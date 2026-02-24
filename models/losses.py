"""
Multi-task loss functions for SVAMITVA feature extraction.

Supports all 10 shapefile categories.

NaN Safety Strategy:
  - All loss modules clamp logits before sigmoid/softmax
  - DiceLoss uses smooth + ε to handle all-zero masks
  - MultiTaskLoss stores sub-losses in nn.ModuleDict so .to(device) works
  - Zero-total guard returns a tiny 1e-7 sentinel to prevent GradScaler Inf injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation. Receives probabilities (post-sigmoid), not logits.
    Handles class imbalance — critical for sparse features like bridges, railways.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.contiguous().view(-1).float()
        target = target.contiguous().view(-1).float()

        pred   = torch.nan_to_num(pred,   nan=0.0, posinf=1.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        intersection = (pred * target).sum()
        denominator  = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth + 1e-7)

        if not torch.isfinite(dice):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return 1.0 - dice


# ── Binary Focal Loss ─────────────────────────────────────────────────────────

class BinaryFocalWithLogitsLoss(nn.Module):
    """
    Binary Focal Loss for highly unbalanced segmentation.
    Used for sparse features: waterbody_line, utility_line, etc.

    Formula: -α(1-p)^γ log(p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred.float(), -100, 100)

        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        bce_loss  = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        pt         = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        loss       = focal_loss.mean()

        if not torch.isfinite(loss):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return loss


# ── Multi-Class Focal Loss ────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance in multi-class tasks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred  = torch.clamp(pred.float(), -100, 100)
        logpt = -F.cross_entropy(pred, target.long(), reduction="none")
        pt    = torch.exp(logpt)
        loss  = -(self.alpha * (1 - pt) ** self.gamma * logpt).mean()

        if not torch.isfinite(loss):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return loss


# ── Combined Segmentation Loss ────────────────────────────────────────────────

class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + Binary Focal loss for binary segmentation tasks."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss()
        self.bce_loss    = BinaryFocalWithLogitsLoss(alpha=0.25, gamma=2.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred.float(), -100, 100)

        if pred.dim() == 4 and pred.shape[1] == 1:
            pred_sq = pred.squeeze(1)
        else:
            pred_sq = pred

        pred_prob    = torch.sigmoid(pred_sq)
        target_float = torch.clamp(torch.nan_to_num(target.float(), nan=0.0), 0.0, 1.0)

        dice  = self.dice_loss(pred_prob, target_float)
        bce   = self.bce_loss(pred_sq, target_float)
        total = self.dice_weight * dice + self.bce_weight * bce

        if not torch.isfinite(total):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return total


# ── Multi-Task Loss ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss covering all 10 SVAMITVA shapefile categories.

    Stored as nn.ModuleDict so model.to(device) / loss.to(device) moves
    all sub-module parameters and buffers correctly.

    Bridge and railway are downweighted by default since they are often absent
    from training data — all-zero masks produce a near-zero loss, which is
    valid supervision ("predict nothing here").
    """

    def __init__(
        self,
        building_weight:        float = 1.0,
        roof_weight:            float = 0.5,
        road_weight:            float = 0.8,
        waterbody_weight:       float = 0.8,
        road_centerline_weight: float = 0.7,
        waterbody_line_weight:  float = 0.7,
        waterbody_point_weight: float = 0.9,
        utility_line_weight:    float = 0.7,
        utility_poly_weight:    float = 0.8,
        bridge_weight:          float = 0.5,
        railway_weight:         float = 0.5,
    ):
        super().__init__()

        self.weights = {
            "building":         building_weight,
            "roof":             roof_weight,
            "road":             road_weight,
            "road_centerline":  road_centerline_weight,
            "waterbody":        waterbody_weight,
            "waterbody_line":   waterbody_line_weight,
            "waterbody_point":  waterbody_point_weight,
            "utility_line":     utility_line_weight,
            "utility_poly":     utility_poly_weight,
            "bridge":           bridge_weight,
            "railway":          railway_weight,
        }

        # nn.ModuleDict ensures sub-losses move with .to(device)
        self.loss_fns = nn.ModuleDict({
            "building":        CombinedSegmentationLoss(),
            "road":            CombinedSegmentationLoss(),
            "road_centerline": CombinedSegmentationLoss(),
            "waterbody":       CombinedSegmentationLoss(),
            "waterbody_line":  CombinedSegmentationLoss(),
            "waterbody_point": CombinedSegmentationLoss(),
            "utility_line":    CombinedSegmentationLoss(),
            "utility_poly":    CombinedSegmentationLoss(),
            "bridge":          CombinedSegmentationLoss(),
            "railway":         CombinedSegmentationLoss(),
        })

        # ignore_index=0 → background pixels never contribute to roof CE loss
        self.roof_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, predictions: dict, targets: dict) -> tuple:
        losses     = {}
        out_device = next(iter(predictions.values())).device

        # ── Segmentation tasks ──────────────────────────────────────────────
        for key, loss_fn in self.loss_fns.items():
            mask_key = f"{key}_mask"
            if mask_key not in predictions or mask_key not in targets:
                continue
            loss = loss_fn(predictions[mask_key], targets[mask_key])
            if torch.isfinite(loss):
                losses[key] = loss

        # ── Roof type multi-class ───────────────────────────────────────────
        if "roof_type" in predictions and "roof_type_mask" in targets:
            preds_rt = torch.nan_to_num(
                torch.clamp(predictions["roof_type"].float(), -100, 100), nan=0.0
            )
            rt_tgt = targets["roof_type_mask"]
            if rt_tgt.dim() == 4:
                rt_tgt = rt_tgt.squeeze(1)
            rt_tgt = torch.clamp(
                torch.nan_to_num(rt_tgt, nan=0).long(), 0, preds_rt.shape[1] - 1
            )

            if (rt_tgt != 0).any():
                l_roof = self.roof_loss_fn(preds_rt, rt_tgt)
                if torch.isfinite(l_roof):
                    losses["roof"] = l_roof
            else:
                # Dummy to keep graph connected (avoids DDP issues)
                losses["roof"] = (preds_rt * 0).sum()

        # ── Weighted sum ────────────────────────────────────────────────────
        total_loss = torch.zeros(1, device=out_device, dtype=torch.float32).sum()

        for key, value in losses.items():
            if key in self.weights and torch.isfinite(value):
                total_loss = total_loss + self.weights[key] * value

        # Sentinel guard: prevent GradScaler injecting Inf on zero-loss steps
        if total_loss.item() <= 1e-7:
            total_loss = total_loss + 1e-7

        return total_loss, losses
