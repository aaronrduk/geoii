"""
Custom loss functions for multi-task learning.

Supports all 10 shapefile categories in the SVAMITVA dataset.

NaN safety strategy
-------------------
1. DiceLoss     – clamps inputs, adds smooth+ε, final NaN guard.
2. FocalLoss    – fp32 upcasting, clamp logits, NaN guard.
3. CombinedSeg  – clamp logits, sanitize target, NaN guard.
4. MultiTask    – loss_fns stored in nn.ModuleDict so .to(device) works;
                  all-zero total_loss returns a tiny positive scalar so the
                  scaler does not inject Inf into future steps.
5. Roof head    – CrossEntropy only on non-background pixels; zero-tensor
                  fallback on same device when all pixels are background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ─────────────────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    Receives *probabilities* (after sigmoid), not raw logits.
    Handles class imbalance — crucial for sparse features like
    bridges, railway lines, and waterbody points.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force fp32 for high-precision summation (fp16 overflows at 65 504)
        pred = pred.contiguous().view(-1).float()
        target = target.contiguous().view(-1).float()

        # NaN / Inf guard on inputs
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()

        # ε added to denominator prevents 0/0 when mask is all-zero
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth + 1e-7)

        if not torch.isfinite(dice):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return 1.0 - dice


# ── Binary Focal Loss ─────────────────────────────────────────────────────────


class BinaryFocalWithLogitsLoss(nn.Module):
    """
    Binary Focal Loss for highly unbalanced segmentation tasks.
    Used for 'waterbody_line', 'utility_line', etc. where positive pixels
    are extremely sparse.

    Formula: - \alpha (1 - p)^\gamma \log(p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp logits — prevents sigmoid overflow in fp16
        pred = torch.clamp(pred.float(), -100, 100)

        # Squeeze channel dim if present
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target.float(), reduction="none"
        )

        pt = torch.exp(
            -bce_loss
        )  # pt is the predicted probability for the actual class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        loss = focal_loss.mean()

        if not torch.isfinite(loss):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return loss


# ── Multi-Class Focal Loss ────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred.float(), -100, 100)

        # Focal Loss Rewrite: Remove one-hot memory expansion
        logpt = -F.cross_entropy(pred, target.long(), reduction="none")
        pt = torch.exp(logpt)
        loss = -(self.alpha * (1 - pt) ** self.gamma * logpt).mean()

        if not torch.isfinite(loss):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return loss


# ── Combined Segmentation Loss ────────────────────────────────────────────────


class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + Binary Focal loss for binary segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()

        # Adopted from .ipynb analysis: use Focal Loss to handle extreme class imbalance
        # for sparse features instead of regular BCE.
        self.bce_loss = BinaryFocalWithLogitsLoss(alpha=0.25, gamma=2.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp logits — prevents sigmoid overflow in fp16
        pred = torch.clamp(pred.float(), -100, 100)

        # Squeeze channel dim if present (N,1,H,W) → (N,H,W)
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred_sq = pred.squeeze(1)
        else:
            pred_sq = pred

        pred_prob = torch.sigmoid(pred_sq)

        target_float = target.float()
        target_float = torch.nan_to_num(target_float, nan=0.0)
        target_float = torch.clamp(target_float, 0.0, 1.0)

        dice = self.dice_loss(pred_prob, target_float)
        bce = self.bce_loss(pred_sq, target_float)

        total = self.dice_weight * dice + self.bce_weight * bce

        if not torch.isfinite(total):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        return total


# ── Multi-Task Loss ───────────────────────────────────────────────────────────


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for all 10 shapefile categories.

    Key correctness notes
    ─────────────────────
    • loss_fns is stored as nn.ModuleDict so that model.to(device) /
      loss_fn.to(device) correctly moves all sub-module parameters and
      buffers to the target device.
    • When ALL task losses are absent (e.g. only bridge/railway tiles with
      no positive pixels and those tasks skipped), we return a tiny sentinel
      value (1e-7) instead of 0.  This prevents the GradScaler from seeing
      a gradient of 0 and then amplifying it to Inf on the next step.
    • Bridge and railway tasks are included in the loss exactly like other
      tasks — their all-zero masks just produce a near-zero Dice+BCE loss,
      which is correct supervision signal ("predict nothing here").
    """

    def __init__(
        self,
        building_weight: float = 1.0,
        roof_weight: float = 0.5,
        road_weight: float = 0.8,
        waterbody_weight: float = 0.8,
        road_centerline_weight: float = 0.7,
        waterbody_line_weight: float = 0.7,
        waterbody_point_weight: float = 0.9,
        utility_line_weight: float = 0.7,
        utility_poly_weight: float = 0.8,
        bridge_weight: float = 0.5,  # downweighted — no .shp in dataset
        railway_weight: float = 0.5,  # downweighted — no .shp in dataset
    ):
        super().__init__()

        self.weights = {
            "building": building_weight,
            "roof": roof_weight,
            "road": road_weight,
            "road_centerline": road_centerline_weight,
            "waterbody": waterbody_weight,
            "waterbody_line": waterbody_line_weight,
            "waterbody_point": waterbody_point_weight,
            "utility_line": utility_line_weight,
            "utility_poly": utility_poly_weight,
            "bridge": bridge_weight,
            "railway": railway_weight,
        }

        # ★ nn.ModuleDict — sub-losses are properly registered as children
        #   and moved to GPU when loss_fn.to(device) is called.
        self.loss_fns = nn.ModuleDict(
            {
                "building": CombinedSegmentationLoss(),
                "road": CombinedSegmentationLoss(),
                "road_centerline": CombinedSegmentationLoss(),
                "waterbody": CombinedSegmentationLoss(),
                "waterbody_line": CombinedSegmentationLoss(),
                "waterbody_point": CombinedSegmentationLoss(),
                "utility_line": CombinedSegmentationLoss(),
                "utility_poly": CombinedSegmentationLoss(),
                "bridge": CombinedSegmentationLoss(),
                "railway": CombinedSegmentationLoss(),
            }
        )

        # ignore_index=0 → background pixels never contribute to roof CE loss
        self.roof_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, predictions: dict, targets: dict) -> tuple:
        losses = {}

        # Determine output device from first prediction tensor
        out_device = next(iter(predictions.values())).device

        # ── Segmentation tasks ─────────────────────────────────────────────
        for key, loss_fn in self.loss_fns.items():
            mask_key = f"{key}_mask"
            if mask_key not in predictions or mask_key not in targets:
                continue

            loss = loss_fn(predictions[mask_key], targets[mask_key])

            if torch.isfinite(loss):
                losses[key] = loss

        # ── Roof type classification ────────────────────────────────────────
        # Restored CrossEntropyLoss supervision safely
        # Prevent unsupervised logits from destabilizing shared features
        if "roof_type" in predictions and "roof_type_mask" in targets:
            preds_rt = torch.clamp(predictions["roof_type"].float(), -100, 100)
            preds_rt = torch.nan_to_num(preds_rt, nan=0.0)
            rt_tgt = targets["roof_type_mask"]
            if rt_tgt.dim() == 4:
                rt_tgt = rt_tgt.squeeze(1)
            rt_tgt = torch.nan_to_num(rt_tgt, nan=0).long()
            rt_tgt = torch.clamp(rt_tgt, 0, preds_rt.shape[1] - 1)

            # Guard against all-background batch
            if (rt_tgt != 0).any():
                l_roof = self.roof_loss_fn(preds_rt, rt_tgt)
                if torch.isfinite(l_roof):
                    losses["roof"] = l_roof
            else:
                # Add a dummy loss to prevent DDP/Gradient issues and maintain graph connection
                losses["roof"] = (preds_rt * 0).sum()

        # ── Safe weighted summation ────────────────────────────────────────
        total_loss = torch.zeros(1, device=out_device, dtype=torch.float32).sum()

        for key, value in losses.items():
            if key in self.weights:
                if torch.isfinite(value):
                    total_loss = total_loss + self.weights[key] * value

        # ★ Guard: exact-zero sentinel replacement
        if total_loss.item() <= 1e-7:
            total_loss = total_loss + 1e-7

        return total_loss, losses
