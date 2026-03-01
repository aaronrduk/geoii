"""
Multi-task loss functions for SVAMITVA feature extraction.

Supports all 10 shapefile categories.

NaN Safety Strategy:
  - All loss modules clamp logits before sigmoid/softmax
  - DiceLoss uses smooth + ε to handle all-zero masks
  - MultiTaskLoss stores sub-losses in nn.ModuleDict so .to(device) works
  - Zero-total guard returns a tiny 1e-7 sentinel to prevent GradScaler Inf injection

Lovász hinge loss is a direct IoU surrogate — maximises the Jaccard index
directly via its tight convex extension.
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

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
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

# ── Boundary Loss ──────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Penalises prediction errors near ground-truth boundaries.
    Extracts edges from the target mask using a Laplacian-like kernel,
    then computes BCE only on boundary pixels — sharpens predicted edges.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # 3×3 Laplacian kernel to detect edges
        k = torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("edge_kernel", k)
        self.kernel_size = kernel_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: logits (N,H,W), target: binary float (N,H,W)."""
        t4d = target.unsqueeze(1).float()
        edges = F.conv2d(t4d, self.edge_kernel, padding=1).squeeze(1)
        boundary_mask = (edges.abs() > 0.1).float()

        if boundary_mask.sum() < 1:
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        # BCE only on boundary pixels
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        loss = (bce * boundary_mask).sum() / boundary_mask.sum().clamp(min=1)

        if not torch.isfinite(loss):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()
        return loss


# ── Lovász Hinge Loss ──────────────────────────────────────────────────────────

class LovaszHingeLoss(nn.Module):
    """
    Lovász hinge loss for binary segmentation.

    Direct surrogate optimisation of the IoU (Jaccard) score.
    Operates on raw logits (not sigmoid probabilities).

    Reference: Berman et al., "The Lovász-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks", CVPR 2018.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t. sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union.clamp(min=1e-6)
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: logits (N, H, W) or (N, 1, H, W)
            target: binary float (N, H, W)
        """
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        losses = []
        for p, t in zip(pred, target):
            p_flat = p.reshape(-1)
            t_flat = t.reshape(-1).float()
            signs = 2.0 * t_flat - 1.0
            errors = 1.0 - p_flat * signs
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            gt_sorted = t_flat[perm]
            grad = self._lovasz_grad(gt_sorted)
            loss = torch.dot(F.relu(errors_sorted), grad)
            losses.append(loss)

        if not losses:
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()

        total = sum(losses) / len(losses)
        if not torch.isfinite(total):
            return torch.zeros(1, device=pred.device, dtype=torch.float32).sum()
        return total


# ── Combined Segmentation Loss ────────────────────────────────────────────────

class CombinedSegmentationLoss(nn.Module):
    """
    Combined Dice + Binary Focal + Lovász + Boundary loss for binary segmentation.

    Lovász hinge is a direct IoU surrogate — adding it to the mix provides
    strong gradient signal that directly pushes IoU upward, complementing
    the region-based Dice and pixel-based Focal losses.
    """

    def __init__(
        self,
        dice_weight: float = 0.3,
        bce_weight: float = 0.3,
        lovasz_weight: float = 0.25,
        boundary_weight: float = 0.15,
    ):
        super().__init__()
        self.dice_weight     = dice_weight
        self.bce_weight      = bce_weight
        self.lovasz_weight   = lovasz_weight
        self.boundary_weight = boundary_weight
        self.dice_loss       = DiceLoss()
        self.bce_loss        = BinaryFocalWithLogitsLoss(alpha=0.75, gamma=2.0)
        self.lovasz_loss     = LovaszHingeLoss()
        self.boundary_loss   = BoundaryLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred.float(), -100, 100)

        if pred.dim() == 4 and pred.shape[1] == 1:
            pred_sq = pred.squeeze(1)
        else:
            pred_sq = pred

        pred_prob    = torch.sigmoid(pred_sq)
        target_float = torch.clamp(torch.nan_to_num(target.float(), nan=0.0), 0.0, 1.0)

        dice     = self.dice_loss(pred_prob, target_float)
        bce      = self.bce_loss(pred_sq, target_float)
        lovasz   = self.lovasz_loss(pred_sq, target_float)
        boundary = self.boundary_loss(pred_sq, target_float)
        total    = (self.dice_weight * dice
                    + self.bce_weight * bce
                    + self.lovasz_weight * lovasz
                    + self.boundary_weight * boundary)

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
        road_weight:            float = 1.0,
        waterbody_weight:       float = 1.0,
        road_centerline_weight: float = 1.0,
        waterbody_line_weight:  float = 1.0,
        waterbody_point_weight: float = 1.2,
        utility_line_weight:    float = 1.0,
        utility_poly_weight:    float = 1.0,
        bridge_weight:          float = 1.2,
        railway_weight:         float = 1.2,
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
