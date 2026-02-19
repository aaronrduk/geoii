"""
Custom loss functions for multi-task learning.

This module implements loss functions optimized for segmentation
and classification tasks in geospatial imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    Dice loss is particularly effective for handling class imbalance
    in segmentation tasks, which is common in geospatial imagery.

    Student note: Dice coefficient measures overlap between prediction and target.
    Dice Loss = 1 - Dice Coefficient, so we want to minimize it.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.

        Args:
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            pred (torch.Tensor): Predicted probabilities, shape (N, H, W)
            target (torch.Tensor): Ground truth binary mask, shape (N, H, W)

        Returns:
            torch.Tensor: Dice loss value
        """
        # Flatten the tensors
        # Student note: We flatten to compute overlap across all pixels
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()

        # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        # Return Dice loss
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses training on hard negatives.
    This is especially useful when there's class imbalance (e.g., background >> buildings).

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha (float): Weighting factor for class imbalance
            gamma (float): Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.

        Args:
            pred (torch.Tensor): Predicted logits, shape (N, C, H, W)
            target (torch.Tensor): Ground truth labels, shape (N, H, W)

        Returns:
            torch.Tensor: Focal loss value
        """
        # Get probabilities using softmax
        # Student note: Softmax converts logits to probabilities that sum to 1
        pred_prob = F.softmax(pred, dim=1)

        # Get the probability of the correct class for each pixel
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Calculate focal weight: (1 - p_t)^gamma
        # This reduces loss for well-classified examples
        pt = (pred_prob * target_one_hot).sum(1)
        focal_weight = (1 - pt) ** self.gamma

        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation: Dice + BCE.

    Combining multiple loss functions often works better than using just one.
    Dice handles class imbalance, while BCE provides stable gradients.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """
        Initialize combined loss.

        Args:
            dice_weight (float): Weight for Dice loss
            bce_weight (float): Weight for Binary Cross-Entropy loss
        """
        super(CombinedSegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            pred (torch.Tensor): Predicted logits, shape (N, 1, H, W)
            target (torch.Tensor): Ground truth binary mask, shape (N, H, W)

        Returns:
            torch.Tensor: Combined loss value
        """
        # For Dice, we need probabilities
        pred_prob = torch.sigmoid(pred.squeeze(1))
        target_float = target.float()

        # Calculate individual losses
        dice = self.dice_loss(pred_prob, target_float)
        bce = self.bce_loss(pred.squeeze(1), target_float)

        # Combine with weights
        total_loss = self.dice_weight * dice + self.bce_weight * bce

        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for the complete feature extraction model.

    Combines losses from all tasks: building segmentation, roof classification,
    road segmentation, waterbody segmentation, and utility detection.

    Student note: Multi-task learning requires balancing losses from different tasks.
    We use task weighting to ensure one task doesn't dominate training.
    """

    def __init__(
        self,
        building_weight: float = 1.0,
        roof_weight: float = 0.5,
        road_weight: float = 0.8,
        waterbody_weight: float = 0.8,
        utility_weight: float = 0.6,
    ):
        """
        Initialize multi-task loss.

        Args:
            building_weight (float): Weight for building segmentation loss
            roof_weight (float): Weight for roof type classification loss
            road_weight (float): Weight for road segmentation loss
            waterbody_weight (float): Weight for waterbody segmentation loss
            utility_weight (float): Weight for utility detection loss
        """
        super(MultiTaskLoss, self).__init__()

        # Task weights
        # Student note: These can be tuned based on task importance
        self.building_weight = building_weight
        self.roof_weight = roof_weight
        self.road_weight = road_weight
        self.waterbody_weight = waterbody_weight
        self.utility_weight = utility_weight

        # Loss functions for different tasks
        self.building_loss_fn = CombinedSegmentationLoss()
        self.road_loss_fn = CombinedSegmentationLoss()
        self.waterbody_loss_fn = CombinedSegmentationLoss()
        self.utility_loss_fn = CombinedSegmentationLoss()

        # Roof type is multi-class classification
        self.roof_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore background

    def forward(self, predictions: dict, targets: dict) -> tuple:
        """
        Calculate total multi-task loss.

        Args:
            predictions (dict): Dictionary with predicted outputs from each head
            targets (dict): Dictionary with ground truth labels

        Returns:
            tuple: (total_loss, loss_dict) where loss_dict contains individual losses
        """
        losses = {}

        # Building segmentation loss
        if "building_mask" in predictions and "building_mask" in targets:
            losses["building"] = self.building_loss_fn(
                predictions["building_mask"], targets["building_mask"]
            )

        # Roof type classification loss (only on building pixels)
        if "roof_type" in predictions and "roof_type_mask" in targets:
            losses["roof"] = self.roof_loss_fn(
                predictions["roof_type"], targets["roof_type_mask"]
            )

        # Road segmentation loss
        if "road_mask" in predictions and "road_mask" in targets:
            losses["road"] = self.road_loss_fn(
                predictions["road_mask"], targets["road_mask"]
            )

        # Waterbody segmentation loss
        if "waterbody_mask" in predictions and "waterbody_mask" in targets:
            losses["waterbody"] = self.waterbody_loss_fn(
                predictions["waterbody_mask"], targets["waterbody_mask"]
            )

        # Utility detection loss
        if "utility_mask" in predictions and "utility_mask" in targets:
            losses["utility"] = self.utility_loss_fn(
                predictions["utility_mask"], targets["utility_mask"]
            )

        # Calculate weighted total loss
        total_loss = (
            self.building_weight * losses.get("building", 0.0)
            + self.roof_weight * losses.get("roof", 0.0)
            + self.road_weight * losses.get("road", 0.0)
            + self.waterbody_weight * losses.get("waterbody", 0.0)
            + self.utility_weight * losses.get("utility", 0.0)
        )

        return total_loss, losses


# Utility function for testing loss functions
if __name__ == "__main__":
    """
    Test the loss functions with dummy data.
    Student note: This is useful for debugging and understanding how losses work.
    """
    # Create dummy predictions and targets
    batch_size = 4
    height, width = 128, 128
    num_classes = 5

    # Test Dice Loss
    pred = torch.rand(batch_size, height, width)
    target = torch.randint(0, 2, (batch_size, height, width)).float()
    dice_loss = DiceLoss()
    loss_value = dice_loss(pred, target)
    print(f"Dice Loss: {loss_value.item():.4f}")

    # Test Focal Loss
    pred_logits = torch.randn(batch_size, num_classes, height, width)
    target_labels = torch.randint(0, num_classes, (batch_size, height, width))
    focal_loss = FocalLoss()
    loss_value = focal_loss(pred_logits, target_labels)
    print(f"Focal Loss: {loss_value.item():.4f}")

    # Test Multi-Task Loss
    predictions = {
        "building_mask": torch.randn(batch_size, 1, height, width),
        "road_mask": torch.randn(batch_size, 1, height, width),
        "roof_type": torch.randn(batch_size, 5, height, width),
    }
    targets = {
        "building_mask": torch.randint(0, 2, (batch_size, height, width)),
        "road_mask": torch.randint(0, 2, (batch_size, height, width)),
        "roof_type_mask": torch.randint(0, 5, (batch_size, height, width)),
    }

    mtl = MultiTaskLoss()
    total_loss, loss_dict = mtl(predictions, targets)
    print(f"Multi-Task Loss: {total_loss.item():.4f}")
    print(
        f"Individual losses: {', '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])}"
    )
