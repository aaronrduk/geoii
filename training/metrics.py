"""
Training metrics for SVAMITVA feature extraction.

This module implements evaluation metrics for segmentation and classification tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class IoUMetric:
    """
    Intersection over Union (IoU) metric for segmentation.

    Also known as Jaccard Index, IoU measures the overlap between
    predicted and ground truth masks.

    IoU = |A ∩ B| / |A ∪ B|

    Student note: IoU is the most common metric for segmentation tasks.
    A value of 1.0 means perfect overlap, 0.0 means no overlap.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize IoU metric.

        Args:
            threshold (float): Threshold for converting probabilities to binary mask
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset the metric state."""
        self.intersection = 0
        self.union = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with a batch of predictions.

        Args:
            pred (torch.Tensor): Predicted logits or probabilities
            target (torch.Tensor): Ground truth binary mask
        """
        # Convert to binary predictions
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        target = target.float()

        # Calculate intersection and union
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection

        self.intersection += intersection.item()
        self.union += union.item()

    def compute(self) -> float:
        """
        Compute final IoU score.

        Returns:
            float: IoU score
        """
        if self.union == 0:
            return 0.0
        return self.intersection / self.union


class PixelAccuracy:
    """
    Pixel-wise accuracy metric.

    Measures the percentage of correctly classified pixels.

    Student note: While simple, pixel accuracy can be misleading
    when classes are imbalanced.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize pixel accuracy metric."""
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset the metric state."""
        self.correct = 0
        self.total = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with a batch of predictions."""
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        pred_binary = (torch.sigmoid(pred) > self.threshold).long()
        target = target.long()

        correct = (pred_binary == target).sum()
        total = target.numel()

        self.correct += correct.item()
        self.total += total.item()

    def compute(self) -> float:
        """Compute final accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class F1Score:
    """
    F1 Score metric (harmonic mean of precision and recall).

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Student note: F1 score is useful when you care about both
    false positives and false negatives.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize F1 score metric."""
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset the metric state."""
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with a batch of predictions."""
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        target = target.float()

        # Calculate TP, FP, FN
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        fn = ((1 - pred_binary) * target).sum()

        self.true_positive += tp.item()
        self.false_positive += fp.item()
        self.false_negative += fn.item()

    def compute(self) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score.

        Returns:
            dict: Dictionary with precision, recall, and f1 scores
        """
        # Calculate precision
        if (self.true_positive + self.false_positive) == 0:
            precision = 0.0
        else:
            precision = self.true_positive / (self.true_positive + self.false_positive)

        # Calculate recall
        if (self.true_positive + self.false_negative) == 0:
            recall = 0.0
        else:
            recall = self.true_positive / (self.true_positive + self.false_negative)

        # Calculate F1
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}


class MultiClassAccuracy:
    """
    Multi-class classification accuracy for roof types.

    Student note: This is for the roof type classification task.
    """

    def __init__(self, num_classes: int = 5, ignore_index: int = 0):
        """
        Initialize multi-class accuracy.

        Args:
            num_classes (int): Number of classes
            ignore_index (int): Class index to ignore (e.g., background)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset the metric state."""
        self.correct = 0
        self.total = 0
        self.per_class_correct = [0] * self.num_classes
        self.per_class_total = [0] * self.num_classes

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with a batch of predictions.

        Args:
            pred (torch.Tensor): Predicted logits, shape (N, C, H, W)
            target (torch.Tensor): Ground truth labels, shape (N, H, W)
        """
        # Get predicted classes
        pred_classes = pred.argmax(dim=1)

        # Create mask for valid pixels (ignore background)
        valid_mask = target != self.ignore_index

        if valid_mask.sum() == 0:
            return

        # Calculate overall accuracy
        correct = (pred_classes[valid_mask] == target[valid_mask]).sum()
        total = valid_mask.sum()

        self.correct += correct.item()
        self.total += total.item()

        # Calculate per-class accuracy
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            cls_mask = target == cls
            if cls_mask.sum() == 0:
                continue

            cls_correct = (pred_classes[cls_mask] == target[cls_mask]).sum()
            self.per_class_correct[cls] += cls_correct.item()
            self.per_class_total[cls] += cls_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        """
        Compute overall and per-class accuracy.

        Returns:
            dict: Dictionary with overall and per-class accuracy
        """
        if self.total == 0:
            overall_acc = 0.0
        else:
            overall_acc = self.correct / self.total

        per_class_acc = {}
        for cls in range(self.num_classes):
            if cls == self.ignore_index or self.per_class_total[cls] == 0:
                per_class_acc[f"class_{cls}"] = 0.0
            else:
                per_class_acc[f"class_{cls}"] = (
                    self.per_class_correct[cls] / self.per_class_total[cls]
                )

        return {"overall": overall_acc, **per_class_acc}


class MetricTracker:
    """
    Tracker for all metrics across different tasks.

    Student note: This class aggregates metrics from all tasks
    for comprehensive evaluation.
    """

    def __init__(self):
        """Initialize metric tracker."""
        # Segmentation metrics for each task
        self.building_iou = IoUMetric()
        self.road_iou = IoUMetric()
        self.waterbody_iou = IoUMetric()
        self.utility_iou = IoUMetric()

        # F1 scores
        self.building_f1 = F1Score()
        self.road_f1 = F1Score()
        self.waterbody_f1 = F1Score()
        self.utility_f1 = F1Score()

        # Roof type classification
        self.roof_acc = MultiClassAccuracy(num_classes=5, ignore_index=0)

    def reset(self):
        """Reset all metrics."""
        self.building_iou.reset()
        self.road_iou.reset()
        self.waterbody_iou.reset()
        self.utility_iou.reset()

        self.building_f1.reset()
        self.road_f1.reset()
        self.waterbody_f1.reset()
        self.utility_f1.reset()

        self.roof_acc.reset()

    def update(self, predictions: Dict, targets: Dict):
        """
        Update all metrics with a batch of predictions.

        Args:
            predictions (dict): Model predictions
            targets (dict): Ground truth labels
        """
        # Update segmentation metrics
        if "building_mask" in predictions:
            self.building_iou.update(
                predictions["building_mask"], targets["building_mask"]
            )
            self.building_f1.update(
                predictions["building_mask"], targets["building_mask"]
            )

        if "road_mask" in predictions:
            self.road_iou.update(predictions["road_mask"], targets["road_mask"])
            self.road_f1.update(predictions["road_mask"], targets["road_mask"])

        if "waterbody_mask" in predictions:
            self.waterbody_iou.update(
                predictions["waterbody_mask"], targets["waterbody_mask"]
            )
            self.waterbody_f1.update(
                predictions["waterbody_mask"], targets["waterbody_mask"]
            )

        if "utility_mask" in predictions:
            self.utility_iou.update(
                predictions["utility_mask"], targets["utility_mask"]
            )
            self.utility_f1.update(predictions["utility_mask"], targets["utility_mask"])

        # Update roof classification metric
        if "roof_type" in predictions:
            self.roof_acc.update(predictions["roof_type"], targets["roof_type_mask"])

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            dict: Dictionary with all computed metrics
        """
        metrics = {}

        # IoU scores
        metrics["building_iou"] = self.building_iou.compute()
        metrics["road_iou"] = self.road_iou.compute()
        metrics["waterbody_iou"] = self.waterbody_iou.compute()
        metrics["utility_iou"] = self.utility_iou.compute()

        # F1 scores
        building_f1_results = self.building_f1.compute()
        metrics["building_precision"] = building_f1_results["precision"]
        metrics["building_recall"] = building_f1_results["recall"]
        metrics["building_f1"] = building_f1_results["f1"]

        # Average IoU (main metric for hackathon)
        # Student note: This is the key metric to meet the 95% target
        avg_iou = (
            metrics["building_iou"]
            + metrics["road_iou"]
            + metrics["waterbody_iou"]
            + metrics["utility_iou"]
        ) / 4.0
        metrics["avg_iou"] = avg_iou

        # Roof type accuracy
        roof_results = self.roof_acc.compute()
        metrics["roof_accuracy"] = roof_results["overall"]

        return metrics
