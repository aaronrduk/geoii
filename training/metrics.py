"""
Training metrics for all 10 SVAMITVA shapefile tasks.
"""

import torch
import numpy as np
from typing import Dict


class IoUMetric:
    """Intersection over Union (IoU / Jaccard Index) for binary segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        target = target.float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        self.intersection += intersection.item()
        self.union += union.item()

    def compute(self) -> float:
        return self.intersection / self.union if self.union > 0 else 0.0


class PixelAccuracy:
    """Pixel-wise accuracy metric."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred_binary = (torch.sigmoid(pred) > self.threshold).long()
        target = target.long()
        self.correct += (pred_binary == target).sum().item()
        self.total += target.numel()

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class F1Score:
    """F1 Score (harmonic mean of precision and recall)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        target = target.float()
        self.true_positive += (pred_binary * target).sum().item()
        self.false_positive += (pred_binary * (1 - target)).sum().item()
        self.false_negative += ((1 - pred_binary) * target).sum().item()

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.true_positive, self.false_positive, self.false_negative
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}


class MultiClassAccuracy:
    """Multi-class pixel accuracy for roof type classification."""

    def __init__(self, num_classes: int = 5, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_classes = pred.argmax(dim=1)
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return
        self.correct += (pred_classes[valid_mask] == target[valid_mask]).sum().item()
        self.total += valid_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        overall = self.correct / self.total if self.total > 0 else 0.0
        return {"overall": overall}


# ── All tasks tracked ──────────────────────────────────────────────────────────
_SEG_TASKS = [
    "building",
    "road",
    "road_centerline",
    "waterbody",
    "waterbody_line",
    "waterbody_point",
    "utility_line",
    "utility_poly",
    "bridge",
    "railway",
]

# Map task → (prediction key, target key)
_TASK_KEYS = {
    "building": ("building_mask", "building_mask"),
    "road": ("road_mask", "road_mask"),
    "road_centerline": ("road_centerline_mask", "road_centerline_mask"),
    "waterbody": ("waterbody_mask", "waterbody_mask"),
    "waterbody_line": ("waterbody_line_mask", "waterbody_line_mask"),
    "waterbody_point": ("waterbody_point_mask", "waterbody_point_mask"),
    "utility_line": ("utility_line_mask", "utility_line_mask"),
    "utility_poly": ("utility_poly_mask", "utility_poly_mask"),
    "bridge": ("bridge_mask", "bridge_mask"),
    "railway": ("railway_mask", "railway_mask"),
}


class MetricTracker:
    """
    Aggregates IoU + F1 metrics for all 10 segmentation tasks
    plus roof type accuracy.
    """

    def __init__(self):
        # One IoU and F1 per segmentation task
        self.iou = {t: IoUMetric() for t in _SEG_TASKS}
        self.f1 = {t: F1Score() for t in _SEG_TASKS}
        self.roof_acc = MultiClassAccuracy(num_classes=5, ignore_index=0)

    def reset(self):
        for t in _SEG_TASKS:
            self.iou[t].reset()
            self.f1[t].reset()
        self.roof_acc.reset()

    def update(self, predictions: Dict, targets: Dict):
        for task, (pred_key, tgt_key) in _TASK_KEYS.items():
            if pred_key in predictions and tgt_key in targets:
                self.iou[task].update(predictions[pred_key], targets[tgt_key])
                self.f1[task].update(predictions[pred_key], targets[tgt_key])

        if "roof_type" in predictions and "roof_type_mask" in targets:
            self.roof_acc.update(predictions["roof_type"], targets["roof_type_mask"])

    def compute(self) -> Dict[str, float]:
        metrics = {}

        iou_values = []
        for task in _SEG_TASKS:
            val = self.iou[task].compute()
            metrics[f"{task}_iou"] = val
            iou_values.append(val)

            f1_result = self.f1[task].compute()
            metrics[f"{task}_f1"] = f1_result["f1"]
            metrics[f"{task}_precision"] = f1_result["precision"]
            metrics[f"{task}_recall"] = f1_result["recall"]

        # Average IoU over all 10 tasks (primary training metric)
        metrics["avg_iou"] = sum(iou_values) / len(iou_values)

        # Roof classification accuracy
        metrics["roof_accuracy"] = self.roof_acc.compute()["overall"]

        return metrics
