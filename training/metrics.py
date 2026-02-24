"""
Training metrics for all 10 SVAMITVA shapefile tasks.
Tracks IoU, F1, Precision, Recall per task + roof type accuracy.
"""

import torch
import numpy as np
from typing import Dict


class IoUMetric:
    """Intersection over Union (Jaccard) for binary segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union        = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred_bin     = (torch.sigmoid(pred) > self.threshold).float()
        target_f     = target.float()
        intersection = (pred_bin * target_f).sum()
        union        = pred_bin.sum() + target_f.sum() - intersection
        self.intersection += intersection.item()
        self.union        += union.item()

    def compute(self) -> float:
        return self.intersection / self.union if self.union > 0 else 0.0


class F1Score:
    """F1, Precision, Recall for binary segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred_bin = (torch.sigmoid(pred) > self.threshold).float()
        target_f = target.float()
        self.tp  += (pred_bin * target_f).sum().item()
        self.fp  += (pred_bin * (1 - target_f)).sum().item()
        self.fn  += ((1 - pred_bin) * target_f).sum().item()

    def compute(self) -> Dict[str, float]:
        tp, fp, fn  = self.tp, self.fp, self.fn
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1          = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1}


class MultiClassAccuracy:
    """Pixel accuracy for roof type multi-class classification."""

    def __init__(self, num_classes: int = 5, ignore_index: int = 0):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.reset()

    def reset(self):
        self.correct = 0
        self.total   = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_cls   = pred.argmax(dim=1)
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return
        self.correct += (pred_cls[valid_mask] == target[valid_mask]).sum().item()
        self.total   += valid_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        return {"overall": self.correct / self.total if self.total > 0 else 0.0}


# ── All tasks ──────────────────────────────────────────────────────────────────

_SEG_TASKS = [
    "building", "road", "road_centerline",
    "waterbody", "waterbody_line", "waterbody_point",
    "utility_line", "utility_poly", "bridge", "railway",
]

_TASK_KEYS = {t: (f"{t}_mask", f"{t}_mask") for t in _SEG_TASKS}


class MetricTracker:
    """
    Aggregates IoU + F1 for all 10 segmentation tasks plus roof type accuracy.
    Instantiate fresh each epoch, call update() per batch, compute() at end.
    """

    def __init__(self):
        self.iou      = {t: IoUMetric() for t in _SEG_TASKS}
        self.f1       = {t: F1Score()   for t in _SEG_TASKS}
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
                self.f1[task].update(predictions[pred_key],  targets[tgt_key])

        if "roof_type" in predictions and "roof_type_mask" in targets:
            self.roof_acc.update(predictions["roof_type"], targets["roof_type_mask"])

    def compute(self) -> Dict[str, float]:
        metrics    = {}
        iou_values = []

        for task in _SEG_TASKS:
            iou_val = self.iou[task].compute()
            metrics[f"{task}_iou"] = iou_val
            iou_values.append(iou_val)

            f1_result = self.f1[task].compute()
            metrics[f"{task}_f1"]        = f1_result["f1"]
            metrics[f"{task}_precision"] = f1_result["precision"]
            metrics[f"{task}_recall"]    = f1_result["recall"]

        metrics["avg_iou"]       = sum(iou_values) / len(iou_values)
        metrics["roof_accuracy"] = self.roof_acc.compute()["overall"]

        return metrics
