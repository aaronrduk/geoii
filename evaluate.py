"""
Model evaluation script for SVAMITVA feature extraction.

This script evaluates a trained model on a test dataset and generates
comprehensive performance metrics and visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.feature_extractor import FeatureExtractorModel
from models.losses import MultiTaskLoss
from training.metrics import MetricTracker
from data.dataset import SvamitvaDataset
from data.augmentation import get_test_transforms

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        tuple: (model, checkpoint_dict)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    backbone = config.get("backbone", "resnet50")
    num_roof_classes = config.get("num_roof_classes", 5)

    model = FeatureExtractorModel(
        backbone=backbone, pretrained=False, num_roof_classes=num_roof_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion, device: str
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Test data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metric_tracker = MetricTracker()
    total_loss = 0.0
    num_batches = 0

    logger.info("Starting evaluation...")

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        targets = {
            "building_mask": batch["building_mask"].to(device),
            "road_mask": batch["road_mask"].to(device),
            "waterbody_mask": batch["waterbody_mask"].to(device),
            "utility_mask": batch["utility_mask"].to(device),
            "roof_type_mask": batch["roof_type_mask"].to(device),
        }

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)

        total_loss += loss.item()
        metric_tracker.update(predictions, targets)
        num_batches += 1

    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

    return metrics


def save_evaluation_report(metrics: Dict, output_path: Path):
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics: Evaluation metrics
        output_path: Path to save report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "evaluation_metrics": metrics,
        "summary": {
            "avg_iou": metrics.get("avg_iou", 0.0),
            "building_iou": metrics.get("building_iou", 0.0),
            "road_iou": metrics.get("road_iou", 0.0),
            "waterbody_iou": metrics.get("waterbody_iou", 0.0),
            "utility_iou": metrics.get("utility_iou", 0.0),
            "roof_accuracy": metrics.get("roof_accuracy", 0.0),
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_path}")


def print_evaluation_summary(metrics: Dict):
    """
    Print evaluation summary to console.

    Args:
        metrics: Evaluation metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Loss: {metrics.get('loss', 0.0):.4f}")
    logger.info(f"Average IoU: {metrics.get('avg_iou', 0.0):.4f}")
    logger.info("-" * 60)
    logger.info("Task-Specific Metrics:")
    logger.info(f"  Building IoU: {metrics.get('building_iou', 0.0):.4f}")
    logger.info(f"  Building Precision: {metrics.get('building_precision', 0.0):.4f}")
    logger.info(f"  Building Recall: {metrics.get('building_recall', 0.0):.4f}")
    logger.info(f"  Building F1: {metrics.get('building_f1', 0.0):.4f}")
    logger.info(f"  Road IoU: {metrics.get('road_iou', 0.0):.4f}")
    logger.info(f"  Waterbody IoU: {metrics.get('waterbody_iou', 0.0):.4f}")
    logger.info(f"  Utility IoU: {metrics.get('utility_iou', 0.0):.4f}")
    logger.info(f"  Roof Type Accuracy: {metrics.get('roof_accuracy', 0.0):.4f}")
    logger.info("=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SVAMITVA Feature Extraction Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_report.json",
        help="Path to save evaluation report",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    output_path = Path(args.output)
    device = args.device

    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return

    logger.info(f"Using device: {device}")

    model, checkpoint = load_checkpoint(checkpoint_path, device)

    test_dataset = SvamitvaDataset(
        root_dir=test_dir,
        image_size=args.image_size,
        transform=get_test_transforms(args.image_size),
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    config = checkpoint.get("config", {})
    criterion = MultiTaskLoss(
        building_weight=config.get("building_weight", 1.0),
        roof_weight=config.get("roof_weight", 0.5),
        road_weight=config.get("road_weight", 0.8),
        waterbody_weight=config.get("waterbody_weight", 0.8),
        utility_weight=config.get("utility_weight", 0.6),
    )

    metrics = evaluate_model(model, test_loader, criterion, device)

    print_evaluation_summary(metrics)

    save_evaluation_report(metrics, output_path)

    if metrics.get("avg_iou", 0.0) >= 0.95:
        logger.info("\n✓ Model meets the 95% IoU target!")
    else:
        logger.info(f"\n⚠ Model IoU ({metrics.get('avg_iou', 0.0):.2%}) is below 95% target")


if __name__ == "__main__":
    main()
