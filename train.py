"""
Main training script for SVAMITVA feature extraction model.

This script orchestrates the complete training process including:
- Model initialization
- Data loading
- Training loop
- Validation
- Checkpointing
- Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.feature_extractor import FeatureExtractorModel
from models.losses import MultiTaskLoss
from training.config import (
    TrainingConfig,
    get_config_from_args,
    get_full_training_config,
)
from training.metrics import MetricTracker
from data.dataset import SvamitvaDataset, create_dataloaders
from data.augmentation import get_train_transforms, get_val_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Student note: This ensures experiments can be reproduced.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training

    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0.0
    metric_tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch["image"].to(device)
        targets = {
            "building_mask": batch["building_mask"].to(device),
            "road_mask": batch["road_mask"].to(device),
            "waterbody_mask": batch["waterbody_mask"].to(device),
            "utility_mask": batch["utility_mask"].to(device),
            "roof_type_mask": batch["roof_type_mask"].to(device),
        }

        # Forward pass with mixed precision
        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast("cuda"):
                predictions = model(images)
                loss, loss_dict = criterion(predictions, targets)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update metrics
        total_loss += loss.item()
        metric_tracker.update(predictions, targets)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0.0
    metric_tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        # Move data to device
        images = batch["image"].to(device)
        targets = {
            "building_mask": batch["building_mask"].to(device),
            "road_mask": batch["road_mask"].to(device),
            "waterbody_mask": batch["waterbody_mask"].to(device),
            "utility_mask": batch["utility_mask"].to(device),
            "roof_type_mask": batch["roof_type_mask"].to(device),
        }

        # Forward pass
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)

        # Update metrics
        total_loss += loss.item()
        metric_tracker.update(predictions, targets)

    # Compute metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def main(config: TrainingConfig):
    """
    Main training function.

    Args:
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("SVAMITVA Feature Extraction - Training")
    logger.info("=" * 60)

    # Set seed
    set_seed(config.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info(f"Loading data from {config.train_data_dir}")
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    logger.info(f"Training samples: {len(train_loader.dataset)}")

    # Check if we have enough samples
    if len(train_loader.dataset) < 1:
        logger.error("Not enough training samples! Need at least 1 sample.")
        logger.info("Please ensure your dataset has:")
        logger.info("  - Orthophoto files (.tif)")
        logger.info("  - Matching annotations folder with shapefiles")
        return
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    logger.info(f"Creating model with {config.backbone} backbone")
    model = FeatureExtractorModel(
        backbone=config.backbone,
        pretrained=config.pretrained,
        num_roof_classes=config.num_roof_classes,
    )
    model = model.to(device)

    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Loss function
    criterion = MultiTaskLoss(
        building_weight=config.building_weight,
        roof_weight=config.roof_weight,
        road_weight=config.road_weight,
        waterbody_weight=config.waterbody_weight,
        utility_weight=config.utility_weight,
    )

    # Optimizer
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Learning rate scheduler
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.lr_min
        )
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=config.lr_patience
        )

    # Mixed precision scaler
    scaler = (
        torch.amp.GradScaler("cuda")
        if (config.mixed_precision and device.type == "cuda")
        else None
    )

    # Training loop
    best_metric = 0.0
    patience_counter = 0
    val_metrics = {}  # Initialize to avoid reference before assignment

    logger.info("\nStarting training...")
    logger.info("-" * 60)

    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Avg IoU: {train_metrics['avg_iou']:.4f}")

        # Validate
        if val_loader and (epoch + 1) % config.eval_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, criterion, device)

            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Avg IoU: {val_metrics['avg_iou']:.4f}")
            logger.info(f"Val Building IoU: {val_metrics['building_iou']:.4f}")
            logger.info(f"Val Road IoU: {val_metrics['road_iou']:.4f}")
            logger.info(f"Val Waterbody IoU: {val_metrics['waterbody_iou']:.4f}")
            logger.info(f"Val Utility IoU: {val_metrics['utility_iou']:.4f}")
            logger.info(f"Val Roof Accuracy: {val_metrics['roof_accuracy']:.4f}")

            # Check if best model
            current_metric = val_metrics[config.metric_for_best]
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

                # Save best model
                checkpoint_path = config.checkpoint_dir / f"best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "config": config.__dict__,
                    },
                    checkpoint_path,
                )

                logger.info(f"✓ Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1

            # Early stopping
            if config.early_stopping and patience_counter >= config.patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Step scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if val_loader:
                scheduler.step(val_metrics[config.metric_for_best])
        else:
            scheduler.step()

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Learning Rate: {current_lr:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

        # Save last model (every epoch)
        final_checkpoint_path = config.checkpoint_dir / "last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics if val_metrics else {},
                "config": config.__dict__,
            },
            final_checkpoint_path,
        )
        logger.info(f"Saved last model to {final_checkpoint_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best {config.metric_for_best}: {best_metric:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" Train SVAMITVA Feature Extraction Model"
    )
    parser.add_argument(
        "--train_dir", type=str, default="dataset/train", help="Training data directory"
    )
    parser.add_argument(
        "--val_dir", type=str, default=None, help="Validation data directory"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name")
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--quick_test", action="store_true", help="Use quick test config for debugging"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Create config
    if args.quick_test:
        from training.config import get_quick_test_config

        config = get_quick_test_config()
    else:
        config = get_config_from_args(args)

    # Run training
    main(config)
