"""
Training script for SVAMITVA feature extraction model.

Supports all 10 shapefile tasks with multi-task learning.

Usage:
    python train.py --train_dir /data/maps --batch_size 8 --epochs 100
    python train.py --train_dir /data/maps --val_dir /data/maps_val --epochs 50
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

# ★ Set before any CUDA context is created
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.feature_extractor import FeatureExtractorModel
from models.losses import MultiTaskLoss
from training.config import TrainingConfig, get_config_from_args
from training.metrics import MetricTracker
from data.dataset import create_dataloaders
from utils.checkpoint import CheckpointManager, resume_training
from utils.logging_config import setup_logging

# Setup logger properly using the corrected config safely
logger = setup_logging()

# ── All binary mask keys to move to GPU ───────────────────────────────────────
TARGET_MASK_KEYS = [
    "building_mask",
    "road_mask",
    "road_centerline_mask",
    "waterbody_mask",
    "waterbody_line_mask",
    "waterbody_point_mask",
    "utility_line_mask",
    "utility_poly_mask",
    "bridge_mask",
    "railway_mask",
    "roof_type_mask",
]


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def move_targets(batch: dict, device: torch.device) -> dict:
    """Move all target masks to device, skipping metadata/string fields."""
    targets = {}
    for k in TARGET_MASK_KEYS:
        if k in batch:
            targets[k] = batch[k].to(device)
    return targets


# ── One epoch ─────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    config: TrainingConfig,
    amp_device: str = "cpu",
) -> dict:
    model.train()
    metrics = MetricTracker()

    # Use tensors for accumulation to prevent CUDA syncs
    running_loss = torch.tensor(0.0, device=device)
    running_task_losses: dict = {}
    step = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = move_targets(batch, device)

        optimizer.zero_grad(set_to_none=True)  # More performant than zero_grad()

        with autocast(
            device_type=amp_device,
            enabled=config.mixed_precision and amp_device == "cuda",
        ):
            predictions = model(images)
            total_loss, task_losses = loss_fn(predictions, targets)

        # ★ NaN guard: finite check strictly replacing isnan/isinf
        if not torch.isfinite(total_loss):
            logger.warning(
                f"Epoch {epoch} step {step}: total_loss is NaN/Inf — skipping"
            )
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue

        scaler.scale(total_loss).backward()

        if config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.detach()
        for k, v in task_losses.items():
            if k not in running_task_losses:
                running_task_losses[k] = torch.tensor(0.0, device=device)
            running_task_losses[k] += v.detach()

        metrics.update(predictions, targets)
        step += 1

        if step % config.log_every_n_steps == 0:
            # Sync needed for log only periodically
            task_loss_str = "  ".join(
                f"{k}={(v.item()/step):.4f}" for k, v in running_task_losses.items()
            )
            logger.info(
                f"Epoch {epoch} [{step}/{len(loader)}]  "
                f"loss={(running_loss.item()/step):.4f}  {task_loss_str}"
            )

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = running_loss.item() / max(step, 1)

    for k, v in running_task_losses.items():
        epoch_metrics[f"loss_{k}"] = v.item() / max(step, 1)

    return epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    config: TrainingConfig,
    amp_device: str = "cpu",
) -> dict:
    model.eval()
    metrics = MetricTracker()
    running_loss = torch.tensor(0.0, device=device)
    step = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = move_targets(batch, device)

        with autocast(
            device_type=amp_device,
            enabled=config.mixed_precision and amp_device == "cuda",
        ):
            predictions = model(images)
            total_loss, _ = loss_fn(predictions, targets)

        running_loss += total_loss.detach()
        metrics.update(predictions, targets)
        step += 1

    val_metrics = metrics.compute()
    val_metrics["loss"] = running_loss.item() / max(step, 1)
    return val_metrics


# ── Main training loop ────────────────────────────────────────────────────────


def train(config: TrainingConfig):
    set_seed(config.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        val_split=config.val_split,
    )
    logger.info(
        f"Data: {len(train_loader)} train batches"
        + (f", {len(val_loader)} val batches" if val_loader else "")
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FeatureExtractorModel(
        backbone=config.backbone,
        pretrained=config.pretrained,
        num_roof_classes=config.num_roof_classes,
    ).to(device)
    logger.info(f"Parameters: {model.get_num_parameters():,}")

    if config.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        logger.info(
            f"Backbone frozen targeted for {config.freeze_backbone_epochs} epochs"
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = MultiTaskLoss(
        building_weight=config.building_weight,
        roof_weight=config.roof_weight,
        road_weight=config.road_weight,
        waterbody_weight=config.waterbody_weight,
        road_centerline_weight=config.road_centerline_weight,
        waterbody_line_weight=config.waterbody_line_weight,
        waterbody_point_weight=config.waterbody_point_weight,
        utility_line_weight=config.utility_line_weight,
        utility_poly_weight=config.utility_poly_weight,
        bridge_weight=config.bridge_weight,
        railway_weight=config.railway_weight,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler: torch.optim.lr_scheduler.LRScheduler
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.lr_min
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.lr_patience,
            factor=0.5,
            min_lr=config.lr_min,
        )

    # GradScaler logic fixed
    scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")

    ckpt_manager = CheckpointManager(
        config.checkpoint_dir,
        metric=config.metric_for_best,
        keep_top_k=config.save_top_k,
    )

    start_epoch = 1
    best_metric = 0.0
    no_improve = 0

    # ── Resume Logic ──────────────────────────────────────────────────────────
    latest_ckpt = ckpt_manager.get_latest_checkpoint()
    if latest_ckpt is not None:
        logger.info(f"Resuming from found checkpoint: {latest_ckpt}")
        start_epoch, loaded_metrics = resume_training(
            latest_ckpt, model, optimizer, scheduler, scaler, device=str(device)
        )
        best_metric = loaded_metrics.get(config.metric_for_best, 0.0)

        # Robustly determine if backbone should be frozen or unfrozen upon resume
        if (
            start_epoch > config.freeze_backbone_epochs
            and config.freeze_backbone_epochs > 0
        ):
            model.unfreeze_backbone()
            logger.info(
                "Resumed epoch implies backbone should be unfrozen. Proceeding unfrozen."
            )

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.num_epochs + 1):
        t0 = time.time()

        # Unfreeze backbone based on config strictly
        if (
            epoch == config.freeze_backbone_epochs + 1
            and config.freeze_backbone_epochs > 0
        ):
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen")

        # Training
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            epoch,
            config,
            amp_device=amp_device,
        )

        # Validation
        val_metrics = {}
        if val_loader is not None and epoch % config.eval_every_n_epochs == 0:
            val_metrics = validate(
                model,
                val_loader,
                loss_fn,
                device,
                config,
                amp_device=amp_device,
            )
            current = val_metrics.get(config.metric_for_best, 0.0)
            is_best = current > best_metric

            # Log
            iou_str = "  ".join(
                f"{k}={v:.3f}" for k, v in val_metrics.items() if k.endswith("_iou")
            )
            logger.info(
                f"Epoch {epoch}/{config.num_epochs}  "
                f"val_loss={val_metrics.get('loss', 0):.4f}  "
                f"avg_iou={val_metrics.get('avg_iou', 0):.4f}  "
                f"time={time.time()-t0:.1f}s"
            )
            logger.info(f"  Per-task IoU: {iou_str}")

            # Save atomic checkpoint incorporating scheduler & scaler properly
            ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config.__dict__,
                is_best=is_best,
                scheduler=scheduler,
                scaler=scaler,
            )

            if is_best:
                best_metric = current
                no_improve = 0
            else:
                if config.early_stopping:
                    no_improve += 1
                    if no_improve >= config.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(no improvement for {no_improve} epochs)"
                        )
                        break
        else:
            logger.info(
                f"Epoch {epoch}/{config.num_epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"time={time.time()-t0:.1f}s"
            )

        # Scheduler step handled correctly based on data availability
        if config.scheduler == "reduce_on_plateau":
            if val_loader is not None and val_metrics:
                scheduler.step(val_metrics.get(config.metric_for_best, 0.0))
        else:
            scheduler.step()

    logger.info(f"Training complete. Best {config.metric_for_best}: {best_metric:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train SVAMITVA model")
    p.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Root dir containing MAP1, MAP2, … folders",
    )
    p.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Validation dir (optional; else auto-split)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader (use 0 for Jupyter/shared servers)",
    )
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--name", type=str, default="baseline")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args)
    # Propagate any CLI overrides
    cfg.backbone = args.backbone
    cfg.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
