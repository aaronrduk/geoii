"""
Training script for SVAMITVA feature extraction model.
Supports all 10 shapefile tasks with multi-task learning.

Usage:
    # Full training
    python train.py --train_dir /data/maps --batch_size 8 --epochs 100

    # With separate validation folder
    python train.py --train_dir /data/maps --val_dir /data/maps_val --epochs 50

    # Quick smoke-test (5 epochs, resnet34)
    python train.py --train_dir /data/maps --quick_test
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.dataset import create_dataloaders
from models.feature_extractor import FeatureExtractor
from models.losses import MultiTaskLoss
from training.config import TrainingConfig, get_config_from_args
from training.metrics import MetricTracker
from utils.checkpoint import CheckpointManager, resume_training
from utils.logging_config import setup_logging

# Set before any CUDA context is created
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = setup_logging()

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
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def move_targets(batch: dict, device: torch.device) -> dict:
    """Move all target mask tensors to device, skip metadata / string fields."""
    return {k: batch[k].to(device) for k in TARGET_MASK_KEYS if k in batch}


# ── Training epoch ────────────────────────────────────────────────────────────


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
    running_loss = torch.tensor(0.0, device=device)
    running_task: dict = {}
    step = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = move_targets(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=amp_device,
            enabled=config.mixed_precision and amp_device == "cuda",
        ):
            predictions = model(images)
            total_loss, task_losses = loss_fn(predictions, targets)

        if not torch.isfinite(total_loss):
            logger.warning(f"Epoch {epoch} step {step}: NaN/Inf loss — skipping step")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue

        scaler.scale(total_loss).backward()

        if config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.detach()
        for k, v in task_losses.items():
            running_task.setdefault(k, torch.tensor(0.0, device=device))
            running_task[k] += v.detach()

        metrics.update(predictions, targets)
        step += 1

        if step % config.log_every_n_steps == 0:
            task_str = "  ".join(
                f"{k}={(v.item()/step):.4f}" for k, v in running_task.items()
            )
            logger.info(
                f"Epoch {epoch} [{step}/{len(loader)}]  "
                f"loss={(running_loss.item()/step):.4f}  {task_str}"
            )

    result = metrics.compute()
    result["loss"] = running_loss.item() / max(step, 1)
    for k, v in running_task.items():
        result[f"loss_{k}"] = v.item() / max(step, 1)

    return result


# ── Validation epoch ──────────────────────────────────────────────────────────


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

    result = metrics.compute()
    result["loss"] = running_loss.item() / max(step, 1)
    return result


# ── Main training loop ────────────────────────────────────────────────────────


def train(config: TrainingConfig):
    set_seed(config.seed, deterministic=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_device = "mps"
    else:
        device = torch.device("cpu")
        amp_device = "cpu"

    logger.info(f"Device: {device}")

    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            g = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {g.name}  ({g.total_memory/1e9:.1f} GB)")
    elif device.type == "mps":
        logger.info("  Using Apple Silicon GPU (MPS)")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        val_split=config.val_split,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FeatureExtractor(
        backbone=config.backbone,
        pretrained=config.pretrained,
        num_roof_classes=config.num_roof_classes,
    ).to(device)
    logger.info(f"Parameters: {model.get_num_parameters():,}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    if config.freeze_backbone_epochs > 0:
        (
            model.module if isinstance(model, nn.DataParallel) else model
        ).freeze_backbone()
        logger.info(f"Backbone frozen for first {config.freeze_backbone_epochs} epochs")

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
    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.optimizer.lower() == "adamw"
        else torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    if config.scheduler == "cosine":
        # Cosine annealing with optional linear warmup
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.lr_min
        )
        warmup_epochs = getattr(config, "warmup_epochs", 0)
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.lr_patience, factor=0.5, min_lr=config.lr_min
        )

    scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    ckpt_manager = CheckpointManager(
        config.checkpoint_dir,
        metric=config.metric_for_best,
        keep_top_k=config.save_top_k,
    )

    start_epoch = 1
    best_metric = 0.0
    no_improve = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    latest_ckpt = ckpt_manager.get_latest_checkpoint()
    if latest_ckpt is not None:
        logger.info(f"Resuming from {latest_ckpt}")
        start_epoch, loaded_metrics = resume_training(
            latest_ckpt, model, optimizer, scheduler, scaler, device=str(device)
        )
        best_metric = loaded_metrics.get(config.metric_for_best, 0.0)
        inner = model.module if isinstance(model, nn.DataParallel) else model
        if start_epoch > config.freeze_backbone_epochs > 0:
            inner.unfreeze_backbone()
            logger.info("Resumed: backbone unfrozen")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.num_epochs + 1):
        t0 = time.time()

        # Unfreeze backbone at scheduled epoch
        inner = model.module if isinstance(model, nn.DataParallel) else model
        if (
            epoch == config.freeze_backbone_epochs + 1
            and config.freeze_backbone_epochs > 0
        ):
            inner.unfreeze_backbone()
            logger.info("Backbone unfrozen")

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
                model, val_loader, loss_fn, device, config, amp_device=amp_device
            )
            current = val_metrics.get(config.metric_for_best, 0.0)
            is_best = current > best_metric

            iou_str = "  ".join(
                f"{k}={v:.3f}" for k, v in val_metrics.items() if k.endswith("_iou")
            )
            logger.info(
                f"Epoch {epoch}/{config.num_epochs}  "
                f"val_loss={val_metrics.get('loss', 0):.4f}  "
                f"avg_iou={val_metrics.get('avg_iou', 0):.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"time={time.time()-t0:.1f}s"
            )
            logger.info(f"  Per-task: {iou_str}")

            ckpt_manager.save_checkpoint(
                model=inner,
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
                no_improve += 1
                if config.early_stopping and no_improve >= config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} ({no_improve} epochs without improvement)"
                    )
                    break
        else:
            logger.info(
                f"Epoch {epoch}/{config.num_epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"time={time.time()-t0:.1f}s"
            )

        # Scheduler step
        if config.scheduler == "reduce_on_plateau":
            if val_metrics:
                scheduler.step(val_metrics.get(config.metric_for_best, 0.0))
        else:
            scheduler.step()

    logger.info(f"Training complete. Best {config.metric_for_best}: {best_metric:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train SVAMITVA feature extraction model")
    p.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Root dir with MAP1, MAP2, … folders",
    )
    p.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Validation dir (optional; else auto-split)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument(
        "--backbone", type=str, default="resnet50", choices=["resnet50", "resnet34"]
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="0 = main process (required for Jupyter/shared servers)",
    )
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--name", type=str, default="baseline")
    p.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick 5-epoch smoke-test with resnet34",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args)
    cfg.backbone = args.backbone
    cfg.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.quick_test:
        cfg.backbone = "resnet34"
        cfg.num_epochs = 5
        cfg.image_size = 256
        cfg.batch_size = 4
        logger.info("Quick-test mode: resnet34, 5 epochs, 256px")

    train(cfg)
