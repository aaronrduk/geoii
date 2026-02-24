"""
Training configuration for SVAMITVA feature extraction model.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    All hyperparameters and settings for training.
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    train_data_dir: str = "dataset/train"
    val_data_dir: Optional[str] = None  # If None, split from train
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Model Architecture ────────────────────────────────────────────────────
    backbone: str = "resnet50"
    pretrained: bool = True
    num_roof_classes: int = 5  # background + 4 roof types

    # ── Training Hyperparameters ──────────────────────────────────────────────
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    scheduler: str = "cosine"  # 'cosine', 'step', 'reduce_on_plateau'
    lr_min: float = 1e-6
    lr_patience: int = 10

    # ── Data ──────────────────────────────────────────────────────────────────
    image_size: int = 512
    num_workers: int = 0  # 0 = main process; required for Jupyter/shared servers
    pin_memory: bool = True
    val_split: float = 0.2

    # ── Loss Weights (original tasks) ─────────────────────────────────────────
    building_weight: float = 1.0
    roof_weight: float = 0.5
    road_weight: float = 0.8
    waterbody_weight: float = 0.8

    # ── Loss Weights (new tasks) ──────────────────────────────────────────────
    road_centerline_weight: float = 0.7
    waterbody_line_weight: float = 0.7
    waterbody_point_weight: float = 0.9  # sparse → upweighted
    utility_line_weight: float = 0.7
    utility_poly_weight: float = 0.8
    bridge_weight: float = 1.0  # rare → highest weight
    railway_weight: float = 0.9

    # ── Optimization ──────────────────────────────────────────────────────────
    optimizer: str = "adamw"
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_every_n_epochs: int = 1
    save_top_k: int = 3
    metric_for_best: str = "avg_iou"

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping: bool = True
    patience: int = 20

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "svamitva-extraction"
    experiment_name: str = "baseline"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    deterministic: bool = True

    # ── Advanced ──────────────────────────────────────────────────────────────
    freeze_backbone_epochs: int = 0
    use_tta: bool = False

    def __post_init__(self):
        self.train_data_dir = Path(self.train_data_dir)
        if self.val_data_dir:
            self.val_data_dir = Path(self.val_data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_quick_test_config() -> TrainingConfig:
    """Quick config for debugging (small model, 5 epochs)."""
    return TrainingConfig(
        backbone="resnet34",
        batch_size=4,
        num_epochs=5,
        image_size=256,
        num_workers=0,
        eval_every_n_epochs=1,
    )


def get_full_training_config() -> TrainingConfig:
    """Full config for final model training."""
    return TrainingConfig(
        backbone="resnet50",
        batch_size=8,
        num_epochs=100,
        image_size=512,
        num_workers=0,
        mixed_precision=True,
        early_stopping=True,
        patience=20,
    )


def get_config_from_args(args) -> TrainingConfig:
    # ★ Bug 11 fix: propagate all CLI arguments including num_workers
    return TrainingConfig(
        train_data_dir=getattr(args, "train_dir", "dataset/train"),
        val_data_dir=getattr(args, "val_dir", None),
        batch_size=getattr(args, "batch_size", 8),
        num_epochs=getattr(args, "epochs", 100),
        learning_rate=getattr(args, "lr", 0.001),
        image_size=getattr(args, "image_size", 512),
        num_workers=getattr(args, "num_workers", 0),
        experiment_name=getattr(args, "name", "baseline"),
        use_wandb=getattr(args, "wandb", False),
    )
