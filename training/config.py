"""
Training configuration for SVAMITVA feature extraction model.

This module defines all hyperparameters and settings for training.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration class for training.

    Student note: Using dataclasses makes configuration clean and type-safe.
    All hyperparameters are documented here for easy tuning.
    """

    # --- Paths ---
    train_data_dir: str = "dataset/train"
    val_data_dir: Optional[str] = None  # If None, will split from train
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Model Architecture ---
    backbone: str = "resnet50"  # 'resnet50' or 'resnet34'
    pretrained: bool = True  # Use ImageNet pretrained weights
    num_roof_classes: int = 5  # Background + 4 roof types

    # --- Training Hyperparameters ---
    batch_size: int = 8  # Adjust based on GPU memory
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', or 'reduce_on_plateau'
    lr_min: float = 1e-6  # Minimum learning rate for cosine annealing
    lr_patience: int = 10  # For ReduceLROnPlateau

    # --- Data ---
    image_size: int = 512  # Input image size (square)
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True

    # Train/val split (if val_data_dir is None)
    val_split: float = 0.2  # 20% for validation

    # --- Loss Weights ---
    # Student note: These weights balance different tasks
    # Tune these if one task is performing poorly
    building_weight: float = 1.0
    roof_weight: float = 0.5
    road_weight: float = 0.8
    waterbody_weight: float = 0.8
    utility_weight: float = 0.6

    # --- Optimization ---
    optimizer: str = "adamw"  # 'adam' or 'adamw'
    gradient_clip: float = 1.0  # Gradient clipping for stability
    mixed_precision: bool = True  # Use AMP for faster training

    # --- Evaluation ---
    eval_every_n_epochs: int = 1  # Evaluate every N epochs
    save_top_k: int = 3  # Save top K best checkpoints
    metric_for_best: str = "avg_iou"  # Metric to determine best model

    # --- Early Stopping ---
    early_stopping: bool = True
    patience: int = 20  # Epochs without improvement before stopping

    # --- Logging ---
    log_every_n_steps: int = 50  # Log training metrics every N steps
    use_wandb: bool = False  # Use Weights & Biases for logging
    wandb_project: str = "svamitva-extraction"
    experiment_name: str = "baseline"

    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True

    # --- Advanced ---
    freeze_backbone_epochs: int = 0  # Freeze backbone for first N epochs
    use_tta: bool = False  # Test-time augmentation for evaluation

    def __post_init__(self):
        """Validate and convert paths to Path objects."""
        self.train_data_dir = Path(self.train_data_dir)
        if self.val_data_dir:
            self.val_data_dir = Path(self.val_data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Preset configurations for different scenarios


def get_quick_test_config() -> TrainingConfig:
    """
    Configuration for quick testing (small model, few epochs).

    Student note: Use this for debugging and quick iterations.
    """
    return TrainingConfig(
        backbone="resnet34",  # Smaller model
        batch_size=4,
        num_epochs=5,
        image_size=256,  # Smaller images
        num_workers=2,
        eval_every_n_epochs=1,
    )


def get_full_training_config() -> TrainingConfig:
    """
    Configuration for full training (for hackathon submission).

    Student note: Use this for final model training.
    """
    return TrainingConfig(
        backbone="resnet50",
        batch_size=8,
        num_epochs=100,
        image_size=512,
        num_workers=4,
        mixed_precision=True,
        early_stopping=True,
        patience=20,
    )


def get_config_from_args(args) -> TrainingConfig:
    """
    Create configuration from command-line arguments.

    Args:
        args: argparse.Namespace with arguments

    Returns:
        TrainingConfig: Configuration object
    """
    return TrainingConfig(
        train_data_dir=getattr(args, "train_dir", "dataset/train"),
        val_data_dir=getattr(args, "val_dir", None),
        batch_size=getattr(args, "batch_size", 8),
        num_epochs=getattr(args, "epochs", 100),
        learning_rate=getattr(args, "lr", 0.001),
        image_size=getattr(args, "image_size", 512),
        experiment_name=getattr(args, "name", "baseline"),
        use_wandb=getattr(args, "wandb", False),
    )
