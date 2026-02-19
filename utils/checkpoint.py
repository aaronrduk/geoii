"""
Checkpoint management utilities.

Handles saving and loading model checkpoints with proper error handling.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for model checkpoints.

    Handles:
    - Saving checkpoints
    - Loading checkpoints
    - Checkpoint versioning
    - Automatic cleanup of old checkpoints
    """

    def __init__(self, checkpoint_dir: Path, keep_top_k: int = 3):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_top_k: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k

        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Dict,
        config: Dict,
        is_best: bool = False,
        scheduler=None,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Validation metrics
            config: Training configuration
            is_best: Whether this is the best model
            scheduler: Learning rate scheduler (optional)

        Returns:
            Path: Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            self.metadata["checkpoints"].append(
                {
                    "epoch": epoch,
                    "path": str(checkpoint_path),
                    "metrics": metrics,
                    "is_best": is_best,
                }
            )

            if is_best:
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")

            self._save_metadata()
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self, checkpoint_path: Path, model, optimizer=None, scheduler=None, device="cpu"
    ) -> Tuple[int, Dict]:
        """
        Load checkpoint and restore model state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load on

        Returns:
            tuple: (epoch, metrics)
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            metrics = checkpoint.get("metrics", {})

            logger.info(f"Loaded checkpoint from epoch {epoch}")

            return epoch, metrics

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.metadata["checkpoints"]:
            return None

        latest = max(self.metadata["checkpoints"], key=lambda x: x["epoch"])
        return Path(latest["path"])

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            return best_path
        return None

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        return {"checkpoints": []}

    def _save_metadata(self):
        """Save checkpoint metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only top K."""
        if len(self.metadata["checkpoints"]) <= self.keep_top_k:
            return

        sorted_checkpoints = sorted(
            self.metadata["checkpoints"], key=lambda x: x["epoch"], reverse=True
        )

        to_keep = sorted_checkpoints[: self.keep_top_k]
        to_remove = sorted_checkpoints[self.keep_top_k :]

        for checkpoint in to_remove:
            if checkpoint.get("is_best", False):
                continue

            path = Path(checkpoint["path"])
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {path}: {e}")

        self.metadata["checkpoints"] = to_keep
        self._save_metadata()


def resume_training(
    checkpoint_path: Path, model, optimizer, scheduler=None, device="cpu"
) -> Tuple[int, Dict]:
    """
    Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore (optional)
        device: Device to load on

    Returns:
        tuple: (start_epoch, best_metrics)
    """
    manager = CheckpointManager(checkpoint_path.parent)
    start_epoch, metrics = manager.load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )

    return start_epoch + 1, metrics
