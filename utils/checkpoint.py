"""
Checkpoint management utilities.

Handles saving and loading model checkpoints safely.
Ensures atomic writes and comprehensive state resumption.
"""

import torch
import logging
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def save_inference_model(model: torch.nn.Module, path: Path | str):
    """Save ONLY the model's state_dict to a file for minimal inference size."""
    model_cpu = getattr(model, "module", model).to("cpu")
    torch.save(model_cpu.state_dict(), path)


def save_half_precision_model(model: torch.nn.Module, path: Path | str):
    """Save ONLY the model's state_dict in FP16 for even smaller inference size."""
    model_cpu = getattr(model, "module", model).to("cpu").half()
    torch.save(model_cpu.state_dict(), path)


class CheckpointManager:
    """
    Manager for model checkpoints.

    Handles:
    - Atomic saving of checkpoints
    - Loading checkpoints with full state restoration
    - Automatic cleanup of old checkpoints (saving the best)
    """

    def __init__(
        self, checkpoint_dir: Path, metric: str = "avg_iou", keep_top_k: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric

        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        config: Dict,
        is_best: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Path:
        """
        Save model checkpoint atomically.
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

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        checkpoint_train_path = (
            self.checkpoint_dir / f"checkpoint_epoch_{epoch}_train.pth"
        )
        checkpoint_inf_path = (
            self.checkpoint_dir / f"checkpoint_epoch_{epoch}_inference.pth"
        )

        try:
            # Atomic save: write to temp file first, then replace
            fd, temp_path = tempfile.mkstemp(
                dir=self.checkpoint_dir, prefix="tmp_ckpt_", suffix=".pth"
            )
            os.close(fd)
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_train_path)

            # Save inference checkpoint natively
            save_inference_model(model, checkpoint_inf_path)

            logger.info(f"Saved checkpoint to {checkpoint_train_path}")

            # Update metadata safely
            checkpoint_info = {
                "epoch": epoch,
                "path": str(checkpoint_train_path.name),  # Store relative path
                "inf_path": str(checkpoint_inf_path.name),
                "metrics": metrics,
                "is_best": is_best,
            }

            # Remove existing entry for same epoch if it exists
            self.metadata["checkpoints"] = [
                c for c in self.metadata["checkpoints"] if c["epoch"] != epoch
            ]
            self.metadata["checkpoints"].append(checkpoint_info)

            if is_best:
                best_train_path = self.checkpoint_dir / "best_model_train.pth"
                best_inf_path = self.checkpoint_dir / "best_model.pth"
                # Atomic save for best model
                fd_best, temp_best = tempfile.mkstemp(
                    dir=self.checkpoint_dir, prefix="tmp_best_", suffix=".pth"
                )
                os.close(fd_best)
                torch.save(checkpoint, temp_best)
                os.replace(temp_best, best_train_path)

                # Save best natively
                save_inference_model(model, best_inf_path)
                logger.info(f"Saved best inference model to {best_inf_path}")

            self._save_metadata()
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: str = "cpu",
    ) -> Tuple[int, Dict]:
        """
        Load checkpoint and restore full training state.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path} to {device}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            metrics = checkpoint.get("metrics", {})

            logger.info(f"Successfully loaded checkpoint from epoch {epoch}")

            return epoch, metrics

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.metadata["checkpoints"]:
            return None

        latest = max(self.metadata["checkpoints"], key=lambda x: x["epoch"])
        path = self.checkpoint_dir / latest["path"]
        return path if path.exists() else None

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
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")

        return {"checkpoints": []}

    def _save_metadata(self):
        """Save checkpoint metadata atomically."""
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=self.checkpoint_dir, prefix="tmp_meta_", suffix=".json"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(self.metadata, f, indent=2)
            os.replace(temp_path, self.metadata_file)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only top K."""
        if len(self.metadata["checkpoints"]) <= self.keep_top_k:
            return

        # Sort by metric (descending) so the best ones are kept
        # Fallback to epoch if metric is missing
        def get_score(ckpt):
            metrics = ckpt.get("metrics", {})
            return metrics.get(self.metric, ckpt.get("epoch", 0))

        sorted_checkpoints = sorted(
            self.metadata["checkpoints"], key=get_score, reverse=True
        )

        to_keep = sorted_checkpoints[: self.keep_top_k]
        to_remove = sorted_checkpoints[self.keep_top_k :]

        # Enforce that is_best is kept if somehow not in top score
        kept_epochs = {c["epoch"] for c in to_keep}
        final_remove = []

        for checkpoint in to_remove:
            if checkpoint.get("is_best", False):
                to_keep.append(checkpoint)
                kept_epochs.add(checkpoint["epoch"])
            else:
                final_remove.append(checkpoint)

        for checkpoint in final_remove:
            paths_to_remove = [self.checkpoint_dir / checkpoint["path"]]
            if "inf_path" in checkpoint:
                paths_to_remove.append(self.checkpoint_dir / checkpoint["inf_path"])

            for path in paths_to_remove:
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"Removed old checkpoint: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {path}: {e}")

        self.metadata["checkpoints"] = to_keep
        self._save_metadata()


def resume_training(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = "cpu",
) -> Tuple[int, Dict]:
    """
    Resume training from checkpoint.
    """
    manager = CheckpointManager(checkpoint_path.parent)
    start_epoch, metrics = manager.load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, scaler, device
    )

    return start_epoch + 1, metrics
