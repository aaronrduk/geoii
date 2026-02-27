import os
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

NOTEBOOK_DIR = Path.cwd()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

DATA_DIR = Path("/Users/aaronr/Desktop/DATA")
CKPT_DIR = NOTEBOOK_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
(NOTEBOOK_DIR / "logs").mkdir(exist_ok=True)

# Device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Apple Silicon GPU (MPS) \u2705")
else:
    device = torch.device("cpu")
    print("Running on CPU \u26a0\ufe0f")

print(f"DATA: {DATA_DIR}  (exists={DATA_DIR.exists()})")

CONFIG = dict(
    backbone="resnet34",  # Lighter for local test
    pretrained=True,
    image_size=256,  # Smaller for local test
    batch_size=4,
    epochs_per_map=5,  # Less epochs for local test
    learning_rate=2e-4,
    weight_decay=1e-4,
    num_workers=0,
    mixed_precision=False,  # MPS AMP unstable
    gradient_clip=1.0,
    building_weight=1.0,
    roof_weight=0.5,
    road_weight=0.8,
    waterbody_weight=0.8,
    road_centerline_weight=0.7,
    waterbody_line_weight=0.7,
    waterbody_point_weight=0.9,
    utility_line_weight=0.7,
    utility_poly_weight=0.8,
    bridge_weight=1.0,
    railway_weight=0.9,
)

TARGET_KEYS = [
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

print("\nSetup complete \u2713")

from models.feature_extractor import FeatureExtractor
from models.losses import MultiTaskLoss
from training.metrics import MetricTracker
from data.dataset import SvamitvaDataset
from data.augmentation import get_train_transforms


def move_targets(batch):
    return {k: batch[k].to(device) for k in TARGET_KEYS if k in batch}


def build_model(load_from: Path = None):
    m = FeatureExtractor(
        backbone=CONFIG["backbone"],
        pretrained=CONFIG["pretrained"],
        num_roof_classes=5,
    )
    if load_from and load_from.exists():
        state = torch.load(load_from, map_location="cpu", weights_only=False)
        weights = state.get("model") or state.get("model_state_dict") or state
        m.load_state_dict(weights, strict=False)
        print(f"Loaded: {load_from.name}")
    return m.to(device)


def train_map(map_name: str, resume_from: Path = None):
    map_dir = DATA_DIR / map_name
    best_out = CKPT_DIR / f"{map_name}_best.pt"
    last_out = CKPT_DIR / f"{map_name}_latest.pt"

    if not map_dir.exists():
        print(f"[SKIP] {map_dir} not found")
        return best_out if best_out.exists() else None

    print(f"\n{'='*70}")
    print(f"  Region     : {map_name}")
    print(
        f"  Checkpoint : {resume_from.name if resume_from and resume_from.exists() else 'SCRATCH'}"
    )
    print(f"{'='*70}")

    model_w = build_model(load_from=resume_from)

    try:
        ds = SvamitvaDataset(
            root_dir=DATA_DIR,
            image_size=CONFIG["image_size"],
            transform=get_train_transforms(CONFIG["image_size"]),
            mode="train",
        )
        ds.samples = [s for s in ds.samples if s["map_name"] == map_name]
        print(f"  Tiles (KMeans Filtered): {len(ds)}")
    except Exception as e:
        print(f"Dataset failed for {map_name}: {e}")
        return None

    loader = DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    loss_fn = MultiTaskLoss(
        **{k: v for k, v in CONFIG.items() if k.endswith("_weight")}
    ).to(device)
    optimizer = torch.optim.AdamW(
        model_w.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs_per_map"], eta_min=1e-6
    )

    best_iou = 0.0
    for epoch in range(1, CONFIG["epochs_per_map"] + 1):
        model_w.train()
        tracker = MetricTracker()
        run_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch in loader:
            images = batch["image"].to(device)
            targets = move_targets(batch)
            optimizer.zero_grad(set_to_none=True)

            # NaN check the images
            if not torch.isfinite(images).all():
                print("  [NaN SKIP] NaN in images detected")
                continue

            preds = model_w(images)
            total_loss, loss_d = loss_fn(preds, targets)

            if not torch.isfinite(total_loss):
                print(f"  [NaN SKIP] epoch {epoch} loss is NaN")
                continue

            total_loss.backward()
            if CONFIG["gradient_clip"] > 0:
                nn.utils.clip_grad_norm_(model_w.parameters(), CONFIG["gradient_clip"])
            optimizer.step()

            run_loss += total_loss.item()
            tracker.update(preds, targets)
            n_steps += 1

            if n_steps >= 10:
                print(
                    "  [QUICK TEST] Stopping early after 10 steps to save checkpoint."
                )
                break

        scheduler.step()
        m = tracker.compute()
        avg_loss = run_loss / max(n_steps, 1)
        avg_iou = m.get("avg_iou", 0.0)

        print(
            f"  Epoch {epoch:2d}/{CONFIG['epochs_per_map']} | loss: {avg_loss:.4f} | iou: {avg_iou:.4f} | {time.time()-t0:.0f}s"
        )

        ckpt = {
            "model": model_w.state_dict(),
            "epoch": epoch,
            "map_name": map_name,
            "best_iou": best_iou,
            "metrics": m,
        }
        torch.save(ckpt, last_out)
        if avg_iou > best_iou:
            best_iou = avg_iou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, best_out)
            print(f"    \u2192 New best! IoU = {best_iou:.4f}")

        # Break early for quick test
        break

    print(f"\n  [DONE] {map_name}  Best IoU={best_iou:.4f}")
    return best_out


print("Training engine ready \u2713")

cpt1 = train_map("MAP1", resume_from=None)
print("\n*** LOCAL TRAINING COMPLETE ***")
