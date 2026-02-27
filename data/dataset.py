"""
PyTorch Dataset for SVAMITVA orthophotos and annotations.

Design decisions:
- Each large MAP TIF is tiled into overlapping 512×512 patches so
  DataLoader gets many samples per MAP instead of one giant image.
- Shapefiles are matched first by explicit glob patterns, then by
  keyword search across all .shp files in the folder (handles
  non-standard naming like 'Abadi.shp', 'built_up.shp', etc.)
- GeoDataFrame is always reprojected to the TIF's CRS before
  rasterization so masks are never empty due to CRS mismatch.
- Bridge and railway shapefiles are optional — if absent, the mask
  stays all-zero (correct: "no such feature here").
- GeoDataFrames are cached per (map_name, task_key) so shapefiles
  are only read and reprojected once per dataset lifetime.

Dataset structure expected:
    DATA/
      MAP1/
        anything.tif
        Road.shp
        Built_Up_Area.shp
        ...
      MAP2/ ... MAP5/
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from .augmentation import get_train_transforms, get_val_transforms
from .preprocessing import OrthophotoPreprocessor, ShapefileAnnotationParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shapefile task definitions ─────────────────────────────────────────────────
# Each entry: (explicit_glob_patterns, keyword_fallback_list, mask_key)
# Keywords matched case-insensitively against all .shp filenames.
# Bridge and Railway are included but gracefully tolerated when absent.
SHAPEFILE_TASKS = [
    (
        ["Build_up*.shp", "Built_Up_Area*.shp", "Abadi*.shp"],
        ["build_up", "built_up", "built", "abadi", "building", "structure"],
        "building",
    ),
    (["Road.shp"], ["^road$"], "road"),
    (
        ["Road_centre_line*.shp", "Road_Centre_Line*.shp"],
        ["centre_line", "centerline", "centreline", "center_line"],
        "road_centerline",
    ),
    (
        ["Waterbody_1*.shp", "Water_Body.shp", "Waterbody.shp", "Water_Body_1.shp"],
        ["^waterbody$", "^waterbody_1$", "^water_body$", "pond", "lake"],
        "waterbody",
    ),
    (
        ["Waterbody_line_1*.shp", "Water_Body_Line*.shp", "Waterbody_Line*.shp"],
        ["waterbody_line", "water_body_line", "canal", "drain"],
        "waterbody_line",
    ),
    (
        ["Waterbody_point_1*.shp", "Waterbody_Point*.shp", "Water_Body_Point*.shp"],
        ["waterbody_point", "water_body_point", "well"],
        "waterbody_point",
    ),
    # utility_poly BEFORE utility_line to avoid keyword cross-match
    (
        ["Utility_poly_1*.shp", "Utility_Poly*.shp"],
        ["utility_poly", "utility_area", "transformer", "tank"],
        "utility_poly",
    ),
    (
        ["Utility_1*.shp", "Utility.shp", "Utility_Line*.shp"],
        ["^utility$", "^utility_1$", "utility_line", "pipeline", "wire"],
        "utility_line",
    ),
    # Optional — all-zero mask when absent
    (["Bridge*.shp", "Bridge.shp"], ["bridge"], "bridge"),
    (
        ["Railway*.shp", "Rail*.shp", "Railway.shp"],
        ["railway", "railroad", "rail"],
        "railway",
    ),
]

ALL_MASK_KEYS = [task for _, _, task in SHAPEFILE_TASKS] + ["roof_type"]

# Tasks where missing shapefiles produce a debug log, not a warning
OPTIONAL_TASKS = {"bridge", "railway"}

TILE_SIZE = 512
TILE_OVERLAP = 64


class SvamitvaDataset(Dataset):
    """
    PyTorch Dataset for SVAMITVA drone imagery.

    Returns 512×512 tile-level samples (not MAP-level).
    Each large orthophoto is split into overlapping tiles during __init__.
    """

    def __init__(
        self,
        root_dir: Path,
        image_size: int = TILE_SIZE,
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.ortho_preprocessor = OrthophotoPreprocessor()
        self.anno_parser = ShapefileAnnotationParser()
        self._gdf_cache: Dict[Tuple, gpd.GeoDataFrame] = {}  # keyed (map_name, task)

        self.samples = self._scan_dataset()
        logger.info(f"[{mode}] dataset ready: {len(self.samples)} tile samples")

    # ── Shapefile finder ───────────────────────────────────────────────────────

    def _find_shapefile(
        self,
        folder: Path,
        patterns: List[str],
        keywords: List[str],
        taken: set,
    ) -> Optional[Path]:
        """
        Find a shapefile by explicit glob patterns first,
        then keyword fallback across all .shp in folder.
        Skips files already claimed by another task.
        """
        # 1. Explicit glob
        for pat in patterns:
            for hit in sorted(folder.glob(pat)):
                if hit not in taken:
                    return hit

        # 2. Keyword fallback — case-insensitive substring / anchored match
        for shp in sorted(folder.glob("*.shp")):
            stem = shp.stem.lower()
            for kw in keywords:
                if kw.startswith("^") and kw.endswith("$"):
                    if stem == kw[1:-1] and shp not in taken:
                        return shp
                elif kw in stem and shp not in taken:
                    return shp

        return None

    # ── Tile calculator ────────────────────────────────────────────────────────

    def _compute_tiles(self, tif_path: Path):
        """Open the TIF, apply K-Means clustering to skip NoData, compute tile windows."""
        try:
            with rasterio.open(tif_path) as src:
                H, W = src.height, src.width
                tif_crs = src.crs
                tif_tf = src.transform

                # Fetch a thumbnail to perform K-Means quickly without OOM
                try:
                    thumb_size = 1024
                    scale = min(thumb_size / max(H, W), 1.0)
                    out_shape = (src.count, int(H * scale), int(W * scale))
                    thumb = src.read(
                        out_shape=out_shape,
                        resampling=rasterio.enums.Resampling.bilinear,
                    )
                    thumb_h, thumb_w = out_shape[1], out_shape[2]
                except Exception as e:
                    logger.warning(f"Could not read thumbnail for KMeans: {e}")
                    thumb = None
                    scale = 1.0

        except Exception as e:
            logger.error(f"Cannot open {tif_path}: {e}")
            return None

        # Apply K-Means
        valid_mask_thumb = None
        if thumb is not None and thumb.shape[0] >= 3:
            try:
                import cv2

                img_rgb = np.transpose(thumb[:3, :, :], (1, 2, 0))
                pixels = img_rgb.reshape(-1, 3).astype(np.float32)

                # K-Means clustering (k=3 or 4 usually separates background, vegetation, built-up)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = 3
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
                )

                # The corners of the orthophoto are almost always NoData background
                corner_indices = [
                    0,
                    thumb_w - 1,
                    (thumb_h - 1) * thumb_w,
                    (thumb_h - 1) * thumb_w + thumb_w - 1,
                ]
                corner_labels = labels.flatten()[corner_indices]

                from collections import Counter

                bg_label = Counter(corner_labels).most_common(1)[0][0]

                valid_mask_thumb = (labels.flatten() != bg_label).reshape(
                    thumb_h, thumb_w
                )
            except Exception as e:
                logger.warning(
                    f"K-Means clustering failed, falling back to full map: {e}"
                )
                valid_mask_thumb = None

        stride = TILE_SIZE - TILE_OVERLAP
        ys_all = list(range(0, max(H - TILE_SIZE, 0) + 1, stride)) or [0]
        xs_all = list(range(0, max(W - TILE_SIZE, 0) + 1, stride)) or [0]

        valid_tiles = []
        for y0 in ys_all:
            for x0 in xs_all:
                if valid_mask_thumb is not None:
                    # Project tile box to thumbnail space
                    tx0 = int(x0 * scale)
                    ty0 = int(y0 * scale)
                    tx1 = int(min(x0 + TILE_SIZE, W) * scale)
                    ty1 = int(min(y0 + TILE_SIZE, H) * scale)

                    # If this tile region in the thumbnail is 100% background, skip it
                    tile_valid = valid_mask_thumb[ty0:ty1, tx0:tx1]
                    # We require at least 1% of the tile to roughly contain some valid pixels
                    if tile_valid.size > 0 and tile_valid.mean() < 0.01:
                        continue

                valid_tiles.append((y0, x0))

        return valid_tiles, H, W, tif_crs, tif_tf

    # ── Dataset scanner ────────────────────────────────────────────────────────

    def _scan_dataset(self) -> List[Dict]:
        samples = []
        candidate_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        if not candidate_dirs:
            raise ValueError(f"No subdirectories found in {self.root_dir}")

        for map_dir in candidate_dirs:
            # ── Find orthophoto ──────────────────────────────────────────────
            ortho = None
            for ext in [".tif", ".tiff", ".TIF", ".TIFF", ".ecw", ".ECW"]:
                hits = list(map_dir.glob(f"*{ext}"))
                if hits:
                    ortho = hits[0]
                    break

            if ortho is None:
                logger.warning(f"No orthophoto in {map_dir}, skipping")
                continue

            # ── Find shapefiles ──────────────────────────────────────────────
            annotations: Dict[str, Path] = {}
            taken: set = set()
            for patterns, keywords, task_key in SHAPEFILE_TASKS:
                shp = self._find_shapefile(map_dir, patterns, keywords, taken)
                if shp:
                    annotations[task_key] = shp
                    taken.add(shp)
                else:
                    if task_key in OPTIONAL_TASKS:
                        logger.debug(
                            f"[{map_dir.name}] Optional '{task_key}' not found — all-zero mask"
                        )
                    else:
                        logger.warning(
                            f"[{map_dir.name}] No shapefile for '{task_key}'"
                        )

            if not annotations:
                logger.warning(f"No shapefiles in {map_dir}, skipping")
                continue

            logger.info(
                f"[{map_dir.name}] {ortho.name} | tasks: {', '.join(annotations.keys())}"
            )

            # ── Compute tiles ────────────────────────────────────────────────
            result = self._compute_tiles(ortho)
            if result is None:
                continue
            tiles, H, W, tif_crs, tif_tf = result

            n_before = len(samples)
            for y0, x0 in tiles:
                y1 = min(y0 + TILE_SIZE, H)
                x1 = min(x0 + TILE_SIZE, W)
                win = Window(x0, y0, x1 - x0, y1 - y0)
                samples.append(
                    {
                        "map_name": map_dir.name,
                        "tif_path": ortho,
                        "annotations": annotations,
                        "window": win,
                        "tif_crs": tif_crs,
                        "tif_tf": tif_tf,
                        "H": H,
                        "W": W,
                    }
                )
            logger.info(
                f"  → {len(samples) - n_before} tiles from {map_dir.name} ({H}×{W}px)"
            )

        if not samples:
            raise ValueError(
                f"No tiles found in {self.root_dir}. "
                "Each MAP folder needs a .tif + at least one .shp."
            )

        return samples

    # ── GeoDataFrame cache ─────────────────────────────────────────────────────

    def _get_cached_gdf(
        self,
        map_name: str,
        task_key: str,
        path: Path,
        target_crs,
    ) -> Optional[gpd.GeoDataFrame]:
        cache_key = (map_name, task_key)
        if cache_key in self._gdf_cache:
            return self._gdf_cache[cache_key]

        try:
            gdf = gpd.read_file(path)
            # FORCE strict projection alignment
            if target_crs is not None:
                if gdf.crs is None:
                    gdf.set_crs(target_crs, inplace=True)
                elif gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)

            # Drop null / invalid geometries up front (saves repeated checks)
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(
                drop=True
            )
            self._gdf_cache[cache_key] = gdf
            return gdf
        except Exception as e:
            logger.error(f"Error loading GDF {path}: {e}")
            return None

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        win = sample["window"]

        # ── Load tile pixels ─────────────────────────────────────────────────
        try:
            with rasterio.open(sample["tif_path"]) as src:
                tile_data = src.read(
                    window=win,
                    out_shape=(src.count, TILE_SIZE, TILE_SIZE),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                tile_tf = src.window_transform(win)
                tif_crs = src.crs
        except Exception as e:
            logger.error(f"Error reading tile {sample['tif_path']}: {e}")
            tile_data = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            tile_tf = sample["tif_tf"]
            tif_crs = sample["tif_crs"]

        # Build (H,W,C) float32 in [0,1]
        if tile_data.shape[0] >= 3:
            image = np.stack([tile_data[0], tile_data[1], tile_data[2]], axis=-1)
        else:
            image = np.stack([tile_data[0]] * 3, axis=-1)

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            image = image.astype(np.float32)

        # Numerical stability: replace NaN/Inf from NoData regions, clamp to [0,1]
        image = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # ── Build masks ───────────────────────────────────────────────────────
        output_shape = (TILE_SIZE, TILE_SIZE)
        masks: Dict[str, np.ndarray] = {}

        for _, _, task_key in SHAPEFILE_TASKS:
            mask_key = f"{task_key}_mask"
            if task_key in sample["annotations"]:
                gdf = self._get_cached_gdf(
                    sample["map_name"],
                    task_key,
                    sample["annotations"][task_key],
                    tif_crs,
                )
                if gdf is not None and len(gdf) > 0:
                    masks[mask_key] = self.anno_parser.rasterize_annotations(
                        gdf, tile_tf, output_shape, task_key
                    )
                else:
                    masks[mask_key] = np.zeros(output_shape, dtype=np.uint8)
            else:
                masks[mask_key] = np.zeros(output_shape, dtype=np.uint8)

        # ── Roof type mask ────────────────────────────────────────────────────
        if "building" in sample["annotations"]:
            bgdf = self._get_cached_gdf(
                sample["map_name"],
                "building",
                sample["annotations"]["building"],
                tif_crs,
            )
            if bgdf is not None and len(bgdf) > 0:
                masks["roof_type_mask"] = self.anno_parser.extract_roof_types(
                    bgdf, tile_tf, output_shape
                )
            else:
                masks["roof_type_mask"] = np.zeros(output_shape, dtype=np.uint8)
        else:
            masks["roof_type_mask"] = np.zeros(output_shape, dtype=np.uint8)

        # ── Augmentation / tensor conversion ──────────────────────────────────
        if self.transform:
            transformed = self.transform(image=image, **masks)
            result: Dict[str, torch.Tensor] = {"image": transformed["image"]}
            for k in masks:
                result[k] = transformed[k].long()
        else:
            result = {"image": torch.from_numpy(image).permute(2, 0, 1).float()}
            for k, v in masks.items():
                result[k] = torch.from_numpy(v).long()

        result["metadata"] = {
            "map_name": sample["map_name"],
            "idx": idx,
            "tile_x": win.col_off,
            "tile_y": win.row_off,
        }
        return result


# ── DataLoader factory ─────────────────────────────────────────────────────────


def create_dataloaders(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: int = TILE_SIZE,
    val_split: float = 0.2,
) -> Tuple:
    """
    Build train and validation DataLoaders.

    If val_dir is provided, creates a separate validation dataset.
    Otherwise, splits training data by val_split fraction.

    num_workers defaults to 0 — required for Jupyter / shared-server
    environments where forked workers cannot be tested.
    """
    train_ds = SvamitvaDataset(
        train_dir, image_size, get_train_transforms(image_size), "train"
    )

    if val_dir is not None:
        val_ds_full = SvamitvaDataset(
            val_dir, image_size, get_val_transforms(image_size), "val"
        )
        tr_ds = train_ds
        val_ds = val_ds_full
        logger.info(f"Separate val dir: {len(tr_ds)} train / {len(val_ds)} val tiles")
    else:
        total = len(train_ds)
        val_size = max(1, int(total * val_split))
        from torch.utils.data import Subset

        tr_ds = Subset(train_ds, list(range(total - val_size)))
        val_ds = Subset(train_ds, list(range(total - val_size, total)))
        logger.info(f"Auto-split: {len(tr_ds)} train / {len(val_ds)} val tiles")

    train_loader = torch.utils.data.DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=len(tr_ds) > batch_size,
        persistent_workers=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    logger.info(
        f"DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches"
    )
    return train_loader, val_loader
