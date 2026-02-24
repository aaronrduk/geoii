"""
PyTorch Dataset for SVAMITVA orthophotos and annotations.

Key design decisions:
- Each large MAP TIF is tiled into overlapping 512×512 patches
  so DataLoader gets many samples per MAP instead of one.
- Shapefiles are matched first by explicit glob patterns,
  then by keyword search across all .shp files in the folder
  (handles non-standard naming like 'Abadi.shp', 'built_up.shp', etc.)
- The GeoDataFrame is always reprojected to the TIF's CRS before
  rasterization so masks are never empty due to CRS mismatch.
- Bridge and railway shapefiles are optional — if absent, the mask
  remains all-zero (correct: no bridge/railway features to learn).

Dataset structure expected:
    DATA/
      MAP1/
        anything.tif      ← orthophoto
        Road.shp          ← road features
        Built_Up_Area.shp ← buildings (any name matching keyword 'built')
        ...
      MAP2/ ... MAP5/
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
import rasterio
from rasterio.windows import Window
import geopandas as gpd

from .preprocessing import OrthophotoPreprocessor, ShapefileAnnotationParser
from .augmentation import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shapefile task definitions ─────────────────────────────────────────────────
# Each entry: (explicit_glob_patterns, keyword_fallback_list, mask_key)
# Keywords are matched case-insensitively against all .shp filenames in folder.
# Bridge and Railway patterns are included but gracefully tolerated when absent.
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
    # ★ Bridge and Railway — optional, all-zero mask when absent
    (
        ["Bridge*.shp", "Bridge.shp"],
        ["bridge"],
        "bridge",
    ),
    (
        ["Railway*.shp", "Rail*.shp", "Railway.shp"],
        ["railway", "railroad", "rail"],
        "railway",
    ),
]

ALL_MASK_KEYS = [task for _, _, task in SHAPEFILE_TASKS] + ["roof_type"]

# Tasks where missing shapefiles are expected and should only produce a debug log
OPTIONAL_TASKS = {"bridge", "railway"}

# Tiling config
TILE_SIZE = 512
TILE_OVERLAP = 64  # overlap between adjacent tiles


class SvamitvaDataset(Dataset):
    """
    PyTorch Dataset for SVAMITVA drone imagery.

    Returns 512×512 tile-level samples (not MAP-level).
    Each large orthophoto is split into overlapping tiles during scan.
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

        # ★ Bug 4 fix: initialise cache in __init__ so it is always present
        self._gdf_cache: Dict[Tuple, gpd.GeoDataFrame] = {}

        self.samples = self._scan_dataset()
        logger.info(f"Initialized {mode} dataset with {len(self.samples)} tile samples")

    # ── Shapefile finder ───────────────────────────────────────────────────────

    def _find_shapefile(
        self, folder: Path, patterns: List[str], keywords: List[str], taken: set
    ) -> Optional[Path]:
        """
        Find a shapefile matching explicit glob patterns first,
        then fall back to keyword search across all .shp in folder.
        Skips files already claimed by another task (stored in `taken`).
        """
        # 1. Exact glob patterns
        for pat in patterns:
            for hit in sorted(folder.glob(pat)):
                if hit not in taken:
                    return hit

        # 2. Keyword fallback — case-insensitive substring / anchored match
        all_shps = sorted(folder.glob("*.shp"))
        for shp in all_shps:
            stem_lower = shp.stem.lower()
            for kw in keywords:
                if kw.startswith("^") and kw.endswith("$"):
                    if stem_lower == kw[1:-1] and shp not in taken:
                        return shp
                elif kw in stem_lower and shp not in taken:
                    return shp

        return None

    # ── Dataset scanner ────────────────────────────────────────────────────────

    def _compute_tiles(self, tif_path: Path):
        """
        Open the TIF, compute tile windows, and return scanning info.
        Returns (ys, xs, H, W, tif_crs, tif_tf) or None on error.
        """
        try:
            with rasterio.open(tif_path) as src:
                H, W = src.height, src.width
                tif_crs = src.crs
                tif_tf = src.transform
        except Exception as e:
            logger.error(f"Cannot open {tif_path}: {e}")
            return None

        stride = TILE_SIZE - TILE_OVERLAP
        ys = list(range(0, max(H - TILE_SIZE, 0) + 1, stride))
        xs = list(range(0, max(W - TILE_SIZE, 0) + 1, stride))
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]

        return ys, xs, H, W, tif_crs, tif_tf

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
                            f"  [{map_dir.name}] Optional '{task_key}' shapefile"
                            " not found — mask will be all-zero"
                        )
                    else:
                        logger.warning(
                            f"  [{map_dir.name}] No shapefile for '{task_key}'"
                        )

            if not annotations:
                logger.warning(f"No shapefiles in {map_dir}, skipping")
                continue

            found = ", ".join(annotations.keys())
            logger.info(f"Found {map_dir.name}: {ortho.name} | tasks: {found}")

            # ── Compute tiles ────────────────────────────────────────────────
            result = self._compute_tiles(ortho)
            if result is None:
                continue
            ys, xs, H, W, tif_crs, tif_tf = result

            n_before = len(samples)
            for y0 in ys:
                for x0 in xs:
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
            n_tiles = len(samples) - n_before
            logger.info(f"  → {n_tiles} tiles from {map_dir.name} ({H}×{W} px)")

        if not samples:
            raise ValueError(
                f"No tiles found in {self.root_dir}. "
                "Each MAP folder must contain a .tif + at least one .shp."
            )

        return samples

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cached_gdf(
        self, map_name: str, task_key: str, path: Path, target_crs
    ) -> Optional[gpd.GeoDataFrame]:
        cache_key = (map_name, task_key)
        if cache_key in self._gdf_cache:
            return self._gdf_cache[cache_key]

        try:
            gdf = gpd.read_file(path)
            if gdf.crs is not None and target_crs is not None and gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            # Drop rows with null / invalid geometries early
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(
                drop=True
            )

            self._gdf_cache[cache_key] = gdf
            return gdf
        except Exception as e:
            logger.error(f"Error loading GDF {path}: {e}")
            return None

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
            logger.error(f"Error reading tile from {sample['tif_path']}: {e}")
            tile_data = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            tile_tf = sample["tif_tf"]
            tif_crs = sample["tif_crs"]

        # Build image array (H,W,C) float32 in [0,1]
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

        # ── Numerical stability shield ───────────────────────────────────────
        # 1. Replace NaN/Inf from NoData regions (common in drone TIFs)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        # 2. Hard-clamp to [0, 1]
        image = np.clip(image, 0.0, 1.0)

        # ── Build masks for this tile ─────────────────────────────────────────
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
                # Optional task (bridge/railway) with no shapefile → all-zero
                masks[mask_key] = np.zeros(output_shape, dtype=np.uint8)

        # ── Roof type mask ───────────────────────────────────────────────────
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

        # ── Augmentation / tensor conversion ─────────────────────────────────
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

    If val_dir is provided, creates a separate validation dataset from that
    folder. Otherwise, splits training data by val_split fraction.

    num_workers defaults to 0 — required for Jupyter / shared-server
    environments where forked workers cannot be tested.
    """
    train_ds = SvamitvaDataset(
        train_dir, image_size, get_train_transforms(image_size), "train"
    )

    # ── Validation dataset ──────────────────────────────────────────────────
    if val_dir is not None:
        # ★ Bug 6 fix: use separate val folder when provided
        val_ds_full = SvamitvaDataset(
            val_dir, image_size, get_val_transforms(image_size), "val"
        )
        tr_ds = train_ds
        val_ds = val_ds_full
        logger.info(
            f"Using separate val dir: {len(tr_ds)} train / {len(val_ds)} val tiles"
        )
    else:
        total = len(train_ds)
        val_size = max(1, int(total * val_split))
        val_idx = list(range(total - val_size, total))
        tr_idx = list(range(total - val_size))

        from torch.utils.data import Subset

        tr_ds = Subset(train_ds, tr_idx)
        val_ds = Subset(train_ds, val_idx)
        logger.info(f"Auto-split: {len(tr_ds)} train / {len(val_ds)} val tiles")

    train_loader = torch.utils.data.DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),  # pin_memory only useful with workers
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
        f"DataLoaders ready — {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches"
    )
    return train_loader, val_loader
