"""
Preprocessing utilities for SVAMITVA drone imagery and annotations.

Handles orthophoto loading, shapefile parsing and rasterization.
Line and point geometries are automatically buffered so they produce
visible pixel masks.
"""

import numpy as np
import rasterio
from rasterio.warp import Resampling
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrthophotoPreprocessor:
    """Loads and normalises orthophoto TIFF/ECW imagery."""

    def __init__(self, target_crs: str = "EPSG:32643"):
        self.target_crs = target_crs
        logger.info(f"OrthophotoPreprocessor: target CRS = {target_crs}")

    def load_orthophoto(
        self,
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Load and normalise an orthophoto.

        Args:
            image_path: Path to .tif / .ecw file
            target_size: (height, width) to resample to; None = full resolution

        Returns:
            (image np.float32 HxWxC in [0,1], metadata dict)
        """
        logger.info(f"Loading orthophoto: {image_path}")
        try:
            with rasterio.open(image_path) as src:
                if target_size:
                    out_shape = (src.count, target_size[0], target_size[1])
                    image = src.read(
                        out_shape=out_shape,
                        resampling=Resampling.bilinear,
                    )
                    transform = src.transform * src.transform.scale(
                        src.width / target_size[1],
                        src.height / target_size[0],
                    )
                    height, width = target_size
                else:
                    image = src.read()
                    transform = src.transform
                    height, width = src.height, src.width

                image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)

                # Keep only first 3 bands (drop alpha if present)
                if image.shape[2] > 3:
                    image = image[:, :, :3]

                # Normalise to [0, 1]
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                elif image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0
                else:
                    image = image.astype(np.float32)

                metadata = {
                    "transform": transform,
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "height": height,
                    "width": width,
                    "original_height": src.height,
                    "original_width": src.width,
                }
                logger.info(f"Loaded image shape={image.shape}, CRS={src.crs}")
            return image, metadata

        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            size = target_size if target_size else (512, 512)
            return np.zeros((*size, 3), dtype=np.float32), {}

    def standardize_image(
        self,
        image: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """ImageNet-style standardisation (mean/std per channel)."""
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406])
        if std is None:
            std = np.array([0.229, 0.224, 0.225])
        return ((image - mean) / std).astype(np.float32)


class ShapefileAnnotationParser:
    """
    Converts shapefile vector annotations to raster masks.

    Geometry type handling:
        Polygon  → rasterised directly (buildings, roads, waterbodies, …)
        LineString → buffered by LINE_BUFFER_PX pixels then rasterised
        Point    → buffered by POINT_BUFFER_PX pixels then rasterised
    """

    # ── Buffer sizes (in map units, computed from pixel size in rasterize) ────
    # These are overridden per-call when we know the pixel size.
    LINE_BUFFER_M: float = 2.0  # ~2.0 m buffer for lines (wider = stronger training signal)
    POINT_BUFFER_M: float = 3.0  # ~3.0 m buffer for points

    # Geometry types that need buffering
    LINE_TASKS = {
        "road_centerline",
        "utility_line",
        "waterbody_line",
        "railway",
    }
    POINT_TASKS = {"waterbody_point"}

    # Roof type lookup — covers common SVAMITVA naming variants
    ROOF_TYPES = {
        "RCC": 1, "rcc": 1, "Rcc": 1, "R.C.C": 1, "R.C.C.": 1,
        "Tiled": 2, "tiled": 2, "TILED": 2, "Tile": 2, "tile": 2,
        "Tin": 3, "tin": 3, "TIN": 3, "GI Sheet": 3, "GI": 3,
        "gi sheet": 3, "Sheet": 3, "AC Sheet": 3, "Asbestos": 3,
        "Others": 4, "others": 4, "OTHERS": 4, "Other": 4, "other": 4,
        "Mixed": 4, "Kutcha": 4, "Thatch": 4, "Thatched": 4,
        "Unknown": 0, "unknown": 0, "": 0, "NA": 0, "None": 0,
    }

    def __init__(self):
        logger.info("ShapefileAnnotationParser initialised")
        self.roof_types = self.ROOF_TYPES

    # ── Shapefile loading ──────────────────────────────────────────────────────

    def load_shapefile(self, shapefile_path: Path) -> gpd.GeoDataFrame:
        logger.info(f"Loading: {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"  → {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"Error loading {shapefile_path}: {e}")
            raise

    # ── Rasterisation ──────────────────────────────────────────────────────────

    def rasterize_annotations(
        self,
        gdf: gpd.GeoDataFrame,
        reference_transform: rasterio.Affine,
        output_shape: Tuple[int, int],
        feature_type: str,
        target_crs=None,
    ) -> np.ndarray:
        """
        Convert a GeoDataFrame to a binary raster mask.

        The GDF is reprojected to target_crs when provided.
        Lines and points are automatically buffered.

        Args:
            gdf: GeoDataFrame with annotation features
            reference_transform: Affine geotransform from the orthophoto tile
            output_shape: (height, width) of output mask
            feature_type: Task name (e.g. 'road_centerline', 'building')
            target_crs: CRS the raster is in; GDF is reprojected if different

        Returns:
            Binary numpy mask (H, W) uint8
        """
        from rasterio.features import rasterize

        if gdf is None or len(gdf) == 0:
            return np.zeros(output_shape, dtype=np.uint8)

        # ★ Reproject shapefile to match the raster CRS
        if target_crs is not None and gdf.crs is not None:
            if str(gdf.crs).upper() != str(target_crs).upper():
                logger.info(f"  Reprojecting {feature_type}: {gdf.crs} → {target_crs}")
                gdf = gdf.to_crs(target_crs)

        # Estimate pixel size from affine transform (metres per pixel)
        pixel_size = abs(reference_transform.a) if reference_transform.a != 0 else 1.0

        # Buffer lines and points so they produce visible pixel masks
        if feature_type in self.LINE_TASKS:
            buffer_m = max(pixel_size * 4, self.LINE_BUFFER_M)  # wider receptive field
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(buffer_m)
        elif feature_type in self.POINT_TASKS:
            buffer_m = max(pixel_size * 5, self.POINT_BUFFER_M)  # more visible points
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(buffer_m)

        # Drop null / invalid geometries
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

        if len(gdf) == 0:
            return np.zeros(output_shape, dtype=np.uint8)

        shapes = [(geom, 1) for geom in gdf.geometry]

        try:
            mask = rasterize(
                shapes=shapes,
                out_shape=output_shape,
                transform=reference_transform,
                fill=0,
                dtype=np.uint8,
            )
        except Exception as e:
            logger.warning(f"  rasterize failed for {feature_type}: {e}")
            return np.zeros(output_shape, dtype=np.uint8)

        n_pos = int(mask.sum())
        if n_pos == 0:
            # Many tiles legitimately contain no features — keep at DEBUG
            logger.debug(f"  [{feature_type}] tile has no features (all-zero mask)")
        else:
            logger.info(f"  [{feature_type}] mask positive pixels: {n_pos}")
        return mask

    # ── Roof type mask ─────────────────────────────────────────────────────────

    def extract_roof_types(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        reference_transform: rasterio.Affine,
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Multi-class mask for roof types.

        Values: 0=unknown/background, 1=RCC, 2=Tiled, 3=Tin, 4=Others
        """
        from rasterio.features import rasterize

        if buildings_gdf is None or len(buildings_gdf) == 0:
            return np.zeros(output_shape, dtype=np.uint8)

        # Find the roof type column
        roof_col = None
        for col in [
            "roof_type",
            "ROOF_TYPE",
            "RoofType",
            "Roof_Type",
            "roof",
            "ROOF",
            "Sl_Typ",
            "SL_TYP",
            "sl_typ",
            "Roof_typ",
            "ROOF_TYP",
            "Rooftype",
            "Structure",
            "STRUCTURE",
            "type",
            "Type",
            "TYPE",
        ]:
            if col in buildings_gdf.columns:
                roof_col = col
                break

        if roof_col is None:
            logger.debug("No roof type column found; all buildings → 0")
            shapes = [(geom, 0) for geom in buildings_gdf.geometry]
        else:
            shapes = []
            for _, row in buildings_gdf.iterrows():
                geom = row.geometry
                if geom is None or not geom.is_valid:
                    continue
                class_id = self.roof_types.get(str(row[roof_col]).strip(), 0)
                shapes.append((geom, class_id))

        if not shapes:
            return np.zeros(output_shape, dtype=np.uint8)

        return rasterize(
            shapes=shapes,
            out_shape=output_shape,
            transform=reference_transform,
            fill=0,
            dtype=np.uint8,
        )

    # ── Validation helper ──────────────────────────────────────────────────────

    def validate_annotations(self, shapefile_path: Path) -> bool:
        try:
            gdf = self.load_shapefile(shapefile_path)
            if len(gdf) == 0:
                logger.warning(f"Empty shapefile: {shapefile_path}")
                return False
            invalid = (~gdf.geometry.is_valid).sum()
            if invalid:
                logger.warning(f"{invalid} invalid geometries in {shapefile_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation failed for {shapefile_path}: {e}")
            return False


# ── Standalone utilities ───────────────────────────────────────────────────────


def check_crs_match(orthophoto_path: Path, shapefile_path: Path) -> bool:
    with rasterio.open(orthophoto_path) as src:
        r_crs = src.crs
    gdf = gpd.read_file(shapefile_path)
    match = r_crs == gdf.crs
    if not match:
        logger.warning(f"CRS mismatch — raster: {r_crs}, vector: {gdf.crs}")
    return match


def reproject_shapefile(
    shapefile_path: Path,
    target_crs: str,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shapefile_path).to_crs(target_crs)
    if output_path:
        gdf.to_file(output_path)
    return gdf
