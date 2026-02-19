"""
Preprocessing utilities for SVAMITVA drone imagery and annotations.

This module handles:
- Orthophoto normalization and standardization
- Shapefile parsing and validation
- Coordinate system transformations
- Data quality checks
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Setup logging for student developers to track preprocessing steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrthophotoPreprocessor:
    """
    Preprocessor for orthophoto images.

    Handles normalization, resizing, and validation of drone imagery.
    Designed to be student-friendly with clear documentation.
    """

    def __init__(self, target_crs: str = "EPSG:32643"):
        """
        Initialize the preprocessor.

        Args:
            target_crs (str): Target coordinate reference system for reprojection.
                             Default is UTM Zone 43N (EPSG:32643) which covers most of India.
        """
        self.target_crs = target_crs
        logger.info(f"OrthophotoPreprocessor initialized with target CRS: {target_crs}")

    def load_orthophoto(
        self, image_path: Path, target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Load an orthophoto from a TIFF file.

        Args:
            image_path (Path): Path to the orthophoto file (.tif)
            target_size (Tuple[int, int], optional): Target size (H, W) to resize to.
                                                    If None, loads full resolution.

        Returns:
            Tuple containing:
                - numpy.ndarray: Image data (H, W, C) normalized to [0, 1]
                - dict: Metadata including transform, CRS, bounds
        """
        logger.info(f"Loading orthophoto from: {image_path}")

        try:
            with rasterio.open(image_path) as src:
                # Read specific size if requested (efficient reading)
                if target_size:
                    # rasterio uses (bands, height, width)
                    out_shape = (src.count, target_size[0], target_size[1])
                    image = src.read(
                        out_shape=out_shape, resampling=Resampling.bilinear
                    )

                    # Update transform for the new size
                    transform = src.transform * src.transform.scale(
                        (src.width / target_size[1]), (src.height / target_size[0])
                    )
                    height, width = target_size
                else:
                    image = src.read()  # Shape: (C, H, W)
                    transform = src.transform
                    height, width = src.height, src.width

                # Transpose to (H, W, C) for easier processing
                image = np.transpose(image, (1, 2, 0))

                # Handle 4-channel images (RGBA) -> RGB
                if image.shape[2] > 3:
                    logger.info(f"Image has {image.shape[2]} channels, keeping first 3")
                    image = image[:, :, :3]

                # Normalize to [0, 1] range
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                elif image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0

                # Extract metadata
                metadata = {
                    "transform": transform,
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "height": height,
                    "width": width,
                    "count": src.count,
                    "original_height": src.height,
                    "original_width": src.width,
                }

                logger.info(f"Loaded image with shape: {image.shape}, CRS: {src.crs}")

            return image, metadata

        except Exception as e:
            logger.error(f"Error loading orthophoto {image_path}: {e}")
            # Return a dummy black image to avoid crashing the entire training
            if target_size:
                dummy_shape = (target_size[0], target_size[1], 3)
            else:
                dummy_shape = (512, 512, 3)
            logger.warning("Returning dummy black image")
            return np.zeros(dummy_shape, dtype=np.float32), {}

    def standardize_image(
        self,
        image: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Standardize image using mean and standard deviation.

        Args:
            image (np.ndarray): Input image (H, W, C)
            mean (np.ndarray, optional): Channel-wise mean.
                                        If None, ImageNet mean is used.
            std (np.ndarray, optional): Channel-wise std.
                                       If None, ImageNet std is used.

        Returns:
            np.ndarray: Standardized image
        """
        # Use ImageNet statistics as default (works well for transfer learning)
        # Student note: These are standard values used in computer vision
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406])  # ImageNet RGB mean
        if std is None:
            std = np.array([0.229, 0.224, 0.225])  # ImageNet RGB std

        # Apply standardization
        # Formula: (x - mean) / std
        standardized = (image - mean) / std

        return standardized.astype(np.float32)


class ShapefileAnnotationParser:
    """
    Parser for shapefile annotations.

    Converts vector annotations (buildings, roads, etc.) into raster masks
    for training the segmentation model.
    """

    # Class mapping for different feature types
    # Student note: These correspond to the categories we want to detect
    FEATURE_CLASSES = {"building": 1, "road": 2, "waterbody": 3, "utility": 4}

    # Roof type mapping for building classification
    ROOF_TYPES = {
        "RCC": 1,
        "Tiled": 2,
        "Tin": 3,
        "Others": 4,
        "Unknown": 0,  # For buildings without roof type annotation
    }

    def __init__(self):
        """Initialize the annotation parser."""
        logger.info("ShapefileAnnotationParser initialized")
        self.feature_classes = self.FEATURE_CLASSES
        self.roof_types = self.ROOF_TYPES

    def load_shapefile(self, shapefile_path: Path) -> gpd.GeoDataFrame:
        """
        Load a shapefile into a GeoDataFrame.

        Args:
            shapefile_path (Path): Path to the shapefile

        Returns:
            gpd.GeoDataFrame: Loaded geospatial data
        """
        logger.info(f"Loading shapefile: {shapefile_path}")

        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded {len(gdf)} features from shapefile")
            return gdf
        except Exception as e:
            logger.error(f"Error loading shapefile {shapefile_path}: {e}")
            raise

    def rasterize_annotations(
        self,
        gdf: gpd.GeoDataFrame,
        reference_transform: rasterio.Affine,
        output_shape: Tuple[int, int],
        feature_type: str,
    ) -> np.ndarray:
        """
        Convert vector annotations to raster mask.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame with annotations
            reference_transform (rasterio.Affine): Geotransform from reference orthophoto
            output_shape (Tuple[int, int]): (height, width) of output mask
            feature_type (str): Type of feature ('building', 'road', etc.)

        Returns:
            np.ndarray: Binary mask (H, W) with 1 for feature, 0 for background
        """
        from rasterio.features import rasterize

        logger.info(f"Rasterizing {feature_type} annotations...")

        # Create list of (geometry, value) tuples for rasterization
        # All features get value 1 (foreground), background is 0
        shapes = [(geom, 1) for geom in gdf.geometry]

        # Rasterize the geometries
        # Student note: This converts vector polygons/lines to pixel masks
        mask = rasterize(
            shapes=shapes,
            out_shape=output_shape,
            transform=reference_transform,
            fill=0,  # Background value
            dtype=np.uint8,
        )

        logger.info(f"Created mask with {np.sum(mask)} positive pixels")

        return mask

    def extract_roof_types(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        reference_transform: rasterio.Affine,
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract roof type classification mask for buildings.

        Args:
            buildings_gdf (gpd.GeoDataFrame): Building annotations
            reference_transform (rasterio.Affine): Geotransform
            output_shape (Tuple[int, int]): Output mask shape

        Returns:
            np.ndarray: Roof type mask where values correspond to ROOF_TYPES
        """
        from rasterio.features import rasterize

        logger.info("Extracting roof type annotations...")

        # Check if roof type column exists
        # Student note: Different datasets may use different column names
        roof_column = None
        for col in ["roof_type", "ROOF_TYPE", "RoofType", "type"]:
            if col in buildings_gdf.columns:
                roof_column = col
                break

        if roof_column is None:
            logger.warning(
                "No roof type column found, using 'Unknown' for all buildings"
            )
            # Create mask with all buildings marked as Unknown (0)
            shapes = [(geom, 0) for geom in buildings_gdf.geometry]
        else:
            # Map roof types to class IDs
            shapes = []
            for idx, row in buildings_gdf.iterrows():
                geom = row.geometry
                roof_type = row[roof_column]

                # Map to class ID
                class_id = self.roof_types.get(roof_type, 0)  # Default to Unknown
                shapes.append((geom, class_id))

        # Rasterize
        roof_mask = rasterize(
            shapes=shapes,
            out_shape=output_shape,
            transform=reference_transform,
            fill=0,  # Background (no building)
            dtype=np.uint8,
        )

        return roof_mask

    def validate_annotations(self, shapefile_path: Path) -> bool:
        """
        Validate that a shapefile is properly formatted.

        Args:
            shapefile_path (Path): Path to shapefile

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            gdf = self.load_shapefile(shapefile_path)

            # Check if GeoDataFrame is not empty
            if len(gdf) == 0:
                logger.warning(f"Shapefile is empty: {shapefile_path}")
                return False

            # Check if geometries are valid
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(
                    f"Found {invalid_geoms.sum()} invalid geometries in {shapefile_path}"
                )
                return False

            logger.info(f"Shapefile validation passed: {shapefile_path}")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {shapefile_path}: {e}")
            return False


def check_crs_match(orthophoto_path: Path, shapefile_path: Path) -> bool:
    """
    Check if orthophoto and shapefile have matching CRS.

    Args:
        orthophoto_path (Path): Path to orthophoto
        shapefile_path (Path): Path to shapefile

    Returns:
        bool: True if CRS match, False otherwise
    """
    with rasterio.open(orthophoto_path) as src:
        raster_crs = src.crs

    gdf = gpd.read_file(shapefile_path)
    vector_crs = gdf.crs

    match = raster_crs == vector_crs

    if not match:
        logger.warning(f"CRS mismatch - Raster: {raster_crs}, Vector: {vector_crs}")
    else:
        logger.info(f"CRS match confirmed: {raster_crs}")

    return match


def reproject_shapefile(
    shapefile_path: Path, target_crs: str, output_path: Optional[Path] = None
) -> gpd.GeoDataFrame:
    """
    Reproject shapefile to target CRS.

    Args:
        shapefile_path (Path): Input shapefile
        target_crs (str): Target CRS (e.g., 'EPSG:32643')
        output_path (Path, optional): If provided, save reprojected shapefile

    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame
    """
    logger.info(f"Reprojecting {shapefile_path} to {target_crs}")

    gdf = gpd.read_file(shapefile_path)
    gdf_reprojected = gdf.to_crs(target_crs)

    if output_path:
        gdf_reprojected.to_file(output_path)
        logger.info(f"Saved reprojected shapefile to {output_path}")

    return gdf_reprojected
