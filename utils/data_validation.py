"""
Data validation utilities for SVAMITVA dataset.

Validates dataset integrity, file formats, and data quality before training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import geopandas as gpd
import rasterio
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validator for SVAMITVA dataset.

    Checks:
    - File existence and accessibility
    - Correct file formats
    - CRS consistency
    - Data quality metrics
    - Annotation completeness
    """

    def __init__(self):
        """Initialize data validator."""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_dataset(self, root_dir: Path) -> Tuple[bool, Dict]:
        """
        Validate entire dataset.

        Args:
            root_dir: Root directory of dataset

        Returns:
            tuple: (is_valid, validation_report)
        """
        logger.info(f"Validating dataset at {root_dir}")

        self.validation_errors = []
        self.validation_warnings = []

        if not root_dir.exists():
            self.validation_errors.append(f"Dataset directory does not exist: {root_dir}")
            return False, self._generate_report()

        village_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

        if len(village_dirs) == 0:
            self.validation_errors.append(f"No village directories found in {root_dir}")
            return False, self._generate_report()

        villages_validated = 0
        for village_dir in village_dirs:
            if self._validate_village(village_dir):
                villages_validated += 1

        is_valid = len(self.validation_errors) == 0 and villages_validated > 0

        report = self._generate_report()
        report["total_villages"] = len(village_dirs)
        report["validated_villages"] = villages_validated

        return is_valid, report

    def _validate_village(self, village_dir: Path) -> bool:
        """
        Validate a single village directory.

        Args:
            village_dir: Path to village directory

        Returns:
            bool: True if valid
        """
        village_name = village_dir.name

        orthophoto_path = self._find_orthophoto(village_dir)
        if orthophoto_path is None:
            self.validation_errors.append(f"{village_name}: No orthophoto found")
            return False

        if not self._validate_orthophoto(orthophoto_path, village_name):
            return False

        anno_dir = village_dir / "annotations"
        if not anno_dir.exists():
            self.validation_warnings.append(f"{village_name}: No annotations directory")
            return False

        crs = self._get_orthophoto_crs(orthophoto_path)

        annotation_files = self._find_annotations(anno_dir)
        if len(annotation_files) == 0:
            self.validation_warnings.append(f"{village_name}: No annotation files found")

        for feature_type, shapefile_path in annotation_files.items():
            self._validate_shapefile(shapefile_path, feature_type, village_name, crs)

        return True

    def _find_orthophoto(self, village_dir: Path) -> Optional[Path]:
        """Find orthophoto file in village directory."""
        for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
            potential_path = village_dir / f"orthophoto{ext}"
            if potential_path.exists():
                return potential_path

            for file in village_dir.glob(f"*{ext}"):
                if "ORTHO" in file.stem.upper():
                    return file

        return None

    def _validate_orthophoto(self, orthophoto_path: Path, village_name: str) -> bool:
        """
        Validate orthophoto file.

        Args:
            orthophoto_path: Path to orthophoto
            village_name: Name of village

        Returns:
            bool: True if valid
        """
        try:
            with rasterio.open(orthophoto_path) as src:
                if src.count < 3:
                    self.validation_warnings.append(
                        f"{village_name}: Orthophoto has less than 3 bands"
                    )

                if src.crs is None:
                    self.validation_errors.append(
                        f"{village_name}: Orthophoto has no CRS defined"
                    )
                    return False

                if src.width < 256 or src.height < 256:
                    self.validation_warnings.append(
                        f"{village_name}: Orthophoto is very small ({src.width}x{src.height})"
                    )

                data = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
                if np.all(data == data[0, 0]):
                    self.validation_warnings.append(
                        f"{village_name}: Orthophoto appears to be constant/empty"
                    )

            return True

        except Exception as e:
            self.validation_errors.append(
                f"{village_name}: Failed to read orthophoto: {str(e)}"
            )
            return False

    def _get_orthophoto_crs(self, orthophoto_path: Path):
        """Get CRS from orthophoto."""
        try:
            with rasterio.open(orthophoto_path) as src:
                return src.crs
        except:
            return None

    def _find_annotations(self, anno_dir: Path) -> Dict[str, Path]:
        """Find annotation shapefiles."""
        annotations = {}

        search_patterns = {
            "buildings": ["buildings.shp", "Buildings.shp", "Built_Up_Area*.shp"],
            "roads": ["roads.shp", "Roads.shp", "Road*.shp"],
            "waterbodies": ["waterbodies.shp", "Waterbodies.shp", "Water*.shp"],
            "utilities": ["utilities.shp", "Utilities.shp", "Utility*.shp"],
        }

        for feature_type, patterns in search_patterns.items():
            for pattern in patterns:
                matches = list(anno_dir.glob(pattern))
                if matches:
                    annotations[feature_type] = matches[0]
                    break

        return annotations

    def _validate_shapefile(
        self, shapefile_path: Path, feature_type: str, village_name: str, expected_crs
    ):
        """Validate shapefile."""
        try:
            gdf = gpd.read_file(shapefile_path)

            if len(gdf) == 0:
                self.validation_warnings.append(
                    f"{village_name}: {feature_type} shapefile is empty"
                )
                return

            if not gdf.geometry.is_valid.all():
                invalid_count = (~gdf.geometry.is_valid).sum()
                self.validation_warnings.append(
                    f"{village_name}: {feature_type} has {invalid_count} invalid geometries"
                )

            if gdf.crs is None:
                self.validation_warnings.append(
                    f"{village_name}: {feature_type} has no CRS defined"
                )
            elif expected_crs and gdf.crs != expected_crs:
                self.validation_warnings.append(
                    f"{village_name}: {feature_type} CRS mismatch (expected {expected_crs}, got {gdf.crs})"
                )

        except Exception as e:
            self.validation_errors.append(
                f"{village_name}: Failed to read {feature_type} shapefile: {str(e)}"
            )

    def _generate_report(self) -> Dict:
        """Generate validation report."""
        report = {
            "is_valid": len(self.validation_errors) == 0,
            "num_errors": len(self.validation_errors),
            "num_warnings": len(self.validation_warnings),
            "errors": self.validation_errors.copy(),
            "warnings": self.validation_warnings.copy(),
        }

        return report

    def print_report(self, report: Dict):
        """Print validation report."""
        logger.info("\n" + "=" * 60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 60)

        if report["is_valid"]:
            logger.info("✓ Dataset validation PASSED")
        else:
            logger.error("✗ Dataset validation FAILED")

        logger.info(f"Total villages: {report.get('total_villages', 0)}")
        logger.info(f"Validated villages: {report.get('validated_villages', 0)}")
        logger.info(f"Errors: {report['num_errors']}")
        logger.info(f"Warnings: {report['num_warnings']}")

        if report["errors"]:
            logger.error("\nErrors:")
            for error in report["errors"]:
                logger.error(f"  - {error}")

        if report["warnings"]:
            logger.warning("\nWarnings:")
            for warning in report["warnings"]:
                logger.warning(f"  - {warning}")

        logger.info("=" * 60)


def validate_dataset_cli():
    """CLI entry point for dataset validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate SVAMITVA dataset")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()

    validator = DataValidator()
    is_valid, report = validator.validate_dataset(Path(args.dataset_dir))
    validator.print_report(report)

    return 0 if is_valid else 1


if __name__ == "__main__":
    import sys

    sys.exit(validate_dataset_cli())
