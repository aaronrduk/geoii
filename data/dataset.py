"""
PyTorch Dataset class for SVAMITVA orthophotos and annotations.

This module implements the main dataset class that loads drone imagery
and corresponding shapefile annotations for training the model.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
import random

from .preprocessing import OrthophotoPreprocessor, ShapefileAnnotationParser
from .augmentation import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SvamitvaDataset(Dataset):
    """
    PyTorch Dataset for SVAMITVA drone imagery and annotations.

    This dataset handles:
    - Loading orthophoto tiles
    - Loading and rasterizing shapefile annotations
    - Creating multi-task labels (buildings, roads, waterbodies, utilities)
    - Roof type classification for buildings
    - Data augmentation

    Student note: This is the main dataset class that PyTorch uses for training.
    The __getitem__ method is called for each training sample.
    """

    def __init__(
        self,
        root_dir: Path,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        mode: str = "train",
        tile_size: Optional[int] = None,
    ):
        """
        Initialize the SVAMITVA dataset.

        Args:
            root_dir (Path): Root directory containing village subdirectories
            image_size (int): Size to resize images to (square)
            transform (Callable, optional): Augmentation/transform pipeline
            mode (str): One of 'train', 'val', or 'test'
            tile_size (int, optional): If provided, extract tiles of this size
                                      from large orthophotos
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.tile_size = tile_size if tile_size else image_size

        # Initialize preprocessors
        self.ortho_preprocessor = OrthophotoPreprocessor()
        self.anno_parser = ShapefileAnnotationParser()

        # Scan for villages and their data
        self.samples = self._scan_dataset()

        logger.info(f"Initialized {mode} dataset with {len(self.samples)} samples")

    def _scan_dataset(self) -> List[Dict]:
        """
        Scan the dataset directory and create a list of samples.

        Each sample contains paths to orthophoto and annotation shapefiles.

        Returns:
            List[Dict]: List of sample dictionaries
        """
        samples = []

        # Student note: The expected directory structure is:
        # root_dir/
        #   village_01/
        #     orthophoto.tif
        #     annotations/
        #       buildings.shp
        #       roads.shp
        #       waterbodies.shp
        #       utilities.shp

        for village_dir in self.root_dir.iterdir():
            if not village_dir.is_dir():
                continue

            # Look for orthophoto file
            orthophoto_path = None
            for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
                potential_path = village_dir / f"orthophoto{ext}"
                if potential_path.exists():
                    orthophoto_path = potential_path
                    break
                # Also check for files with village name
                for file in village_dir.glob(f"*{ext}"):
                    if "ORTHO" in file.stem.upper():
                        orthophoto_path = file
                        break

            if not orthophoto_path:
                logger.warning(f"No orthophoto found in {village_dir}")
                continue

            # Look for annotation directory
            anno_dir = village_dir / "annotations"
            if not anno_dir.exists():
                logger.warning(f"No annotations directory in {village_dir}")
                continue

            # Collect annotation files
            annotations = {}
            for feature_type in ["buildings", "roads", "waterbodies", "utilities"]:
                # Try different possible filenames
                shapefile_path = None
                for pattern in [
                    f"{feature_type}.shp",
                    f"{feature_type.capitalize()}.shp",
                    f"Built_Up_Area*.shp" if feature_type == "buildings" else None,
                    f"Road*.shp" if feature_type == "roads" else None,
                    f"Water*.shp" if feature_type == "waterbodies" else None,
                    f"Utility*.shp" if feature_type == "utilities" else None,
                ]:
                    if pattern:
                        matches = list(anno_dir.glob(pattern))
                        if matches:
                            shapefile_path = matches[0]
                            break

                if shapefile_path and shapefile_path.exists():
                    annotations[feature_type] = shapefile_path
                else:
                    logger.warning(f"No {feature_type} shapefile in {anno_dir}")

            # Only add sample if we have at least one annotation type
            if annotations:
                sample = {
                    "village_name": village_dir.name,
                    "orthophoto": orthophoto_path,
                    "annotations": annotations,
                }
                samples.append(sample)
                logger.info(
                    f"Added sample for {village_dir.name} with {len(annotations)} annotation types"
                )

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.root_dir}")

        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dict containing:
                - 'image': Tensor of shape (C, H, W)
                - 'building_mask': Binary mask for buildings (H, W)
                - 'road_mask': Binary mask for roads (H, W)
                - 'waterbody_mask': Binary mask for waterbodies (H, W)
                - 'utility_mask': Binary mask for utilities (H, W)
                - 'roof_type_mask': Multi-class mask for roof types (H, W)
                - 'metadata': Dictionary with additional information
        """
        sample = self.samples[idx]

        # Load orthophoto
        # Student note: This loads the full high-resolution image
        image, metadata = self.ortho_preprocessor.load_orthophoto(
            sample["orthophoto"], target_size=(self.image_size, self.image_size)
        )

        # Create masks for each feature type
        # Student note: Masks are binary images where 1=feature, 0=background
        height, width = image.shape[:2]
        transform = metadata["transform"]

        masks = {}

        # Process each annotation type
        for feature_type in ["buildings", "roads", "waterbodies", "utilities"]:
            if feature_type in sample["annotations"]:
                # Load shapefile
                shapefile_path = sample["annotations"][feature_type]
                gdf = self.anno_parser.load_shapefile(shapefile_path)

                # Rasterize to create mask
                mask = self.anno_parser.rasterize_annotations(
                    gdf, transform, (height, width), feature_type
                )
                masks[
                    f'{feature_type[:-1] if feature_type.endswith("s") else feature_type}_mask'
                ] = mask
            else:
                # Create empty mask if annotation doesn't exist
                masks[
                    f'{feature_type[:-1] if feature_type.endswith("s") else feature_type}_mask'
                ] = np.zeros((height, width), dtype=np.uint8)

        # Special handling for building roof types
        if "buildings" in sample["annotations"]:
            gdf = self.anno_parser.load_shapefile(sample["annotations"]["buildings"])
            roof_type_mask = self.anno_parser.extract_roof_types(
                gdf, transform, (height, width)
            )
            masks["roof_type_mask"] = roof_type_mask
        else:
            masks["roof_type_mask"] = np.zeros((height, width), dtype=np.uint8)

        # Apply transforms if provided
        # Student note: Transforms handle both image and mask augmentation
        if self.transform:
            transformed = self.transform(
                image=image,
                building_mask=masks.get(
                    "building_mask", np.zeros((height, width), dtype=np.uint8)
                ),
                road_mask=masks.get(
                    "road_mask", np.zeros((height, width), dtype=np.uint8)
                ),
                waterbody_mask=masks.get(
                    "waterbody_mask", np.zeros((height, width), dtype=np.uint8)
                ),
                utility_mask=masks.get(
                    "utility_mask", np.zeros((height, width), dtype=np.uint8)
                ),
                roof_type_mask=masks.get(
                    "roof_type_mask", np.zeros((height, width), dtype=np.uint8)
                ),
            )

            # Extract transformed data
            output = {
                "image": transformed["image"],
                "building_mask": transformed["building_mask"].long(),
                "road_mask": transformed["road_mask"].long(),
                "waterbody_mask": transformed["waterbody_mask"].long(),
                "utility_mask": transformed["utility_mask"].long(),
                "roof_type_mask": transformed["roof_type_mask"].long(),
            }
        else:
            # If no transform, manually convert to tensors
            output = {
                "image": torch.from_numpy(image)
                .permute(2, 0, 1)
                .float(),  # (H,W,C) -> (C,H,W)
                "building_mask": torch.from_numpy(
                    masks.get(
                        "building_mask", np.zeros((height, width), dtype=np.uint8)
                    )
                ).long(),
                "road_mask": torch.from_numpy(
                    masks.get("road_mask", np.zeros((height, width), dtype=np.uint8))
                ).long(),
                "waterbody_mask": torch.from_numpy(
                    masks.get(
                        "waterbody_mask", np.zeros((height, width), dtype=np.uint8)
                    )
                ).long(),
                "utility_mask": torch.from_numpy(
                    masks.get("utility_mask", np.zeros((height, width), dtype=np.uint8))
                ).long(),
                "roof_type_mask": torch.from_numpy(
                    masks.get(
                        "roof_type_mask", np.zeros((height, width), dtype=np.uint8)
                    )
                ).long(),
            }

        # Add metadata
        output["metadata"] = {
            "village_name": sample["village_name"],
            "idx": idx,
        }

        return output

    def get_class_distribution(self) -> Dict[str, float]:
        """
        Calculate the distribution of different classes in the dataset.

        Useful for understanding class imbalance and setting loss weights.

        Returns:
            Dict[str, float]: Class distribution statistics
        """
        # This would require iterating through the entire dataset
        # For now, return placeholder - implement if needed for loss weighting
        logger.info("Class distribution calculation not implemented yet")
        return {}


def create_dataloaders(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512,
    val_split: float = 0.2,
) -> Tuple:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_dir (Path): Training data directory
        val_dir (Path, optional): Validation data directory. If None, will split train_dir
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        image_size (int): Image size
        val_split (float): Fraction of data to use for validation when val_dir is None

    Returns:
        Tuple: (train_loader, val_loader) where val_loader may be None if no validation
    """
    # Student note: DataLoaders handle batching and shuffling of data

    # Create validation dataset if directory is provided
    if val_dir:
        # Separate train and val directories provided
        train_dataset = SvamitvaDataset(
            root_dir=train_dir,
            image_size=image_size,
            transform=get_train_transforms(image_size),
            mode="train",
        )

        val_dataset = SvamitvaDataset(
            root_dir=val_dir,
            image_size=image_size,
            transform=get_val_transforms(image_size),
            mode="val",
        )
    else:
        # Split train_dir into train and validation
        full_dataset = SvamitvaDataset(
            root_dir=train_dir,
            image_size=image_size,
            transform=get_train_transforms(image_size),
            mode="train",
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        if val_size == 0:
            logger.warning("Validation split is 0. No validation will be performed.")
            train_dataset = full_dataset
            val_dataset = None
        else:
            # Split dataset randomly
            from torch.utils.data import random_split

            train_dataset, val_dataset_temp = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),  # For reproducibility
            )

            # Create new dataset for validation with val transforms
            val_dataset = SvamitvaDataset(
                root_dir=train_dir,
                image_size=image_size,
                transform=get_val_transforms(image_size),
                mode="val",
            )
            # Use the same indices as the split
            val_dataset.samples = [
                full_dataset.samples[i] for i in val_dataset_temp.indices
            ]

            logger.info(
                f"Split dataset: {train_size} train, {val_size} validation samples"
            )

    # Create train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        drop_last=(
            True if len(train_dataset) > batch_size else False
        ),  # Drop incomplete batches
    )

    # Create validation loader if dataset exists
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    return train_loader, None
