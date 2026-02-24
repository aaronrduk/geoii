"""
Data augmentation pipeline for SVAMITVA imagery.

This module provides augmentation transforms for training robust models.
Uses albumentations library which is optimized for geospatial data.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Any

# Student note: Data augmentation is crucial for training with limited data
# It creates variations of the same image to help the model generalize better


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        image_size (int): Size to resize images to (square)

    Returns:
        A.Compose: Compiled augmentation pipeline
    """
    # Student note: These transforms are carefully chosen for geospatial imagery
    # We avoid transforms that would distort spatial relationships

    return A.Compose(
        [
            # Geometric transforms
            # These help the model be invariant to rotation and flipping
            A.RandomRotate90(p=0.5),  # Rotate by 0, 90, 180, or 270 degrees
            A.HorizontalFlip(p=0.5),  # Horizontal flip
            A.VerticalFlip(p=0.5),  # Vertical flip
            A.Transpose(p=0.3),  # Transpose the image
            # Random crop and resize
            # This helps the model focus on different parts of the image
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),  # Crop between 80-100% of original
                ratio=(0.9, 1.1),  # Maintain roughly square aspect ratio
                p=1.0,
            ),
            # Color augmentations
            # These help the model handle different lighting conditions
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=15,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0
                    ),
                ],
                p=0.7,
            ),
            # Blur and noise
            # These simulate different image quality conditions
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            # Add noise
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            # Normalize with ImageNet statistics
            # Student note: This is standard practice for transfer learning
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,  # Our images are already normalized to [0, 1]
            ),
            # Convert to PyTorch tensor
            ToTensorV2(),
        ],
        # Additional targets for masks
        # Student note: This ensures masks are transformed along with images
        additional_targets={
            "building_mask": "mask",
            "road_mask": "mask",
            "road_centerline_mask": "mask",
            "waterbody_mask": "mask",
            "waterbody_line_mask": "mask",
            "waterbody_point_mask": "mask",
            "utility_line_mask": "mask",
            "utility_poly_mask": "mask",
            "bridge_mask": "mask",
            "railway_mask": "mask",
            "roof_type_mask": "mask",
        },
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Get validation data transforms (no augmentation, just preprocessing).

    Args:
        image_size (int): Size to resize images to

    Returns:
        A.Compose: Compiled transform pipeline
    """
    # Student note: Validation should not use augmentation
    # We want to evaluate on clean data to get accurate metrics

    return A.Compose(
        [
            # Just resize to target size
            A.Resize(height=image_size, width=image_size),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
            ),
            # Convert to tensor
            ToTensorV2(),
        ],
        additional_targets={
            "building_mask": "mask",
            "road_mask": "mask",
            "road_centerline_mask": "mask",
            "waterbody_mask": "mask",
            "waterbody_line_mask": "mask",
            "waterbody_point_mask": "mask",
            "utility_line_mask": "mask",
            "utility_poly_mask": "mask",
            "bridge_mask": "mask",
            "railway_mask": "mask",
            "roof_type_mask": "mask",
        },
    )


def get_test_transforms(image_size: int = 512) -> A.Compose:
    """
    Get test-time transforms (same as validation).

    Args:
        image_size (int): Size to resize images to

    Returns:
        A.Compose: Compiled transform pipeline
    """
    # Test transforms are identical to validation transforms
    return get_val_transforms(image_size)


def get_tta_transforms(image_size: int = 512) -> list:
    """
    Get Test-Time Augmentation (TTA) transforms.

    TTA applies multiple augmentations at test time and averages the results
    to improve prediction robustness.

    Args:
        image_size (int): Size to resize images to

    Returns:
        list: List of transform pipelines for TTA
    """
    # Student note: TTA can improve accuracy but slows down inference
    # Use this for the final evaluation on test dataset

    tta_transforms = []

    # Original (no augmentation)
    tta_transforms.append(get_test_transforms(image_size))

    # Horizontal flip
    tta_transforms.append(
        A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    )

    # Vertical flip
    tta_transforms.append(
        A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    )

    # 90 degree rotation
    tta_transforms.append(
        A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Rotate(limit=(90, 90), p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    )

    return tta_transforms


# Utility function for visualizing augmentations
def visualize_augmentation(
    image, masks: Dict[str, Any], transform, num_examples: int = 4
):
    """
    Visualize the effect of augmentation on images and masks.

    Useful for debugging and understanding what augmentations are doing.

    Args:
        image: Input image (numpy array)
        masks: Dictionary of masks
        transform: Augmentation pipeline
        num_examples: Number of augmented examples to show
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))

    for i in range(num_examples):
        # Apply augmentation
        augmented = transform(image=image, **masks)

        aug_image = augmented["image"]
        aug_building_mask = augmented.get("building_mask", None)

        # Display original image
        if i == 0:
            axes[i, 0].set_title("Augmented Image")
            axes[i, 1].set_title("Building Mask")
            axes[i, 2].set_title("Overlay")

        # Convert tensor back to numpy if needed
        if hasattr(aug_image, "numpy"):
            aug_image = aug_image.numpy().transpose(1, 2, 0)
            # Denormalize for visualization
            aug_image = aug_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            aug_image = np.clip(aug_image, 0, 1)

        axes[i, 0].imshow(aug_image)
        axes[i, 0].axis("off")

        if aug_building_mask is not None:
            axes[i, 1].imshow(aug_building_mask, cmap="gray")
            axes[i, 1].axis("off")

            # Overlay
            overlay = aug_image.copy()
            overlay[aug_building_mask > 0] = [1, 0, 0]  # Red overlay for buildings
            axes[i, 2].imshow(overlay)
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=150, bbox_inches="tight")
    print("Saved augmentation examples to augmentation_examples.png")
    plt.close()
