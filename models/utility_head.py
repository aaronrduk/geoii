"""
Utility infrastructure detection head.

This module implements the decoder head for utility detection
(transformers, overhead tanks, wells, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityHead(nn.Module):
    """
    Neural network head for utility infrastructure detection.

    Utilities are point-like or small polygon features that require
    combining segmentation with object detection principles.

    Student note: We use segmentation approach for utilities,
    treating them as small regions rather than explicit bounding boxes.
    """

    def __init__(self, in_channels: int = 256):
        """
        Initialize the utility head.

        Args:
            in_channels (int): Number of input channels from backbone
        """
        super(UtilityHead, self).__init__()

        # Decoder with attention to small objects
        # Student note: We use smaller kernel sizes to preserve fine details
        self.decoder = nn.Sequential(
            # First block - preserve spatial resolution
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # Second block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # Refinement with smaller kernels for fine details
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Segmentation head for utilities
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary segmentation
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the utility head.

        Args:
            x (torch.Tensor): Input features from backbone, shape (N, C, H, W)

        Returns:
            dict: Dictionary containing 'utility_mask' (N, 1, H, W)
        """
        # Apply decoder
        features = self.decoder(x)

        # Generate utility mask
        utility_mask = self.seg_head(features)

        return {"utility_mask": utility_mask}
