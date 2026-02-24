"""
Waterbody segmentation head.

This module implements the decoder head for waterbody feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaterbodyHead(nn.Module):
    """
    Neural network head for waterbody segmentation.

    Waterbodies include ponds, rivers, lakes, and other water features.

    Student note: Waterbody segmentation is a binary task.
    """

    def __init__(self, in_channels: int = 256):
        """
        Initialize the waterbody head.

        Args:
            in_channels (int): Number of input channels from backbone
        """
        super(WaterbodyHead, self).__init__()

        # Decoder layers
        # Student note: Standard decoder architecture for segmentation
        self.decoder = nn.Sequential(
            # First upsample block
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Second upsample block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Refinement layer
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary segmentation
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the waterbody head.

        Args:
            x (torch.Tensor): Input features from backbone, shape (N, C, H, W)

        Returns:
            dict: Dictionary containing 'waterbody_mask' (N, 1, H, W)
        """
        # Apply decoder
        features = self.decoder(x)

        # Generate waterbody mask
        waterbody_mask = self.seg_head(features)

        return {"waterbody_mask": waterbody_mask}
