"""
Building segmentation and roof classification head.

This module implements the decoder head for building footprint extraction
and roof-type classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingHead(nn.Module):
    """
    Neural network head for building extraction with roof type classification.

    This module performs two tasks:
    1. Binary segmentation: Detect building footprints
    2. Multi-class classification: Classify roof types (RCC, Tiled, Tin, Others)

    Student note: This is a multi-task head that shares features between
    segmentation and classification for efficiency.
    """

    def __init__(self, in_channels: int = 256, num_roof_classes: int = 5):
        """
        Initialize the building head.

        Args:
            in_channels (int): Number of input channels from backbone
            num_roof_classes (int): Number of roof type classes (including background)
        """
        super(BuildingHead, self).__init__()

        # Shared decoder layers
        # Student note: These layers upsample and refine features from the backbone
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
            # Third upsample block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Building segmentation head
        # Student note: This outputs a single channel (building vs background)
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # 1 channel for binary segmentation
        )

        # Roof type classification head
        # Student note: This outputs multiple channels (one per roof type)
        self.roof_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_roof_classes, kernel_size=1),  # Multi-class output
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the building head.

        Args:
            x (torch.Tensor): Input features from backbone, shape (N, C, H, W)

        Returns:
            dict: Dictionary containing:
                - 'building_mask': Building segmentation logits (N, 1, H, W)
                - 'roof_type': Roof type classification logits (N, num_classes, H, W)
        """
        # Apply shared decoder
        # Student note: Both tasks benefit from the same high-level features
        features = self.decoder(x)

        # Building segmentation output
        building_mask = self.seg_head(features)

        # Roof type classification output
        roof_type = self.roof_head(features)

        return {"building_mask": building_mask, "roof_type": roof_type}
