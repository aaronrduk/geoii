"""
Road segmentation head.

This module implements the decoder head for road feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadHead(nn.Module):
    """
    Neural network head for road segmentation.

    Roads are linear features that require special attention to connectivity
    and continuity in the segmentation.

    Student note: Road segmentation is a binary task (road vs non-road).
    """

    def __init__(self, in_channels: int = 256):
        """
        Initialize the road head.

        Args:
            in_channels (int): Number of input channels from backbone
        """
        super(RoadHead, self).__init__()

        # Decoder with attention to linear features
        # Student note: We use dilated convolutions to capture long-range context
        # which is important for road connectivity
        self.decoder = nn.Sequential(
            # Dilated conv for wider receptive field
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Standard conv
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Refinement layer
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary segmentation
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the road head.

        Args:
            x (torch.Tensor): Input features from backbone, shape (N, C, H, W)

        Returns:
            dict: Dictionary containing 'road_mask' (N, 1, H, W)
        """
        # Apply decoder
        features = self.decoder(x)

        # Generate road mask
        road_mask = self.seg_head(features)

        return {"road_mask": road_mask}
