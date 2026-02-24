"""
Waterbody Point detection head.

Detects point water features such as wells, ponds, and water sources.
"""

import torch
import torch.nn as nn


class WaterbodyPointHead(nn.Module):
    """
    Neural network head for point waterbody feature extraction.

    Point features (wells, water sources) are very small objects.
    We use aggressive multi-scale context with small kernels to
    detect these precise locations as small circular blobs.
    """

    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels (int): Number of input channels from FPN.
        """
        super(WaterbodyPointHead, self).__init__()

        # Multi-scale atrous spatial pyramid for small-object context
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Fuse + upsample
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: point feature vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x (torch.Tensor): FPN features (N, C, H, W)
        Returns:
            dict: {'waterbody_point_mask': (N, 1, H, W)}
        """
        # Global context branch (upsample to same spatial size as x)
        pool = self.pool_branch(x)
        pool = pool.expand(-1, -1, x.shape[2], x.shape[3])

        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)

        fused = self.fuse(torch.cat([pool, c1, c2, c3], dim=1))
        return {"waterbody_point_mask": self.seg_head(fused)}
