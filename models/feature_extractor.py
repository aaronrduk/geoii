"""
Main feature extractor model combining backbone and all task heads.

This is the complete model architecture for SVAMITVA feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

from .building_head import BuildingHead
from .road_head import RoadHead
from .waterbody_head import WaterbodyHead
from .utility_head import UtilityHead


class FeatureExtractorModel(nn.Module):
    """
    Multi-task feature extraction model for SVAMITVA drone imagery.

    Architecture:
        1. Shared Backbone (ResNet50) - Extracts features from input image
        2. Feature Pyramid Network (FPN) -  Multi-scale feature fusion
        3. Task-specific Heads:
           - Building Head (segmentation + roof classification)
           - Road Head (segmentation)
           - Waterbody Head (segmentation)
           - Utility Head (detection/segmentation)

    Student note: This is a multi-task learning architecture where all tasks
    share the same backbone to learn complementary features efficiently.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_roof_classes: int = 5,
    ):
        """
        Initialize the feature extractor model.

        Args:
            backbone (str): Backbone architecture ('resnet50', 'resnet34', etc.)
            pretrained (bool): Whether to use ImageNet pretrained weights
            num_roof_classes (int): Number of roof type classes (including background)
        """
        super(FeatureExtractorModel, self).__init__()

        # Initialize backbone
        # Student note: We use a pre-trained backbone for transfer learning
        # This helps the model converge faster with limited data
        self.backbone_name = backbone
        self.backbone, self.feature_channels = self._create_backbone(
            backbone, pretrained
        )

        # Feature Pyramid Network (FPN) layers
        # Student note: FPN combines features from different scales
        # This helps detect objects of varying sizes
        self.fpn = self._create_fpn(self.feature_channels)

        # Task-specific decoder heads
        # Student note: Each head is specialized for a specific task
        fpn_channels = 256  # Standard FPN output channels

        self.building_head = BuildingHead(
            in_channels=fpn_channels, num_roof_classes=num_roof_classes
        )
        self.road_head = RoadHead(in_channels=fpn_channels)
        self.waterbody_head = WaterbodyHead(in_channels=fpn_channels)
        self.utility_head = UtilityHead(in_channels=fpn_channels)

        # Initialize weights for new layers
        self._initialize_weights()

    def _create_backbone(self, backbone: str, pretrained: bool):
        """
        Create the backbone network.

        Args:
            backbone (str): Backbone architecture name
            pretrained (bool): Whether to use pretrained weights

        Returns:
            tuple: (backbone_model, feature_channels dict)
        """
        if backbone == "resnet50":
            # Load ResNet50
            # Student note: ResNet50 is a proven architecture for computer vision
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            resnet = models.resnet50(weights=weights)

            # Extract feature layers
            # We'll extract features at multiple scales (C2, C3, C4, C5)
            # Student note: Ci represents features at different spatial resolutions
            backbone_model = nn.ModuleDict(
                {
                    "conv1": resnet.conv1,
                    "bn1": resnet.bn1,
                    "relu": resnet.relu,
                    "maxpool": resnet.maxpool,
                    "layer1": resnet.layer1,  # C2: 256 channels
                    "layer2": resnet.layer2,  # C3: 512 channels
                    "layer3": resnet.layer3,  # C4: 1024 channels
                    "layer4": resnet.layer4,  # C5: 2048 channels
                }
            )

            # Feature channels at each level
            feature_channels = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}

        elif backbone == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = None
            resnet = models.resnet34(weights=weights)
            backbone_model = nn.ModuleDict(
                {
                    "conv1": resnet.conv1,
                    "bn1": resnet.bn1,
                    "relu": resnet.relu,
                    "maxpool": resnet.maxpool,
                    "layer1": resnet.layer1,
                    "layer2": resnet.layer2,
                    "layer3": resnet.layer3,
                    "layer4": resnet.layer4,
                }
            )
            feature_channels = {"C2": 64, "C3": 128, "C4": 256, "C5": 512}
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        return backbone_model, feature_channels

    def _create_fpn(self, feature_channels: Dict[str, int]) -> nn.ModuleDict:
        """
        Create Feature Pyramid Network layers.

        FPN creates a multi-scale feature pyramid by combining features
        from different backbone layers.

        Args:
            feature_channels (dict): Channels at each feature level

        Returns:
            nn.ModuleDict: FPN layers
        """
        fpn_channels = 256  # Standard FPN channel dimension

        # Lateral connections (1x1 conv to reduce channels)
        # Student note: These project features to a common channel dimension
        fpn = nn.ModuleDict(
            {
                "lateral_c5": nn.Conv2d(feature_channels["C5"], fpn_channels, 1),
                "lateral_c4": nn.Conv2d(feature_channels["C4"], fpn_channels, 1),
                "lateral_c3": nn.Conv2d(feature_channels["C3"], fpn_channels, 1),
                "lateral_c2": nn.Conv2d(feature_channels["C2"], fpn_channels, 1),
                # Refinement convolutions (3x3)
                # Student note: These smooth the upsampled features
                "refine_c5": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c4": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c3": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c2": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            }
        )

        return fpn

    def _initialize_weights(self):
        """
        Initialize weights for new layers (not loaded from pretrained).

        Student note: Proper initialization is important for training stability.
        """
        for module in [
            self.fpn,
            self.building_head,
            self.road_head,
            self.waterbody_head,
            self.utility_head,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    # Kaiming initialization for Conv layers
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # Initialize BN layers
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the backbone to extract multi-scale features.

        Args:
            x (torch.Tensor): Input image, shape (N, 3, H, W)

        Returns:
            dict: Multi-scale features {C2, C3, C4, C5}
        """
        # Initial layers
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        x = self.backbone["maxpool"](x)

        # Extract features at multiple scales
        c2 = self.backbone["layer1"](x)  # 1/4 resolution
        c3 = self.backbone["layer2"](c2)  # 1/8 resolution
        c4 = self.backbone["layer3"](c3)  # 1/16 resolution
        c5 = self.backbone["layer4"](c4)  # 1/32 resolution

        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

    def forward_fpn(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through FPN to create multi-scale feature pyramid.

        Args:
            features (dict): Multi-scale features from backbone

        Returns:
            torch.Tensor: Fused FPN features
        """
        # Top-down pathway
        # Student note: We start from the coarsest features and progressively refine

        # C5 (coarsest)
        p5 = self.fpn["lateral_c5"](features["C5"])

        # C4
        p4 = self.fpn["lateral_c4"](features["C4"])
        p4 = p4 + nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")

        # C3
        p3 = self.fpn["lateral_c3"](features["C3"])
        p3 = p3 + nn.functional.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # C2 (finest)
        p2 = self.fpn["lateral_c2"](features["C2"])
        p2 = p2 + nn.functional.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        # Apply refinement convolutions
        p5 = self.fpn["refine_c5"](p5)
        p4 = self.fpn["refine_c4"](p4)
        p3 = self.fpn["refine_c3"](p3)
        p2 = self.fpn["refine_c2"](p2)

        # For this model, we'll use P3 as the main feature map
        # Student note: P3 provides a good balance between spatial resolution
        # and semantic information
        return p3

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through the model.

        Args:
            x (torch.Tensor): Input image batch, shape (N, 3, H, W)

        Returns:
            dict: Dictionary with predictions from all heads
        """
        # Extract backbone features
        backbone_features = self.forward_backbone(x)

        # Apply FPN
        fpn_features = self.forward_fpn(backbone_features)

        # Apply task-specific heads
        # Student note: Each head processes the same FPN features
        # to produce task-specific outputs
        building_outputs = self.building_head(fpn_features)
        road_outputs = self.road_head(fpn_features)
        waterbody_outputs = self.waterbody_head(fpn_features)
        utility_outputs = self.utility_head(fpn_features)

        # Combine all outputs
        outputs = {
            **building_outputs,  # 'building_mask' and 'roof_type'
            **road_outputs,  # 'road_mask'
            **waterbody_outputs,  # 'waterbody_mask'
            **utility_outputs,  # 'utility_mask'
        }

        # Upsample all outputs to match input spatial resolution
        input_h, input_w = x.shape[2], x.shape[3]
        for key in outputs:
            if outputs[key].shape[-2:] != (input_h, input_w):
                outputs[key] = F.interpolate(
                    outputs[key],
                    size=(input_h, input_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return outputs

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """
        Freeze backbone parameters for fine-tuning.

        Student note: Useful when you want to train only the task heads
        while keeping the backbone fixed.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# Test function
if __name__ == "__main__":
    """
    Test the model with dummy input.
    Student note: This is helpful for debugging and understanding model I/O.
    """
    # Create model
    model = FeatureExtractorModel(backbone="resnet50", pretrained=False)

    # Print model info
    print(f"Model created: {model.backbone_name}")
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)

    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        outputs = model(dummy_input)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
