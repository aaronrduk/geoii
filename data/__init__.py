"""
Data processing module for SVAMITVA feature extraction.

This module contains:
- Dataset classes for loading orthophotos and annotations
- Data augmentation pipelines
- Preprocessing utilities
- DataLoader configurations
"""

__all__ = [
    "SvamitvaDataset",
    "get_train_transforms",
    "get_val_transforms",
    "create_dataloaders",
]
