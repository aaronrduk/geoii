"""
SVAMITVA Feature Extraction Model

A production-ready AI/ML model for automated feature extraction from drone imagery.
"""

__version__ = "1.0.0"
__author__ = "Your Team Name"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "configs"
