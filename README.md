# SVAMITVA Feature Extraction Model

A production-ready AI/ML model for automated feature extraction from SVAMITVA drone orthophotos. This model can identify and classify building footprints (with roof types), roads, waterbodies, and utilities with high accuracy.

## 🎯 Features

- **Building Footprint Extraction**: Detects buildings and classifies roof types (RCC, Tiled, Tin, Others)
- **Road Feature Extraction**: Segments roads and extracts centerlines
- **Waterbody Detection**: Identifies water bodies (ponds, rivers, lakes)
- **Utility Infrastructure**: Detects transformers, overhead tanks, and wells
- **QGIS Plugin**: One-click feature extraction directly in QGIS
- **High Accuracy**: Target accuracy of 95% on test datasets

## 🏗️ Architecture

The model uses a multi-task learning approach with:
- **Backbone**: ResNet50 pre-trained encoder
- **Feature Pyramid Network (FPN)** for multi-scale features
- **Task-specific heads**: Separate decoders for each feature type
- **Roof-type classifier**: Multi-class classification for building roofs

```
Input Orthophoto (512×512)
    ↓
ResNet50 Encoder
    ↓
Feature Pyramid Network
    ↓
├── Building Head → Segmentation + Roof Classification
├── Road Head → Segmentation
├── Waterbody Head → Segmentation
└── Utility Head → Object Detection
```

## 📦 Installation

### 1. Clone the Repository
```bash
cd /Users/aaronr/Desktop/DEVELOPMENT/AI/AA
git clone <repository-url> svamitva_model  # If using git
cd svamitva_model
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Organize your training data in the following structure:
```
dataset/
├── train/
│   ├── village_01/
│   │   ├── orthophoto.tif
│   │   └── annotations/
│   │       ├── buildings.shp
│   │       ├── roads.shp
│   │       ├── waterbodies.shp
│   │       └── utilities.shp
│   ├── village_02/
│   └── ...
└── test/
    └── ... (testing villages - to be added later)
```

## 🚀 Quick Start

### Training

```bash
# Basic training
python train.py --config configs/default.yaml

# Custom configuration
python train.py \
  --config configs/default.yaml \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 0.001
```

### Inference

```bash
# Run inference on a single orthophoto
python inference/predictor.py \
  --checkpoint checkpoints/best_model.pth \
  --input dataset/test/village_01/orthophoto.tif \
  --output results/village_01/

# Batch processing
python inference/predictor.py \
  --checkpoint checkpoints/best_model.pth \
  --input_dir dataset/test/ \
  --output_dir results/
```

### Evaluation

```bash
# Evaluate on test dataset
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --test_dir dataset/test/ \
  --output results/evaluation_report.json
```

## 🔌 QGIS Plugin

### Installation
1. Open QGIS
2. Go to `Plugins → Manage and Install Plugins → Install from ZIP`
3. Select `qgis_plugin.zip`
4. Restart QGIS

### Usage
1. Load your orthophoto in QGIS
2. Open the SVAMITVA Feature Extraction panel
3. Select the loaded raster layer
4. Click "Extract Features"
5. Separate vector layers will be created for each feature type

## 📊 Performance Metrics

Target accuracy metrics:
- **Building IoU**: > 0.85
- **Roof Classification Accuracy**: > 0.90
- **Road IoU**: > 0.80
- **Waterbody IoU**: > 0.85
- **Utility Detection mAP**: > 0.80

## 📁 Project Structure

```
svamitva_model/
├── data/               # Data loading and preprocessing
├── models/             # Neural network architectures
├── training/           # Training pipeline and metrics
├── inference/          # Inference and post-processing
├── qgis_plugin/        # QGIS plugin code
├── utils/              # Utility functions
├── tests/              # Unit and integration tests
├── docs/               # Documentation
├── configs/            # Configuration files
├── checkpoints/        # Saved model checkpoints
├── dataset/            # Training and testing data
└── results/            # Inference and evaluation results
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
The codebase follows best practices with:
- Comprehensive inline comments for student developers
- Modular architecture for easy understanding
- Type hints for better code clarity
- Detailed docstrings for all functions

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed model architecture
- [Training Guide](docs/TRAINING_GUIDE.md) - Step-by-step training instructions
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment
- [User Manual](docs/USER_MANUAL.md) - QGIS plugin usage

## 🤝 Contributing

This project is designed for the SVAMITVA hackathon. The code is structured to be:
- **Student-friendly**: Well-commented and easy to understand
- **Modular**: Each component can be modified independently
- **Production-ready**: Optimized for deployment and real-world use

## 📄 License

[Add your license here]

## 👥 Authors

[Add your team information]

## 🙏 Acknowledgments

- SVAMITVA Scheme for providing the dataset
- TerraLab AI for inspiration from their AI Segmentation plugin
