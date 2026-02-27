# How the SVAMITVA Model Works

## 1. Data Preparation
### The Input Data
The model operates on high-resolution drone Orthophotos (GeoTIFFs) accompanied by manually digitized ground-truth Shapefiles (`.shp`). Each Shapefile represents a distinct geographical feature class, such as "Buildings", "Roads", or "Waterbodies".

### Tiling & Geometry Parsing
Because Orthophotos are far too large to fit into VRAM natively, they are sliced into **512x512 pixel tiles** (with 64px sliding overlap) by a PyTorch Dataset class during processing.

- **CRS Enforcement:** Before Shapefiles are processed, `data/preprocessing.py` checks exactly what Coordinate Reference System (CRS) the GeoTIFF image is inherently tied to (e.g., EPSG:32643). The Shapefile geometrical vectors are forcefully re-projected (`to_crs()`) into this native coordinate system to guarantee absolute geographical alignment across all datasets.
- **Rasterization:** The aligned Polygons, LineStrings, and Points are "burned" into physical binary image masks (1 for presence, 0 for background) using the `rasterio.features.rasterize()` function, forming identical 512x512 target boundaries.

### Smart K-Means Area Filtering
Since an Orthophoto is generally rotated over an arbitrary bounding box, loading and training on huge black/transparent edge "NoData" regions is computationally wasteful and mathematically unstable.
- `data/dataset.py` reads a 1024x1024 thumbnail of the image and performs OpenCV **K-Means Clustering** (k=3 or 4) on it.
- It automatically isolates the dominating color cluster (which is universally the `#000000` or `#FFFFFF` boundary edges).
- As the 512x512 tile window slides dynamically across the enormous GeoTIFF, if its corresponding position in the thumbnail is 100% assigned to the background cluster, the tile is completely skipped before any disk I/O occurs.

## 2. Model Architecture
### Backbone & Predictor
The architecture utilizes a unified ResNet backbone (`ResNet34` locally or `ResNet50` on DGX) mapped into a PyTorch Feature Pyramid Network (`FPN`). This configuration detects multiscale features (like massive factories spanning entire tiles down to ultra-narrow utility centerlines) hierarchically.

### Independent Task Heads
For every 10 features, there is a dedicated localized "Head" branch connected to the FPN (`models/building_head.py`, `models/road_head.py`, etc.). 
- The model essentially accepts a `[3, 512, 512]` RGB terrain matrix. 
- It isolates 10 separate `[H, W]` independent binary logit masks, plus an optional Multi-Class probability tensor for Roof Type classification over buildings.

## 3. Training Process
### Loss Tracking
Because certain features naturally manifest significantly more frequently than others (e.g. massive building footprints vs. sparse railway networks), the universal `MultiTaskLoss` loss function computes Binary Cross-Entropy (BCE) combined seamlessly with Dice Loss arrays, weighted via the hyperparameter config dictionary specifically balancing the 10 branches.

### Hardware Validation & Checkpoints
- The training relies on PyTorch's atomic check-pointing logic, inherently preserving the precise epoch where the validation Intersection-over-Union (IoU) was highest.
- The DGX Server orchestrator (`SVAMITVA_DGX_Train.ipynb`) loops sequentially across all folders (`MAP1`, `MAP2`), hot-loading from the latest `.pt` snapshot sequentially so the training accumulates intelligence sequentially using the `GradScaler` for Mixed-Precision speed without exploding `NaN` values.

## 4. Inference & Feature Export
The Streamlit frontend (`app.py`) demonstrates real-time inference effortlessly:
1. A user uploads an arbitrary GeoTIFF, and PyTorch dynamically feeds memory-safe chunks via sliding windows to the checkpoint.
2. OpenCV overlays the reconstructed probability arrays visibly on screen using color-coded metrics.
3. Upon downloading, the script reverses the pipeline: it uses `fiona/shapely` to convert the binary prediction arrays *back into Geometries* (MultiPolygons, LineStrings), copying the Native Transform affine array from the source image. Resultingly, outputs slide perfectly back into tools like QGIS/ArcGIS as flawless, fully-projected geographical Shapefiles.
