# SVAMITVA Feature Extraction Model

A highly optimized Multi-Task Deep Learning model designed to rapidly extract 10 distinct geospatial features from large-scale drone Orthophotos (GeoTIFFs). Developed for the SVAMITVA land survey programme.

## Features
- **10 Core Feature Masks**: Extracts Buildings, Roads, Waterbodies, Utilities, Bridges, and Railways representing various geometries (Polygons, Lines, Points). Includes a secondary classifier for Roof Type classification.
- **Intelligent Preprocessing (K-Means)**: Automatically filters out massive "NoData" background regions using OpenCV K-Means clustering, drastically speeding up data loading and guaranteeing the model only looks at real terrain.
- **Strict CRS Alignment**: Enforces perfect Coordinate Reference System (CRS) matching between Shapefiles and GeoTIFFs. 
- **Universal Training**: Dedicated pipelines for both Apple Silicon (Local macOS) and NVIDIA CUDA (DGX Servers) with `NaN` explosion protection, Mixed Precision support, and resume-ready check-pointing.
- **Interactive Frontend**: A Streamlit application capable of processing huge TIFFs via sliding windows and exporting results back directly into fully-georeferenced Shapefiles (.shp).

## Dataset Structure (Remote Server)
Every `MAP` folder sits flat — containing the orthophoto and corresponding shapefiles directly.

```
/path/to/DATA/
    MAP1/
        Orthophoto.tif
        Built_Up_Area_typ.shp
        Road.shp
        ...
    MAP2/
```

## Model Outputs (10 Shapefile Tasks)

| Output key | Target Feature | Geometry |
|---|---|---|
| `building_mask` | Built_Up_Area_typ | Polygon |
| `roof_type` | Built_Up_Area_typ | Polygon (Multi-class) |
| `road_mask` | Road | Polygon |
| `road_centerline_mask` | Road_Centre_Line | Line |
| `waterbody_mask` | Water_Body | Polygon |
| `waterbody_line_mask` | Water_Body_Line | Line |
| `waterbody_point_mask` | Waterbody_Point | Point |
| `utility_line_mask` | Utility | Line |
| `utility_poly_mask` | Utility_Poly_ | Polygon |
| `bridge_mask` | Bridge | Polygon/Line |
| `railway_mask` | Railway | Line |

## Running the Pipelines

### 1. DGX Server Training (Sequential Maps)
```bash
jupyter nbconvert SVAMITVA_DGX_Train.ipynb --execute
```
*Automatically finds all MAP folders, sets up Mixed Precision, and loops sequentially keeping weights persistent.*

### 2. Local Apple Silicon Prototype
```bash
jupyter nbconvert SVAMITVA_Local_Train.ipynb --execute
```
*Configured for `mps` devices to train a lighter ResNet34 strictly on MAP1.*

### 3. Streamlit Interface
```bash
streamlit run app.py
```
*Exposes a local web-UI on `localhost:8501`. Upload any image to evaluate the latest `.pt` checkpoint and download Shapefile exports.*

---
To learn exactly how data is loaded, how annotations are rasterized, and how the model trains, please read `WORKING.md`.
