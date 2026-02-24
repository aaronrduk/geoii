# SVAMITVA Feature Extraction Model

Multi-task deep learning model for extracting geospatial features from
drone orthophotos, as part of the SVAMITVA land survey programme.

## Dataset Structure (Remote Server)

Each MAP folder sits flat — the TIF and all shapefiles are directly inside
the MAP folder. No `annotations/` sub-folder is needed.

```
/path/to/maps/
    MAP1/
        village.tif
        Built_Up_Area_typ.shp
        Road.shp
        Road_Centre_Line.shp
        Utility.shp
        Utility_Poly_.shp
        Water_Body.shp
        Water_Body_Line.shp
        Waterbody_Point.shp
        Bridge.shp
        Railway.shp
    MAP2/ ... MAP5/
```

## Model Outputs (10 shapefile tasks)

| Output key | Shapefile | Geometry |
|---|---|---|
| `building_mask` | Built_Up_Area_typ | Polygon |
| `roof_type` | Built_Up_Area_typ | Polygon (multi-class) |
| `road_mask` | Road | Polygon |
| `road_centerline_mask` | Road_Centre_Line | Line |
| `waterbody_mask` | Water_Body | Polygon |
| `waterbody_line_mask` | Water_Body_Line | Line |
| `waterbody_point_mask` | Waterbody_Point | Point |
| `utility_line_mask` | Utility | Line |
| `utility_poly_mask` | Utility_Poly_ | Polygon |
| `bridge_mask` | Bridge | Polygon/Line |
| `railway_mask` | Railway | Line |

## Training

```bash
python3 train.py \
  --train_dir /path/to/MAP_parent_folder \
  --batch_size 8 \
  --epochs 100 \
  --backbone resnet50 \
  --image_size 512
```

## Project Structure

```
svamitva_model/
├── models/
│   ├── feature_extractor.py     # Main model (backbone + FPN + all heads)
│   ├── building_head.py
│   ├── road_head.py
│   ├── road_centerline_head.py
│   ├── waterbody_head.py
│   ├── waterbody_line_head.py
│   ├── waterbody_point_head.py
│   ├── utility_head.py          # UtilityLineHead + UtilityPolyHead
│   ├── bridge_head.py
│   ├── railway_head.py
│   └── losses.py
├── data/
│   ├── dataset.py               # MAP*-aware dataset scanner
│   ├── preprocessing.py         # Orthophoto loader + shapefile rasteriser
│   └── augmentation.py
├── training/
│   ├── config.py
│   └── metrics.py
├── utils/
│   ├── checkpoint.py
│   └── logging_config.py
├── app.py                       # Streamlit inference app
└── train.py
```

## Requirements

```
torch torchvision rasterio geopandas shapely albumentations tqdm
```
