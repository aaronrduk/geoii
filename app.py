"""
SVAMITVA Feature Extraction — Streamlit App
Supports JPG, JPEG, PNG, TIF, TIFF (including large GeoTIFFs).
Extracts 10 feature layers, exports Shapefiles with original CRS.
"""

import io
import os
import sys
import zipfile
import tempfile
from pathlib import Path


import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection for large images

sys.path.insert(0, str(Path(__file__).parent))
from models.feature_extractor import FeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVAMITVA Feature Extraction",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
<style> 
.stApp { background: linear-gradient(135deg,#0f0c29,#302b63,#24243e); }
.main .block-container { padding-top:2rem; }
h1,h2,h3 { color:#e0e0ff !important; }
.stMetric label { color:#a0a0d0 !important; }
.stMetric [data-testid="stMetricValue"] { color:#ffffff !important; }
.feature-card {
    background:rgba(255,255,255,0.06); border-radius:12px;
    padding:1.2rem; border:1px solid rgba(255,255,255,0.1); margin-bottom:1rem;
}
.not-detected {
    background:rgba(255,80,80,0.15); border:1px solid rgba(255,80,80,0.3);
    border-radius:8px; padding:1rem; text-align:center;
    color:#ff9999; font-size:1.05rem; margin-bottom:0.5rem;
}
.detected-ok {
    background:rgba(50,220,100,0.12); border:1px solid rgba(50,220,100,0.3);
    border-radius:8px; padding:0.6rem 1rem; color:#80ffbb;
    display:inline-block; margin-bottom:0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — 10 feature layers
# ─────────────────────────────────────────────────────────────────────────────
TILE_SIZE = 512
OVERLAP = 64

FEATURE_META = {
    # key : (emoji+label, colour RGB, geometry_type, attr_prefix, sub_types)
    # sub_types is a list of names cycled when labelling individual features.
    # For most layers it is a single-element list; for utility_poly it lists
    # the distinct infrastructure objects so exported attributes read
    # "Overhead Tank 1", "Transformer 2", etc.
    "building_mask": (
        "🏠 Buildings", (255, 60, 60), "Polygon",
        "Building", ["Building"],
    ),
    "road_mask": (
        "🛣️ Roads", (255, 220, 50), "Polygon",
        "Road", ["Road"],
    ),
    "road_centerline_mask": (
        "〰️ Road Centrelines", (255, 160, 30), "LineString",
        "Road Centreline", ["Road Centreline"],
    ),
    "waterbody_mask": (
        "💧 Waterbodies", (50, 120, 255), "Polygon",
        "Waterbody", ["Waterbody"],
    ),
    "waterbody_line_mask": (
        "〜 Water Lines", (80, 180, 255), "LineString",
        "Water Line", ["Canal", "Drain", "Water Line"],
    ),
    "waterbody_point_mask": (
        "🔵 Water Points", (150, 220, 255), "Point",
        "Water Point", ["Well", "Water Point"],
    ),
    "utility_line_mask": (
        "⚡ Utility Lines", (50, 220, 100), "LineString",
        "Utility Line", ["Pipeline", "Overhead Wire", "Utility Line"],
    ),
    "utility_poly_mask": (
        "🔌 Utility Areas", (100, 255, 150), "Polygon",
        "Utility Structure", ["Overhead Tank", "Transformer", "Pump House", "Substation"],
    ),
    "bridge_mask": (
        "🌉 Bridges", (220, 130, 50), "Polygon",
        "Bridge", ["Bridge"],
    ),
    "railway_mask": (
        "🚂 Railways", (180, 80, 255), "LineString",
        "Railway", ["Railway"],
    ),
}

ROOF_TYPES = ["RCC", "Tiled", "Tin", "Others", "Unknown"]

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint auto-detection
# ─────────────────────────────────────────────────────────────────────────────
CKPT_DIR = Path(__file__).parent / "checkpoints"


def _find_latest_checkpoint() -> Path | None:
    """Return the most recently modified .pt file in checkpoints/."""
    if not CKPT_DIR.exists():
        return None
    pts = sorted(
        CKPT_DIR.rglob("*best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return pts[0] if pts else None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_backbone(weights: dict) -> str:
    """Auto-detect backbone from checkpoint weights (resnet34 vs resnet50).

    ResNet-50 layer1.0 has 3 conv layers (bottleneck) while ResNet-34 has 2
    (basic block).  We check for the presence of a bottleneck key.
    """
    for key in weights:
        if "backbone.layer1.0.conv3.weight" in key:
            return "resnet50"
    return "resnet34"


@st.cache_resource
def load_model(ckpt_path_str: str):
    ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None

    # Try to detect backbone from checkpoint first
    backbone = "resnet50"  # default for DGX-trained models
    if ckpt_path and ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = (
            state_dict.get("model") or state_dict.get("model_state_dict") or state_dict
        )
        backbone = _detect_backbone(weights)
    else:
        weights = None

    model = FeatureExtractor(backbone=backbone, pretrained=False, num_roof_classes=5)

    if weights is not None:
        model.load_state_dict(weights, strict=False)
        return model.to(DEVICE).eval(), str(ckpt_path.name), True
    return model.to(DEVICE).eval(), "untrained", False


# ─────────────────────────────────────────────────────────────────────────────
# Image loading — JPG / PNG / TIF with CRS preservation
# ─────────────────────────────────────────────────────────────────────────────
def load_image_with_meta(uploaded_file):
    """
    Returns (image_np [H,W,3] uint8, geo_meta dict or None).
    geo_meta contains rasterio transform + crs for use in shapefile export.
    """
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()

    if ext in ("jpg", "jpeg", "png"):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img), None

    if ext in ("tif", "tiff"):
        try:
            import rasterio
            from rasterio.io import MemoryFile

            with MemoryFile(raw) as memfile:
                with memfile.open() as src:
                    transform = src.transform
                    crs = src.crs
                    bands = src.read()  # (C, H, W)
                    profile = src.profile.copy()

            if bands.shape[0] >= 3:
                img = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            else:
                img = np.stack([bands[0]] * 3, axis=-1)

            # Normalise to uint8
            if img.dtype != np.uint8:
                mn, mx = img.min(), img.max()
                if mx > mn:
                    img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            geo_meta = {"transform": transform, "crs": crs, "profile": profile}
            return img, geo_meta

        except ImportError:
            st.error("rasterio not installed — cannot read GeoTIFF.")
            return None, None

    st.error(f"Unsupported file type: .{ext}")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Tiled inference (handles very large images)
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_tile(tile_np: np.ndarray) -> torch.Tensor:
    t = tile_np.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    return torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).float()


def run_inference(image_np: np.ndarray, model: nn.Module, selected: list[str]):
    """
    Run tiled inference on image_np.
    Returns dict[str -> np.ndarray(H,W) float32] for each selected mask key.
    """
    h, w = image_np.shape[:2]
    stride = TILE_SIZE - OVERLAP
    all_keys = selected + (["roof_type"] if "building_mask" in selected else [])

    accum = {k: np.zeros((h, w), dtype=np.float32) for k in selected}
    counts = np.zeros((h, w), dtype=np.float32)
    roof_accum = (
        np.zeros((5, h, w), dtype=np.float32) if "building_mask" in selected else None
    )

    ys = list(range(0, max(h - TILE_SIZE, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - TILE_SIZE, 0) + 1, stride)) or [0]
    total = len(ys) * len(xs)
    prog = st.progress(0, text="Running inference…")

    done = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + TILE_SIZE, h)
            x1 = min(x0 + TILE_SIZE, w)
            th, tw = y1 - y0, x1 - x0

            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            tile[:th, :tw] = image_np[y0:y1, x0:x1]

            inp = _preprocess_tile(tile).to(DEVICE)
            with torch.no_grad():
                out = model(inp)

            for k in selected:
                if k in out:
                    pred = torch.sigmoid(out[k]).cpu().numpy()[0, 0]
                    accum[k][y0:y1, x0:x1] += pred[:th, :tw]

            if roof_accum is not None and "roof_type" in out:
                rp = out["roof_type"].cpu().numpy()[0]
                roof_accum[:, y0:y1, x0:x1] += rp[:, :th, :tw]

            counts[y0:y1, x0:x1] += 1.0
            done += 1
            prog.progress(done / total, text=f"Tile {done}/{total}")

    prog.empty()

    results = {k: accum[k] / np.maximum(counts, 1.0) for k in selected}
    if roof_accum is not None:
        results["roof_type"] = roof_accum / np.maximum(counts[None], 1.0)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Shapefile export — preserves CRS from input GeoTIFF
# ─────────────────────────────────────────────────────────────────────────────
def _convert_geometry(geom, target_type: str):
    """
    Convert a shapely geometry to the requested target type.

    Polygon  → Polygon  : identity
    Polygon  → LineString: extract boundary (exterior ring)
    Polygon  → Point     : centroid
    Multi*   → handled recursively via .geoms
    """
    from shapely.geometry import (
        MultiPolygon, MultiLineString, MultiPoint,
    )
    from shapely.ops import linemerge

    gtype = type(geom).__name__

    if target_type == "Polygon":
        # Already polygon from rasterio.features.shapes → keep as-is
        return geom

    if target_type == "LineString":
        if gtype in ("Polygon",):
            return geom.exterior  # returns LinearRing (valid LineString)
        if gtype == "MultiPolygon":
            lines = [p.exterior for p in geom.geoms]
            merged = linemerge(lines)
            return merged
        return geom  # already a line

    if target_type == "Point":
        return geom.centroid

    return geom


def mask_to_shp_zip(
    mask: np.ndarray,
    threshold: float,
    feature_name: str,
    geo_meta: dict | None,
    geom_type: str = "Polygon",
    attr_prefix: str = "Feature",
    sub_types: list[str] | None = None,
    roof_cls_map: np.ndarray | None = None,
) -> bytes | None:
    """
    Vectorise a probability mask → shapefile ZIP.

    Geometry handling:
        - Polygon layers  → exported as Polygon
        - LineString layers (roads centrelines, railways, etc.) → polygon
          boundaries are converted to LineStrings
        - Point layers (water points) → polygon centroids become Points

    Attribute naming:
        Each feature gets a sequential, human-readable name built from
        *sub_types* cycled over the features:
            "Building 1", "Building 2", …
            "Overhead Tank 1", "Transformer 2", "Pump House 3", …
    """
    try:
        import rasterio.features
        import fiona
        from shapely.geometry import shape, mapping
    except ImportError:
        st.error("Install shapely + fiona + rasterio for SHP export.")
        return None

    binary = (mask > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    # ── Vectorise raster mask ──────────────────────────────────────────────
    if geo_meta and geo_meta.get("transform") is not None:
        shapes_iter = rasterio.features.shapes(binary, transform=geo_meta["transform"])
    else:
        shapes_iter = rasterio.features.shapes(binary)

    raw_shapes = [(shape(geom), int(val)) for geom, val in shapes_iter if int(val) == 1]
    raw_shapes = [(g, v) for g, v in raw_shapes if g.is_valid and not g.is_empty]
    if not raw_shapes:
        return None

    # ── Convert geometries to the correct target type ──────────────────────
    converted = []
    for g, v in raw_shapes:
        cg = _convert_geometry(g, geom_type)
        if cg is not None and not cg.is_empty:
            # Flatten Multi* into individual geometries
            gname = type(cg).__name__
            if gname.startswith("Multi"):
                for sub in cg.geoms:
                    if not sub.is_empty:
                        converted.append(sub)
            else:
                converted.append(cg)
    if not converted:
        return None

    # ── Choose the correct Fiona geometry type string ──────────────────────
    fiona_geom_type = geom_type  # "Polygon", "LineString", or "Point"

    # ── Build attribute schema ─────────────────────────────────────────────
    #   name   : "Building 1", "Overhead Tank 2", …
    #   layer  : layer-level label ("Buildings", "Utility Areas", …)
    #   type   : sub-type from sub_types list
    #   measure: area for polygons, length for lines, 0 for points
    schema = {
        "geometry": fiona_geom_type,
        "properties": {
            "id": "int",
            "name": "str",
            "layer": "str",
            "type": "str",
            "measure": "float",
        },
    }

    # Add roof_type column for building features
    if roof_cls_map is not None:
        schema["properties"]["roof_type"] = "str"

    if sub_types is None or len(sub_types) == 0:
        sub_types = [attr_prefix]

    # ── CRS ────────────────────────────────────────────────────────────────
    if geo_meta and geo_meta.get("crs"):
        try:
            crs_dict = dict(geo_meta["crs"])
        except Exception:
            crs_dict = {"init": "epsg:4326"}
    else:
        crs_dict = {"init": "epsg:4326"}

    # ── Write shapefile ────────────────────────────────────────────────────
    # Track per-subtype counters so names read "Overhead Tank 1", "Transformer 1"
    subtype_counters: dict[str, int] = {st_name: 0 for st_name in sub_types}

    with tempfile.TemporaryDirectory() as tmp:
        shp_path = os.path.join(tmp, f"{feature_name}.shp")
        with fiona.open(
            shp_path, "w", driver="ESRI Shapefile", crs=crs_dict, schema=schema
        ) as dst:
            for i, geom in enumerate(converted):
                try:
                    # Cycle through sub-types
                    st_name = sub_types[i % len(sub_types)]
                    subtype_counters[st_name] += 1
                    feat_label = f"{st_name} {subtype_counters[st_name]}"

                    # Measure: area for polygons, length for lines, 0 for points
                    if geom_type == "Polygon":
                        measure = float(geom.area)
                    elif geom_type == "LineString":
                        measure = float(geom.length)
                    else:
                        measure = 0.0

                    props = {
                        "id": i + 1,
                        "name": feat_label,
                        "layer": attr_prefix,
                        "type": st_name,
                        "measure": measure,
                    }

                    # Add roof classification per building polygon
                    if roof_cls_map is not None:
                        try:
                            centroid = geom.centroid
                            rh, rw = roof_cls_map.shape
                            # Convert geo-coordinates → pixel row/col
                            if geo_meta and geo_meta.get("transform") is not None:
                                inv_tf = ~geo_meta["transform"]
                                px, py = inv_tf * (centroid.x, centroid.y)
                                cx, cy = int(px), int(py)
                            else:
                                cx, cy = int(centroid.x), int(centroid.y)
                            if 0 <= cy < rh and 0 <= cx < rw:
                                props["roof_type"] = ROOF_TYPES[int(roof_cls_map[cy, cx])]
                            else:
                                props["roof_type"] = "Unknown"
                        except Exception:
                            props["roof_type"] = "Unknown"

                    dst.write(
                        {
                            "geometry": mapping(geom),
                            "properties": props,
                        }
                    )
                except Exception:
                    continue

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
                fp = os.path.join(tmp, f"{feature_name}{ext}")
                if os.path.exists(fp):
                    zf.write(fp, f"{feature_name}{ext}")
        buf.seek(0)
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
def overlay(image_np, mask, color, alpha=0.45, threshold=0.5):
    out = image_np.copy().astype(np.float32)
    bin_m = (mask > threshold).astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(
            bin_m,
            np.clip(image_np[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255),
            image_np[:, :, c],
        )
    return out.astype(np.uint8)


def combined_overlay(image_np, preds, selected, threshold=0.5):
    out = image_np.copy().astype(np.float32)
    for k in selected:
        if k not in preds:
            continue
        color = FEATURE_META[k][1]
        bm = (preds[k] > threshold).astype(np.float32)
        for c in range(3):
            out[:, :, c] += bm * color[c] * 0.4
    return np.clip(out, 0, 255).astype(np.uint8)


def to_png_bytes(arr: np.ndarray) -> bytes:
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr.astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🛰️ SVAMITVA Feature Extraction")
    st.markdown(
        "Upload drone/aerial imagery (**JPG · PNG · TIF**) to extract "
        "10 geospatial feature layers and export as **Shapefiles** with original CRS."
    )

    # ── Sidebar — checkpoint ────────────────────────────────────────────────
    st.sidebar.header("🗂️ Checkpoint")
    auto_ckpt = _find_latest_checkpoint()

    use_uploaded = st.sidebar.checkbox("Upload my own checkpoint", value=False)
    if use_uploaded:
        ckpt_file = st.sidebar.file_uploader("Upload .pt file", type=["pt", "pth"])
        if ckpt_file:
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            tmp.write(ckpt_file.read())
            tmp.flush()
            ckpt_path_str = tmp.name
        else:
            ckpt_path_str = ""
    else:
        ckpt_path_str = str(auto_ckpt) if auto_ckpt else ""

    if ckpt_path_str:
        st.sidebar.caption(f"Using: `{Path(ckpt_path_str).name}`")
    else:
        st.sidebar.warning(
            "No checkpoint found — model is untrained.\nResults will not be meaningful."
        )

    model, ckpt_name, loaded = load_model(ckpt_path_str)

    if loaded:
        st.sidebar.success(f"✅ Loaded: `{ckpt_name}`")
    else:
        st.sidebar.error("⚠️ Untrained model — upload or train a checkpoint first.")

    # ── Sidebar — features ──────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Feature Layers")
    selected = []
    for key, meta in FEATURE_META.items():
        label = meta[0]
        if st.sidebar.checkbox(label, value=True, key=f"chk_{key}"):
            selected.append(key)

    if not selected:
        st.sidebar.error("Select at least one layer.")

    # ── Sidebar — settings ──────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.info(
        f"**Device:** {str(DEVICE).upper()}\n\n"
        f"**Tile size:** {TILE_SIZE}px\n\n"
        f"**Overlap:** {OVERLAP}px"
    )

    # ── File upload ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Image (JPG · JPEG · PNG · TIF · TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="GeoTIFFs retain CRS in the exported shapefiles.",
    )

    if uploaded is None:
        st.markdown(
            """
<div class="feature-card">
<h3>Getting Started</h3>
<p>Upload an aerial or drone image to begin feature extraction.<br>
Supported: <strong>JPG · JPEG · PNG · TIF · TIFF</strong> (including large GeoTIFFs).</p>
<p>The model detects 10 layers:</p>
<ul>
  <li>🏠 Buildings + roof type classification</li>
  <li>🛣️ Roads and road centrelines</li>
  <li>💧 Waterbodies, water lines, and water points</li>
  <li>⚡ Utility lines and utility areas</li>
  <li>🌉 Bridges</li>
  <li>🚂 Railways</li>
</ul>
<p><strong>Export:</strong> Each detected layer exports as a Shapefile (<code>.shp</code>) with the same CRS as input + PNG masks.</p>
</div>""",
            unsafe_allow_html=True,
        )
        return

    if not selected:
        st.error("Please select at least one layer in the sidebar.")
        return

    # ── Load image ──────────────────────────────────────────────────────────
    image_np, geo_meta = load_image_with_meta(uploaded)
    if image_np is None:
        return

    h, w = image_np.shape[:2]
    crs_str = (
        str(geo_meta["crs"])
        if geo_meta and geo_meta.get("crs")
        else "None (pixel coords)"
    )
    st.success(f"📐 {w}×{h}px · {uploaded.name} · CRS: `{crs_str}`")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Input Image", use_container_width=True)

    # ── Inference ───────────────────────────────────────────────────────────
    with st.spinner("🔄 Extracting features…"):
        preds = run_inference(image_np, model, selected)

    # ── Split detected / not-detected ───────────────────────────────────────
    detected, not_detected = [], []
    for k in selected:
        mask = preds.get(k)
        if mask is not None and (mask > threshold).sum() > 0:
            detected.append(k)
        else:
            not_detected.append(k)

    # ── Combined overlay ────────────────────────────────────────────────────
    with col2:
        comb = combined_overlay(image_np, preds, detected, threshold)
        st.image(comb, caption="Combined Detected Features", use_container_width=True)

    # ── Not-detected banners ────────────────────────────────────────────────
    if not_detected:
        st.markdown("---")
        st.subheader("⚠️ Layers Not Found")
        for k in not_detected:
            label = FEATURE_META[k][0]
            st.markdown(
                f'<div class="not-detected">⚠️ <strong>{label}</strong> — '
                f"Layer not found in this image</div>",
                unsafe_allow_html=True,
            )

    # ── Statistics ──────────────────────────────────────────────────────────
    if detected:
        st.markdown("---")
        st.subheader("📊 Detection Statistics")
        total_px = h * w
        cols = st.columns(min(len(detected), 5))
        for i, k in enumerate(detected):
            n_px = int((preds[k] > threshold).sum())
            pct = n_px / total_px * 100
            with cols[i % 5]:
                st.metric(FEATURE_META[k][0], f"{pct:.2f}%", delta=f"{n_px:,} px")

    # ── Per-feature tabs ─────────────────────────────────────────────────────
    all_dl_files = {}  # filename → bytes (for master ZIP)
    shp_zips = {}  # feature_name → bytes

    if detected:
        st.markdown("---")
        st.subheader("🔍 Per-Layer Results")
        tabs = st.tabs([FEATURE_META[k][0] for k in detected])

        for tab, key in zip(tabs, detected):
            label, color, geom_type, attr_prefix, sub_types = FEATURE_META[key]
            mask = preds[key]
            ov_img = overlay(image_np, mask, color, threshold=threshold)
            bin_vis = ((mask > threshold) * 255).astype(np.uint8)

            with tab:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        ov_img, caption=f"{label} — Overlay", use_container_width=True
                    )
                with c2:
                    st.image(
                        bin_vis,
                        caption=f"{label} — Binary Mask",
                        use_container_width=True,
                    )

                # Generate files
                mask_png = to_png_bytes(bin_vis)
                overlay_png = to_png_bytes(ov_img)
                fname = key.replace("_mask", "")

                all_dl_files[f"{fname}_mask.png"] = mask_png
                all_dl_files[f"{fname}_overlay.png"] = overlay_png

                # Shapefile — correct geometry type + named attributes
                # For buildings, pass per-pixel roof classification
                roof_cls = None
                if key == "building_mask" and "roof_type" in preds:
                    roof_cls = np.argmax(preds["roof_type"], axis=0).astype(np.uint8)

                shp_bytes = mask_to_shp_zip(
                    mask, threshold, fname, geo_meta,
                    geom_type=geom_type,
                    attr_prefix=attr_prefix,
                    sub_types=sub_types,
                    roof_cls_map=roof_cls,
                )
                if shp_bytes:
                    shp_zips[fname] = shp_bytes
                    all_dl_files[f"shapefiles/{fname}.zip"] = shp_bytes

                # Download buttons
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.download_button(
                        "⬇️ Mask (PNG)",
                        mask_png,
                        f"{fname}_mask.png",
                        "image/png",
                        key=f"dl_mask_{key}",
                    )
                with d2:
                    st.download_button(
                        "⬇️ Overlay (PNG)",
                        overlay_png,
                        f"{fname}_overlay.png",
                        "image/png",
                        key=f"dl_ov_{key}",
                    )
                with d3:
                    if shp_bytes:
                        st.download_button(
                            "⬇️ Shapefile (.shp)",
                            shp_bytes,
                            f"{fname}_shapefile.zip",
                            "application/zip",
                            key=f"dl_shp_{key}",
                        )
                    else:
                        st.info("No geometries above threshold to export.")

    # ── Roof Type Distribution ───────────────────────────────────────────────
    if "roof_type" in preds and "building_mask" in detected:
        st.markdown("---")
        st.subheader("🏠 Roof Type Distribution (within detected buildings)")
        roof_probs = preds["roof_type"]
        bldg_mask = preds["building_mask"] > threshold
        if bldg_mask.any():
            roof_cls = np.argmax(roof_probs, axis=0)
            in_bldg = roof_cls[bldg_mask]
            total_b = len(in_bldg)
            rc = st.columns(len(ROOF_TYPES))
            for i, (col, name) in enumerate(zip(rc, ROOF_TYPES)):
                cnt = int((in_bldg == i).sum())
                with col:
                    st.metric(name, f"{cnt/total_b*100:.1f}%", delta=f"{cnt:,}")

    # ── Download All ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📦 Download All Results")

    comb_png = to_png_bytes(comb if detected else image_np)
    all_dl_files["combined_overlay.png"] = comb_png

    master_zip = io.BytesIO()
    with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in all_dl_files.items():
            zf.writestr(fname, data)
    master_zip.seek(0)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ All Results (ZIP)",
            master_zip.getvalue(),
            "svamitva_results.zip",
            "application/zip",
            key="dl_all",
        )
    if shp_zips:
        shp_only = io.BytesIO()
        with zipfile.ZipFile(shp_only, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in shp_zips.items():
                zf.writestr(f"{fname}.zip", data)
        shp_only.seek(0)
        with dl2:
            st.download_button(
                "⬇️ Shapefiles Only (ZIP)",
                shp_only.getvalue(),
                "svamitva_shapefiles.zip",
                "application/zip",
                key="dl_shp_only",
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"**Checkpoint:** `{ckpt_name}` · "
        f"**Detected:** {len(detected)}/10 layers · "
        f"**Device:** {str(DEVICE).upper()}"
    )


if __name__ == "__main__":
    main()
