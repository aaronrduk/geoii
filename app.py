"""
SVAMITVA Feature Extraction — Production Streamlit App
Supports JPG, PNG, JPEG, and GeoTIFF uploads with feature selection and SHP export.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import zipfile
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models.feature_extractor import FeatureExtractorModel

# ─────────────────────────────────────────────────
# Page Config & Theme
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SVAMITVA Feature Extraction",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #e0e0ff !important; }
    .stMetric label { color: #a0a0d0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; }
    .feature-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    .not-detected {
        background: rgba(255,80,80,0.15);
        border: 1px solid rgba(255,80,80,0.3);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #ff9999;
        font-size: 1.1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE_SIZE = 512
OVERLAP = 64

FEATURE_COLORS = {
    "building_mask": (255, 60, 60),  # Red
    "road_mask": (255, 220, 50),  # Yellow
    "waterbody_mask": (50, 120, 255),  # Blue
    "utility_mask": (50, 220, 100),  # Green
}

FEATURE_LABELS = {
    "building_mask": "🏠 Buildings",
    "road_mask": "🛣️ Roads",
    "waterbody_mask": "💧 Waterbodies",
    "utility_mask": "⚡ Utilities",
}

FEATURE_KEYS = list(FEATURE_LABELS.keys())

ROOF_TYPES = ["RCC", "Tiled", "Tin", "Others", "Unknown"]


@st.cache_resource
def load_model():
    model = FeatureExtractorModel(backbone="resnet50", pretrained=False)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        st.sidebar.success("✅ Model loaded from checkpoint")
    else:
        st.sidebar.warning("⚠️ No checkpoint — using untrained model")
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────────
# Image Handling
# ─────────────────────────────────────────────────
def load_image(uploaded_file):
    """Load uploaded image as numpy array (H, W, 3) uint8."""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg", "png"):
        img = Image.open(uploaded_file).convert("RGB")
        return np.array(img)
    elif ext in ("tif", "tiff"):
        try:
            import rasterio

            with rasterio.open(uploaded_file) as src:
                bands = src.read()
                if bands.shape[0] >= 3:
                    img = np.stack([bands[0], bands[1], bands[2]], axis=-1)
                else:
                    img = np.stack([bands[0]] * 3, axis=-1)
                # Handle 16-bit or float images
                if img.dtype != np.uint8:
                    img_min, img_max = img.min(), img.max()
                    if img_max > 255:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(
                            np.uint8
                        )
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                return img
        except ImportError:
            st.error("Install `rasterio` for GeoTIFF support: `pip install rasterio`")
            return None
    return None


def preprocess_tile(tile_np):
    """Convert (H,W,3) uint8 tile to model input tensor."""
    tile = tile_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tile = (tile - mean) / std
    tile = np.transpose(tile, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(tile).unsqueeze(0).float()


def tile_and_predict(image_np, model, selected_features):
    """Tile image, run inference, stitch results for selected features only."""
    h, w = image_np.shape[:2]

    # Small images: run directly
    if h <= TILE_SIZE and w <= TILE_SIZE:
        padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        padded[:h, :w] = image_np
        inp = preprocess_tile(padded).to(DEVICE)
        with torch.no_grad():
            out = model(inp)
        results = {}
        for k, v in out.items():
            mask = v.cpu().numpy()[0]
            if k == "roof_type":
                if "building_mask" in selected_features:
                    results[k] = mask[:, :h, :w]
            elif k in selected_features:
                results[k] = mask[0, :h, :w]
        return results

    # Large images: tile
    stride = TILE_SIZE - OVERLAP
    accum = {k: np.zeros((h, w), dtype=np.float32) for k in selected_features}
    counts = np.zeros((h, w), dtype=np.float32)
    roof_accum = None
    if "building_mask" in selected_features:
        roof_accum = np.zeros((5, h, w), dtype=np.float32)

    tiles_done = 0
    total = ((h - 1) // stride + 1) * ((w - 1) // stride + 1)
    prog = st.progress(0, text="Running inference on tiles...")

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + TILE_SIZE, h)
            x1 = min(x0 + TILE_SIZE, w)
            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            tile[: y1 - y0, : x1 - x0] = image_np[y0:y1, x0:x1]

            inp = preprocess_tile(tile).to(DEVICE)
            with torch.no_grad():
                out = model(inp)

            th, tw = y1 - y0, x1 - x0
            for k in selected_features:
                if k in out:
                    pred = torch.sigmoid(out[k]).cpu().numpy()[0, 0]
                    accum[k][y0:y1, x0:x1] += pred[:th, :tw]

            if roof_accum is not None and "roof_type" in out:
                roof_pred = out["roof_type"].cpu().numpy()[0]
                roof_accum[:, y0:y1, x0:x1] += roof_pred[:, :th, :tw]

            counts[y0:y1, x0:x1] += 1.0
            tiles_done += 1
            prog.progress(tiles_done / total, text=f"Tile {tiles_done}/{total}")

    prog.empty()

    results = {}
    for k in selected_features:
        results[k] = accum[k] / np.maximum(counts, 1.0)
    if roof_accum is not None:
        results["roof_type"] = roof_accum / np.maximum(counts[None], 1.0)
    return results


# ─────────────────────────────────────────────────
# Mask to Shapefile Conversion
# ─────────────────────────────────────────────────
def mask_to_shapefile_bytes(mask, threshold=0.5, feature_name="feature"):
    """
    Convert a binary mask to a shapefile (as ZIP bytes).
    Uses rasterio.features to vectorize the mask into polygons,
    then writes them as a shapefile using fiona.
    Returns ZIP bytes containing .shp, .dbf, .shx, .prj files.
    """
    try:
        from shapely.geometry import shape, mapping
        import fiona
        from fiona.crs import from_epsg
        import rasterio.features
    except ImportError:
        st.error(
            "Missing dependencies for SHP export. Install: `pip install shapely fiona rasterio`"
        )
        return None

    binary = (mask > threshold).astype(np.uint8)

    # Check if anything was detected
    if binary.sum() == 0:
        return None

    # Extract polygon geometries from the binary mask
    polygons = []
    for geom, value in rasterio.features.shapes(binary, transform=None):
        if value == 1:
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        return None

    # Write shapefile to a temp directory, then ZIP it
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, f"{feature_name}.shp")

        schema = {
            "geometry": "Polygon",
            "properties": {
                "feature": "str",
                "area_px": "float",
                "id": "int",
            },
        }

        # Use pixel coordinate CRS (no georeferencing)
        with fiona.open(shp_path, "w", driver="ESRI Shapefile", schema=schema) as dst:
            for i, poly in enumerate(polygons):
                dst.write(
                    {
                        "geometry": mapping(poly),
                        "properties": {
                            "feature": feature_name,
                            "area_px": poly.area,
                            "id": i + 1,
                        },
                    }
                )

        # Bundle all shapefile components into a ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
                fpath = os.path.join(tmpdir, f"{feature_name}{ext}")
                if os.path.exists(fpath):
                    zf.write(fpath, f"{feature_name}{ext}")

        zip_buffer.seek(0)
        return zip_buffer.getvalue()


# ─────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────
def create_overlay(image_np, mask, color, alpha=0.45):
    """Create colored overlay on image."""
    overlay = image_np.copy()
    binary = (mask > 0.5).astype(np.uint8)
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary,
            np.clip(image_np[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255).astype(
                np.uint8
            ),
            image_np[:, :, c],
        )
    return overlay


def create_combined_overlay(image_np, predictions, selected_features):
    """Create overlay with selected features only."""
    overlay = image_np.copy().astype(np.float32)
    for key, color in FEATURE_COLORS.items():
        if key in selected_features and key in predictions:
            mask = (predictions[key] > 0.5).astype(np.float32)
            for c in range(3):
                overlay[:, :, c] += mask * color[c] * 0.4
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_to_png_bytes(mask):
    """Convert binary mask to downloadable PNG bytes."""
    binary = ((mask > 0.5) * 255).astype(np.uint8)
    img = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_to_png_bytes(overlay_np):
    """Convert overlay to downloadable PNG bytes."""
    img = Image.fromarray(overlay_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────
def main():
    st.title("🛰️ SVAMITVA Feature Extraction")
    st.markdown(
        "Upload aerial/drone imagery to extract buildings, "
        "roads, waterbodies, and utilities. Export as **shapefiles**."
    )

    model = load_model()

    # ── Sidebar ──
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Select Features to Extract")
    selected_features = []
    for key, label in FEATURE_LABELS.items():
        if st.sidebar.checkbox(label, value=True, key=f"sel_{key}"):
            selected_features.append(key)

    if not selected_features:
        st.sidebar.error("Select at least one feature!")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Device:** {DEVICE.upper()}\n\n"
        f"**Tile Size:** {TILE_SIZE}px\n\n"
        f"**Overlap:** {OVERLAP}px"
    )

    # ── File upload ──
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="JPG, JPEG, PNG, TIF, or TIFF aerial imagery",
    )

    if uploaded is None:
        st.markdown(
            """
        <div class="feature-card">
            <h3>Getting Started</h3>
            <p>Upload an aerial or drone image to begin feature extraction.
            Supported formats: <strong>JPG, JPEG, PNG, TIF, TIFF</strong></p>
            <p>The model detects:</p>
            <ul>
                <li>🏠 Buildings with roof type classification</li>
                <li>🛣️ Roads and pathways</li>
                <li>💧 Waterbodies (ponds, rivers, tanks)</li>
                <li>⚡ Utilities (transformers, wells)</li>
            </ul>
            <p><strong>Export:</strong> Download detected features as Shapefiles (.shp) or PNG masks.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    if not selected_features:
        st.error("⚠️ Please select at least one feature from the sidebar to extract.")
        return

    # ── Load and display image ──
    image_np = load_image(uploaded)
    if image_np is None:
        st.error(
            "❌ Could not load image. Please upload a valid JPG, PNG, or TIF file."
        )
        return

    h, w = image_np.shape[:2]
    st.success(f"📐 Image loaded: {w}×{h} pixels ({uploaded.name})")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original Image", use_container_width=True)

    # ── Run inference ──
    with st.spinner("🔄 Running feature extraction..."):
        predictions = tile_and_predict(image_np, model, selected_features)

    # Apply threshold and check detections
    binary_preds = {}
    detected_features = []
    not_detected_features = []

    for key in selected_features:
        if key in predictions:
            binary_mask = predictions[key] > threshold
            pixel_count = int(binary_mask.sum())
            binary_preds[key] = binary_mask
            if pixel_count > 0:
                detected_features.append(key)
            else:
                not_detected_features.append(key)
        else:
            not_detected_features.append(key)

    # ── Combined overlay ──
    with col2:
        combined = create_combined_overlay(image_np, predictions, selected_features)
        st.image(combined, caption="Combined Features", use_container_width=True)

    # ── Not Detected Warnings ──
    if not_detected_features:
        st.markdown("---")
        for key in not_detected_features:
            label = FEATURE_LABELS.get(key, key)
            st.markdown(
                f'<div class="not-detected">⚠️ <strong>{label}</strong> — '
                f"No features detected in this image</div>",
                unsafe_allow_html=True,
            )

    # ── Statistics ──
    if detected_features:
        st.markdown("---")
        st.subheader("📊 Detection Statistics")

        total_pixels = h * w
        stat_cols = st.columns(len(detected_features))
        for i, key in enumerate(detected_features):
            label = FEATURE_LABELS[key]
            count = int(binary_preds[key].sum())
            pct = count / total_pixels * 100
            with stat_cols[i]:
                st.metric(label, f"{pct:.1f}%", delta=f"{count:,} pixels")

    # ── Per-feature tabs ──
    if detected_features:
        st.markdown("---")
        st.subheader("🔍 Per-Feature Analysis")

        tab_labels = [FEATURE_LABELS[k] for k in detected_features]
        tabs = st.tabs(tab_labels)
        download_files = {}
        shp_files = {}

        for tab, key in zip(tabs, detected_features):
            with tab:
                label = FEATURE_LABELS[key]
                c1, c2 = st.columns(2)
                mask = predictions[key]
                color = FEATURE_COLORS[key]
                overlay = create_overlay(image_np, mask, color)

                with c1:
                    st.image(
                        overlay, caption=f"{label} Overlay", use_container_width=True
                    )
                with c2:
                    binary_vis = ((mask > threshold) * 255).astype(np.uint8)
                    st.image(
                        binary_vis, caption=f"{label} Mask", use_container_width=True
                    )

                # Create downloads
                mask_bytes = mask_to_png_bytes(mask)
                overlay_bytes = overlay_to_png_bytes(overlay)
                download_files[f"{key}_mask.png"] = mask_bytes
                download_files[f"{key}_overlay.png"] = overlay_bytes

                # SHP export
                feature_name = key.replace("_mask", "")
                shp_bytes = mask_to_shapefile_bytes(mask, threshold, feature_name)

                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.download_button(
                        f"⬇️ Mask (PNG)",
                        mask_bytes,
                        f"{key}_mask.png",
                        "image/png",
                        key=f"dl_mask_{key}",
                    )
                with dc2:
                    st.download_button(
                        f"⬇️ Overlay (PNG)",
                        overlay_bytes,
                        f"{key}_overlay.png",
                        "image/png",
                        key=f"dl_overlay_{key}",
                    )
                with dc3:
                    if shp_bytes:
                        shp_files[f"{feature_name}.zip"] = shp_bytes
                        st.download_button(
                            f"⬇️ Shapefile (SHP)",
                            shp_bytes,
                            f"{feature_name}_shapefile.zip",
                            "application/zip",
                            key=f"dl_shp_{key}",
                        )
                    else:
                        st.info("No polygons to export")

    # ── Roof Type Distribution ──
    if "roof_type" in predictions and "building_mask" in detected_features:
        st.markdown("---")
        st.subheader("🏠 Roof Type Distribution")

        roof_probs = predictions["roof_type"]
        building_mask = predictions["building_mask"] > threshold
        if building_mask.any():
            roof_classes = np.argmax(roof_probs, axis=0)
            roof_in_buildings = roof_classes[building_mask]
            n = len(roof_in_buildings)
            dist = {}
            for i, name in enumerate(ROOF_TYPES[: roof_probs.shape[0]]):
                count = int((roof_in_buildings == i).sum())
                dist[name] = count

            rc = st.columns(len(dist))
            for col, (name, count) in zip(rc, dist.items()):
                with col:
                    st.metric(name, f"{count / n * 100:.1f}%", delta=f"{count:,}")

    # ── Download All Results ──
    st.markdown("---")
    st.subheader("📦 Download All Results")

    all_download_files = {}

    # Add PNG masks and overlays
    if detected_features:
        all_download_files.update(download_files)

    # Add combined overlay
    combined_bytes = overlay_to_png_bytes(combined)
    all_download_files["combined_overlay.png"] = combined_bytes

    # Add all shapefiles
    if shp_files:
        for fname, data in shp_files.items():
            all_download_files[f"shapefiles/{fname}"] = data

    # Create master ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in all_download_files.items():
            zf.writestr(fname, data)
    zip_buffer.seek(0)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download All Results (ZIP)",
            zip_buffer.getvalue(),
            "svamitva_results.zip",
            "application/zip",
            key="dl_all",
        )

    # SHP-only download
    if shp_files:
        with col_dl2:
            shp_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(shp_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in shp_files.items():
                    zf.writestr(fname, data)
            shp_zip_buffer.seek(0)
            st.download_button(
                "⬇️ Download Shapefiles Only (ZIP)",
                shp_zip_buffer.getvalue(),
                "svamitva_shapefiles.zip",
                "application/zip",
                key="dl_shp_all",
            )


if __name__ == "__main__":
    main()
