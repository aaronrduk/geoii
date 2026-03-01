"""
Pre-clip large MAP orthophotos into small tile sub-maps inside DATA/MAPC/.

Usage:
    python clip_maps.py --data /Users/aaronr/Desktop/DATA --maps MAP1 --tile-size 512 --overlap 96

Output:
    DATA/
      MAP1/            <- original, UNTOUCHED
      MAP2/            <- original, UNTOUCHED
      MAPC/
        MAP1.1/
          Orthophoto.tif
          Built_Up_Area_typ.shp
          Road.shp
          ...
        MAP1.2/
        ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Shapefile name patterns — matched case-insensitively against .shp stems.
# Multiple keywords per layer so we catch all naming variants across MAPs.
# ---------------------------------------------------------------------------
SHAPEFILE_KEYWORDS = {
    "Built_Up_Area_typ": ["built_up", "build_up", "abadi", "building", "structure"],
    "Road": ["^road$", "^road_1$"],
    "Road_Centre_Line": ["centre_line", "center_line", "centreline", "centerline"],
    "Water_Body": ["^waterbody$", "^waterbody_1$", "^water_body$", "water_body_1", "pond", "lake"],
    "Water_Body_Line": ["waterbody_line", "water_body_line"],
    "Waterbody_Point": ["waterbody_point", "water_body_point"],
    "Utility": ["^utility$", "^utility_1$", "utility_line"],
    "Utility_Poly_": ["utility_poly", "utility_area"],
    "Bridge": ["bridge"],
    "Railway": ["railway", "railroad", "rail"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tif(map_dir: Path) -> Path:
    """Return the first .tif/.tiff found in *map_dir*, or None."""
    for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        matches = list(map_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def find_shapefiles(map_dir: Path) -> dict:
    """Return {layer_name: Path} for every shapefile matching a known keyword."""
    import re
    found = {}
    all_shps = sorted(map_dir.glob("*.shp"))
    taken = set()

    for layer_name, keywords in SHAPEFILE_KEYWORDS.items():
        for shp in all_shps:
            if shp in taken:
                continue
            stem = shp.stem.lower()
            for kw in keywords:
                # Support anchored patterns like ^road$
                if kw.startswith("^") and kw.endswith("$"):
                    if re.fullmatch(kw[1:-1], stem):
                        found[layer_name] = shp
                        taken.add(shp)
                        break
                elif kw in stem:
                    found[layer_name] = shp
                    taken.add(shp)
                    break
            if layer_name in found:
                break

    return found


def is_nodata_tile(rgb: np.ndarray, threshold: float = 0.85) -> bool:
    """True when >*threshold* of pixels are near-black or near-white (NoData)."""
    if rgb is None or rgb.size == 0:
        return True
    brightness = rgb.reshape(-1, 3 if rgb.shape[-1] >= 3 else 1).astype(np.float32)
    if brightness.shape[-1] == 3:
        brightness = brightness.mean(axis=1)
    else:
        brightness = brightness.squeeze()
    dark = (brightness < 15).sum()
    bright = (brightness > 240).sum()
    return (dark + bright) / len(brightness) > threshold


# ---------------------------------------------------------------------------
# Core clipper
# ---------------------------------------------------------------------------

def clip_one_map(
    src_tif: Path,
    shapefiles: dict,
    mapc_dir: Path,
    map_name: str,
    tile_size: int = 512,
    overlap: int = 96,
    nodata_threshold: float = 0.85,
) -> int:
    """Clip *src_tif* + its shapefiles into MAPC/<map_name>.<n>/ sub-folders."""

    with rasterio.open(src_tif) as src:
        img_h, img_w = src.height, src.width
        crs = src.crs
        full_transform = src.transform
        band_count = src.count

        print(f"  Source : {img_w} x {img_h} px, {band_count} bands, CRS = {crs}")

        # Load & reproject every shapefile once --------------------------------
        gdf_cache = {}
        for name, shp_path in shapefiles.items():
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.crs and gdf.crs != crs:
                    gdf = gdf.to_crs(crs)
                gdf_cache[name] = gdf
                print(f"    + {name}: {len(gdf)} features")
            except Exception as e:
                print(f"    x {name}: {e}")

        # Tile loop ------------------------------------------------------------
        step = tile_size - overlap
        saved = 0
        skipped_nodata = 0
        total = 0

        for row_off in range(0, img_h, step):
            for col_off in range(0, img_w, step):
                win_h = min(tile_size, img_h - row_off)
                win_w = min(tile_size, img_w - col_off)

                # Skip tiny edge slivers
                if win_h < tile_size // 2 or win_w < tile_size // 2:
                    continue

                total += 1
                window = Window(col_off, row_off, win_w, win_h)

                # --- Read pixels ---
                tile_data = src.read(window=window)  # (C, H, W)

                # --- NoData filter ---
                rgb = np.moveaxis(tile_data[:min(3, band_count)], 0, -1)
                if is_nodata_tile(rgb, threshold=nodata_threshold):
                    skipped_nodata += 1
                    continue

                # --- Geo bounds of this tile ---
                tile_transform = rasterio.windows.transform(window, full_transform)
                left = tile_transform.c
                top = tile_transform.f
                right = left + win_w * tile_transform.a
                bottom = top + win_h * tile_transform.e  # e is negative

                tile_box = box(
                    min(left, right),
                    min(top, bottom),
                    max(left, right),
                    max(top, bottom),
                )

                # --- Clip shapefiles ---
                clipped_gdfs = {}
                for name, gdf in gdf_cache.items():
                    try:
                        clipped = gpd.clip(gdf, tile_box)
                        if len(clipped) > 0 and not clipped.is_empty.all():
                            clipped_gdfs[name] = clipped
                    except Exception:
                        pass

                # --- Create sub-map folder ---
                saved += 1
                sub_name = f"{map_name}.{saved}"
                sub_dir = mapc_dir / sub_name
                sub_dir.mkdir(parents=True, exist_ok=True)

                # Save mini-TIFF
                profile = src.profile.copy()
                profile.update(
                    width=win_w,
                    height=win_h,
                    transform=tile_transform,
                    compress="deflate",
                )
                with rasterio.open(sub_dir / "Orthophoto.tif", "w", **profile) as dst:
                    dst.write(tile_data)

                # Save clipped shapefiles
                for name, clipped_gdf in clipped_gdfs.items():
                    try:
                        clipped_gdf.to_file(sub_dir / f"{name}.shp")
                    except Exception as e:
                        print(f"      [WARN] {sub_name}/{name}.shp - {e}")

                if saved % 200 == 0:
                    print(f"    ... {saved} saved / {skipped_nodata} skipped so far")

    print(f"\n  RESULT: {saved} sub-maps saved, {skipped_nodata} NoData skipped, "
          f"{total} tiles examined")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-clip MAP orthophotos into DATA/MAPC/<MAP>.<n>/ sub-maps"
    )
    parser.add_argument(
        "--data", type=str, default="/Users/aaronr/Desktop/DATA",
        help="Root DATA directory containing MAP1, MAP2, ...",
    )
    parser.add_argument(
        "--maps", nargs="+", default=["MAP1"],
        help="Which MAPs to clip (e.g. --maps MAP1 MAP2 MAP3)",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--nodata-threshold", type=float, default=0.85)
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    # All clipped sub-maps go into DATA/MAPC/
    mapc_dir = data_dir / "MAPC"
    mapc_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {mapc_dir}")

    grand_total = 0

    for map_name in args.maps:
        map_dir = data_dir / map_name
        if not map_dir.exists():
            print(f"\n[SKIP] {map_dir} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Clipping: {map_name}  ->  MAPC/{map_name}.*")
        print(f"{'=' * 60}")

        tif_path = find_tif(map_dir)
        if not tif_path:
            print(f"  [SKIP] No TIF found in {map_dir}")
            continue

        shps = find_shapefiles(map_dir)
        print(f"  Found {len(shps)} shapefiles")

        n = clip_one_map(
            src_tif=tif_path,
            shapefiles=shps,
            mapc_dir=mapc_dir,
            map_name=map_name,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nodata_threshold=args.nodata_threshold,
        )
        grand_total += n

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE - {grand_total} total sub-maps in {mapc_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
