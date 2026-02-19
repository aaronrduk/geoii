#!/usr/bin/env python3
"""
Organize raw DATASET folder into the training directory structure.

Usage:
    python prepare_dataset.py --dataset_dir /path/to/DATASET --output_dir dataset/train_full
"""

import argparse
import os
import shutil
import glob
import re
from pathlib import Path


# Mapping from shapefile name patterns to feature types
FEATURE_PATTERNS = {
    "buildings": [
        r"(?i)build",
        r"(?i)built.?up",
        r"(?i)structure",
        r"(?i)bua",
        r"(?i)house",
    ],
    "roads": [
        r"(?i)road",
        r"(?i)street",
        r"(?i)path",
        r"(?i)highway",
        r"(?i)lane",
    ],
    "waterbodies": [
        r"(?i)water",
        r"(?i)pond",
        r"(?i)river",
        r"(?i)lake",
        r"(?i)tank",
        r"(?i)drain",
    ],
    "utilities": [
        r"(?i)util",
        r"(?i)transform",
        r"(?i)tank",
        r"(?i)well",
        r"(?i)electric",
        r"(?i)power",
    ],
}


def classify_shapefile(filename: str) -> str | None:
    """Classify a shapefile into a feature type based on its name."""
    base = Path(filename).stem
    for feature_type, patterns in FEATURE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, base):
                return feature_type
    return None


def find_orthophotos(dataset_dir: str) -> list[dict]:
    """Find all orthophoto .tif files and their associated shapefiles."""
    tif_files = glob.glob(os.path.join(dataset_dir, "**", "*.tif"), recursive=True)
    # Filter to large files (likely orthophotos, > 100MB)
    ortho_files = []
    for f in tif_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        if size_mb > 50:
            ortho_files.append(f)

    if not ortho_files:
        # Fallback: just take all .tif files
        ortho_files = tif_files

    # Group by parent directory
    groups = {}
    for f in ortho_files:
        parent = os.path.dirname(f)
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(f)

    # Find associated shapefiles
    result = []
    for parent, tifs in groups.items():
        # Look for shapefiles in same dir or subdirs
        shp_files = glob.glob(os.path.join(parent, "**", "*.shp"), recursive=True)
        for tif in tifs:
            entry = {
                "orthophoto": tif,
                "shapefiles": shp_files,
                "name": Path(tif).stem,
            }
            result.append(entry)

    return result


def create_village_name(entry: dict, idx: int) -> str:
    """Create a clean village name from the orthophoto path."""
    name = entry["name"]
    # Clean up name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = f"village_{idx:02d}"

    # Determine state prefix from path
    path_str = entry["orthophoto"].lower()
    if "cg" in path_str or "chhatisgarh" in path_str:
        prefix = "CG"
    elif "pb" in path_str or "punjab" in path_str:
        prefix = "PB"
    else:
        prefix = "XX"

    return f"{prefix}_{name}"


def prepare_dataset(dataset_dir: str, output_dir: str, use_symlinks: bool = True):
    """Organize the dataset into the training directory structure."""
    print(f"📂 Scanning dataset: {dataset_dir}")

    entries = find_orthophotos(dataset_dir)
    if not entries:
        print("❌ No orthophotos found!")
        return

    print(f"📊 Found {len(entries)} orthophoto(s)")

    os.makedirs(output_dir, exist_ok=True)

    for idx, entry in enumerate(entries):
        village_name = create_village_name(entry, idx)
        village_dir = os.path.join(output_dir, village_name)
        anno_dir = os.path.join(village_dir, "annotations")
        os.makedirs(anno_dir, exist_ok=True)

        # Link/copy orthophoto
        ortho_dst = os.path.join(village_dir, "orthophoto.tif")
        if not os.path.exists(ortho_dst):
            src = os.path.abspath(entry["orthophoto"])
            if use_symlinks:
                os.symlink(src, ortho_dst)
                print(f"  🔗 Linked orthophoto: {village_name}")
            else:
                shutil.copy2(src, ortho_dst)
                print(f"  📋 Copied orthophoto: {village_name}")

        # Classify and copy shapefiles
        features_found = set()
        for shp in entry["shapefiles"]:
            feature_type = classify_shapefile(shp)
            if feature_type and feature_type not in features_found:
                features_found.add(feature_type)
                # Copy all shapefile components (.shp, .dbf, .shx, .prj)
                shp_stem = Path(shp).stem
                shp_dir = os.path.dirname(shp)
                for ext in [
                    ".shp",
                    ".dbf",
                    ".shx",
                    ".prj",
                    ".sbn",
                    ".sbx",
                    ".cpg",
                    ".qpj",
                ]:
                    src_file = os.path.join(shp_dir, shp_stem + ext)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(anno_dir, feature_type + ext)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)

        print(f"  ✅ {village_name}: " f"annotations={list(features_found)}")

    # Summary
    villages = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    print(f"\n🏁 Dataset prepared: {len(villages)} villages " f"in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SVAMITVA training dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to raw DATASET folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/train_full",
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    args = parser.parse_args()

    prepare_dataset(
        args.dataset_dir,
        args.output_dir,
        use_symlinks=not args.copy,
    )
