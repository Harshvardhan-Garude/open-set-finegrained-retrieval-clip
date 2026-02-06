"""Merge multiple vehicle datasets into a single ImageFolder-style directory.

This script was used to create a unified dataset with folder-per-class format:
    merged_dataset/<class_name>/*.jpg

Supported sources (optional):
  - Vehicle-10: expects class folders directly under --vehicle10_root
  - Stanford Cars: expects names.csv + anno_train.csv + anno_test.csv and images under car_data/train and car_data/test
  - OpenImages vehicle subset: expects class folders under --openimages_root (e.g., Car/ Truck/ ...)

IMPORTANT:
  - This script copies images into the output directory.
  - It does not download datasets.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge Vehicle-10 + Stanford Cars + OpenImages into one folder tree.")
    p.add_argument("--vehicle10_root", type=str, default=None, help="Path to Vehicle-10 root (folders per class).")
    p.add_argument("--stanford_root", type=str, default=None, help="Path to Stanford Cars root.")
    p.add_argument("--openimages_root", type=str, default=None, help="Path to OpenImages vehicle subset root.")
    p.add_argument("--out_root", type=str, default="data/merged_dataset", help="Output merged dataset directory.")
    return p.parse_args()


def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_images(src_dir: Path, dst_dir: Path, rename_prefix: str = "") -> int:
    """Copy all image files from src_dir to dst_dir. Returns count."""
    safe_makedirs(dst_dir)
    n = 0
    for fname in os.listdir(src_dir):
        src_path = src_dir / fname
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in IMAGE_EXTS:
            continue
        dst_name = f"{rename_prefix}{fname}" if rename_prefix else fname
        shutil.copyfile(src_path, dst_dir / dst_name)
        n += 1
    return n


def merge_vehicle10(vehicle10_root: Path, out_root: Path) -> None:
    classes = sorted([d for d in os.listdir(vehicle10_root) if (vehicle10_root / d).is_dir()])
    for cls in classes:
        src = vehicle10_root / cls
        dst = out_root / cls
        print(f"[merge] Vehicle-10: {cls}")
        copy_images(src, dst)


def merge_stanford(stanford_root: Path, out_root: Path) -> None:
    names_csv = stanford_root / "names.csv"
    anno_train = stanford_root / "anno_train.csv"
    anno_test = stanford_root / "anno_test.csv"
    img_train = stanford_root / "car_data" / "train"
    img_test = stanford_root / "car_data" / "test"

    name_df = pd.read_csv(names_csv, header=None)
    name_map = {}
    for _, row in name_df.iterrows():
        cls_id = str(row[0])
        cls_name = str(row[1])
        safe_name = cls_name.replace("/", "-").replace(" ", "_")
        name_map[cls_id] = safe_name

    for split_name, anno_file, img_dir in [
        ("TRAIN", anno_train, img_train),
        ("TEST", anno_test, img_test),
    ]:
        df = pd.read_csv(anno_file)
        print(f"[merge] Stanford Cars {split_name}: {len(df)} rows")

        for _, row in df.iterrows():
            img_fname = row["image"]
            cls_id = str(row["class_id"])
            if cls_id not in name_map:
                continue

            cls_folder = f"Stanford_{name_map[cls_id]}"
            dst_dir = out_root / cls_folder
            safe_makedirs(dst_dir)

            src_path = img_dir / img_fname
            if not src_path.is_file():
                continue

            dst_fname = f"STAN_{split_name}_{img_fname}"
            shutil.copyfile(src_path, dst_dir / dst_fname)


def merge_openimages(openimages_root: Path, out_root: Path) -> None:
    # You can extend this list based on your subset
    oi_classes = ["Car", "Boat", "Bicycle", "Truck", "Motorcycle"]
    for cls in oi_classes:
        src = openimages_root / cls
        if not src.is_dir():
            print(f"[merge] OpenImages: missing {src} (skipping)")
            continue
        dst = out_root / cls
        print(f"[merge] OpenImages: {cls}")
        copy_images(src, dst, rename_prefix=f"OI_{cls}_")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    safe_makedirs(out_root)

    if args.vehicle10_root:
        merge_vehicle10(Path(args.vehicle10_root), out_root)

    if args.stanford_root:
        merge_stanford(Path(args.stanford_root), out_root)

    if args.openimages_root:
        merge_openimages(Path(args.openimages_root), out_root)

    # Print a quick summary
    print("\n[merge] DONE. Output:", out_root)
    for sub in sorted(os.listdir(out_root)):
        cls_dir = out_root / sub
        if cls_dir.is_dir():
            print(f"  - {sub}: {len(os.listdir(cls_dir))} images")


if __name__ == "__main__":
    main()
