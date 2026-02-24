"""
Convert satellite flood data to Palette format.

Our structure:
    data/images/{idx:03d}_{lon}_{lat}/{year}_{month:02d}_{day:02d}.png

Output:
    datasets/flood/images/cond/00001.png   ← condition (March 21), RGB
    datasets/flood/images/gt/00001.png     ← ground truth target, RGB
    datasets/flood/flist/train.flist
    datasets/flood/flist/test.flist

Split is by LOCATION (no data leakage).

Usage:
    python prepare_data.py
    python prepare_data.py --src data/images --dst datasets/flood --test_ratio 0.1
    python prepare_data.py --src dataset/images --dst datasets/flood
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


INPUT_SUFFIX = "03_21"
TARGET_SUFFIXES = ["04_07", "04_21", "05_07", "05_21", "06_07", "07_07"]


def collect_pairs(src_path, skip_cloudy=True):
    pairs = []
    for folder in sorted(os.listdir(src_path)):
        folder_path = os.path.join(src_path, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split("_")
        if len(parts) < 3:
            continue
        try:
            lon, lat = float(parts[1]), float(parts[2])
        except ValueError:
            continue

        files_in_folder = set(os.listdir(folder_path))
        found_years = set()
        for f in files_in_folder:
            if f.endswith(".png") and not f.endswith("_C.png"):
                found_years.add(int(f.split("_")[0]))

        for year in sorted(found_years):
            input_file = f"{year}_{INPUT_SUFFIX}.png"
            input_path = os.path.join(folder_path, input_file)
            if not os.path.exists(input_path):
                continue
            if skip_cloudy and f"{year}_{INPUT_SUFFIX}_C.png" in files_in_folder:
                continue

            for suffix in TARGET_SUFFIXES:
                target_file = f"{year}_{suffix}.png"
                target_path = os.path.join(folder_path, target_file)
                if not os.path.exists(target_path):
                    continue
                if skip_cloudy and f"{year}_{suffix}_C.png" in files_in_folder:
                    continue

                pairs.append({
                    'cond_path': input_path,
                    'gt_path': target_path,
                    'loc': folder,
                    'year': year,
                    'suffix': suffix,
                })
    return pairs


def save_rgba(src_path, dst_path):
    """Copy RGBA image as-is (preserves cloud mask in alpha channel)."""
    img = Image.open(src_path).convert("RGBA")
    img.save(dst_path)


def main(args):
    print(f"Scanning {args.src} ...")
    pairs = collect_pairs(args.src, skip_cloudy=not args.include_cloudy)
    print(f"Found {len(pairs)} valid pairs")

    if len(pairs) == 0:
        print("No pairs found! Check --src path.")
        return

    # Split by location
    locations = sorted(set(p['loc'] for p in pairs))
    random.seed(args.seed)
    random.shuffle(locations)

    n_test_locs = max(1, int(len(locations) * args.test_ratio))
    test_locs = set(locations[:n_test_locs])
    train_locs = set(locations[n_test_locs:])

    train_pairs = [p for p in pairs if p['loc'] in train_locs]
    test_pairs = [p for p in pairs if p['loc'] in test_locs]

    print(f"Locations: {len(train_locs)} train, {len(test_locs)} test")
    print(f"Samples:   {len(train_pairs)} train, {len(test_pairs)} test")

    # Create directories
    dst = Path(args.dst)
    (dst / "images" / "cond").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "gt").mkdir(parents=True, exist_ok=True)
    (dst / "flist").mkdir(parents=True, exist_ok=True)

    # Process all pairs
    all_pairs = train_pairs + test_pairs
    train_indices = []
    test_indices = []

    for idx, pair in enumerate(tqdm(all_pairs, desc="Converting")):
        file_name = f"{idx:05d}.png"
        save_rgba(pair['cond_path'], dst / "images" / "cond" / file_name)
        save_rgba(pair['gt_path'], dst / "images" / "gt" / file_name)

        if pair['loc'] in train_locs:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    # Write flist files
    with open(dst / "flist" / "train.flist", 'w') as f:
        for idx in train_indices:
            f.write(f"{idx}\n")

    with open(dst / "flist" / "test.flist", 'w') as f:
        for idx in test_indices:
            f.write(f"{idx}\n")

    print(f"\nDone! Output in {dst}/")
    print(f"  images/cond/  — {len(all_pairs)} condition images")
    print(f"  images/gt/    — {len(all_pairs)} ground truth images")
    print(f"  flist/train.flist — {len(train_indices)} indices")
    print(f"  flist/test.flist  — {len(test_indices)} indices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/images')
    parser.add_argument('--dst', type=str, default='datasets/flood')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_cloudy', action='store_true')
    main(parser.parse_args())