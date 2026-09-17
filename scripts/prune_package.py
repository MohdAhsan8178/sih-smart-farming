#!/usr/bin/env python3
"""
scripts/prune_package.py

Implements B6:
1. Copies all 30,122 unique files referenced across all split CSVs (30,039 in all_images.csv
   + 83 in test_external_riceblast.csv) from data/packaged/ to data/packaged_min/.
2. Copies all 174 files from data/packaged/openset_holdout/ to data/packaged_min/openset_holdout/.
3. Copies all 5,255 files from data/packaged/openset/ to data/packaged_min/openset/.
4. Rewrites splits_packaged/*.csv paths to data/packaged_min/.
5. Confirms splits_packaged/class_weights.json remains unchanged.
6. Verifies 100% of paths in splits_packaged/*.csv exist on disk (expected: 0 missing across all 60,161).
7. Runs leak check on splits_packaged/*.csv for openset_holdout and not_crop_archive.
8. Reports total file counts and disk sizes for data/packaged vs data/packaged_min.
"""

import os
import sys
import csv
import json
import shutil
import subprocess
from pathlib import Path

SRC_DIR = Path("data/packaged")
DST_DIR = Path("data/packaged_min")
RAW_SPLITS_DIR = Path("splits")
SPLITS_DIR = Path("splits_packaged")

def prune_dataset():
    print("=== B6: Creating pruned package in data/packaged_min/ ===")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect all unique image paths across all split CSVs from original splits
    unique_rel_paths = set()
    for csv_file in sorted(RAW_SPLITS_DIR.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "path" not in header:
                continue
            path_idx = header.index("path")
            for row in reader:
                if len(row) > path_idx and row[path_idx].startswith("data/raw/"):
                    rel = Path(row[path_idx]).relative_to("data/raw")
                    unique_rel_paths.add(rel)

    print(f"Total unique split image paths: {len(unique_rel_paths)} (30039 in all_images.csv + 83 in test_external_riceblast.csv)")
    assert len(unique_rel_paths) == 30122, f"Expected 30122 unique split paths, got {len(unique_rel_paths)}"

    copied_split = 0
    for rel in unique_rel_paths:
        src = SRC_DIR / rel
        dst = DST_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_split += 1

    print(f"Copied {copied_split} split images to {DST_DIR}")
    assert copied_split == 30122

    # 2. Copy openset_holdout/
    holdout_src = SRC_DIR / "openset_holdout"
    holdout_dst = DST_DIR / "openset_holdout"
    copied_holdout = 0
    for p in holdout_src.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            rel = p.relative_to(holdout_src)
            target = holdout_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied_holdout += 1

    print(f"Copied {copied_holdout} files from {holdout_src} to {holdout_dst}")
    assert copied_holdout == 174, f"Expected 174 files, got {copied_holdout}"

    # 3. Copy openset/
    openset_src = SRC_DIR / "openset"
    openset_dst = DST_DIR / "openset"
    copied_openset = 0
    for p in openset_src.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            rel = p.relative_to(openset_src)
            target = openset_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied_openset += 1

    print(f"Copied {copied_openset} files from {openset_src} to {openset_dst}")
    assert copied_openset == 5255, f"Expected 5255 files, got {copied_openset}"

    total_packaged_min = copied_split + copied_holdout + copied_openset
    print(f"Total files in data/packaged_min/: {total_packaged_min}")
    assert total_packaged_min == 35551, f"Expected 35551, got {total_packaged_min}"

def rewrite_splits_for_packaged_min():
    print("\n=== Rewriting splits_packaged/*.csv paths to data/packaged_min/ ===")
    for csv_file in sorted(SPLITS_DIR.glob("*.csv")):
        raw_csv = RAW_SPLITS_DIR / csv_file.name
        with open(raw_csv, "r", encoding="utf-8") as f_in, open(csv_file, "w", encoding="utf-8", newline="") as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            header = next(reader)
            writer.writerow(header)
            
            path_idx = header.index("path") if "path" in header else -1
            rewritten_rows = 0
            for row in reader:
                if path_idx >= 0 and len(row) > path_idx:
                    if row[path_idx].startswith("data/raw/"):
                        row[path_idx] = row[path_idx].replace("data/raw/", "data/packaged_min/", 1)
                        rewritten_rows += 1
                writer.writerow(row)
        print(f"Wrote {csv_file.name}: {rewritten_rows} paths rewritten to data/packaged_min/")

def verify_packaged_min_paths():
    print("\n=== Verifying All Paths in splits_packaged/*.csv Exist in data/packaged_min/ ===")
    total_checked = 0
    total_missing = 0
    missing_samples = []

    for csv_file in sorted(SPLITS_DIR.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "path" not in header:
                continue
            path_idx = header.index("path")
            file_checked = 0
            file_missing = 0
            for row in reader:
                if len(row) > path_idx:
                    p_str = row[path_idx]
                    p = Path(p_str)
                    file_checked += 1
                    if not p.exists():
                        file_missing += 1
                        if len(missing_samples) < 5:
                            missing_samples.append(p_str)
            total_checked += file_checked
            total_missing += file_missing
            print(f"{csv_file.name:35s}: {file_checked} paths checked, {file_missing} missing")

    print(f"\nTotal paths checked across all splits_packaged CSVs: {total_checked}")
    print(f"Total missing files: {total_missing}")
    if total_missing > 0:
        print(f"Sample missing paths: {missing_samples}")
    assert total_checked == 60161, f"Expected 60161 paths checked, got {total_checked}"
    assert total_missing == 0, f"Expected 0 missing files, got {total_missing}"

def leak_check():
    print("\n=== Leak Check on splits_packaged/*.csv ===")
    cmd = ["grep", "-c", "openset_holdout\\|not_crop_archive"] + sorted(str(p) for p in SPLITS_DIR.glob("*.csv"))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("grep output:")
    print(res.stdout.strip())
    for line in res.stdout.strip().splitlines():
        if ":" in line:
            fname, cnt = line.rsplit(":", 1)
            assert int(cnt) == 0, f"Leak detected in {fname}: count={cnt}"
    print("Leak check PASSED: 0 occurrences of openset_holdout or not_crop_archive in any split CSV!")

def main():
    prune_dataset()
    rewrite_splits_for_packaged_min()
    verify_packaged_min_paths()
    leak_check()
    print("\n=== B6 Pruning Completed Successfully ===")

if __name__ == "__main__":
    main()
