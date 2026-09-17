#!/usr/bin/env python3
"""
scripts/package_dataset.py

Implements Phase B: Measure, Resize, Package.
1. Resizes 980 field photos in not_crop/ and 174 photos in openset_holdout/
   to max edge 1024 px, aspect ratio preserved, JPEG quality 90,
   with ImageOps.exif_transpose applied and orientation tag stripped.
2. Copies all other images in data/raw/ verbatim without re-encoding to data/packaged/.
3. Verifies per-folder file counts are identical before and after.
4. Generates splits_packaged/ with data/raw/ rewritten to data/packaged/.
5. Copies splits/class_weights.json unchanged.
6. Verifies 100% of paths in splits_packaged/*.csv exist on disk.
7. Performs leak check on splits_packaged/*.csv for openset_holdout and not_crop_archive.
"""

import os
import sys
import csv
import json
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageOps

RAW_DIR = Path("data/raw")
PACKAGED_DIR = Path("data/packaged")
SPLITS_DIR = Path("splits")
SPLITS_PACKAGED_DIR = Path("splits_packaged")

TARGET_NOT_CROP_SUBFOLDERS = {
    "hands", "feet_shoes", "pavement", "walls", "soil_field", "green_noncrop"
}
TARGET_HOLDOUT_SUBFOLDERS = {
    "hands", "feet_shoes", "pavement", "walls", "soil", "green_noncrop"
}

def is_hidden_or_git(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)

def resize_and_save_image(src_path: Path, dst_path: Path, max_edge: int = 1024, quality: int = 90):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        transposed = ImageOps.exif_transpose(img)
        if transposed.mode != "RGB":
            transposed = transposed.convert("RGB")
        w, h = transposed.size
        if max(w, h) > max_edge:
            if w >= h:
                new_w = max_edge
                new_h = int(round(h * max_edge / w))
            else:
                new_h = max_edge
                new_w = int(round(w * max_edge / h))
            resized = transposed.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            resized = transposed
        
        # Save with no orientation tag (default PIL save excludes EXIF unless passed)
        resized.save(dst_path, format="JPEG", quality=quality)

def package_images():
    print("=== Step B2: Packaging and Resizing Images ===")
    PACKAGED_DIR.mkdir(parents=True, exist_ok=True)

    resized_count = 0
    copied_count = 0

    all_raw_files = [p for p in RAW_DIR.rglob("*") if p.is_file() and not is_hidden_or_git(p)]
    print(f"Total non-hidden files in {RAW_DIR}: {len(all_raw_files)}")

    for p in all_raw_files:
        rel = p.relative_to(RAW_DIR)
        dst = PACKAGED_DIR / rel

        # Determine if this is one of the 1,154 target images
        is_target = False
        parts = rel.parts
        if len(parts) >= 2:
            top_folder = parts[0]
            sub_folder = parts[1]
            if top_folder == "not_crop" and sub_folder in TARGET_NOT_CROP_SUBFOLDERS:
                is_target = True
            elif top_folder == "openset_holdout" and sub_folder in TARGET_HOLDOUT_SUBFOLDERS:
                is_target = True

        if is_target:
            resize_and_save_image(p, dst, max_edge=1024, quality=90)
            resized_count += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            copied_count += 1

    print(f"Resized field/holdout images: {resized_count} (expected 1154: 980 not_crop + 174 holdout)")
    print(f"Copied unchanged files: {copied_count}")
    print(f"Total packaged files: {resized_count + copied_count}")
    assert resized_count == 1154, f"Expected 1154 resized images, got {resized_count}"

def verify_resized_properties():
    print("\n=== Verifying Resized Images Properties (Dimensions <= 1024, No Orientation Tag) ===")
    checked = 0
    for top_folder, subfolders in [("not_crop", TARGET_NOT_CROP_SUBFOLDERS), ("openset_holdout", TARGET_HOLDOUT_SUBFOLDERS)]:
        for sub in subfolders:
            folder = PACKAGED_DIR / top_folder / sub
            for img_p in folder.glob("*"):
                if img_p.is_file() and not img_p.name.startswith("."):
                    with Image.open(img_p) as img:
                        w, h = img.size
                        assert max(w, h) <= 1024, f"Image {img_p} exceeds 1024: size={img.size}"
                        exif = img.getexif()
                        assert 0x0112 not in exif, f"Image {img_p} has orientation tag {exif.get(0x0112)}"
                        checked += 1
    print(f"Verified all {checked} resized images: max dimension <= 1024 px and orientation tag absent.")
    assert checked == 1154, f"Expected 1154 checked, got {checked}"

def verify_folder_counts():
    print("\n=== Verifying Per-Folder File Counts Between raw and packaged ===")
    mismatches = []
    print(f"{'Folder':25s} | {'data/raw':10s} | {'data/packaged':13s} | Status")
    print("-" * 62)
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            raw_count = len([p for p in d.rglob("*") if p.is_file() and not is_hidden_or_git(p)])
            pkg_d = PACKAGED_DIR / d.name
            pkg_count = len([p for p in pkg_d.rglob("*") if p.is_file() and not is_hidden_or_git(p)]) if pkg_d.exists() else 0
            status = "IDENTICAL" if raw_count == pkg_count else "MISMATCH"
            print(f"{d.name:25s} | {raw_count:10d} | {pkg_count:13d} | {status}")
            if raw_count != pkg_count:
                mismatches.append((d.name, raw_count, pkg_count))
    
    assert len(mismatches) == 0, f"Folder count mismatches found: {mismatches}"
    print("All folder counts are 100% IDENTICAL!")

def rewrite_splits():
    print("\n=== Step B3: Rewriting Split Paths into splits_packaged/ ===")
    SPLITS_PACKAGED_DIR.mkdir(parents=True, exist_ok=True)
    
    for split_file in sorted(SPLITS_DIR.iterdir()):
        if split_file.name.startswith("."):
            continue
        dst_file = SPLITS_PACKAGED_DIR / split_file.name
        
        if split_file.suffix == ".json":
            # Copy class_weights.json unchanged
            shutil.copy2(split_file, dst_file)
            print(f"Copied {split_file.name} unchanged to {dst_file}")
        elif split_file.suffix == ".csv":
            with open(split_file, "r", encoding="utf-8") as f_in, open(dst_file, "w", encoding="utf-8", newline="") as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                header = next(reader)
                writer.writerow(header)
                
                path_idx = header.index("path") if "path" in header else -1
                rewritten_rows = 0
                for row in reader:
                    if path_idx >= 0 and len(row) > path_idx:
                        if row[path_idx].startswith("data/raw/"):
                            row[path_idx] = row[path_idx].replace("data/raw/", "data/packaged/", 1)
                            rewritten_rows += 1
                    writer.writerow(row)
            print(f"Wrote {dst_file.name}: {rewritten_rows} paths rewritten")

def verify_packaged_split_paths():
    print("\n=== Verifying All Paths in splits_packaged/*.csv Exist on Disk ===")
    total_checked = 0
    total_missing = 0
    missing_samples = []

    for csv_file in sorted(SPLITS_PACKAGED_DIR.glob("*.csv")):
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
    assert total_missing == 0, f"Expected 0 missing files, got {total_missing}"

def leak_check():
    print("\n=== Step B4: Leak Check ===")
    cmd = ["grep", "-c", "openset_holdout\\|not_crop_archive"] + sorted(str(p) for p in SPLITS_PACKAGED_DIR.glob("*.csv"))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("grep output:")
    print(res.stdout.strip())
    # Verify count is 0 in all lines
    for line in res.stdout.strip().splitlines():
        if ":" in line:
            fname, cnt = line.rsplit(":", 1)
            assert int(cnt) == 0, f"Leak detected in {fname}: count={cnt}"
    print("Leak check PASSED: 0 occurrences of openset_holdout or not_crop_archive in any split CSV!")

def main():
    package_images()
    verify_resized_properties()
    verify_folder_counts()
    rewrite_splits()
    verify_packaged_split_paths()
    leak_check()
    print("\n=== Phase B Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
