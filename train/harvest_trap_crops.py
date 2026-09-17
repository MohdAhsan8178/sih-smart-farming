"""
Harvest and deduplicate 64x64 trap patch crops for Model B (3-class taxonomy).

Classes:
  0: small_pale_winged (Wageningen 4TU 'WF' + PST whiteflies)
  1: larger_insect     (Wageningen 4TU 'MR'/'NC' + Ong & Høye DSLR/Webcam/Smartphone beetles)
  2: debris            (Wageningen/PST clear background + Ong & Høye other objects)

Perceptual hash deduplication (dhash <= 4) is enforced to ensure no near-duplicate
crops exist within or across splits.
"""

import sys
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm

from configs.classes_model_b import CLASS_NAMES, IDX
from configs.paths import DATA, PROCESSED, SPLITS

PATCH_DIR = PROCESSED / "model_b_patches"
MANIFEST_PATH = SPLITS / "model_b_manifest.csv"

# Target counts per class
TARGET_PER_CLASS = 3000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def compute_dhash(img_bgr: np.ndarray) -> imagehash.ImageHash:
    """Compute 64-bit difference hash for 64x64 BGR patch."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return imagehash.dhash(pil_img, hash_size=8)


def is_near_duplicate(new_hash: imagehash.ImageHash, hash_list: List[imagehash.ImageHash], max_dist: int = 4) -> bool:
    """Check if new_hash is within max_dist Hamming distance of any hash in hash_list."""
    for h in hash_list:
        if new_hash - h <= max_dist:
            return True
    return False


def harvest_4tu() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Harvest crops from Wageningen 4TU dataset."""
    print("[harvest] Processing Wageningen 4TU dataset...")
    four_tu_dir = DATA / "model_b_sources" / "4tu" / "4TUDatasetAnonymised"
    if not four_tu_dir.exists():
        print("  Warning: 4TU directory not found.")
        return [], [], []

    xml_files = sorted(list(four_tu_dir.glob("*.xml")))
    pale_winged = []
    larger_insects = []
    debris_crops = []

    for xf in tqdm(xml_files, desc="4TU files"):
        img_p = xf.with_suffix(".jpg")
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        tree = ET.parse(xf)
        boxes = []
        for obj in tree.findall("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            xmin = int(float(b.find("xmin").text))
            ymin = int(float(b.find("ymin").text))
            xmax = int(float(b.find("xmax").text))
            ymax = int(float(b.find("ymax").text))
            boxes.append((name, xmin, ymin, xmax, ymax))

            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # Extract 64x64 centered crop
            x0 = max(0, min(w - 64, cx - 32))
            y0 = max(0, min(h - 64, cy - 32))
            crop = img[y0:y0 + 64, x0:x0 + 64]
            if crop.shape[:2] != (64, 64):
                continue

            entry = {
                "crop": crop,
                "source": "Wageningen_4TU",
                "device": "DSLR_highres",
                "original_label": name,
            }

            if name == "WF":
                pale_winged.append(entry)
            elif name in ("MR", "NC"):
                larger_insects.append(entry)

        # Harvest 4 background debris crops per card (regions > 60px from any box)
        bg_attempts = 0
        bg_harvested = 0
        while bg_attempts < 30 and bg_harvested < 4:
            bg_attempts += 1
            rx = random.randint(32, w - 32)
            ry = random.randint(32, h - 32)
            # Check distance to all boxes
            too_close = False
            for _, bx0, by0, bx1, by1 in boxes:
                bcx, bcy = (bx0 + bx1) // 2, (by0 + by1) // 2
                if abs(rx - bcx) < 60 and abs(ry - bcy) < 60:
                    too_close = True
                    break
            if not too_close:
                x0 = rx - 32
                y0 = ry - 32
                crop = img[y0:y0 + 64, x0:x0 + 64]
                if crop.shape[:2] == (64, 64):
                    debris_crops.append({
                        "crop": crop,
                        "source": "Wageningen_4TU",
                        "device": "DSLR_highres",
                        "original_label": "background",
                    })
                    bg_harvested += 1

    print(f"  4TU harvested: {len(pale_winged)} WF, {len(larger_insects)} MR/NC, {len(debris_crops)} debris")
    return pale_winged, larger_insects, debris_crops


def harvest_pst() -> Tuple[List[Dict], List[Dict]]:
    """Harvest crops from PST Whitefly dataset (Zenodo 7801239)."""
    print("[harvest] Processing PST Whitefly dataset...")
    pst_dir = DATA / "model_b_sources" / "pst" / "extracted"
    if not pst_dir.exists():
        print("  Warning: PST directory not found.")
        return [], []

    pale_winged = []
    debris_crops = []

    for split in ["train", "test"]:
        split_dir = pst_dir / split
        ann_p = split_dir / "annotations.csv"
        frames_dir = split_dir / "fullFrames"
        if not ann_p.exists() or not frames_dir.exists():
            continue

        df = pd.read_csv(ann_p)
        grouped = df.groupby("imgName")

        for img_name, group in tqdm(grouped, desc=f"PST {split}"):
            img_p = frames_dir / img_name
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]

            points = list(zip(group["X"].astype(float), group["Y"].astype(float)))

            # Sample up to 150 whiteflies per frame to avoid card dominance
            sampled_pts = random.sample(points, min(150, len(points)))
            for x, y in sampled_pts:
                cx = int(round(x))
                cy = int(round(y))
                x0 = max(0, min(w - 64, cx - 32))
                y0 = max(0, min(h - 64, cy - 32))
                crop = img[y0:y0 + 64, x0:x0 + 64]
                if crop.shape[:2] == (64, 64):
                    pale_winged.append({
                        "crop": crop,
                        "source": "PST_Zenodo",
                        "device": "DSLR_highres",
                        "original_label": "whitefly",
                    })

            # Background samples (> 60px from any point)
            bg_attempts = 0
            bg_harvested = 0
            while bg_attempts < 50 and bg_harvested < 10:
                bg_attempts += 1
                rx = random.randint(32, w - 32)
                ry = random.randint(32, h - 32)
                too_close = False
                for px, py in points:
                    if abs(rx - px) < 60 and abs(ry - py) < 60:
                        too_close = True
                        break
                if not too_close:
                    x0 = rx - 32
                    y0 = ry - 32
                    crop = img[y0:y0 + 64, x0:x0 + 64]
                    if crop.shape[:2] == (64, 64):
                        debris_crops.append({
                            "crop": crop,
                            "source": "PST_Zenodo",
                            "device": "DSLR_highres",
                            "original_label": "background",
                        })
                        bg_harvested += 1

    print(f"  PST harvested: {len(pale_winged)} whitefly, {len(debris_crops)} debris")
    return pale_winged, debris_crops


def harvest_ong_hoye() -> Tuple[List[Dict], List[Dict]]:
    """Harvest crops from Ong & Høye (Figshare 23617383)."""
    print("[harvest] Processing Ong & Høye dataset...")
    oh_dir = DATA / "model_b_sources" / "ong_hoye"
    if not oh_dir.exists():
        print("  Warning: Ong & Høye directory not found.")
        return [], []

    larger_insects = []
    debris_crops = []

    # 1. Harvest beetles by device
    for device, d_tag in [("DSLR", "DSLR"), ("Webcam", "Webcam"), ("Smart_phone", "Smartphone")]:
        dp = oh_dir / device
        if not dp.exists():
            continue
        imgs = sorted(list(dp.rglob("*.jpg")) + list(dp.rglob("*.png")))
        # Sample up to 600 per device to keep balanced
        sampled = random.sample(imgs, min(600, len(imgs)))
        for ip in sampled:
            img = cv2.imread(str(ip))
            if img is None:
                continue
            crop = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            larger_insects.append({
                "crop": crop,
                "source": "Ong_Hoye_Figshare",
                "device": d_tag,
                "original_label": "stored_beetle",
            })

    # 2. Harvest Other_objects-samples (debris)
    other_p = oh_dir / "Other_objects-samples"
    if other_p.exists():
        imgs = sorted(list(other_p.glob("*.jpg")) + list(other_p.glob("*.png")))
        for ip in imgs:
            img = cv2.imread(str(ip))
            if img is None:
                continue
            crop = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            debris_crops.append({
                "crop": crop,
                "source": "Ong_Hoye_Figshare",
                "device": "Smartphone_webcam_mix",
                "original_label": "other_objects",
            })

    print(f"  Ong & Høye harvested: {len(larger_insects)} beetle crops, {len(debris_crops)} other_objects debris")
    return larger_insects, debris_crops


def run_harvest():
    """Run full harvest, dhash dedup, and split creation."""
    four_tu_wf, four_tu_larger, four_tu_debris = harvest_4tu()
    pst_wf, pst_debris = harvest_pst()
    oh_larger, oh_debris = harvest_ong_hoye()

    # Combine by class
    all_pale_winged = four_tu_wf + pst_wf
    all_larger = four_tu_larger + oh_larger
    all_debris = four_tu_debris + pst_debris + oh_debris

    print(f"\n[harvest] Total raw collected before dedup:")
    print(f"  small_pale_winged : {len(all_pale_winged)}")
    print(f"  larger_insect     : {len(all_larger)}")
    print(f"  debris            : {len(all_debris)}")

    # Create destination directories
    for cname in CLASS_NAMES:
        (PATCH_DIR / cname).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    class_pools = {
        "small_pale_winged": all_pale_winged,
        "larger_insect": all_larger,
        "debris": all_debris,
    }

    # Deduplicate and sample up to TARGET_PER_CLASS per class
    for cname, items in class_pools.items():
        c_idx = IDX[cname]
        random.shuffle(items)
        accepted_hashes = []
        accepted_count = 0

        print(f"\n[harvest] Deduplicating {cname} (target: {TARGET_PER_CLASS})...")
        for item in tqdm(items, desc=cname):
            if accepted_count >= TARGET_PER_CLASS:
                break
            crop = item["crop"]
            chash = compute_dhash(crop)
            if is_near_duplicate(chash, accepted_hashes, max_dist=4):
                continue

            accepted_hashes.append(chash)
            img_id = f"{cname}_{accepted_count:05d}"
            save_path = PATCH_DIR / cname / f"{img_id}.png"
            cv2.imwrite(str(save_path), crop)

            manifest_rows.append({
                "crop_id": img_id,
                "class_name": cname,
                "class_idx": c_idx,
                "source_dataset": item["source"],
                "device": item["device"],
                "original_label": item["original_label"],
                "dhash": str(chash),
                "rel_path": str(save_path.relative_to(PATCH_DIR.parent.parent)),
            })
            accepted_count += 1

        print(f"  -> Accepted {accepted_count} unique crops for {cname}")

    df_manifest = pd.DataFrame(manifest_rows)
    print(f"\n[harvest] Total deduplicated crops: {len(df_manifest)}")
    print(df_manifest["class_name"].value_counts())

    # Create train / val / test_clean / test_degraded splits
    print("\n[harvest] Creating balanced splits...")
    df_manifest["split"] = "train"

    for cname in CLASS_NAMES:
        c_rows = df_manifest[df_manifest["class_name"] == cname]
        indices = c_rows.index.tolist()
        random.shuffle(indices)

        n_total = len(indices)
        n_val = int(0.15 * n_total)
        n_test = int(0.15 * n_total)
        n_train = n_total - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        df_manifest.loc[train_idx, "split"] = "train"
        df_manifest.loc[val_idx, "split"] = "val"

        # Split test into clean vs degraded
        for tidx in test_idx:
            dev = df_manifest.loc[tidx, "device"]
            if dev in ("Webcam", "Smartphone", "Smartphone_webcam_mix"):
                df_manifest.loc[tidx, "split"] = "test_degraded"
            else:
                df_manifest.loc[tidx, "split"] = "test_clean"

    SPLITS.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved manifest to {MANIFEST_PATH}")
    print("Split breakdown:")
    print(pd.crosstab(df_manifest["class_name"], df_manifest["split"]))


if __name__ == "__main__":
    run_harvest()
