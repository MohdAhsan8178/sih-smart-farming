"""
Rebuild Model B Dataset with Group-Aware Splitting & Leakage Governance.

Key Architectural Guarantees:
1. Group-Aware Splitting:
   - Grouping key:
     * PST: group_id = f"pst_{card_name}" (28 trap cards).
     * Ong & Hoye: group_id = f"ong_specimen_{specimen_id}" (same specimen across DSLR/Webcam/Smartphone).
     * 4TU: group_id = f"4tu_{card_name}" (284 trap cards).
   - All crops from the same group belong strictly to ONE split.
2. Cross-Source Generalization Holdout:
   - 100% of Wageningen 4TU (284 cards, all classes) held out in 'test_cross_source'.
3. In-Distribution Pool (PST + Ong & Hoye):
   - StratifiedGroupKFold on group_id:
     * Train: ~70-75%
     * Val: ~12-15%
     * Test In-Distribution: ~12-15%
4. No Arbitrary Data Capping:
   - Harvests all valid deduplicated crops (dhash <= 4).
   - Handles class imbalance via weighted cross-entropy loss.
5. Hard Open-Set (OOD) Directory:
   - Weighted heavily toward hard OOD:
     * Insects on real yellow glue that are NOT our target classes (from PST & 4TU).
     * Card defects, grid lines, fiducials, and dried glue bubbles.
     * Foreign debris and seeds on yellow stages (Ong & Hoye other_objects).
     * Tight agricultural pest wax morphology crops (e.g. woolly aphid / scale wax).
6. Contact Sheets:
   - 10x10 grid of 100 random crops per class saved to artifacts/inspection/
"""

import sys
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

from configs.classes_model_b import CLASS_NAMES, IDX
from configs.paths import DATA, PROCESSED, SPLITS, ROOT

PATCH_DIR_V2 = PROCESSED / "model_b_patches_v2"
MANIFEST_V2_PATH = SPLITS / "model_b_manifest_v2.csv"
INSPECTION_DIR = ROOT / "artifacts" / "inspection"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def compute_dhash(img_bgr: np.ndarray) -> imagehash.ImageHash:
    """Compute 64-bit difference hash for 64x64 BGR patch."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return imagehash.dhash(pil_img, hash_size=8)


def is_near_duplicate(new_hash: imagehash.ImageHash, hash_set: Set[imagehash.ImageHash], max_dist: int = 4) -> bool:
    """Check if new_hash is within max_dist Hamming distance of any hash in hash_set."""
    for h in hash_set:
        if new_hash - h <= max_dist:
            return True
    return False


def harvest_pst_full() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Harvest crops from PST Whitefly dataset (Zenodo 7801239).
    Returns:
      (whiteflies, background_debris, hard_ood_crops)
    """
    print("[harvest] Processing PST dataset (all 28 cards)...")
    pst_dir = DATA / "model_b_sources" / "pst" / "extracted"
    if not pst_dir.exists():
        print("  Warning: PST directory not found.")
        return [], [], []

    whiteflies = []
    debris_crops = []
    hard_ood_crops = []

    for split_folder in ["train", "test"]:
        s_dir = pst_dir / split_folder
        ann_p = s_dir / "annotations.csv"
        frames_dir = s_dir / "fullFrames"
        if not ann_p.exists() or not frames_dir.exists():
            continue

        df = pd.read_csv(ann_p)
        grouped = df.groupby("imgName")

        for img_name, group in tqdm(grouped, desc=f"PST {split_folder}"):
            img_p = frames_dir / img_name
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            card_stem = Path(img_name).stem
            card_group_id = f"pst_{card_stem}"

            pts = group[["X", "Y"]].values.astype(float)

            # 1. Extract ALL valid whiteflies (no 150-per-card cap)
            for x, y in pts:
                cx = int(round(x))
                cy = int(round(y))
                if cx < 32 or cx > w - 32 or cy < 32 or cy > h - 32:
                    continue
                crop = img[cy - 32:cy + 32, cx - 32:cx + 32]
                if crop.shape[:2] == (64, 64):
                    whiteflies.append({
                        "crop": crop,
                        "class_name": "small_pale_winged",
                        "source": "PST_Zenodo",
                        "group_id": card_group_id,
                        "device": "DSLR_highres",
                        "original_label": "whitefly",
                    })

            # 2. Extract genuine clean glue debris (background regions far from annotations)
            bg_attempts = 0
            bg_harvested = 0
            while bg_attempts < 400 and bg_harvested < 150:
                bg_attempts += 1
                rx = random.randint(40, w - 40)
                ry = random.randint(40, h - 40)
                dists = np.sqrt((pts[:, 0] - rx)**2 + (pts[:, 1] - ry)**2)
                if dists.min() > 80: # Strictly far from any whitefly
                    crop = img[ry - 32:ry + 32, rx - 32:rx + 32]
                    # Verify crop is not empty black/border
                    if crop.shape[:2] == (64, 64) and crop[:, :, 1].mean() > 60:
                        debris_crops.append({
                            "crop": crop,
                            "class_name": "debris",
                            "source": "PST_Zenodo",
                            "group_id": card_group_id,
                            "device": "DSLR_highres",
                            "original_label": "card_glue_background",
                        })
                        bg_harvested += 1

            # 3. Extract HARD OOD: Non-whitefly incidental dark bodies & card artifacts
            b_chan = img[:, :, 0]
            thresh = cv2.adaptiveThreshold(b_chan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ood_harvested = 0
            for c in cnts:
                if ood_harvested >= 25:
                    break
                area = cv2.contourArea(c)
                if 80 < area < 4000:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        if 32 < cx < w - 32 and 32 < cy < h - 32:
                            dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
                            if dists.min() > 100: # Far from any whitefly
                                crop = img[cy - 32:cy + 32, cx - 32:cx + 32]
                                if crop.shape[:2] == (64, 64):
                                    hard_ood_crops.append({
                                        "crop": crop,
                                        "class_name": "openset_hard",
                                        "source": "PST_Zenodo",
                                        "group_id": card_group_id,
                                        "device": "DSLR_highres",
                                        "original_label": "pst_non_whitefly_blob_or_defect",
                                    })
                                    ood_harvested += 1

    print(f"  PST harvested: {len(whiteflies)} WF, {len(debris_crops)} debris, {len(hard_ood_crops)} hard OOD")
    return whiteflies, debris_crops, hard_ood_crops


def harvest_ong_hoye_full() -> Tuple[List[Dict], List[Dict]]:
    """
    Harvest ALL beetle crops and Other_objects from Ong & Hoye.
    Guarantees specimen grouping so the same specimen across DSLR/Webcam/Smartphone
    shares the exact same group_id.
    """
    print("[harvest] Processing Ong & Hoye dataset (all specimens)...")
    oh_dir = DATA / "model_b_sources" / "ong_hoye"
    if not oh_dir.exists():
        print("  Warning: Ong & Hoye directory not found.")
        return [], []

    larger_insects = []
    debris_crops = []

    # 1. Harvest beetles across DSLR, Webcam, Smartphone
    for dev_folder, d_tag in [("DSLR", "DSLR"), ("Webcam", "Webcam"), ("Smart_phone", "Smartphone")]:
        dp = oh_dir / dev_folder
        if not dp.exists():
            continue
        imgs = sorted(list(dp.rglob("*.jpg")) + list(dp.rglob("*.png")))
        for ip in tqdm(imgs, desc=f"Ong & Hoye {d_tag}"):
            specimen_stem = ip.stem
            # Group key: specimen ID ensures DSLR/Webcam/Smartphone views stay in same split
            group_id = f"ong_specimen_{specimen_stem}"
            img = cv2.imread(str(ip))
            if img is None:
                continue
            crop = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            larger_insects.append({
                "crop": crop,
                "class_name": "larger_insect",
                "source": "Ong_Hoye_Figshare",
                "group_id": group_id,
                "device": d_tag,
                "original_label": "stored_beetle",
            })

    # 2. Harvest Other_objects-samples
    other_p = oh_dir / "Other_objects-samples"
    if other_p.exists():
        imgs = sorted(list(other_p.glob("*.jpg")) + list(other_p.glob("*.png")))
        for ip in imgs:
            specimen_stem = ip.stem
            group_id = f"ong_other_{specimen_stem}"
            img = cv2.imread(str(ip))
            if img is None:
                continue
            crop = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            debris_crops.append({
                "crop": crop,
                "class_name": "debris",
                "source": "Ong_Hoye_Figshare",
                "group_id": group_id,
                "device": "Smartphone_webcam_mix",
                "original_label": "other_objects",
            })

    print(f"  Ong & Hoye harvested: {len(larger_insects)} beetles, {len(debris_crops)} other_objects debris")
    return larger_insects, debris_crops


def harvest_4tu_holdout() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Harvest Wageningen 4TU dataset as the HELD-OUT CROSS-SOURCE EVALUATION DATASET.
    All 284 cards will be assigned to split='test_cross_source'.
    Also extracts hard OOD (e.g. thrips annotations and card border/grid markings).
    """
    print("[harvest] Processing Wageningen 4TU dataset (100% held-out cross-source)...")
    four_tu_dir = DATA / "model_b_sources" / "4tu" / "4TUDatasetAnonymised"
    if not four_tu_dir.exists():
        print("  Warning: 4TU directory not found.")
        return [], [], [], []

    xml_files = sorted(list(four_tu_dir.glob("*.xml")))
    wf_crops = []
    mr_nc_crops = []
    debris_crops = []
    hard_ood_crops = []

    for xf in tqdm(xml_files, desc="4TU holdout cards"):
        img_p = xf.with_suffix(".jpg")
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        card_stem = xf.stem
        group_id = f"4tu_{card_stem}"

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
            x0 = max(0, min(w - 64, cx - 32))
            y0 = max(0, min(h - 64, cy - 32))
            crop = img[y0:y0 + 64, x0:x0 + 64]
            if crop.shape[:2] != (64, 64):
                continue

            entry = {
                "crop": crop,
                "source": "Wageningen_4TU",
                "group_id": group_id,
                "device": "DSLR_highres",
                "original_label": name,
            }

            if name == "WF":
                entry["class_name"] = "small_pale_winged"
                wf_crops.append(entry)
            elif name in ("MR", "NC"):
                entry["class_name"] = "larger_insect"
                mr_nc_crops.append(entry)
            elif name == "TH":
                # Thrips on glue: hard OOD!
                entry["class_name"] = "openset_hard"
                entry["original_label"] = "thrips_on_glue"
                hard_ood_crops.append(entry)

        # Harvest 6 clean glue background debris per card
        bg_attempts = 0
        bg_harvested = 0
        while bg_attempts < 60 and bg_harvested < 6:
            bg_attempts += 1
            rx = random.randint(40, w - 40)
            ry = random.randint(40, h - 40)
            too_close = False
            for _, bx0, by0, bx1, by1 in boxes:
                bcx, bcy = (bx0 + bx1) // 2, (by0 + by1) // 2
                if abs(rx - bcx) < 70 and abs(ry - bcy) < 70:
                    too_close = True
                    break
            if not too_close:
                crop = img[ry - 32:ry + 32, rx - 32:rx + 32]
                if crop.shape[:2] == (64, 64):
                    debris_crops.append({
                        "crop": crop,
                        "class_name": "debris",
                        "source": "Wageningen_4TU",
                        "group_id": group_id,
                        "device": "DSLR_highres",
                        "original_label": "card_glue_background",
                    })
                    bg_harvested += 1

        # Harvest 1 card border / grid line marking as hard OOD
        grid_crop = img[max(0, h-70):max(0, h-70)+64, max(0, w//2 - 32):max(0, w//2 - 32)+64]
        if grid_crop.shape[:2] == (64, 64):
            hard_ood_crops.append({
                "crop": grid_crop,
                "class_name": "openset_hard",
                "source": "Wageningen_4TU",
                "group_id": group_id,
                "device": "DSLR_highres",
                "original_label": "4tu_card_border_or_grid",
            })

    print(f"  4TU held-out harvested: {len(wf_crops)} WF, {len(mr_nc_crops)} MR/NC, {len(debris_crops)} debris, {len(hard_ood_crops)} hard OOD")
    return wf_crops, mr_nc_crops, debris_crops, hard_ood_crops


def harvest_hard_agricultural_wax_ood() -> List[Dict]:
    """
    Harvest hard close-up crops representing pest wax morphology and tight non-crop textures
    to ensure the open-set evaluates woolly aphid / mealybug wax and non-leaf textures.
    """
    print("[harvest] Harvesting hard agricultural wax/scale OOD crops...")
    hard_crops = []
    openset_dir = DATA / "raw" / "openset"
    if not openset_dir.exists():
        return hard_crops

    candidate_subdirs = [
        "dtd_textures", "broadleaf_weeds", "plastic_mulch", "straw_mulch",
        "banglariceleaf_sheath_blight", "dhan_shomadhan_sheath_blight"
    ]
    for sub in candidate_subdirs:
        sp = openset_dir / sub
        if not sp.exists():
            continue
        imgs = sorted(list(sp.glob("*.jpg")) + list(sp.glob("*.png")))
        sampled = random.sample(imgs, min(30, len(imgs)))
        for ip in sampled:
            im = cv2.imread(str(ip))
            if im is None:
                continue
            h, w = im.shape[:2]
            if h >= 64 and w >= 64:
                cy, cx = h // 2, w // 2
                crop = im[cy-32:cy+32, cx-32:cx+32]
            else:
                crop = cv2.resize(im, (64, 64))
            hard_crops.append({
                "crop": crop,
                "class_name": "openset_hard",
                "source": f"hard_texture_{sub}",
                "group_id": f"ood_{sub}_{ip.stem}",
                "device": "field_camera",
                "original_label": sub,
            })
    print(f"  Harvested {len(hard_crops)} hard texture/morphology OOD crops")
    return hard_crops


def generate_contact_sheet(crops: List[np.ndarray], out_path: Path, grid_size: int = 10, patch_size: int = 64):
    """Generate a 10x10 contact sheet of 100 randomly sampled crops with 1px border."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = grid_size * grid_size
    if len(crops) == 0:
        print(f"  Warning: No crops to generate contact sheet for {out_path.name}")
        return
    sampled = random.sample(crops, min(n_samples, len(crops)))
    if len(sampled) < n_samples:
        sampled = (sampled * (n_samples // len(sampled) + 1))[:n_samples]

    cell_size = patch_size + 2 # 1px border on each side
    sheet_w = grid_size * cell_size
    sheet_h = grid_size * cell_size
    sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8) # Dark gray background

    for idx, crop in enumerate(sampled):
        r = idx // grid_size
        c = idx % grid_size
        y0 = r * cell_size + 1
        x0 = c * cell_size + 1
        if crop.shape[:2] != (patch_size, patch_size):
            crop = cv2.resize(crop, (patch_size, patch_size))
        sheet[y0:y0+patch_size, x0:x0+patch_size] = crop

    cv2.imwrite(str(out_path), sheet)
    print(f"[inspection] Saved contact sheet: {out_path}")


def run_rebuild():
    """Main workflow to rebuild dataset with group-aware splits and contact sheets."""
    # 1. Harvest from all sources
    pst_wf, pst_debris, pst_ood = harvest_pst_full()
    oh_larger, oh_debris = harvest_ong_hoye_full()
    tu_wf, tu_larger, tu_debris, tu_ood = harvest_4tu_holdout()
    wax_ood = harvest_hard_agricultural_wax_ood()

    all_hard_ood = pst_ood + tu_ood + wax_ood
    print(f"\n[harvest] Total Raw Harvest Summary:")
    print(f"  PST In-Dist Whitefly     : {len(pst_wf)}")
    print(f"  PST In-Dist Debris       : {len(pst_debris)}")
    print(f"  Ong & Hoye Larger Insect : {len(oh_larger)}")
    print(f"  Ong & Hoye Debris        : {len(oh_debris)}")
    print(f"  4TU Held-Out Whitefly    : {len(tu_wf)}")
    print(f"  4TU Held-Out Larger      : {len(tu_larger)}")
    print(f"  4TU Held-Out Debris      : {len(tu_debris)}")
    print(f"  Hard OOD (Open-Set)      : {len(all_hard_ood)}")

    # 2. Reset output directory
    if PATCH_DIR_V2.exists():
        shutil.rmtree(PATCH_DIR_V2)
    for cname in CLASS_NAMES + ["openset_hard"]:
        (PATCH_DIR_V2 / cname).mkdir(parents=True, exist_ok=True)

    # 3. Deduplicate In-Distribution Pool & Held-Out Pool
    print("\n[dedup] Performing dhash deduplication...")

    def dedup_pool(items: List[Dict], cname: str, split_override: str = None) -> Tuple[List[Dict], List[np.ndarray]]:
        accepted_rows = []
        accepted_crops = []
        seen_hashes = set()
        c_idx = IDX.get(cname, -1)

        for item in tqdm(items, desc=f"Dedup {cname}"):
            crop = item["crop"]
            chash = compute_dhash(crop)
            if is_near_duplicate(chash, seen_hashes, max_dist=4):
                continue
            seen_hashes.add(chash)

            count = len(accepted_rows)
            crop_id = f"{cname}_{count:06d}"
            save_path = PATCH_DIR_V2 / cname / f"{crop_id}.png"
            cv2.imwrite(str(save_path), crop)

            row = {
                "crop_id": crop_id,
                "class_name": cname,
                "class_idx": c_idx,
                "group_id": item["group_id"],
                "source_dataset": item["source"],
                "device": item["device"],
                "original_label": item["original_label"],
                "dhash": str(chash),
                "rel_path": str(save_path.relative_to(ROOT)),
                "split": split_override if split_override else "unassigned",
            }
            accepted_rows.append(row)
            accepted_crops.append(crop)
        return accepted_rows, accepted_crops

    # A. Dedup In-Distribution Whiteflies (from PST)
    pst_wf_rows, pst_wf_crops = dedup_pool(pst_wf, "small_pale_winged")
    # B. Dedup In-Distribution Larger Insects (from Ong & Hoye)
    oh_larger_rows, oh_larger_crops = dedup_pool(oh_larger, "larger_insect")
    # C. Dedup In-Distribution Debris (from PST background + Ong & Hoye other_objects)
    in_dist_debris = pst_debris + oh_debris
    random.shuffle(in_dist_debris)
    in_dist_debris_rows, in_dist_debris_crops = dedup_pool(in_dist_debris, "debris")

    # D. Dedup Held-Out 4TU (All assigned to split='test_cross_source')
    tu_wf_rows, tu_wf_crops = dedup_pool(tu_wf, "small_pale_winged", split_override="test_cross_source")
    tu_larger_rows, tu_larger_crops = dedup_pool(tu_larger, "larger_insect", split_override="test_cross_source")
    tu_debris_rows, tu_debris_crops = dedup_pool(tu_debris, "debris", split_override="test_cross_source")

    # E. Dedup Hard OOD (Assigned to split='openset_eval')
    random.shuffle(all_hard_ood)
    ood_rows, ood_crops = dedup_pool(all_hard_ood, "openset_hard", split_override="openset_eval")

    # 4. Perform StratifiedGroupKFold on In-Distribution Pool
    in_dist_rows = pst_wf_rows + oh_larger_rows + in_dist_debris_rows
    df_in_dist = pd.DataFrame(in_dist_rows)

    print(f"\n[split] Splitting in-distribution pool ({len(df_in_dist)} crops across {df_in_dist['group_id'].nunique()} groups)...")

    # 5-fold StratifiedGroupKFold -> Fold 0 = test_indist (20%), Folds 1..4 = train + val
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(sgkf.split(df_in_dist, y=df_in_dist["class_idx"], groups=df_in_dist["group_id"]))

    df_train_val = df_in_dist.iloc[train_val_idx].copy().reset_index(drop=True)
    df_test = df_in_dist.iloc[test_idx].copy().reset_index(drop=True)

    # Further split train_val into train (85%) and val (15%) with StratifiedGroupKFold
    sgkf_val = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED)
    tr_idx, val_idx = next(sgkf_val.split(df_train_val, y=df_train_val["class_idx"], groups=df_train_val["group_id"]))

    df_train = df_train_val.iloc[tr_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test_indist"

    # Separate test_indist into clean vs degraded based on capture device
    for idx_row in df_test.index:
        dev = df_test.loc[idx_row, "device"]
        if dev in ("Webcam", "Smartphone", "Smartphone_webcam_mix"):
            df_test.loc[idx_row, "split"] = "test_indist_degraded"
        else:
            df_test.loc[idx_row, "split"] = "test_indist_clean"

    # Assemble master manifest
    df_4tu_wf = pd.DataFrame(tu_wf_rows)
    df_4tu_larger = pd.DataFrame(tu_larger_rows)
    df_4tu_debris = pd.DataFrame(tu_debris_rows)
    df_ood = pd.DataFrame(ood_rows)

    df_all = pd.concat([df_train, df_val, df_test, df_4tu_wf, df_4tu_larger, df_4tu_debris, df_ood], ignore_index=True)

    # 5. Verify Zero Group Leakage
    train_groups = set(df_all[df_all["split"] == "train"]["group_id"])
    val_groups = set(df_all[df_all["split"] == "val"]["group_id"])
    test_indist_groups = set(df_all[df_all["split"].str.startswith("test_indist")]["group_id"])
    cross_groups = set(df_all[df_all["split"] == "test_cross_source"]["group_id"])

    assert len(train_groups & val_groups) == 0, "FATAL: Group leakage between train and val!"
    assert len(train_groups & test_indist_groups) == 0, "FATAL: Group leakage between train and test_indist!"
    assert len(val_groups & test_indist_groups) == 0, "FATAL: Group leakage between val and test_indist!"
    assert len(train_groups & cross_groups) == 0, "FATAL: 4TU group leaked into train!"

    print("[split] Zero Group Leakage ASSERTION PASSED.")

    # Save manifest
    SPLITS.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(MANIFEST_V2_PATH, index=False)
    print(f"[manifest] Saved master manifest to {MANIFEST_V2_PATH}")

    # Print breakdown
    print("\n" + "=" * 60)
    print("MASTER DATASET SPLIT SUMMARY:")
    print("=" * 60)
    print(pd.crosstab(df_all["class_name"], df_all["split"], margins=True))

    # 6. Generate Contact Sheets
    print("\n[inspection] Generating contact sheets for human visual audit...")
    generate_contact_sheet(pst_wf_crops, INSPECTION_DIR / "contact_sheet_small_pale_winged.png")
    generate_contact_sheet(oh_larger_crops, INSPECTION_DIR / "contact_sheet_larger_insect.png")
    generate_contact_sheet(in_dist_debris_crops, INSPECTION_DIR / "contact_sheet_debris.png")
    generate_contact_sheet(ood_crops, INSPECTION_DIR / "contact_sheet_openset_hard.png")

    print("\nRebuild and contact sheet generation successfully finished!")


if __name__ == "__main__":
    run_rebuild()
