"""
Build the unified image manifest and record taxonomy mapping decisions.

Generates:
  - splits/all_images.csv    (path, label, source_dataset, orig_folder)
  - splits/class_mapping.csv (source_dataset, orig_folder, canonical_label, rationale)
  - artifacts/reports/class_distribution.txt
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import (
    ROOT, RAW, PADDY, SUGAR_THITE, SUGAR_DAPHAL, NOTCROP, SPLITS, REPORTS
)
from configs.classes import CLASS_NAMES

def get_image_paths(folder_path):
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in img_exts:
                paths.append(Path(root) / f)
    return paths

def build_manifest():
    print("[build_manifest] Walking raw dataset directories...")
    
    mapping_records = []
    image_records = []

    # 1. Paddy Doctor (train_images)
    paddy_train = PADDY / 'train_images'
    paddy_map = {
        'normal': ('rice__normal', 'Healthy paddy control from field survey'),
        'bacterial_leaf_blight': ('rice__bacterial_leaf_blight', 'Xanthomonas oryzae pv. oryzae'),
        'bacterial_leaf_streak': ('rice__bacterial_leaf_streak', 'Xanthomonas oryzae pv. oryzicola'),
        'bacterial_panicle_blight': ('rice__bacterial_panicle_blight', 'Burkholderia glumae'),
        'blast': ('rice__blast', 'Magnaporthe oryzae'),
        'brown_spot': ('rice__brown_spot', 'Bipolaris oryzae'),
        'dead_heart': ('rice__yellow_stem_borer', 'Yellow stem borer (Scirpophaga incertulas) larval damage symptom'),
        'downy_mildew': ('rice__downy_mildew', 'Sclerophthora macrospora'),
        'hispa': ('rice__hispa', 'Dicladispa armigera leaf scraping damage'),
        'tungro': ('rice__tungro', 'Rice tungro bacilliform and spherical viruses (RTBV/RTSV)')
    }

    if paddy_train.exists():
        for d in sorted(paddy_train.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                orig_name = d.name
                if orig_name in paddy_map:
                    canon, rationale = paddy_map[orig_name]
                    mapping_records.append({
                        'source_dataset': 'paddy_doctor',
                        'orig_folder': orig_name,
                        'canonical_label': canon,
                        'rationale': rationale
                    })
                    imgs = get_image_paths(d)
                    for p in imgs:
                        image_records.append({
                            'path': str(p.relative_to(ROOT)),
                            'label': canon,
                            'source_dataset': 'paddy_doctor',
                            'orig_folder': orig_name
                        })
                else:
                    print(f"[build_manifest] Skipping unmapped Paddy Doctor folder: {orig_name}")

    # 2. Sugarcane Thite (355y629ynj)
    thite_map = {
        'Banded Chlorosis': ('sugarcane__banded_chlorosis', 'Cold/environmental chlorotic banding'),
        'Brown Spot': ('sugarcane__brown_spot', 'Cercospora longipes'),
        'BrownRust': ('sugarcane__rust', 'Puccinia melanocephala; merged with rust per spec'),
        'Dried Leaves': ('sugarcane__dried_leaf', 'Physiological/senescent dried leaf tissue'),
        'Grassy shoot': ('sugarcane__grassy_shoot', 'Candidatus Phytoplasma sacchari'),
        'Healthy Leaves': ('sugarcane__healthy', 'Healthy sugarcane leaf tissue'),
        'Pokkah Boeng': ('sugarcane__pokkah_boeng', 'Fusarium moniliforme complex'),
        'Sett Rot': ('sugarcane__sett_rot', 'Ceratocystis paradoxa'),
        'Viral Disease': ('sugarcane__mosaic', 'Sugarcane Mosaic Virus (SCMV)'),
        'Yellow Leaf': ('sugarcane__yellow_leaf', 'Sugarcane Yellow Leaf Virus (SCYLV)'),
        'smut': ('sugarcane__smut', 'Sporisorium scitamineum')
    }

    if SUGAR_THITE.exists():
        for d in sorted(SUGAR_THITE.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                orig_name = d.name
                if orig_name in thite_map:
                    canon, rationale = thite_map[orig_name]
                    mapping_records.append({
                        'source_dataset': 'sugarcane_thite',
                        'orig_folder': orig_name,
                        'canonical_label': canon,
                        'rationale': rationale
                    })
                    imgs = get_image_paths(d)
                    for p in imgs:
                        image_records.append({
                            'path': str(p.relative_to(ROOT)),
                            'label': canon,
                            'source_dataset': 'sugarcane_thite',
                            'orig_folder': orig_name
                        })
                else:
                    print(f"[build_manifest] Skipping unmapped Sugarcane Thite folder: {orig_name}")

    # 3. Sugarcane Daphal (9424skmnrk)
    daphal_map = {
        'Healthy': ('sugarcane__healthy', 'Healthy sugarcane leaf control across diverse smartphones'),
        'Mosaic': ('sugarcane__mosaic', 'Sugarcane Mosaic Virus across diverse smartphones'),
        'RedRot': ('sugarcane__red_rot', 'Colletotrichum falcatum; taken exclusively from Daphal per spec'),
        'Rust': ('sugarcane__rust', 'Puccinia melanocephala; merged with brown rust per spec'),
        'Yellow': ('sugarcane__yellow_leaf', 'Sugarcane yellow leaf disease; merged per spec')
    }

    if SUGAR_DAPHAL.exists():
        for d in sorted(SUGAR_DAPHAL.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                orig_name = d.name
                if orig_name in daphal_map:
                    canon, rationale = daphal_map[orig_name]
                    mapping_records.append({
                        'source_dataset': 'sugarcane_daphal',
                        'orig_folder': orig_name,
                        'canonical_label': canon,
                        'rationale': rationale
                    })
                    imgs = get_image_paths(d)
                    for p in imgs:
                        image_records.append({
                            'path': str(p.relative_to(ROOT)),
                            'label': canon,
                            'source_dataset': 'sugarcane_daphal',
                            'orig_folder': orig_name
                        })
                else:
                    print(f"[build_manifest] Skipping unmapped Sugarcane Daphal folder: {orig_name}")

    # 4. not_crop
    if NOTCROP.exists():
        for d in sorted(NOTCROP.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                imgs = get_image_paths(d)
                if imgs:
                    orig_name = d.name
                    mapping_records.append({
                        'source_dataset': 'not_crop',
                        'orig_folder': orig_name,
                        'canonical_label': 'not_crop',
                        'rationale': f'Field background negative category: {orig_name}'
                    })
                    for p in imgs:
                        image_records.append({
                            'path': str(p.relative_to(ROOT)),
                            'label': 'not_crop',
                            'source_dataset': 'not_crop',
                            'orig_folder': orig_name
                        })

    # Build DataFrames
    df_map = pd.DataFrame(mapping_records)
    df_imgs = pd.DataFrame(image_records)

    SPLITS.mkdir(parents=True, exist_ok=True)
    map_csv = SPLITS / 'class_mapping.csv'
    all_csv = SPLITS / 'all_images.csv'

    df_map.to_csv(map_csv, index=False)
    df_imgs.to_csv(all_csv, index=False)
    print(f"[build_manifest] Wrote {len(df_map)} mapping rows to {map_csv}")
    print(f"[build_manifest] Wrote {len(df_imgs)} image records to {all_csv}")

    # Validation
    unknown = set(df_imgs.label) - set(CLASS_NAMES)
    assert not unknown, f"Found labels outside canonical taxonomy: {unknown}"

    # Class distribution report
    dist = df_imgs.label.value_counts()
    REPORTS.mkdir(parents=True, exist_ok=True)
    dist_file = REPORTS / 'class_distribution.txt'
    
    header = f"{'Canonical Class':<35} | {'Count':<8} | {'% Total':<8}"
    sep = "-" * len(header)
    lines = [f"TOTAL IMAGES: {len(df_imgs)}", f"TOTAL CLASSES POPULATED: {df_imgs.label.nunique()}", "", header, sep]
    for cls_name, count in dist.items():
        pct = count * 100 / len(df_imgs)
        lines.append(f"{cls_name:<35} | {count:<8} | {pct:>6.2f}%")
    
    report_text = "\n".join(lines) + "\n"
    dist_file.write_text(report_text)
    print(f"[build_manifest] Wrote class distribution to {dist_file}")
    print("\n" + report_text)

if __name__ == "__main__":
    build_manifest()
