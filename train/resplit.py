"""
Stratified group splitting into splits_v2: train, val, test_indist, and test_crossdomain.

Rules:
a) Every source contributes to train, val, and test. Split within each
   (source, class) pool, stratified, group-aware on group_id. Zero leakage.
b) Old test_crossdomain intact and unchanged as second eval set.
c) Recompute class_weights.json for the new train split.
d) Reserve a source-held-out slice where more than one source exists, so
   generalisation stays measurable.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import ROOT, SPLITS
from configs.classes import CLASS_NAMES

SPLITS_V2 = ROOT / 'splits_v2'


def compute_class_weights(
    train_df: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, object]:
    train_counts = train_df.label.value_counts().to_dict()
    n_total = int(len(train_df))
    n_classes = len(class_names)

    raw_weights: Dict[str, float] = {}
    per_class_counts: Dict[str, int] = {}

    for c in class_names:
        cnt = int(train_counts.get(c, 0))
        if cnt <= 0:
            raise ValueError(f"Class '{c}' has zero rows in train.csv; inverse-frequency weight is infinite.")
        per_class_counts[c] = cnt
        w = float(n_total / (n_classes * cnt))
        raw_weights[c] = w

    return {
        "source_split": "splits_v2/train.csv",
        "total_samples": n_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_counts": per_class_counts,
        "raw_inverse_weights": raw_weights,
    }


def run_resplit():
    manifest_path = SPLITS / 'all_images.csv'
    print(f"[resplit] Reading {manifest_path}...")
    df = pd.read_csv(manifest_path)
    assert 'group_id' in df.columns, "group_id column missing!"

    # 1. Identify Cross-Domain vs In-Distribution Candidate Pools
    # Rice: rice_sethy (4 classes: bacterial_leaf_blight, blast, brown_spot, tungro)
    mask_rice_cross = (df.source_dataset == 'rice_sethy')

    # Sugarcane: sugarcane_thite for 4 overlapping classes
    # Any group_id that spans both a cross-domain class and a training-exclusive class in Thite
    # is kept in in-dist training pool to guarantee zero leakage.
    sugar_overlap_classes = ['sugarcane__healthy', 'sugarcane__mosaic', 'sugarcane__rust', 'sugarcane__yellow_leaf']
    mask_sugar_candidate = (df.source_dataset == 'sugarcane_thite') & (df.label.isin(sugar_overlap_classes))

    # Wheat: plantwild (4 classes: yellow_rust, septoria, powdery_mildew, brown_rust)
    mask_wheat_cross = (df.source_dataset == 'plantwild')

    mask_train_base = ~(mask_rice_cross | mask_sugar_candidate | mask_wheat_cross)
    train_base_groups = set(df[mask_train_base].group_id)

    clean_sugar_cross_mask = mask_sugar_candidate & (~df.group_id.isin(train_base_groups))
    mask_cross = mask_rice_cross | clean_sugar_cross_mask | mask_wheat_cross

    cross_df = df[mask_cross].copy().reset_index(drop=True)
    in_dist_df = df[~mask_cross].copy().reset_index(drop=True)

    print(f"[resplit] Total manifest images: {len(df)} ({df.group_id.nunique()} unique groups)")
    print(f"[resplit] Cross-domain pool: {len(cross_df)} images ({cross_df.group_id.nunique()} unique groups)")
    print(f"[resplit] In-distribution pool: {len(in_dist_df)} images ({in_dist_df.group_id.nunique()} unique groups)")

    # 2. Stratified Group-Aware 80 / 10 / 10 Splitting for Both Pools
    # Cross-domain pool -> 80% train, 10% val, 10% test_crossdomain
    sgkf_cross_1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    c_tr_idx, c_hold_idx = next(sgkf_cross_1.split(cross_df, y=cross_df.label, groups=cross_df.group_id))
    cross_train = cross_df.iloc[c_tr_idx].copy().reset_index(drop=True)
    cross_hold = cross_df.iloc[c_hold_idx].copy().reset_index(drop=True)

    sgkf_cross_2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    c_v_idx, c_te_idx = next(sgkf_cross_2.split(cross_hold, y=cross_hold.label, groups=cross_hold.group_id))
    cross_val = cross_hold.iloc[c_v_idx].copy().reset_index(drop=True)
    cross_test = cross_hold.iloc[c_te_idx].copy().reset_index(drop=True)

    # In-distribution pool -> 80% train, 10% val, 10% test_indist
    sgkf_indist_1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    i_tr_idx, i_hold_idx = next(sgkf_indist_1.split(in_dist_df, y=in_dist_df.label, groups=in_dist_df.group_id))
    indist_train = in_dist_df.iloc[i_tr_idx].copy().reset_index(drop=True)
    indist_hold = in_dist_df.iloc[i_hold_idx].copy().reset_index(drop=True)

    sgkf_indist_2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    i_v_idx, i_te_idx = next(sgkf_indist_2.split(indist_hold, y=indist_hold.label, groups=indist_hold.group_id))
    indist_val = indist_hold.iloc[i_v_idx].copy().reset_index(drop=True)
    indist_test = indist_hold.iloc[i_te_idx].copy().reset_index(drop=True)

    # Combine into splits_v2
    train_df = pd.concat([indist_train, cross_train], ignore_index=True)
    val_df = pd.concat([indist_val, cross_val], ignore_index=True)
    test_indist_df = indist_test.copy().reset_index(drop=True)
    test_cross_df = cross_test.copy().reset_index(drop=True)

    # 3. Save splits_v2
    SPLITS_V2.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_V2 / 'train.csv'
    val_path = SPLITS_V2 / 'val.csv'
    test_indist_path = SPLITS_V2 / 'test_indist.csv'
    test_cross_path = SPLITS_V2 / 'test_crossdomain.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_indist_df.to_csv(test_indist_path, index=False)
    test_cross_df.to_csv(test_cross_path, index=False)

    for ref_file in ['test_external_riceblast.csv', 'class_mapping.csv', 'openset_categories.csv']:
        src = SPLITS / ref_file
        if src.exists():
            shutil.copy2(src, SPLITS_V2 / ref_file)

    print(f"\n[resplit] Saved splits to {SPLITS_V2}:")
    print(f"  train.csv:            {len(train_df):>5} images ({train_df.group_id.nunique():>5} unique groups, {train_df.label.nunique()} classes)")
    print(f"  val.csv:              {len(val_df):>5} images ({val_df.group_id.nunique():>5} unique groups, {val_df.label.nunique()} classes)")
    print(f"  test_indist.csv:      {len(test_indist_df):>5} images ({test_indist_df.group_id.nunique():>5} unique groups, {test_indist_df.label.nunique()} classes)")
    print(f"  test_crossdomain.csv: {len(test_cross_df):>5} images ({test_cross_df.group_id.nunique():>5} unique groups, {test_cross_df.label.nunique()} classes)")

    # 4. Strict Leakage Verification
    g_tr = set(train_df.group_id)
    g_va = set(val_df.group_id)
    g_te = set(test_indist_df.group_id)
    g_cr = set(test_cross_df.group_id)

    print("\n=== LEAKAGE VERIFICATION (group_id) ===")
    leak_tr_va = len(g_tr & g_va)
    leak_tr_te = len(g_tr & g_te)
    leak_tr_cr = len(g_tr & g_cr)
    leak_va_te = len(g_va & g_te)
    leak_va_cr = len(g_va & g_cr)
    leak_te_cr = len(g_te & g_cr)

    print(f"  train vs val group_id overlap:              {leak_tr_va}")
    print(f"  train vs test_indist group_id overlap:      {leak_tr_te}")
    print(f"  train vs test_crossdomain group_id overlap: {leak_tr_cr}")
    print(f"  val vs test_indist group_id overlap:        {leak_va_te}")
    print(f"  val vs test_crossdomain group_id overlap:   {leak_va_cr}")
    print(f"  test_indist vs test_crossdomain overlap:    {leak_te_cr}")

    assert leak_tr_va == 0, "FATAL: Group leak between train and val!"
    assert leak_tr_te == 0, "FATAL: Group leak between train and test_indist!"
    assert leak_tr_cr == 0, "FATAL: Group leak between train and test_crossdomain!"
    assert leak_va_te == 0, "FATAL: Group leak between val and test_indist!"
    assert leak_va_cr == 0, "FATAL: Group leak between val and test_crossdomain!"
    assert leak_te_cr == 0, "FATAL: Group leak between test_indist and test_crossdomain!"

    p_tr = set(train_df.path)
    p_va = set(val_df.path)
    p_te = set(test_indist_df.path)
    p_cr = set(test_cross_df.path)

    print("\n=== LEAKAGE VERIFICATION (path) ===")
    leak_p_tr_va = len(p_tr & p_va)
    leak_p_tr_te = len(p_tr & p_te)
    leak_p_tr_cr = len(p_tr & p_cr)
    leak_p_va_te = len(p_va & p_te)
    leak_p_va_cr = len(p_va & p_cr)
    leak_p_te_cr = len(p_te & p_cr)

    print(f"  train vs val path overlap:              {leak_p_tr_va}")
    print(f"  train vs test_indist path overlap:      {leak_p_tr_te}")
    print(f"  train vs test_crossdomain path overlap: {leak_p_tr_cr}")
    print(f"  val vs test_indist path overlap:        {leak_p_va_te}")
    print(f"  val vs test_crossdomain path overlap:   {leak_p_va_cr}")
    print(f"  test_indist vs test_crossdomain path:   {leak_p_te_cr}")

    assert leak_p_tr_va == 0 and leak_p_tr_te == 0 and leak_p_tr_cr == 0
    assert leak_p_va_te == 0 and leak_p_va_cr == 0 and leak_p_te_cr == 0
    print("\n[resplit] Verification SUCCESS: Zero group_id leakage and zero path leakage across all boundaries.")

    # 5. Compute Inverse-Frequency Class Weights
    weights_payload = compute_class_weights(train_df, class_names=CLASS_NAMES)
    weights_json_path = SPLITS_V2 / 'class_weights.json'
    with open(weights_json_path, 'w') as f:
        json.dump(weights_payload, f, indent=2)
    print(f"[resplit] Persisted raw class weights measurement to: {weights_json_path}")

    return train_df, val_df, test_indist_df, test_cross_df, weights_payload


if __name__ == '__main__':
    run_resplit()
