"""
Stratified group splitting into train, val, test_indist, and test_crossdomain splits.

Ensures that near-duplicate image groups (group_id) strictly never leak
across any split boundary.

Cross-Domain Split Architecture (4b):
- Rice: rice_sethy (4 classes: bacterial_leaf_blight, blast, brown_spot, tungro)
- Sugarcane: sugarcane_thite (4 overlapping classes: healthy, mosaic, rust, yellow_leaf)
- Wheat: plantwild (4 classes: yellow_rust, septoria, powdery_mildew, brown_rust)
Total 12 cross-domain evaluation classes across all 3 crops.
All taxonomy classes remain populated in the training pool.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import ROOT, SPLITS
from configs.classes import CLASS_NAMES


def compute_class_weights(
    train_df: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, object]:
    """
    Computes raw inverse-frequency class weights from a training split DataFrame.
    Enforces a strict zero-count guard: raises ValueError if any class in class_names
    has zero rows in the training split.
    """
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
        "source_split": "splits/train.csv",
        "total_samples": n_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_counts": per_class_counts,
        "raw_inverse_weights": raw_weights,
    }


def run_split():
    manifest_path = SPLITS / 'all_images.csv'
    print(f"[split] Reading {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    assert 'group_id' in df.columns, "group_id column missing! Run dedup.py first."

    # 1. Isolate the Held-Out Cross-Domain Evaluation Set (4b)
    # Rice: rice_sethy (4 classes)
    mask_rice_cross = (df.source_dataset == 'rice_sethy')

    # Sugarcane: sugarcane_thite for 4 overlapping classes
    # Any group_id that spans both a cross-domain class and a training-exclusive class in Thite
    # is kept in training to prevent cross-split group_id leakage.
    sugar_overlap_classes = ['sugarcane__healthy', 'sugarcane__mosaic', 'sugarcane__rust', 'sugarcane__yellow_leaf']
    mask_sugar_candidate = (df.source_dataset == 'sugarcane_thite') & (df.label.isin(sugar_overlap_classes))

    # Wheat: plantwild (4 classes)
    mask_wheat_cross = (df.source_dataset == 'plantwild')

    # Define tentative training candidate to identify groups that must remain in train
    mask_train_base = ~(mask_rice_cross | mask_sugar_candidate | mask_wheat_cross)
    train_base_groups = set(df[mask_train_base].group_id)

    # Clean sugar cross mask: only groups strictly disjoint from training pool
    clean_sugar_cross_mask = mask_sugar_candidate & (~df.group_id.isin(train_base_groups))

    mask_cross = mask_rice_cross | clean_sugar_cross_mask | mask_wheat_cross

    cross_df = df[mask_cross].copy().reset_index(drop=True)
    train_pool_df = df[~mask_cross].copy().reset_index(drop=True)

    print(f"[split] Total manifest images: {len(df)} (unique groups: {df.group_id.nunique()})")
    print(f"[split] Cross-domain held-out pool: {len(cross_df)} images (unique groups: {cross_df.group_id.nunique()})")
    print(f"[split] Training pool: {len(train_pool_df)} images (unique groups: {train_pool_df.group_id.nunique()})")

    # Verify all taxonomy classes are populated in the training pool
    train_pool_classes = set(train_pool_df.label.unique())
    all_classes = set(CLASS_NAMES)
    missing = all_classes - train_pool_classes
    assert not missing, f"FATAL: Classes missing from training pool: {missing}"
    print(f"[split] Confirmed: All {len(train_pool_classes)} classes populated in training pool.")

    # 2. Split Training Pool into Train (80%), Val (10%), and Test In-Dist (10%)
    # Using StratifiedGroupKFold on group_id to ensure zero near-duplicate leakage
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, hold_idx = next(sgkf.split(train_pool_df, y=train_pool_df.label, groups=train_pool_df.group_id))
    train_df = train_pool_df.iloc[train_idx].copy().reset_index(drop=True)
    hold_df = train_pool_df.iloc[hold_idx].copy().reset_index(drop=True)

    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    v_idx, t_idx = next(sgkf2.split(hold_df, y=hold_df.label, groups=hold_df.group_id))
    val_df = hold_df.iloc[v_idx].copy().reset_index(drop=True)
    test_indist_df = hold_df.iloc[t_idx].copy().reset_index(drop=True)

    # 3. Save All Splits
    SPLITS.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS / 'train.csv'
    val_path = SPLITS / 'val.csv'
    test_indist_path = SPLITS / 'test_indist.csv'
    test_cross_path = SPLITS / 'test_crossdomain.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_indist_df.to_csv(test_indist_path, index=False)
    cross_df.to_csv(test_cross_path, index=False)

    print(f"\n[split] Saved splits to {SPLITS}:")
    print(f"  train.csv:            {len(train_df):>5} images ({train_df.group_id.nunique():>5} unique groups, {train_df.label.nunique()} classes)")
    print(f"  val.csv:              {len(val_df):>5} images ({val_df.group_id.nunique():>5} unique groups, {val_df.label.nunique()} classes)")
    print(f"  test_indist.csv:      {len(test_indist_df):>5} images ({test_indist_df.group_id.nunique():>5} unique groups, {test_indist_df.label.nunique()} classes)")
    print(f"  test_crossdomain.csv: {len(cross_df):>5} images ({cross_df.group_id.nunique():>5} unique groups, {cross_df.label.nunique()} classes)")

    # 4. Strict Leakage Verification
    g_tr = set(train_df.group_id)
    g_va = set(val_df.group_id)
    g_te = set(test_indist_df.group_id)
    g_cr = set(cross_df.group_id)

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

    # Path leakage check
    p_tr = set(train_df.path)
    p_va = set(val_df.path)
    p_te = set(test_indist_df.path)
    p_cr = set(cross_df.path)

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

    assert leak_p_tr_va == 0 and leak_p_tr_te == 0 and leak_p_tr_cr == 0, "FATAL: Path leak in train!"
    assert leak_p_va_te == 0 and leak_p_va_cr == 0 and leak_p_te_cr == 0, "FATAL: Path leak in eval splits!"

    print("\n[split] Verification SUCCESS: Zero group_id leakage and zero path leakage across all boundaries.")

    # 5. Compute Inverse-Frequency Class Weights for train.csv
    print("\n=== PER-CLASS INVERSE-FREQUENCY WEIGHT TABLE (train.csv) ===")
    weights_payload = compute_class_weights(train_df, class_names=CLASS_NAMES)

    weights_json_path = SPLITS / "class_weights.json"
    with open(weights_json_path, "w") as f:
        json.dump(weights_payload, f, indent=2)
    print(f"[split] Persisted raw class weights measurement to: {weights_json_path}")

    # Display preview table
    raw_weights = weights_payload["raw_inverse_weights"]
    per_class_counts = weights_payload["class_counts"]
    mean_raw = np.mean(list(raw_weights.values()))
    norm_weights = {c: w / mean_raw for c, w in raw_weights.items()}

    print(f"{'Class Name':<35} | {'Train Count':<11} | {'Raw Weight':<11} | {'Norm Weight':<11}")
    print("-" * 75)
    for c in sorted(CLASS_NAMES, key=lambda x: norm_weights[x], reverse=True):
        cnt = per_class_counts[c]
        print(f"{c:<35} | {cnt:>11} | {raw_weights[c]:>11.4f} | {norm_weights[c]:>11.4f}")

if __name__ == '__main__':
    run_split()
