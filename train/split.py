"""
Stratified group splitting into train, val, and test_indist splits,
plus assembly of the independent test_crossdomain split.

Ensures that near-duplicate image groups (group_id) strictly never leak
across train, val, and test.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import ROOT, SPLITS, PLANTWILD, PLANTDOC

def run_split():
    manifest_path = SPLITS / 'all_images.csv'
    print(f"[split] Reading {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    assert 'group_id' in df.columns, "group_id column missing! Run dedup.py first."

    print(f"[split] Splitting {len(df)} images across {df.group_id.nunique()} unique groups (~80/10/10)...")
    
    # Split 1: 80% train, 20% holdout (5 folds)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, hold_idx = next(sgkf.split(df, y=df.label, groups=df.group_id))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    hold_df = df.iloc[hold_idx].copy().reset_index(drop=True)

    # Split 2: Holdout divided 50/50 into val and test_indist (2 folds -> 10% / 10%)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    v_idx, t_idx = next(sgkf2.split(hold_df, y=hold_df.label, groups=hold_df.group_id))
    val_df = hold_df.iloc[v_idx].copy().reset_index(drop=True)
    test_indist_df = hold_df.iloc[t_idx].copy().reset_index(drop=True)

    # Save in-distribution splits
    train_path = SPLITS / 'train.csv'
    val_path = SPLITS / 'val.csv'
    test_indist_path = SPLITS / 'test_indist.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_indist_df.to_csv(test_indist_path, index=False)

    print(f"[split] Train count: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"[split] Val count:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"[split] Test count:  {len(test_indist_df)} ({len(test_indist_df)/len(df):.1%})")

    # Build test_crossdomain.csv from out-of-domain PlantWild / PlantDoc targets
    cross_records = []
    # 1. PlantWild rice blast
    pw_rice_blast = PLANTWILD / 'rice blast'
    if pw_rice_blast.exists():
        for p in pw_rice_blast.glob('*'):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                cross_records.append({
                    'path': str(p.relative_to(ROOT)),
                    'label': 'rice__blast',
                    'source_dataset': 'plantwild',
                    'orig_folder': 'rice blast'
                })

    test_cross_df = pd.DataFrame(cross_records)
    test_cross_path = SPLITS / 'test_crossdomain.csv'
    test_cross_df.to_csv(test_cross_path, index=False)
    print(f"[split] Assembled {len(test_cross_df)} cross-domain test records into {test_cross_path}")

    # Validation assertions
    g_tr, g_va, g_te = set(train_df.group_id), set(val_df.group_id), set(test_indist_df.group_id)
    assert not (g_tr & g_va), "GROUP LEAK between train and val!"
    assert not (g_tr & g_te), "GROUP LEAK between train and test_indist!"
    assert not (g_va & g_te), "GROUP LEAK between val and test_indist!"

    p_tr, p_va, p_te = set(train_df.path), set(val_df.path), set(test_indist_df.path)
    assert not (p_tr & p_va), "PATH LEAK between train and val!"
    assert not (p_tr & p_te), "PATH LEAK between train and test_indist!"
    assert not (p_va & p_te), "PATH LEAK between val and test_indist!"

    print("[split] Verification: All group and path disjointness checks PASSED.")

if __name__ == '__main__':
    run_split()
