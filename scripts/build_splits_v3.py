#!/usr/bin/env python3
"""
scripts/build_splits_v3.py

Builds Phase 3 splits (splits_v3) according to the corrected held-out rule:
- Classes with >= 3 sources: exactly 1 entire source held out in test_sourceheldout.csv.
  Train keeps >= 2 sources.
- Classes with exactly 2 sources: BOTH sources stay in train. Not source-measurable.
- Classes with 1 source: unchanged, no held-out. Not source-measurable.
"""

import os
import sys
import json
import filecmp
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.classes import CLASS_NAMES

OUT_DIR = Path("splits_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== 1. ASSEMBLING V3 ALL_IMAGES MANIFEST ===")
df_v2 = pd.read_csv("splits_v2/all_images.csv")
print(f"Loaded splits_v2/all_images.csv: {len(df_v2)} rows")

rename_df = pd.read_csv("data/packaged_min/splits_packaged/rename_map.csv")
new_entries = rename_df.tail(6021)
tax_entries = new_entries[~new_entries["new_relative_path"].str.startswith("openset/")].copy()
print(f"New taxonomy entries from rename_map: {len(tax_entries)}")

def get_meta(rel_path):
    parts = rel_path.split("/")
    src, folder = parts[0], parts[1]
    mapping = {
        ("banglariceleaf", "bacterial_leaf_blight"): ("rice__bacterial_leaf_blight", "banglariceleaf", "bacterial_leaf_blight"),
        ("banglariceleaf", "bacterial_leaf_streak"): ("rice__bacterial_leaf_streak", "banglariceleaf", "bacterial_leaf_streak"),
        ("banglariceleaf", "leaf_blast"): ("rice__blast", "banglariceleaf", "leaf_blast"),
        ("banglariceleaf", "normal"): ("rice__normal", "banglariceleaf", "normal"),
        ("dhan_shomadhan", "blast"): ("rice__blast", "dhan_shomadhan", "blast"),
        ("dhan_shomadhan", "brown_spot"): ("rice__brown_spot", "dhan_shomadhan", "brown_spot"),
        ("dhan_shomadhan", "tungro"): ("rice__tungro", "dhan_shomadhan", "tungro"),
        ("rice_mendeley", "bacterial_leaf_blight"): ("rice__bacterial_leaf_blight", "rice_mendeley", "bacterial_leaf_blight"),
        ("rice_mendeley", "brown_spot"): ("rice__brown_spot", "rice_mendeley", "brown_spot"),
        ("rice_mendeley", "leaf_blast"): ("rice__blast", "rice_mendeley", "leaf_blast"),
        ("kushagra_wheat", "powdery_mildew"): ("wheat__powdery_mildew", "kushagra_wheat", "powdery_mildew"),
        ("sugarcane_ld_bd", "healthy"): ("sugarcane__healthy", "sugarcane_ld_bd", "healthy"),
        ("sugarcane_ld_bd", "red_rot"): ("sugarcane__red_rot", "sugarcane_ld_bd", "red_rot"),
    }
    return mapping[(src, folder)]

new_rows = []
max_g = df_v2["group_id"].max()
for idx, p in enumerate(tax_entries["new_relative_path"]):
    lbl, src, orig = get_meta(p)
    new_rows.append({
        "path": f"data/raw/{p}",
        "label": lbl,
        "source_dataset": src,
        "orig_folder": orig,
        "group_id": max_g + 1 + idx
    })

df_new = pd.DataFrame(new_rows)
df_all = pd.concat([df_v2, df_new], ignore_index=True)
print(f"Total V3 manifest images: {len(df_all)} (groups: {df_all.group_id.nunique()})")
df_all.to_csv(OUT_DIR / "all_images.csv", index=False)

print("\n=== 2. ISOLATING HELD-OUT SOURCE (test_sourceheldout.csv) ===")
# Exactly 9 classes with >= 3 sources qualifying for a held-out source
held_out_spec = {
    "rice__normal": "rice_hasan",                # 155 (smallest)
    "rice__bacterial_leaf_blight": "rice_mendeley", # 180 (smallest)
    "rice__blast": "dhan_shomadhan",             # 255 (smallest)
    "rice__brown_spot": "dhan_shomadhan",        # 133 (smallest)
    "rice__tungro": "dhan_shomadhan",            # 188 (smallest)
    "sugarcane__healthy": "sugarcane_ld_bd",     # 194 (smallest, low-res)
    "wheat__yellow_rust": "wheat_mendeley",      # 208 (smallest)
    "wheat__septoria": "wheat_mendeley",         # 97 (smallest)
    "wheat__powdery_mildew": "wheat_small",      # 161 (smallest)
}

print(f"Qualifying classes for source-heldout: {len(held_out_spec)}")

held_out_masks = []
for cls, src in held_out_spec.items():
    held_out_masks.append((df_all["label"] == cls) & (df_all["source_dataset"] == src))

is_held_out = pd.concat(held_out_masks, axis=1).any(axis=1)

df_heldout = df_all[is_held_out].copy().reset_index(drop=True)
df_train_pool = df_all[~is_held_out].copy().reset_index(drop=True)

df_heldout.to_csv(OUT_DIR / "test_sourceheldout.csv", index=False)
print(f"Saved test_sourceheldout.csv: {len(df_heldout)} images ({df_heldout.label.nunique()} classes)")

print("\n=== 3. SPLITTING REMAINING TRAINING POOL (80 / 10 / 10) ===")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, hold_idx = next(sgkf.split(df_train_pool, y=df_train_pool.label, groups=df_train_pool.group_id))
train_df = df_train_pool.iloc[train_idx].copy().reset_index(drop=True)
hold_df = df_train_pool.iloc[hold_idx].copy().reset_index(drop=True)

sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
v_idx, t_idx = next(sgkf2.split(hold_df, y=hold_df.label, groups=hold_df.group_id))
val_df = hold_df.iloc[v_idx].copy().reset_index(drop=True)
test_indist_df = hold_df.iloc[t_idx].copy().reset_index(drop=True)

train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_indist_df.to_csv(OUT_DIR / "test_indist.csv", index=False)

print(f"Saved train.csv:       {len(train_df):>5} images ({train_df.group_id.nunique()} groups, {train_df.label.nunique()} classes)")
print(f"Saved val.csv:         {len(val_df):>5} images ({val_df.group_id.nunique()} groups, {val_df.label.nunique()} classes)")
print(f"Saved test_indist.csv: {len(test_indist_df):>5} images ({test_indist_df.group_id.nunique()} groups, {test_indist_df.label.nunique()} classes)")

# Assert train count meaningfully above V2
assert len(train_df) > 24083, f"Train count {len(train_df)} is NOT above V2 (24083)!"
print(f"Train count assertion PASSED: {len(train_df)} is meaningfully above V2 (24,083) by +{len(train_df) - 24083} images.")

# Assert BLS has both sources in train
bls_train = train_df[train_df["label"] == "rice__bacterial_leaf_streak"]
bls_sources = sorted(bls_train["source_dataset"].unique().tolist())
assert bls_sources == ["banglariceleaf", "paddy_doctor"], f"BLS missing sources: {bls_sources}"
print(f"BLS train assertion PASSED: {len(bls_train)} images across both sources {bls_sources}.")

print("\n=== 4. COPYING test_external_riceblast.csv & AUXILIARIES ===")
import shutil
shutil.copy2("splits_v2/test_external_riceblast.csv", OUT_DIR / "test_external_riceblast.csv")
assert filecmp.cmp("splits_v2/test_external_riceblast.csv", OUT_DIR / "test_external_riceblast.csv", shallow=False)
print("test_external_riceblast.csv: BYTE-IDENTICAL ASSERTION PASSED.")

if Path("splits_v2/class_mapping.csv").exists():
    shutil.copy2("splits_v2/class_mapping.csv", OUT_DIR / "class_mapping.csv")
if Path("splits_v2/openset_categories.csv").exists():
    shutil.copy2("splits_v2/openset_categories.csv", OUT_DIR / "openset_categories.csv")

df_riceblast = pd.read_csv(OUT_DIR / "test_external_riceblast.csv")

print("\n=== 5. RECOMPUTING CLASS_WEIGHTS.JSON FOR splits_v3/train.csv ===")
train_counts = train_df.label.value_counts().to_dict()
n_total = len(train_df)
n_classes = len(CLASS_NAMES)
raw_weights = {}
per_class_counts = {}

for c in CLASS_NAMES:
    cnt = int(train_counts.get(c, 0))
    assert cnt > 0, f"Class {c} has zero count in train.csv!"
    per_class_counts[c] = cnt
    raw_weights[c] = float(n_total / (n_classes * cnt))

weights_payload = {
    "source_split": "splits_v3/train.csv",
    "total_samples": n_total,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "class_counts": per_class_counts,
    "raw_inverse_weights": raw_weights,
}

with open(OUT_DIR / "class_weights.json", "w") as f:
    json.dump(weights_payload, f, indent=2)
print(f"Persisted splits_v3/class_weights.json with {len(raw_weights)} classes.")

print("\n=== 6. VERIFYING LEAKAGE ACROSS ALL SPLITS ===")
all_splits = {
    "train": train_df,
    "val": val_df,
    "test_indist": test_indist_df,
    "test_sourceheldout": df_heldout,
    "test_external_riceblast": df_riceblast,
}

# Group ID leakage check
print("--- GROUP ID LEAKAGE CHECK ---")
split_names = list(all_splits.keys())
group_leaks = 0
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        s1, s2 = split_names[i], split_names[j]
        df1, df2 = all_splits[s1], all_splits[s2]
        if "group_id" in df1.columns and "group_id" in df2.columns:
            g1, g2 = set(df1.group_id), set(df2.group_id)
            overlap = len(g1 & g2)
            print(f"  {s1:<25} vs {s2:<25} group_id overlap: {overlap}")
            if overlap > 0:
                group_leaks += overlap

assert group_leaks == 0, f"FATAL: Found {group_leaks} group_id leaks!"
print("Group ID leakage assertion PASSED: Zero group_id collisions across all splits.")

# Path leakage check
print("\n--- PATH LEAKAGE CHECK ---")
path_leaks = 0
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        s1, s2 = split_names[i], split_names[j]
        p1 = set(all_splits[s1].path)
        p2 = set(all_splits[s2].path)
        overlap = len(p1 & p2)
        print(f"  {s1:<25} vs {s2:<25} path overlap: {overlap}")
        if overlap > 0:
            path_leaks += overlap

assert path_leaks == 0, f"FATAL: Found {path_leaks} path leaks!"
print("Path leakage assertion PASSED: Zero path collisions across all splits.")

print("\n=== 7. 29-ROW SUMMARY TABLE ===")
rows = []
for c in CLASS_NAMES:
    tr_sub = train_df[train_df["label"] == c]
    tr_srcs = sorted(tr_sub["source_dataset"].unique().tolist())
    tr_src_str = ", ".join(tr_srcs)
    held_src = held_out_spec.get(c, "-")
    held_cnt = len(df_heldout[df_heldout["label"] == c]) if held_src != "-" else 0
    is_measurable = "yes" if held_src != "-" else "no"
    tr_cnt = len(tr_sub)
    rows.append({
        "class": c,
        "train_sources": tr_src_str,
        "train_count": tr_cnt,
        "held_out_source": held_src,
        "held_out_count": held_cnt,
        "source_measurable": is_measurable
    })

summary_df = pd.DataFrame(rows)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 50)
print(summary_df.to_string(index=False))
