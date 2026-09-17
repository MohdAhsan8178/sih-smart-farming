#!/usr/bin/env python3
"""
scripts/audit_e1_citations.py

E1: Cat JSON keys and sed lines for all Task 3 claims.
Extracts exact lines and JSON values from:
1. eval_indist.json (Headline in-distribution accuracy & macro-F1)
2. eval_crossdomain.json (Source-held-out macro-F1 and generalization gap)
3. configs/train_config.py (T_CAL, TAU_ENERGY, TAU_CONF, TAU_PRIOR)
4. splits_v3/class_weights.json & configs/train_config.py (Class weighting formula)
5. splits_v3/openset_categories.csv & artifacts/reports/ood_metrics.json (Open-set held-out data)
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

print("=================================================================")
print("PART E1: MODEL A PROVENANCE & AUDIT CITATIONS")
print("=================================================================")

# 1. Headline indist test performance
print("\n--- 1. CLAIMED HEADLINE PERFORMANCE (test_indist) ---")
indist_path = ROOT / "artifacts" / "reports" / "eval_indist.json"
if indist_path.exists():
    with open(indist_path) as f:
        d = json.load(f)
    print(f"File: {indist_path.relative_to(ROOT)}")
    print(f"  top1_accuracy : {d.get('top1_accuracy')} ({d.get('top1_accuracy')*100:.2f}%)")
    print(f"  macro_f1      : {d.get('macro_f1')} ({d.get('macro_f1')*100:.2f}%)")
    print(f"  rows_evaluated: {d.get('rows_evaluated')}")
    print(f"  checkpoint    : {d.get('checkpoint')} (weights: {d.get('weights_used')})")

# 2. Source-held-out test performance
print("\n--- 2. SOURCE-HELDOUT PERFORMANCE & COLLAPSE GAP (test_sourceheldout) ---")
cross_path = ROOT / "artifacts" / "reports" / "eval_crossdomain.json"
if cross_path.exists():
    with open(cross_path) as f:
        d = json.load(f)
    gap = d.get("HEADLINE_GENERALIZATION_GAP", {})
    sh = d.get("source_heldout_evaluation", {})
    print(f"File: {cross_path.relative_to(ROOT)}")
    print(f"  indist_macro_f1         : {gap.get('indist_macro_f1')} ({gap.get('indist_macro_f1')*100:.2f}%)")
    print(f"  sourceheldout_macro_f1  : {gap.get('sourceheldout_macro_f1')} ({gap.get('sourceheldout_macro_f1')*100:.2f}%)")
    print(f"  macro_f1_gap_percentage_points : {gap.get('macro_f1_gap_percentage_points'):.2f} points")
    print(f"  sourceheldout_top1_acc  : {sh.get('top1_accuracy')} ({sh.get('top1_accuracy')*100:.2f}%)")
    print(f"  severity_assessment     : {gap.get('severity_assessment')}")
    print(f"  shortcut_learning_resid : {gap.get('shortcut_learning_residual')}")

# 3. Calibration parameters in train_config.py
print("\n--- 3. CALIBRATION CONSTANTS (configs/train_config.py) ---")
train_cfg_path = ROOT / "configs" / "train_config.py"
print(f"File: {train_cfg_path.relative_to(ROOT)} lines 25-33:")
res = subprocess.run(["sed", "-n", "25,33p", str(train_cfg_path)], capture_output=True, text=True)
print(res.stdout)

# 4. Open-set rejection calibration
print("\n--- 4. OPEN-SET EVALUATION & METRICS (ood_metrics.json) ---")
ood_path = ROOT / "artifacts" / "reports" / "ood_metrics.json"
if ood_path.exists():
    with open(ood_path) as f:
        d = json.load(f)
    print(f"File: {ood_path.relative_to(ROOT)}")
    for k, v in d.items():
        print(f"  {k}: {v}")

# 5. Open-set held-out categories
print("\n--- 5. OPEN-SET HELD-OUT CATEGORIES (splits_v3/openset_categories.csv) ---")
openset_cat_path = ROOT / "splits_v3" / "openset_categories.csv"
if openset_cat_path.exists():
    print(f"File: {openset_cat_path.relative_to(ROOT)}:")
    res = subprocess.run(["head", "-n", "20", str(openset_cat_path)], capture_output=True, text=True)
    print(res.stdout)

# 6. Class weights formula and values
print("\n--- 6. CLASS WEIGHTS (splits_v3/class_weights.json) ---")
weights_path = ROOT / "splits_v3" / "class_weights.json"
if weights_path.exists():
    with open(weights_path) as f:
        d = json.load(f)
    weights = d.get("raw_inverse_weights", {})
    for k in list(weights.keys())[:5]:
        print(f"  {k:<35}: weight={weights[k]:.4f}")
