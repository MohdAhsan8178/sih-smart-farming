#!/usr/bin/env python3
"""
scripts/generate_cross_source_reliability.py

Generates artifacts/reports/model_a_cross_source_reliability.json
using the exact spec rule:
    TESTED_ROBUST : recall >= 0.60
    TESTED_WEAK   : 0.30 <= recall < 0.60
    TESTED_FAILED : recall < 0.30
    UNTESTED      : support == 0 (no source-heldout data)

Sourced from the reconciled stage1.pt EMA + eval_transform run
(artifacts/reports/eval_crossdomain.json) that reproduced 37.40% macro-F1.
"""
import json
from pathlib import Path
import sys

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.classes import CLASS_NAMES

EVAL_JSON = REPO_ROOT / "artifacts/reports/eval_crossdomain.json"
OUTPUT_JSON = REPO_ROOT / "artifacts/reports/model_a_cross_source_reliability.json"


def assign_tier(recall, support: int) -> str:
    if support == 0 or recall is None:
        return "UNTESTED"
    if recall >= 0.60:
        return "TESTED_ROBUST"
    elif recall >= 0.30:
        return "TESTED_WEAK"
    else:
        return "TESTED_FAILED"


def main():
    if not EVAL_JSON.exists():
        print(f"Error: {EVAL_JSON} not found!")
        sys.exit(1)

    with open(EVAL_JSON, "r") as f:
        data = json.load(f)

    heldout = data.get("source_heldout_evaluation", {})
    per_class = heldout.get("per_class", {})
    top1 = heldout.get("top1_accuracy", 0.0)
    macro_f1 = heldout.get("macro_f1", 0.0)

    print(f"Reconciled Evaluation Reference: {EVAL_JSON}")
    print(f"Checkpoint: {heldout.get('checkpoint_path')} | Weights: {heldout.get('weights_used')}")
    print(f"Top-1 Accuracy: {top1 * 100:.2f}% | Macro-F1: {macro_f1 * 100:.2f}%\n")

    print("=" * 96)
    print(f"{'Class Name':<35} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Tier':<14}")
    print("-" * 96)

    reliability_dict = {}

    for cls in CLASS_NAMES:
        metrics = per_class.get(cls, {"support": 0, "precision": None, "recall": None, "f1": None})
        support = metrics.get("support", 0)
        prec = metrics.get("precision")
        rec = metrics.get("recall")
        f1 = metrics.get("f1")

        tier = assign_tier(rec, support)

        reliability_dict[cls] = {
            "tier": tier,
            "support": support,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

        prec_str = f"{prec:.4f}" if prec is not None else "N/A"
        rec_str = f"{rec:.4f}" if rec is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"

        print(f"{cls:<35} | {support:<8} | {prec_str:<10} | {rec_str:<10} | {f1_str:<10} | {tier:<14}")

    print("=" * 96)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(reliability_dict, f, indent=2)

    print(f"\nSaved updated reliability tiers to: {OUTPUT_JSON}")

    tier_counts = {}
    for v in reliability_dict.values():
        t = v["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Tier counts: {tier_counts}")


if __name__ == "__main__":
    main()
