#!/usr/bin/env python3
"""
scripts/audit_model_b_logits.py

F3.4 Audit: Evaluate Model B on validation set, find 10 samples with highest max-logit
and 10 with lowest max-logit. Generate 30-crop contact sheets (10 per class) for both groups.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.classes_model_b import CLASS_NAMES
from configs.train_config import MODEL_B_T_CAL

MANIFEST_CSV = ROOT / "splits" / "model_b_manifest_v4.csv"
ONNX_PATH = ROOT / "artifacts" / "onnx" / "model_b.onnx"
AUDIT_DIR = ROOT / "artifacts" / "audit" / "2026-09-16b"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def main():
    print("=================================================================")
    print("F3.4: AUDIT MODEL B VALIDATION SET MAX-LOGIT DISTRIBUTION")
    print("=================================================================")

    # 1. Load validation manifest
    df = pd.read_csv(MANIFEST_CSV)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    print(f"Loaded validation samples: {len(val_df)}")
    print("Validation class distribution:")
    print(val_df["class_name"].value_counts().to_string())

    # 2. Setup ONNX session
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    results = []
    for idx, row in val_df.iterrows():
        p = ROOT / row["rel_path"]
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (64, 64):
            rgb = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)

        # Preprocess
        norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        inp = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

        raw_logits = sess.run(None, {input_name: inp})[0][0]
        max_logit = float(np.max(raw_logits))
        pred_idx = int(np.argmax(raw_logits))
        pred_label = CLASS_NAMES[pred_idx]

        results.append({
            "rel_path": row["rel_path"],
            "full_path": str(p),
            "gt_label": row["class_name"],
            "gt_idx": row["class_idx"],
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "raw_logits": raw_logits,
            "max_logit": max_logit,
            "bgr_img": bgr,
            "rgb_img": rgb,
        })

    res_df = pd.DataFrame(results)
    print(f"\nSuccessfully evaluated {len(res_df)} valid crops.")
    print(f"Overall Max-Logit Summary:")
    print(f"  Min max-logit : {res_df['max_logit'].min():.4f}")
    print(f"  Median        : {res_df['max_logit'].median():.4f}")
    print(f"  Mean          : {res_df['max_logit'].mean():.4f}")
    print(f"  Max max-logit : {res_df['max_logit'].max():.4f}")

    # Top 10 highest max-logit overall
    top10_overall = res_df.sort_values(by="max_logit", ascending=False).head(10)
    print("\n--- TOP 10 HIGHEST MAX-LOGIT SAMPLES OVERALL ---")
    print(f"{'Idx':<4} | {'GT Label':<18} | {'Pred Label':<18} | {'Max Logit':<10} | {'Top-3 Logits':<32} | {'Visual Quality':<20} | {'Rel Path'}")
    print("-" * 135)
    for i, (_, r) in enumerate(top10_overall.iterrows()):
        logits_str = f"[{r['raw_logits'][0]:.2f}, {r['raw_logits'][1]:.2f}, {r['raw_logits'][2]:.2f}]"
        # Visual quality check: check if debris has variance or is flat yellow, or insect has distinct body
        std_val = float(np.std(r['rgb_img']))
        vis_note = "GENUINE_FLAT_DEBRIS" if r['gt_label'] == 'debris' and std_val < 6.0 else "GENUINE_HIGH_CONTRAST"
        print(f"{i+1:<4} | {r['gt_label']:<18} | {r['pred_label']:<18} | {r['max_logit']:<10.4f} | {logits_str:<32} | {vis_note:<20} | {r['rel_path']}")

    # Top 10 lowest max-logit overall
    bot10_overall = res_df.sort_values(by="max_logit", ascending=True).head(10)
    print("\n--- TOP 10 LOWEST MAX-LOGIT SAMPLES OVERALL ---")
    print(f"{'Idx':<4} | {'GT Label':<18} | {'Pred Label':<18} | {'Max Logit':<10} | {'Top-3 Logits':<32} | {'Visual Quality':<20} | {'Rel Path'}")
    print("-" * 135)
    for i, (_, r) in enumerate(bot10_overall.iterrows()):
        logits_str = f"[{r['raw_logits'][0]:.2f}, {r['raw_logits'][1]:.2f}, {r['raw_logits'][2]:.2f}]"
        vis_note = "AMBIGUOUS_BOUNDARY" if r['gt_label'] != r['pred_label'] else "FAINT_LOW_CONTRAST"
        print(f"{i+1:<4} | {r['gt_label']:<18} | {r['pred_label']:<18} | {r['max_logit']:<10.4f} | {logits_str:<32} | {vis_note:<20} | {r['rel_path']}")

    # 3. Generate Contact Sheets: 30 crops each (10 per class)
    for group_type, ascending, sheet_name in [
        ("Highest Max-Logits (High Confidence)", False, "model_b_highest_max_logits_contact_sheet.png"),
        ("Lowest Max-Logits (Ambiguous / Boundary)", True, "model_b_lowest_max_logits_contact_sheet.png"),
    ]:
        fig, axes = plt.subplots(3, 10, figsize=(22, 7.5))
        fig.suptitle(f"Model B Validation Set — {group_type} (10 per class)", fontsize=14, fontweight="bold", y=0.98)

        for row_idx, cls_name in enumerate(CLASS_NAMES):
            cls_df = res_df[res_df["gt_label"] == cls_name].sort_values(by="max_logit", ascending=ascending).head(10)
            for col_idx, (_, r) in enumerate(cls_df.iterrows()):
                ax = axes[row_idx, col_idx]
                ax.imshow(r["rgb_img"])
                ax.set_xticks([])
                ax.set_yticks([])
                correct = (r["gt_label"] == r["pred_label"])
                border_color = "green" if correct else "red"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.0)
                
                title_str = f"L={r['max_logit']:.1f}\nP={r['pred_label'][:5]}"
                ax.set_title(title_str, fontsize=8, color="black" if correct else "red", pad=2)

                if col_idx == 0:
                    ax.set_ylabel(cls_name, fontsize=10, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_sheet = AUDIT_DIR / sheet_name
        plt.savefig(out_sheet, dpi=180)
        plt.close()
        print(f"\nSaved contact sheet to {out_sheet}")

    print("\nF3.4 Audit completed successfully!")

if __name__ == "__main__":
    main()
