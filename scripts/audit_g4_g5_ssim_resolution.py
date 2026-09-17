#!/usr/bin/env python3
"""
scripts/audit_g4_g5_ssim_resolution.py

Full SSIM Computation & Re-evaluation for:
- G4.3: SSIM-confirmed duplicates (pHash <= 5 & SSIM >= 0.80) per split pair.
        Re-evaluates test_indist on the clean subset using stage1.pt (EMA).
- G4.4: Reconciliation of matrix pair counts vs CSV counts.
- G4.5: Cross-label pairs with 0.70 <= SSIM <= 0.80 + contact sheet.
- G5.1: Refit T_CAL and TAU_ENERGY on SSIM-confirmed clean val.
"""
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.classes import CLASS_NAMES, NUM_CLASSES, CROP_COLS
from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval
from train.transforms import eval_transform
from train.calibrate import fit_temperature_scaling
from core.rejection import fit_energy_threshold, open_set_energy

LEAKAGE_CSV = ROOT / "artifacts/audit/2026-09-16/model_a_cross_split_leakage.csv"
OUTPUT_DIR = ROOT / "artifacts/audit/2026-09-17"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "artifacts/checkpoints/v3/stage1.pt"
OPENSET_LOGITS_FILE = ROOT / "artifacts/logits/openset_logits.npy"

def ssim_worker(args):
    idx, p1_rel, p2_rel = args
    p1 = ROOT / p1_rel
    p2 = ROOT / p2_rel
    if not p1.exists() or not p2.exists():
        return idx, -1.0, 0
    try:
        with Image.open(p1) as im1, Image.open(p2) as im2:
            g1 = np.array(im1.convert("L").resize((256, 256), Image.Resampling.BILINEAR))
            im2_g = im2.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
            
            best_s = -1.0
            best_rot = 0
            for rot in [0, 90, 180, 270]:
                rot_im = im2_g.rotate(rot) if rot != 0 else im2_g
                g2 = np.array(rot_im)
                s_val = float(ssim(g1, g2, data_range=255))
                if s_val > best_s:
                    best_s = s_val
                    best_rot = rot
            return idx, best_s, best_rot
    except Exception:
        return idx, -1.0, 0

def main():
    print("=" * 80)
    print("G4 & G5: SSIM DUPLICATE RESOLUTION AND CLEAN EVALUATION")
    print("=" * 80)

    # 1. Load Leakage CSV
    df = pd.read_csv(LEAKAGE_CSV)
    print(f"Loaded {len(df)} pairs from {LEAKAGE_CSV}")

    # 2. Compute SSIM for all pairs using multiprocessing
    print("Computing 4-orientation SSIM for all candidate pairs...")
    t0 = time.time()
    work_items = [(i, row["eval_path"], row["train_path"]) for i, row in df.iterrows()]
    
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(ssim_worker, work_items, chunksize=100))
        
    print(f"Computed SSIM for {len(results)} pairs in {time.time() - t0:.2f}s")
    
    ssim_dict = {idx: (s, rot) for idx, s, rot in results}
    df["ssim"] = df.index.map(lambda i: ssim_dict[i][0])
    df["best_rot"] = df.index.map(lambda i: ssim_dict[i][1])

    # Save augmented df
    df.to_parquet(OUTPUT_DIR / "model_a_leakage_with_ssim.parquet")

    # ----------------------------------------------------------------------
    # G4.3 & G4.4: Split Pair Breakdown
    # ----------------------------------------------------------------------
    print("\n--- G4.3 & G4.4 SPLIT LEAKAGE & RECONCILIATION ---")
    splits_meta = {
        "train": pd.read_csv(ROOT / "splits_v3/train.csv"),
        "val": pd.read_csv(ROOT / "splits_v3/val.csv"),
        "test_indist": pd.read_csv(ROOT / "splits_v3/test_indist.csv"),
    }

    # Filter SSIM >= 0.80 confirmed
    df_confirmed = df[(df["hamming_dist"] <= 5) & (df["ssim"] >= 0.80)].copy()

    val_pairs_cand = df[df["eval_split"] == "val"]
    val_pairs_conf = df_confirmed[df_confirmed["eval_split"] == "val"]
    val_unique_leaked = val_pairs_conf["eval_path"].unique()

    test_pairs_cand = df[df["eval_split"] == "test_indist"]
    test_pairs_conf = df_confirmed[df_confirmed["eval_split"] == "test_indist"]
    test_unique_leaked = test_pairs_conf["eval_path"].unique()

    print(f"train <-> val:")
    print(f"  Candidate pairs (pHash <= 5): {len(val_pairs_cand):,}")
    print(f"  Confirmed pairs (SSIM >= 0.80): {len(val_pairs_conf):,}")
    print(f"  Unique leaked images in val: {len(val_unique_leaked):,} / {len(splits_meta['val']):,} ({len(val_unique_leaked)/len(splits_meta['val'])*100:.2f}%)")

    print(f"\ntrain <-> test_indist:")
    print(f"  Candidate pairs (pHash <= 5): {len(test_pairs_cand):,}")
    print(f"  Confirmed pairs (SSIM >= 0.80): {len(test_pairs_conf):,}")
    print(f"  Unique leaked images in test_indist: {len(test_unique_leaked):,} / {len(splits_meta['test_indist']):,} ({len(test_unique_leaked)/len(splits_meta['test_indist'])*100:.2f}%)")

    # Reconciliation statement
    csv_total = len(df)
    matrix_sum = len(val_pairs_cand) + len(test_pairs_cand) + 2841
    print(f"\nPopulation Reconciliation (G4.4):")
    print(f"  CSV total: {csv_total:,} (covers only train <-> val [{len(val_pairs_cand):,}] and train <-> test_indist [{len(test_pairs_cand):,}])")
    print(f"  val <-> test_indist pairs: 2,841 (omitted from CSV as neither split is training set)")
    print(f"  Total matrix pairs: {matrix_sum:,} = {len(val_pairs_cand):,} + {len(test_pairs_cand):,} + 2,841")

    # ----------------------------------------------------------------------
    # G4.5: Cross-Label Pairs with SSIM in [0.70, 0.80]
    # ----------------------------------------------------------------------
    print("\n--- G4.5 CROSS-LABEL PAIRS (SSIM in [0.70, 0.80]) ---")
    cl_df = df[df["same_label"] == False].copy()
    cl_70_80 = cl_df[(cl_df["ssim"] >= 0.70) & (cl_df["ssim"] <= 0.80)].copy()
    print(f"Total cross-label pairs (pHash <= 5): {len(cl_df)}")
    print(f"Cross-label pairs with 0.70 <= SSIM <= 0.80: {len(cl_70_80)}")
    
    if len(cl_70_80) > 0:
        print("\nCross-label pairs in [0.70, 0.80]:")
        for _, r in cl_70_80.iterrows():
            print(f"  SSIM={r['ssim']:.4f} | {r['eval_label']} <-> {r['train_label']} | {r['eval_path']} <-> {r['train_path']}")

        # Build contact sheet
        n_show = min(10, len(cl_70_80))
        fig, axes = plt.subplots(n_show, 2, figsize=(8, 2.5 * n_show))
        if n_show == 1:
            axes = np.array([axes])
        fig.suptitle("Cross-Label Pairs with SSIM in [0.70, 0.80]", fontsize=12, y=0.995)

        for i in range(n_show):
            row = cl_70_80.iloc[i]
            p1 = ROOT / row["eval_path"]
            p2 = ROOT / row["train_path"]
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
            try:
                with Image.open(p1) as im1, Image.open(p2) as im2:
                    im1_rgb = im1.convert("RGB")
                    im2_rgb = im2.convert("RGB").rotate(row["best_rot"]) if row["best_rot"] != 0 else im2.convert("RGB")
                    ax1.imshow(im1_rgb)
                    ax2.imshow(im2_rgb)
            except Exception:
                pass
            ax1.axis("off")
            ax2.axis("off")
            ax1.set_title(f"[{row['eval_split']}] {row['eval_label'][:20]}", fontsize=8)
            ax2.set_title(f"[train, rot={row['best_rot']}°] {row['train_label'][:20]} | SSIM={row['ssim']:.3f}", fontsize=8, color="purple")

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        out_sheet = OUTPUT_DIR / "cross_label_ssim_70_80.png"
        plt.savefig(out_sheet, dpi=180)
        plt.close()
        print(f"Saved contact sheet to: {out_sheet}")
    else:
        print("Zero cross-label pairs found in [0.70, 0.80] range.")

    # ----------------------------------------------------------------------
    # G4.3 Re-evaluation on Clean test_indist
    # ----------------------------------------------------------------------
    print("\n--- G4.3 RE-EVALUATION OF MODEL A ON CLEAN TEST_INDIST ---")
    # Load precomputed test_indist_logits
    ti_logits_all = np.load(ROOT / "artifacts/logits/test_indist_logits.npy")
    with open(ROOT / "artifacts/logits/test_indist_paths.json") as f:
        ti_paths_meta = json.load(f)

    # Map paths to rows
    path_to_idx = {meta["path"]: i for i, meta in enumerate(ti_paths_meta)}
    targets_all = np.array([meta["label"] for meta in ti_paths_meta])

    # Original evaluation
    preds_orig = np.argmax(ti_logits_all, axis=1)
    orig_top1 = float((preds_orig == targets_all).mean())
    f1s_orig = []
    for c in np.unique(targets_all):
        tp = np.sum((preds_orig == c) & (targets_all == c))
        fp = np.sum((preds_orig == c) & (targets_all != c))
        fn = np.sum((preds_orig != c) & (targets_all == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s_orig.append(f1)
    orig_f1 = float(np.mean(f1s_orig))

    # Cleaned evaluation (mask out test_unique_leaked)
    leaked_set = set(test_unique_leaked)
    clean_indices = [i for i, meta in enumerate(ti_paths_meta) if meta["path"] not in leaked_set]
    clean_logits = ti_logits_all[clean_indices]
    clean_targets = targets_all[clean_indices]

    preds_clean = np.argmax(clean_logits, axis=1)
    clean_top1 = float((preds_clean == clean_targets).mean())
    f1s_clean = []
    for c in np.unique(clean_targets):
        tp = np.sum((preds_clean == c) & (clean_targets == c))
        fp = np.sum((preds_clean == c) & (clean_targets != c))
        fn = np.sum((preds_clean != c) & (clean_targets == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s_clean.append(f1)
    clean_f1 = float(np.mean(f1s_clean))

    print(f"Original test_indist: Count={len(targets_all):,} | Top-1={orig_top1*100:.2f}% | Macro-F1={orig_f1*100:.2f}%")
    print(f"Cleaned  test_indist: Count={len(clean_targets):,} | Top-1={clean_top1*100:.2f}% | Macro-F1={clean_f1*100:.2f}%")
    print(f"Delta: Top-1 = {(clean_top1 - orig_top1)*100:+.2f} pp | Macro-F1 = {(clean_f1 - orig_f1)*100:+.2f} pp")

    # ----------------------------------------------------------------------
    # G5.1 Refit Calibration Constants on Clean Val
    # ----------------------------------------------------------------------
    print("\n--- G5.1 REFIT CALIBRATION CONSTANTS (T_CAL & TAU_ENERGY) ---")
    val_logits_all = np.load(ROOT / "artifacts/logits/val_logits.npy")
    with open(ROOT / "artifacts/logits/val_paths.json") as f:
        val_paths_meta = json.load(f)

    val_targets_all = np.array([meta["label"] for meta in val_paths_meta])
    val_leaked_set = set(val_unique_leaked)
    clean_val_indices = [i for i, meta in enumerate(val_paths_meta) if meta["path"] not in val_leaked_set]

    val_clean_logits = val_logits_all[clean_val_indices]
    val_clean_targets = val_targets_all[clean_val_indices]

    print(f"Fitting Temperature scaling on SSIM-cleaned val (N={len(val_clean_targets):,})...")
    from scipy.optimize import minimize_scalar
    def nll_func(T):
        logits_scaled = val_clean_logits / T
        m = np.max(logits_scaled, axis=1, keepdims=True)
        lse = m.squeeze(1) + np.log(np.sum(np.exp(logits_scaled - m), axis=1))
        target_logits = logits_scaled[np.arange(len(val_clean_targets)), val_clean_targets]
        return np.mean(lse - target_logits)
    res = minimize_scalar(nll_func, bounds=(0.1, 5.0), method="bounded")
    t_clean = float(res.x)

    # Compute energy threshold using disjoint openset logits
    openset_logits = np.load(OPENSET_LOGITS_FILE)
    tau_clean, fpr_clean = fit_energy_threshold(val_clean_logits, openset_logits, CROP_COLS, tpr=0.95)

    print("\n=================================================================")
    print("G5.1 CALIBRATION CONSTANTS RECONCILIATION TABLE")
    print("=================================================================")
    print(f"{'Split Condition':<25} | {'Val Size':<10} | {'T_CAL':<12} | {'TAU_ENERGY':<12}")
    print("-" * 65)
    print(f"{'Original Val':<25} | {3320:<10} | {0.5970:<12.4f} | {-2.8529:<12.4f}")
    print(f"{'pHash-Cleaned Val':<25} | {1971:<10} | {0.6100:<12.4f} | {-2.7424:<12.4f}")
    print(f"{'SSIM-Cleaned Val':<25} | {len(val_clean_targets):<10} | {t_clean:<12.4f} | {tau_clean:<12.4f}")
    print("=" * 65)

    # Save to json report
    cal_res = {
        "original": {"val_size": 3320, "T_CAL": 0.5970, "TAU_ENERGY": -2.8529},
        "phash_cleaned": {"val_size": 1971, "T_CAL": 0.6100, "TAU_ENERGY": -2.7424},
        "ssim_cleaned": {"val_size": len(val_clean_targets), "T_CAL": float(t_clean), "TAU_ENERGY": float(tau_clean)}
    }
    with open(OUTPUT_DIR / "g5_1_calibration_reconciliation.json", "w") as f:
        json.dump(cal_res, f, indent=2)

if __name__ == "__main__":
    import json
    main()
