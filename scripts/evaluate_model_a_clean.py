#!/usr/bin/env python3
"""
scripts/evaluate_model_a_clean.py

E3: Re-evaluates Model A on cleaned test_indist (without train duplicates).
E4: Re-evaluates Model A on cleaned val (without train duplicates),
    refits T_cal and tau_energy on cleaned val (report only).
E5: Generates per-class recall table for test_sourceheldout (9 classes).
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.paths import CKPT, ROOT as DATA_ROOT
from configs.train_config import IMAGE_SIZE
from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval, compute_split_metrics
from train.transforms import eval_transform

AUDIT_DIR = ROOT / "artifacts" / "audit" / "2026-09-16"
LEAKAGE_CSV = AUDIT_DIR / "model_a_cross_split_leakage.csv"
CKPT_PATH = ROOT / "artifacts" / "checkpoints" / "v3" / "stage1.pt"

class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

class EvalDataset(Dataset):
    def __init__(self, df, transform):
        self.transform = transform
        self.samples = []
        for _, row in df.iterrows():
            p = ROOT / row["path"]
            if p.exists() and row["label"] in class_to_idx:
                self.samples.append((str(p), class_to_idx[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, target = self.samples[idx]
        with Image.open(path_str) as img:
            rgb = np.array(img.convert("RGB"))
        tensor = self.transform(image=rgb)["image"]
        return tensor, target

def run_eval(model, df, tf, device, desc):
    ds = EvalDataset(df, tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_preds, all_targets = [], []
    all_logits = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_logits.append(logits.cpu().numpy())
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
            
    top1, macro_f1, micro_f1, per_class, cm = compute_split_metrics(
        all_targets=all_targets,
        all_preds=all_preds,
        class_names=CLASS_NAMES,
    )
    logits_cat = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, NUM_CLASSES))
    return top1, macro_f1, micro_f1, per_class, cm, logits_cat, np.array(all_targets)

def main():
    print("=================================================================")
    print("PART E3 - E5: MODEL A RIGOROUS CLEAN EVALUATION & RE-CALIBRATION")
    print("=================================================================")

    # Load leakage mapping
    leak_df = pd.read_csv(LEAKAGE_CSV)
    test_leaked_paths = set(leak_df[leak_df["eval_split"] == "test_indist"]["eval_path"].unique())
    val_leaked_paths = set(leak_df[leak_df["eval_split"] == "val"]["eval_path"].unique())

    print(f"Loaded leakage records from {LEAKAGE_CSV}:")
    print(f"  test_indist leaked image count : {len(test_leaked_paths):,}")
    print(f"  val leaked image count         : {len(val_leaked_paths):,}")

    # Load splits
    test_indist_df = pd.read_csv(ROOT / "splits_v3" / "test_indist.csv")
    val_df = pd.read_csv(ROOT / "splits_v3" / "val.csv")
    test_sh_df = pd.read_csv(ROOT / "splits_v3" / "test_sourceheldout.csv")

    clean_test_df = test_indist_df[~test_indist_df["path"].isin(test_leaked_paths)].reset_index(drop=True)
    clean_val_df = val_df[~val_df["path"].isin(val_leaked_paths)].reset_index(drop=True)

    device = torch.device("cpu")
    print(f"\nBuilding Model A (tf_efficientnet_lite0, num_classes=29)...")
    raw_model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model, weights_used = load_checkpoint_for_eval(
        checkpoint_path=CKPT_PATH,
        model=raw_model,
        eval_weights="ema",
        device=device
    )
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint {CKPT_PATH} (weights used: {weights_used})")

    tf = eval_transform(size=IMAGE_SIZE)

    # ----------------------------------------------------------------------
    # E3: Re-evaluate on Original vs Cleaned test_indist
    # ----------------------------------------------------------------------
    print("\n--- E3. IN-DISTRIBUTION TEST SET: ORIGINAL vs CLEANED ---")
    top1_orig, f1_orig, _, pc_orig, _, l_orig, y_orig = run_eval(model, test_indist_df, tf, device, "Original test_indist")
    top1_clean, f1_clean, _, pc_clean, _, l_clean, y_clean = run_eval(model, clean_test_df, tf, device, "Cleaned test_indist")

    print(f"Original test_indist (N = {len(y_orig)}):")
    print(f"  Top-1 Accuracy : {top1_orig:.4f} ({top1_orig*100:.2f}%)")
    print(f"  Macro-F1       : {f1_orig:.4f} ({f1_orig*100:.2f}%)")

    print(f"\nCleaned test_indist (N = {len(y_clean)}, {len(test_leaked_paths)} duplicates removed):")
    print(f"  Top-1 Accuracy : {top1_clean:.4f} ({top1_clean*100:.2f}%)")
    print(f"  Macro-F1       : {f1_clean:.4f} ({f1_clean*100:.2f}%)")

    print(f"\nLeakage Inflation Delta on test_indist:")
    print(f"  Top-1 Accuracy Drop : {(top1_clean - top1_orig)*100:+.2f} percentage points")
    print(f"  Macro-F1 Drop       : {(f1_clean - f1_orig)*100:+.2f} percentage points")

    # ----------------------------------------------------------------------
    # E4: Check Val Leakage, Refit T_cal and tau_energy on Cleaned Val
    # ----------------------------------------------------------------------
    print("\n--- E4. VALIDATION SET LEAKAGE & RE-CALIBRATION ---")
    top1_v_orig, f1_v_orig, _, _, _, l_v_orig, y_v_orig = run_eval(model, val_df, tf, device, "Original val")
    top1_v_clean, f1_v_clean, _, _, _, l_v_clean, y_v_clean = run_eval(model, clean_val_df, tf, device, "Cleaned val")

    print(f"Original val (N = {len(y_v_orig)}):")
    print(f"  Top-1 Accuracy : {top1_v_orig:.4f} ({top1_v_orig*100:.2f}%)")
    print(f"  Macro-F1       : {f1_v_orig:.4f} ({f1_v_orig*100:.2f}%)")

    print(f"\nCleaned val (N = {len(y_v_clean)}, {len(val_leaked_paths)} duplicates removed):")
    print(f"  Top-1 Accuracy : {top1_v_clean:.4f} ({top1_v_clean*100:.2f}%)")
    print(f"  Macro-F1       : {f1_v_clean:.4f} ({f1_v_clean*100:.2f}%)")

    # Refit T_cal via LBFGS on cleaned val logits
    logits_t = torch.tensor(l_v_clean, dtype=torch.float32)
    targets_t = torch.tensor(y_v_clean, dtype=torch.long)

    temperature = nn.Parameter(torch.ones(1) * 0.597)
    opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    crit = nn.CrossEntropyLoss()

    def eval_t():
        opt.zero_grad()
        loss = crit(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    opt.step(eval_t)
    refit_T_val = float(temperature.item())

    nll_orig_t = float(crit(logits_t / 0.597, targets_t).item())
    nll_refit_t = float(crit(logits_t / refit_T_val, targets_t).item())

    # Refit tau_energy on cleaned val (RAW logits, crop classes only)
    crop_cols = [i for i in range(NUM_CLASSES) if CLASS_NAMES[i] != "not_crop"]
    z_raw = l_v_clean[:, crop_cols]
    m_raw = z_raw.max(axis=1, keepdims=True)
    energy_clean_val = -1.0 * (m_raw.squeeze(1) + np.log(np.exp(z_raw - m_raw).sum(axis=1)))
    refit_tau_energy = float(np.percentile(energy_clean_val, 95.0))

    print(f"\nRe-Calibration Comparison (Cleaned Val Set):")
    print(f"  T_CAL (Original)        : 0.5970 (NLL on clean val: {nll_orig_t:.6f})")
    print(f"  T_CAL (Refitted on clean): {refit_T_val:.4f} (NLL on clean val: {nll_refit_t:.6f})")
    print(f"  TAU_ENERGY (Original)   : -2.8529")
    print(f"  TAU_ENERGY (Refitted)   : {refit_tau_energy:.4f}")

    # ----------------------------------------------------------------------
    # E5: Per-Class Recall Table for test_sourceheldout (9 Classes)
    # ----------------------------------------------------------------------
    print("\n--- E5. PER-CLASS RECALL TABLE FOR test_sourceheldout (9 CLASSES) ---")
    top1_sh, f1_sh, micro_sh, pc_sh, cm_sh, l_sh, y_sh = run_eval(model, test_sh_df, tf, device, "test_sourceheldout")

    print(f"Overall test_sourceheldout Performance (N = {len(y_sh)}):")
    print(f"  Top-1 Accuracy : {top1_sh:.4f} ({top1_sh*100:.2f}%)")
    print(f"  Macro-F1       : {f1_sh:.4f} ({f1_sh*100:.2f}%)")

    active_classes = [c for c in CLASS_NAMES if pc_sh.get(c, {}).get("support", 0) > 0]
    print(f"\nActive Classes in test_sourceheldout ({len(active_classes)} classes):")
    print(f"{'Class Name':<32} | {'Support':<8} | {'Precision':<10} | {'Recall (%)':<12} | {'F1-Score':<10} | {'Diagnosis'}")
    print("-" * 92)

    for c in sorted(active_classes):
        m = pc_sh[c]
        supp = m["support"]
        prec = m["precision"]
        rec = m["recall"]
        f1 = m["f1"]
        
        prec_str = f"{prec:.4f}" if prec is not None else "N/A"
        rec_str = f"{rec*100:.2f}%" if rec is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        
        if rec is None or rec < 0.05:
            diag = "TOTAL COLLAPSE (<5%)"
        elif rec < 0.50:
            diag = "DEGRADED (<50%)"
        else:
            diag = "ROBUST (>=50%)"
            
        print(f"{c:<32} | {supp:<8} | {prec_str:<10} | {rec_str:<12} | {f1_str:<10} | {diag}")

if __name__ == "__main__":
    main()
