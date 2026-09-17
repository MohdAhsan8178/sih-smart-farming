#!/usr/bin/env python3
"""
scripts/repro_model_b_energy.py

Reproduce and verify Model B energy computation, overflow diagnosis, and OOD rejection.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import logsumexp
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "splits" / "model_b_manifest_v4.csv"
PATCHES_DIR = ROOT / "data" / "processed" / "model_b_patches_v4"
CKPT_PATH = ROOT / "artifacts" / "checkpoints" / "model_b_best.pt"
if not CKPT_PATH.exists():
    CKPT_PATH = ROOT / "kaggle" / "sih_model_b_kernel" / "output" / "best_model_b.pt"

print(f"=== REPRODUCING MODEL B ENERGY COMPUTATION ===")
print(f"Manifest path: {MANIFEST_PATH}")
print(f"Checkpoint   : {CKPT_PATH}")

df_master = pd.read_csv(MANIFEST_PATH)
val_df = df_master[df_master["split"] == "val"].reset_index(drop=True)
ood_df = df_master[df_master["split"] == "openset_eval"].reset_index(drop=True)

print(f"Val rows: {len(val_df)} | OOD rows: {len(ood_df)}")

class PatchDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = PATCHES_DIR / row["class_name"] / f"{row['crop_id']}.png"
        img = np.array(Image.open(p).convert("RGB"))
        tensor = self.transform(image=img)["image"]
        return tensor, int(row["class_idx"])

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Inference device: {device}")

model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=3)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

def compute_logits(df):
    loader = DataLoader(PatchDataset(df, transform), batch_size=128, shuffle=False)
    logits_list = []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))
            logits_list.append(out.cpu().numpy())
    return np.concatenate(logits_list, axis=0)

val_logits = compute_logits(val_df)
ood_logits = compute_logits(ood_df)

print("\n--- 1. LOGIT DISTRIBUTIONS ---")
for name, l in [("Validation (In-Distribution)", val_logits), ("Hard OOD (Openset Eval)", ood_logits)]:
    max_l = np.max(l, axis=1)
    print(f"{name}:")
    print(f"  dtype       : {l.dtype}")
    print(f"  shape       : {l.shape}")
    print(f"  min / max   : {l.min():.4f} / {l.max():.4f}")
    print(f"  mean        : {l.mean():.4f}")
    print(f"  max-logit p1/p50/p99: {np.percentile(max_l, 1):.4f} / {np.percentile(max_l, 50):.4f} / {np.percentile(max_l, 99):.4f}")

T_cal = 0.9997
print(f"\nUsing Temperature T_cal = {T_cal:.4f}")

print("\n--- 2. NAIVE FORMULA VS STABLE FORMULA (OVERFLOW AUDIT) ---")
# Test Naive formula on Val
naive_warnings_val = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    id_energy_naive = -T_cal * np.log(np.sum(np.exp(val_logits / T_cal), axis=1))
    naive_warnings_val = [str(item.message) for item in w]

val_naive_neginf = np.isneginf(id_energy_naive).sum()
val_naive_nan = np.isnan(id_energy_naive).sum()

print("Val Naive Formula:")
print(f"  Warnings caught: {naive_warnings_val}")
print(f"  -inf count     : {val_naive_neginf} / {len(id_energy_naive)} ({val_naive_neginf/len(id_energy_naive)*100:.2f}%)")
print(f"  NaN count      : {val_naive_nan}")

# Test Naive formula on OOD
naive_warnings_ood = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ood_energy_naive = -T_cal * np.log(np.sum(np.exp(ood_logits / T_cal), axis=1))
    naive_warnings_ood = [str(item.message) for item in w]

ood_naive_neginf = np.isneginf(ood_energy_naive).sum()
ood_naive_nan = np.isnan(ood_energy_naive).sum()

print("OOD Naive Formula:")
print(f"  Warnings caught: {naive_warnings_ood}")
print(f"  -inf count     : {ood_naive_neginf} / {len(ood_energy_naive)} ({ood_naive_neginf/len(ood_energy_naive)*100:.2f}%)")
print(f"  NaN count      : {ood_naive_nan}")

# Test Stable formula on Val and OOD
stable_warnings = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    id_energy_stable = -T_cal * logsumexp(val_logits / T_cal, axis=1)
    ood_energy_stable = -T_cal * logsumexp(ood_logits / T_cal, axis=1)
    stable_warnings = [str(item.message) for item in w]

val_stable_neginf = np.isneginf(id_energy_stable).sum()
ood_stable_neginf = np.isneginf(ood_energy_stable).sum()

print("Stable logsumexp Formula:")
print(f"  Warnings caught: {stable_warnings} (Zero warnings)")
print(f"  Val -inf count : {val_stable_neginf} / {len(id_energy_stable)}")
print(f"  OOD -inf count : {ood_stable_neginf} / {len(ood_energy_stable)}")
print(f"  Val energy min / max: {id_energy_stable.min():.4f} / {id_energy_stable.max():.4f}")
print(f"  OOD energy min / max: {ood_energy_stable.min():.4f} / {ood_energy_stable.max():.4f}")

print("\n--- 3. THRESHOLD & OOD REJECTION SUMMARY ---")
tau_energy_stable = float(np.percentile(id_energy_stable, 95.0))
tau_energy_naive = float(np.percentile(id_energy_naive, 95.0))
print(f"Fitted TAU_ENERGY (Stable logsumexp 95th pct) : {tau_energy_stable:.6f}")
print(f"Fitted TAU_ENERGY (Naive formula 95th pct)    : {tau_energy_naive:.6f}")

ood_rej_stable_count = int((ood_energy_stable > tau_energy_stable).sum())
ood_rej_stable_rate = (ood_rej_stable_count / len(ood_energy_stable)) * 100.0

ood_rej_naive_count = int((ood_energy_naive > tau_energy_naive).sum())
ood_rej_naive_rate = (ood_rej_naive_count / len(ood_energy_naive)) * 100.0

print(f"Stable OOD Rejection: {ood_rej_stable_count} / {len(ood_energy_stable)} ({ood_rej_stable_rate:.2f}%)")
print(f"Naive OOD Rejection : {ood_rej_naive_count} / {len(ood_energy_naive)} ({ood_rej_naive_rate:.2f}%)")
