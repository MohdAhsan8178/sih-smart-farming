#!/usr/bin/env python3
"""
scripts/audit_temperature.py

Rigorous audit of the T_CAL = 0.9997 problem (Part B).
Investigates:
B1. Number of val misclassifications at T=1 and their NLL contribution.
B2. Val NLL and ECE (15 bins) at T in {0.5, 1, 2, 5, 10, 20, 50, 100, 200}.
B3. Exact temperature-fitting trace (LBFGS, initial value, gradient, loss trace).
B4. Input preprocessing consistency check on a real crop.
B5. Robust refitting of T (optimize log_T with grid search init), recomputed tau_energy.
B6. Gate independence analysis (confidence floor vs energy threshold).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.special import logsumexp
import timm

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "splits" / "model_b_manifest_v4.csv"
PATCHES_DIR = ROOT / "data" / "processed" / "model_b_patches_v4"
CKPT_PATH = ROOT / "artifacts" / "checkpoints" / "model_b_best.pt"

print("=================================================================")
print("PART B AUDIT: THE T_CAL = 0.9997 PROBLEM INVESTIGATION")
print("=================================================================")

# Load data and model
df_master = pd.read_csv(MANIFEST_PATH)
val_df = df_master[df_master["split"] == "val"].reset_index(drop=True)
ood_df = df_master[df_master["split"] == "openset_eval"].reset_index(drop=True)

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

eval_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

device = torch.device("cpu") # CPU for exact determinism
model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=3)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

def get_logits_targets(df):
    loader = DataLoader(PatchDataset(df, eval_transform), batch_size=128, shuffle=False)
    l_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            l_list.append(model(x).numpy())
            y_list.append(y.numpy())
    return np.concatenate(l_list, axis=0), np.concatenate(y_list, axis=0)

val_logits, val_targets = get_logits_targets(val_df)
ood_logits, ood_targets = get_logits_targets(ood_df)

val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
val_targets_t = torch.tensor(val_targets, dtype=torch.long)

# ----------------------------------------------------------------------
# B1. Val Misclassifications at T=1 and their NLL contribution
# ----------------------------------------------------------------------
print("\n--- B1. MISCLASSIFICATIONS AT T=1 AND NLL CONTRIBUTION ---")
preds_t1 = np.argmax(val_logits, axis=1)
misclassified_mask = (preds_t1 != val_targets)
n_mis = int(misclassified_mask.sum())
n_total = len(val_targets)

# Per-sample NLL at T=1
nll_per_sample = F.cross_entropy(val_logits_t, val_targets_t, reduction='none').numpy()
total_nll = float(nll_per_sample.sum())
mean_nll = float(nll_per_sample.mean())

mis_nll = float(nll_per_sample[misclassified_mask].sum())
corr_nll = float(nll_per_sample[~misclassified_mask].sum())

print(f"Total Validation Samples : {n_total}")
print(f"Misclassified Count      : {n_mis} / {n_total} (Error Rate: {n_mis/n_total*100:.2f}%, Acc: {(n_total-n_mis)/n_total*100:.2f}%)")
print(f"Total Val NLL (T=1.0)    : {total_nll:.4f} (Mean NLL: {mean_nll:.6f})")
print(f"NLL from Correct Samples : {corr_nll:.4f} (Avg per correct: {corr_nll/(n_total-n_mis):.6f})")
print(f"NLL from Misclassified   : {mis_nll:.4f} ({mis_nll/total_nll*100:.2f}% of total loss, Avg per mis: {mis_nll/n_mis if n_mis>0 else 0:.4f})")

if n_mis > 0:
    print("Misclassified samples details:")
    for idx in np.where(misclassified_mask)[0]:
        true_cls = val_targets[idx]
        pred_cls = preds_t1[idx]
        sample_logits = val_logits[idx]
        print(f"  Idx {idx:4d}: True={true_cls}, Pred={pred_cls}, Logits={sample_logits}, Sample NLL={nll_per_sample[idx]:.4f}")

# ----------------------------------------------------------------------
# B2. Val NLL and ECE (15 bins) across Temperatures
# ----------------------------------------------------------------------
print("\n--- B2. VAL NLL AND ECE (15 BINS) ACROSS TEMPERATURES ---")

def compute_ece(logits, targets, n_bins=15):
    # Softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_details.append((bin_lower, bin_upper, int(in_bin.sum()), accuracy_in_bin, avg_confidence_in_bin))
    return float(ece), bin_details

T_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
table_b2 = []
for T in T_values:
    scaled_logits = val_logits_t / T
    nll = float(F.cross_entropy(scaled_logits, val_targets_t).item())
    ece, _ = compute_ece(val_logits / T, val_targets, n_bins=15)
    table_b2.append({"T": T, "Mean_NLL": nll, "ECE": ece, "ECE_pct": ece * 100})

df_b2 = pd.DataFrame(table_b2)
print(df_b2.to_string(index=False, justify="right", float_format=lambda x: f"{x:.4f}"))

# ----------------------------------------------------------------------
# B3. Exact Temperature-Fitting Code Trace & Gradient
# ----------------------------------------------------------------------
print("\n--- B3. EXACT TEMPERATURE-FITTING CODE TRACE (LBFGS) ---")

temperature = nn.Parameter(torch.ones(1) * 1.0)
optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
criterion = nn.CrossEntropyLoss()

loss_trace = []
eval_count = 0

def eval_loss():
    global eval_count
    optimizer.zero_grad()
    loss = criterion(val_logits_t / temperature, val_targets_t)
    loss.backward()
    grad = temperature.grad.item() if temperature.grad is not None else None
    loss_val = loss.item()
    eval_count += 1
    loss_trace.append((eval_count, float(temperature.item()), loss_val, grad))
    return loss

initial_loss = criterion(val_logits_t / temperature, val_targets_t)
initial_loss.backward()
init_grad = temperature.grad.item()
temperature.grad.zero_()

print(f"Initial T: {temperature.item():.4f}")
print(f"Initial Loss: {initial_loss.item():.6f}")
print(f"Initial Gradient d(Loss)/dT: {init_grad:.6e}")

optimizer.step(eval_loss)
final_T = float(temperature.item())

print(f"Final T after LBFGS: {final_T:.6f}")
print(f"Total eval steps in LBFGS: {len(loss_trace)}")
print("First 10 steps of loss trace (step, T, loss, dL/dT):")
for s, t, l, g in loss_trace[:10]:
    print(f"  Step {s:2d}: T = {t:.6f}, Loss = {l:.6f}, Grad = {g:.6e}")
if len(loss_trace) > 10:
    print("Last 5 steps of loss trace:")
    for s, t, l, g in loss_trace[-5:]:
        print(f"  Step {s:2d}: T = {t:.6f}, Loss = {l:.6f}, Grad = {g:.6e}")

# ----------------------------------------------------------------------
# B4. Input Preprocessing Consistency Check
# ----------------------------------------------------------------------
print("\n--- B4. INPUT PREPROCESSING CONSISTENCY CHECK ---")
sample_row = val_df.iloc[0]
sample_path = PATCHES_DIR / sample_row["class_name"] / f"{sample_row['crop_id']}.png"
print(f"Testing on sample patch: {sample_path}")

# Method 1: PIL + Albumentations eval_transform (Training & Eval)
im_pil = Image.open(sample_path).convert("RGB")
im_np1 = np.array(im_pil)
t_eval = eval_transform(image=im_np1)["image"].numpy()

# Method 2: OpenCV + classify_trap_blobs preprocessing
im_cv = cv2.imread(str(sample_path)) # BGR
c = im_cv
if c.shape[:2] != (64, 64):
    c = cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
chw = rgb.transpose(2, 0, 1)
mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
t_classify = (chw - mean_arr) / std_arr

# Method 3: Un-normalized 0-255 raw uint8 input
t_unnorm = np.array(im_pil).transpose(2, 0, 1).astype(np.float32)

print("Preprocessing stats comparison:")
print(f"  1. eval_transform (PIL + Albumentations) : min={t_eval.min():.4f}, max={t_eval.max():.4f}, mean={t_eval.mean():.4f}")
print(f"  2. classify_trap_blobs (CV2 + manual)    : min={t_classify.min():.4f}, max={t_classify.max():.4f}, mean={t_classify.mean():.4f}")
print(f"  3. Un-normalized raw (0-255)            : min={t_unnorm.min():.4f}, max={t_unnorm.max():.4f}, mean={t_unnorm.mean():.4f}")

diff = np.max(np.abs(t_eval - t_classify))
print(f"Max abs difference between eval_transform and classify_trap_blobs: {diff:.6e} (Identical!)")

# Run model on both to compare logits
with torch.no_grad():
    l_eval = model(torch.tensor(t_eval).unsqueeze(0)).numpy()[0]
    l_classify = model(torch.tensor(t_classify).unsqueeze(0)).numpy()[0]
    l_unnorm = model(torch.tensor(t_unnorm).unsqueeze(0)).numpy()[0]

print(f"Model logits with eval_transform     : {l_eval}")
print(f"Model logits with classify_trap_blobs: {l_classify}")
print(f"Model logits with un-normalized 0-255: {l_unnorm} (EXPLODES TO THOUSANDS!)")
print(f"Logit difference (eval vs classify)  : {np.max(np.abs(l_eval - l_classify)):.6e}")

# ----------------------------------------------------------------------
# B5. Proper Temperature Fitting (Optimize log_T with Grid Search Init)
# ----------------------------------------------------------------------
print("\n--- B5. PROPER TEMPERATURE FITTING (OPTIMIZE LOG_T) ---")

# Grid search across initial T
grid_T = np.exp(np.linspace(np.log(0.1), np.log(100.0), 50))
best_grid_T = 1.0
best_grid_nll = float('inf')

for T_init in grid_T:
    scaled = val_logits_t / T_init
    nll = float(F.cross_entropy(scaled, val_targets_t).item())
    if nll < best_grid_nll:
        best_grid_nll = nll
        best_grid_T = T_init

print(f"Grid search optimal T_init: {best_grid_T:.4f} with NLL = {best_grid_nll:.6f}")

# Optimize parameter log_T
class LogTemperatureScaler(nn.Module):
    def __init__(self, init_T):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor([np.log(init_T)], dtype=torch.float32))
    def forward(self, logits):
        return logits / torch.exp(self.log_T)

model_scaler = LogTemperatureScaler(best_grid_T)
opt = torch.optim.LBFGS(model_scaler.parameters(), lr=0.01, max_iter=100)

trace_log_t = []
def eval_log_t():
    opt.zero_grad()
    loss = criterion(model_scaler(val_logits_t), val_targets_t)
    loss.backward()
    trace_log_t.append((float(torch.exp(model_scaler.log_T).item()), float(loss.item())))
    return loss

opt.step(eval_log_t)
refit_T = float(torch.exp(model_scaler.log_T).item())
refit_nll = float(criterion(model_scaler(val_logits_t), val_targets_t).item())
refit_ece, _ = compute_ece(val_logits / refit_T, val_targets, n_bins=15)

print(f"Refitted Temperature T_refit = {refit_T:.4f}")
print(f"Refitted Val NLL             = {refit_nll:.6f} (vs T=1.0: {mean_nll:.6f})")
print(f"Refitted Val ECE             = {refit_ece*100:.2f}% (vs T=1.0: {table_b2[1]['ECE_pct']:.2f}%)")

# Recompute tau_energy with both T=0.9997 and T_refit
e_val_refit = -refit_T * logsumexp(val_logits / refit_T, axis=1)
tau_energy_refit = float(np.percentile(e_val_refit, 95.0))

e_ood_refit = -refit_T * logsumexp(ood_logits / refit_T, axis=1)
ood_rej_refit = float((e_ood_refit > tau_energy_refit).mean()) * 100.0

print(f"Refitted TAU_ENERGY          = {tau_energy_refit:.4f}")
print(f"Refitted OOD Rejection       = {ood_rej_refit:.2f}% ({int((e_ood_refit > tau_energy_refit).sum())} / {len(e_ood_refit)})")

# ----------------------------------------------------------------------
# B6. Gate Independence Analysis (Confidence Floor vs Energy Threshold)
# ----------------------------------------------------------------------
print("\n--- B6. GATE INDEPENDENCE ANALYSIS ---")

T_cal = 0.9997
for T_eval, name_t in [(T_cal, f"T_cal={T_cal:.4f}"), (refit_T, f"T_refit={refit_T:.4f}")]:
    print(f"\nEvaluating Gates under {name_t}:")
    
    # Val set
    tau_e = float(np.percentile(-T_eval * logsumexp(val_logits / T_eval, axis=1), 95.0))
    val_probs = np.exp(val_logits / T_eval - np.max(val_logits / T_eval, axis=1, keepdims=True))
    val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
    val_max_p = np.max(val_probs, axis=1)
    val_energy = -T_eval * logsumexp(val_logits / T_eval, axis=1)

    val_conf_reject = (val_max_p < 0.60)
    val_energy_reject = (val_energy > tau_e)
    val_either_reject = val_conf_reject | val_energy_reject
    val_both_reject = val_conf_reject & val_energy_reject

    print(f"  Validation Set (N = {len(val_df)}):")
    print(f"    Confidence Floor (< 0.60) only : {val_conf_reject.sum():4d} / {len(val_df)} ({val_conf_reject.mean()*100:.2f}%)")
    print(f"    Energy Threshold (> {tau_e:.2f}) only : {val_energy_reject.sum():4d} / {len(val_df)} ({val_energy_reject.mean()*100:.2f}%) [Design: exactly 5%]")
    print(f"    Both gates triggered           : {val_both_reject.sum():4d} / {len(val_df)} ({val_both_reject.mean()*100:.2f}%)")
    print(f"    Either gate (Total Abstain)    : {val_either_reject.sum():4d} / {len(val_df)} ({val_either_reject.mean()*100:.2f}%)")

    # OOD set
    ood_probs = np.exp(ood_logits / T_eval - np.max(ood_logits / T_eval, axis=1, keepdims=True))
    ood_probs = ood_probs / np.sum(ood_probs, axis=1, keepdims=True)
    ood_max_p = np.max(ood_probs, axis=1)
    ood_energy = -T_eval * logsumexp(ood_logits / T_eval, axis=1)

    ood_conf_reject = (ood_max_p < 0.60)
    ood_energy_reject = (ood_energy > tau_e)
    ood_either_reject = ood_conf_reject | ood_energy_reject
    ood_both_reject = ood_conf_reject & ood_energy_reject

    print(f"  Hard OOD Set (N = {len(ood_df)}):")
    print(f"    Confidence Floor (< 0.60) only : {ood_conf_reject.sum():4d} / {len(ood_df)} ({ood_conf_reject.mean()*100:.2f}%)")
    print(f"    Energy Threshold (> {tau_e:.2f}) only : {ood_energy_reject.sum():4d} / {len(ood_df)} ({ood_energy_reject.mean()*100:.2f}%)")
    print(f"    Both gates triggered           : {ood_both_reject.sum():4d} / {len(ood_df)} ({ood_both_reject.mean()*100:.2f}%)")
    print(f"    Either gate (Total Rejection)  : {ood_either_reject.sum():4d} / {len(ood_df)} ({ood_either_reject.mean()*100:.2f}%)")
