"""
Train existing TrapPestCNN architecture on the group-aware rebuilt splits (manifest v2)
to measure the honest re-measured baseline without card-level leakage.

Evaluates:
1. Clean In-Distribution Test (unseen PST cards + unseen Ong & Hoye specimens)
2. Degraded In-Distribution Test (ESP32-CAM degradation simulation)
3. Cross-Source Generalization Test (100% held-out Wageningen 4TU - 284 unseen cards)
4. Critical ETL error rate: small_pale_winged -> larger_insect
5. Hard Open-Set (OOD) rejection score distribution
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

from configs.classes_model_b import CLASS_NAMES, NUM_CLASSES, IDX, TARGET_CLASS_IDX
from configs.paths import ROOT, SPLITS, CKPT, ONNX_DIR, REPORTS
from train.model_b import TrapPestCNN, build_model_b
from train.transforms_model_b import train_transform_model_b, eval_transform_model_b

MANDATORY_CAVEAT: str = (
    "NOTICE: Reported accuracy measures benchmark performance on group-isolated trap cards "
    "and cross-source European benchmarks; it predicts nothing about Indian field recall."
)


class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        rel_path = str(row['rel_path'])
        abs_path = ROOT / rel_path
        img = cv2.imread(str(abs_path))
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=rgb)
            tensor = augmented['image']
        else:
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        label = int(row['class_idx'])
        crop_id = str(row.get('crop_id', ''))
        return tensor, label, crop_id


def compute_class_weights(df_train: pd.DataFrame) -> torch.Tensor:
    counts = df_train['class_idx'].value_counts().to_dict()
    n_total = len(df_train)
    weights = []
    for c in range(NUM_CLASSES):
        n_c = counts.get(c, 1)
        w = np.sqrt(n_total / (NUM_CLASSES * n_c))
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_logits = []

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.concatenate(all_logits, axis=0)
    loss = total_loss / len(all_targets)
    acc = (all_preds == all_targets).mean()
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return loss, acc, macro_f1, all_targets, all_preds, all_logits


def run_honest_baseline(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Using compute device: {device}")

    manifest_path = SPLITS / "model_b_manifest_v2.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found.")

    df_all = pd.read_csv(manifest_path)
    df_train = df_all[df_all['split'] == 'train'].reset_index(drop=True)
    df_val = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    df_test_indist_clean = df_all[df_all['split'] == 'test_indist_clean'].reset_index(drop=True)
    df_test_indist_deg = df_all[df_all['split'] == 'test_indist_degraded'].reset_index(drop=True)
    df_test_cross = df_all[df_all['split'] == 'test_cross_source'].reset_index(drop=True)
    df_openset = df_all[df_all['split'] == 'openset_eval'].reset_index(drop=True)

    print(f"[baseline] Master dataset splits loaded:")
    print(f"  Train               : {len(df_train)}")
    print(f"  Val                 : {len(df_val)}")
    print(f"  Test In-Dist Clean  : {len(df_test_indist_clean)}")
    print(f"  Test In-Dist Degraded: {len(df_test_indist_deg)}")
    print(f"  Test Cross-Source   : {len(df_test_cross)} (Wageningen 4TU)")
    print(f"  Open-Set Hard (OOD) : {len(df_openset)}")

    train_ds = PatchDataset(df_train, transform=train_transform_model_b())
    val_ds = PatchDataset(df_val, transform=eval_transform_model_b())
    test_clean_ds = PatchDataset(df_test_indist_clean, transform=eval_transform_model_b())
    test_deg_ds = PatchDataset(df_test_indist_deg, transform=eval_transform_model_b())
    test_cross_ds = PatchDataset(df_test_cross, transform=eval_transform_model_b())
    openset_ds = PatchDataset(df_openset, transform=eval_transform_model_b())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_clean_loader = DataLoader(test_clean_ds, batch_size=batch_size, shuffle=False)
    test_deg_loader = DataLoader(test_deg_ds, batch_size=batch_size, shuffle=False)
    test_cross_loader = DataLoader(test_cross_ds, batch_size=batch_size, shuffle=False)
    openset_loader = DataLoader(openset_ds, batch_size=batch_size, shuffle=False)

    # Initialize model from scratch
    model = TrapPestCNN(num_classes=NUM_CLASSES)
    model.to(device)

    weights = compute_class_weights(df_train).to(device)
    print(f"[baseline] Training class weights: {weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n" + "=" * 70)
    print(f"[baseline] Retraining TrapPestCNN on Group-Aware Splits ({epochs} epochs)...")
    print("=" * 70)

    best_val_f1 = -1.0
    best_ckpt_path = CKPT / "model_b_honest_baseline.pt"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_f1': float(val_f1),
                'val_acc': float(val_acc),
                'num_classes': NUM_CLASSES,
                'class_names': CLASS_NAMES,
            }, best_ckpt_path)

    print(f"\n[baseline] Best validation macro-F1: {best_val_f1:.4f} saved to {best_ckpt_path}")

    # Load best checkpoint for thorough evaluation
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    # 1. Clean In-Distribution Test
    _, clean_acc, clean_f1, clean_y, clean_pred, clean_logits = evaluate(model, test_clean_loader, criterion, device)
    cm_clean = confusion_matrix(clean_y, clean_pred, labels=list(range(NUM_CLASSES)))

    # 2. Degraded In-Distribution Test
    _, deg_acc, deg_f1, deg_y, deg_pred, _ = evaluate(model, test_deg_loader, criterion, device)
    cm_deg = confusion_matrix(deg_y, deg_pred, labels=list(range(NUM_CLASSES)))

    # 3. Cross-Source Holdout Test (Wageningen 4TU)
    _, cross_acc, cross_f1, cross_y, cross_pred, _ = evaluate(model, test_cross_loader, criterion, device)
    cm_cross = confusion_matrix(cross_y, cross_pred, labels=list(range(NUM_CLASSES)))

    # 4. Critical Error Rates
    # small_pale_winged -> larger_insect
    wf_total_clean = (clean_y == TARGET_CLASS_IDX).sum()
    wf_mis_clean = cm_clean[TARGET_CLASS_IDX, IDX['larger_insect']] if wf_total_clean > 0 else 0
    err_rate_clean = (wf_mis_clean / wf_total_clean) * 100.0 if wf_total_clean > 0 else 0.0

    wf_total_cross = (cross_y == TARGET_CLASS_IDX).sum()
    wf_mis_cross = cm_cross[TARGET_CLASS_IDX, IDX['larger_insect']] if wf_total_cross > 0 else 0
    err_rate_cross = (wf_mis_cross / wf_total_cross) * 100.0 if wf_total_cross > 0 else 0.0

    # 5. Open-Set Evaluation
    # Model produces predictions on hard OOD
    _, _, _, _, ood_pred, ood_logits = evaluate(model, openset_loader, criterion, device)
    ood_class_counts = {CLASS_NAMES[i]: int((ood_pred == i).sum()) for i in range(NUM_CLASSES)}

    # Energy calculations
    id_energy = -np.log(np.sum(np.exp(clean_logits), axis=1))
    ood_energy = -np.log(np.sum(np.exp(ood_logits), axis=1))
    tau_energy_95 = float(np.percentile(id_energy, 95.0))
    ood_rejection_rate = float((ood_energy > tau_energy_95).mean()) * 100.0

    report_lines = [
        "HONEST RE-MEASURED BASELINE EVALUATION REPORT (MODEL B)",
        "=" * 60,
        "CORRECTION NOTICE: The previous 76.79% clean accuracy is INVALIDATED by",
        "card-level leakage (crops from the same card spanned train and test).",
        "The metrics below reflect strict group-aware splits and true cross-source holdout.",
        "=" * 60,
        "",
        "1. IN-DISTRIBUTION TEST PERFORMANCE (Group-Isolated Cards):",
        f"  Total Clean In-Dist Samples    : {len(clean_y)}",
        f"  Clean In-Dist Accuracy         : {clean_acc * 100:.2f}%",
        f"  Clean In-Dist Macro-F1         : {clean_f1:.4f}",
        "",
        "  Clean In-Dist Confusion Matrix (Rows=True, Cols=Predicted):",
        "  " + "  ".join([f"{c[:10]:>12}" for c in CLASS_NAMES]),
    ]
    for i, row in enumerate(cm_clean):
        report_lines.append(f"  {CLASS_NAMES[i][:12]:<12} " + "  ".join([f"{val:>12}" for val in row]))

    report_lines.extend([
        "",
        f"  Total Degraded In-Dist Samples : {len(deg_y)}",
        f"  Degraded In-Dist Accuracy      : {deg_acc * 100:.2f}%",
        f"  Degraded In-Dist Macro-F1      : {deg_f1:.4f}",
        "",
        "2. CROSS-SOURCE GENERALIZATION TEST (100% Held-Out Wageningen 4TU - 284 Cards):",
        f"  Total Cross-Source Samples     : {len(cross_y)}",
        f"  Cross-Source Accuracy          : {cross_acc * 100:.2f}%",
        f"  Cross-Source Macro-F1          : {cross_f1:.4f}",
        "",
        "  Cross-Source Confusion Matrix (Rows=True, Cols=Predicted):",
        "  " + "  ".join([f"{c[:10]:>12}" for c in CLASS_NAMES]),
    ])
    for i, row in enumerate(cm_cross):
        report_lines.append(f"  {CLASS_NAMES[i][:12]:<12} " + "  ".join([f"{val:>12}" for val in row]))

    report_lines.extend([
        "",
        "3. CRITICAL OPERATIONAL ERROR RATES (small_pale_winged -> larger_insect):",
        f"  In-Dist Clean Leakage Rate     : {err_rate_clean:.2f}% ({wf_mis_clean}/{wf_total_clean})",
        f"  Cross-Source Leakage Rate      : {err_rate_cross:.2f}% ({wf_mis_cross}/{wf_total_cross})",
        "",
        "4. HARD OPEN-SET (OOD) FORCED CLASSIFICATION DISTRIBUTION (No abstention):",
        f"  Total Hard OOD Samples         : {len(ood_pred)}",
        f"  Forced small_pale_winged       : {ood_class_counts['small_pale_winged']} ({ood_class_counts['small_pale_winged']/len(ood_pred)*100:.1f}%)",
        f"  Forced larger_insect           : {ood_class_counts['larger_insect']} ({ood_class_counts['larger_insect']/len(ood_pred)*100:.1f}%)",
        f"  Forced debris                  : {ood_class_counts['debris']} ({ood_class_counts['debris']/len(ood_pred)*100:.1f}%)",
        f"  OOD Rejection Rate at 95% TPR  : {ood_rejection_rate:.2f}% (Energy tau = {tau_energy_95:.4f})",
        "",
        "5. PROVENANCE & GOVERNANCE AUDIT:",
        "  - Status of 76.79% metric      : STRUCK & INVALIDATED (Card-level leakage)",
        "  - Corrected In-Dist Baseline   : " + f"{clean_acc*100:.2f}% (Group-Isolated)",
        "  - Corrected Cross-Source Metric: " + f"{cross_acc*100:.2f}% (Wageningen 4TU)",
    ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    out_rep_path = REPORTS / "model_b_honest_baseline_report.txt"
    with open(out_rep_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved honest report to {out_rep_path}")


if __name__ == "__main__":
    run_honest_baseline()
