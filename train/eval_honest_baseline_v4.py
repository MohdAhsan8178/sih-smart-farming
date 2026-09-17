"""
Train existing TrapPestCNN architecture on manifest v4 (cleanliness, mirid recovery & card split).

Evaluates:
1. In-Distribution Test (group-isolated).
2. Cross-Card Holdout Test (held-out Wageningen cards).
3. Hard Open-Set (OOD) evaluation (verified non-target crops).

Outputs:
- Macro-F1 (primary headline)
- Balanced Accuracy (mean per-class recall)
- Per-Class Recall
- Critical operational error rate (small_pale_winged -> larger_insect)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
from tqdm import tqdm

from configs.classes_model_b import CLASS_NAMES, NUM_CLASSES, IDX, TARGET_CLASS_IDX
from configs.paths import ROOT, SPLITS, CKPT, REPORTS
from train.model_b import TrapPestCNN
from train.transforms_model_b import train_transform_model_b, eval_transform_model_b

MANDATORY_CAVEAT: str = (
    "NOTICE: Reported metrics measure benchmark performance on group-isolated trap cards "
    "and held-out card splits; they predict nothing about Indian field recall."
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
             device: torch.device) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
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
    all_logits = np.concatenate(all_logits, axis=0) if len(all_logits) > 0 else np.zeros((0, NUM_CLASSES))
    loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0
    acc = (all_preds == all_targets).mean() if len(all_targets) > 0 else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if len(all_targets) > 0 else 0.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
    return loss, acc, macro_f1, bal_acc, all_targets, all_preds, all_logits


def run_baseline_v4(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline_v4] Compute device: {device}")

    manifest_path = SPLITS / "model_b_manifest_v4.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df_all = pd.read_csv(manifest_path)
    df_train = df_all[df_all['split'] == 'train'].reset_index(drop=True)
    df_val = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    df_test_indist = df_all[df_all['split'] == 'test_indist'].reset_index(drop=True)
    df_test_cross = df_all[df_all['split'] == 'test_cross_card'].reset_index(drop=True)
    df_ood = df_all[df_all['split'] == 'openset_eval'].reset_index(drop=True)

    print(f"[baseline_v4] Splits loaded:")
    print(f"  Train           : {len(df_train)}")
    print(f"  Val             : {len(df_val)}")
    print(f"  Test In-Dist    : {len(df_test_indist)}")
    print(f"  Test Cross-Card : {len(df_test_cross)} (84 held-out 4TU cards)")
    print(f"  Open-Set Hard   : {len(df_ood)}")

    train_ds = PatchDataset(df_train, transform=train_transform_model_b())
    val_ds = PatchDataset(df_val, transform=eval_transform_model_b())
    test_indist_ds = PatchDataset(df_test_indist, transform=eval_transform_model_b())
    test_cross_ds = PatchDataset(df_test_cross, transform=eval_transform_model_b())
    ood_ds = PatchDataset(df_ood, transform=eval_transform_model_b())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_indist_loader = DataLoader(test_indist_ds, batch_size=batch_size, shuffle=False)
    test_cross_loader = DataLoader(test_cross_ds, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)

    model = TrapPestCNN(num_classes=NUM_CLASSES).to(device)
    weights = compute_class_weights(df_train).to(device)
    print(f"[baseline_v4] Class weights: {weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_ckpt_path = CKPT / "model_b_v4_honest_baseline.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_bal, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val Macro-F1: {val_f1:.4f} Bal-Acc: {val_bal*100:.2f}% (Acc: {val_acc*100:.2f}%)")

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_f1': float(val_f1),
                'val_bal_acc': float(val_bal),
                'num_classes': NUM_CLASSES,
                'class_names': CLASS_NAMES,
            }, best_ckpt_path)

    # Load best weights
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    # 1. In-Dist Test Eval
    _, indist_acc, indist_f1, indist_bal, indist_y, indist_pred, indist_logits = evaluate(model, test_indist_loader, criterion, device)
    cm_indist = confusion_matrix(indist_y, indist_pred, labels=list(range(NUM_CLASSES)))

    indist_recalls = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tot = (indist_y == c_idx).sum()
        cor = cm_indist[c_idx, c_idx]
        indist_recalls[c_name] = (cor / tot) if tot > 0 else 0.0

    # 2. Cross-Card Test Eval (84 held-out 4TU cards)
    _, cross_acc, cross_f1, cross_bal, cross_y, cross_pred, cross_logits = evaluate(model, test_cross_loader, criterion, device)
    cm_cross = confusion_matrix(cross_y, cross_pred, labels=list(range(NUM_CLASSES)))

    cross_recalls = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tot = (cross_y == c_idx).sum()
        cor = cm_cross[c_idx, c_idx]
        cross_recalls[c_name] = (cor / tot) if tot > 0 else 0.0

    # 3. Critical Error Rate: small_pale_winged -> larger_insect
    wf_tot_indist = (indist_y == TARGET_CLASS_IDX).sum()
    wf_mis_indist = cm_indist[TARGET_CLASS_IDX, IDX['larger_insect']] if wf_tot_indist > 0 else 0
    err_indist = (wf_mis_indist / wf_tot_indist) * 100.0 if wf_tot_indist > 0 else 0.0

    wf_tot_cross = (cross_y == TARGET_CLASS_IDX).sum()
    wf_mis_cross = cm_cross[TARGET_CLASS_IDX, IDX['larger_insect']] if wf_tot_cross > 0 else 0
    err_cross = (wf_mis_cross / wf_tot_cross) * 100.0 if wf_tot_cross > 0 else 0.0

    # 4. Open-Set Evaluation
    _, _, _, _, _, ood_pred, ood_logits = evaluate(model, ood_loader, criterion, device)
    ood_class_counts = {CLASS_NAMES[i]: int((ood_pred == i).sum()) for i in range(NUM_CLASSES)}

    from scipy.special import logsumexp
    id_energy = -logsumexp(indist_logits, axis=1) if len(indist_logits) > 0 else np.array([0.0])
    ood_energy = -logsumexp(ood_logits, axis=1) if len(ood_logits) > 0 else np.array([0.0])
    tau_95 = float(np.percentile(id_energy, 95.0))
    ood_rejection_rate = float((ood_energy > tau_95).mean()) * 100.0

    report_lines = [
        "HONEST RE-MEASURED BASELINE REPORT v4 (STRICT SUBSTRATE & MIRID RECOVERY)",
        "=" * 65,
        "DISCIPLINE AUDIT: Raw accuracy has been STRUCK as a headline metric.",
        "Both 76.79% (leaked) and 97.11% (class-imbalanced) are invalid headlines.",
        "Primary metrics are Macro-F1, Balanced Accuracy, and Per-Class Recall.",
        "=" * 65,
        "",
        "1. IN-DISTRIBUTION TEST SET (Group-Isolated Cards & Specimens):",
        f"   Macro-F1 (HEADLINE) : {indist_f1:.4f}",
        f"   Balanced Accuracy   : {indist_bal * 100:.2f}%",
        f"   Raw Accuracy        : {indist_acc * 100:.2f}%  [non-headline]",
        "   Per-Class Recall:",
    ]
    for c_name in CLASS_NAMES:
        report_lines.append(f"     - {c_name:18s}: {indist_recalls[c_name]*100:.2f}%")
    report_lines.append(f"   Whitefly->Larger Error Rate: {err_indist:.2f}%")
    report_lines.append(f"   Confusion Matrix:\n{cm_indist}\n")

    report_lines.extend([
        "2. CROSS-CARD TEST SET (84 Held-Out Wageningen 4TU Cards):",
        f"   Macro-F1 (HEADLINE) : {cross_f1:.4f}",
        f"   Balanced Accuracy   : {cross_bal * 100:.2f}%",
        f"   Raw Accuracy        : {cross_acc * 100:.2f}%  [non-headline]",
        "   Per-Class Recall:",
    ])
    for c_name in CLASS_NAMES:
        report_lines.append(f"     - {c_name:18s}: {cross_recalls[c_name]*100:.2f}%")
    report_lines.append(f"   Whitefly->Larger Error Rate: {err_cross:.2f}%")
    report_lines.append(f"   Confusion Matrix:\n{cm_cross}\n")

    report_lines.extend([
        "3. HARD OPEN-SET (OOD) EVALUATION (Verified Non-Targets & Markings):",
        f"   Total OOD Samples   : {len(df_ood)}",
        f"   Energy Rejection Rate (at 95% ID TPR): {ood_rejection_rate:.2f}%",
        f"   Predictions on OOD (without rejection): {ood_class_counts}\n",
        "4. SUMMARY COMPARISON AGAINST LEAKED/IMBALANCED FIGURES:",
        "   - Leaked v1 Raw Acc     : 76.79% [STRUCK: specimen leakage across splits]",
        "   - Imbalanced v2 Raw Acc : 97.11% [STRUCK: 85.6% whitefly class imbalance]",
        f"   - v4 In-Dist Macro-F1   : {indist_f1:.4f} (Balanced Acc: {indist_bal*100:.2f}%)",
        f"   - v4 Cross-Card Macro-F1: {cross_f1:.4f} (Balanced Acc: {cross_bal*100:.2f}%)",
        "=" * 65,
        MANDATORY_CAVEAT
    ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_rep = REPORTS / "model_b_evaluation_v4.txt"
    out_rep.write_text(report_text)
    print(f"\n[baseline_v4] Saved report to {out_rep}")


if __name__ == "__main__":
    run_baseline_v4()
