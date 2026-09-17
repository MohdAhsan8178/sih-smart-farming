#!/usr/bin/env python3
"""
Step 15: Calibrate all decision thresholds.
Fits in this exact sequence:
1. Temperature scaling -> T_CAL (LBFGS on validation NLL).
2. Energy threshold -> TAU_ENERGY (core.rejection.fit_energy_threshold on disjoint open-set data).
3. Post-hoc prior strength -> TAU_PRIOR (sweep [0, 0.25, 0.5, 0.75, 1.0] on validation macro-F1).
4. Aggregation thresholds -> TAU_DISEASE, TAU_MARGIN (grid-sweep, plot precision/recall).

Produces:
- artifacts/reports/ood_metrics.json
- artifacts/reports/threshold_sweep.png
- configs/train_config.py (updated in-place when --write-config is passed)

Reference: docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 15
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn

from configs.classes import (
    CLASS_NAMES,
    CROP_COLS,
    DISEASE_COLS,
    HEALTHY_COLS,
    NOTCROP_COL,
    NUM_CLASSES,
)
from configs.paths import CKPT, REPORTS, ROOT, SPLITS
from configs.train_config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    TAU_CONF,
    TAU_DISEASE as DEFAULT_TAU_DISEASE,
    TAU_MARGIN as DEFAULT_TAU_MARGIN,
    TAU_PRIOR as DEFAULT_TAU_PRIOR,
)
from core.aggregate import aggregate_frame
from core.rejection import (
    fit_energy_threshold,
    open_set_energy,
    posthoc_logit_adjust,
    softmax,
)
from scripts.extract_logits import extract_logits
from train.model import build_model
from train.train_model_a import build_test_loader, load_checkpoint_for_eval


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 100,
) -> Tuple[float, float, float]:
    """
    Fits temperature scaling parameter T_CAL using LBFGS on validation NLL.
    Returns:
        (T_CAL, nll_before, nll_after)
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    targets_t = torch.tensor(val_targets, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    nll_before = float(criterion(logits_t, targets_t).item())

    # Optimize single temperature parameter (initialized to 1.5)
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    t_cal = float(temperature.item())
    nll_after = float(criterion(logits_t / temperature, targets_t).item())

    return t_cal, nll_before, nll_after


def fit_tau_prior(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    train_csv: Path,
    tau_candidates: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> Tuple[float, Dict[float, float]]:
    """
    Sweeps post-hoc prior strength tau across [0, 0.25, 0.5, 0.75, 1.0] and picks
    the value maximizing validation macro-F1.
    """
    train_df = pd.read_csv(train_csv)
    class_counts = train_df["label"].value_counts()
    priors = np.array([class_counts.get(name, 1) for name in CLASS_NAMES], dtype=np.float64)
    priors = priors / priors.sum()
    log_priors = np.log(priors)

    sweep_results: Dict[float, float] = {}
    for tau in tau_candidates:
        adj_logits = posthoc_logit_adjust(val_logits, log_priors, tau=tau)
        preds = np.argmax(adj_logits, axis=1)
        macro_f1 = float(f1_score(val_targets, preds, average="macro"))
        sweep_results[tau] = macro_f1

    best_tau = max(sweep_results, key=sweep_results.get)
    return best_tau, sweep_results


def sweep_aggregation_thresholds(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    t_cal: float,
    tau_disease_grid: List[float] = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    tau_margin_grid: List[float] = [0.00, 0.05, 0.10, 0.15, 0.20],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Simulates multi-tile frames from validation data and sweeps (tau_disease, tau_margin)
    for the DISEASE verdict. Chooses optimal operating point.
    """
    probs = softmax(val_logits, T=t_cal)

    disease_indices = np.where(np.isin(val_targets, DISEASE_COLS))[0]
    healthy_indices = np.where(np.isin(val_targets, HEALTHY_COLS))[0]
    notcrop_indices = np.where(val_targets == NOTCROP_COL)[0]

    np.random.seed(42)
    frames = []
    labels = []

    # 1. Pure diseased frames (4 tiles)
    for i in range(0, len(disease_indices) - 4, 4):
        frames.append(probs[disease_indices[i:i+4]])
        labels.append("DISEASE")

    # 2. Mixed frames (1 disease tile + 3 healthy tiles) - testing MIL sensitivity
    for i in range(min(len(disease_indices), len(healthy_indices) // 3)):
        d_idx = disease_indices[i]
        h_idxs = healthy_indices[i*3:(i+1)*3]
        tile_set = np.vstack([probs[d_idx:d_idx+1], probs[h_idxs]])
        frames.append(tile_set)
        labels.append("DISEASE")

    # 3. Pure healthy frames (4 tiles)
    for i in range(0, len(healthy_indices) - 4, 4):
        frames.append(probs[healthy_indices[i:i+4]])
        labels.append("HEALTHY")

    # 4. Pure not-crop frames (4 tiles)
    for i in range(0, len(notcrop_indices) - 4, 4):
        frames.append(probs[notcrop_indices[i:i+4]])
        labels.append("NOT_CROP")

    best_f1 = -1.0
    best_point = None
    all_records = []

    for td in tau_disease_grid:
        for tm in tau_margin_grid:
            tp, fp, fn = 0, 0, 0
            for frame_probs, true_state in zip(frames, labels):
                state, _, _ = aggregate_frame(
                    frame_probs,
                    HEALTHY_COLS,
                    NOTCROP_COL,
                    tau_disease=td,
                    tau_margin=tm,
                )
                if state == "DISEASE":
                    if true_state == "DISEASE":
                        tp += 1
                    else:
                        fp += 1
                else:
                    if true_state == "DISEASE":
                        fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            rec_dict = {
                "tau_disease": float(td),
                "tau_margin": float(tm),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
            all_records.append(rec_dict)

            # Optimization criterion: maximize F1 with constraint precision >= 0.95
            if prec >= 0.95 and f1 > best_f1:
                best_f1 = f1
                best_point = rec_dict

    if best_point is None:
        best_point = max(all_records, key=lambda r: r["f1"])

    return best_point["tau_disease"], best_point["tau_margin"], best_point, all_records


def plot_threshold_sweep(
    id_energy: np.ndarray,
    ood_energy: np.ndarray,
    tau_energy: float,
    fpr_95: float,
    auroc: float,
    tau_prior_sweep: Dict[float, float],
    best_tau_prior: float,
    agg_sweep_records: List[Dict[str, Any]],
    best_agg_point: Dict[str, Any],
    output_png: Path,
) -> Path:
    """
    Generates threshold_sweep.png containing:
    Panel 1: Open-set energy distribution (ID vs OOD) with TAU_ENERGY threshold.
    Panel 2: ROC curve for OOD rejection.
    Panel 3: Post-hoc prior strength sweep curve.
    Panel 4: Aggregation precision-recall sweep across (tau_disease, tau_margin).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Energy distributions
    ax = axes[0, 0]
    ax.hist(id_energy, bins=60, alpha=0.6, label="In-Distribution (Val Crop)", density=True, color="green")
    ax.hist(ood_energy, bins=60, alpha=0.6, label="Open-Set OOD (raw/openset)", density=True, color="red")
    ax.axvline(tau_energy, color="black", linestyle="--", linewidth=2, label=f"TAU_ENERGY = {tau_energy:.3f} (95% TPR)")
    ax.set_title(f"Open-Set Energy Distribution (AUROC: {auroc*100:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Energy E(x)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: ROC Curve
    ax = axes[0, 1]
    y_true = np.concatenate([np.zeros(len(id_energy)), np.ones(len(ood_energy))])
    scores = np.concatenate([id_energy, ood_energy])
    thresholds = np.linspace(scores.min(), scores.max(), 300)
    tprs = []
    fprs = []
    for th in thresholds:
        # Score > th is classified as OOD
        tp = np.sum((scores > th) & (y_true == 1))
        fn = np.sum((scores <= th) & (y_true == 1))
        fp = np.sum((scores > th) & (y_true == 0))
        tn = np.sum((scores <= th) & (y_true == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    ax.plot(fprs, tprs, color="blue", linewidth=2, label=f"ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax.plot(fpr_95, 0.95, marker="*", markersize=12, color="red", label=f"Operating Point: FPR={fpr_95*100:.1f}% @ 95% TPR")
    ax.set_title("OOD Rejection ROC Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (OOD accepted as crop)", fontsize=10)
    ax.set_ylabel("True Positive Rate (OOD rejected)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: TAU_PRIOR sweep
    ax = axes[1, 0]
    taus = sorted(tau_prior_sweep.keys())
    f1s = [tau_prior_sweep[t] * 100 for t in taus]
    ax.plot(taus, f1s, marker="o", color="purple", linewidth=2)
    ax.scatter([best_tau_prior], [tau_prior_sweep[best_tau_prior] * 100], color="red", s=120, zorder=5, label=f"Best: tau={best_tau_prior:.2f} ({tau_prior_sweep[best_tau_prior]*100:.2f}%)")
    ax.set_title("Post-Hoc Prior Strength Sweep (Validation Macro-F1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("TAU_PRIOR Strength", fontsize=10)
    ax.set_ylabel("Val Macro-F1 (%)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Aggregation sweep
    ax = axes[1, 1]
    df_agg = pd.DataFrame(agg_sweep_records)
    for tm in df_agg["tau_margin"].unique():
        sub = df_agg[df_agg["tau_margin"] == tm]
        ax.plot(sub["recall"], sub["precision"], marker="s", label=f"tau_margin = {tm:.2f}")
    ax.scatter(
        [best_agg_point["recall"]],
        [best_agg_point["precision"]],
        color="red",
        s=150,
        marker="*",
        zorder=6,
        label=f"Selected: td={best_agg_point['tau_disease']:.2f}, tm={best_agg_point['tau_margin']:.2f} (F1={best_agg_point['f1']*100:.1f}%)",
    )
    ax.set_title("MIL Aggregation Precision-Recall Decision Surface", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall for DISEASE", fontsize=10)
    ax.set_ylabel("Precision for DISEASE", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[calibrate.py] Saved threshold sweep plot to: {output_png}")
    return output_png


def update_train_config_file(
    config_path: Path,
    tau_energy: float,
    t_cal: float,
    tau_prior: float,
    tau_disease: float,
    tau_margin: float,
) -> None:
    """
    Updates configs/train_config.py in-place with fitted threshold constants.
    Preserves exact comments, structure, and formatting.
    """
    content = config_path.read_text()

    content = re.sub(
        r"TAU_ENERGY\s*=\s*.*",
        f"TAU_ENERGY  = {round(tau_energy, 4):<8}  # <- fitted in STEP 15 on true open-set data (95% TPR)",
        content,
    )
    content = re.sub(
        r"T_CAL\s*=\s*.*",
        f"T_CAL       = {round(t_cal, 4):<8}  # <- fitted in STEP 15 (temperature scaling via LBFGS on validation NLL)",
        content,
    )
    content = re.sub(
        r"TAU_PRIOR\s*=\s*.*",
        f"TAU_PRIOR   = {round(tau_prior, 4):<8}  # <- swept in STEP 15 (best validation macro-F1 with weighted loss)",
        content,
    )
    content = re.sub(
        r"TAU_DISEASE\s*=\s*.*",
        f"TAU_DISEASE = {round(tau_disease, 4):<8}  # per-tile disease probability floor (calibrated STEP 15)",
        content,
    )
    content = re.sub(
        r"TAU_MARGIN\s*=\s*.*",
        f"TAU_MARGIN  = {round(tau_margin, 4):<8}  # disease must beat healthy on the SAME tile by this (calibrated STEP 15)",
        content,
    )

    config_path.write_text(content)
    print(f"[calibrate.py] Successfully wrote calibrated thresholds to: {config_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 15: Calibrate all decision thresholds.")
    parser.add_argument("--ckpt", type=str, default="artifacts/checkpoints/v3/stage1.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--openset", type=str, default="data/raw/openset", help="Path to open-set directory")
    parser.add_argument("--splits_dir", type=str, default="splits_v3", help="Directory containing split CSVs")
    parser.add_argument("--reports_dir", type=str, default=str(REPORTS), help="Directory for reports")
    parser.add_argument("--logits_dir", type=str, default="artifacts/logits", help="Directory for saved logits")
    parser.add_argument("--write-config", action="store_true", default=False, help="Write fitted values to configs/train_config.py")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    openset_dir = Path(args.openset)
    splits_dir = Path(args.splits_dir)
    reports_dir = Path(args.reports_dir)
    logits_dir = Path(args.logits_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "configs" / "train_config.py"

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[calibrate.py] Checkpoint: {ckpt_path} | Open-set: {openset_dir} | Device: {device}")

    # 1. Validation logits resolution
    val_logits_file = logits_dir / "val_logits.npy"
    val_paths_file = logits_dir / "val_paths.json"

    if not val_logits_file.exists() or not val_paths_file.exists():
        print(f"[calibrate.py] Validation logits not found in {logits_dir}, generating...")
        from train.evaluate import evaluate_split_dataset
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        model, weights_used = load_checkpoint_for_eval(ckpt_path, model=model, eval_weights="auto", device=device)
        model.to(device)
        _, _, val_logits, val_paths = evaluate_split_dataset(
            model=model,
            split_file=splits_dir / "val.csv",
            class_names=CLASS_NAMES,
            device=device,
            batch_size=args.batch_size,
        )
        np.save(val_logits_file, val_logits)
        with open(val_paths_file, "w") as f:
            json.dump(val_paths, f, indent=2)
    else:
        val_logits = np.load(val_logits_file)
        val_paths = json.load(open(val_paths_file))

    val_targets = np.array([p["label"] for p in val_paths], dtype=int)

    # ----------------------------------------------------
    # STEP 1: Temperature Scaling -> T_CAL
    # ----------------------------------------------------
    print("\n[calibrate.py] [1/4] Fitting Temperature Scaling (T_CAL) via LBFGS on validation NLL...")
    t_cal, nll_before, nll_after = fit_temperature_scaling(val_logits, val_targets)
    print(f"[calibrate.py] T_CAL = {t_cal:.4f} (Validation NLL: {nll_before:.4f} -> {nll_after:.4f})")

    # ----------------------------------------------------
    # STEP 2: Open-Set Energy Threshold -> TAU_ENERGY
    # ----------------------------------------------------
    print("\n[calibrate.py] [2/4] Fitting Open-Set Energy Threshold (TAU_ENERGY)...")
    openset_logits_file = logits_dir / "openset_logits.npy"
    if not openset_logits_file.exists():
        print(f"[calibrate.py] Extracting open-set logits from {openset_dir}...")
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        model, weights_used = load_checkpoint_for_eval(ckpt_path, model=model, eval_weights="auto", device=device)
        model.to(device)
        openset_logits, _ = extract_logits(
            model=model,
            input_dir=openset_dir,
            out=openset_logits_file,
            checkpoint_path=ckpt_path,
            weights_used=weights_used,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        openset_logits = np.load(openset_logits_file)

    tau_energy, fpr_95 = fit_energy_threshold(val_logits, openset_logits, CROP_COLS, tpr=0.95)

    id_e = open_set_energy(val_logits, CROP_COLS)
    ood_e = open_set_energy(openset_logits, CROP_COLS)
    y_true_ood = np.concatenate([np.zeros(len(id_e)), np.ones(len(ood_e))])
    scores_ood = np.concatenate([id_e, ood_e])
    auroc = float(roc_auc_score(y_true_ood, scores_ood))

    print(f"[calibrate.py] TAU_ENERGY = {tau_energy:.4f} | FPR@95TPR = {fpr_95*100:.2f}% | AUROC = {auroc*100:.2f}%")

    # ----------------------------------------------------
    # STEP 3: Post-Hoc Prior Strength -> TAU_PRIOR
    # ----------------------------------------------------
    print("\n[calibrate.py] [3/4] Sweeping Post-Hoc Prior Strength (TAU_PRIOR)...")
    train_csv = splits_dir / "train.csv"
    best_tau_prior, tau_prior_sweep = fit_tau_prior(val_logits, val_targets, train_csv)
    for tau, score in tau_prior_sweep.items():
        mark = " <-- BEST" if tau == best_tau_prior else ""
        print(f"  tau = {tau:<4.2f} : Validation Macro-F1 = {score*100:.2f}%{mark}")
    print(f"[calibrate.py] Selected TAU_PRIOR = {best_tau_prior:.2f}")

    # ----------------------------------------------------
    # STEP 4: Aggregation Thresholds -> TAU_DISEASE, TAU_MARGIN
    # ----------------------------------------------------
    print("\n[calibrate.py] [4/4] Grid-sweeping MIL Aggregation Thresholds (TAU_DISEASE, TAU_MARGIN)...")
    tau_disease, tau_margin, best_agg_point, agg_records = sweep_aggregation_thresholds(
        val_logits, val_targets, t_cal=t_cal
    )
    print(
        f"[calibrate.py] Selected Operating Point: TAU_DISEASE = {tau_disease:.2f}, "
        f"TAU_MARGIN = {tau_margin:.2f} (Precision = {best_agg_point['precision']*100:.1f}%, "
        f"Recall = {best_agg_point['recall']*100:.1f}%, F1 = {best_agg_point['f1']*100:.1f}%)"
    )

    # ----------------------------------------------------
    # Save Reports: ood_metrics.json & threshold_sweep.png
    # ----------------------------------------------------
    ood_metrics_payload = {
        "title": "Model A Threshold Calibration & OOD Metrics (Step 15)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint": str(ckpt_path),
        "temperature_scaling": {
            "T_CAL": float(t_cal),
            "val_nll_before": float(nll_before),
            "val_nll_after": float(nll_after),
            "optimization_method": "LBFGS on validation negative log-likelihood",
        },
        "open_set_rejection": {
            "TAU_ENERGY": float(tau_energy),
            "target_tpr": 0.95,
            "fpr_at_95tpr": float(fpr_95),
            "fpr_at_95tpr_percentage": float(fpr_95 * 100.0),
            "auroc": float(auroc),
            "auroc_percentage": float(auroc * 100.0),
            "in_distribution_samples": int(len(id_e)),
            "open_set_samples": int(len(ood_e)),
            "open_set_source": str(openset_dir),
        },
        "post_hoc_prior": {
            "selected_TAU_PRIOR": float(best_tau_prior),
            "sweep": {str(k): float(v) for k, v in tau_prior_sweep.items()},
            "note": "Stage 1/2 trained with sqrt-inverse-frequency class weights; tau=0.0 avoids double-correction",
        },
        "aggregation_thresholds": {
            "selected_TAU_DISEASE": float(tau_disease),
            "selected_TAU_MARGIN": float(tau_margin),
            "operating_point": best_agg_point,
        },
    }

    ood_json_path = reports_dir / "ood_metrics.json"
    with open(ood_json_path, "w") as f:
        json.dump(ood_metrics_payload, f, indent=2)
    print(f"[calibrate.py] Saved OOD and calibration report to: {ood_json_path}")

    plot_png_path = reports_dir / "threshold_sweep.png"
    plot_threshold_sweep(
        id_energy=id_e,
        ood_energy=ood_e,
        tau_energy=tau_energy,
        fpr_95=fpr_95,
        auroc=auroc,
        tau_prior_sweep=tau_prior_sweep,
        best_tau_prior=best_tau_prior,
        agg_sweep_records=agg_records,
        best_agg_point=best_agg_point,
        output_png=plot_png_path,
    )

    # ----------------------------------------------------
    # Update Config if --write-config passed
    # ----------------------------------------------------
    if args.write_config:
        update_train_config_file(
            config_path=config_path,
            tau_energy=tau_energy,
            t_cal=t_cal,
            tau_prior=best_tau_prior,
            tau_disease=tau_disease,
            tau_margin=tau_margin,
        )

    print("\n[calibrate.py] Step 15 calibration successfully finished.")


if __name__ == "__main__":
    main()
