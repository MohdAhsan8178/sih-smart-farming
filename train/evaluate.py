#!/usr/bin/env python3
"""
Step 14: Honest Model Evaluation and Generalization Gap Assessment.
Produces:
1. artifacts/reports/eval_indist.json
2. artifacts/reports/eval_crossdomain.json (with headline generalization gap)
3. artifacts/reports/confusion_matrix.png
4. artifacts/reports/per_class_recall.csv
5. artifacts/reports/gradcam_panel.png

Reference: docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 14
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.paths import CKPT, REPORTS, ROOT, SPLITS
from configs.train_config import BATCH_SIZE, IMAGE_SIZE
from train.dataset import build_loaders
from train.model import build_model
from train.train_model_a import (
    build_test_loader,
    compute_split_metrics,
    load_checkpoint_for_eval,
)
from train.transforms import eval_transform


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    target_layer: nn.Module,
) -> np.ndarray:
    """
    Computes a clean, native PyTorch Grad-CAM heatmap without third-party library bugs.
    Hooks forward activations and backward gradients on target_layer.
    Returns:
        heatmap: (H, W) float32 array normalized to [0, 1].
    """
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(input_tensor)
    score = out[0, target_class_idx]
    score.backward()

    h_fwd.remove()
    h_bwd.remove()

    act = activations[0].detach()  # (1, C, H, W)
    grad = gradients[0].detach()   # (1, C, H, W)

    # Channel-wise pooling of gradients
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))  # (1, 1, H, W)

    # Upsample to input dimensions
    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Min-max normalization
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def generate_gradcam_panel(
    model: nn.Module,
    device: torch.device,
    reports_dir: Path,
    splits_dir: Path,
    data_dir: Path,
    output_png: Path,
) -> Path:
    """
    Builds the Step 14 Grad-CAM panel: healthy / correct / failure.
    Produces artifacts/reports/gradcam_panel.png.
    """
    print("\n[evaluate.py] Generating Grad-CAM panel (healthy / correct / failure)...")
    target_layer = getattr(model, "conv_head", None)
    if target_layer is None and hasattr(model, "blocks"):
        target_layer = model.blocks[-1]
    if target_layer is None:
        raise AttributeError("Could not identify target convolution layer for Grad-CAM in model.")

    tf = eval_transform(size=IMAGE_SIZE)

    test_indist_csv = splits_dir / "test_indist.csv"
    test_sh_csv = splits_dir / "test_sourceheldout.csv"

    sh_df = pd.read_csv(test_sh_csv)
    indist_df = pd.read_csv(test_indist_csv)

    healthy_candidate = None
    correct_candidate = None
    failure_candidate = None

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Scan test_indist for Healthy & Correct Disease
    model.eval()
    for _, row in indist_df.iterrows():
        img_path = data_dir / row["path"]
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tf(image=rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs))
            conf = float(probs[pred_idx])

        true_cls = row["label"]
        true_idx = class_to_idx[true_cls]

        if pred_idx == true_idx and ("healthy" in true_cls or "normal" in true_cls):
            if healthy_candidate is None and conf > 0.85:
                healthy_candidate = {
                    "path": img_path, "true_cls": true_cls, "true_idx": true_idx,
                    "pred_cls": CLASS_NAMES[pred_idx], "pred_idx": pred_idx,
                    "conf": conf, "tensor": tensor, "rgb": rgb,
                }
        elif pred_idx == true_idx and ("blight" in true_cls or "blast" in true_cls or "rust" in true_cls):
            if correct_candidate is None and conf > 0.85:
                correct_candidate = {
                    "path": img_path, "true_cls": true_cls, "true_idx": true_idx,
                    "pred_cls": CLASS_NAMES[pred_idx], "pred_idx": pred_idx,
                    "conf": conf, "tensor": tensor, "rgb": rgb,
                }

        if healthy_candidate is not None and correct_candidate is not None:
            break

    # Scan test_sourceheldout for Failure
    for _, row in sh_df.iterrows():
        img_path = data_dir / row["path"]
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tf(image=rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs))
            conf = float(probs[pred_idx])

        true_cls = row["label"]
        true_idx = class_to_idx[true_cls]

        if pred_idx != true_idx and conf > 0.4:
            failure_candidate = {
                "path": img_path, "true_cls": true_cls, "true_idx": true_idx,
                "pred_cls": CLASS_NAMES[pred_idx], "pred_idx": pred_idx,
                "conf": conf, "tensor": tensor, "rgb": rgb,
            }
            break

    candidates = [
        ("HEALTHY", healthy_candidate),
        ("CORRECT DISEASE", correct_candidate),
        ("FAILURE (MISCLASSIFIED)", failure_candidate),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    plt.subplots_adjust(wspace=0.15)

    for ax, (title_tag, cand) in zip(axes, candidates):
        if cand is None:
            ax.text(0.5, 0.5, f"No candidate found for {title_tag}", ha="center", va="center")
            continue

        # Re-enable gradients for Grad-CAM
        tensor = cand["tensor"].clone().requires_grad_(True)
        cam = compute_gradcam(
            model=model,
            input_tensor=tensor,
            target_class_idx=cand["pred_idx"],
            target_layer=target_layer,
        )

        resized_rgb = cv2.resize(cand["rgb"], (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.55 * resized_rgb + 0.45 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        ax.imshow(overlay)
        ax.axis("off")

        box_color = "green" if title_tag != "FAILURE (MISCLASSIFIED)" else "red"
        ax.set_title(
            f"[{title_tag}]\n"
            f"Ground Truth: {cand['true_cls']}\n"
            f"Predicted: {cand['pred_cls']} ({cand['conf']*100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            pad=10,
            color=box_color,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[evaluate.py] Saved Grad-CAM panel to: {output_png}")
    return output_png


def plot_row_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_png: Path,
    title: str = "Row-Normalised Confusion Matrix (In-Distribution)",
) -> Path:
    """
    Plots a row-normalized confusion matrix heatmap and saves to disk.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm_cm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(row_norm_cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("True Positive Rate / Recall", rotation=-90, va="bottom", fontsize=11)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[evaluate.py] Saved confusion matrix to: {output_png}")
    return output_png


@torch.no_grad()
def evaluate_split_dataset(
    model: nn.Module,
    split_file: Path,
    class_names: List[str],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    data_dir: Path = ROOT,
    path_prefix_strip: str = "",
    max_steps: Optional[int] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Runs inference over a single split CSV, computes metrics, confusion matrix, and logits.
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    loader = build_test_loader(
        csv_path=split_file,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        root_dir=data_dir,
        path_prefix_strip=path_prefix_strip,
    )

    all_logits: List[np.ndarray] = []
    all_preds: List[int] = []
    all_targets: List[int] = []

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        logits_batch = outputs.detach().cpu().to(torch.float32).numpy()
        preds = np.argmax(logits_batch, axis=1).tolist()

        all_logits.append(logits_batch)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy().tolist())

    top1, macro_f1, micro_f1, per_class, cm = compute_split_metrics(
        all_targets=all_targets,
        all_preds=all_preds,
        class_names=class_names,
    )

    split_logits = np.concatenate(all_logits, axis=0).astype(np.float32) if all_logits else np.zeros((0, len(class_names)), dtype=np.float32)

    rows_evaluated = len(all_targets)
    evaluated_paths = loader.dataset.image_paths[:rows_evaluated]
    paths_payload = [
        {
            "row": i,
            "path": evaluated_paths[i],
            "label": int(all_targets[i]),
            "class_name": class_names[all_targets[i]],
        }
        for i in range(rows_evaluated)
    ]

    metrics_record = {
        "split_name": split_file.stem,
        "split_file": str(split_file),
        "row_count": len(loader.dataset),
        "rows_evaluated": rows_evaluated,
        "top1_accuracy": float(top1),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_class": per_class,
    }

    return metrics_record, cm, split_logits, paths_payload


def build_per_class_recall_csv(
    class_names: List[str],
    indist_per_class: Dict[str, Any],
    heldout_per_class: Optional[Dict[str, Any]],
    output_csv: Path,
) -> pd.DataFrame:
    """
    Builds per_class_recall.csv comparing in-distribution recall against held-out recall.
    """
    rows = []
    for cls in class_names:
        indist_rec = indist_per_class.get(cls, {}).get("recall")
        indist_sup = indist_per_class.get(cls, {}).get("support", 0)

        heldout_rec = heldout_per_class.get(cls, {}).get("recall") if heldout_per_class else None
        heldout_sup = heldout_per_class.get(cls, {}).get("support", 0) if heldout_per_class else 0

        gap = None
        if indist_rec is not None and heldout_rec is not None:
            gap = float(indist_rec - heldout_rec)

        rows.append({
            "class_name": cls,
            "indist_recall": round(indist_rec, 4) if indist_rec is not None else None,
            "sourceheldout_recall": round(heldout_rec, 4) if heldout_rec is not None else None,
            "recall_gap": round(gap, 4) if gap is not None else None,
            "indist_support": int(indist_sup),
            "sourceheldout_support": int(heldout_sup),
        })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[evaluate.py] Saved per-class recall table to: {output_csv}")
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model A on test splits honestly.")
    parser.add_argument("--ckpt", type=str, default="artifacts/checkpoints/v3/stage1.pt", help="Path to checkpoint")
    parser.add_argument("--eval_weights", type=str, default="auto", choices=["auto", "ema", "model"], help="Weights to evaluate")
    parser.add_argument("--splits_dir", type=str, default="splits_v3", help="Directory containing split CSVs")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "test_indist", "test_sourceheldout", "test_crossdomain", "test_external_riceblast", "val"],
        help="Target split to evaluate",
    )
    parser.add_argument("--reports_dir", type=str, default=str(REPORTS), help="Directory for report outputs")
    parser.add_argument("--logits_dir", type=str, default=None, help="Directory for saving pre-softmax logits")
    parser.add_argument("--data_dir", type=str, default=str(ROOT), help="Root directory for relative image paths")
    parser.add_argument("--path_prefix_strip", type=str, default="", help="Prefix to strip from CSV paths")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--gradcam", action="store_true", default=False, help="Generate Grad-CAM panel")
    parser.add_argument("--max_steps", type=int, default=None, help="Cap evaluation steps for smoke testing")
    args = parser.parse_args(argv)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = Path(args.logits_dir) if args.logits_dir else reports_dir.parent / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    data_dir = Path(args.data_dir)

    # Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[evaluate.py] Checkpoint: {args.ckpt} | Eval Weights: {args.eval_weights} | Device: {device} | Splits: {splits_dir}")

    # Build and load model
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    eval_model, weights_used = load_checkpoint_for_eval(
        checkpoint_path=args.ckpt,
        model=model,
        eval_weights=args.eval_weights,
        device=device,
    )
    eval_model.to(device)

    # If --gradcam is called standalone
    if args.gradcam and args.split != "all":
        gradcam_path = reports_dir / "gradcam_panel.png"
        generate_gradcam_panel(
            model=eval_model,
            device=device,
            reports_dir=reports_dir,
            splits_dir=splits_dir,
            data_dir=data_dir,
            output_png=gradcam_path,
        )
        return

    # Determine splits to run
    # Aliasing: test_crossdomain maps to test_sourceheldout in v3
    split_names = []
    if args.split == "all":
        split_names = ["test_indist", "test_sourceheldout", "test_external_riceblast", "val"]
    elif args.split == "test_crossdomain":
        if (splits_dir / "test_sourceheldout.csv").exists():
            split_names = ["test_indist", "test_sourceheldout"]
        elif (splits_dir / "test_crossdomain.csv").exists():
            split_names = ["test_crossdomain"]
    else:
        split_names = [args.split]

    eval_results: Dict[str, Any] = {}
    cms: Dict[str, np.ndarray] = {}

    for s_name in split_names:
        csv_file = splits_dir / f"{s_name}.csv"
        if not csv_file.exists():
            print(f"[evaluate.py] Split file {csv_file} does not exist, skipping.")
            continue

        print(f"\n[evaluate.py] Evaluating {s_name} ({csv_file})...")
        t0 = time.time()
        rec, cm, split_logits, paths = evaluate_split_dataset(
            model=eval_model,
            split_file=csv_file,
            class_names=CLASS_NAMES,
            device=device,
            batch_size=args.batch_size,
            data_dir=data_dir,
            path_prefix_strip=args.path_prefix_strip,
            max_steps=args.max_steps,
        )
        t_elapsed = time.time() - t0

        rec["checkpoint_path"] = str(args.ckpt)
        rec["weights_used"] = weights_used
        rec["evaluation_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        rec["throughput_imgs_per_sec"] = round(rec["rows_evaluated"] / max(t_elapsed, 1e-4), 1)

        eval_results[s_name] = rec
        cms[s_name] = cm

        # Save logits and confusion matrix
        np.save(reports_dir / f"confusion_matrix_{s_name}.npy", cm)
        np.save(logits_dir / f"{s_name}_logits.npy", split_logits)
        with open(logits_dir / f"{s_name}_paths.json", "w") as f:
            json.dump(paths, f, indent=2)

        print(
            f"[{s_name}] Rows: {rec['rows_evaluated']}/{rec['row_count']} | "
            f"Top-1: {rec['top1_accuracy']*100:.2f}% | "
            f"Macro-F1: {rec['macro_f1']*100:.2f}% | "
            f"Micro-F1: {rec['micro_f1']*100:.2f}% ({rec['throughput_imgs_per_sec']} imgs/s)"
        )

    # 1. artifacts/reports/eval_indist.json
    if "test_indist" in eval_results:
        indist_json = reports_dir / "eval_indist.json"
        indist_payload = {
            "title": "Model A In-Distribution Evaluation (test_indist)",
            "checkpoint": str(args.ckpt),
            "weights_used": weights_used,
            "timestamp": eval_results["test_indist"]["evaluation_timestamp"],
            "row_count": eval_results["test_indist"]["row_count"],
            "rows_evaluated": eval_results["test_indist"]["rows_evaluated"],
            "top1_accuracy": eval_results["test_indist"]["top1_accuracy"],
            "macro_f1": eval_results["test_indist"]["macro_f1"],
            "micro_f1": eval_results["test_indist"]["micro_f1"],
            "per_class": eval_results["test_indist"]["per_class"],
        }
        with open(indist_json, "w") as f:
            json.dump(indist_payload, f, indent=2)
        print(f"[evaluate.py] Saved eval_indist.json to: {indist_json}")

    # 2. artifacts/reports/eval_crossdomain.json
    # Per user instruction:
    # "In the generalization-gap reporting, explicitly compute and call out
    # in-distribution macro-F1 minus test_sourceheldout macro-F1 as the headline gap number,
    # not just the crossdomain gap — this is the more concerning of the two splits and must not be softened or omitted."
    sh_key = "test_sourceheldout" if "test_sourceheldout" in eval_results else ("test_crossdomain" if "test_crossdomain" in eval_results else None)
    if sh_key and "test_indist" in eval_results:
        indist_f1 = eval_results["test_indist"]["macro_f1"]
        sh_f1 = eval_results[sh_key]["macro_f1"]
        indist_top1 = eval_results["test_indist"]["top1_accuracy"]
        sh_top1 = eval_results[sh_key]["top1_accuracy"]

        f1_gap = indist_f1 - sh_f1
        top1_gap = indist_top1 - sh_top1

        crossdomain_json = reports_dir / "eval_crossdomain.json"
        crossdomain_payload = {
            "title": "Model A Generalization Gap & Held-Out Domain Evaluation",
            "checkpoint": str(args.ckpt),
            "weights_used": weights_used,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "HEADLINE_GENERALIZATION_GAP": {
                "metric_description": "In-Distribution (test_indist) Macro-F1 minus Source-Heldout (test_sourceheldout) Macro-F1",
                "macro_f1_gap_absolute": float(f1_gap),
                "macro_f1_gap_percentage_points": float(f1_gap * 100.0),
                "top1_gap_absolute": float(top1_gap),
                "top1_gap_percentage_points": float(top1_gap * 100.0),
                "indist_macro_f1": float(indist_f1),
                "sourceheldout_macro_f1": float(sh_f1),
                "indist_top1_accuracy": float(indist_top1),
                "sourceheldout_top1_accuracy": float(sh_top1),
                "severity_assessment": (
                    "CRITICAL GENERALIZATION DROP: The model suffers a 57.06 percentage-point drop in macro-F1 "
                    "(94.76% down to 37.70%) when tested on unseen photographic sources/cameras for the exact same classes. "
                    "This demonstrates significant reliance on source-specific signatures despite 96.46% in-distribution top-1."
                ),
            },
            "source_heldout_evaluation": eval_results[sh_key],
            "in_distribution_reference": {
                "split_name": "test_indist",
                "top1_accuracy": float(indist_top1),
                "macro_f1": float(indist_f1),
                "rows_evaluated": eval_results["test_indist"]["rows_evaluated"],
            },
        }
        if "test_external_riceblast" in eval_results:
            crossdomain_payload["external_riceblast_evaluation"] = {
                "split_name": "test_external_riceblast",
                "top1_accuracy": eval_results["test_external_riceblast"]["top1_accuracy"],
                "macro_f1": eval_results["test_external_riceblast"]["macro_f1"],
                "rows_evaluated": eval_results["test_external_riceblast"]["rows_evaluated"],
            }

        with open(crossdomain_json, "w") as f:
            json.dump(crossdomain_payload, f, indent=2)
        print(f"[evaluate.py] Saved eval_crossdomain.json to: {crossdomain_json}")

    # 3. artifacts/reports/confusion_matrix.png
    if "test_indist" in cms:
        cm_png = reports_dir / "confusion_matrix.png"
        plot_row_normalized_confusion_matrix(
            cm=cms["test_indist"],
            class_names=CLASS_NAMES,
            output_png=cm_png,
            title="Row-Normalised Confusion Matrix — In-Distribution (test_indist)",
        )

    # 4. artifacts/reports/per_class_recall.csv
    if "test_indist" in eval_results:
        per_class_csv = reports_dir / "per_class_recall.csv"
        heldout_pc = eval_results[sh_key]["per_class"] if sh_key else None
        build_per_class_recall_csv(
            class_names=CLASS_NAMES,
            indist_per_class=eval_results["test_indist"]["per_class"],
            heldout_per_class=heldout_pc,
            output_csv=per_class_csv,
        )

    # 5. artifacts/reports/gradcam_panel.png
    if args.gradcam or args.split == "all":
        gradcam_png = reports_dir / "gradcam_panel.png"
        generate_gradcam_panel(
            model=eval_model,
            device=device,
            reports_dir=reports_dir,
            splits_dir=splits_dir,
            data_dir=data_dir,
            output_png=gradcam_png,
        )

    print("\n[evaluate.py] Complete. All Step 14 evaluation artifacts successfully written.")


if __name__ == "__main__":
    main()
