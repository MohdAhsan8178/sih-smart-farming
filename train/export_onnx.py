#!/usr/bin/env python3
"""
Step 16 & Step 17: Fused export model and ONNX export.

Builds a fused model accepting raw uint8 BGR (B, H, W, 3) straight from OpenCV
and executing permutation, BGR->RGB conversion, float scaling, and ImageNet
normalization on GPU/engine.

Exports static-shape ONNX (Batch=9, Opset=13) and verifies:
- CHECK 1: RED FLAG IMAGE (strongly asymmetric BGR [20, 40, 200] catching channel swap)
- CHECK 2: End-to-end macro-F1 parity on real cv2.imread images (|F1_fused - F1_ref| < 0.005)
- CHECK 3: ONNX export fidelity (|out_onnx - out_pytorch| < 1e-3)

Reference: docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 16 & STEP 17
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.paths import CKPT, REPORTS
from configs.train_config import ENGINE_BATCH, IMAGE_SIZE, ONNX_OPSET
from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval

# Standard ImageNet statistics (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FusedModel(nn.Module):
    """
    Accepts (B, H, W, 3) BGR (uint8 or float32) straight from OpenCV / camera pipeline.
    Permute + BGR->RGB + scale + normalize on GPU.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Standard ImageNet stats in TRUE RGB order.
        # The channel swap happens on the TENSOR, not on these vectors.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)               # (B, 3, H, W) — still BGR
        x = x[:, [2, 1, 0], :, :].float() / 255.0     # <-- BGR->RGB. NEVER OMIT.
        return self.backbone((x - self.mean) / self.std)


def reference_preprocess(bgr_img: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Reference CPU preprocessing: resize -> BGR2RGB -> /255.0 -> ImageNet norm -> CHW.
    """
    if bgr_img.shape[:2] != (target_size, target_size):
        bgr_img = cv2.resize(bgr_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = normalized.transpose(2, 0, 1)
    return chw


def check_1_red_flag(
    fused_callable,
    backbone: nn.Module,
    batch_size: int = ENGINE_BATCH,
    target_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    CHECK 1 — RED FLAG IMAGE.
    Uniform random noise CANNOT catch a channel swap because uniform noise
    has identical statistics in all three channels.
    Constructs a strongly asymmetric RED image: B=20, G=40, R=200.
    Returns:
        (max_abs_diff_correct_fused, max_abs_diff_swapped_hypothetical)
    """
    bgr = np.zeros((batch_size, target_size, target_size, 3), dtype=np.uint8)
    bgr[..., 2] = 200  # Red
    bgr[..., 1] = 40   # Green
    bgr[..., 0] = 20   # Blue

    # Reference PyTorch output with standard CPU preprocessing
    rgb = cv2.cvtColor(bgr[0], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]  # (1, 3, H, W)
    ref_t = torch.from_numpy(ref).repeat(batch_size, 1, 1, 1)
    if device is not None:
        ref_t = ref_t.to(device)

    with torch.no_grad():
        out_ref = backbone(ref_t).cpu().numpy()

    # Fused output (PyTorch or ONNX callable)
    if isinstance(fused_callable, nn.Module):
        bgr_t = torch.from_numpy(bgr)
        if device is not None:
            bgr_t = bgr_t.to(device)
        with torch.no_grad():
            out_fused = fused_callable(bgr_t).cpu().numpy()
    else:
        # Fused callable is ONNX session or function
        out_fused = fused_callable(bgr)

    diff = float(np.abs(out_fused - out_ref).max())

    # Verify that a deliberate failure (feeding BGR as RGB without swap) yields massive diff
    ref_bgr_order = ((bgr[0].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    ref_bgr_t = torch.from_numpy(ref_bgr_order).repeat(batch_size, 1, 1, 1)
    if device is not None:
        ref_bgr_t = ref_bgr_t.to(device)
    with torch.no_grad():
        out_swapped = backbone(ref_bgr_t).cpu().numpy()
    diff_swapped = float(np.abs(out_swapped - out_ref).max())

    return diff, diff_swapped


def check_2_real_images(
    fused_callable,
    backbone: nn.Module,
    val_csv: Path,
    batch_size: int = ENGINE_BATCH,
    target_size: int = IMAGE_SIZE,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float, int]:
    """
    CHECK 2 — END TO END on real images loaded by cv2.imread.
    Evaluates macro-F1 of fused model vs reference preprocessing pipeline.
    Returns:
        (f1_reference, f1_fused, delta_f1, total_evaluated)
    """
    df = pd.read_csv(val_csv)
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].copy()

    # Map string labels to integer classes
    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    valid_records = []
    for _, row in df.iterrows():
        p = ROOT / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
        if p.exists():
            lbl = label_to_idx.get(row["label"], -1)
            if lbl >= 0:
                valid_records.append((str(p), lbl))

    total = len(valid_records)
    if total == 0:
        raise RuntimeError(f"No valid images found from {val_csv}")

    ref_preds = []
    fused_preds = []
    y_true = []

    # Process in batches of batch_size
    for i in range(0, total, batch_size):
        batch_slice = valid_records[i:i + batch_size]
        actual_len = len(batch_slice)

        bgr_batch = np.zeros((batch_size, target_size, target_size, 3), dtype=np.uint8)
        ref_batch = np.zeros((batch_size, 3, target_size, target_size), dtype=np.float32)

        for j, (p_str, lbl) in enumerate(batch_slice):
            img = cv2.imread(p_str)
            if img is None:
                raise IOError(f"Failed to read image with cv2.imread: {p_str}")
            if img.shape[:2] != (target_size, target_size):
                img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            bgr_batch[j] = img
            ref_batch[j] = reference_preprocess(img, target_size=target_size)
            y_true.append(lbl)

        # Reference forward
        ref_t = torch.from_numpy(ref_batch)
        if device is not None:
            ref_t = ref_t.to(device)
        with torch.no_grad():
            out_ref = backbone(ref_t).cpu().numpy()
        ref_preds.extend(np.argmax(out_ref[:actual_len], axis=1).tolist())

        # Fused forward
        if isinstance(fused_callable, nn.Module):
            bgr_t = torch.from_numpy(bgr_batch)
            if device is not None:
                bgr_t = bgr_t.to(device)
            with torch.no_grad():
                out_fused = fused_callable(bgr_t).cpu().numpy()
        else:
            out_fused = fused_callable(bgr_batch)
        fused_preds.extend(np.argmax(out_fused[:actual_len], axis=1).tolist())

    f1_ref = float(f1_score(y_true, ref_preds, average="macro"))
    f1_fused = float(f1_score(y_true, fused_preds, average="macro"))
    delta_f1 = float(abs(f1_fused - f1_ref))

    return f1_ref, f1_fused, delta_f1, total


def check_3_onnx_fidelity(
    onnx_path: Path,
    fused_model: nn.Module,
    batch_size: int = ENGINE_BATCH,
    target_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    CHECK 3 — ONNX export fidelity.
    Compares ONNX Runtime outputs with PyTorch FusedModel outputs on random uint8 input.
    Returns:
        (max_abs_diff, mean_abs_diff)
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    np.random.seed(42)
    bgr_random = np.random.randint(0, 256, (batch_size, target_size, target_size, 3), dtype=np.uint8)

    # PyTorch output
    bgr_t = torch.from_numpy(bgr_random)
    if device is not None:
        bgr_t = bgr_t.to(device)
    with torch.no_grad():
        out_pytorch = fused_model(bgr_t).cpu().numpy()

    # ONNX output (feed as float32 if ONNX expects float)
    input_type = sess.get_inputs()[0].type
    feed_data = bgr_random.astype(np.float32) if "float" in input_type else bgr_random
    out_onnx = sess.run(None, {input_name: feed_data})[0]

    max_diff = float(np.abs(out_pytorch - out_onnx).max())
    mean_diff = float(np.abs(out_pytorch - out_onnx).mean())

    return max_diff, mean_diff


def export_onnx(
    ckpt_path: Path,
    out_path: Path,
    batch_size: int = ENGINE_BATCH,
    opset_version: int = ONNX_OPSET,
    eval_weights: str = "auto",
    device: Optional[torch.device] = None,
) -> Tuple[FusedModel, Path]:
    """
    Constructs FusedModel and exports to static-shape ONNX (batch, 224, 224, 3) with opset 13.
    """
    dev = device if device is not None else torch.device("cpu")

    backbone = build_model(num_classes=NUM_CLASSES, pretrained=False)
    backbone, weights_used = load_checkpoint_for_eval(ckpt_path, backbone, eval_weights=eval_weights, device=dev)
    backbone.to(dev)
    backbone.eval()

    fused_model = FusedModel(backbone)
    fused_model.to(dev)
    fused_model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.float32, device=dev)

    print(f"[export_onnx] Exporting FusedModel to ONNX (Batch={batch_size}, Opset={opset_version}, Weights={weights_used})...")
    torch.onnx.export(
        fused_model,
        dummy_input,
        str(out_path),
        input_names=["input_bgr"],
        output_names=["logits"],
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"[export_onnx] Successfully exported raw ONNX to: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    return fused_model, out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 16 & 17: FusedModel and ONNX static export.")
    parser.add_argument("--ckpt", type=str, default="artifacts/checkpoints/v3/stage1.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="artifacts/onnx/model_a_fused.onnx", help="Path to exported ONNX model")
    parser.add_argument("--batch", type=int, default=ENGINE_BATCH, help="Static batch size (default: ENGINE_BATCH = 9)")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset (default: ONNX_OPSET = 13)")
    parser.add_argument("--eval_weights", type=str, default="auto", choices=["auto", "ema", "model"], help="Checkpoint weights")
    parser.add_argument("--verify", type=str, default=None, help="Verify existing simplified ONNX model file against PyTorch")
    parser.add_argument("--splits_dir", type=str, default="splits_v3", help="Splits directory containing val.csv")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max val samples for Check 2 (default: all)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    splits_dir = Path(args.splits_dir)
    val_csv = splits_dir / "val.csv"
    device = torch.device(args.device)

    # 1. Verification Mode for an existing ONNX model
    if args.verify:
        onnx_file = Path(args.verify)
        if not onnx_file.exists():
            raise FileNotFoundError(f"Verification target ONNX file not found: {onnx_file}")

        print(f"\n=======================================================")
        print(f"STEP 17 VERIFICATION: Simplified ONNX vs PyTorch")
        print(f"ONNX Model: {onnx_file}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"=======================================================\n")

        # Load reference backbone and fused PyTorch model
        backbone = build_model(num_classes=NUM_CLASSES, pretrained=False)
        backbone, weights_used = load_checkpoint_for_eval(ckpt_path, backbone, eval_weights=args.eval_weights, device=device)
        backbone.to(device)
        backbone.eval()

        fused_pt = FusedModel(backbone)
        fused_pt.to(device)
        fused_pt.eval()

        # Build ONNX callable
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        input_type = sess.get_inputs()[0].type

        def onnx_callable(bgr_batch_np: np.ndarray) -> np.ndarray:
            feed = bgr_batch_np.astype(np.float32) if "float" in input_type and bgr_batch_np.dtype == np.uint8 else bgr_batch_np
            return sess.run(None, {input_name: feed})[0]

        # CHECK 1: RED FLAG IMAGE on ONNX
        print("[Step 17] Executing CHECK 1 (RED FLAG asymmetric image test) on simplified ONNX...")
        diff_1, diff_swapped_1 = check_1_red_flag(
            onnx_callable, backbone=backbone, batch_size=args.batch, device=device
        )
        print(f"  -> ONNX vs Reference PyTorch Max Abs Diff: {diff_1:.6e}")
        print(f"  -> Hypothetical Swapped BGR Max Abs Diff: {diff_swapped_1:.4f}")
        assert diff_1 < 1e-3, f"CHECK 1 FAILED: CHANNEL ORDER MISMATCH on ONNX! Max abs diff: {diff_1}"
        assert diff_swapped_1 > 0.1, f"CHECK 1 SANITY FAILED: Swapped BGR diff too small: {diff_swapped_1}"
        print("  [PASS] CHECK 1: Channel order verified. Red flag test passed on simplified ONNX.")

        # CHECK 2: END TO END on real images loaded by cv2.imread
        print(f"\n[Step 17] Executing CHECK 2 (End-to-End real cv2.imread images) on simplified ONNX...")
        f1_ref, f1_onnx, delta_f1, total_eval = check_2_real_images(
            onnx_callable,
            backbone=backbone,
            val_csv=val_csv,
            batch_size=args.batch,
            max_samples=args.max_val_samples,
            device=device,
        )
        print(f"  -> Total Images Evaluated: {total_eval}")
        print(f"  -> Reference Preprocessing Macro-F1: {f1_ref * 100:.3f}%")
        print(f"  -> Simplified ONNX Macro-F1:         {f1_onnx * 100:.3f}%")
        print(f"  -> Macro-F1 Delta (|F1_onnx - F1_ref|): {delta_f1:.6e} ({delta_f1 * 100:.4f} percentage points)")
        assert delta_f1 < 0.005, f"CHECK 2 FAILED: Macro-F1 delta exceeds 0.005: {delta_f1}"
        print("  [PASS] CHECK 2: End-to-end macro-F1 parity confirmed on simplified ONNX.")

        # CHECK 3: ONNX export fidelity
        print(f"\n[Step 17] Executing CHECK 3 (ONNX export numerical fidelity vs PyTorch FusedModel)...")
        max_diff_3, mean_diff_3 = check_3_onnx_fidelity(
            onnx_file, fused_model=fused_pt, batch_size=args.batch, device=device
        )
        print(f"  -> Max Absolute Difference:  {max_diff_3:.6e}")
        print(f"  -> Mean Absolute Difference: {mean_diff_3:.6e}")
        assert max_diff_3 < 1e-3, f"CHECK 3 FAILED: Numerical fidelity mismatch: {max_diff_3}"
        print("  [PASS] CHECK 3: Numerical fidelity verified against PyTorch FusedModel.")

        print(f"\n=======================================================")
        print(f"ALL THREE CHECKS PASSED AGAINST SIMPLIFIED ONNX!")
        print(f"=======================================================\n")
        return

    # 2. Export Mode: Export FusedModel to ONNX
    print(f"\n=======================================================")
    print(f"STEP 16: Building and Verifying FusedModel in PyTorch")
    print(f"Checkpoint: {ckpt_path}")
    print(f"=======================================================\n")

    backbone = build_model(num_classes=NUM_CLASSES, pretrained=False)
    backbone, weights_used = load_checkpoint_for_eval(ckpt_path, backbone, eval_weights=args.eval_weights, device=device)
    backbone.to(device)
    backbone.eval()

    fused_model = FusedModel(backbone)
    fused_model.to(device)
    fused_model.eval()

    # CHECK 1 on PyTorch FusedModel
    print("[Step 16] Executing CHECK 1 (RED FLAG asymmetric image test) on PyTorch FusedModel...")
    diff_pt, diff_swapped_pt = check_1_red_flag(
        fused_model, backbone=backbone, batch_size=args.batch, device=device
    )
    print(f"  -> PyTorch Fused vs Reference Max Abs Diff: {diff_pt:.6e}")
    print(f"  -> Hypothetical Swapped BGR Max Abs Diff:   {diff_swapped_pt:.4f}")
    assert diff_pt < 1e-3, f"CHECK 1 FAILED: CHANNEL ORDER MISMATCH on PyTorch FusedModel! Max abs diff: {diff_pt}"
    print("  [PASS] CHECK 1: Channel order verified on PyTorch FusedModel.")

    # CHECK 2 on PyTorch FusedModel
    print(f"\n[Step 16] Executing CHECK 2 (End-to-End real cv2.imread images) on PyTorch FusedModel...")
    f1_ref, f1_pt, delta_f1_pt, total_eval_pt = check_2_real_images(
        fused_model,
        backbone=backbone,
        val_csv=val_csv,
        batch_size=args.batch,
        max_samples=args.max_val_samples,
        device=device,
    )
    print(f"  -> Total Images Evaluated: {total_eval_pt}")
    print(f"  -> Reference Preprocessing Macro-F1: {f1_ref * 100:.3f}%")
    print(f"  -> PyTorch FusedModel Macro-F1:      {f1_pt * 100:.3f}%")
    print(f"  -> Macro-F1 Delta (|F1_pt - F1_ref|): {delta_f1_pt:.6e}")
    assert delta_f1_pt < 0.005, f"CHECK 2 FAILED: Macro-F1 delta exceeds 0.005: {delta_f1_pt}"
    print("  [PASS] CHECK 2: End-to-end macro-F1 parity confirmed on PyTorch FusedModel.")

    # Export to ONNX
    print(f"\n=======================================================")
    print(f"STEP 17: Exporting to Static-Shape ONNX (Opset={args.opset}, Batch={args.batch})")
    print(f"=======================================================\n")
    out_file = Path(args.out)
    export_onnx(
        ckpt_path=ckpt_path,
        out_path=out_file,
        batch_size=args.batch,
        opset_version=args.opset,
        eval_weights=args.eval_weights,
        device=device,
    )


if __name__ == "__main__":
    main()
