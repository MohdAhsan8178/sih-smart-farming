#!/usr/bin/env python3
"""
scripts/audit_g4_2_parity.py

G4.2 Parity Check:
Compares stage1.pt (EMA weights) vs artifacts/onnx/model_a_fused.onnx
on 50 images from splits_v3/test_indist.csv.
Computes max absolute difference across all logits.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval
from train.export_onnx import reference_preprocess

CKPT_PATH = ROOT / "artifacts/checkpoints/v3/stage1.pt"
ONNX_PATH = ROOT / "artifacts/onnx/model_a_fused.onnx"
CSV_PATH = ROOT / "splits_v3/test_indist.csv"

def main():
    print("=" * 80)
    print("G4.2 ONNX vs PyTorch EMA Parity Check")
    print(f"PyTorch Checkpoint: {CKPT_PATH}")
    print(f"ONNX Model        : {ONNX_PATH}")
    print(f"Test Split        : {CSV_PATH}")
    print("=" * 80)

    # 1. Load PyTorch model with EMA weights
    device = torch.device("cpu")
    model = build_model(num_classes=29, pretrained=False)
    model, weights_used = load_checkpoint_for_eval(CKPT_PATH, model, eval_weights="ema", device=device)
    model.eval()
    print(f"PyTorch model successfully loaded with weights: {weights_used}")

    # 2. Load ONNX model
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    print(f"ONNX session loaded. Input: {input_name} {input_shape}, Output: {output_name}")

    # 3. Load 50 test images
    df = pd.read_csv(CSV_PATH)

    valid_rows = []
    for _, row in df.iterrows():
        p = ROOT / row["path"]
        if p.exists():
            valid_rows.append(p)
        if len(valid_rows) >= 50:
            break

    print(f"Collected {len(valid_rows)} valid images for parity check.")

    diffs = []
    max_abs_diff_global = 0.0

    for idx, p in enumerate(valid_rows):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if bgr.shape[:2] != (224, 224):
            bgr_resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            bgr_resized = bgr

        # PyTorch reference forward
        pt_input_np = reference_preprocess(bgr_resized, target_size=224) # (1, 3, 224, 224)
        tensor_in = torch.from_numpy(pt_input_np).unsqueeze(0).to(device)
        with torch.no_grad():
            pt_out = model(tensor_in).cpu().numpy() # (1, 29)
            
        # ONNX forward (FusedModel expects NHWC BGR float32)
        onnx_in = bgr_resized.astype(np.float32)[np.newaxis, ...] # (1, 224, 224, 3)
        if isinstance(input_shape[0], int) and input_shape[0] == 9:
            # Batch size fixed to 9
            batch_9 = np.repeat(onnx_in, 9, axis=0) # (9, 224, 224, 3)
            onnx_out = session.run([output_name], {input_name: batch_9})[0] # (9, 29)
            onnx_logits = onnx_out[0:1] # (1, 29)
        else:
            onnx_out = session.run([output_name], {input_name: onnx_in})[0]
            onnx_logits = onnx_out

        abs_diff = np.abs(pt_out - onnx_logits)
        max_d = float(np.max(abs_diff))
        diffs.append(max_d)
        if max_d > max_abs_diff_global:
            max_abs_diff_global = max_d

    print("-" * 80)
    print(f"Evaluated {len(diffs)} images.")
    print(f"Mean Max Abs Diff : {np.mean(diffs):.8f}")
    print(f"Global Max Abs Diff: {max_abs_diff_global:.8f}")
    print(f"Parity Verdict     : {'EXCELLENT (< 1e-4)' if max_abs_diff_global < 1e-4 else 'DISCREPANCY'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
