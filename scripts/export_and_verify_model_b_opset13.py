#!/usr/bin/env python3
"""
scripts/export_and_verify_model_b_opset13.py

Re-exports Model B at ONNX opset 13 for TensorRT 8.2 compatibility on Jetson Nano.
Verifies with onnx.checker.
Performs 50-crop parity test between PyTorch model_b_best.pt and ONNX Runtime.
Eliminates duplicate ONNX by symlinking model_b.onnx -> model_b_calibrated.onnx.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import onnx
import onnxruntime as ort
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "artifacts" / "checkpoints" / "model_b_best.pt"
ONNX_CALIBRATED_PATH = ROOT / "artifacts" / "onnx" / "model_b_calibrated.onnx"
ONNX_SYMLINK_PATH = ROOT / "artifacts" / "onnx" / "model_b.onnx"
MANIFEST_PATH = ROOT / "splits" / "model_b_manifest_v4.csv"
PATCHES_DIR = ROOT / "data" / "processed" / "model_b_patches_v4"

print("=================================================================")
print("PART C: NANO DEPLOYABILITY - RE-EXPORT MODEL B AT OPSET 13")
print("=================================================================")

# 1. Load PyTorch model
print(f"Loading checkpoint: {CKPT_PATH}")
device = torch.device("cpu")
model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=3)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2. Export to ONNX opset 13
print(f"Exporting to ONNX at opset 13: {ONNX_CALIBRATED_PATH}")
if ONNX_CALIBRATED_PATH.exists():
    ONNX_CALIBRATED_PATH.unlink()
data_file = ONNX_CALIBRATED_PATH.with_suffix(".onnx.data")
if data_file.exists():
    data_file.unlink()
if ONNX_SYMLINK_PATH.is_symlink() or ONNX_SYMLINK_PATH.exists():
    ONNX_SYMLINK_PATH.unlink()

dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_CALIBRATED_PATH),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=13,
    do_constant_folding=True,
    dynamo=False,
)
print(f"Export completed. File size: {ONNX_CALIBRATED_PATH.stat().st_size / 1e6:.2f} MB")

# 3. ONNX Checker
print("\n--- ONNX CHECKER ---")
onnx_model = onnx.load(str(ONNX_CALIBRATED_PATH))
onnx.checker.check_model(onnx_model)
print("onnx.checker.check_model passed successfully!")
opsets = {op.domain or "ai.onnx": op.version for op in onnx_model.opset_import}
print(f"IR Version    : {onnx_model.ir_version}")
print(f"Opset Imports : {opsets}")
print(f"Producer      : {onnx_model.producer_name} {onnx_model.producer_version}")

# 4. 50-crop parity test
print("\n--- 50-CROP PARITY TEST (PyTorch vs ONNX Runtime) ---")
df = pd.read_csv(MANIFEST_PATH)
val_df = df[df["split"] == "val"].reset_index(drop=True)
sample_50 = val_df.sample(n=50, random_state=42).reset_index(drop=True)

eval_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

ort_session = ort.InferenceSession(str(ONNX_CALIBRATED_PATH), providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name

diffs = []
for i, row in sample_50.iterrows():
    p = PATCHES_DIR / row["class_name"] / f"{row['crop_id']}.png"
    img = np.array(Image.open(p).convert("RGB"))
    t = eval_transform(image=img)["image"].unsqueeze(0) # (1, 3, 64, 64)
    
    with torch.no_grad():
        pt_logits = model(t).numpy()[0]
    
    ort_logits = ort_session.run(None, {input_name: t.numpy()})[0][0]
    
    diff = np.abs(pt_logits - ort_logits)
    max_d = float(np.max(diff))
    diffs.append(max_d)
    
    if i < 5:
        print(f"  Crop {i:2d} ({row['class_name']}): PT={pt_logits}, ORT={ort_logits}, MaxDiff={max_d:.2e}")

diffs = np.array(diffs)
print(f"\nTotal test crops: {len(diffs)}")
print(f"Max absolute diff across all 50 crops: {diffs.max():.6e}")
print(f"Mean absolute diff across all 50 crops: {diffs.mean():.6e}")
print(f"Min absolute diff across all 50 crops: {diffs.min():.6e}")
assert diffs.max() < 1e-3, f"Fidelity check failed! Max diff {diffs.max()} >= 1e-3"
print("Parity test PASSED: max diff < 1e-3 threshold!")

# 5. C3: Eliminate duplicate ONNX by creating symlink
print("\n--- C3. ELIMINATE DUPLICATE ONNX ---")
rel_target = "model_b_calibrated.onnx"
os.symlink(rel_target, str(ONNX_SYMLINK_PATH))
print(f"Created symlink: {ONNX_SYMLINK_PATH} -> {rel_target}")
print(f"Symlink verification: is_symlink={ONNX_SYMLINK_PATH.is_symlink()}, points_to={os.readlink(str(ONNX_SYMLINK_PATH))}")

# Check onnx directory contents
print("\nDirectory contents of artifacts/onnx/:")
for p in sorted((ROOT / "artifacts" / "onnx").iterdir()):
    if p.is_symlink():
        print(f"  {p.name:30s} -> {os.readlink(str(p))}")
    else:
        print(f"  {p.name:30s} {p.stat().st_size / 1e6:.2f} MB")
