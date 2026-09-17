# Local MacBook Testing Harness (`AI_macbook_test/`)

## Overview & Scope
This directory contains a self-contained local testing harness designed specifically for **macOS / Apple Silicon** (or CPU) to visually sanity-check the behavior of the trained crop disease model before field deployment.

> [!IMPORTANT]
> **LOCAL DEVELOPMENT TOOL ONLY — NEVER DEPLOYED TO JETSON NANO**
> - This testing harness runs PyTorch directly on macOS using Apple Silicon MPS or CPU. It **never runs on and is never deployed to the NVIDIA Jetson Nano**.
> - The Jetson Nano edge deployment path uses TensorRT engine compilation (`artifacts/engines/`) and `edge/trt_classifier.py`.
> - This visual sanity tool **does not replace** the official benchmark evaluation metrics produced by `train/evaluate.py` across the test splits (`splits/` or `splits_v3/`).

---

## 1. Installation

This harness uses a modern, lightweight set of dependencies suitable for macOS and Python 3.9–3.13 without the Jetson Nano / Python 3.6 legacy constraints.

To install in a local virtual environment:
```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
pip install -r AI_macbook_test/requirements-macbook-test.txt
```

---

## 2. Spatial Tiling & Aggregation (Real Edge Pipeline Parity)

Whole-frame inference often dilutes early-stage or localized lesions because resizing high-resolution images directly to 224×224 blurs fine disease symptoms.

To match the real deployment pipeline:
1. **3×3 Overlapping Grid**: The frame is tiled into 9 overlapping `320×320` patches with `20%` overlap (step `256` px, footprint `832×832` px).
2. **Batched Inference**: All 9 tiles are processed through the model to obtain per-tile logits and calibrated softmax probabilities (`T_CAL = 0.597`).
3. **MIL Aggregation**: [`core.aggregate.aggregate_frame()`](../core/aggregate.py) performs Multiple Instance Learning (MIL) max-pooling over tiles:
   - Any tile exceeding `TAU_DISEASE = 0.40` by `TAU_MARGIN = 0.15` over healthy probabilities carries the frame into `DISEASE`.
   - Frames dominated by healthy leaves resolve to `HEALTHY`.
   - Frames dominated by background resolve to `NOT_CROP`.
   - Frames lacking sufficient evidence or confidence resolve to `UNCERTAIN`.
4. **Side-by-Side HUD & Output**: Both whole-frame and tiled+aggregated results are computed and displayed side-by-side for comparison.

---

## 3. Usage

### A. Live Webcam Mode
Runs the real-time inference loop using the MacBook's built-in webcam (`cv2.VideoCapture(0)`).

```bash
python AI_macbook_test/webcam_test.py
```

**Key Controls in Live Window:**
- `q` or `ESC`: Quit the live camera loop and release resources.
- `s`: Save an annotated snapshot frame to `AI_macbook_test/snapshots/snapshot_<timestamp>.jpg`.
- `t`: Toggle Test-Time Augmentation (TTA) on/off in real-time.

**Options:**
```bash
# Enable Test-Time Augmentation (3 views per tile)
python AI_macbook_test/webcam_test.py --tta

# Run whole-frame only (disable 3x3 spatial tiling)
python AI_macbook_test/webcam_test.py --no-tiling
```

### B. Single Static Image Mode (`--image`)
Evaluates a single image without opening a camera stream, displaying both whole-frame and tiled+aggregated results:

```bash
# Standard comparison run (whole-frame vs 3x3 tiled)
python AI_macbook_test/webcam_test.py --image AI_macbook_test/sample_images/train/train_rice_bacterial_leaf_blight.jpg

# Enable Test-Time Augmentation (TTA: orig, h-flip, 1.1x scale)
python AI_macbook_test/webcam_test.py --image AI_macbook_test/sample_images/test_crossdomain/cross_sugarcane_rust.jpg --tta

# Disable tiling (run whole-frame single forward pass only)
python AI_macbook_test/webcam_test.py --image AI_macbook_test/sample_images/test_indist/indist_wheat_healthy.png --no-tiling
```

**Additional Flags:**
- `--save-annotated <path>`: Save an image with the dual-column HUD overlay banner.
- `--show`: Display an OpenCV pop-up window showing the annotated image.

---

## 4. Test-Time Augmentation (TTA)

When `--tta` is enabled, the model evaluates 3 views per tile/frame:
1. **Original View**: Standard deterministic `eval_transform()` crop.
2. **Horizontally Flipped View**: `cv2.flip(img, 1)` to provide horizontal invariance.
3. **Scaled View**: Resized 1.1x then center-cropped to 224 to provide zoomed inspection of lesion textures.

Logits across the 3 views are averaged before decision gates and aggregation. Output explicitly reports `TTA Mode: ENABLED (3 views averaged)` for full transparency.

---

## 5. Sample Test Images

Subfolders under `AI_macbook_test/sample_images/`:
- `train/`: 4 images copied from `splits/train.csv` (rice blight, sugarcane red rot, rice normal, not_crop).
- `test_indist/`: 4 images copied from `splits/test_indist.csv` (rice brown spot, sugarcane mosaic, wheat healthy, not_crop).
- `test_crossdomain/`: 4 images copied from `splits/test_crossdomain.csv` (sugarcane rust, wheat stripe rust, sugarcane healthy, rice blast).
- `external_unseen/`: 4 images from `data/raw/openset/` that do not appear in any `splits/*.csv` split (apple rust leaf, broadleaf weed, Philippines Bakanae, sugarcane ring spot).

Detailed paths, labels, and file metadata are recorded in [`AI_macbook_test/sample_images/manifest.csv`](sample_images/manifest.csv).
