#!/usr/bin/env python3
"""
Local MacBook Webcam and Single-Image Testing Harness.
Visually sanity-checks the trained model against the MacBook's webcam or static images.

Features:
1. Loads the trained PyTorch checkpoint directly (artifacts/checkpoints/v3/stage1.pt)
2. Runs on Apple Silicon (MPS) or CPU (not CUDA)
3. Strictly uses official eval_transform() from train/transforms.py (as in train/dataset.py)
4. Whole-frame single forward pass + core.rejection.decide()
5. Real deployment 3x3 spatial tiling (320px tiles, 20% overlap) + core.aggregate.aggregate_frame()
6. Optional Test-Time Augmentation (--tta): horizontal flip + 1.1x scale center-cropped, logit-averaged
7. Dual-column HUD overlay and side-by-side stdout output for direct comparison
8. CLI flags: --image <path>, --tta, --no-tiling, --camera-index, --device, --eval-weights, --save-annotated, --show

NOTE: This is a local development/demo tool only. It is never deployed to the Jetson Nano
and does not replace official evaluation metrics in train/evaluate.py.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
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
from configs.paths import CKPT
from configs.train_config import (
    IMAGE_SIZE,
    N_TILES,
    T_CAL,
    TAU_CONF,
    TAU_DISEASE,
    TAU_ENERGY,
    TAU_MARGIN,
    TAU_PRIOR,
    TILE_GRID,
    TILE_OVERLAP,
    TILE_SIZE,
)
from core.aggregate import aggregate_frame
from core.rejection import decide, open_set_energy, softmax
from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval
from train.transforms import eval_transform


def select_device(user_device: Optional[str] = None) -> torch.device:
    """Selects mps if available on Apple Silicon, else cpu."""
    if user_device and user_device != "auto":
        return torch.device(user_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_log_priors(splits_csv: Optional[Path] = None) -> np.ndarray:
    """
    Computes empirical training log-priors for decide() and posthoc_logit_adjust.
    Uses splits_v3/train.csv if present, else splits/train.csv.
    """
    if splits_csv is None:
        cand_v3 = REPO_ROOT / "splits_v3" / "train.csv"
        cand_v1 = REPO_ROOT / "splits" / "train.csv"
        splits_csv = cand_v3 if cand_v3.exists() else cand_v1

    if splits_csv.exists():
        df = pd.read_csv(splits_csv)
        counts = df["label"].value_counts()
        priors = np.array([counts.get(name, 1) for name in CLASS_NAMES], dtype=np.float64)
        priors = priors / priors.sum()
        return np.log(priors)
    else:
        return np.full(NUM_CLASSES, -np.log(NUM_CLASSES), dtype=np.float64)


def extract_3x3_tiles(
    img_bgr: np.ndarray,
    tile_size: int = TILE_SIZE,      # 320
    grid_size: int = TILE_GRID,      # 3
    overlap: float = TILE_OVERLAP,   # 0.20
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Extracts deterministic 3x3 grid of 320x320 tiles with 20% overlap.

    Tiling geometry:
      - tile_size = 320 px
      - overlap = 20% -> step = tile_size * (1 - overlap) = 256 px
      - 3 tiles across each dimension require a footprint of:
        tile_size + (grid_size - 1) * step = 320 + 2 * 256 = 832 px
        Offsets: [0, 256, 512]
        Tile 0: [0, 320]
        Tile 1: [256, 576]   (overlap with Tile 0: 64 px = 20%)
        Tile 2: [512, 832]   (overlap with Tile 1: 64 px = 20%)

    If the input image is not 832x832, it is resized to (832, 832) using INTER_LINEAR
    so the 3x3 grid covers the complete field of view uniformly.

    NOTE on edge/tiler.py: edge/tiler.py is Step 20 in the implementation plan
    and has not been authored in this branch. Field-specific ExG vegetation-mask
    filtering is intentionally omitted here for local test photos and webcam frames.
    """
    h, w = img_bgr.shape[:2]
    step = int(tile_size * (1.0 - overlap))  # 256
    footprint = tile_size + (grid_size - 1) * step  # 832

    if (h, w) != (footprint, footprint):
        canvas = cv2.resize(img_bgr, (footprint, footprint), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = img_bgr

    offsets = [i * step for i in range(grid_size)]  # [0, 256, 512]

    tiles = []
    boxes = []
    for y in offsets:
        for x in offsets:
            tile = canvas[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            boxes.append((x, y, x + tile_size, y + tile_size))

    return tiles, boxes


def run_decision(
    logits_np: np.ndarray,
    log_priors: np.ndarray,
    tau_energy: float = TAU_ENERGY,
    t_cal: float = T_CAL,
    tau_conf: float = TAU_CONF,
    tau_prior: float = TAU_PRIOR,
    tau_disease: float = TAU_DISEASE,
    tau_margin: float = TAU_MARGIN,
) -> Dict[str, Union[str, float, int]]:
    """
    Applies core.rejection.decide() with calibrated thresholds and maps
    to the 4 deployment states: DISEASE, HEALTHY, NOT_CROP, UNCERTAIN.
    """
    dec = decide(
        logits_np,
        crop_cols=CROP_COLS,
        notcrop_col=NOTCROP_COL,
        log_priors=log_priors,
        tau_energy=tau_energy,
        T_cal=t_cal,
        tau_conf=tau_conf,
        tau_prior=tau_prior,
    )[0]

    probs = softmax(logits_np, T=t_cal)[0]
    raw_state = dec["state"]
    energy = float(dec["energy"])

    if raw_state == "NOT_CROP":
        return {
            "state": "NOT_CROP",
            "class_name": CLASS_NAMES[NOTCROP_COL],
            "class_id": int(NOTCROP_COL),
            "confidence": float(probs[NOTCROP_COL]),
            "energy": energy,
            "raw_state": raw_state,
        }

    if raw_state in ("UNKNOWN", "ABSTAIN"):
        top_cid = int(np.argmax(probs))
        return {
            "state": "UNCERTAIN",
            "class_name": CLASS_NAMES[top_cid],
            "class_id": top_cid,
            "confidence": float(dec["conf"]),
            "energy": energy,
            "raw_state": raw_state,
        }

    # raw_state == "OK": passed open-set energy and confidence gates
    d_best = float(probs[DISEASE_COLS].max())
    d_arg = int(DISEASE_COLS[probs[DISEASE_COLS].argmax()])
    h_best = float(probs[HEALTHY_COLS].max())
    h_arg = int(HEALTHY_COLS[probs[HEALTHY_COLS].argmax()])

    if (d_best >= tau_disease) and ((d_best - h_best) >= tau_margin):
        return {
            "state": "DISEASE",
            "class_name": CLASS_NAMES[d_arg],
            "class_id": d_arg,
            "confidence": d_best,
            "energy": energy,
            "raw_state": raw_state,
        }
    elif h_best > d_best:
        return {
            "state": "HEALTHY",
            "class_name": CLASS_NAMES[h_arg],
            "class_id": h_arg,
            "confidence": h_best,
            "energy": energy,
            "raw_state": raw_state,
        }
    else:
        return {
            "state": "UNCERTAIN",
            "class_name": CLASS_NAMES[d_arg],
            "class_id": d_arg,
            "confidence": d_best,
            "energy": energy,
            "raw_state": raw_state,
        }


def forward_with_optional_tta(
    bgr_list: List[np.ndarray],
    model: nn.Module,
    transform,
    device: torch.device,
    use_tta: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Runs model inference on a list of BGR images, with optional 3-view TTA:
    - View 1: Original image
    - View 2: Horizontal flip (cv2.flip(img, 1))
    - View 3: Scaled 1.1x (cv2.resize 1.1x, center-cropped to 224 by eval_transform)

    Averages logits across views when use_tta is True.
    Returns:
      (logits_np, n_views_averaged)
    """
    n_images = len(bgr_list)
    if n_images == 0:
        return np.empty((0, NUM_CLASSES), dtype=np.float64), 1

    # Prepare standard views
    orig_tensors = []
    for bgr in bgr_list:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_tensors.append(transform(image=rgb)["image"])
    batch_orig = torch.stack(orig_tensors, dim=0).to(device)

    if not use_tta:
        with torch.no_grad():
            logits = model(batch_orig)
        return logits.detach().cpu().numpy().astype(np.float64), 1

    # TTA View 2: Horizontally flipped
    flip_tensors = []
    for bgr in bgr_list:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_flip = cv2.flip(rgb, 1)
        flip_tensors.append(transform(image=rgb_flip)["image"])
    batch_flip = torch.stack(flip_tensors, dim=0).to(device)

    # TTA View 3: Scaled 1.1x
    scale_tensors = []
    for bgr in bgr_list:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scaled = cv2.resize(rgb, (int(w * 1.1), int(h * 1.1)), interpolation=cv2.INTER_LINEAR)
        scale_tensors.append(transform(image=scaled)["image"])
    batch_scale = torch.stack(scale_tensors, dim=0).to(device)

    # Batched forward pass for all 3 views
    batch_all = torch.cat([batch_orig, batch_flip, batch_scale], dim=0)
    with torch.no_grad():
        logits_all = model(batch_all)

    logits_orig = logits_all[:n_images]
    logits_flip = logits_all[n_images : 2 * n_images]
    logits_scale = logits_all[2 * n_images :]

    logits_avg = (logits_orig + logits_flip + logits_scale) / 3.0
    return logits_avg.detach().cpu().numpy().astype(np.float64), 3


def predict_pipeline(
    bgr_img: np.ndarray,
    model: nn.Module,
    transform,
    device: torch.device,
    log_priors: np.ndarray,
    use_tiling: bool = True,
    use_tta: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full evaluation pipeline:
    1. Whole-frame single forward pass + decide()
    2. (Optional) 3x3 spatial tiling (9 tiles) + aggregate_frame()
    Returns dictionary with results for both paths and latency metrics.
    """
    # ----------------------------------------------------
    # 1. Whole-frame path
    # ----------------------------------------------------
    t0_wf = time.perf_counter()
    wf_logits, wf_views = forward_with_optional_tta(
        [bgr_img], model, transform, device, use_tta=use_tta
    )
    wf_result = run_decision(wf_logits, log_priors)
    t1_wf = time.perf_counter()
    wf_latency_ms = (t1_wf - t0_wf) * 1000.0

    output = {
        "whole_frame": wf_result,
        "wf_latency_ms": wf_latency_ms,
        "wf_views": wf_views,
        "tiled": None,
        "tiled_latency_ms": 0.0,
        "tiled_views": 0,
        "tile_breakdown": None,
    }

    if not use_tiling:
        return output

    # ----------------------------------------------------
    # 2. Tiled + MIL aggregation path
    # ----------------------------------------------------
    t0_tiled = time.perf_counter()
    tiles, boxes = extract_3x3_tiles(bgr_img)
    tile_logits, tiled_views = forward_with_optional_tta(
        tiles, model, transform, device, use_tta=use_tta
    )

    # Compute per-tile probabilities with T_CAL
    tile_probs = softmax(tile_logits, T=T_CAL)
    tile_energies = open_set_energy(tile_logits, CROP_COLS)

    # Run per-tile decisions
    tile_decisions = [
        run_decision(tile_logits[i : i + 1], log_priors) for i in range(len(tiles))
    ]

    # Call MIL frame-level aggregation
    agg_state, agg_cid, agg_score = aggregate_frame(
        tile_probs=tile_probs,
        healthy_cols=HEALTHY_COLS,
        notcrop_col=NOTCROP_COL,
        tau_disease=TAU_DISEASE,
        tau_margin=TAU_MARGIN,
        min_tiles=2,
        tau_healthy=0.50,
        notcrop_frac=0.50,
    )

    agg_class_name = CLASS_NAMES[agg_cid] if agg_cid is not None else "uncertain"

    n_disease = sum(1 for d in tile_decisions if d["state"] == "DISEASE")
    n_healthy = sum(1 for d in tile_decisions if d["state"] == "HEALTHY")
    n_notcrop = sum(1 for d in tile_decisions if d["state"] == "NOT_CROP")
    n_uncertain = sum(1 for d in tile_decisions if d["state"] == "UNCERTAIN")

    t1_tiled = time.perf_counter()
    tiled_latency_ms = (t1_tiled - t0_tiled) * 1000.0

    output["tiled"] = {
        "state": agg_state,
        "class_name": agg_class_name,
        "class_id": agg_cid,
        "confidence": float(agg_score),
        "mean_energy": float(tile_energies.mean()),
        "min_energy": float(tile_energies.min()),
        "tile_decisions": tile_decisions,
    }
    output["tiled_latency_ms"] = tiled_latency_ms
    output["tiled_views"] = tiled_views
    output["tile_breakdown"] = {
        "disease": n_disease,
        "healthy": n_healthy,
        "not_crop": n_notcrop,
        "uncertain": n_uncertain,
        "total": len(tiles),
    }

    return output


def draw_side_by_side_overlay(
    frame: np.ndarray,
    res: Dict[str, Any],
    fps: float = 0.0,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Renders dual-column HUD overlay banner on the video frame:
    Column 1: Whole-Frame Result
    Column 2: 3x3 Tiled + Aggregated Result
    """
    annotated = frame.copy()
    wf = res["whole_frame"]
    tiled = res.get("tiled")

    def get_colors(state: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        if state == "HEALTHY":
            return (46, 204, 113), (39, 174, 96)    # Emerald green
        elif state == "DISEASE":
            return (52, 73, 235), (41, 55, 195)     # Coral / Red
        elif state == "NOT_CROP":
            return (200, 160, 50), (160, 120, 30)   # Blue-gray
        else:
            return (0, 165, 255), (0, 140, 215)     # Amber / Orange

    wf_box_col, wf_hdr_col = get_colors(wf["state"])

    # Draw dark backing card (540 px wide x 175 px high)
    overlay = annotated.copy()
    card_w = 550 if tiled else 460
    cv2.rectangle(overlay, (10, 10), (10 + card_w, 180), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)

    # Top status bar
    tta_str = "TTA: 3 views" if use_tta else "TTA: OFF"
    fps_str = f"FPS: {fps:.1f}" if fps > 0 else "Static"
    cv2.rectangle(annotated, (10, 10), (10 + card_w, 38), (35, 35, 35), -1)
    cv2.putText(
        annotated,
        f"MACBOOK MONITOR  |  {fps_str}  |  {tta_str}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if not tiled:
        # Single column overlay (tiling disabled)
        cv2.rectangle(annotated, (15, 45), (10 + card_w - 5, 75), wf_hdr_col, -1)
        cv2.rectangle(annotated, (15, 45), (10 + card_w - 5, 175), wf_box_col, 2)
        cv2.putText(
            annotated,
            f"STATE: {wf['state']}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Class: {wf['class_name']}",
            (25, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Conf: {wf['confidence']*100:.1f}%  |  Energy: {wf['energy']:.2f}",
            (25, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Time: {res['wf_latency_ms']:.1f}ms  |  Gate: {wf['raw_state']}",
            (25, 162),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return annotated

    # Dual column overlay (Whole-Frame vs 3x3 Tiled)
    tiled_box_col, tiled_hdr_col = get_colors(tiled["state"])
    col_w = 260

    # Column 1: Whole Frame
    cv2.rectangle(annotated, (15, 44), (15 + col_w, 70), wf_hdr_col, -1)
    cv2.rectangle(annotated, (15, 44), (15 + col_w, 175), wf_box_col, 2)
    cv2.putText(
        annotated,
        f"[WF] {wf['state']}",
        (22, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{wf['class_name'][:20]}",
        (22, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Conf: {wf['confidence']*100:.1f}%",
        (22, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Energy: {wf['energy']:.2f}",
        (22, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Time: {res['wf_latency_ms']:.1f}ms",
        (22, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Column 2: 3x3 Tiled
    x2 = 25 + col_w
    cv2.rectangle(annotated, (x2, 44), (x2 + col_w, 70), tiled_hdr_col, -1)
    cv2.rectangle(annotated, (x2, 44), (x2 + col_w, 175), tiled_box_col, 2)
    cv2.putText(
        annotated,
        f"[TILED] {tiled['state']}",
        (x2 + 7, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{tiled['class_name'][:20]}",
        (x2 + 7, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Score: {tiled['confidence']*100:.1f}%",
        (x2 + 7, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    bd = res["tile_breakdown"]
    cv2.putText(
        annotated,
        f"Tiles: {bd['disease']}D/{bd['healthy']}H/{bd['not_crop']}N/{bd['uncertain']}U",
        (x2 + 7, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Time: {res['tiled_latency_ms']:.1f}ms",
        (x2 + 7, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    return annotated


def run_single_image(
    image_path: Path,
    model: nn.Module,
    transform,
    device: torch.device,
    log_priors: np.ndarray,
    weights_used: str,
    use_tiling: bool = True,
    use_tta: bool = False,
    show_window: bool = False,
    save_annotated: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs single-image evaluation and prints formatted side-by-side output."""
    if not image_path.exists():
        raise FileNotFoundError(f"Target image does not exist: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    res = predict_pipeline(
        bgr,
        model,
        transform,
        device,
        log_priors,
        use_tiling=use_tiling,
        use_tta=use_tta,
    )

    wf = res["whole_frame"]
    tiled = res.get("tiled")

    print("=" * 70)
    print("MACBOOK MODEL TEST: SINGLE IMAGE INFERENCE")
    print("=" * 70)
    print(f"Image Path     : {image_path}")
    print(f"Device         : {device}")
    print(f"Weights Used   : {weights_used}")
    tta_desc = f"ENABLED ({res['wf_views']} views averaged: orig, h-flip, 1.1x scale)" if use_tta else "DISABLED (1 view)"
    print(f"TTA Mode       : {tta_desc}")
    print("-" * 70)

    # [1] Whole-frame output
    print("[1] WHOLE-FRAME INFERENCE (Single forward pass):")
    print(f"    Predicted Class: {wf['class_name']} (ID: {wf['class_id']})")
    print(f"    Decide State   : {wf['state']} (Gate: {wf['raw_state']})")
    print(f"    Confidence     : {wf['confidence'] * 100:.2f}%")
    print(f"    Open-Set Energy: {wf['energy']:.4f} (Threshold TAU_ENERGY: {TAU_ENERGY})")
    print(f"    Latency        : {res['wf_latency_ms']:.2f} ms")

    # [2] Tiled output (if enabled)
    if tiled:
        bd = res["tile_breakdown"]
        print(f"\n[2] TILED + AGGREGATED INFERENCE (3x3 grid, 320px, 20% overlap, {bd['total']} tiles):")
        print(f"    Aggregated State: {tiled['state']}")
        print(f"    Aggregated Class: {tiled['class_name']} (ID: {tiled['class_id']})")
        print(f"    Confidence Score: {tiled['confidence'] * 100:.2f}%")
        print(f"    Tile Energies   : Mean = {tiled['mean_energy']:.4f}, Min = {tiled['min_energy']:.4f}")
        print(f"    Tile Breakdown  : {bd['disease']} Disease, {bd['healthy']} Healthy, {bd['not_crop']} Not-Crop, {bd['uncertain']} Uncertain")
        print(f"    Latency         : {res['tiled_latency_ms']:.2f} ms")

        slowdown = res['tiled_latency_ms'] / max(res['wf_latency_ms'], 1e-6)
        print("-" * 70)
        print("TIMING & OVERHEAD:")
        print(f"    Whole-Frame    : {res['wf_latency_ms']:.2f} ms")
        print(f"    3x3 Tiled (MIL): {res['tiled_latency_ms']:.2f} ms")
        print(f"    Overhead Ratio : {slowdown:.2f}x")
        print("-" * 70)
        print("COMPARISON SUMMARY:")
        print(f"    Whole-Frame State: {wf['state']:<10} --> Tiled State: {tiled['state']}")
    else:
        print("\n[2] TILED INFERENCE: DISABLED (--no-tiling specified)")

    print("-" * 70)
    print("Thresholds Applied:")
    print(f"  T_CAL={T_CAL}, TAU_ENERGY={TAU_ENERGY}, TAU_CONF={TAU_CONF}")
    print(f"  TAU_DISEASE={TAU_DISEASE}, TAU_MARGIN={TAU_MARGIN}, TAU_PRIOR={TAU_PRIOR}")
    print("=" * 70)

    if save_annotated or show_window:
        annotated = draw_side_by_side_overlay(bgr, res, use_tta=use_tta)
        if save_annotated:
            save_annotated.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_annotated), annotated)
            print(f"Annotated preview saved to: {save_annotated}")
        if show_window:
            cv2.imshow("MacBook Model Test - Side-by-Side", annotated)
            print("Press any key in image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return res


def run_webcam_loop(
    camera_index: int,
    model: nn.Module,
    transform,
    device: torch.device,
    log_priors: np.ndarray,
    use_tiling: bool = True,
    use_tta: bool = False,
    snapshot_dir: Optional[Path] = None,
) -> None:
    """Runs real-time webcam inference loop with dual-column HUD overlay."""
    print(f"\n[webcam_test] Opening webcam (device index {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open webcam at index {camera_index}. "
            "Check camera permissions in macOS System Settings -> Privacy & Security -> Camera."
        )

    print("\nLive webcam running! Controls:")
    print("  [q] or [ESC] : Quit")
    print("  [s]          : Save current annotated snapshot")
    print("  [t]          : Toggle TTA on/off")
    print("-" * 50)

    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[webcam_test] Error: Failed to grab frame from camera.")
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
            prev_time = curr_time

            res = predict_pipeline(
                frame,
                model,
                transform,
                device,
                log_priors,
                use_tiling=use_tiling,
                use_tta=use_tta,
            )
            annotated = draw_side_by_side_overlay(frame, res, fps=fps, use_tta=use_tta)

            cv2.imshow("MacBook Crop Disease Monitor (Press 'q' to Quit, 's' to Save, 't' for TTA)", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                print("[webcam_test] Exit requested by user.")
                break
            elif key == ord('s') and snapshot_dir:
                snap_path = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                print(f"[webcam_test] Saved snapshot to: {snap_path}")
            elif key == ord('t'):
                use_tta = not use_tta
                print(f"[webcam_test] TTA toggled: {'ENABLED' if use_tta else 'DISABLED'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[webcam_test] Camera released and window closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local MacBook testing harness with live webcam and single-image inference."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="artifacts/checkpoints/v3/stage1.pt",
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single static image (runs image mode instead of webcam loop)",
    )
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="Disable 3x3 spatial tiling (run whole-frame single forward pass only)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (3 views: original, h-flip, 1.1x scale)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV VideoCapture camera index (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Compute device (default: auto -> mps if available else cpu)",
    )
    parser.add_argument(
        "--eval-weights",
        type=str,
        default="auto",
        choices=["auto", "ema", "model_state_dict"],
        help="Checkpoint weights to load (default: auto -> EMA if available)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV window during --image mode",
    )
    parser.add_argument(
        "--save-annotated",
        type=str,
        default=None,
        help="Optional path to save annotated preview image in --image mode",
    )
    args = parser.parse_args()

    # 1. Resolve paths
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Select device & load model
    device = select_device(args.device)
    print(f"[webcam_test] Using compute device: {device}")
    print(f"[webcam_test] Loading architecture (EfficientNet-Lite0, {NUM_CLASSES} classes)...")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model, weights_used = load_checkpoint_for_eval(
        ckpt_path, model, eval_weights=args.eval_weights, device=device
    )
    model.to(device)
    model.eval()
    print(f"[webcam_test] Successfully loaded checkpoint from {ckpt_path} (weights: {weights_used})")

    # 3. Preprocessing transform (strictly matching train/dataset.py:21)
    tf = eval_transform(size=IMAGE_SIZE)

    # 4. Training log-priors for decide()
    log_priors = get_log_priors()

    use_tiling = not args.no_tiling

    # 5. Dispatch mode: single static image vs live webcam
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = REPO_ROOT / img_path
        save_ann = Path(args.save_annotated) if args.save_annotated else None
        run_single_image(
            image_path=img_path,
            model=model,
            transform=tf,
            device=device,
            log_priors=log_priors,
            weights_used=weights_used,
            use_tiling=use_tiling,
            use_tta=args.tta,
            show_window=args.show,
            save_annotated=save_ann,
        )
    else:
        snap_dir = REPO_ROOT / "AI_macbook_test" / "snapshots"
        run_webcam_loop(
            camera_index=args.camera_index,
            model=model,
            transform=tf,
            device=device,
            log_priors=log_priors,
            use_tiling=use_tiling,
            use_tta=args.tta,
            snapshot_dir=snap_dir,
        )


if __name__ == "__main__":
    main()
