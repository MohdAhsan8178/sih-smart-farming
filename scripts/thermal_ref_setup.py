#!/usr/bin/env python3
"""
scripts/thermal_ref_setup.py -- Reference Region Setup & Calibration Tool (M2.2).

Captures a 24x32 thermal frame from MLX90640, saves an upscaled 20x PNG (480x640)
with a labelled row/col grid and raw .npy file, and writes configs/thermal_refs.json
when wet/dry pixel boxes are specified by the user.

Python 3.6 compatible.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.thermal_capture import MLX90640


def parse_box_arg(box_str: str) -> Dict[str, int]:
    """Parses 'row_min,row_max,col_min,col_max' into a coordinate dict."""
    parts = [int(p.strip()) for p in box_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Box must contain 4 integers: row_min,row_max,col_min,col_max (got '%s')" % box_str)
    r_min, r_max, c_min, c_max = parts
    if not (0 <= r_min <= r_max < 24):
        raise ValueError("Invalid row range: 0 <= %d <= %d < 24 required" % (r_min, r_max))
    if not (0 <= c_min <= c_max < 32):
        raise ValueError("Invalid col range: 0 <= %d <= %d < 32 required" % (c_min, c_max))
    return {
        "row_min": r_min,
        "row_max": r_max,
        "col_min": c_min,
        "col_max": c_max,
    }


def render_grid_overlay_png(thermal_array: np.ndarray, output_png_path: str) -> None:
    """
    Renders 24x32 thermal array upscaled 20x to 480x640 with labelled row/col grid.
    """
    arr = np.asarray(thermal_array, dtype=np.float32)
    t_min = float(np.min(arr))
    t_max = float(np.max(arr))
    span = max(0.1, t_max - t_min)

    # Normalize to 0-255 uint8 and apply colormap
    norm = np.clip((arr - t_min) / span * 255.0, 0, 255).astype(np.uint8)
    colored_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

    # Upscale 20x using nearest neighbor to preserve individual pixel cell boundaries
    scale = 20
    upscaled = cv2.resize(colored_small, (32 * scale, 24 * scale), interpolation=cv2.INTER_NEAREST)

    # Draw grid lines
    grid_img = upscaled.copy()
    for r in range(25):
        y = r * scale
        cv2.line(grid_img, (0, y), (32 * scale, y), (80, 80, 80), 1)
    for c in range(33):
        x = c * scale
        cv2.line(grid_img, (x, 0), (x, 24 * scale), (80, 80, 80), 1)

    # Draw row labels along left and col labels along top
    for r in range(0, 24, 2):
        y = r * scale + scale - 4
        cv2.putText(grid_img, str(r), (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(0, 32, 2):
        x = c * scale + 3
        cv2.putText(grid_img, str(c), (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    cv2.imwrite(output_png_path, grid_img)


def main():
    parser = argparse.ArgumentParser(description="MLX90640 Reference Region Setup (M2.2)")
    parser.add_argument("--capture", action="store_true", help="Capture a new thermal frame from sensor")
    parser.add_argument("--thermal-source", choices=["hardware", "mock"], default="hardware", help="Thermal capture source")
    parser.add_argument("--output-dir", default="data/thermal_calibration", help="Directory to store calibration artifacts")
    parser.add_argument("--wet-box", type=str, default=None, help="Wet reference box: r_min,r_max,c_min,c_max")
    parser.add_argument("--dry-box", type=str, default=None, help="Dry reference box: r_min,r_max,c_min,c_max")
    parser.add_argument("--config-path", default="configs/thermal_refs.json", help="Path to write thermal_refs.json")
    parser.add_argument("--image-path", type=str, default=None, help="Reference image path for existing capture")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    image_path = args.image_path
    npy_path = None

    if args.capture:
        print("[SETUP] Capturing frame from MLX90640 (source=%s)..." % args.thermal_source)
        sensor = MLX90640(mock=(args.thermal_source == "mock"))
        frame = sensor.capture_frame()
        if not frame.get("available") or frame.get("temperature_array") is None:
            print("[ERROR] Thermal capture failed: %s" % frame.get("reason", "unknown error"))
            sys.exit(1)

        thermal_arr = frame["temperature_array"]
        npy_path = os.path.join(args.output_dir, "thermal_ref_frame_%s.npy" % ts)
        image_path = os.path.join(args.output_dir, "thermal_ref_frame_%s.png" % ts)

        np.save(npy_path, thermal_arr)
        render_grid_overlay_png(thermal_arr, image_path)

        print("[SUCCESS] Thermal frame captured:")
        print("  - Raw NumPy Array: %s" % npy_path)
        print("  - Labelled Grid Image: %s" % image_path)
        print("\n--- INSTRUCTIONS ---")
        print("1. Open the image '%s' to inspect the thermal view." % image_path)
        print("2. Locate the wet reference surface (coolest artificial target) and dry reference surface (hottest target).")
        print("3. Note the row ranges (0-23) and column ranges (0-31) covering each reference pad.")
        print("4. Re-run this script with the box coordinates to save configuration:")
        print("   python3 scripts/thermal_ref_setup.py --wet-box <r_min>,<r_max>,<c_min>,<c_max> --dry-box <r_min>,<r_max>,<c_min>,<c_max> --image-path %s\n" % image_path)

    if args.wet_box and args.dry_box:
        wet_dict = parse_box_arg(args.wet_box)
        dry_dict = parse_box_arg(args.dry_box)

        # Check for overlap between wet and dry reference boxes
        w_rmin, w_rmax = wet_dict["row_min"], wet_dict["row_max"]
        w_cmin, w_cmax = wet_dict["col_min"], wet_dict["col_max"]
        d_rmin, d_rmax = dry_dict["row_min"], dry_dict["row_max"]
        d_cmin, d_cmax = dry_dict["col_min"], dry_dict["col_max"]

        row_overlap = max(0, min(w_rmax, d_rmax) - max(w_rmin, d_rmin) + 1)
        col_overlap = max(0, min(w_cmax, d_cmax) - max(w_cmin, d_cmin) + 1)
        if row_overlap > 0 and col_overlap > 0:
            print("[ERROR] Wet and dry reference boxes overlap! They must be distinct regions.")
            sys.exit(1)

        config_data = {
            "status": "MEASURED",
            "wet_ref": wet_dict,
            "dry_ref": dry_dict,
            "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference_image_path": image_path,
        }

        os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
        with open(args.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        print("[SUCCESS] Reference region configuration saved to %s" % args.config_path)
        print(json.dumps(config_data, indent=2))


if __name__ == "__main__":
    main()
