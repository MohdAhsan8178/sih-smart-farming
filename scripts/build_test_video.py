#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sample_dir = Path("AI_macbook_test/sample_images")
manifest_csv = sample_dir / "manifest.csv"
df = pd.read_csv(manifest_csv)

video_path = Path("data/video/test_video_from_dataset_images.mp4")
manifest_out = Path("data/video/test_video_from_dataset_images_manifest.json")

width, height = 1280, 720
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

frames_meta = []
frame_idx = 0

def prepare_frame(img, target_w=1280, target_h=720):
    h, w = img.shape[:2]
    if h < target_h or w < target_w:
        rep_y = int(np.ceil(target_h / h))
        rep_x = int(np.ceil(target_w / w))
        tiled = np.tile(img, (rep_y, rep_x, 1))[:target_h, :target_w]
        res = tiled
    else:
        res = cv2.resize(img, (target_w, target_h))

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if var < 150.0:
        gaussian = cv2.GaussianBlur(res, (0, 0), 2.0)
        res = cv2.addWeighted(res, 1.8, gaussian, -0.8, 0)
    return res

for _, row in df.iterrows():
    img_path = sample_dir / row["folder"] / row["filename"]
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img_resized = prepare_frame(img, width, height)
    label = str(row["true_label"])
    fname = str(row["filename"])
    folder = str(row["folder"])

    # Frame 1: novel scene
    writer.write(img_resized)
    frames_meta.append({
        "frame_idx": frame_idx,
        "source_image": fname,
        "folder": folder,
        "true_label": label,
        "expected_gate": "pass",
        "description": "Real dataset image: " + label
    })
    frame_idx += 1

    # Frame 2: duplicate frame immediately following (fails novelty gate)
    writer.write(img_resized)
    frames_meta.append({
        "frame_idx": frame_idx,
        "source_image": "duplicate_" + fname,
        "folder": folder,
        "true_label": label,
        "expected_gate": "fail_novelty",
        "description": "Duplicate frame (tests novelty rejection)"
    })
    frame_idx += 1

# Deliberate failure frames using first image
ref_img = cv2.imread(str(sample_dir / df.iloc[0]["folder"] / df.iloc[0]["filename"]))
ref_resized = cv2.resize(ref_img, (width, height))

# 1. Artificially blurred frame -> fails blur gate
blurred = cv2.GaussianBlur(ref_resized, (51, 51), 30.0)
writer.write(blurred)
frames_meta.append({
    "frame_idx": frame_idx,
    "source_image": "synthetic_blurred_frame.jpg",
    "folder": "test_failure_cases",
    "true_label": "none",
    "expected_gate": "fail_blur",
    "description": "Artificially blurred frame (tests blur rejection)"
})
frame_idx += 1

# 2. Overexposed frame -> fails exposure gate
overexposed = np.full_like(ref_resized, 255)
writer.write(overexposed)
frames_meta.append({
    "frame_idx": frame_idx,
    "source_image": "synthetic_overexposed_frame.jpg",
    "folder": "test_failure_cases",
    "true_label": "none",
    "expected_gate": "fail_exposure_overexposed",
    "description": "Artificially clipped overexposed frame (tests exposure rejection)"
})
frame_idx += 1

# 3. Underexposed frame -> fails exposure gate
underexposed = np.zeros_like(ref_resized)
writer.write(underexposed)
frames_meta.append({
    "frame_idx": frame_idx,
    "source_image": "synthetic_underexposed_frame.jpg",
    "folder": "test_failure_cases",
    "true_label": "none",
    "expected_gate": "fail_exposure_underexposed",
    "description": "Artificially clipped underexposed frame (tests exposure rejection)"
})
frame_idx += 1

writer.release()

with open(manifest_out, "w") as f:
    json.dump(frames_meta, f, indent=2)

shutil.copy(str(video_path), "test_video_from_dataset_images.mp4")
shutil.copy(str(manifest_out), "test_video_from_dataset_images_manifest.json")

print("Generated " + str(frame_idx) + " frames in " + str(video_path) + " and copied to repo root.")
