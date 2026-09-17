#!/usr/bin/env python3
"""
Extract pre-softmax logits from an arbitrary image directory using a trained Model A checkpoint.
Used for Step 15 energy calibration on unlabelled holdout sets (e.g. data/raw/openset_holdout/).

Constraints:
- Must use eval_transform(size=IMAGE_SIZE) — identical to validation.
- Captures raw float32 pre-softmax logits (before softmax, before temperature, before prior adjustment).
- Fails loudly with filename if any image is corrupted or unreadable (never silently skips).
- Preserves exact row correspondence between logits.shape[0] and paths list.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.classes import CLASS_NAMES
from configs.paths import CKPT, ROOT
from configs.train_config import BATCH_SIZE, IMAGE_SIZE
from train.model import build_model
from train.train_model_a import load_checkpoint_for_eval
from train.transforms import eval_transform


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnlabelledImageFolderDataset(Dataset):
    """
    Dataset iterating over an arbitrary list of image paths without labels or manifest.
    Applies eval_transform and fails loudly on corrupt or unreadable images.
    """

    def __init__(self, image_paths: List[Path], transform: object, input_dir: Path):
        self.image_paths = image_paths
        self.transform = transform
        self.input_dir = Path(input_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str]:
        path = self.image_paths[idx]
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Corrupted or unreadable image file: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb)["image"]
        try:
            rel_path = str(path.relative_to(self.input_dir))
        except ValueError:
            rel_path = str(path)

        parent_folder = path.parent.name
        return tensor, str(path), rel_path, parent_folder


def discover_images(input_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively discovers all image files in input_dir, sorted deterministically.
    Ignores hidden files / dotfiles (e.g. .DS_Store).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    found_images = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.startswith("."):
                continue
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                found_images.append(Path(root) / f)

    found_images.sort()
    return found_images


def extract_logits(
    model: nn.Module,
    input_dir: Union[str, Path],
    out: Union[str, Path],
    checkpoint_path: Union[str, Path] = "in_memory",
    weights_used: str = "unknown",
    batch_size: int = BATCH_SIZE,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts pre-softmax float32 logits from input_dir using model and eval_transform.
    Saves logits to .npy and parallel metadata/paths to .json.
    """
    input_path = Path(input_dir)
    dev = device if device is not None else torch.device("cpu")
    model.eval()

    image_paths = discover_images(input_path)
    if not image_paths:
        raise ValueError(f"No valid image files found in {input_path}")

    tf = eval_transform(size=IMAGE_SIZE)
    dataset = UnlabelledImageFolderDataset(image_paths=image_paths, transform=tf, input_dir=input_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    all_logits = []
    records = []
    category_counts = Counter()

    with torch.no_grad():
        for tensors, abs_paths, rel_paths, parent_folders in loader:
            tensors = tensors.to(dev, non_blocking=True)
            outputs = model(tensors)  # Raw float32 pre-softmax logits
            logits_np = outputs.detach().cpu().to(torch.float32).numpy()
            all_logits.append(logits_np)

            for i in range(len(abs_paths)):
                row_idx = len(records)
                records.append({
                    "row": row_idx,
                    "path": abs_paths[i],
                    "relative_path": rel_paths[i],
                    "parent_folder": parent_folders[i],
                })
                category_counts[parent_folders[i]] += 1

    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)

    # Determine destination paths
    out_p = Path(out)
    if out_p.suffix == ".npy":
        logits_file = out_p
        paths_file = out_p.with_name(out_p.stem.removesuffix("_logits") + "_paths.json")
    elif out_p.is_dir() or str(out).endswith(os.sep):
        out_p.mkdir(parents=True, exist_ok=True)
        logits_file = out_p / "logits.npy"
        paths_file = out_p / "paths.json"
    else:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        logits_file = Path(f"{out}_logits.npy")
        paths_file = Path(f"{out}_paths.json")

    logits_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(logits_file, logits_arr)

    meta_payload = {
        "checkpoint_path": str(checkpoint_path),
        "weights_used": weights_used,
        "input_dir": str(input_path),
        "row_count": len(records),
        "category_counts": dict(category_counts),
        "paths": [r["path"] for r in records],
        "images": records,
    }

    with open(paths_file, "w") as f:
        json.dump(meta_payload, f, indent=2)

    print(
        f"[extract_logits] Extracted {len(records)} rows | "
        f"Shape: {logits_arr.shape} | "
        f"Categories: {dict(category_counts)} | "
        f"Logits: {logits_file} | "
        f"Paths: {paths_file}"
    )

    return logits_arr, meta_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw logits from image folder")
    parser.add_argument("--checkpoint", type=str, default=str(CKPT / "stage1.pt"), help="Path to checkpoint (.pt)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--out", type=str, required=True, help="Output destination prefix or directory")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--data_dir", type=str, default=str(ROOT), help="Root directory for relative paths")
    parser.add_argument("--eval_weights", type=str, default="auto", choices=["auto", "ema", "model"], help="Which weights to evaluate: auto, ema, model")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    model, weights_used = load_checkpoint_for_eval(
        checkpoint_path=ckpt_path,
        model=model,
        eval_weights=args.eval_weights,
        device=device,
    )

    extract_logits(
        model=model,
        input_dir=args.input_dir,
        out=args.out,
        checkpoint_path=ckpt_path,
        weights_used=weights_used,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
