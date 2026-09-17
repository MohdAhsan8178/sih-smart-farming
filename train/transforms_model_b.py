"""
Augmentation and preprocessing transforms for Model B (sticky-trap patch classifier).
Simulates low-cost ESP32-CAM (OV2640) hardware characteristics on 64x64 patches.

Python 3.6 compatible.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
PATCH_SIZE: int = 64


def train_transform_model_b(size: int = PATCH_SIZE) -> A.Compose:
    """
    Aggressive ESP32-CAM simulation pipeline on 64x64 trap patches:
      1. Extreme downscale-then-upscale to mimic 6-pixel insect footprint at nominal resolution.
      2. Heavy JPEG compression artefacts (quality 15-60) matching OV2640 hardware encoder.
      3. Motion blur, focal defocus, and Gaussian blur simulating camera vibration / wind smear.
      4. Sensor noise (ISO noise + Gaussian noise) and mild colour cast / white-balance drift.
      5. Geometric nadir invariance (flips and 90-degree rotations).
      6. Coarse dropout simulating dust specks, glue bubbles, and partial occlusion.
    """
    return A.Compose([
        A.Resize(size, size),

        # 1. Extreme resolution degradation: simulate 6-pixel insect footprint
        A.Downscale(scale_range=(0.25, 0.60), p=0.6),

        # 2. Aggressive OV2640 JPEG compression artefacts
        A.ImageCompression(quality_range=(15, 60), p=0.7),

        # 3. Motion blur & focal defocus
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),

        # 4. Sensor noise & colour drift
        A.OneOf([
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.1, 0.4), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.20), p=1.0),
        ], p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.6),

        # 5. Nadir invariance
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 6. Glue glare / dust occlusion
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(4, 12),
            hole_width_range=(4, 12),
            fill=255,
            p=0.4
        ),

        # 7. Normalization and tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform_model_b(size: int = PATCH_SIZE) -> A.Compose:
    """
    Deterministic evaluation transform for Model B.
    Resizes patch to (size, size) and normalizes.
    """
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
