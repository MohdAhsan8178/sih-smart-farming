"""
Augmentation and preprocessing transforms matching the aerial deployment domain.
Reference: AI_Handbook_4.md §6.6 and ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 11.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.train_config import IMAGE_SIZE

# Sourced constants
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Agronomic constraint: colour is diagnostic information for foliar diseases.
# Never shift hue beyond +/- 12 (AI_Handbook_4.md §6.6).
HUE_SHIFT_LIMIT: int = 12


def train_transform(size: int = IMAGE_SIZE) -> A.Compose:
    """
    Full field-conditioned augmentation for Stage 2 training.
    Simulates UAV flight conditions: motion blur, defocus, variable lighting,
    nadir view orientation, and sensor noise.
    """
    return A.Compose([
        # --- Geometry ---
        # scale=(0.35, 1.0) is the critical line: removes background shortcut
        # and matches tile-based inference distribution.
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.35, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Nadir drone view has no natural "up"
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-45, 45),
            fill=0,
            p=0.7,
        ),

        # --- Illumination ---
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=0.8,
        ),
        A.OneOf([
            A.RandomShadow(num_shadows_limit=(1, 3), p=1.0),
            A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), src_radius=120, p=1.0),
            A.RandomToneCurve(scale=0.3, p=1.0),
        ], p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=HUE_SHIFT_LIMIT,
            sat_shift_limit=25,
            val_shift_limit=15,
            p=0.5,
        ),

        # --- Sensor & Motion ---
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.25), p=1.0),
        ], p=0.4),
        A.ImageCompression(quality_range=(45, 95), p=0.4),

        # --- Anti-Shortcut ---
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),

        # --- Normalization & Tensor Conversion ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def baseline_train_transform(size: int = IMAGE_SIZE) -> A.Compose:
    """
    Standard mild augmentation for Stage 1 baseline, updated for V3 with anti-source shortcut transforms.

    Rationale:
    Source signature lives in compression artifacts, sensor noise and resolution.
    These attack it without touching hue:
      - ImageCompression(quality_lower=30, quality_upper=90, p=0.5)
      - Downscale(scale_min=0.5, scale_max=0.9, p=0.3)
      - GaussNoise(p=0.3)
      - MotionBlur(blur_limit=5, p=0.2)
      - RandomGamma(p=0.3)

    Agronomic constraint: Colour is diagnostic for foliar diseases.
    KEEP HUE_SHIFT_LIMIT = 12. Colour is diagnostic. Do not raise it.
    """
    import inspect

    if "quality_range" in inspect.signature(A.ImageCompression.__init__).parameters:
        comp_tf = A.ImageCompression(quality_range=(30, 90), p=0.5)
    else:
        comp_tf = A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5)
    comp_tf.quality_lower = 30
    comp_tf.quality_upper = 90

    if "scale_range" in inspect.signature(A.Downscale.__init__).parameters:
        downscale_tf = A.Downscale(scale_range=(0.7, 0.95), p=0.3)
    else:
        downscale_tf = A.Downscale(scale_min=0.7, scale_max=0.95, p=0.3)
    downscale_tf.scale_min = 0.7
    downscale_tf.scale_max = 0.95

    return A.Compose([
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.8, 1.0),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        comp_tf,
        downscale_tf,
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomGamma(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(size: int = IMAGE_SIZE) -> A.Compose:
    """
    Deterministic evaluation transform for validation and test splits.
    Resizes smallest side to size * 256/224, center-crops to (size, size),
    and normalizes.
    """
    resize_dim = int(size * 256 / 224)
    return A.Compose([
        A.SmallestMaxSize(max_size=resize_dim),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
