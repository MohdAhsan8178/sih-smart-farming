"""
PyTorch Dataset and DataLoader constructors for aerial crop disease training.
Reference: ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 11 and AI_Handbook_4.md §6.6.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.paths import ROOT, SPLITS, REPORTS
from configs.train_config import BATCH_SIZE, IMAGE_SIZE, TILE_SIZE
from train.transforms import baseline_train_transform, eval_transform, train_transform, IMAGENET_MEAN, IMAGENET_STD

# Sourced constants
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_TILE_SIZE: int = TILE_SIZE  # 320


class PlantDataset(Dataset):
    """
    Dataset for plant leaf imagery.
    Loads images in RGB format and applies Albumentations transformations.
    Class mappings and dataset sizes are dynamically resolved from the input
    manifest or configs/classes.py at runtime.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        transform: Optional[object] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        is_train: bool = False,
        slicing_factor: int = 0,
        tile_size: int = DEFAULT_TILE_SIZE,
        enforce_manifest_agreement: bool = False,
        root_dir: Optional[Union[str, Path]] = None,
        path_prefix_strip: str = "",
    ):
        self.root_dir = Path(root_dir) if root_dir is not None else ROOT
        self.transform = transform
        self.is_train = is_train
        self.slicing_factor = max(0, int(slicing_factor))
        self.tile_size = int(tile_size)

        # 1. Load DataFrame
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        else:
            path = Path(data_source)
            if not path.is_absolute():
                path = self.root_dir / path
            if not path.exists():
                raise FileNotFoundError(f"Dataset split CSV not found: {path}")
            self.df = pd.read_csv(path).reset_index(drop=True)

        if 'path' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError(f"Split CSV must contain 'path' and 'label' columns. Found: {list(self.df.columns)}")

        # 1b. Strip path prefix if provided (vectorised, once at construction)
        if path_prefix_strip:
            mask = ~self.df['path'].str.startswith(path_prefix_strip)
            n_offenders = int(mask.sum())
            if n_offenders > 0:
                first_offender = self.df.loc[mask, 'path'].iloc[0]
                raise ValueError(
                    f"path_prefix_strip='{path_prefix_strip}' but {n_offenders} path(s) "
                    f"do not start with it. First offending path: '{first_offender}'"
                )
            self.df['path'] = self.df['path'].str[len(path_prefix_strip):]

        # 2. Resolve Class Taxonomy Dynamically
        if class_to_idx is None:
            # Read from single source of truth at runtime
            from configs.classes import IDX
            self.class_to_idx = dict(IDX)
        else:
            self.class_to_idx = dict(class_to_idx)

        # 3. Upfront Manifest vs Taxonomy Config Agreement Guard
        if enforce_manifest_agreement:
            manifest_classes = set(self.df['label'].unique())
            taxonomy_classes = set(self.class_to_idx.keys())
            if manifest_classes != taxonomy_classes:
                unknown = manifest_classes - taxonomy_classes
                missing = taxonomy_classes - manifest_classes
                raise ValueError(
                    f"Taxonomy agreement failure between manifest and configs/classes.py! "
                    f"Unknown in config: {unknown} | Missing from manifest: {missing}"
                )

        # Pre-filter or validate labels
        self.labels = self.df['label'].astype(str).tolist()
        self.image_paths = self.df['path'].astype(str).tolist()

        # 1c. Early path-existence check on first 20 stripped paths (fires at
        # construction, never inside a DataLoader worker).
        if path_prefix_strip:
            check_count = min(20, len(self.image_paths))
            for i in range(check_count):
                raw_csv_value = self.image_paths[i]
                resolved = self.root_dir / raw_csv_value if not Path(raw_csv_value).is_absolute() else Path(raw_csv_value)
                if not resolved.exists():
                    raise RuntimeError(
                        f"path_prefix_strip sanity check failed at construction. "
                        f"root_dir='{self.root_dir}', "
                        f"raw CSV value (after strip)='{raw_csv_value}', "
                        f"original CSV value='{path_prefix_strip}{raw_csv_value}', "
                        f"joined result='{resolved}' does not exist on disk."
                    )

        # Build index mapping: if slicing_factor > 0 in training, each image emits
        # (1 + slicing_factor) samples (original + random tile crops).
        self.samples: List[Tuple[int, bool]] = []
        for i in range(len(self.image_paths)):
            self.samples.append((i, False))  # Whole image
            if self.is_train and self.slicing_factor > 0:
                for _ in range(self.slicing_factor):
                    self.samples.append((i, True))  # Sliced tile crop

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_image(self, path_str: str) -> np.ndarray:
        img_path = Path(path_str)
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image path does not exist on disk: {img_path}")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Corrupted or unreadable image: {img_path}")

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _extract_tile_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        crop_h = min(self.tile_size, h)
        crop_w = min(self.tile_size, w)
        if h <= crop_h and w <= crop_w:
            return img

        y = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        x = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        return img[y:y + crop_h, x:x + crop_w]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_idx, is_slice = self.samples[index]
        path_str = self.image_paths[img_idx]
        label_str = self.labels[img_idx]

        if label_str not in self.class_to_idx:
            raise KeyError(
                f"Label '{label_str}' not present in class_to_idx taxonomy mapping. "
                f"Known classes ({len(self.class_to_idx)}): {list(self.class_to_idx.keys())}"
            )
        target = int(self.class_to_idx[label_str])

        img = self._load_rgb_image(path_str)

        if is_slice:
            img = self._extract_tile_crop(img)

        if self.transform is not None:
            augmented = self.transform(image=img)
            tensor = augmented['image']
        else:
            # Fallback if no transform passed
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return tensor, target


def seed_worker(worker_id: int) -> None:
    """
    Seeds DataLoader worker processes distinctly and reproducibly.
    Combines torch.initial_seed() % (2**32) with worker_id so that each worker
    receives an independent, deterministic random stream for NumPy and Python random.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    train_csv: Optional[Union[str, Path]] = None,
    val_csv: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    stage: int = 1,
    class_to_idx: Optional[Dict[str, int]] = None,
    slicing_factor: int = 0,
    enforce_manifest_agreement: bool = True,
    root_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    path_prefix_strip: str = "",
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders.
    Stage 1: baseline transforms (mild resize + flip + normalize).
    Stage 2: full field-conditioned augmentation.
    """
    train_path = Path(train_csv) if train_csv is not None else (SPLITS / 'train.csv')
    val_path = Path(val_csv) if val_csv is not None else (SPLITS / 'val.csv')
    bs = int(batch_size) if batch_size is not None else BATCH_SIZE

    if stage == 1:
        train_tf = baseline_train_transform(size=IMAGE_SIZE)
    else:
        train_tf = train_transform(size=IMAGE_SIZE)

    val_tf = eval_transform(size=IMAGE_SIZE)

    train_ds = PlantDataset(
        data_source=train_path,
        transform=train_tf,
        class_to_idx=class_to_idx,
        is_train=True,
        slicing_factor=slicing_factor if stage == 2 else 0,
        enforce_manifest_agreement=enforce_manifest_agreement,
        root_dir=root_dir,
        path_prefix_strip=path_prefix_strip,
    )

    val_ds = PlantDataset(
        data_source=val_path,
        transform=val_tf,
        class_to_idx=class_to_idx,
        is_train=False,
        slicing_factor=0,
        enforce_manifest_agreement=enforce_manifest_agreement,
        root_dir=root_dir,
        path_prefix_strip=path_prefix_strip,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader


def save_augmentation_grid(
    train_csv: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    n_samples: int = 16,
    seed: int = 42,
) -> Path:
    """
    Saves a 4x4 visual verification grid of field-augmented samples
    to artifacts/reports/aug_samples.png as required by STEP 11.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(out_path) if out_path is not None else (REPORTS / 'aug_samples.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tf = train_transform(size=IMAGE_SIZE)
    ds = PlantDataset(
        data_source=train_csv or (SPLITS / 'train.csv'),
        transform=tf,
        is_train=True,
    )

    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    tensors = [ds[i][0] for i in indices]

    # Unnormalize: img * std + mean -> [0, 255]
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)

    grid_images = []
    for t in tensors:
        arr = t.numpy() * std + mean
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        # Convert RGB to BGR for cv2 saving
        bgr = cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        grid_images.append(bgr)

    # Pad if fewer than 16
    while len(grid_images) < 16:
        grid_images.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

    rows = []
    for r in range(4):
        row = np.hstack(grid_images[r * 4:(r + 1) * 4])
        rows.append(row)
    grid = np.vstack(rows)

    cv2.imwrite(str(save_path), grid)
    return save_path
