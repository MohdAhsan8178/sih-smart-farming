#!/usr/bin/env python3
"""
Step 20: Edge Tiler for spatial crop inspection.

Extracts a deterministic 3x3 grid of 320x320 tiles with 20% overlap on an 832x832 canvas.
Filters tiles keeping only those with >40% vegetation using absolute ExG thresholding
(vegetation_mask from core.indices, thresh=PROVISIONAL_EXG_VEG_THRESHOLD).
Guarantees the output batch size is ALWAYS padded to exactly N_TILES (9) to match
the static batch size of the TensorRT engine.

Python 3.6 compatible (no f-string '=', no walrus, no dataclasses).
"""

import argparse
from pathlib import Path
import sys
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.train_config import (
    IMAGE_SIZE,
    N_TILES,
    TILE_GRID,
    TILE_MIN_VEG_FRACTION,
    TILE_OVERLAP,
    TILE_SIZE,
)
from core.indices import (
    PROVISIONAL_EXG_VEG_THRESHOLD,
    vegetation_mask,
)


class TileBatch(object):
    """
    Container for extracted and padded tile batches.
    Guarantees len(tiles) == N_TILES (9).
    Supports attribute access and tuple unpacking.
    """

    def __init__(
        self,
        tiles: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        veg_fractions: List[float],
        n_valid: int,
    ):
        self.tiles = tiles
        self.boxes = boxes
        self.veg_fractions = veg_fractions
        self.n_valid = int(n_valid)

    def __iter__(self):
        return iter((self.tiles, self.boxes, self.veg_fractions, self.n_valid))

    def __getitem__(self, idx):
        return (self.tiles, self.boxes, self.veg_fractions, self.n_valid)[idx]

    def __len__(self):
        return 4

    def as_numpy_batch(self) -> np.ndarray:
        """Returns tiles stacked as a single uint8 numpy array (N_TILES, H, W, 3)."""
        return np.stack(self.tiles, axis=0)


class Tiler(object):
    """
    Deterministic 3x3 spatial tiler with vegetation mask filtering and static batch padding.
    """

    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        grid_size: int = TILE_GRID,
        overlap: float = TILE_OVERLAP,
        min_veg_fraction: float = TILE_MIN_VEG_FRACTION,
        n_tiles: int = N_TILES,
        exg_thresh: int = PROVISIONAL_EXG_VEG_THRESHOLD,
    ):
        self.tile_size = int(tile_size)
        self.grid_size = int(grid_size)
        self.overlap = float(overlap)
        self.min_veg_fraction = float(min_veg_fraction)
        self.n_tiles = int(n_tiles)
        self.exg_thresh = int(exg_thresh)

        # Step between tile origins: step = tile_size * (1 - overlap) = 256 px
        self.step = int(self.tile_size * (1.0 - self.overlap))
        # Total required canvas footprint: tile_size + (grid_size - 1) * step = 320 + 2 * 256 = 832 px
        self.footprint = self.tile_size + (self.grid_size - 1) * self.step

        # Deterministic grid coordinate offsets: [0, 256, 512]
        self.offsets = [i * self.step for i in range(self.grid_size)]

    def extract(
        self,
        img_bgr: np.ndarray,
        target_size: Optional[int] = None,
    ) -> TileBatch:
        """
        Extracts 3x3 tiles, filters by vegetation fraction, and pads to exactly n_tiles.

        Parameters:
            img_bgr: Input BGR image (H, W, 3) or (H, W).
            target_size: Optional target size (e.g. IMAGE_SIZE = 224) to resize output tiles.
                         If None, tiles remain at tile_size (320x320).

        Returns:
            TileBatch containing exactly n_tiles (9) tiles.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is None or empty.")

        h, w = img_bgr.shape[:2]

        # Ensure image is in 3-channel BGR
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Uniformly scale/resize input frame to 832x832 canvas if necessary
        if (h, w) != (self.footprint, self.footprint):
            canvas = cv2.resize(
                img_bgr,
                (self.footprint, self.footprint),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            canvas = img_bgr

        # Extract deterministic 3x3 candidate tiles and evaluate vegetation fraction
        raw_candidates = []
        for y in self.offsets:
            for x in self.offsets:
                tile = canvas[y : y + self.tile_size, x : x + self.tile_size]
                box = (x, y, x + self.tile_size, y + self.tile_size)

                # Absolute ExG vegetation mask (deliberately NOT Otsu)
                _, veg_frac = vegetation_mask(tile, thresh=self.exg_thresh)
                raw_candidates.append((tile, box, float(veg_frac)))

        # Keep tiles with > min_veg_fraction
        qualifying = [c for c in raw_candidates if c[2] > self.min_veg_fraction]
        n_valid = len(qualifying)

        # Pad to exactly n_tiles (9)
        if n_valid == self.n_tiles:
            # All 9 tiles qualify
            selected = qualifying
        elif n_valid > 0:
            # 1 to 8 tiles qualify: cyclically replicate qualifying tiles to reach n_tiles
            selected = list(qualifying)
            needed = self.n_tiles - n_valid
            for i in range(needed):
                selected.append(qualifying[i % n_valid])
        else:
            # Degenerate case: 0 tiles qualify (e.g. pure soil, pavement, all-black frame).
            # Retain all 9 geometric candidates so downstream classifier can classify
            # real scene contents as not_crop or abstain.
            selected = raw_candidates[: self.n_tiles]

        # Resize to target_size if requested
        final_tiles = []
        final_boxes = []
        final_veg_fracs = []

        for tile, box, vf in selected:
            if target_size is not None and (tile.shape[0] != target_size or tile.shape[1] != target_size):
                t_resized = cv2.resize(
                    tile,
                    (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                t_resized = tile.copy()
            final_tiles.append(t_resized)
            final_boxes.append(box)
            final_veg_fracs.append(vf)

        assert len(final_tiles) == self.n_tiles, (
            "Tiler must always return exactly %d tiles, got %d" % (self.n_tiles, len(final_tiles))
        )

        return TileBatch(
            tiles=final_tiles,
            boxes=final_boxes,
            veg_fractions=final_veg_fracs,
            n_valid=n_valid,
        )


def extract_tiles(
    img_bgr: np.ndarray,
    tile_size: int = TILE_SIZE,
    grid_size: int = TILE_GRID,
    overlap: float = TILE_OVERLAP,
    min_veg_fraction: float = TILE_MIN_VEG_FRACTION,
    n_tiles: int = N_TILES,
    target_size: Optional[int] = None,
    exg_thresh: int = PROVISIONAL_EXG_VEG_THRESHOLD,
) -> TileBatch:
    """
    Convenience wrapper to extract tiles with default parameters.
    Always returns exactly n_tiles (9).
    """
    tiler = Tiler(
        tile_size=tile_size,
        grid_size=grid_size,
        overlap=overlap,
        min_veg_fraction=min_veg_fraction,
        n_tiles=n_tiles,
        exg_thresh=exg_thresh,
    )
    return tiler.extract(img_bgr, target_size=target_size)


def run_selftest() -> bool:
    """
    Self-test routine confirming Tiler always returns exactly N_TILES (9)
    across all input scenarios including degenerate cases.
    """
    print("=== Tiler Self-Test ===")
    tiler = Tiler()
    target_batch = N_TILES  # 9

    # 1. Pure Vegetation Frame (all 9 tiles pass >40% vegetation)
    pure_veg = np.full((832, 832, 3), (25, 160, 35), dtype=np.uint8)
    batch_veg = tiler.extract(pure_veg)
    print("Test 1: Pure Vegetation Frame        -> Tiles: %d, Valid: %d (PASS)" % (len(batch_veg.tiles), batch_veg.n_valid))
    assert len(batch_veg.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert batch_veg.n_valid == target_batch, "Expected all 9 tiles to be valid"

    # 2. Partial Vegetation Frame (only top row is vegetation, bottom 2 rows soil)
    partial = np.full((832, 832, 3), (40, 60, 90), dtype=np.uint8)  # soil BGR
    partial[:320, :] = (25, 160, 35)  # top 320px is vegetation
    batch_partial = tiler.extract(partial)
    print("Test 2: Partial Vegetation Frame     -> Tiles: %d, Valid: %d (Padded to 9) (PASS)" % (len(batch_partial.tiles), batch_partial.n_valid))
    assert len(batch_partial.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert 0 < batch_partial.n_valid < target_batch, "Expected partial validity"

    # 3. Degenerate Bare Soil Frame (0% vegetation)
    soil = np.full((832, 832, 3), (40, 60, 90), dtype=np.uint8)  # brown soil
    batch_soil = tiler.extract(soil)
    print("Test 3: Degenerate Bare Soil Frame   -> Tiles: %d, Valid: %d (Padded to 9) (PASS)" % (len(batch_soil.tiles), batch_soil.n_valid))
    assert len(batch_soil.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert batch_soil.n_valid == 0, "Expected 0 valid vegetation tiles"

    # 4. Degenerate Blank / Black Frame (all zeros)
    blank = np.zeros((832, 832, 3), dtype=np.uint8)
    batch_blank = tiler.extract(blank)
    print("Test 4: Degenerate Black Frame       -> Tiles: %d, Valid: %d (Padded to 9) (PASS)" % (len(batch_blank.tiles), batch_blank.n_valid))
    assert len(batch_blank.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert batch_blank.n_valid == 0, "Expected 0 valid vegetation tiles"

    # 5. Arbitrary Resolution Frame (1920x1080 resized to 832x832)
    hd_frame = np.full((1080, 1920, 3), (25, 160, 35), dtype=np.uint8)
    batch_hd = tiler.extract(hd_frame)
    print("Test 5: Arbitrary 1080p Frame        -> Tiles: %d, Valid: %d (Resized to 832x832) (PASS)" % (len(batch_hd.tiles), batch_hd.n_valid))
    assert len(batch_hd.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert batch_hd.tiles[0].shape == (TILE_SIZE, TILE_SIZE, 3), "Expected tile shape (%d, %d, 3)" % (TILE_SIZE, TILE_SIZE)

    # 6. Target Size Resizing (IMAGE_SIZE = 224 for TRTClassifier)
    batch_224 = tiler.extract(pure_veg, target_size=IMAGE_SIZE)
    print("Test 6: Target Size (224x224) Resize -> Tiles: %d, Shape: %s (PASS)" % (len(batch_224.tiles), str(batch_224.tiles[0].shape)))
    assert len(batch_224.tiles) == target_batch, "Expected %d tiles" % target_batch
    assert batch_224.tiles[0].shape == (IMAGE_SIZE, IMAGE_SIZE, 3), "Expected tile shape (%d, %d, 3)" % (IMAGE_SIZE, IMAGE_SIZE)

    # 7. Confirm Absolute ExG Thresholding (NOT Otsu bisection)
    # A 100% closed green canopy must report ~100% vegetation, never ~50%
    _, veg_frac = vegetation_mask(pure_veg, thresh=PROVISIONAL_EXG_VEG_THRESHOLD)
    print("Test 7: Absolute ExG Canopy Purity   -> Veg Fraction: %.4f (Deliberately NOT Otsu) (PASS)" % veg_frac)
    assert veg_frac >= 0.99, "Pure canopy vegetation fraction should be ~1.0, got %.4f" % veg_frac

    print("ALL 7 TILER SELF-TEST CHECKS PASSED: ALWAYS returns exactly %d tiles." % target_batch)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Tiler Self-Test")
    parser.add_argument("--selftest", action="store_true", help="Run self-test routine")
    args = parser.parse_args()

    if args.selftest:
        success = run_selftest()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(0)
