"""
Unit tests for Step 20: Edge Frame Gate (edge/frame_gate.py) and Edge Tiler (edge/tiler.py).
Follows repo testing conventions: verbose assertion, failure mode isolation, no synthetic shortcuts.
"""

import cv2
import numpy as np
import pytest

from configs.train_config import (
    GATE_ALTITUDE_MAX_M,
    GATE_ALTITUDE_MIN_M,
    GATE_MIN_NOVELTY_DISPLACEMENT,
    IMAGE_SIZE,
    N_TILES,
    PROVISIONAL_GATE_CLIPPING_BRIGHT_DN,
    PROVISIONAL_GATE_CLIPPING_DARK_DN,
    PROVISIONAL_GATE_CLIPPING_MAX_FRACTION,
    PROVISIONAL_GATE_MAX_PITCH_DEG,
    PROVISIONAL_GATE_MAX_ROLL_DEG,
    PROVISIONAL_TAU_BLUR,
    TILE_MIN_VEG_FRACTION,
    TILE_OVERLAP,
    TILE_SIZE,
)
from core.indices import PROVISIONAL_EXG_VEG_THRESHOLD, vegetation_mask
from edge.frame_gate import FrameGate, run_selftest as run_gate_selftest
from edge.tiler import TileBatch, Tiler, extract_tiles, run_selftest as run_tiler_selftest


def _make_sharp_vegetation_frame(size=832, seed=42):
    """Creates a high-contrast, textured vegetation frame with high blur score and >40% ExG."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), (30, 150, 40), dtype=np.uint8)
    noise = (rng.randn(size, size, 3) * 40.0).clip(-35, 35).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 10, 240).astype(np.uint8)
    # Add strong high-frequency edges so variance of Laplacian is well above TAU_BLUR
    for _ in range(40):
        pt1 = (int(rng.randint(20, size - 20)), int(rng.randint(20, size - 20)))
        pt2 = (pt1[0] + int(rng.randint(-40, 40)), pt1[1] + int(rng.randint(-40, 40)))
        cv2.line(img, pt1, pt2, (10, 70, 20), 2)
    return img


# ==============================================================================
# FrameGate Tests
# ==============================================================================

def test_frame_gate_altitude_in_band_passes():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 2.0, "roll_deg": 1.0, "pitch_deg": -1.0}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is True
    assert reason == "ok"
    assert metrics["altitude_m"] == 2.0


def test_frame_gate_altitude_too_low_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 1.2, "roll_deg": 0.0, "pitch_deg": 0.0}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is False
    assert reason == "altitude_out_of_band"
    assert metrics["altitude_m"] == 1.2
    assert gate.rejection_counts["altitude_out_of_band"] == 1


def test_frame_gate_altitude_too_high_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 2.8, "roll_deg": 0.0, "pitch_deg": 0.0}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is False
    assert reason == "altitude_out_of_band"
    assert metrics["altitude_m"] == 2.8
    assert gate.rejection_counts["altitude_out_of_band"] == 1


def test_frame_gate_attitude_roll_exceeded_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 2.0, "roll_deg": 18.5, "pitch_deg": 2.0}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is False
    assert reason == "attitude_exceeds_threshold"
    assert metrics["roll_deg"] == 18.5
    assert gate.rejection_counts["attitude_exceeds_threshold"] == 1


def test_frame_gate_attitude_pitch_exceeded_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 2.0, "roll_deg": -3.0, "pitch_deg": -19.0}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is False
    assert reason == "attitude_exceeds_threshold"
    assert metrics["pitch_deg"] == -19.0
    assert gate.rejection_counts["attitude_exceeds_threshold"] == 1


def test_frame_gate_attitude_in_range_passes():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    telem = {"altitude_m": 2.0, "roll_deg": 14.5, "pitch_deg": -14.5}
    passed, reason, metrics = gate.evaluate(frame, telemetry=telem)
    assert passed is True
    assert reason == "ok"


def test_frame_gate_sharpness_blur_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    # Apply severe Gaussian blur
    blurred = cv2.GaussianBlur(frame, (51, 51), 20.0)
    passed, reason, metrics = gate.evaluate(blurred)
    assert passed is False
    assert reason == "frame_blurry"
    assert metrics["blur_score"] < gate.tau_blur
    assert gate.rejection_counts["frame_blurry"] == 1


def test_frame_gate_sharpness_textured_passes():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    passed, reason, metrics = gate.evaluate(frame)
    assert passed is True
    assert reason == "ok"
    assert metrics["blur_score"] >= gate.tau_blur


def test_frame_gate_exposure_underexposed_clipped_rejected():
    gate = FrameGate()
    # Create dark frame with >2% DN <= 5
    dark = np.full((832, 832, 3), 3, dtype=np.uint8)
    passed, reason, metrics = gate.evaluate(dark)
    assert passed is False
    assert reason == "exposure_underexposed"
    assert metrics["dark_fraction"] > gate.clipping_max_fraction
    assert gate.rejection_counts["exposure_underexposed"] == 1


def test_frame_gate_exposure_overexposed_clipped_rejected():
    gate = FrameGate()
    # Create bright blown-out frame with >2% DN >= 250
    bright = np.full((832, 832, 3), 253, dtype=np.uint8)
    passed, reason, metrics = gate.evaluate(bright)
    assert passed is False
    assert reason == "exposure_overexposed"
    assert metrics["bright_fraction"] > gate.clipping_max_fraction
    assert gate.rejection_counts["exposure_overexposed"] == 1


def test_frame_gate_exposure_normal_passes():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    dark_frac, bright_frac = gate.compute_exposure_fractions(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert dark_frac <= gate.clipping_max_fraction
    assert bright_frac <= gate.clipping_max_fraction


def test_frame_gate_novelty_first_frame_passes():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    passed, reason, metrics = gate.evaluate(frame)
    assert passed is True
    assert reason == "ok"
    assert metrics["displacement"] == 1.0
    assert gate.last_kept_frame_thumb is not None


def test_frame_gate_novelty_identical_frame_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    # First frame passes
    p1, r1, _ = gate.evaluate(frame)
    assert p1 is True
    # Exact duplicate frame evaluated immediately after
    p2, r2, m2 = gate.evaluate(frame)
    assert p2 is False
    assert r2 == "scene_not_novel"
    assert m2["displacement"] <= gate.min_novelty_displacement
    assert gate.rejection_counts["scene_not_novel"] == 1


def test_frame_gate_novelty_small_shift_rejected():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    p1, _, _ = gate.evaluate(frame)
    assert p1 is True

    # Small translation (shift 20px out of 832px = ~2.4% << 60%)
    shifted_small = np.roll(frame, 20, axis=1)
    p2, r2, m2 = gate.evaluate(shifted_small)
    assert p2 is False
    assert r2 == "scene_not_novel"
    assert m2["displacement"] <= gate.min_novelty_displacement


def test_frame_gate_novelty_large_shift_passes_and_updates_reference():
    gate = FrameGate()
    frame1 = _make_sharp_vegetation_frame(seed=100)
    p1, _, _ = gate.evaluate(frame1)
    assert p1 is True

    # A novel frame at a new location along the flight path (non-circular independent patch)
    frame2 = _make_sharp_vegetation_frame(seed=200)
    p2, r2, m2 = gate.evaluate(frame2)
    assert p2 is True
    assert r2 == "ok"
    assert m2["displacement"] > gate.min_novelty_displacement


def test_frame_gate_novelty_telemetry_displacement_overrides_visual():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    gate.evaluate(frame)
    # Even if identical visual frame, explicit flight telemetry displacement > 60% takes effect
    passed, reason, metrics = gate.evaluate(frame, telemetry={"displacement": 0.65})
    assert passed is True
    assert metrics["displacement"] == 0.65


def test_frame_gate_rejected_frame_does_not_advance_novelty_reference():
    gate = FrameGate()
    frame1 = _make_sharp_vegetation_frame(seed=1)
    p1, _, _ = gate.evaluate(frame1)
    assert p1 is True
    ref1 = gate.last_kept_frame_thumb.copy()

    # Frame 2 is severely blurry -> rejected
    blurred = cv2.GaussianBlur(frame1, (51, 51), 20.0)
    p2, r2, _ = gate.evaluate(blurred)
    assert p2 is False
    assert r2 == "frame_blurry"
    # Reference thumb must NOT have changed!
    assert np.array_equal(gate.last_kept_frame_thumb, ref1)


def test_frame_gate_require_telemetry_missing_rejected():
    gate = FrameGate(require_telemetry=True)
    frame = _make_sharp_vegetation_frame()
    passed, reason, _ = gate.evaluate(frame, telemetry=None)
    assert passed is False
    assert reason == "telemetry_missing"
    assert gate.rejection_counts["telemetry_missing"] == 1


def test_frame_gate_reset_clears_state():
    gate = FrameGate()
    frame = _make_sharp_vegetation_frame()
    gate.evaluate(frame)
    assert gate.frames_evaluated == 1
    assert gate.frames_passed == 1
    assert gate.last_kept_frame_thumb is not None

    gate.reset()
    assert gate.frames_evaluated == 0
    assert gate.frames_passed == 0
    assert gate.last_kept_frame_thumb is None


def test_frame_gate_empty_or_invalid_frame_rejected():
    gate = FrameGate()
    passed, reason, _ = gate.evaluate(np.empty((0, 0, 3), dtype=np.uint8))
    assert passed is False
    assert reason == "empty_frame"

    passed2, reason2, _ = gate.evaluate(np.zeros((10, 10, 5), dtype=np.uint8))
    assert passed2 is False
    assert reason2 == "invalid_dimensions"


def test_frame_gate_selftest_executes_successfully():
    success = run_gate_selftest(video_path=None)
    assert success is True


# ==============================================================================
# Tiler Tests
# ==============================================================================

def test_tiler_geometry_and_footprint():
    tiler = Tiler(tile_size=320, grid_size=3, overlap=0.20)
    assert tiler.step == 256
    assert tiler.footprint == 832
    assert tiler.offsets == [0, 256, 512]


def test_tiler_pure_vegetation_all_nine_valid():
    tiler = Tiler()
    frame = _make_sharp_vegetation_frame(size=832)
    batch = tiler.extract(frame)

    assert len(batch.tiles) == N_TILES  # 9
    assert len(batch.boxes) == N_TILES
    assert len(batch.veg_fractions) == N_TILES
    assert batch.n_valid == N_TILES  # 9 valid
    for t in batch.tiles:
        assert t.shape == (TILE_SIZE, TILE_SIZE, 3)


def test_tiler_partial_vegetation_pads_cyclically_to_exactly_nine():
    tiler = Tiler()
    # Top 320 px is vegetation (green), bottom is bare soil (brown)
    canvas = np.full((832, 832, 3), (40, 60, 90), dtype=np.uint8)  # soil BGR
    canvas[:320, :] = (25, 160, 35)  # green canopy

    batch = tiler.extract(canvas)
    assert len(batch.tiles) == N_TILES  # exactly 9
    assert 0 < batch.n_valid < N_TILES  # only top row qualifying
    # Verify cyclic padding preserved shapes
    for t in batch.tiles:
        assert t.shape == (TILE_SIZE, TILE_SIZE, 3)


def test_tiler_degenerate_bare_soil_returns_exactly_nine():
    tiler = Tiler()
    soil = np.full((832, 832, 3), (40, 60, 90), dtype=np.uint8)
    batch = tiler.extract(soil)

    assert len(batch.tiles) == N_TILES  # 9
    assert batch.n_valid == 0  # 0 valid vegetation tiles
    for t in batch.tiles:
        assert t.shape == (TILE_SIZE, TILE_SIZE, 3)


def test_tiler_degenerate_blank_black_returns_exactly_nine():
    tiler = Tiler()
    blank = np.zeros((832, 832, 3), dtype=np.uint8)
    batch = tiler.extract(blank)

    assert len(batch.tiles) == N_TILES
    assert batch.n_valid == 0
    for t in batch.tiles:
        assert t.shape == (TILE_SIZE, TILE_SIZE, 3)


def test_tiler_arbitrary_image_size_resized_to_footprint():
    tiler = Tiler()
    # 1080p full frame input
    hd_img = np.full((1080, 1920, 3), (25, 160, 35), dtype=np.uint8)
    batch = tiler.extract(hd_img)

    assert len(batch.tiles) == N_TILES
    assert batch.tiles[0].shape == (TILE_SIZE, TILE_SIZE, 3)


def test_tiler_target_size_resizing_for_engine():
    tiler = Tiler()
    frame = _make_sharp_vegetation_frame()
    batch = tiler.extract(frame, target_size=IMAGE_SIZE)

    assert len(batch.tiles) == N_TILES
    for t in batch.tiles:
        assert t.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_tiler_uses_absolute_exg_not_otsu_on_pure_canopy():
    pure_canopy = np.full((320, 320, 3), (25, 160, 35), dtype=np.uint8)
    mask, veg_frac = vegetation_mask(pure_canopy, thresh=PROVISIONAL_EXG_VEG_THRESHOLD)
    # Absolute threshold keeps 100% of pure canopy. Otsu would bisect it to ~50%.
    assert veg_frac >= 0.99
    assert np.all(mask == 255)


def test_tiler_batch_container_iteration_and_indexing():
    tiles = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(9)]
    boxes = [(0, 0, 10, 10) for _ in range(9)]
    veg_fractions = [0.5 for _ in range(9)]
    n_valid = 9

    batch = TileBatch(tiles, boxes, veg_fractions, n_valid)
    assert len(batch) == 4
    t, b, v, nv = batch
    assert len(t) == 9
    assert len(b) == 9
    assert len(v) == 9
    assert nv == 9

    arr = batch.as_numpy_batch()
    assert arr.shape == (9, 10, 10, 3)


def test_tiler_convenience_extract_tiles_wrapper():
    frame = _make_sharp_vegetation_frame()
    batch = extract_tiles(frame, target_size=224)
    assert len(batch.tiles) == 9
    assert batch.tiles[0].shape == (224, 224, 3)


def test_tiler_empty_image_raises_value_error():
    tiler = Tiler()
    with pytest.raises(ValueError):
        tiler.extract(None)
    with pytest.raises(ValueError):
        tiler.extract(np.empty((0, 0, 3), dtype=np.uint8))


def test_tiler_selftest_executes_successfully():
    success = run_tiler_selftest()
    assert success is True
