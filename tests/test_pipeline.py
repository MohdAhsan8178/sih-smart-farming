"""
Regression tests. Each test names the red-team finding it guards against.
Run before every training job, every export, and once on the Nano.

    python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from configs.classes import (CLASS_NAMES, NUM_CLASSES, IDX,
                             HEALTHY_COLS, NOTCROP_COL, CROP_COLS, DISEASE_COLS)
from core.aggregate import aggregate_frame, aggregate_cell
from core.thermal import canopy_temperature, vegetation_mask
from core.rejection import open_set_energy, posthoc_logit_adjust, decide
from core.trap_segmentation import segment_trap_blobs

RICE_NORMAL = IDX['rice__normal']
RICE_BLAST = IDX['rice__blast']
RARE = IDX['sugarcane__sett_rot']          # stands in for the 43-image class


# ---------------------------------------------------------------- round 2 ---
def test_r2_healthy_frame_is_not_disease():
    """R2 Flaw 2: aggregate_frame had no HEALTHY path; noise won the argmax."""
    p = np.full((8, NUM_CLASSES), 0.002)
    p[:, RICE_NORMAL] = 0.98
    p[:, RICE_BLAST] = 0.008
    state, cid, score = aggregate_frame(p, HEALTHY_COLS, NOTCROP_COL)
    assert state == 'HEALTHY', f'healthy field returned {state}/{cid}'


def test_r2_single_lesion_tile_is_detected():
    """R2 Flaw 2: top-2 mean diluted 0.92 -> 0.47 and failed the 0.6 gate."""
    p = np.full((8, NUM_CLASSES), 0.002)
    p[:, RICE_NORMAL] = 0.98
    p[3, RICE_NORMAL] = 0.05
    p[3, RICE_BLAST] = 0.92
    state, cid, score = aggregate_frame(p, HEALTHY_COLS, NOTCROP_COL)
    assert state == 'DISEASE' and cid == RICE_BLAST, f'{state}/{cid}'
    assert score > 0.85, f'single-tile lesion diluted to {score:.2f}'


def test_r2_empty_and_tiny_input_never_nan():
    """R2 Flaw 2: n_tiles=0 gave k=0, [-0:] returned the full array, mean->NaN."""
    for arr in (np.zeros((0, NUM_CLASSES)), np.zeros((1, NUM_CLASSES)), None):
        state, cid, score = aggregate_frame(arr, HEALTHY_COLS, NOTCROP_COL)
        assert state == 'UNCERTAIN' and not np.isnan(score)


def test_r2_bare_soil_rejected_for_cwsi():
    """R2 Flaw 4: the spread gate was inverted; uniform hot soil PASSED."""
    soil = np.full((24, 32), 58.0)
    tc, info = canopy_temperature(soil, air_temp_c=38.0)
    assert tc is None, f'bare soil accepted as canopy at {tc}'
    assert info == 'no_vegetation_bare_soil'


def test_r2_mixed_canopy_soil_accepted():
    """R2 Flaw 4: valid mixed frames were rejected by the >25 degC spread gate."""
    frame = np.full((24, 32), 58.0)
    frame[:12, :] = 29.0
    tc, frac = canopy_temperature(frame, air_temp_c=38.0)
    assert tc is not None and 27.0 < tc < 32.0, f'mixed frame gave {tc}'


def test_r2_energy_excludes_notcrop_column():
    """R2 Flaw 3: energy over all logits made a trained not_crop look in-dist."""
    soil = np.full((1, NUM_CLASSES), -5.0)
    soil[0, NOTCROP_COL] = 14.0
    e = open_set_energy(soil, CROP_COLS)
    assert e[0] > 0, f'not_crop image scored in-distribution (E={e[0]:.2f})'


# ---------------------------------------------------------------- round 3 ---
def test_r3_stressed_canopy_is_not_rejected():
    """
    R3 Flaw 2: the T_air+7 gate discarded severely water-stressed canopies -
    the exact drought the sensor exists to catch.
    """
    ta = 38.0
    for tc_true in (44.0, 45.5, 47.0, 48.0):
        frame = np.full((24, 32), tc_true)
        tc, info = canopy_temperature(frame, air_temp_c=ta)
        assert tc is not None, (
            f'stressed canopy at {tc_true} degC (Ta+{tc_true-ta:.1f}) '
            f'rejected as "{info}" - drought alarm suppressed')
        assert abs(tc - tc_true) < 0.5


def test_r3_stressed_canopy_still_separates_from_soil():
    """The widened gate must still reject genuine bare soil."""
    ta = 38.0
    for soil_t in (56.0, 60.0, 65.0):
        tc, info = canopy_temperature(np.full((24, 32), soil_t), air_temp_c=ta)
        assert tc is None, f'soil at {soil_t} accepted as canopy'


def test_r3_pure_canopy_vegetation_mask_not_bisected():
    """
    R3 Flaw 3: Otsu on a unimodal ExG histogram bisects a 100% green field,
    reporting ~50% vegetation and failing every downstream purity gate.
    """
    green = np.zeros((240, 320, 3), np.uint8)
    green[..., 0] = 40                     # B
    green[..., 1] = 150                    # G
    green[..., 2] = 45                     # R
    green = green + np.random.RandomState(0).randint(-5, 6, green.shape).astype(np.int16)
    green = np.clip(green, 0, 255).astype(np.uint8)
    mask, frac = vegetation_mask(green)
    assert frac > 0.95, f'pure canopy reported only {frac:.2%} vegetation'


def test_r3_soil_frame_vegetation_mask_near_zero():
    """The absolute threshold must still reject bare soil."""
    soil = np.zeros((240, 320, 3), np.uint8)
    soil[..., 0], soil[..., 1], soil[..., 2] = 60, 90, 120   # brownish
    mask, frac = vegetation_mask(soil)
    assert frac < 0.05, f'soil reported {frac:.2%} vegetation'


def test_r3_ood_input_does_not_explode_into_rare_class():
    """
    R3 Flaw 4: post-hoc prior adjustment adds ~+6 to a 43-image class, so a
    zero-evidence OOD input could reach 85%+ confidence on the rarest disease.
    The energy gate must fire first, and the confidence gate must use
    UNADJUSTED probabilities.
    """
    counts = np.full(NUM_CLASSES, 1000.0)
    counts[RARE] = 43.0
    log_priors = np.log(counts / counts.sum())

    id_like = np.full((200, NUM_CLASSES), -2.0)
    id_like[np.arange(200), np.random.RandomState(1).randint(0, 13, 200)] = 11.0
    tau_e = float(np.percentile(open_set_energy(id_like, CROP_COLS), 95))

    ood = np.random.RandomState(2).normal(0.0, 0.4, (50, NUM_CLASSES))
    out = decide(ood, CROP_COLS, NOTCROP_COL, log_priors,
                 tau_energy=tau_e, tau_conf=0.60)
    bad = [d for d in out if d['state'] == 'OK' and d['class_id'] == RARE]
    assert not bad, f'{len(bad)}/50 OOD inputs became confident rare-class calls'


def test_r3_real_crop_still_passes_the_gates():
    """The OOD defence must not reject genuine in-distribution inputs."""
    counts = np.full(NUM_CLASSES, 1000.0); counts[RARE] = 43.0
    log_priors = np.log(counts / counts.sum())
    id_like = np.full((200, NUM_CLASSES), -2.0)
    id_like[np.arange(200), np.random.RandomState(1).randint(0, 13, 200)] = 11.0
    tau_e = float(np.percentile(open_set_energy(id_like, CROP_COLS), 95))

    out = decide(id_like[:50], CROP_COLS, NOTCROP_COL, log_priors,
                 tau_energy=tau_e, tau_conf=0.60)
    ok = sum(d['state'] == 'OK' for d in out)
    assert ok >= 45, f'only {ok}/50 real crop tiles accepted'


def test_r3_cell_states_are_distinguishable():
    """
    R3 Flaw 5: v3 returned None for healthy cells, uncertain cells and
    never-visited cells alike, so prescription mapping could not tell
    "do not spray" from "re-fly".
    """
    healthy = [('HEALTHY', RICE_NORMAL, 0.97)] * 3
    nodata = []
    weak = [('DISEASE', RICE_BLAST, 0.09)] * 3

    assert aggregate_cell(healthy)['state'] == 'HEALTHY'
    assert aggregate_cell(nodata)['state'] == 'NO_DATA'
    assert aggregate_cell(weak)['state'] == 'UNCERTAIN'

    strong = [('DISEASE', RICE_BLAST, 0.88)] * 3
    r = aggregate_cell(strong)
    assert r['state'] == 'DISEASE' and r['class_id'] == RICE_BLAST


def test_r3_energy_handles_1d_and_2d():
    """R3 Flaw 8: the runtime helper hardcoded axis=1 and crashed on 1D input."""
    v1 = np.random.RandomState(3).normal(0, 1, NUM_CLASSES)
    v2 = np.random.RandomState(3).normal(0, 1, (4, NUM_CLASSES))
    e1 = open_set_energy(v1, CROP_COLS)
    e2 = open_set_energy(v2, CROP_COLS)
    assert e1.shape == (1,) and e2.shape == (4,)


def test_r3_micro_pests_survive_next_to_a_large_insect():
    """
    R3 Flaw 1: a global 0.3*dist.max() threshold set by a large insect
    erased every whitefly and thrip on the board.
    """
    img = np.full((400, 400, 3), (60, 200, 230), np.uint8)     # yellow trap
    cv2.circle(img, (80, 80), 40, (30, 30, 30), -1)            # large moth
    micro = [(250, 120), (300, 160), (200, 260), (330, 300), (150, 330)]
    for (x, y) in micro:
        cv2.circle(img, (x, y), 3, (25, 25, 25), -1)           # whiteflies

    blobs = segment_trap_blobs(img, min_area=6, max_area=8000,
                               abs_floor_px=1.5)
    found = 0
    for _, (cx, cy), _ in blobs:
        if any(abs(cx - x) < 12 and abs(cy - y) < 12 for x, y in micro):
            found += 1
    assert found >= 4, (
        f'only {found}/5 micro-pests detected alongside a large insect '
        f'(total blobs: {len(blobs)})')
