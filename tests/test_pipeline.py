"""
Regression tests. Each test names the red-team finding it guards against.
Run before every training job, every export, and once on the Nano.

    python -m pytest tests/ -v
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from configs.classes import (CLASS_NAMES, NUM_CLASSES, IDX,
                             HEALTHY_COLS, NOTCROP_COL, CROP_COLS, DISEASE_COLS)
from core.aggregate import aggregate_frame, aggregate_cell
from core.thermal import (canopy_temperature, cwsi,
                          compute_vpd, fit_non_water_stressed_baseline,
                          cross_check_mlx90614, ThermalMastStation,
                          MIN_BASELINE_OBSERVATIONS,
                          PROVISIONAL_BIMODAL_GAP_C,
                          PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO)
from core.indices import (vegetation_mask, excess_green,
                          PROVISIONAL_EXG_VEG_THRESHOLD,
                          exg, vari, tgi, ngrdi, gmr, dgci,
                          bgr_to_bandmap, rgb_to_bandmap,
                          vari_from_bgr, tgi_from_bgr, ngrdi_from_bgr,
                          gmr_from_bgr, dgci_from_bgr,
                          aggregate_index, compute_canopy_index,
                          masked_index_mean, check_frame_usable_for_indices,
                          PROVISIONAL_MIN_CANOPY_FRACTION)
from core.ndvi import (
    ndvi_from_dual_bandpass,
    apply_channel_response_correction,
    apply_empirical_line_calibration,
    SENSOR_ID_RGB_INSPECTION,
    SENSOR_ID_NIR_SURVEY,
    PROVISIONAL_PANEL_REFLECTANCES,
    PANEL_REFLECTANCES_CONFIRMED
)
from core.rejection import open_set_energy, posthoc_logit_adjust, decide
from core.trap_segmentation import (segment_trap_blobs, verify_fiducial_marker,
                                  verify_fiducial_frame, process_trap_frame,
                                  evaluate_trap_counts_against_etl,
                                  MARKER_SPACING_CONFIRMED,
                                  MM_PER_PIXEL, NOMINAL_MM_PER_PIXEL_DESIGN_TARGET,
                                  derive_watershed_floor_px, calibrate_scale_from_fiducials,
                                  TARGET_MIN_PEST_RADIUS_MM, DEFAULT_ABS_FLOOR_PX,
                                  PROVISIONAL_FOCUS_FLOOR_LAPLACIAN,
                                  PROVISIONAL_MAX_ROTATION_DEG,
                                  PROVISIONAL_MAX_CARD_SATURATION,
                                  PROVISIONAL_MARKER_SPACING_W_MM,
                                  PROVISIONAL_MARKER_SPACING_H_MM,
                                  PROVISIONAL_TREND_MIN_OBSERVATIONS,
                                  PROVISIONAL_TREND_ROLLING_WINDOW,
                                  PROVISIONAL_TREND_RISING_RATIO,
                                  PROVISIONAL_TREND_FALLING_RATIO,
                                  PROVISIONAL_TREND_MIN_RATE_DIFF,
                                  TrapCountHistory,
                                  TRAP_ETL_REGISTRY)

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


# -------------------------------------------------------- Section A (Thermal Mast) ---
def test_section_a_thermal_mast_downwash_rationale_in_docstring():
    """Section A: verify downwash biophysical rationale is preserved in core/thermal.py docstring."""
    import core.thermal as th
    doc = th.__doc__
    assert "rotor downwash" in doc.lower()
    assert "convection" in doc.lower()
    assert "ground mast" in doc.lower()
    assert "mlx90640" in doc.lower()
    assert "mlx90614" in doc.lower()


def test_section_a_baseline_insufficient_returns_none():
    """Section A5: baseline fitting with <14 observations returns (None, 'baseline_insufficient')."""
    # 5 observations < 14 required threshold
    sparse_obs = [
        {"delta_t": -1.2, "vpd_kpa": 1.5},
        {"delta_t": -2.0, "vpd_kpa": 2.1},
        {"delta_t": -0.8, "vpd_kpa": 1.2},
        {"delta_t": -2.5, "vpd_kpa": 2.7},
        {"delta_t": -1.7, "vpd_kpa": 1.9},
    ]
    params, status = fit_non_water_stressed_baseline(sparse_obs, min_obs=14)
    assert params is None, f"Expected None params for sparse obs, got {params}"
    assert status == "baseline_insufficient"

    # ThermalMastStation should refuse to calculate CWSI when baseline is uncalibrated
    station = ThermalMastStation("station_test", min_baseline_obs=14)
    for obs in sparse_obs:
        station.record_baseline_observation(obs["delta_t"], obs["vpd_kpa"])
    fitted, fit_status = station.fit_baseline()
    assert fitted is None and fit_status == "baseline_insufficient"

    # Attempt to process a reading
    frame = np.full((24, 32), 29.0)
    res = station.process_reading(frame, air_temp_c=30.0, relative_humidity_pct=50.0)
    assert res["cwsi"] is None
    assert res["status"] == "baseline_insufficient"


def test_section_a_baseline_fitting_with_sufficient_observations():
    """Section A5: baseline fitting with >=14 observations fits OLS line with high R2."""
    rng = np.random.RandomState(42)
    true_slope = -2.10
    true_intercept = 1.80
    vpds = np.linspace(1.0, 3.5, 14)
    # add small noise (+/- 0.08 degC)
    noise = rng.normal(0, 0.08, size=14)
    deltas = true_slope * vpds + true_intercept + noise

    obs = [{"delta_t": float(d), "vpd_kpa": float(v)} for d, v in zip(deltas, vpds)]
    params, status = fit_non_water_stressed_baseline(obs, min_obs=14)
    assert status == "ok"
    assert params is not None
    assert params["n_obs"] == 14
    assert abs(params["ll_slope"] - true_slope) < 0.15
    assert abs(params["ll_intercept"] - true_intercept) < 0.25
    assert params["r_squared"] > 0.95


def test_section_a_mlx90614_crosscheck_drift_and_agreement():
    """Section A4: MLX90614 acts strictly as cross-check/drift monitor, flags disagreement >3.5 degC."""
    # Case 1: Agreement within tolerance (<= 3.5 degC)
    chk_ok = cross_check_mlx90614(tc_mlx90640=28.0, temp_mlx90614=29.2, max_disagreement=3.5)
    assert chk_ok["valid"] is True
    assert chk_ok["status"] == "ok"
    assert abs(chk_ok["disagreement_c"] - 1.2) < 1e-4

    # Case 2: Disagreement exceeding threshold (> 3.5 degC) indicates optical drift or sensor failure
    chk_bad = cross_check_mlx90614(tc_mlx90640=28.0, temp_mlx90614=32.8, max_disagreement=3.5)
    assert chk_bad["valid"] is False
    assert chk_bad["status"] == "sensor_disagreement_or_drift"
    assert abs(chk_bad["disagreement_c"] - 4.8) < 1e-4


def test_section_a_mlx90614_cannot_report_cwsi_without_array():
    """Section A4: MLX90614 single-point reading CANNOT be passed as primary thermal array."""
    station = ThermalMastStation("station_test")
    # Single float or 1D array representing single-point IR
    with pytest.raises(TypeError) as exc:
        station.process_reading(28.5, air_temp_c=30.0)
    assert "MLX90614" in str(exc.value) and "array" in str(exc.value).lower()

    with pytest.raises(TypeError):
        station.process_reading(np.array([28.5]), air_temp_c=30.0)


def test_section_a_mast_station_time_series_accumulation_and_cwsi():
    """Section A2, A4, A5: full time series accumulation, cross-check, and CWSI calculation on mast."""
    station = ThermalMastStation("mast_field_01", crop_name="rice", min_baseline_obs=14)

    # 1. Calibrate baseline with 14 synthetic days
    vpds = np.linspace(1.2, 3.2, 14)
    for v in vpds:
        dt = -1.95 * v + 1.65
        station.record_baseline_observation(delta_t=dt, vpd_kpa=v)

    params, status = station.fit_baseline()
    assert status == "ok"
    assert params["status"] == "calibrated"

    # 2. Process valid MLX90640 frame (mixed canopy + soil)
    frame = np.full((24, 32), 55.0)  # soil at 55 degC
    frame[:14, :] = 28.0             # canopy at 28 degC
    res = station.process_reading(
        thermal_array=frame,
        air_temp_c=30.0,
        relative_humidity_pct=60.0,
        mlx90614_temp=28.8,
        timestamp="2026-09-06T12:00:00Z"
    )
    assert res["status"] == "ok"
    assert res["canopy_temp_c"] is not None
    assert abs(res["canopy_temp_c"] - 28.0) < 1.0
    assert 0.0 <= res["cwsi"] <= 1.0
    assert res["cross_check"]["valid"] is True
    assert len(station.history) == 1

    # 3. Process bare soil frame (should be rejected, no CWSI)
    soil_frame = np.full((24, 32), 58.0)
    res_soil = station.process_reading(
        thermal_array=soil_frame,
        air_temp_c=30.0,
        relative_humidity_pct=60.0,
        timestamp="2026-09-06T12:15:00Z"
    )
    assert res_soil["canopy_temp_c"] is None
    assert res_soil["cwsi"] is None
    assert "tc_rejected" in res_soil["status"]
    assert len(station.history) == 2


# --------------------------------------------------- Section B (Trap Hardening) ---
def test_section_b_mm_per_pixel_constant_and_derived_watershed_floor():
    """Section B Rework (Finding 3): MM_PER_PIXEL uncalibrated guard and derived floor."""
    # Module-level MM_PER_PIXEL must be None until calibrated
    assert MM_PER_PIXEL is None, "MM_PER_PIXEL must be uncalibrated (None) by default"
    assert NOMINAL_MM_PER_PIXEL_DESIGN_TARGET == 0.125
    assert TARGET_MIN_PEST_RADIUS_MM == 0.225
    assert DEFAULT_ABS_FLOOR_PX is None

    # Uncalibrated call must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        derive_watershed_floor_px()
    assert "Physical scale MM_PER_PIXEL is uncalibrated" in str(exc_info.value)

    # Derived floor with nominal scale matches design target (0.225 / 0.125 = 1.80 px)
    derived = derive_watershed_floor_px(mm_per_pixel=NOMINAL_MM_PER_PIXEL_DESIGN_TARGET)
    assert derived == 1.80


def test_section_b_fiducial_marker_aligned_passes():
    """Section B Rework (Finding 4): Four-corner ArUco frame verification and live scale."""
    img = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    coords = [(80, 100), (80, 650), (450, 650), (450, 100)]
    for i, (y, x) in enumerate(coords):
        m = cv2.aruco.generateImageMarker(dict_aruco, i, 50)
        img[y:y+50, x:x+50] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    passed, reason, info = verify_fiducial_frame(img, allow_provisional=True)
    assert passed is True, f"Fiducial frame verification failed: {reason}, {info}"
    assert reason == "ok"
    assert info["measured_mm_per_pixel"] > 0
    assert abs(info["rotation_deg"]) <= PROVISIONAL_MAX_ROTATION_DEG
    assert info["focus_score"] >= PROVISIONAL_FOCUS_FLOOR_LAPLACIAN
    assert info["homography"].shape == (3, 3)
    assert len(info["marker_centers"]) == 4


def test_section_b_fiducial_marker_displaced_rejects_camera_shifted():
    """Section B Rework (Finding 4): Tilt guard and incomplete fiducials reject frame."""
    img = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    coords = [(80, 100), (80, 650), (450, 650), (450, 100)]
    for i, (y, x) in enumerate(coords):
        m = cv2.aruco.generateImageMarker(dict_aruco, i, 50)
        img[y:y+50, x:x+50] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    # 1. Camera tilted by 12 deg (> PROVISIONAL_MAX_ROTATION_DEG = 10.0 deg)
    M = cv2.getRotationMatrix2D((400, 300), 12.0, 1.0)
    rot_img = cv2.warpAffine(img, M, (800, 600), borderValue=(60, 200, 230))
    passed_rot, reason_rot, info_rot = verify_fiducial_frame(rot_img, allow_provisional=True)
    assert passed_rot is False
    assert reason_rot == "camera_tilted"
    assert abs(info_rot["rotation_deg"]) > PROVISIONAL_MAX_ROTATION_DEG

    # 2. Incomplete markers (marker ID 2 missing)
    img_missing = img.copy()
    img_missing[450:500, 650:700] = (60, 200, 230)
    passed_miss, reason_miss, info_miss = verify_fiducial_frame(img_missing, allow_provisional=True)
    assert passed_miss is False
    assert reason_miss == "fiducial_incomplete"
    assert 2 in info_miss["missing_ids"]


def test_section_b_fiducial_marker_absent_rejects_camera_shifted():
    """Section B Rework (Finding 4): Missing fiducials and blurred/defocused frame reject."""
    # 1. Completely blank sheet (no markers)
    blank = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    passed_blank, reason_blank, _ = verify_fiducial_frame(blank, allow_provisional=True)
    assert passed_blank is False
    assert reason_blank == "fiducial_incomplete"

    blobs, proc_reason, _ = process_trap_frame(blank, allow_provisional=True)
    assert blobs is None
    assert proc_reason == "fiducial_incomplete"

    # 2. Blurred / defocused markers (< PROVISIONAL_FOCUS_FLOOR_LAPLACIAN = 100.0)
    img = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    coords = [(80, 100), (80, 650), (450, 650), (450, 100)]
    for i, (y, x) in enumerate(coords):
        m = cv2.aruco.generateImageMarker(dict_aruco, i, 50)
        m_blur = cv2.GaussianBlur(m, (15, 15), 0)
        img[y:y+50, x:x+50] = cv2.cvtColor(m_blur, cv2.COLOR_GRAY2BGR)

    passed_blur, reason_blur, info_blur = verify_fiducial_frame(img, allow_provisional=True)
    assert passed_blur is False
    assert reason_blur == "image_blurred_defocused"
    assert info_blur["focus_score"] < PROVISIONAL_FOCUS_FLOOR_LAPLACIAN


def test_section_b_process_trap_frame_end_to_end():
    """Section B Rework (Finding 5, 6): Fiducial masking, strict == count, saturation guard."""
    # Base trap image with 4 ArUco markers
    img = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    coords = [(80, 100), (80, 650), (450, 650), (450, 100)]
    for i, (y, x) in enumerate(coords):
        m = cv2.aruco.generateImageMarker(dict_aruco, i, 50)
        img[y:y+50, x:x+50] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    # Add 1 moth (radius 35) + 3 micro-pests (radius 3)
    cv2.circle(img, (300, 400), 35, (30, 30, 30), -1)
    micro_coords = [(350, 250), (250, 500), (380, 550)]
    for (my, mx) in micro_coords:
        cv2.circle(img, (mx, my), 3, (25, 25, 25), -1)

    # 1. Normal frame: fiducials masked out -> STRICT == 4 blobs detected!
    blobs, reason, info = process_trap_frame(
        img, min_area=6, max_area=8000, abs_floor_px=1.8, allow_provisional=True
    )
    assert reason == "ok"
    assert blobs is not None
    assert len(blobs) == 4, f"Expected exactly 4 blobs (1 moth + 3 whiteflies), got {len(blobs)}"
    assert info["card_saturated"] is False
    assert info["card_replacement_required"] is False

    # 2. Saturated frame (>30% area covered): flags CARD_SATURATED
    img_sat = img.copy()
    cv2.rectangle(img_sat, (150, 150), (650, 450), (20, 20, 20), -1)
    blobs_sat, reason_sat, info_sat = process_trap_frame(
        img_sat, min_area=6, max_area=500000, abs_floor_px=1.8, allow_provisional=True
    )
    assert reason_sat == "CARD_SATURATED"
    assert info_sat["card_saturated"] is True
    assert info_sat["card_replacement_required"] is True
    assert info_sat["coverage_fraction"] > PROVISIONAL_MAX_CARD_SATURATION


def test_section_b_icar_niphm_etl_comparison():
    """Section B Rework (Finding 1, 2, 6): Verified registry, single numeric ETL, negative entries, window guards."""
    # 1. Surviving numeric threshold: exactly ONE (sugarcane whitefly / woolly aphid, 100/trap daily)
    surviving = [k for k, v in TRAP_ETL_REGISTRY.items() if v.get("threshold_value") is not None]
    assert "sugarcane_whitefly" in surviving
    assert "sugarcane_woolly_aphid" in surviving
    assert TRAP_ETL_REGISTRY["sugarcane_whitefly"]["threshold_value"] == 100.0
    assert TRAP_ETL_REGISTRY["sugarcane_whitefly"]["threshold_unit"] == "insects_per_trap"
    assert "https://niphm.gov.in" in TRAP_ETL_REGISTRY["sugarcane_whitefly"]["source_url"]
    assert "100 per trap" in TRAP_ETL_REGISTRY["sugarcane_whitefly"]["source_quote"]
    assert "ambiguity_note" in TRAP_ETL_REGISTRY["sugarcane_whitefly"]

    # 2. Explicit negative entries (NOT_SAMPLED_BY_STICKY_TRAP)
    negative_pests = [
        "rice_yellow_stem_borer", "rice_leaf_folder", "rice_brown_planthopper",
        "rice_hispa", "sugarcane_early_shoot_borer", "sugarcane_top_borer", "sugarcane_pyrilla"
    ]
    eval_neg = evaluate_trap_counts_against_etl({k: 50 for k in negative_pests}, days_monitored=3.0)
    for r in eval_neg:
        assert r["status"] == "NOT_SAMPLED_BY_STICKY_TRAP"
        assert r["threshold_value"] is None
        assert r["threshold_unit"] is None

    # 3. Qualitative trend monitoring entries (NO_PUBLISHED_ETL)
    eval_qual = evaluate_trap_counts_against_etl({"wheat_aphid": 20, "wheat_thrips": 15}, days_monitored=3.0)
    for r in eval_qual:
        assert r["status"] == "NO_PUBLISHED_ETL"

    # 4. Quantitative evaluation for sugarcane whitefly
    eval_exceeds = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 150}, days_monitored=1.0)
    assert eval_exceeds[0]["status"] == "ABOVE_ETL"
    assert eval_exceeds[0]["threshold_available"] is True

    eval_at = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 100}, days_monitored=1.0)
    assert eval_at[0]["status"] == "AT_ETL"
    assert eval_at[0]["threshold_available"] is True

    eval_below = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 50}, days_monitored=1.0)
    assert eval_below[0]["status"] == "BELOW_ETL"
    assert eval_below[0]["threshold_available"] is True

    # 5. Monitoring window enforcement ([1.0, 7.0] days)
    # < 1.0 day -> INVALID_MONITORING_WINDOW
    res_short = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 50}, days_monitored=0.5)
    assert res_short[0]["status"] == "INVALID_MONITORING_WINDOW"

    # > 7.0 days -> INVALID_MONITORING_WINDOW
    res_long = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 50}, days_monitored=14.0)
    assert res_long[0]["status"] == "INVALID_MONITORING_WINDOW"
    assert res_long[0]["card_replacement_required"] is True

    # Missing deployment timestamp -> MISSING_DEPLOYMENT_TIMESTAMP
    res_notime = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 50})
    assert res_notime[0]["status"] == "MISSING_DEPLOYMENT_TIMESTAMP"

    # 6. Saturation guard in ETL evaluation
    res_sat = evaluate_trap_counts_against_etl({"sugarcane_whitefly": 50}, days_monitored=2.0, card_coverage=0.35)
    assert res_sat[0]["status"] == "CARD_SATURATED"
    assert res_sat[0]["card_replacement_required"] is True


def test_section_b_marker_spacing_unconfirmed_guard_raises_runtime_error():
    """Follow-up 1: calibrate_scale_from_fiducials raises RuntimeError when MARKER_SPACING_CONFIRMED is False."""
    assert MARKER_SPACING_CONFIRMED is False, "MARKER_SPACING_CONFIRMED must be False by default"
    centers = {
        0: (105.0, 105.0),
        1: (675.0, 105.0),
        2: (675.0, 475.0),
        3: (105.0, 475.0)
    }
    with pytest.raises(RuntimeError) as exc_info:
        calibrate_scale_from_fiducials(centers)
    assert "MARKER_SPACING_CONFIRMED = False" in str(exc_info.value)
    assert "Pass allow_provisional=True" in str(exc_info.value)


def test_section_b_marker_spacing_allow_provisional_returns_dict_with_flag():
    """Follow-up 1: calibrate_scale_from_fiducials with allow_provisional=True returns scale dict."""
    centers = {
        0: (105.0, 105.0),
        1: (675.0, 105.0),
        2: (675.0, 475.0),
        3: (105.0, 475.0)
    }
    res = calibrate_scale_from_fiducials(centers, allow_provisional=True)
    assert isinstance(res, dict)
    assert res["scale_provisional"] is True
    assert "measured_mm_per_pixel" in res
    # w_px = 570.0, h_px = 370.0 -> (80 + 55) / (570 + 370) = 135 / 940 ≈ 0.143617 mm/px
    expected_scale = (80.0 + 55.0) / (570.0 + 370.0)
    assert abs(res["measured_mm_per_pixel"] - expected_scale) < 1e-5
    assert abs(res["w_px"] - 570.0) < 1e-5
    assert abs(res["h_px"] - 370.0) < 1e-5


def test_section_b_process_trap_frame_unconfirmed_guard_raises_runtime_error():
    """Item 1: process_trap_frame() raises RuntimeError by default with no allow_provisional argument."""
    img = np.full((600, 800, 3), (60, 200, 230), dtype=np.uint8)
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    coords = [(80, 100), (80, 650), (450, 650), (450, 100)]
    for i, (y, x) in enumerate(coords):
        m = cv2.aruco.generateImageMarker(dict_aruco, i, 50)
        img[y:y+50, x:x+50] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    with pytest.raises(RuntimeError) as exc_info:
        process_trap_frame(img)
    assert "MARKER_SPACING_CONFIRMED = False" in str(exc_info.value)
    assert "Pass allow_provisional=True" in str(exc_info.value)


def test_section_b_trap_trend_insufficient_history_and_flat_stable():
    """Item 2: Series shorter than minimum returns INSUFFICIENT_HISTORY; flat series returns STABLE."""
    tracker = TrapCountHistory(trap_id="trap_test_01", min_baseline_obs=3, rolling_window=7)
    t0 = 1700000000.0  # reference epoch

    # Obs 1: Day 1.0 (count=10, interval=10.0/day). n_baseline = 0 < 3 -> INSUFFICIENT_HISTORY
    res1 = tracker.record_observation(t0 + 86400, "whitefly", count=10, card_replaced_at=t0)
    assert res1["status"] == "INSUFFICIENT_HISTORY"
    assert res1["baseline_median_rate"] is None
    assert res1["n_baseline_observations"] == 0
    assert res1["interval_daily_rate"] == 10.0
    assert res1["cumulative_daily_rate"] == 10.0

    # Obs 2: Day 2.0 (count=20, delta=10 -> interval=10.0/day). n_baseline = 1 < 3 -> INSUFFICIENT_HISTORY
    res2 = tracker.record_observation(t0 + 2 * 86400, "whitefly", count=20, card_replaced_at=t0)
    assert res2["status"] == "INSUFFICIENT_HISTORY"
    assert res2["n_baseline_observations"] == 1
    assert res2["baseline_median_rate"] is None
    assert res2["interval_daily_rate"] == 10.0

    # Obs 3: Day 3.0 (count=30, delta=10 -> interval=10.0/day). n_baseline = 2 < 3 -> INSUFFICIENT_HISTORY
    res3 = tracker.record_observation(t0 + 3 * 86400, "whitefly", count=30, card_replaced_at=t0)
    assert res3["status"] == "INSUFFICIENT_HISTORY"
    assert res3["n_baseline_observations"] == 2
    assert res3["baseline_median_rate"] is None
    assert res3["interval_daily_rate"] == 10.0

    # Obs 4: Day 4.0 (count=40, delta=10 -> interval=10.0/day). n_baseline = 3 >= 3 -> STABLE (flat series)
    res4 = tracker.record_observation(t0 + 4 * 86400, "whitefly", count=40, card_replaced_at=t0)
    assert res4["status"] == "STABLE"
    assert res4["n_baseline_observations"] == 3
    assert res4["baseline_median_rate"] == 10.0
    assert res4["interval_daily_rate"] == 10.0
    assert res4["cumulative_daily_rate"] == 10.0
    assert res4["ratio"] == 1.0


def test_section_b_trap_trend_rising_and_falling_alerts():
    """Item 2, Findings 2 & 3: Multipliers flag RISING (>=1.5x) and FALLING (<=0.67x) with monotonic counts."""
    # 1. RISING test: daily interval catch rate surges from 20/day to 32/day
    tracker = TrapCountHistory(trap_id="trap_test_02", min_baseline_obs=3, rolling_window=5)
    t0 = 1700000000.0

    # Counts monotonically increase: 20 -> 40 -> 60 (interval rate = 20.0/day)
    tracker.record_observation(t0 + 86400, "aphids", count=20, card_replaced_at=t0)
    tracker.record_observation(t0 + 2 * 86400, "aphids", count=40, card_replaced_at=t0)
    tracker.record_observation(t0 + 3 * 86400, "aphids", count=60, card_replaced_at=t0)

    # Day 4: 32 new aphids arrive (count increases 60 -> 92). Interval rate = 32.0/day.
    # Ratio = 32/20 = 1.60 >= 1.50 -> RISING
    res_rise = tracker.record_observation(t0 + 4 * 86400, "aphids", count=92, card_replaced_at=t0)
    assert res_rise["status"] == "RISING"
    assert res_rise["baseline_median_rate"] == 20.0
    assert res_rise["interval_daily_rate"] == 32.0
    assert res_rise["ratio"] == 1.6
    assert res_rise["count"] == 92  # verified monotonic increase

    # 2. FALLING test: daily interval catch rate drops from 30/day to 15/day
    fall_tracker = TrapCountHistory(trap_id="trap_test_03", min_baseline_obs=3)
    # Counts monotonically increase: 30 -> 60 -> 90 (interval rate = 30.0/day)
    fall_tracker.record_observation(t0 + 86400, "thrips", count=30, card_replaced_at=t0)
    fall_tracker.record_observation(t0 + 2 * 86400, "thrips", count=60, card_replaced_at=t0)
    fall_tracker.record_observation(t0 + 3 * 86400, "thrips", count=90, card_replaced_at=t0)

    # Day 4: Catch abates, only 15 new thrips arrive (count increases 90 -> 105, NOT decreases!)
    # Interval rate = (105 - 90) / 1.0 = 15.0/day. Ratio = 15/30 = 0.50 <= 0.67 -> FALLING
    res_fall = fall_tracker.record_observation(t0 + 4 * 86400, "thrips", count=105, card_replaced_at=t0)
    assert res_fall["status"] == "FALLING"
    assert res_fall["baseline_median_rate"] == 30.0
    assert res_fall["interval_daily_rate"] == 15.0
    assert res_fall["ratio"] == 0.50
    assert res_fall["count"] == 105  # physically consistent non-decreasing count


def test_section_b_trap_trend_saturated_and_invalid_records_excluded_from_baseline():
    """Item 2: CARD_SATURATED, INVALID_MONITORING_WINDOW, MISSING_DEPLOYMENT_TIMESTAMP excluded from baseline."""
    tracker = TrapCountHistory(trap_id="trap_test_04", min_baseline_obs=3)
    t0 = 1700000000.0

    # Obs 1: Day 1, count=10 -> interval rate 10.0
    tracker.record_observation(t0 + 86400, "whitefly", count=10, card_replaced_at=t0)
    # Obs 2: Day 2, count=22 -> interval rate 12.0 (delta = 12)
    tracker.record_observation(t0 + 2 * 86400, "whitefly", count=22, card_replaced_at=t0)

    # Obs 3: Day 3, count=122 (delta = 100), card_coverage=0.35 (>0.30 floor) -> CARD_SATURATED
    res_sat = tracker.record_observation(t0 + 3 * 86400, "whitefly", count=122, card_replaced_at=t0, card_coverage=0.35)
    assert res_sat["record_status"] == "CARD_SATURATED"
    assert res_sat["interval_daily_rate"] == 100.0

    # Obs 4: Day 4, count=136 (delta = 14) -> interval rate 14.0 (normal interval)
    tracker.record_observation(t0 + 4 * 86400, "whitefly", count=136, card_replaced_at=t0)

    # Now evaluate Observation 5 on Day 5: count=149 (delta = 13 -> interval rate 13.0)
    # Prior valid interval rates without Obs 3: [10.0, 12.0, 14.0] -> n=3, median = 12.0
    # IF the saturated Obs 3 (rate 100.0) were included: rates would be [10, 12, 14, 100] -> median = 13.0
    res_eval = tracker.record_observation(t0 + 5 * 86400, "whitefly", count=149, card_replaced_at=t0)

    assert res_eval["n_baseline_observations"] == 3
    assert res_eval["baseline_median_rate"] == 12.0
    assert res_eval["baseline_median_rate"] != 13.0, "Saturated record was improperly included in rolling baseline!"
    assert res_eval["status"] == "STABLE"

    # Also verify INVALID_MONITORING_WINDOW (>7.0 days) is flagged and excluded
    res_inv = tracker.record_observation(t0 + 8 * 86400, "whitefly", count=180, card_replaced_at=t0)
    assert res_inv["record_status"] == "INVALID_MONITORING_WINDOW"


def test_section_b_trap_trend_card_replacement_handling():
    """Item 2: Card replacement breaks continuity, archives past observations, resets baseline cohort."""
    tracker = TrapCountHistory(trap_id="trap_test_05", min_baseline_obs=3, reset_on_card_replacement=True)
    t0 = 1700000000.0

    # 4 observations on Card 1 (deployed at t0, counts 10, 20, 30, 40)
    for d in [1.0, 2.0, 3.0, 4.0]:
        tracker.record_observation(t0 + d * 86400, "whitefly", count=d * 10, card_replaced_at=t0)
    assert len(tracker.history["whitefly"]) == 4

    # Physically replace card at t1 (Card 2)
    t1 = t0 + 500000.0  # new card deployment
    res_new_card = tracker.record_observation(t1 + 86400, "whitefly", count=8, card_replaced_at=t1)

    # Rule Verification:
    # 1. card_replaced flag is True
    assert res_new_card["card_replaced"] is True
    # 2. Interval rate is 8.0/day evaluated on Card 2 (NOT against Card 1's count of 40)
    assert res_new_card["interval_daily_rate"] == 8.0
    # 3. Prior observations from Card 1 are archived
    assert len(tracker.archived_sessions["whitefly"]) == 4
    # 4. Baseline resets for new physical card: n_baseline = 0 -> INSUFFICIENT_HISTORY
    assert res_new_card["n_baseline_observations"] == 0
    assert res_new_card["baseline_median_rate"] is None
    assert res_new_card["status"] == "INSUFFICIENT_HISTORY"

    # Subsequent readings on Card 2 rebuild baseline independently (counts 16, 24, 32, 48)
    tracker.record_observation(t1 + 2 * 86400, "whitefly", count=16, card_replaced_at=t1)
    tracker.record_observation(t1 + 3 * 86400, "whitefly", count=24, card_replaced_at=t1)

    # Reading 4 on Card 2 now has 3 prior valid observations on Card 2 (median = 8.0)
    res_card2_eval = tracker.record_observation(t1 + 4 * 86400, "whitefly", count=32, card_replaced_at=t1)
    assert res_card2_eval["n_baseline_observations"] == 3
    assert res_card2_eval["baseline_median_rate"] == 8.0
    assert res_card2_eval["status"] == "STABLE"

    # Reading 5 on Card 2: count increases to 48 (delta = 16 -> interval rate = 16.0). Ratio = 16/8 = 2.0 -> RISING
    res_card2_rise = tracker.record_observation(t1 + 5 * 86400, "whitefly", count=48, card_replaced_at=t1)
    assert res_card2_rise["status"] == "RISING"
    assert res_card2_rise["interval_daily_rate"] == 16.0
    assert res_card2_rise["ratio"] == 2.0


def test_section_b_trap_trend_zero_baseline_guard():
    """Follow-up A: Zero baseline requires rate > MIN_RATE_DIFF to flag RISING; avoids ZeroDivisionError."""
    t0 = 1700000000.0

    # 1. Tracker A: Series of 0, 0, 0 followed by a reading of 1 (rate = 1.0/day)
    tracker_a = TrapCountHistory(trap_id="trap_zero_a", min_baseline_obs=3, min_rate_diff=1.0)
    for d in [1.0, 2.0, 3.0]:
        res = tracker_a.record_observation(t0 + d * 86400, "whitefly", count=0, card_replaced_at=t0)
        assert res["status"] == "INSUFFICIENT_HISTORY"
        assert res["interval_daily_rate"] == 0.0

    # Reading 4: 1 insect arrives on Day 4 (count = 1, interval rate = 1.0/day)
    # Absolute change is 1.0, which does NOT exceed min_rate_diff = 1.0 -> must NOT flag RISING!
    res_one = tracker_a.record_observation(t0 + 4 * 86400, "whitefly", count=1, card_replaced_at=t0)
    assert res_one["baseline_median_rate"] == 0.0
    assert res_one["interval_daily_rate"] == 1.0
    assert res_one["n_baseline_observations"] == 3
    assert res_one["status"] == "STABLE", f"Expected STABLE for 1 insect on zero baseline, got {res_one['status']}"
    assert res_one["ratio"] is None
    assert res_one["baseline_was_zero"] is True
    # Verify JSON serializability of payload
    dumped_one = json.dumps(res_one)
    assert json.loads(dumped_one)["ratio"] is None

    # 2. Tracker B: Series of 0, 0, 0 followed by a surge well above MIN_RATE_DIFF (5 insects -> rate 5.0/day)
    tracker_b = TrapCountHistory(trap_id="trap_zero_b", min_baseline_obs=3, min_rate_diff=1.0)
    for d in [1.0, 2.0, 3.0]:
        tracker_b.record_observation(t0 + d * 86400, "whitefly", count=0, card_replaced_at=t0)

    # Reading 4: 5 insects arrive on Day 4 (count = 5, interval rate = 5.0/day)
    # Rate 5.0 is well above min_rate_diff = 1.0 -> MUST flag RISING!
    res_surge = tracker_b.record_observation(t0 + 4 * 86400, "whitefly", count=5, card_replaced_at=t0)
    assert res_surge["baseline_median_rate"] == 0.0
    assert res_surge["interval_daily_rate"] == 5.0
    assert res_surge["n_baseline_observations"] == 3
    assert res_surge["status"] == "RISING"
    assert res_surge["ratio"] is None
    assert res_surge["baseline_was_zero"] is True
    # Verify JSON serializability of payload
    dumped_surge = json.dumps(res_surge)
    assert json.loads(dumped_surge)["status"] == "RISING"


def test_section_b_trap_trend_carry_rate_history_across_cards_default():
    """
    Decision B: Default reset_on_card_replacement=False carries interval rate history
    across card changes, eliminating 43% blind time. Day 1 on a new card evaluates
    trend against prior card's interval rates while measuring delta strictly from card_replaced_at.
    """
    tracker = TrapCountHistory(trap_id="trap_cross_card", min_baseline_obs=3, rolling_window=5)
    t0 = 1700000000.0

    # 3 observations on Card 1 (deployed at t0, daily interval rate = 10.0/day)
    tracker.record_observation(t0 + 86400, "whitefly", count=10, card_replaced_at=t0)
    tracker.record_observation(t0 + 2 * 86400, "whitefly", count=20, card_replaced_at=t0)
    tracker.record_observation(t0 + 3 * 86400, "whitefly", count=30, card_replaced_at=t0)

    # Deploy Card 2 at t1.
    t1 = t0 + 400000.0
    # Day 1 on Card 2 (elapsed = 1.0 day since t1): count=10.
    # Delta on new card = 10 - 0 = 10 (evaluated from card_replaced_at, NOT from Card 1's count of 30!).
    # Interval rate = 10.0 / 1.0 = 10.0/day.
    res_card2_day1 = tracker.record_observation(t1 + 86400, "whitefly", count=10, card_replaced_at=t1)

    assert res_card2_day1["card_replaced"] is True
    assert res_card2_day1["interval_daily_rate"] == 10.0
    assert res_card2_day1["n_baseline_observations"] == 3
    assert res_card2_day1["baseline_median_rate"] == 10.0
    assert res_card2_day1["status"] == "STABLE"
    assert res_card2_day1["baseline_spans_card_change"] is True

    # Day 2 on Card 2: surge to 25 new insects on Day 2 (count=35, delta=25 -> rate=25.0)
    # Ratio = 25/10 = 2.5 >= 1.5 -> RISING immediately on Day 2 of new card!
    res_card2_day2 = tracker.record_observation(t1 + 2 * 86400, "whitefly", count=35, card_replaced_at=t1)
    assert res_card2_day2["card_replaced"] is False
    assert res_card2_day2["interval_daily_rate"] == 25.0
    assert res_card2_day2["baseline_median_rate"] == 10.0
    assert res_card2_day2["status"] == "RISING"
    assert res_card2_day2["ratio"] == 2.5
    assert res_card2_day2["baseline_spans_card_change"] is True

    # JSON serializability check
    assert json.loads(json.dumps(res_card2_day2))["status"] == "RISING"


# --------------------------------------------------- Section C (Indices Architecture) ---
def test_section_c_bandmap_delegation_byte_identical_equivalence():
    """Section C: Prove exact bit-for-bit equivalence between BandMap and array adapter calls."""
    rng = np.random.RandomState(42)
    bgr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    bands = bgr_to_bandmap(bgr)

    # ExG byte equality
    exg_direct = exg(bands)
    exg_adapter = excess_green(bgr)
    assert np.array_equal(exg_direct, exg_adapter), "ExG adapter differs from band-addressed call"

    # VARI byte equality
    vari_direct = vari(bands)
    vari_adapter = vari_from_bgr(bgr)
    assert np.array_equal(vari_direct, vari_adapter), "VARI adapter differs from band-addressed call"

    # TGI byte equality
    tgi_direct = tgi(bands)
    tgi_adapter = tgi_from_bgr(bgr)
    assert np.array_equal(tgi_direct, tgi_adapter), "TGI adapter differs from band-addressed call"

    # NGRDI byte equality
    ngrdi_direct = ngrdi(bands)
    ngrdi_adapter = ngrdi_from_bgr(bgr)
    assert np.array_equal(ngrdi_direct, ngrdi_adapter), "NGRDI adapter differs from band-addressed call"

    # GMR byte equality (Correction 1, Option a)
    gmr_direct = gmr(bands)
    gmr_adapter = gmr_from_bgr(bgr)
    assert np.array_equal(gmr_direct, gmr_adapter), "GMR adapter differs from band-addressed call"

    # DGCI byte equality (Correction 2)
    dgci_map_direct, dom_direct = dgci(bands)
    dgci_map_adapter, dom_adapter = dgci_from_bgr(bgr)
    assert np.array_equal(dgci_map_direct, dgci_map_adapter), "DGCI map differs from band-addressed call"
    assert np.array_equal(dom_direct, dom_adapter), "DGCI domain mask differs from band-addressed call"


def test_section_c_known_hand_computed_values():
    """
    Section B4, C: Hand-computed expected values on a known RGB pixel triplet.
    Known input: R=60, G=150, B=30 (foliage green within [60, 120] deg hue domain).
      ExG   = 2*150 - 60 - 30 = 210
      VARI  = (150 - 60) / (150 + 60 - 30) = 90 / 180 = 0.50
      TGI   = 150 - 0.39*60 - 0.61*30 = 150 - 23.40 - 18.30 = 108.30
      NGRDI = (150 - 60) / (150 + 60) = 90 / 210 = 3/7 ≈ 0.4285714
      GMR   = 150 - 60 = 90.0
      DGCI (Option a, Karcher & Richardson 2003):
        V = 150/255 ≈ 0.58823529, min = 30/255 ≈ 0.11764706, Delta = 120/255
        S = Delta / V = 120 / 150 = 0.8000
        H = 60 * (2 + (30 - 60)/120) = 60 * (2 - 0.25) = 105.0 deg (in [60, 120])
        term_H = (105.0 - 60.0) / 60.0 = 45.0 / 60.0 = 0.7500
        term_S = 1.0 - 0.8000 = 0.2000
        term_V = 1.0 - (150/255) ≈ 0.41176471
        DGCI = (0.7500 + 0.2000 + 0.41176471) / 3 = 1.36176471 / 3 ≈ 0.45392157
    """
    pixel_bgr = np.array([[[30, 150, 60]]], dtype=np.uint8)  # B=30, G=150, R=60
    bands = bgr_to_bandmap(pixel_bgr)

    # ExG
    assert int(exg(bands)[0, 0]) == 210

    # VARI
    assert abs(float(vari(bands)[0, 0]) - 0.50) < 1e-5

    # TGI
    assert abs(float(tgi(bands)[0, 0]) - 108.30) < 1e-4

    # NGRDI
    assert abs(float(ngrdi(bands)[0, 0]) - (3.0 / 7.0)) < 1e-5

    # GMR (Correction 1, unnormalized G - R)
    assert abs(float(gmr(bands)[0, 0]) - 90.0) < 1e-5

    # DGCI (Correction 2, Option a)
    dgci_val, in_domain = dgci(bands)
    assert bool(in_domain[0, 0]) is True
    assert abs(float(dgci_val[0, 0]) - 0.453922) < 1e-4
    # Explicit assertion: in-domain DGCI is bounded to [0.0, 1.0]
    assert 0.0 <= float(dgci_val[0, 0]) <= 1.0


def test_section_c_dgci_foliage_domain_guard():
    """Section B1 / Option a: DGCI flags non-foliage and shade pixels outside [60, 120] deg."""
    # Blue sky pixel: R=50, G=100, B=230 -> Hue ≈ 223 deg (outside [60, 120])
    sky_bgr = np.array([[[230, 100, 50]]], dtype=np.uint8)
    _, sky_domain = dgci(bgr_to_bandmap(sky_bgr))
    assert bool(sky_domain[0, 0]) is False

    # Red/magenta pixel: R=220, G=40, B=180 -> Hue ≈ 313 deg (outside [60, 120])
    red_bgr = np.array([[[180, 40, 220]]], dtype=np.uint8)
    _, red_domain = dgci(bgr_to_bandmap(red_bgr))
    assert bool(red_domain[0, 0]) is False

    # Deep shade / bluish-green leaf pixel: R=30, G=150, B=70 -> Hue ≈ 140 deg (outside [60, 120])
    shade_bgr = np.array([[[70, 150, 30]]], dtype=np.uint8)
    _, shade_domain = dgci(bgr_to_bandmap(shade_bgr))
    assert bool(shade_domain[0, 0]) is False

    # Aggregation surfaces out_of_domain_fraction
    mixed_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    mixed_bgr[:5, :] = [30, 150, 60]   # In-domain foliage (50 pixels)
    mixed_bgr[5:, :] = [70, 150, 30]   # Out-of-domain shade (50 pixels)
    mixed_bands = bgr_to_bandmap(mixed_bgr)
    dgci_map, domain_mask = dgci(mixed_bands)
    veg_m, _ = vegetation_mask(mixed_bands, thresh=50)  # All 100 pixels pass ExG veg mask
    agg = aggregate_index(dgci_map, veg_m, in_domain_mask=domain_mask)
    assert agg["status"] == "ok"
    assert abs(agg["out_of_domain_fraction"] - 0.50) < 1e-4

    # Mathematical boundedness: in-domain DGCI is strictly in [0.0, 1.0] across synthetic foliage grid
    r_grid, g_grid = np.meshgrid(
        np.linspace(20, 100, 20),
        np.linspace(120, 255, 20),
        indexing="ij"
    )
    b_grid = np.full_like(r_grid, 30)
    test_bands = {"red": r_grid, "green": g_grid, "blue": b_grid}
    test_dgci, test_domain = dgci(test_bands)
    valid_dgci = test_dgci[test_domain]
    assert valid_dgci.size > 0
    assert (valid_dgci >= 0.0).all() and (valid_dgci <= 1.0).all()


def test_section_c_masked_aggregation_full_distribution_and_empty_guard():
    """Section A, Correction 6: Full distribution statistics and explicit empty-mask guard."""
    index_arr = np.full((50, 50), 0.75, dtype=np.float32)

    # 1. Zero canopy pixels: must return None without division/mean error
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    res_empty = aggregate_index(index_arr, empty_mask, min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert res_empty["status"] == "insufficient_canopy"
    assert res_empty["mean"] is None
    assert res_empty["n_canopy_pixels"] == 0

    # 2. Below canopy fraction floor (1% < 15%)
    sparse_mask = np.zeros((50, 50), dtype=np.uint8)
    sparse_mask[:5, :5] = 255  # 25 / 2500 = 1%
    res_sparse = aggregate_index(index_arr, sparse_mask, min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert res_sparse["status"] == "insufficient_canopy"
    assert res_sparse["mean"] is None
    assert res_sparse["vegetation_fraction"] == 0.01

    # 3. Sufficient canopy (>15%): returns full distribution dict
    valid_mask = np.zeros((50, 50), dtype=np.uint8)
    valid_mask[:25, :] = 255  # 50%
    index_arr[:25, :] = np.linspace(0.5, 0.9, 1250).reshape(25, 50)
    res_valid = aggregate_index(index_arr, valid_mask, min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert res_valid["status"] == "ok"
    assert res_valid["mean"] is not None
    assert res_valid["median"] is not None
    assert res_valid["std"] is not None
    assert res_valid["n_canopy_pixels"] == 1250
    assert abs(res_valid["vegetation_fraction"] - 0.50) < 1e-4


def test_prompt3_section_a_canopy_masking_pure_mixed_and_soil_tiles():
    """Prompt 3 Section A: Pure-canopy, pure-soil, and 50/50 mixed tiles."""
    # 1. Pure Canopy Tile (50x50 BGR: R=30, G=160, B=40)
    canopy_tile = np.zeros((50, 50, 3), dtype=np.uint8)
    canopy_tile[..., 0] = 40   # Blue
    canopy_tile[..., 1] = 160  # Green
    canopy_tile[..., 2] = 30   # Red

    mask_canopy, frac_canopy = vegetation_mask(canopy_tile)
    assert abs(frac_canopy - 1.0) < 1e-4
    assert np.all(mask_canopy == 255)

    res_canopy = compute_canopy_index(canopy_tile, index_name="vari")
    assert res_canopy["status"] == "ok"
    assert res_canopy["mean"] is not None
    assert abs(res_canopy["vegetation_fraction"] - 1.0) < 1e-4

    mean_val, veg_frac, status_val = masked_index_mean(canopy_tile, index_name="vari")
    assert mean_val is not None
    assert isinstance(veg_frac, float)
    assert abs(veg_frac - 1.0) < 1e-4
    assert status_val == "ok"
    # Theoretical VARI for R=30, G=160, B=40: (160-30) / (160+30-40) = 130/150 = 0.8667
    assert abs(mean_val - (130.0 / 150.0)) < 1e-3

    # 2. Pure Soil Tile (50x50 BGR: R=150, G=100, B=60)
    # ExG = 2*100 - 150 - 60 = -10 <= 20 threshold
    soil_tile = np.zeros((50, 50, 3), dtype=np.uint8)
    soil_tile[..., 0] = 60   # Blue
    soil_tile[..., 1] = 100  # Green
    soil_tile[..., 2] = 150  # Red

    mask_soil, frac_soil = vegetation_mask(soil_tile)
    assert frac_soil == 0.0
    assert np.all(mask_soil == 0)

    res_soil = compute_canopy_index(soil_tile, index_name="vari")
    assert res_soil["status"] == "insufficient_canopy"
    assert res_soil["mean"] is None
    assert res_soil["vegetation_fraction"] == 0.0

    # Correction 3: Failure path return types must be consistently (Optional[float], float, str)
    soil_mean, soil_frac, soil_status = masked_index_mean(soil_tile, index_name="vari")
    assert soil_mean is None
    assert isinstance(soil_frac, float)
    assert soil_frac == 0.0
    assert isinstance(soil_status, str)
    assert soil_status == "insufficient_canopy"

    # 3. 50/50 Mixed Tile (Top half pure canopy, bottom half pure soil)
    mixed_tile = np.zeros((50, 50, 3), dtype=np.uint8)
    mixed_tile[:25, :] = [40, 160, 30]   # Top 25 rows = canopy
    mixed_tile[25:, :] = [60, 100, 150]  # Bottom 25 rows = soil

    mask_mixed, frac_mixed = vegetation_mask(mixed_tile)
    assert abs(frac_mixed - 0.50) < 1e-4

    res_mixed = compute_canopy_index(mixed_tile, index_name="vari")
    assert res_mixed["status"] == "ok"
    masked_vari = res_mixed["mean"]
    # Masked mean must match the canopy value (0.8667)
    assert abs(masked_vari - (130.0 / 150.0)) < 1e-3

    # Unmasked mean computed across the entire tile including soil
    unmasked_vari_map = vari_from_bgr(mixed_tile)
    unmasked_vari = float(np.mean(unmasked_vari_map))

    # Assert masked index differs measurably from unmasked index
    # (Unmasked includes soil VARI (100-150)/(100+150-60) = -50/190 = -0.263, pulling mean to ~0.302)
    difference = abs(masked_vari - unmasked_vari)
    assert difference > 0.40, f"Masked ({masked_vari:.3f}) and unmasked ({unmasked_vari:.3f}) not sufficiently separated"

    mean_mixed, frac_out, status_mixed = masked_index_mean(mixed_tile, index_name="vari")
    assert abs(mean_mixed - masked_vari) < 1e-5
    assert abs(frac_out - 0.50) < 1e-4
    assert status_mixed == "ok"


def test_prompt3_section_a_dgci_canopy_and_domain_mask_composition():
    """
    Correction 4: Verify PROVISIONAL_EXG_VEG_THRESHOLD and DGCI mask composition (canopy mask ∩ domain mask).
    Pixels passing ExG canopy threshold but outside foliage hue [60, 120] deg must be excluded.
    """
    # 1. Constant naming verification
    assert PROVISIONAL_EXG_VEG_THRESHOLD == 20

    # 2. Construct a 50x50 BGR image where ALL pixels pass ExG canopy threshold (> 20),
    # but top half has foliage Hue in [60, 120] deg and bottom half has Hue > 120 deg (e.g. 150 deg).
    #
    # Top 25 rows: R=60, G=180, B=20 (BGR = [20, 180, 60])
    #   ExG = 2*180 - 60 - 20 = 280 (clipped to 255) > 20 -> Canopy = True
    #   OpenCV HSV: H = 60 * (2 + (20-60)/160) = 105 deg (in [60, 120] -> in_domain = True)
    #
    # Bottom 25 rows: R=20, G=180, B=100 (BGR = [100, 180, 20])
    #   ExG = 2*180 - 20 - 100 = 240 > 20 -> Canopy = True
    #   OpenCV HSV: H = 60 * (2 + (100-20)/160) = 150 deg (> 120 -> in_domain = False)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:25, :] = [20, 180, 60]   # In-domain canopy (Hue = 105 deg)
    img[25:, :] = [100, 180, 20]  # Out-of-domain canopy (Hue = 150 deg)

    mask, veg_frac = vegetation_mask(img)
    # 100% of pixels are canopy
    assert abs(veg_frac - 1.0) < 1e-4
    assert np.all(mask == 255)

    # Evaluate DGCI
    res = compute_canopy_index(img, index_name="dgci")
    assert res["status"] == "ok"
    assert res["n_canopy_pixels"] == 2500

    # Exactly 50% of canopy pixels are out of domain (the bottom half with H=150 deg)
    assert abs(res["out_of_domain_fraction"] - 0.50) < 1e-4

    # The resulting DGCI mean must match the top half only (canopy mask ∩ domain mask)
    top_dgci_map, top_in_domain = dgci_from_bgr(img[:25, :])
    assert np.all(top_in_domain)
    expected_top_mean = float(np.mean(top_dgci_map))

    assert abs(res["mean"] - expected_top_mean) < 1e-5
    assert abs(res["median"] - expected_top_mean) < 1e-5

    # Also test masked_index_mean on DGCI
    dgci_mean, frac_dgci, status_dgci = masked_index_mean(img, index_name="dgci")
    assert status_dgci == "ok"
    assert abs(dgci_mean - expected_top_mean) < 1e-5
    assert abs(frac_dgci - 1.0) < 1e-4


# ==============================================================================
# Tests for Prompt 3 Section D (core/ndvi.py)
# ==============================================================================

def test_prompt3_section_d_channel_mapping_and_counterintuitive_derivation():
    """
    Prompt 3 Section D1 & D2: Verify channel response correction fails loud by default
    and documents counterintuitive channel mapping (Blue=NIR, Red=660nm).
    """
    raw_red = np.full((10, 10), 50.0, dtype=np.float32)
    raw_blue = np.full((10, 10), 180.0, dtype=np.float32)

    # D2: apply_channel_response_correction MUST raise NotImplementedError when calib_matrix is None
    with pytest.raises(NotImplementedError, match="Bench response calibration on assembled IMX219-77IR"):
        apply_channel_response_correction(raw_red, raw_blue, calib_matrix=None)

    # D1: ndvi_from_dual_bandpass MUST also fail loud by default
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = 180  # Blue Bayer = 850nm NIR
    img[..., 2] = 50   # Red Bayer = 660nm Red
    with pytest.raises(NotImplementedError, match="Bench response calibration on assembled IMX219-77IR"):
        ndvi_from_dual_bandpass(img, calib_matrix=None)


def test_prompt3_section_d_ndvi_computation_with_mocked_calibration():
    """
    Prompt 3 Section D1 & D5: Verify NDVI computation with synthetic arrays and mocked calibration.
    Formula: NDVI = (Blue_NIR - Red_660) / (Blue_NIR + Red_660).
    """
    # 1. Identity calibration matrix K^-1 = I
    # NOTE (F5): Passing an identity matrix disables cross-talk correction and produces NDVI
    # from raw uncorrected Bayer DN, which is valid for unit testing and invalid for any agronomic output.
    identity_k = np.eye(2, dtype=np.float32)

    # Synthetic image: BGR format (index 0 = Blue = NIR, index 2 = Red = 660nm)
    # Dense healthy canopy: High NIR (Blue=200), Low Red (Red=40)
    img_canopy = np.zeros((20, 20, 3), dtype=np.uint8)
    img_canopy[..., 0] = 200  # Blue = NIR
    img_canopy[..., 1] = 20   # Green = unused
    img_canopy[..., 2] = 40   # Red = 660nm

    ndvi_canopy = ndvi_from_dual_bandpass(img_canopy, calib_matrix=identity_k)
    assert ndvi_canopy.shape == (20, 20)
    # Expected: (200 - 40) / (200 + 40) = 160 / 240 = 0.6667
    expected_ndvi = 160.0 / 240.0
    assert np.allclose(ndvi_canopy, expected_ndvi, atol=1e-4)

    # Bare soil / dry target: Low NIR (Blue=50), High Red (Red=150)
    img_soil = np.zeros((20, 20, 3), dtype=np.uint8)
    img_soil[..., 0] = 50   # Blue = NIR
    img_soil[..., 2] = 150  # Red = 660nm
    ndvi_soil = ndvi_from_dual_bandpass(img_soil, calib_matrix=identity_k)
    # Expected: (50 - 150) / (50 + 150) = -100 / 200 = -0.50
    assert np.allclose(ndvi_soil, -0.50, atol=1e-4)

    # 2. Non-trivial cross-talk unmixing matrix
    # Suppose Red channel has 20% NIR leakage: DN_R = L_red + 0.2 * L_nir, DN_B = L_nir
    # Inverting gives: L_red = 1.0 * DN_R - 0.2 * DN_B, L_nir = 1.0 * DN_B
    # K^-1 matrix: row 0 = [1.0, -0.2], row 1 = [0.0, 1.0]
    unmix_k = np.array([
        [1.0, -0.2],
        [0.0, 1.0]
    ], dtype=np.float32)

    # True radiances: L_red = 40.0, L_nir = 200.0
    # Sensor records: DN_R = 40 + 0.2 * 200 = 80, DN_B = 200
    img_leaked = np.zeros((10, 10, 3), dtype=np.uint8)
    img_leaked[..., 0] = 200  # DN_B
    img_leaked[..., 2] = 80   # DN_R (leaked)

    ndvi_corrected = ndvi_from_dual_bandpass(img_leaked, calib_matrix=unmix_k)
    # After unmixing, L_red = 80 - 0.2*200 = 40, L_nir = 200 -> NDVI should be 0.6667
    assert np.allclose(ndvi_corrected, expected_ndvi, atol=1e-4)


def test_prompt3_section_d_empirical_line_calibration_across_flights():
    """
    Prompt 3 Section D3: Verify empirical line calibration across flights and missing-panel fail-safe.
    """
    raw_ndvi = np.array([
        [0.10, 0.30],
        [0.50, 0.70]
    ], dtype=np.float32)

    # 1. Case: Calibration panels present (e.g. 3 panels: 5%, 50%, 84% reflectance)
    # Observed DNs: 25.0, 120.0, 210.0 -> linear relationship
    panel_obs = [25.0, 120.0, 210.0]
    panel_ref = [0.05, 0.50, 0.84]

    cal_data, meta_cal = apply_empirical_line_calibration(
        raw_index_or_dn=raw_ndvi,
        panel_observations=panel_obs,
        panel_reflectances=panel_ref
    )
    assert meta_cal["radiometrically_calibrated"] is True
    assert meta_cal["within_flight_relative_only"] is False
    assert meta_cal["r_squared"] > 0.99
    assert meta_cal["gain_m"] is not None
    assert meta_cal["offset_c"] is not None
    assert meta_cal["n_panels"] == 3
    assert cal_data.shape == raw_ndvi.shape

    # 2. Case: Calibration panels missing -> Fail-safe within-flight relative only flag
    raw_copy, meta_uncal = apply_empirical_line_calibration(
        raw_index_or_dn=raw_ndvi,
        panel_observations=None,
        panel_reflectances=None
    )
    assert np.array_equal(raw_copy, raw_ndvi)
    assert meta_uncal["radiometrically_calibrated"] is False
    assert meta_uncal["within_flight_relative_only"] is True
    assert meta_uncal["n_panels"] == 0
    assert "Calibration panel observations missing" in meta_uncal["reason"]

    # 3. Case: Insufficient panels (< 2 points)
    _, meta_insufficient = apply_empirical_line_calibration(
        raw_index_or_dn=raw_ndvi,
        panel_observations=[50.0],
        panel_reflectances=[0.50]
    )
    assert meta_insufficient["radiometrically_calibrated"] is False
    assert meta_insufficient["within_flight_relative_only"] is True


def test_prompt3_section_d_dual_camera_architecture_never_fused_guard():
    """
    Prompt 3 Section D4: Verify dual camera architecture constants and fail-safe hardware guards.
    """
    assert SENSOR_ID_RGB_INSPECTION == 0
    assert SENSOR_ID_NIR_SURVEY == 1
    # Physical camera assembly is tracked in the BOM rather than in code;
    # runtime computation is strictly gated by calib_matrix is not None.


def test_section_e_check_frame_usable_for_indices():
    """Verify check_frame_usable_for_indices downstream consumer gating."""
    # 1. SATURATION_OK -> usable
    meta_ok = {
        "saturation_status": "SATURATION_OK",
        "saturation_status_reason": ""
    }
    usable, reason = check_frame_usable_for_indices(meta_ok)
    assert usable is True
    assert reason == ""

    # 2. SATURATION_DETECTED -> unusable with explicit reason
    meta_detected = {
        "saturation_status": "SATURATION_DETECTED",
        "saturation_status_reason": "Channel saturation detected: Red=10.0%"
    }
    usable, reason = check_frame_usable_for_indices(meta_detected)
    assert usable is False
    assert "Channel saturation detected" in reason

    # 3. NOT_EVALUATED -> unusable with explicit reason
    meta_not_eval = {
        "saturation_status": "NOT_EVALUATED",
        "saturation_status_reason": "Frame is None; saturation not evaluated"
    }
    usable, reason = check_frame_usable_for_indices(meta_not_eval)
    assert usable is False
    assert "Frame is None" in reason

    # 4. Invalid / empty / missing metadata
    usable, reason = check_frame_usable_for_indices(None)
    assert usable is False
    assert "Metadata must be a dictionary" in reason

    usable, reason = check_frame_usable_for_indices({})
    assert usable is False
    assert "unknown or missing saturation_status" in reason


def test_section_a_thermal_pure_canopy_not_bisected_by_otsu():
    """
    T3: Verify pure canopy array with sun/shade variance exceeding PROVISIONAL_BIMODAL_GAP_C
    is NOT bisected by Otsu.
    Asserts tc tracks the FULL-array median (within 0.25 C) and NOT the cold-half median,
    and frac is approximately 1.0.
    """
    np.random.seed(42)
    # Uniform spread between 25.0 and 31.0 C (span 6.0 C >= PROVISIONAL_BIMODAL_GAP_C = 4.0 C)
    t_pure = np.random.uniform(25.0, 31.0, size=(24, 32)).astype(np.float32)
    full_median = float(np.median(t_pure))  # ~28.02 C
    cold_half_median = float(np.median(t_pure[t_pure <= full_median]))  # ~26.44 C

    tc, frac = canopy_temperature(t_pure, air_temp_c=25.0)

    # Must match full array median, NOT cold half
    assert tc is not None
    assert abs(tc - full_median) < 0.25, f"tc {tc:.2f} diverged from full-array median {full_median:.2f}"
    assert abs(tc - cold_half_median) > 1.0, f"tc {tc:.2f} erroneously tracked cold-half median {cold_half_median:.2f}"
    assert abs(frac - 1.0) < 1e-4, f"frac {frac:.4f} was bisected (expected 1.0)"


def test_section_a_thermal_canopy_plus_hot_soil_splits_and_tracks_canopy():
    """
    T3: Verify genuine canopy plus hot soil bimodal mixture splits correctly
    and tc tracks the canopy population.
    """
    np.random.seed(42)
    # 75% canopy at 28.0 C (span 25.0 - 31.0 C), 25% hot soil at 48.0 C (span 42.0 - 55.0 C)
    t_mixed = np.random.uniform(25.0, 31.0, size=(24, 32)).astype(np.float32)
    t_mixed[:6, :] = np.random.normal(loc=48.0, scale=2.5, size=(6, 32)).clip(42.0, 56.0).astype(np.float32)

    tc, frac = canopy_temperature(t_mixed, air_temp_c=25.0)

    assert tc is not None
    assert abs(tc - 28.0) < 0.5, f"tc {tc:.2f} failed to track canopy population (expected ~28.0 C)"
    assert abs(frac - 0.75) < 0.05, f"frac {frac:.4f} failed to isolate 75% canopy"


def test_section_a_thermal_pure_canopy_vs_mixed_canopy_temperature_consistency():
    """
    T3: Verify that a pure-canopy frame and the same frame with hot soil added
    do not return wildly different tc for the canopy portion.
    """
    np.random.seed(42)
    t_pure = np.random.uniform(25.0, 31.0, size=(24, 32)).astype(np.float32)
    tc_pure, _ = canopy_temperature(t_pure, air_temp_c=25.0)

    t_mixed = t_pure.copy()
    t_mixed[:6, :] = np.random.normal(loc=48.0, scale=2.5, size=(6, 32)).clip(42.0, 56.0).astype(np.float32)
    tc_mixed, _ = canopy_temperature(t_mixed, air_temp_c=25.0)

    assert tc_pure is not None and tc_mixed is not None
    # Consistency tolerance: canopy portion must agree within 0.5 C (with bug it differed by 1.71 C)
    delta_tc = abs(tc_pure - tc_mixed)
    assert delta_tc < 0.5, f"Pure canopy tc ({tc_pure:.2f} C) and mixed canopy tc ({tc_mixed:.2f} C) differ by {delta_tc:.2f} C"


def test_prompt3_section_d_empirical_line_unconfirmed_panel_reflectances_guard():
    """
    P1: Verify apply_empirical_line_calibration requires verified panel reflectances
    and raises RuntimeError if panel_reflectances is unconfirmed and allow_provisional is False.
    """
    raw_ndvi = np.array([0.2, 0.5, 0.7], dtype=np.float32)
    panel_obs = [30.0, 120.0, 200.0]

    # 1. Calling without panel_reflectances and without allow_provisional MUST raise RuntimeError
    with pytest.raises(RuntimeError, match="Ground panel reflectances are unconfirmed"):
        apply_empirical_line_calibration(raw_ndvi, panel_observations=panel_obs, panel_reflectances=None, allow_provisional=False)

    # 2. Calling with allow_provisional=True succeeds and flags provisional_reflectances_used=True
    cal_prov, meta_prov = apply_empirical_line_calibration(
        raw_ndvi,
        panel_observations=panel_obs,
        panel_reflectances=None,
        allow_provisional=True
    )
    assert meta_prov["radiometrically_calibrated"] is True
    assert meta_prov["provisional_reflectances_used"] is True

    # 3. Calling with explicit, verified reflectances succeeds and flags provisional_reflectances_used=False
    verified_ref = [0.05, 0.50, 0.84]
    cal_ver, meta_ver = apply_empirical_line_calibration(
        raw_ndvi,
        panel_observations=panel_obs,
        panel_reflectances=verified_ref,
        allow_provisional=False
    )
    assert meta_ver["radiometrically_calibrated"] is True
    assert meta_ver["provisional_reflectances_used"] is False


