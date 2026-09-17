"""
Unit and integration tests for core/indices.py integration into edge/pipeline.py and edge/storage.py.
Covers:
  - 0% canopy (pure soil) and 100% canopy (pure green) frame evaluation
  - Cell-level canopy cover persistence and aggregation
  - Advisory vegetation block structure, percentiles, and provisional banding
  - Insufficient canopy (<15%) withholding logic
  - DGCI out-of-domain fraction (>30%) withholding logic
  - Standing guard: threshold_confirmed must NEVER be True for uncalibrated heuristics
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from configs.train_config import (
    PROVISIONAL_DGCI_MAX_OOD_FRACTION,
    PROVISIONAL_EXG_VEG_THRESHOLD,
    PROVISIONAL_MIN_CANOPY_FRACTION,
)
from core.indices import (
    aggregate_index,
    bgr_to_bandmap,
    compute_canopy_index,
    dgci,
    exg,
    tgi,
    vari,
    vegetation_mask,
)
from edge.storage import EdgeStorage


def test_zero_percent_canopy_frame():
    """Verify a pure bare soil frame produces 0.0 canopy fraction and withheld indices."""
    # Synthetic brown soil image (H=100, W=100): B=45, G=70, R=110
    # ExG = 2*70 - 110 - 45 = 140 - 155 = -15 <= 20
    soil_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    soil_bgr[:, :, 0] = 45   # Blue
    soil_bgr[:, :, 1] = 70   # Green
    soil_bgr[:, :, 2] = 110  # Red

    mask, veg_frac = vegetation_mask(soil_bgr, thresh=PROVISIONAL_EXG_VEG_THRESHOLD)
    assert veg_frac == 0.0
    assert np.count_nonzero(mask) == 0

    stats = compute_canopy_index(soil_bgr, index_name="vari", min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert stats["status"] == "insufficient_canopy"
    assert stats["mean"] is None
    assert stats["vegetation_fraction"] == 0.0


def test_hundred_percent_canopy_frame():
    """Verify a pure green canopy frame produces 1.0 canopy fraction and real index values."""
    # Synthetic lush green image (H=100, W=100): B=30, G=200, R=40
    # ExG = 2*200 - 40 - 30 = 400 - 70 = 330 -> clipped to 255 > 20
    green_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    green_bgr[:, :, 0] = 30   # Blue
    green_bgr[:, :, 1] = 200  # Green
    green_bgr[:, :, 2] = 40   # Red

    mask, veg_frac = vegetation_mask(green_bgr, thresh=PROVISIONAL_EXG_VEG_THRESHOLD)
    assert veg_frac == 1.0
    assert np.count_nonzero(mask) == 10000

    vari_stats = compute_canopy_index(green_bgr, index_name="vari", min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert vari_stats["status"] == "ok"
    assert vari_stats["mean"] is not None
    assert vari_stats["mean"] > 0.0

    exg_stats = compute_canopy_index(green_bgr, index_name="exg", min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert exg_stats["status"] == "ok"
    assert exg_stats["mean"] is not None
    assert exg_stats["mean"] > 20.0

    tgi_stats = compute_canopy_index(green_bgr, index_name="tgi", min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert tgi_stats["status"] == "ok"
    assert tgi_stats["mean"] is not None

    dgci_stats = compute_canopy_index(green_bgr, index_name="dgci", min_fraction=PROVISIONAL_MIN_CANOPY_FRACTION)
    assert dgci_stats["status"] == "ok"
    assert dgci_stats["mean"] is not None
    assert 0.0 <= dgci_stats["mean"] <= 1.0


def test_cell_canopy_cover_persistence():
    """Verify canopy cover fraction is recorded per cell in SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_test_canopy_01"
        storage.record_scan_start(scan_id=scan_id)

        # Record cell verdict with canopy_cover
        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_28.6139_77.2090",
            state="HEALTHY",
            class_id=1,
            score=0.92,
            n_frames=3,
            n_agree=3,
            canopy_cover=0.684,
        )

        conn = storage._get_connection()
        row = conn.execute(
            "SELECT canopy_cover FROM cell_verdicts WHERE scan_id = ? AND cell_id = ?;",
            (scan_id, "cell_28.6139_77.2090"),
        ).fetchone()

        assert row is not None
        assert abs(row["canopy_cover"] - 0.684) < 1e-4


def test_advisory_vegetation_block_complete_with_measured_numbers():
    """Verify create_advisory builds vegetation block conforming to §7.3 & §B6/§B7."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_test_veg_01"
        storage.record_scan_start(scan_id=scan_id)

        # Record 5 frame events with real measured indices
        for i in range(5):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-16T10:00:0%dZ" % i,
                cell_id="cell_01",
                gate_passed=True,
                gate_metrics={"blur_score": 150.0},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=1,
                confidence=0.88,
                tile_decisions=[],
                canopy_cover=0.50 + 0.05 * i,
                vari=0.25 + 0.02 * i,
                exg=35.0 + 2.0 * i,
                tgi=15.0 + 1.0 * i,
                dgci=0.55 + 0.01 * i,
                dgci_ood_frac=0.04,
                indices_status="ok",
            )

        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_01",
            state="HEALTHY",
            class_id=1,
            score=0.88,
            n_frames=5,
            n_agree=5,
            canopy_cover=0.60,
        )

        storage.record_scan_end(scan_id=scan_id, frames_evaluated=5, tiles_classified=45)
        advisory = storage.create_advisory(scan_id=scan_id)

        assert "vegetation" in advisory
        veg = advisory["vegetation"]

        assert veg["interpretation_mode"] == "relative"

        # Canopy cover
        assert "canopy_cover" in veg
        assert abs(veg["canopy_cover"]["mean"] - 0.60) < 1e-3
        assert veg["canopy_cover"]["status"] == "OK"
        assert veg["canopy_cover"]["threshold_confirmed"] is False
        assert "PROVISIONAL" in veg["canopy_cover"]["threshold_source"]

        # VARI
        assert "vari" in veg
        assert veg["vari"]["mean"] is not None
        assert veg["vari"]["band"] in ("LOWER_TAIL", "BELOW_TYPICAL", "TYPICAL", "ABOVE_TYPICAL")
        assert veg["vari"]["threshold_confirmed"] is False
        assert "PROVISIONAL" in veg["vari"]["threshold_source"]
        assert veg["vari"]["source"] == "measured"

        # ExG, TGI, DGCI
        assert veg["exg"]["mean"] is not None
        assert veg["exg"]["threshold_confirmed"] is False
        assert veg["tgi"]["mean"] is not None
        assert veg["tgi"]["threshold_confirmed"] is False
        assert veg["dgci"]["mean"] is not None
        assert veg["dgci"]["threshold_confirmed"] is False

        # Reserved NDVI
        assert veg["ndvi"] is None
        assert veg["ndvi_status"] == "PENDING_HARDWARE_FINALIZATION"


def test_advisory_insufficient_canopy_withheld():
    """Verify index means are withheld when scan canopy cover is below 15%."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_test_low_canopy"
        storage.record_scan_start(scan_id=scan_id)

        # 3 frames with canopy cover 0.08 < 0.15
        for i in range(3):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-16T10:00:0%dZ" % i,
                cell_id="cell_01",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="UNCERTAIN",
                class_id=None,
                confidence=0.0,
                tile_decisions=[],
                canopy_cover=0.08,
                vari=0.10,
                exg=15.0,
                tgi=5.0,
                dgci=0.40,
                dgci_ood_frac=0.02,
                indices_status="insufficient_canopy",
            )

        storage.record_scan_end(scan_id=scan_id, frames_evaluated=3, tiles_classified=27)
        advisory = storage.create_advisory(scan_id=scan_id)

        veg = advisory["vegetation"]
        assert veg["canopy_cover"]["status"] == "INSUFFICIENT_CANOPY"
        assert veg["vari"]["mean"] is None
        assert veg["vari"]["reason"] == "INSUFFICIENT_CANOPY_FRACTION"
        assert veg["exg"]["mean"] is None
        assert veg["exg"]["reason"] == "INSUFFICIENT_CANOPY_FRACTION"
        assert veg["tgi"]["mean"] is None
        assert veg["dgci"]["mean"] is None


def test_advisory_dgci_ood_fraction_exceeded():
    """Verify DGCI mean is withheld when out-of-domain fraction exceeds 0.30."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_test_dgci_ood"
        storage.record_scan_start(scan_id=scan_id)

        # 3 frames with good canopy but high DGCI OOD fraction (0.42 > 0.30)
        for i in range(3):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-16T10:00:0%dZ" % i,
                cell_id="cell_01",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=1,
                confidence=0.85,
                tile_decisions=[],
                canopy_cover=0.55,
                vari=0.30,
                exg=40.0,
                tgi=18.0,
                dgci=0.62,
                dgci_ood_frac=0.42,
                indices_status="ok",
            )

        storage.record_scan_end(scan_id=scan_id, frames_evaluated=3, tiles_classified=27)
        advisory = storage.create_advisory(scan_id=scan_id)

        veg = advisory["vegetation"]
        assert veg["vari"]["mean"] is not None
        assert veg["dgci"]["mean"] is None
        assert veg["dgci"]["reason"] == "OUT_OF_DOMAIN_FRACTION_EXCEEDED"
        assert veg["dgci"]["threshold"] == PROVISIONAL_DGCI_MAX_OOD_FRACTION


def test_standing_guard_threshold_confirmed_never_true():
    """
    Standing guard: in any advisory document, threshold_confirmed MUST be boolean False
    for all uncalibrated vegetation heuristics (canopy_cover, vari, exg, tgi, dgci).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_standing_guard"
        storage.record_scan_start(scan_id=scan_id)
        storage.record_frame_event(
            scan_id=scan_id,
            frame_idx=0,
            timestamp_utc="2026-09-16T10:00:00Z",
            cell_id="cell_01",
            gate_passed=True,
            gate_metrics={},
            n_valid_tiles=9,
            frame_state="HEALTHY",
            class_id=1,
            confidence=0.90,
            tile_decisions=[],
            canopy_cover=0.60,
            vari=0.32,
            exg=45.0,
            tgi=20.0,
            dgci=0.65,
            dgci_ood_frac=0.05,
        )
        storage.record_scan_end(scan_id=scan_id, frames_evaluated=1, tiles_classified=9)
        advisory = storage.create_advisory(scan_id=scan_id)

        veg = advisory["vegetation"]
        for key in ("canopy_cover", "vari", "exg", "tgi", "dgci"):
            item = veg[key]
            assert "threshold_confirmed" in item, "Missing threshold_confirmed in %s" % key
            assert item["threshold_confirmed"] is False, (
                "VIOLATION: threshold_confirmed is True in %s! Uncalibrated heuristics must NEVER be marked confirmed." % key
            )
            assert "threshold_source" in item, "Missing threshold_source in %s" % key
            assert item["threshold_source"].startswith("PROVISIONAL"), (
                "VIOLATION: threshold_source in %s does not declare PROVISIONAL!" % key
            )
