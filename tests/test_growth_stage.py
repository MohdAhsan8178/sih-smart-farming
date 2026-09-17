"""
Tests for core/growth_stage.py and growth_stage advisory assembly in edge/storage.py.
"""
import pytest
import numpy as np
from core.growth_stage import (
    estimate_growth_stage,
    get_scaled_stage_lengths,
    STATUS_VERIFIED,
    STATUS_WEB_VERIFIED,
    STATUS_RECALLED_UNVERIFIED,
    STATUS_UNSOURCED,
    FAO56_STAGE_LENGTHS_BASE,
    FAO56_KC_BENCHMARKS,
    FAO56_STAGE_CANOPY_RANGES,
)
from edge.storage import EdgeStorage


FROZEN_ENUM = {STATUS_VERIFIED, STATUS_WEB_VERIFIED, STATUS_RECALLED_UNVERIFIED, STATUS_UNSOURCED}


def test_unknown_planting_date():
    """Verify missing planting date yields stage: None with DAYS_SINCE_PLANTING_REQUIRED."""
    res = estimate_growth_stage("rice", days_since_planting=None, canopy_cover=0.45)
    assert res["crop"] == "rice"
    assert res["stage"] is None
    assert res["stage_code"] is None
    assert res["reason"] == "DAYS_SINCE_PLANTING_REQUIRED"
    assert res["status"] == "AWAITING_PLANTING_DATE"
    assert res["canopy_cover_measured"] == 0.45
    assert res["canopy_cover_expected_range"] is None
    assert res["kc"] is None
    assert res["total_cycle_days"] == 150
    assert res["cycle_source"] == "default_assumption"
    assert res["cycle_verification_status"] == STATUS_RECALLED_UNVERIFIED
    assert res["verification_status"] == STATUS_WEB_VERIFIED
    assert res["verification_status"] in FROZEN_ENUM
    assert res["verification_status"] not in ("SOURCED", "PROVISIONAL")


def test_zero_and_none_canopy_cover():
    """Verify zero or None canopy cover is handled gracefully."""
    res_zero = estimate_growth_stage("wheat", days_since_planting=10, canopy_cover=0.0)
    assert res_zero["stage"] == "initial"
    assert res_zero["stage_code"] == "INI"
    assert res_zero["canopy_cover_measured"] == 0.0
    assert res_zero["canopy_cover_expected_range"] == [0.0, 0.10]
    assert res_zero["status"] == "OK"

    res_none = estimate_growth_stage("wheat", days_since_planting=10, canopy_cover=None)
    assert res_none["stage"] == "initial"
    assert res_none["canopy_cover_measured"] is None
    assert res_none["canopy_cover_expected_range"] == [0.0, 0.10]
    assert res_none["status"] == "OK"


def test_unsupported_or_invalid_crop():
    """Verify unsupported crops or invalid names return UNSOURCED and graceful reason."""
    res_bad = estimate_growth_stage("cotton", days_since_planting=30)
    assert res_bad["stage"] is None
    assert res_bad["reason"] == "UNSUPPORTED_CROP"
    assert res_bad["status"] == "UNSUPPORTED_CROP"
    assert res_bad["verification_status"] == STATUS_UNSOURCED

    res_none = estimate_growth_stage(None, days_since_planting=30)
    assert res_none["stage"] is None
    assert res_none["reason"] == "CROP_NOT_SPECIFIED"
    assert res_none["status"] == "CROP_NOT_SPECIFIED"
    assert res_none["verification_status"] == STATUS_UNSOURCED


def test_negative_days_since_planting():
    """Verify negative days since planting returns invalid input."""
    res = estimate_growth_stage("rice", days_since_planting=-3)
    assert res["stage"] is None
    assert res["reason"] == "INVALID_NEGATIVE_DAYS_SINCE_PLANTING"
    assert res["status"] == "INVALID_INPUT"


def test_variety_cycle_override_scaling():
    """Verify custom total_cycle_days proportionally scales stage lengths and updates verification status."""
    # Short-duration rice: 110 days instead of default 150 days
    scaled = get_scaled_stage_lengths("rice", 110)
    assert sum(scaled) == 110
    assert all(length >= 1 for length in scaled)

    res = estimate_growth_stage("rice", days_since_planting=20, canopy_cover=0.08, total_cycle_days=110)
    assert res["total_cycle_days"] == 110
    assert res["cycle_source"] == "farmer_override"
    assert res["cycle_verification_status"] == STATUS_VERIFIED
    assert res["verification_status"] == STATUS_WEB_VERIFIED
    assert res["verification_status"] in FROZEN_ENUM


def test_rice_stage_transitions():
    """Verify rice stage cutoffs and FAO-56 Table 11/12 parameters."""
    # Rice base durations: [30, 30, 60, 30] -> total 150
    # Day 15 -> initial (INI), kc=1.05
    res_ini = estimate_growth_stage("rice", days_since_planting=15, canopy_cover=0.05)
    assert res_ini["stage"] == "initial"
    assert res_ini["stage_code"] == "INI"
    assert res_ini["kc"] == 1.05
    assert res_ini["canopy_cover_expected_range"] == [0.0, 0.10]

    # Day 45 -> development (DEV)
    res_dev = estimate_growth_stage("rice", days_since_planting=45, canopy_cover=0.40)
    assert res_dev["stage"] == "development"
    assert res_dev["stage_code"] == "DEV"
    assert 1.05 <= res_dev["kc"] <= 1.20
    assert res_dev["canopy_cover_expected_range"] == [0.10, 0.70]

    # Day 90 -> mid_season (MID), kc=1.20
    res_mid = estimate_growth_stage("rice", days_since_planting=90, canopy_cover=0.85)
    assert res_mid["stage"] == "mid_season"
    assert res_mid["stage_code"] == "MID"
    assert res_mid["kc"] == 1.20
    assert res_mid["canopy_cover_expected_range"] == [0.70, 1.00]

    # Day 135 -> late_season (LATE)
    res_late = estimate_growth_stage("rice", days_since_planting=135, canopy_cover=0.55)
    assert res_late["stage"] == "late_season"
    assert res_late["stage_code"] == "LATE"
    assert 0.75 <= res_late["kc"] <= 1.20
    assert res_late["canopy_cover_expected_range"] == [0.20, 0.85]


def test_wheat_stage_transitions():
    """Verify wheat stage cutoffs and FAO-56 Table 11/12 parameters."""
    # Wheat base durations: [15, 25, 50, 30] -> total 120
    # Day 10 -> initial (INI)
    res_ini = estimate_growth_stage("wheat", days_since_planting=10)
    assert res_ini["stage"] == "initial"
    assert res_ini["stage_code"] == "INI"
    assert res_ini["kc"] == 0.50

    # Day 30 -> development (DEV)
    res_dev = estimate_growth_stage("wheat", days_since_planting=30)
    assert res_dev["stage"] == "development"
    assert res_dev["stage_code"] == "DEV"
    assert 0.50 <= res_dev["kc"] <= 1.15

    # Day 65 -> mid_season (MID)
    res_mid = estimate_growth_stage("wheat", days_since_planting=65)
    assert res_mid["stage"] == "mid_season"
    assert res_mid["stage_code"] == "MID"
    assert res_mid["kc"] == 1.15

    # Day 105 -> late_season (LATE)
    res_late = estimate_growth_stage("wheat", days_since_planting=105)
    assert res_late["stage"] == "late_season"
    assert res_late["stage_code"] == "LATE"
    assert 0.30 <= res_late["kc"] <= 1.15


def test_sugarcane_stage_transitions():
    """Verify sugarcane ratoon stage cutoffs and FAO-56 Table 11/12 parameters."""
    # Sugarcane base durations: [25, 70, 135, 50] -> total 280
    # Day 15 -> initial (INI)
    res_ini = estimate_growth_stage("sugarcane", days_since_planting=15)
    assert res_ini["stage"] == "initial"
    assert res_ini["stage_code"] == "INI"
    assert res_ini["kc"] == 0.40

    # Day 60 -> development (DEV)
    res_dev = estimate_growth_stage("sugarcane", days_since_planting=60)
    assert res_dev["stage"] == "development"
    assert res_dev["stage_code"] == "DEV"
    assert 0.40 <= res_dev["kc"] <= 1.25

    # Day 160 -> mid_season (MID)
    res_mid = estimate_growth_stage("sugarcane", days_since_planting=160)
    assert res_mid["stage"] == "mid_season"
    assert res_mid["stage_code"] == "MID"
    assert res_mid["kc"] == 1.25

    # Day 260 -> late_season (LATE)
    res_late = estimate_growth_stage("sugarcane", days_since_planting=260)
    assert res_late["stage"] == "late_season"
    assert res_late["stage_code"] == "LATE"
    assert 0.75 <= res_late["kc"] <= 1.25


def test_no_judgmental_labels_emitted():
    """Verify no judgmental labels like STUNTED, SPARSE, AHEAD are present in output."""
    res = estimate_growth_stage("rice", days_since_planting=80, canopy_cover=0.15)
    res_str = str(res).upper()
    for forbidden in ("STUNTED", "SPARSE", "AHEAD_OF_SCHEDULE", "BEHIND_SCHEDULE", "POOR_GROWTH", "VIGOROUS"):
        assert forbidden not in res_str


def test_storage_create_advisory_growth_stage_integration(tmp_path):
    """Verify create_advisory integrates growth_stage block into payload and database."""
    db_file = tmp_path / "test_growth_advisory.db"
    storage = EdgeStorage(db_path=db_file)
    scan_id = "scan_growth_test"

    # Record scan start with planting date metadata
    storage.record_scan_start(
        scan_id=scan_id,
        metadata={"days_since_planting": 75, "total_cycle_days": 150},
    )

    # Record frame event for rice
    storage.record_frame_event(
        scan_id=scan_id,
        frame_idx=0,
        timestamp_utc="2026-09-16T08:00:00Z",
        cell_id="cell_0",
        gate_passed=True,
        gate_metrics={},
        n_valid_tiles=9,
        frame_state="HEALTHY",
        class_id=0,
        confidence=0.92,
        tile_decisions=[],
        canopy_cover=0.82,
    )
    storage.record_scan_end(scan_id=scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)

    # Create advisory
    advisory = storage.create_advisory(scan_id=scan_id)

    assert "growth_stage" in advisory
    gs = advisory["growth_stage"]
    assert gs["crop"] == "rice"
    assert gs["stage"] == "mid_season"
    assert gs["stage_code"] == "MID"
    assert gs["days_since_planting"] == 75
    assert gs["total_cycle_days"] == 150
    assert gs["canopy_cover_measured"] == 0.82
    assert gs["canopy_cover_expected_range"] == [0.70, 1.00]
    assert gs["kc"] == 1.20
    assert gs["verification_status"] == STATUS_WEB_VERIFIED
    assert gs["verification_status"] in FROZEN_ENUM

    storage.close()


def test_storage_create_advisory_without_planting_date(tmp_path):
    """Verify create_advisory handles scan without planting date by awaiting input."""
    db_file = tmp_path / "test_growth_no_dsp.db"
    storage = EdgeStorage(db_path=db_file)
    scan_id = "scan_no_dsp"

    storage.record_scan_start(scan_id=scan_id)
    storage.record_frame_event(
        scan_id=scan_id,
        frame_idx=0,
        timestamp_utc="2026-09-16T08:00:00Z",
        cell_id="cell_0",
        gate_passed=True,
        gate_metrics={},
        n_valid_tiles=9,
        frame_state="HEALTHY",
        class_id=0,
        confidence=0.92,
        tile_decisions=[],
        canopy_cover=0.35,
    )
    storage.record_scan_end(scan_id=scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)

    advisory = storage.create_advisory(scan_id=scan_id)
    assert "growth_stage" in advisory
    gs = advisory["growth_stage"]
    assert gs["crop"] == "rice"
    assert gs["stage"] is None
    assert gs["reason"] == "DAYS_SINCE_PLANTING_REQUIRED"
    assert gs["status"] == "AWAITING_PLANTING_DATE"
    assert gs["canopy_cover_measured"] == 0.35
    assert gs["verification_status"] == STATUS_WEB_VERIFIED

    storage.close()
