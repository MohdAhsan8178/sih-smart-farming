"""
tests/test_irrigation_wiring.py — Tests for thermal, ndvi, and irrigation wiring behind availability guards (J5).
"""
import datetime
import tempfile
from pathlib import Path
import pytest

from edge.storage import EdgeStorage


def test_j5_hardware_guards_when_absent():
    """J5.2: When hardware/mast data is absent, blocks are emitted with available: false."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_wiring_absent.db"
        storage = EdgeStorage(db_path=db_path)

        # Record a dummy scan and event
        storage.record_scan_start("scan_test_1", mode="handheld_pod")
        storage.record_frame_event(
            scan_id="scan_test_1",
            frame_idx=0,
            timestamp_utc="2026-09-17T06:00:00Z",
            cell_id="cell_0_0",
            gate_passed=True,
            gate_metrics={},
            n_valid_tiles=9,
            frame_state="HEALTHY",
            class_id=0,
            confidence=0.95,
            tile_decisions=[],
        )
        adv = storage.create_advisory("scan_test_1")

        # 1. Thermal
        assert adv["thermal"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["thermal"]["reason"]

        # 2. NDVI
        assert adv["ndvi"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["ndvi"]["reason"]

        # 3. Irrigation (no mast telemetry)
        assert adv["irrigation"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["irrigation"]["reason"]


def test_j5_irrigation_calculation_with_mast_telemetry():
    """J5.3: When mast telemetry is present, irrigation calculates FAO-56 ET0 and Kc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_wiring_present.db"
        storage = EdgeStorage(db_path=db_path)

        # Record fresh mast telemetry (28 C air, 70% RH, -50mm water level)
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        storage.record_mast_telemetry({
            "node_id": "SIH-NODE-01",
            "recorded_at_utc": now_iso,
            "air_temp_c": 28.0,
            "humidity_pct": 70.0,
            "water_level_mm": -50.0,
            "soil_moisture_v": [1.9],
        })

        # Record scan with rice crop
        storage.record_scan_start("scan_rice_1", mode="handheld_pod")
        for i in range(5):
            storage.record_frame_event(
                scan_id="scan_rice_1",
                frame_idx=i,
                timestamp_utc=now_iso,
                cell_id="cell_0_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.95,
                tile_decisions=[],
            )

        adv = storage.create_advisory("scan_rice_1", days_since_planting=45)
        irr = adv["irrigation"]

        assert irr["available"] is True
        assert irr["method"] == "fao56_hargreaves_samani"
        assert irr["air_temp_c"] == 28.0
        assert irr["et0_mm_day"] > 0.0
        assert irr["kc"] > 0.0
        assert irr["crop_et_mm_day"] > 0.0
        assert "paddy_awd" in irr
        assert irr["paddy_awd"]["water_level_mm"] == -50.0
        assert irr["paddy_awd"]["status"] == "SAFE_DRYING"  # -50mm >= -150mm
