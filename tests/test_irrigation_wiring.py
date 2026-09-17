"""
tests/test_irrigation_wiring.py — Tests for thermal, ndvi, and irrigation wiring behind availability guards (K4.2, J5).
"""
import datetime
import tempfile
from pathlib import Path
import pytest

from edge.storage import EdgeStorage, get_utc_iso_now


def test_k4_2_hardware_guards_when_absent():
    """K4.2 / J5.2: When hardware/mast data is absent, blocks are emitted with available: false."""
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

        # 1. Thermal (handheld array absent)
        assert adv["thermal"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["thermal"]["reason"]

        # 2. NDVI (multispectral camera absent)
        assert adv["ndvi"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["ndvi"]["reason"]

        # 3. Irrigation (no mast telemetry)
        assert adv["irrigation"]["available"] is False
        assert "HARDWARE_NOT_CONNECTED" in adv["irrigation"]["reason"]


def test_k4_2_irrigation_insufficient_history():
    """K4.2: When only 1 or 2 mast readings are present (<6 readings or <6h span), irrigation reports available: false."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_wiring_insufficient.db"
        storage = EdgeStorage(db_path=db_path)

        now_utc = get_utc_iso_now()
        # Record only 2 readings taken 10 minutes apart
        storage.record_mast_reading({
            "seq": 1,
            "node_id": "N01",
            "utc": now_utc,
            "rtc_valid": True,
            "air_temp_c": 30.0,
            "rh_pct": 60.0,
        }, log_epoch=1, received_at=now_utc)

        storage.record_scan_start("scan_test_insuf", mode="handheld_pod")
        adv = storage.create_advisory("scan_test_insuf")
        assert adv["irrigation"]["available"] is False
        assert "INSUFFICIENT_24H_HISTORY" in adv["irrigation"]["reason"]


def test_k4_2_irrigation_calculation_with_24h_history():
    """K4.2: When sufficient 24h mast history is present (>=6 readings spanning >=6h), calculates FAO-56 Hargreaves-Samani ET0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_wiring_present.db"
        storage = EdgeStorage(db_path=db_path)

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        # Record 7 readings spanning 12 hours with Tmin=22.0, Tmax=34.0, Tmean=28.0
        temps = [22.0, 24.0, 28.0, 32.0, 34.0, 30.0, 26.0]
        for i, t in enumerate(temps):
            t_dt = now_dt - datetime.timedelta(hours=(12 - i * 2))
            t_iso = t_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            storage.record_mast_reading({
                "seq": i + 1,
                "node_id": "N01",
                "field_id": "F01",
                "utc": t_iso,
                "rtc_valid": True,
                "uptime_s": (i + 1) * 7200,
                "air_temp_c": t,
                "rh_pct": 65.0,
                "ir_object_c": t - 1.5,
                "ir_ambient_c": t + 0.5,
                "lux": 50000.0,
                "soil1_v": 1.85,
                "soil2_v": 1.90,
                "battery_v": 11.8,
                "status": {"sht40": "OK"},
            }, log_epoch=1, received_at=t_iso)

        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        storage.record_scan_start("scan_crop_1", mode="handheld_pod")
        for i in range(5):
            storage.record_frame_event(
                scan_id="scan_crop_1",
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

        adv = storage.create_advisory("scan_crop_1", days_since_planting=45, total_cycle_days=120)
        irr = adv["irrigation"]

        assert irr["available"] is True
        assert irr["method"] == "fao56_hargreaves_samani"
        assert irr["t_min_24h_c"] == 22.0
        assert irr["t_max_24h_c"] == 34.0
        assert irr["ra_mm_day"] == 15.0
        assert irr["et0_mm_day"] > 0.0
        assert irr["kc"] > 0.0
        assert irr["crop_et_mm_day"] > 0.0
        assert irr["soil1_v"] == 1.85
        assert irr["soil2_v"] == 1.90
        assert irr["battery_v"] == 11.8
