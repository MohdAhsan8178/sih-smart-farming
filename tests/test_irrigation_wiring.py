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
        adv = storage.create_advisory("scan_test_1", replay=False)

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
        assert irr["ra_mm_day"] > 0.0
        assert irr["ra_mj_m2_day"] > 0.0
        assert irr["ra_source"] in ("GPS", "CONFIG_LATITUDE")
        assert irr["day_of_year"] >= 1
        assert irr["et0_mm_day"] > 0.0
        assert irr["kc"] > 0.0
        assert irr["crop_et_mm_day"] > 0.0
        assert irr["soil1_v"] == 1.85
        assert irr["soil2_v"] == 1.90
        assert irr["battery_v"] == 11.8


def test_l3_2_fao56_example8_radiation():
    """
    L3.2: Verify calculate_extraterrestrial_radiation_fao56 against FAO-56 Example 8:
      Location: 20°S (latitude = -20.0°)
      Date: 3 September (J = 246)
      FAO-56 Benchmark: Ra = 32.2 MJ / m^2 / day (equivalent to 13.14 mm/day)
      Assertion: Calculated Ra is within +-0.1 MJ / m^2 / day of 32.2.
    """
    from edge.irrigation_model import calculate_extraterrestrial_radiation_fao56

    res = calculate_extraterrestrial_radiation_fao56(day_of_year=246, latitude_deg=-20.0)

    ra_mj = res["ra_mj_m2_day"]
    ra_mm = res["ra_mm_day"]

    # Benchmark: 32.2 MJ/m^2/day
    assert abs(ra_mj - 32.2) <= 0.1, f"Ra {ra_mj} MJ/m^2/day deviates from 32.2 by > 0.1"
    # Benchmark mm/day: 0.408 * 32.2 = 13.1376 -> ~13.14 mm/day
    assert abs(ra_mm - 13.14) <= 0.1, f"Ra mm/day {ra_mm} deviates from 13.14 by > 0.1"
    assert res["dr"] > 0.0
    assert res["solar_declination_rad"] > 0.0
    assert res["sunset_hour_angle_rad"] > 0.0


def test_k4_2_irrigation_with_batched_readings_same_received_at():
    """
    K4.2 / Fix 7: 12 readings with utc spread across 8 hours, all sharing the same received_at
    (simulating a single batch pull from ESP32 buffer), must produce irrigation.available = True.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_batch_same_received_at.db"
        storage = EdgeStorage(db_path=db_path)

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        batch_received_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 12 readings spanning 8 hours (every ~43.6 minutes)
        # Tmin = 20.0, Tmax = 32.0
        temps = [20.0, 21.0, 23.0, 25.0, 28.0, 31.0, 32.0, 30.0, 28.0, 26.0, 24.0, 22.0]
        for i, t in enumerate(temps):
            t_dt = now_dt - datetime.timedelta(hours=(8.0 - i * (8.0 / 11.0)))
            t_iso = t_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            storage.record_mast_reading({
                "seq": i + 1,
                "node_id": "N01",
                "field_id": "F01",
                "utc": t_iso,
                "rtc_valid": True,
                "uptime_s": (i + 1) * 2400,
                "air_temp_c": t,
                "rh_pct": 55.0,
                "ir_object_c": t - 1.0,
                "ir_ambient_c": t + 0.2,
                "lux": 45000.0,
                "soil1_v": 1.80,
                "soil2_v": 1.85,
                "battery_v": 12.1,
                "status": {"sht40": "OK"},
            }, log_epoch=1, received_at=batch_received_at)

        storage.record_scan_start("scan_batch_1", mode="handheld_pod")
        for i in range(5):
            storage.record_frame_event(
                scan_id="scan_batch_1",
                frame_idx=i,
                timestamp_utc=batch_received_at,
                cell_id="cell_0_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.95,
                tile_decisions=[],
            )

        adv = storage.create_advisory("scan_batch_1", days_since_planting=45, total_cycle_days=120)
        irr = adv["irrigation"]

        assert irr["available"] is True
        assert irr["method"] == "fao56_hargreaves_samani"
        assert irr["samples_24h"] == 12
        assert irr["t_min_24h_c"] == 20.0
        assert irr["t_max_24h_c"] == 32.0
        assert irr["et0_mm_day"] > 0.0
        assert irr["crop_et_mm_day"] > 0.0


def test_k4_2_irrigation_excludes_invalid_clock_readings():
    """
    K4.2 / Fix 7: Readings with rtc_valid=0 are excluded from span/Tmin/Tmax calculation
    and reported in the advisory reason string.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_invalid_clock_exclusion.db"
        storage = EdgeStorage(db_path=db_path)

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        batch_received_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 12 readings with invalid clock (rtc_valid=False, default ESP32 boot time 2000-01-01)
        for i in range(12):
            storage.record_mast_reading({
                "seq": i + 1,
                "node_id": "N01",
                "field_id": "F01",
                "utc": "2000-01-01T00:00:00Z",
                "rtc_valid": False,
                "uptime_s": (i + 1) * 60,
                "air_temp_c": 25.0 + i * 0.1,
                "rh_pct": 50.0,
            }, log_epoch=1, received_at=batch_received_at)

        # 2 readings with valid clock taken 12 minutes apart
        for i, dt_offset in enumerate([12, 0]):
            t_dt = now_dt - datetime.timedelta(minutes=dt_offset)
            t_iso = t_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            storage.record_mast_reading({
                "seq": 13 + i,
                "node_id": "N01",
                "field_id": "F01",
                "utc": t_iso,
                "rtc_valid": True,
                "uptime_s": 1000 + i * 720,
                "air_temp_c": 28.0 + i * 2.0,
                "rh_pct": 55.0,
            }, log_epoch=1, received_at=batch_received_at)

        storage.record_scan_start("scan_invalid_clock_1", mode="handheld_pod")
        adv = storage.create_advisory("scan_invalid_clock_1")
        irr = adv["irrigation"]

        assert irr["available"] is False
        assert "INSUFFICIENT_24H_HISTORY" in irr["reason"]
        assert "found 2 readings spanning 0.2h" in irr["reason"]
        assert "12 readings excluded for invalid clock" in irr["reason"]


