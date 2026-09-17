"""
tests/test_sensors.py — Unit and integration tests for edge/sensors.py (K1.5, K2.1, K2.3).
"""
import datetime
import json
from pathlib import Path
import tempfile
import pytest

from edge.sensors import (
    GPS,
    MastTelemetryReader,
    parse_nmea_coordinate,
    parse_nmea_sentence,
)
from edge.storage import EdgeStorage, get_utc_iso_now


def test_k2_1_nmea_coordinate_parsing():
    """Verify conversion of NMEA coordinate formats to decimal degrees."""
    # 2838.5432, N -> 28 + 38.5432 / 60 = 28.642387
    lat = parse_nmea_coordinate("2838.5432", "N")
    assert abs(lat - 28.642387) < 1e-4

    # 07712.3456, E -> 77 + 12.3456 / 60 = 77.20576
    lon = parse_nmea_coordinate("07712.3456", "E")
    assert abs(lon - 77.20576) < 1e-4

    # South and West should be negative
    lat_s = parse_nmea_coordinate("1230.0000", "S")
    assert abs(lat_s - (-12.5)) < 1e-4

    lon_w = parse_nmea_coordinate("04530.0000", "W")
    assert abs(lon_w - (-45.5)) < 1e-4


def test_k2_1_nmea_sentence_parser():
    """Test stdlib parser for GPGGA and GPRMC sentences."""
    # 1. Valid GPGGA
    gga_raw = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    gga = parse_nmea_sentence(gga_raw)
    assert gga is not None
    assert gga["type"] == "GGA"
    assert gga["valid"] is True
    assert gga["fix_quality"] == 1
    assert gga["satellites"] == 8
    assert abs(gga["latitude"] - 48.1173) < 1e-3
    assert abs(gga["longitude"] - 11.5166) < 1e-3
    assert gga["altitude_m"] == 545.4

    # 2. Valid GPRMC (Status A)
    rmc_raw = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230324,003.1,W*61"
    rmc = parse_nmea_sentence(rmc_raw)
    assert rmc is not None
    assert rmc["type"] == "RMC"
    assert rmc["valid"] is True
    assert rmc["status"] == "A"
    assert rmc["utc_iso"] == "2024-03-23T12:35:19Z"
    assert rmc["utc_timestamp"] is not None

    # 3. Invalid GPRMC (Status V / Void)
    rmc_void = "$GPRMC,123519,V,4807.038,N,01131.000,E,022.4,084.4,230324,003.1,W*76"
    rmc_v = parse_nmea_sentence(rmc_void)
    assert rmc_v is not None
    assert rmc_v["valid"] is False

    # 4. Checksum mismatch -> returns None
    corrupt = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*99"
    assert parse_nmea_sentence(corrupt) is None


def test_k2_1_gps_read_stream_and_simulation():
    """Verify GPS read_stream on recorded NMEA sentences and simulation fallback."""
    gps = GPS(port="/dev/nonexistent_uart_test")
    assert gps.is_simulation_mode() is True
    assert gps.read() is None  # Simulation fallback returns None, never fake coords

    # Feed real recorded NMEA stream
    stream = [
        "$GPGGA,092750.000,2838.5432,N,07712.3456,E,1,06,1.2,210.0,M,-35.0,M,,*7E",
        "$GPRMC,092750.000,A,2838.5432,N,07712.3456,E,0.02,120.0,170926,,,A*5B",
    ]
    fix = gps.read_stream(stream)
    assert fix is not None
    assert fix["valid"] is True
    assert fix["satellites"] == 6
    assert abs(fix["latitude"] - 28.642387) < 1e-4
    assert abs(fix["longitude"] - 77.20576) < 1e-4
    assert fix["utc_iso"] == "2026-09-17T09:27:50Z"


def test_k1_5_mast_telemetry_reader_fields_and_staleness():
    """
    K1.5: Verify MastTelemetryReader reads exact Guide §6 fields and
    computes staleness from received_at when rtc_valid is false.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sensors.db"
        storage = EdgeStorage(db_path=db_path)
        reader = MastTelemetryReader(storage)

        # 1. Empty storage -> None
        assert reader.get_latest() is None

        # 2. Store fresh reading with rtc_valid=True
        now_iso = get_utc_iso_now()
        reading_valid = {
            "seq": 10,
            "node_id": "N01",
            "field_id": "F01",
            "utc": now_iso,
            "rtc_valid": True,
            "uptime_s": 3600,
            "air_temp_c": 30.5,
            "rh_pct": 58.0,
            "ir_object_c": 29.1,
            "ir_ambient_c": 31.0,
            "lux": 65000.0,
            "soil1_v": 1.95,
            "soil2_v": 2.01,
            "battery_v": 11.75,
            "status": {"sht40": "OK", "mlx90614": "OK"},
        }
        storage.record_mast_reading(reading_valid, log_epoch=1, received_at=now_iso)

        latest = reader.get_latest()
        assert latest is not None
        assert latest["node_id"] == "N01"
        assert latest["seq"] == 10
        assert latest["air_temp_c"] == 30.5
        assert latest["rh_pct"] == 58.0
        assert latest["ir_object_c"] == 29.1
        assert latest["ir_ambient_c"] == 31.0
        assert latest["lux"] == 65000.0
        assert latest["soil1_v"] == 1.95
        assert latest["soil2_v"] == 2.01
        assert latest["battery_v"] == 11.75
        assert latest["status"] == {"sht40": "OK", "mlx90614": "OK"}
        assert latest["staleness_seconds"] < 5.0

        # 3. Store reading with rtc_valid=False:
        # utc is corrupted/bogus, but received_at is fresh -> staleness is computed from received_at!
        reading_untrusted_rtc = {
            "seq": 11,
            "node_id": "N01",
            "field_id": "F01",
            "utc": "1970-01-01T00:00:00Z",  # Unset/corrupted RTC on mast
            "rtc_valid": False,
            "uptime_s": 3700,
            "air_temp_c": 31.0,
            "rh_pct": 56.0,
            "ir_object_c": 29.5,
            "ir_ambient_c": 31.5,
            "lux": 70000.0,
            "soil1_v": 1.94,
            "soil2_v": 2.00,
            "battery_v": 11.72,
            "status": {"sht40": "OK"},
        }
        storage.record_mast_reading(reading_untrusted_rtc, log_epoch=1, received_at=now_iso)

        latest2 = reader.get_latest()
        assert latest2 is not None
        assert latest2["seq"] == 11
        assert latest2["rtc_valid"] is False
        assert latest2["staleness_seconds"] < 5.0  # Fresh because computed from received_at!

        # 4. Stale reading (>3600 seconds old) -> returns None
        stale_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        reading_stale = {
            "seq": 12,
            "node_id": "N01",
            "utc": stale_iso,
            "rtc_valid": True,
            "air_temp_c": 32.0,
        }
        storage.record_mast_reading(reading_stale, log_epoch=1, received_at=stale_iso)

        latest_stale = reader.get_latest()
        assert latest_stale is None  # Stale data rejected cleanly!
