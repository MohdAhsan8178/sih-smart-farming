"""
tests/test_sensors.py — Unit and integration tests for edge/sensors.py (J4.4).
"""
import datetime
import json
import socket
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
import pytest

from edge.sensors import GPS, MastTelemetryReader
from edge.storage import EdgeStorage
from gateway.server import EdgeGateway


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_j4_4_gps_simulation_fallback():
    """J4.4: GPS fallback / simulation path returns None rather than fabricating coordinates."""
    gps = GPS(port="/dev/nonexistent_serial_port_9999")
    assert gps.is_simulation_mode() is True
    reading = gps.read()
    assert reading is None  # MUST NOT fabricate coordinates


def test_j4_4_mast_telemetry_stale_and_empty():
    """J4.4: Stale path and empty path return None without fabricated defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sensors.db"
        storage = EdgeStorage(db_path=db_path)
        reader = MastTelemetryReader(storage)

        # 1. Empty storage -> None
        assert reader.get_latest() is None

        # 2. Stale storage (>3600 seconds old)
        old_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stale_payload = {
            "node_id": "SIH-NODE-01",
            "recorded_at_utc": old_time,
            "air_temp_c": 32.5,
            "humidity_pct": 65.0,
        }
        storage.record_mast_telemetry(stale_payload)

        # In DB, received_at_utc is now_utc by default in record_mast_telemetry,
        # so let us manually backdate received_at_utc in the database to simulate 2 hours ago
        conn = storage._get_connection()
        with conn:
            conn.execute("UPDATE mast_telemetry SET received_at_utc = ?;", (old_time,))

        assert reader.get_latest() is None  # Stale data returned None!


def test_j4_4_http_post_telemetry_roundtrip():
    """J4.4: Real HTTP POST of telemetry to gateway followed by sensors.py reading it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_gateway_sensors.db"
        storage = EdgeStorage(db_path=db_path)
        port = _find_free_port()
        gw = EdgeGateway(host="127.0.0.1", port=port, storage=storage)
        gw.start_background()

        try:
            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            telemetry_data = {
                "node_id": "SIH-NODE-01",
                "recorded_at_utc": now_iso,
                "firmware_version": "1.0.0",
                "air_temp_c": 28.4,
                "humidity_pct": 72.0,
                "canopy_temp_c": 26.8,
                "soil_moisture_v": [1.82, 1.85, 1.81],
                "water_level_mm": 45.0,
            }

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/mast/telemetry",
                data=json.dumps(telemetry_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                assert resp.status == 200
                res_json = json.loads(resp.read().decode("utf-8"))
                assert res_json["status"] == "ok"
                assert res_json["node_id"] == "SIH-NODE-01"

            # Now read back using MastTelemetryReader
            reader = MastTelemetryReader(storage)
            latest = reader.get_latest()
            assert latest is not None
            assert latest["node_id"] == "SIH-NODE-01"
            assert latest["air_temp_c"] == 28.4
            assert latest["humidity_pct"] == 72.0
            assert latest["canopy_temp_c"] == 26.8
            assert latest["soil_moisture_v"] == [1.82, 1.85, 1.81]
            assert latest["water_level_mm"] == 45.0
            assert latest["staleness_seconds"] < 10.0

        finally:
            gw.stop()
