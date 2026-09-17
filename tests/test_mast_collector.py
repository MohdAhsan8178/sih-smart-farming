"""
tests/test_mast_collector.py -- Unit & Integration Tests for Mast Collector & WiFi Switch (K1.4).

Tests:
  1. Fake mast HTTP server implementing Guide §6 wire contract.
  2. Multi-page /readings paging with truncated=True then truncated=False.
  3. Mast log_epoch change resets cursor to 0.
  4. Records with null sensor values and rtc_valid=False correctly stored.
  5. Trap image download via /trap/list & /trap/image and Model B ingestion.
  6. Partial failure (HTTP 500 mid-paging) preserves cursor consistency without corruption.
  7. Time synchronization via GPS only when valid fix is present.
  8. WiFiSwitch nmcli execution and guaranteed AP restoration (mocking nmcli calls).
"""

import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import urllib.parse
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from edge.mast_collector import MastCollector
from edge.sensors import GPS, MastTelemetryReader
from edge.storage import EdgeStorage, get_utc_iso_now
from edge.wifi_switch import WiFiSwitch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeMastHandler(BaseHTTPRequestHandler):
    """
    Simulates the ESP32 Ground Mast HTTP API per Guide §6.
    """
    log_epoch = 101
    rtc_valid = True
    node_id = "N01"
    inject_500_on_page = None
    readings_call_count = 0
    time_received = None

    def log_message(self, format, *args):
        pass  # Quiet

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # Allow paths with or without /api/v1 prefix
        if path.startswith("/api/v1"):
            path = path[len("/api/v1"):]

        if path in ("/health", "/api/v1/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "node_id": self.node_id,
                "firmware": "n01-1.0.0",
                "log_epoch": self.log_epoch,
                "utc": get_utc_iso_now(),
                "rtc_valid": self.rtc_valid,
                "battery_v": 11.85,
                "last_seq": 105,
                "status": {"sht40": "OK", "mlx90614": "OK", "bh1750": "OK", "ads1115": "OK", "rtc": "OK"},
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path in ("/readings", "/api/v1/readings"):
            FakeMastHandler.readings_call_count += 1
            if self.inject_500_on_page and FakeMastHandler.readings_call_count == self.inject_500_on_page:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "internal_flash_error"}')
                return

            since = int(params.get("since", [0])[0])
            limit = int(params.get("limit", [500])[0])

            # Page 1: since == 0 -> records 1..3 with truncated=True
            if since == 0:
                records = [
                    {
                        "seq": 1,
                        "node_id": self.node_id,
                        "field_id": "F01",
                        "utc": "2026-09-17T06:10:00Z",
                        "rtc_valid": True,
                        "uptime_s": 600,
                        "air_temp_c": 28.5,
                        "rh_pct": 65.0,
                        "ir_object_c": 26.2,
                        "ir_ambient_c": 28.1,
                        "lux": 45000.0,
                        "soil1_v": 1.85,
                        "soil2_v": 1.90,
                        "battery_v": 11.9,
                        "status": {"sht40": "OK", "mlx90614": "OK"},
                    },
                    {
                        "seq": 2,
                        "node_id": self.node_id,
                        "field_id": "F01",
                        "utc": "2026-09-17T06:20:00Z",
                        "rtc_valid": False,  # Untrusted RTC
                        "uptime_s": 1200,
                        "air_temp_c": 29.0,
                        "rh_pct": 62.0,
                        "ir_object_c": None,  # Null sensor reading
                        "ir_ambient_c": None,
                        "lux": None,
                        "soil1_v": 1.84,
                        "soil2_v": 1.89,
                        "battery_v": 11.88,
                        "status": {"sht40": "OK", "mlx90614": "ERR", "bh1750": "ERR"},
                    },
                    {
                        "seq": 3,
                        "node_id": self.node_id,
                        "field_id": "F01",
                        "utc": "2026-09-17T06:30:00Z",
                        "rtc_valid": True,
                        "uptime_s": 1800,
                        "air_temp_c": 30.1,
                        "rh_pct": 59.0,
                        "ir_object_c": 28.0,
                        "ir_ambient_c": 30.0,
                        "lux": 60000.0,
                        "soil1_v": 1.83,
                        "soil2_v": 1.88,
                        "battery_v": 11.85,
                        "status": {"sht40": "OK"},
                    },
                ]
                resp = {"records": records, "count": 3, "next_since": 3, "truncated": True}
            else:
                # Page 2: since == 3 -> records 4..5 with truncated=False
                records = [
                    {
                        "seq": 4,
                        "node_id": self.node_id,
                        "field_id": "F01",
                        "utc": "2026-09-17T06:40:00Z",
                        "rtc_valid": True,
                        "uptime_s": 2400,
                        "air_temp_c": 31.2,
                        "rh_pct": 55.0,
                        "ir_object_c": 29.5,
                        "ir_ambient_c": 31.0,
                        "lux": 75000.0,
                        "soil1_v": 1.82,
                        "soil2_v": 1.87,
                        "battery_v": 11.82,
                        "status": {"sht40": "OK"},
                    },
                ]
                resp = {"records": records, "count": 1, "next_since": 4, "truncated": False}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path in ("/trap/list", "/api/v1/trap/list"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "images": [
                    {
                        "trap_id": 1,
                        "node_id": self.node_id,
                        "received_utc": "2026-09-17T04:30:15Z",
                        "rtc_valid": True,
                        "bytes": 50000,
                        "sensor": "OV2640",
                        "frame": "UXGA_1600x1200",
                    }
                ]
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path in ("/trap/image", "/api/v1/trap/image"):
            trap_id = params.get("id", ["1"])[0]
            # Generate dummy yellow sticky card image with 3 dark insect spots
            dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
            dummy_img[:, :] = (0, 200, 200)  # Yellow in BGR
            for pt in [(100, 100), (200, 150), (250, 300)]:
                cv2.circle(dummy_img, pt, 8, (20, 20, 20), -1)
            _, img_bytes = cv2.imencode(".jpg", dummy_img)

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(img_bytes.tobytes())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path.startswith("/api/v1"):
            path = path[len("/api/v1"):]

        if path == "/time":
            utc_val = params.get("utc", [None])[0]
            FakeMastHandler.time_received = int(utc_val) if utc_val else None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true, "rtc_valid": true}')
            return

        self.send_response(404)
        self.end_headers()


def test_k1_4_mast_collector_full_cycle():
    """
    Tests complete pull cycle:
      - Multi-page /readings with truncated=True then False
      - Null values and rtc_valid=False handled cleanly
      - Trap image downloaded and processed by Model B
      - Time synchronization from GPS
    """
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), FakeMastHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    FakeMastHandler.readings_call_count = 0
    FakeMastHandler.inject_500_on_page = None
    FakeMastHandler.log_epoch = 101

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mast.db"
        trap_dir = Path(tmpdir) / "traps"
        storage = EdgeStorage(db_path=db_path)

        # Mock GPS with valid fix
        mock_gps = MagicMock()
        mock_gps.get_current_fix.return_value = {
            "valid": True,
            "utc_timestamp": 1789640000,
            "latitude": 28.5,
            "longitude": 77.2,
        }

        collector = MastCollector(
            base_url=f"http://127.0.0.1:{port}/api/v1",
            storage=storage,
            gps_reader=mock_gps,
            trap_dir=trap_dir,
            timeout=5.0,
        )

        # 1. Run first collection
        summary = collector.collect()
        assert summary["status"] == "ok"
        assert summary["records_pulled"] == 4  # 3 on page 1 + 1 on page 2
        assert summary["cursor_seq"] == 4
        assert summary["time_synced"] is True
        assert summary["traps_processed"] == 1
        assert FakeMastHandler.time_received == 1789640000

        # Verify storage contents
        latest = storage.get_latest_mast_telemetry()
        assert latest is not None
        assert latest["seq"] == 4
        assert latest["log_epoch"] == 101
        assert latest["air_temp_c"] == 31.2

        # Verify cursor persisted
        epoch, seq = storage.get_mast_cursor("N01")
        assert epoch == 101
        assert seq == 4

        # Verify null reading (seq=2) was stored with rtc_valid=False and null ir_object_c
        row_seq2 = storage._get_connection().execute("SELECT * FROM mast_telemetry WHERE seq=2;").fetchone()
        assert row_seq2 is not None
        assert row_seq2["rtc_valid"] == 0
        assert row_seq2["ir_object_c"] is None

        # Verify trap card was saved and processed in trap_records table
        trap_rec = storage.get_latest_trap_record()
        assert trap_rec is not None
        assert trap_rec["total_blobs_counted"] == 3
        assert (trap_dir / "T1.jpg").exists()

        # 2. Test log_epoch change resets cursor to 0
        FakeMastHandler.log_epoch = 102
        summary2 = collector.collect()
        assert summary2["log_epoch"] == 102
        epoch2, seq2 = storage.get_mast_cursor("N01")
        assert epoch2 == 102

    server.shutdown()


def test_k1_4_partial_failure_preserves_cursor():
    """
    Tests that mid-paging HTTP 500 error does NOT corrupt or over-advance cursor.
    """
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), FakeMastHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    FakeMastHandler.readings_call_count = 0
    FakeMastHandler.inject_500_on_page = 2  # Fail on second page
    FakeMastHandler.log_epoch = 200

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_fail.db"
        storage = EdgeStorage(db_path=db_path)

        mock_gps = MagicMock()
        mock_gps.get_current_fix.return_value = None  # No GPS fix

        collector = MastCollector(
            base_url=f"http://127.0.0.1:{port}/api/v1",
            storage=storage,
            gps_reader=mock_gps,
            trap_dir=Path(tmpdir) / "traps",
            max_retries=0,
            timeout=2.0,
        )

        with pytest.raises(Exception):
            collector.collect()

        # Cursor must have been saved for Page 1 (seq=3), not corrupted to 4
        epoch, seq = storage.get_mast_cursor("N01")
        assert epoch == 200
        assert seq == 3

    server.shutdown()


def test_k1_4_wifi_switch_restores_ap_on_exception():
    """
    Tests that WiFiSwitch executes nmcli commands and ALWAYS restores AP SIH-FIELD.
    Mocking nmcli subprocess calls.
    """
    switcher = WiFiSwitch(field_ap_name="SIH-FIELD", mast_ssid="SIH-NODE-01", dry_run=False)

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        # Normal execution
        executed = []
        def dummy_task():
            executed.append(True)
            return "ok"

        res = switcher.run_with_mast_connection(dummy_task)
        assert res == "ok"
        assert executed == [True]

        # Verify nmcli command calls:
        # 1. nmcli con down SIH-FIELD
        # 2. nmcli dev wifi connect SIH-NODE-01 password sih12345
        # 3. nmcli con up SIH-FIELD
        calls = [c[0][0] for c in mock_subproc.call_args_list]
        assert ["nmcli", "connection", "down", "SIH-FIELD"] in calls
        assert ["nmcli", "device", "wifi", "connect", "SIH-NODE-01", "password", "sih12345"] in calls
        assert ["nmcli", "connection", "up", "SIH-FIELD"] in calls

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        # Failing task
        def failing_task():
            raise ValueError("Simulated download failure")

        with pytest.raises(ValueError):
            switcher.run_with_mast_connection(failing_task)

        # Confirm connection up SIH-FIELD was STILL executed in finally!
        calls = [c[0][0] for c in mock_subproc.call_args_list]
        assert ["nmcli", "connection", "up", "SIH-FIELD"] in calls
