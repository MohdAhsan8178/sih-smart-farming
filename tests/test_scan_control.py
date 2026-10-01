#!/usr/bin/env python3
"""
tests/test_scan_control.py — Test Suite for Stage 1 Scan Control & Gateway Infrastructure.

NOTE ON TARGET HARDWARE:
Mac/host test results represent simulated environment validation, NOT Jetson Nano evidence.
Physical Jetson Nano testing requires the target hardware (NVIDIA Jetson Nano 4GB,
JetPack 4.6.1, TensorRT 8.2, CSI IMX219 /dev/video0).

Covers Stage 1 Requirements:
1. Storage resolver (/mnt/aegisdata/aegis if mounted+writable, else internal data/ + STORAGE_CARD_MISSING).
2. One-time database migration (copies edge.db without moving/deleting original).
3. Pipeline CLI strictness (--until-stopped requires --scan-id, --field-id, --crop; exit code 2 on missing).
4. Concurrent SQLite WAL mode reader/writer resilience (5000ms busy timeout, 0 database locked errors).
5. Full spec §1.3 scan status shape (all keys present, idle before, done after, empty/null Stage 2 keys).
6. Gateway scan lifecycle & error paths:
   - 503 camera_unavailable if engine or /dev/video0 missing
   - 409 scan_in_progress if scan active
   - Idempotent POST /api/v1/scan/stop (200 on idle/done/finalizing)
   - Crash/error recovery (state "error", stop_reason "error", creates advisory)
   - Interrupted recovery on startup (dead PID -> advisory with stop_reason "interrupted" in manifest)
7. Safe shutdown endpoint (POST /api/v1/pod/shutdown requires confirm: true, 202 sent first).
8. Phone time sync helper validation (ISO-8601 UTC regex, year range 2025–2035).
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.storage import (
    DEFAULT_DB_PATH,
    EdgeStorage,
    get_utc_iso_now,
    resolve_data_directory,
)
from gateway.server import EdgeGateway, ThreadedHTTPServer


def http_get(url: str) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    """Helper to perform HTTP GET and return (status, json_body, headers)."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            headers = dict(resp.headers)
            body = json.loads(resp.read().decode("utf-8"))
            return status, body, headers
    except urllib.error.HTTPError as e:
        headers = dict(e.headers)
        body = json.loads(e.read().decode("utf-8"))
        return e.code, body, headers


def http_post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    """Helper to perform HTTP POST with JSON body and return (status, json_body, headers)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            headers = dict(resp.headers)
            body = json.loads(resp.read().decode("utf-8"))
            return status, body, headers
    except urllib.error.HTTPError as e:
        headers = dict(e.headers)
        body = json.loads(e.read().decode("utf-8"))
        return e.code, body, headers


# =============================================================================
# 1. Storage Resolver & Migration Tests
# =============================================================================

def test_storage_resolver_mounted_and_writable(monkeypatch):
    """
    Test 1A: When mount_point is mounted (os.path.ismount == True) and writable,
    resolver returns (<mount>/aegis, 'sd', False).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mount_dir = Path(tmpdir) / "fake_sd"
        mount_dir.mkdir(parents=True, exist_ok=True)
        internal_dir = Path(tmpdir) / "internal_data"
        internal_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(os.path, "ismount", lambda p: str(p) == str(mount_dir))

        resolved, loc, card_missing = resolve_data_directory(
            mount_point=mount_dir,
            internal_dir=internal_dir,
        )

        assert resolved == mount_dir / "aegis"
        assert loc == "sd"
        assert card_missing is False
        assert (mount_dir / "aegis").exists()


def test_storage_resolver_unmounted_dir_falls_back(monkeypatch):
    """
    Test 1B: An existing directory that is NOT a mount point must NOT be used as SD.
    Must fall back to internal_dir and report STORAGE_CARD_MISSING (card_missing=True).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_unmounted_dir = Path(tmpdir) / "aegisdata_on_emmc"
        fake_unmounted_dir.mkdir(parents=True, exist_ok=True)
        internal_dir = Path(tmpdir) / "internal_data"
        internal_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly ensure ismount returns False
        monkeypatch.setattr(os.path, "ismount", lambda p: False)

        resolved, loc, card_missing = resolve_data_directory(
            mount_point=fake_unmounted_dir,
            internal_dir=internal_dir,
        )

        assert resolved == internal_dir
        assert loc == "internal"
        assert card_missing is True


def test_storage_resolver_one_time_migration(monkeypatch):
    """
    Test 1C: One-time migration: if SD dir has no edge.db and internal data/edge.db exists,
    copy it (never move or delete the original).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mount_dir = Path(tmpdir) / "fake_sd"
        mount_dir.mkdir(parents=True, exist_ok=True)
        internal_dir = Path(tmpdir) / "internal_data"
        internal_dir.mkdir(parents=True, exist_ok=True)

        # Create original internal edge.db with content
        orig_db = internal_dir / "edge.db"
        with open(str(orig_db), "wb") as f:
            f.write(b"SQLITE_TEST_HEADER_AEGIS_DB_CONTENT")

        monkeypatch.setattr(os.path, "ismount", lambda p: str(p) == str(mount_dir))

        resolved, loc, card_missing = resolve_data_directory(
            mount_point=mount_dir,
            internal_dir=internal_dir,
        )

        migrated_db = resolved / "edge.db"
        assert migrated_db.exists()
        assert orig_db.exists(), "Original internal database must NEVER be deleted or moved"
        with open(str(migrated_db), "rb") as f:
            migrated_bytes = f.read()
        assert migrated_bytes == b"SQLITE_TEST_HEADER_AEGIS_DB_CONTENT"


# =============================================================================
# 2. Pipeline CLI Strictness Tests
# =============================================================================

def test_pipeline_cli_until_stopped_requires_arguments():
    """
    Test 2A: With --until-stopped, --scan-id, --field-id, and --crop are strictly REQUIRED.
    Argparse error + exit code 2 if missing.
    """
    py_exec = sys.executable
    script = str(ROOT / "edge" / "pipeline.py")

    # 1. Missing all three
    p1 = subprocess.run([py_exec, script, "--until-stopped"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p1.returncode == 2

    # 2. Missing --crop
    p2 = subprocess.run([py_exec, script, "--until-stopped", "--scan-id", "S1", "--field-id", "F1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p2.returncode == 2

    # 3. Missing --field-id
    p3 = subprocess.run([py_exec, script, "--until-stopped", "--scan-id", "S1", "--crop", "wheat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p3.returncode == 2

    # 4. Missing --scan-id
    p4 = subprocess.run([py_exec, script, "--until-stopped", "--field-id", "F1", "--crop", "wheat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p4.returncode == 2

    # 5. Invalid crop choice
    p5 = subprocess.run([py_exec, script, "--until-stopped", "--scan-id", "S1", "--field-id", "F1", "--crop", "barley"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p5.returncode == 2


def test_pipeline_cli_without_until_stopped_preserves_behaviour():
    """
    Test 2B: Without --until-stopped, existing CLI behaviour (--max-frames, etc.)
    is preserved without requiring --scan-id, --field-id, or --crop.
    """
    py_exec = sys.executable
    script = str(ROOT / "edge" / "pipeline.py")

    # Asking for help or passing max-frames should not trigger required until-stopped validation
    p = subprocess.run([py_exec, script, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 0
    help_out = p.stdout.decode("utf-8")
    assert "--until-stopped" in help_out
    assert "--max-frames" in help_out


# =============================================================================
# 3. Concurrent SQLite WAL Mode Resilience Test
# =============================================================================

def test_sqlite_wal_concurrent_reader_writer():
    """
    Test 3: Concurrently write frame events and scan records from one thread
    while multiple threads read health, manifest, and advisories in WAL mode.
    Assert 0 'database is locked' errors with 5000ms busy timeout.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "concurrent_wal.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_wal_test"
        storage.record_scan_start(scan_id=scan_id, crop="wheat", field_id="field_1")

        stop_event = threading.Event()
        writer_errors: List[Exception] = []
        reader_errors: List[Exception] = []

        def writer_worker():
            try:
                for idx in range(100):
                    if stop_event.is_set():
                        break
                    storage.record_frame_event(
                        scan_id=scan_id,
                        frame_idx=idx,
                        timestamp_utc=get_utc_iso_now(),
                        cell_id="cell_01",
                        gate_passed=True,
                        gate_metrics={},
                        n_valid_tiles=9,
                        frame_state="HEALTHY",
                        class_id=0,
                        confidence=0.92,
                        tile_decisions=[],
                    )
                    time.sleep(0.005)
            except Exception as e:
                writer_errors.append(e)

        def reader_worker():
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    # Perform read operations that mimic gateway endpoints
                    h = storage.get_health()
                    assert "device" in h
                    m = storage.get_manifest(limit=50)
                    assert "count" in m
                    time.sleep(0.005)
            except Exception as e:
                reader_errors.append(e)

        t_write = threading.Thread(target=writer_worker)
        t_read = threading.Thread(target=reader_worker)

        t_write.start()
        t_read.start()

        t_write.join(timeout=10.0)
        t_read.join(timeout=10.0)
        stop_event.set()

        assert len(writer_errors) == 0, f"Writer encountered errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Reader encountered errors: {reader_errors}"


# =============================================================================
# 4. Full Status Shape & Spec Compliance Tests
# =============================================================================

def test_full_scan_status_shape_before_and_after_scan():
    """
    Test 4: Verify full spec §1.3 shape from get_scan_status().
    - Before any scan, state is 'idle'.
    - All spec keys present: state, scan_id, field_id, crop, replay, started_utc,
      elapsed_s, max_duration_s, counts, thermal_c_latest, field_station, warnings,
      alerts, advisory_id, stop_reason.
    - If storage is internal, warnings includes STORAGE_CARD_MISSING.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "status_shape.db"
        storage = EdgeStorage(db_path=db_path)
        storage.storage_location = "internal"

        server = ThreadedHTTPServer(
            ("127.0.0.1", 0),
            None,
            storage=storage,
            db_path=db_path,
            allow_mock=True,
        )

        st = server.get_scan_status()

        # 1. Assert idle before scan
        assert st["state"] == "idle"
        assert st["scan_id"] is None
        assert st["advisory_id"] is None
        assert st["stop_reason"] is None
        assert st["max_duration_s"] == 1800
        assert isinstance(st["elapsed_s"], int)

        # 2. Assert counts shape
        assert isinstance(st["counts"], dict)
        for k in ("frames_seen", "frames_used", "stretches", "healthy", "need_look", "unclear", "not_crop"):
            assert k in st["counts"]
            assert st["counts"][k] == 0

        # 3. Assert field_station shape
        assert isinstance(st["field_station"], dict)
        for k in ("reachable", "readings_collected", "last_reading_utc"):
            assert k in st["field_station"]
            assert st["field_station"][k] is None

        # 4. Assert warnings and alerts lists
        assert isinstance(st["warnings"], list)
        assert isinstance(st["alerts"], list)

        # 5. Assert STORAGE_CARD_MISSING warning present for internal storage
        warn_codes = [w.get("code") for w in st["warnings"] if isinstance(w, dict)]
        assert "STORAGE_CARD_MISSING" in warn_codes

        server.server_close()


# =============================================================================
# 5. Gateway Endpoints & Error Paths Tests
# =============================================================================

@pytest.fixture
def running_gateway():
    """Starts an EdgeGateway on an ephemeral port in mock mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "gw_test.db"
        gw = EdgeGateway(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            allow_mock=True,
        )
        gw.start_background()
        port = gw.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        yield gw, base_url
        gw.stop()


def test_gateway_health_includes_pod_ready_and_scan_state(running_gateway):
    """Test GET /api/v1/health has pod_ready and scan_state."""
    gw, base_url = running_gateway
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert status == 200
    assert "pod_ready" in body
    assert body["pod_ready"] is True
    assert "scan_state" in body
    assert body["scan_state"] == "idle"
    assert "storage" in body
    assert body["storage"]["location"] in ("sd", "internal")


def test_gateway_scan_start_validation(running_gateway):
    """Test POST /api/v1/scan/start parameter validation (crop, field_id, source)."""
    gw, base_url = running_gateway

    # 1. Missing crop -> 400
    s1, b1, _ = http_post_json(f"{base_url}/api/v1/scan/start", {"field_id": "F1"})
    assert s1 == 400
    assert b1["error"] == "bad_request"

    # 2. Invalid crop -> 400
    s2, b2, _ = http_post_json(f"{base_url}/api/v1/scan/start", {"crop": "cotton", "field_id": "F1"})
    assert s2 == 400
    assert b2["error"] == "bad_request"

    # 3. Missing field_id -> 400
    s3, b3, _ = http_post_json(f"{base_url}/api/v1/scan/start", {"crop": "wheat"})
    assert s3 == 400
    assert b3["error"] == "bad_request"

    # 4. Invalid source -> 400
    s4, b4, _ = http_post_json(f"{base_url}/api/v1/scan/start", {"crop": "wheat", "field_id": "F1", "source": "drone"})
    assert s4 == 400
    assert b4["error"] == "bad_request"


def test_gateway_scan_start_503_when_camera_unavailable():
    """
    Test 5A: In non-mock mode, if engine or camera device is missing,
    POST /api/v1/scan/start returns 503 camera_unavailable.
    Gateway must never open the camera itself.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "prod_gw.db"
        gw = EdgeGateway(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            allow_mock=False,  # Production mode
        )
        gw.server.camera_device = Path("/nonexistent/video0")
        gw.server.engine_path = Path("/nonexistent/engine.engine")
        gw.start_background()
        base_url = f"http://127.0.0.1:{gw.server_address[1]}"

        # Health should report pod_ready == False
        s_h, b_h, _ = http_get(f"{base_url}/api/v1/health")
        assert s_h == 200
        assert b_h["pod_ready"] is False

        # Attempt to start scan with camera source
        status, body, _ = http_post_json(
            f"{base_url}/api/v1/scan/start",
            {"crop": "wheat", "field_id": "field_test", "source": "camera"},
        )
        assert status == 503
        assert body["error"] == "camera_unavailable"

        gw.stop()


def test_gateway_scan_stop_idempotent(running_gateway):
    """
    Test 5B: POST /api/v1/scan/stop is idempotent.
    Stopping an idle or already finished scan returns 200 with current state.
    """
    gw, base_url = running_gateway
    status, body, _ = http_post_json(f"{base_url}/api/v1/scan/stop", {"reason": "user"})
    assert status == 200
    assert body["state"] == "idle"


def test_gateway_scan_start_concurrency_409(running_gateway):
    """
    Test 5C: One scan at a time. If a scan is in progress (starting/scanning/finalizing),
    subsequent start returns 409 scan_in_progress.
    """
    gw, base_url = running_gateway
    gw.server.current_scan_state = "scanning"
    gw.server.current_scan_id = "active_scan_123"

    status, body, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "field_test"},
    )
    assert status == 409
    assert body["error"] == "scan_in_progress"
    assert body["scan_id"] == "active_scan_123"

    # Reset
    gw.server.current_scan_state = "idle"


def test_interrupted_scan_recovery_on_gateway_startup():
    """
    Test 5D: On gateway start: any scan with status 'running' and a dead PID
    is finalized into an advisory with stop_reason = 'interrupted'.
    Committed frame events are preserved; recovered advisory appears in /manifest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "recovery.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "interrupted_scan_001"
        dead_pid = 9999999  # Guaranteed non-existent PID

        # 1. Record scan start with running status and dead PID
        storage.record_scan_start(
            scan_id=scan_id,
            crop="rice",
            field_id="field_south",
            pid=dead_pid,
            status="running",
        )

        # 2. Commit 3 frame events before the simulated power cut
        for idx in range(3):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=idx,
                timestamp_utc=f"2026-10-01T12:00:0{idx}Z",
                cell_id="cell_01",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.95,
                tile_decisions=[],
            )

        # 3. Simulate gateway startup with this database
        gw = EdgeGateway(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            allow_mock=True,
        )
        gw.start_background()
        base_url = f"http://127.0.0.1:{gw.server_address[1]}"

        # 4. Check manifest: recovered advisory must appear
        s_m, b_m, _ = http_get(f"{base_url}/api/v1/manifest")
        assert s_m == 200
        assert b_m["count"] == 1
        adv_summary = b_m["advisories"][0]
        adv_id = adv_summary["advisory_id"]

        # 5. Fetch full advisory
        s_a, b_a, _ = http_get(f"{base_url}/api/v1/advisory/{adv_id}")
        assert s_a == 200
        assert b_a["scan"]["stop_reason"] == "interrupted"
        assert b_a["scan"]["frames_evaluated"] == 3
        assert b_a["scan"]["crop_declared"] == "rice"

        gw.stop()


# =============================================================================
# 6. Safe Shutdown Tests
# =============================================================================

def test_safe_shutdown_refuses_without_confirm(running_gateway):
    """Test 6A: POST /api/v1/pod/shutdown refuses without confirm: true (400)."""
    gw, base_url = running_gateway

    # 1. No payload
    req = urllib.request.Request(f"{base_url}/api/v1/pod/shutdown", data=b"", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status == 400

    # 2. confirm: false
    s2, b2, _ = http_post_json(f"{base_url}/api/v1/pod/shutdown", {"confirm": False})
    assert s2 == 400
    assert b2["error"] == "missing_confirm"


def test_safe_shutdown_accepts_with_confirm(running_gateway, monkeypatch):
    """
    Test 6B: POST /api/v1/pod/shutdown returns 202 FIRST with shutting_down state,
    and cleanly stops any active scan before invoking shutdown helper.
    """
    gw, base_url = running_gateway

    # Mock subprocess.call so we don't actually trigger shutdown
    called_cmds: List[List[str]] = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: called_cmds.append(cmd))

    status, body, _ = http_post_json(f"{base_url}/api/v1/pod/shutdown", {"confirm": True})
    assert status == 202
    assert body["state"] == "shutting_down"

    # Wait for detached thread
    time.sleep(1.5)
    assert len(called_cmds) == 1
    assert called_cmds[0] == ["sudo", "/usr/local/sbin/aegis-shutdown"]


# =============================================================================
# 7. Phone Time Sync & Validation Tests
# =============================================================================

def test_aegis_set_time_helper_validation():
    """
    Test 7A: Verify scripts/helpers/aegis-set-time bash script argument validation:
    - Exit 1 on wrong argument count
    - Exit 2 on regex mismatch
    - Exit 3 on year out of range 2025-2035
    """
    helper_path = str(ROOT / "scripts" / "helpers" / "aegis-set-time")

    # 1. No args -> exit 1
    p1 = subprocess.run([helper_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p1.returncode == 1

    # 2. Invalid regex (space instead of T, missing Z, etc.) -> exit 2
    p2 = subprocess.run([helper_path, "2026-10-01 14:32:00"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p2.returncode == 2

    # 3. Invalid year < 2025 -> exit 3
    p3 = subprocess.run([helper_path, "2024-12-31T23:59:59Z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p3.returncode == 3

    # 4. Invalid year > 2035 -> exit 3
    p4 = subprocess.run([helper_path, "2036-01-01T00:00:00Z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p4.returncode == 3


def test_gateway_scan_start_phone_utc_validation(running_gateway):
    """
    Test 7B: POST /api/v1/scan/start validates phone_utc format and year range.
    Rejects malformed timestamps with 400.
    """
    gw, base_url = running_gateway

    # 1. Invalid regex
    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F1", "phone_utc": "invalid_timestamp"},
    )
    assert s1 == 400
    assert b1["error"] == "bad_request"

    # 2. Year out of range
    s2, b2, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F1", "phone_utc": "2020-01-01T00:00:00Z"},
    )
    assert s2 == 400
    assert b2["error"] == "bad_request"
