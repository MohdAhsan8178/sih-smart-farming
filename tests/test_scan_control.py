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
        gw.server.pipeline_script = str(ROOT / "tests" / "fixtures" / "fake_pipeline.py")
        gw.start_background()
        port = gw.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        try:
            yield gw, base_url
        finally:
            if getattr(gw.server, "current_scan_proc", None) is not None:
                gw.server.stop_scan()
                for _ in range(30):
                    if getattr(gw.server, "current_scan_proc", None) is None:
                        break
                    time.sleep(0.1)
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


# =============================================================================
# 5. Gateway Endpoints & Process Monitoring Tests (a through j)
# =============================================================================

def test_a_start_scan_spawns_process_and_creates_log(running_gateway):
    """a. start_scan spawns the process, state is 'starting', log file is created in <data_dir>/logs/<scan_id>.log."""
    gw, base_url = running_gateway
    status, body, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert status == 202
    assert body["state"] == "starting"
    scan_id = body["scan_id"]

    log_file = gw.server.storage.data_dir / "logs" / f"{scan_id}.log"
    assert log_file.exists()

    # Clean up
    gw.server.stop_scan()
    for _ in range(30):
        time.sleep(0.1)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] in ("done", "idle", "error"):
            break


def test_b_status_heartbeat_advances_and_state_becomes_scanning(running_gateway):
    """b. status heartbeat advances elapsed_s, frames_seen, frames_used; state becomes 'scanning'."""
    gw, base_url = running_gateway
    status, body, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert status == 202

    reached_scanning = False
    for _ in range(30):
        time.sleep(0.2)
        s, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        assert s == 200
        if b["state"] == "scanning":
            assert b["state"] == "scanning"
            assert b["counts"]["frames_seen"] > 0
            assert b["counts"]["frames_used"] > 0
            reached_scanning = True
            break
    assert reached_scanning is True

    # Clean up
    gw.server.stop_scan()
    for _ in range(30):
        time.sleep(0.1)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] in ("done", "idle", "error"):
            break


def test_c_stop_scan_sends_sigterm_finalizes_to_done(running_gateway):
    """c. stop_scan while scanning sends SIGTERM, state is 'finalizing' -> 'done' with advisory_id, stop_reason is 'user'."""
    gw, base_url = running_gateway
    status, body, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert status == 202

    # Wait for scanning state
    for _ in range(30):
        time.sleep(0.2)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] == "scanning":
            break

    # Stop scan
    s_stop, b_stop, _ = http_post_json(f"{base_url}/api/v1/scan/stop", {"reason": "client_ignored"})
    assert s_stop == 202
    assert b_stop["state"] == "finalizing"

    # Wait for done state
    reached_done = False
    for _ in range(30):
        time.sleep(0.2)
        s, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        assert s == 200
        if b["state"] == "done":
            assert b["state"] == "done"
            assert b["stop_reason"] == "user"
            assert b["advisory_id"] is not None
            reached_done = True
            break
    assert reached_done is True


def test_d_stop_scan_on_already_stopped_returns_200(running_gateway):
    """d. stop_scan on already-stopped returns 200 and changes nothing."""
    gw, base_url = running_gateway
    # 1. Stop on idle
    s1, b1, _ = http_post_json(f"{base_url}/api/v1/scan/stop", {})
    assert s1 == 200
    assert b1["state"] == "idle"

    # 2. Run scan to done
    http_post_json(f"{base_url}/api/v1/scan/start", {"crop": "wheat", "field_id": "F01", "source": "replay"})
    time.sleep(0.5)
    http_post_json(f"{base_url}/api/v1/scan/stop", {})
    for _ in range(30):
        time.sleep(0.2)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] == "done":
            break

    # 3. Stop on already done
    s2, b2, _ = http_post_json(f"{base_url}/api/v1/scan/stop", {})
    assert s2 == 200
    assert b2["state"] == "done"
    assert b2["stop_reason"] == "user"


def test_e_start_scan_while_already_running_returns_409(running_gateway):
    """e. start_scan while already running returns 409 conflict."""
    gw, base_url = running_gateway
    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s1 == 202
    scan_id = b1["scan_id"]

    s2, b2, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "rice", "field_id": "F02", "source": "replay"},
    )
    assert s2 == 409
    assert b2["error"] == "scan_in_progress"
    assert b2["scan_id"] == scan_id

    # Clean up
    gw.server.stop_scan()
    for _ in range(30):
        time.sleep(0.1)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] in ("done", "idle", "error"):
            break


def test_f_pipeline_exit1_state_error_and_preserves_frames(running_gateway, monkeypatch):
    """f. pipeline exit 1 -> state becomes 'error', stop_reason 'error', any saved frames preserved in an advisory."""
    gw, base_url = running_gateway
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "exit1")

    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s1 == 202
    scan_id = b1["scan_id"]

    # Commit frame event to SQLite for this scan before process exits
    gw.server.storage.record_frame_event(
        scan_id=scan_id,
        frame_idx=0,
        timestamp_utc="2026-10-01T12:00:00Z",
        cell_id="cell_01",
        gate_passed=True,
        gate_metrics={},
        n_valid_tiles=9,
        frame_state="HEALTHY",
        class_id=0,
        confidence=0.95,
        tile_decisions=[],
    )

    # Monitor should detect exit 1 and transition to error
    reached_error = False
    for _ in range(30):
        time.sleep(0.2)
        s, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        assert s == 200
        if b["state"] == "error":
            assert b["state"] == "error"
            assert b["stop_reason"] == "error"
            assert b["advisory_id"] is not None
            reached_error = True
            break
    assert reached_error is True

    # Verify advisory preserved the frame event
    adv_id = b["advisory_id"]
    s_a, b_a, _ = http_get(f"{base_url}/api/v1/advisory/{adv_id}")
    assert s_a == 200
    assert b_a["scan"]["stop_reason"] == "error"
    assert b_a["scan"]["frames_evaluated"] == 1


def test_g_pipeline_no_status_file_killed_after_startup_timeout(running_gateway, monkeypatch):
    """g. pipeline does not write status file within startup timeout -> killed, state 'error', stop_reason 'error'."""
    gw, base_url = running_gateway
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "no_status")
    gw.server.startup_timeout_s = 1.0  # Fast timeout for test

    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s1 == 202

    reached_error = False
    for _ in range(30):
        time.sleep(0.2)
        s, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        assert s == 200
        if b["state"] == "error":
            assert b["state"] == "error"
            assert b["stop_reason"] == "error"
            reached_error = True
            break
    assert reached_error is True


def test_h_pipeline_ignores_sigterm_killed_after_watchdog(running_gateway, monkeypatch):
    """h. pipeline ignores SIGTERM -> killed after watchdog timeout, stop_reason 'interrupted'."""
    gw, base_url = running_gateway
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "ignore_sigterm")
    gw.server.watchdog_timeout_s = 1.0  # Fast watchdog for test

    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s1 == 202
    scan_id = b1["scan_id"]

    time.sleep(0.5)
    s_stop, b_stop, _ = http_post_json(f"{base_url}/api/v1/scan/stop", {})
    assert s_stop == 202
    assert b_stop["state"] == "finalizing"

    # Wait for forced kill (1s timeout) + 3s more to verify stability
    time.sleep(4.0)

    s, b, _ = http_get(f"{base_url}/api/v1/scan/status")
    assert s == 200
    assert b["state"] == "done"
    assert b["stop_reason"] == "interrupted"
    assert b["advisory_id"] is not None

    # Verify number of advisories for that scan_id in SQLite == 1
    conn = gw.server.storage._get_connection()
    rows = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchall()
    assert len(rows) == 1


def test_i_max_duration_s_reached_stops_gracefully_with_time_limit():
    """i. max-duration-s reached -> pipeline stops capture, exits gracefully, stop_reason 'time_limit'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "timelimit.db"
        status_file = Path(tmpdir) / "status.json"
        script = str(ROOT / "tests" / "fixtures" / "fake_pipeline.py")

        cmd = [
            sys.executable,
            script,
            "--source", "test_video.mp4",
            "--until-stopped",
            "--scan-id", "scan_tl_01",
            "--field-id", "F01",
            "--crop", "wheat",
            "--status-file", str(status_file),
            "--db-path", str(db_path),
            "--max-duration-s", "1",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        assert res.returncode == 0

        # Verify status file
        assert status_file.exists()
        with open(str(status_file), "r", encoding="utf-8") as f:
            st = json.load(f)
        assert st["state"] == "done"
        assert st["stop_reason"] == "time_limit"

        # Verify advisory in database
        storage = EdgeStorage(db_path=db_path)
        m = storage.get_manifest()
        assert m["count"] == 1
        adv_id = m["advisories"][0]["advisory_id"]
        adv = storage.get_advisory(adv_id)
        assert adv["scan"]["stop_reason"] == "time_limit"


def test_j_crash_recovery_dead_pid_finalized_as_interrupted():
    """j. crash recovery: start gateway with a 'running' scan whose PID is dead -> finalized as 'interrupted'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "crash_recovery.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "crashed_scan_001"
        dead_pid = 9999999

        # Record scan start with running status and dead PID
        storage.record_scan_start(
            scan_id=scan_id,
            crop="wheat",
            field_id="F01",
            pid=dead_pid,
            status="running",
            replay=0,
        )

        # Commit 2 frame events before the crash
        for idx in range(2):
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

        # Start gateway with this database
        gw = EdgeGateway(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            allow_mock=True,
        )
        gw.start_background()
        base_url = f"http://127.0.0.1:{gw.server_address[1]}"

        # Before any new scan, state must be idle
        s_s, b_s, _ = http_get(f"{base_url}/api/v1/scan/status")
        assert s_s == 200
        assert b_s["state"] == "idle"

        # Check manifest: recovered advisory must appear
        s_m, b_m, _ = http_get(f"{base_url}/api/v1/manifest")
        assert s_m == 200
        assert b_m["count"] == 1
        adv_id = b_m["advisories"][0]["advisory_id"]

        # Fetch full advisory
        s_a, b_a, _ = http_get(f"{base_url}/api/v1/advisory/{adv_id}")
        assert s_a == 200
        assert b_a["scan"]["stop_reason"] == "interrupted"
        assert b_a["scan"]["frames_evaluated"] == 2
        assert b_a["replay"] is False

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
    and cleanly stops any active scan before invoking shutdown helper with sudo -n.
    """
    gw, base_url = running_gateway

    called_cmds: List[List[str]] = []
    class DummyCompletedProcess:
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: called_cmds.append(cmd) or DummyCompletedProcess())

    status, body, _ = http_post_json(f"{base_url}/api/v1/pod/shutdown", {"confirm": True})
    assert status == 202
    assert body["state"] == "shutting_down"

    # Wait for detached thread
    time.sleep(1.5)
    assert len(called_cmds) == 1
    assert called_cmds[0] == ["sudo", "-n", "/usr/local/sbin/aegis-shutdown"]


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


# =============================================================================
# 8. Regression & Seam Verification Tests (Tests 2, 3, 4, 5, 7)
# =============================================================================

def test_two_scans_back_to_back_final_status(running_gateway):
    """Test 2: Two scans back-to-back with fake pipeline: scan 2's final status has scan_id == scan 2's id, and advisory_id == scan 2's advisory."""
    gw, base_url = running_gateway

    # 1. Scan 1
    s1, b1, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s1 == 202
    scan_id_1 = b1["scan_id"]

    for _ in range(30):
        time.sleep(0.2)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] == "scanning":
            break

    http_post_json(f"{base_url}/api/v1/scan/stop", {})
    time.sleep(2.0)
    s_st1, b_st1, _ = http_get(f"{base_url}/api/v1/scan/status")
    assert s_st1 == 200
    assert b_st1["state"] == "done"
    assert b_st1["scan_id"] == scan_id_1
    adv_id_1 = b_st1["advisory_id"]
    assert adv_id_1 is not None

    # 2. Scan 2
    s2, b2, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "rice", "field_id": "F02", "source": "replay"},
    )
    assert s2 == 202
    scan_id_2 = b2["scan_id"]
    assert scan_id_2 != scan_id_1

    for _ in range(30):
        time.sleep(0.2)
        _, b, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b["state"] == "scanning":
            break

    http_post_json(f"{base_url}/api/v1/scan/stop", {})
    time.sleep(2.0)
    s_st2, b_st2, _ = http_get(f"{base_url}/api/v1/scan/status")
    assert s_st2 == 200
    assert b_st2["state"] == "done"
    assert b_st2["scan_id"] == scan_id_2
    adv_id_2 = b_st2["advisory_id"]
    assert adv_id_2 is not None
    assert adv_id_2 != adv_id_1


def test_time_limit_real_pipeline_all_rejected():
    """Test 3: Time limit with REAL pipeline: --dry-run --until-stopped --max-duration-s 3 on all-black video."""
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real_timelimit.db"
        status_file = Path(tmpdir) / "status.json"
        video_path = Path(tmpdir) / "all_black.mp4"

        # Generate a 60-frame all-black video (each frame will be rejected by frame gate)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
        for _ in range(60):
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(black_frame)
        out.release()

        cmd = [
            sys.executable,
            str(ROOT / "edge" / "pipeline.py"),
            "--source", str(video_path),
            "--dry-run",
            "--until-stopped",
            "--scan-id", "scan_real_tl_01",
            "--field-id", "F01",
            "--crop", "wheat",
            "--status-file", str(status_file),
            "--db-path", str(db_path),
            "--max-duration-s", "3",
            "--no-realtime",
        ]

        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        seen_elapsed_increase = False
        frames_used_stayed_zero = True

        while proc.poll() is None:
            time.sleep(0.3)
            if status_file.exists():
                try:
                    with open(str(status_file), "r", encoding="utf-8") as f:
                        st = json.load(f)
                    if st.get("elapsed_s", 0) > 0:
                        seen_elapsed_increase = True
                    if st.get("counts", {}).get("frames_used", 0) != 0:
                        frames_used_stayed_zero = False
                except Exception:
                    pass
            if time.monotonic() - t0 > 15.0:
                proc.kill()
                pytest.fail("Pipeline did not exit within 15 seconds")

        ret = proc.wait(timeout=5.0)
        assert ret == 0

        assert status_file.exists()
        with open(str(status_file), "r", encoding="utf-8") as f:
            st = json.load(f)
        assert st["state"] == "done"
        assert st["stop_reason"] == "time_limit"
        assert seen_elapsed_increase is True
        assert frames_used_stayed_zero is True


def test_old_cli_real_pipeline_defaults():
    """Test 4: Old CLI with REAL pipeline: --dry-run --max-frames N, no --until-stopped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "old_cli.db"
        video_path = str(ROOT / "test_video_from_dataset_images.mp4")

        cmd = [
            sys.executable,
            str(ROOT / "edge" / "pipeline.py"),
            "--source", video_path,
            "--dry-run",
            "--max-frames", "5",
            "--db-path", str(db_path),
            "--no-realtime",
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        assert res.returncode == 0

        storage = EdgeStorage(db_path=db_path)
        m = storage.get_manifest()
        assert m["count"] == 1
        adv_id = m["advisories"][0]["advisory_id"]
        assert adv_id.endswith("_F01")
        adv = storage.get_advisory(adv_id)
        assert adv["scan"]["mode"] == "handheld_pod"
        assert adv["scan"]["crop_declared"] is None
        assert adv["scan"]["stop_reason"] is None


def test_sudo_set_time_failure_retains_clock_source(running_gateway, monkeypatch):
    """Test 5: sudo failure for aegis-set-time (returncode 1) -> clock_source != 'phone', advisory time_source != 'phone'."""
    gw, base_url = running_gateway

    class MockFailedProcess:
        returncode = 1
        stdout = b""
        stderr = b"sudo: permission denied"

    orig_run = subprocess.run
    def mock_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 2 and "aegis-set-time" in cmd[2]:
            return MockFailedProcess()
        return orig_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    phone_time = "2026-10-01T15:00:00Z"
    s, b, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay", "phone_utc": phone_time},
    )
    assert s == 202

    s_h, b_h, _ = http_get(f"{base_url}/api/v1/health")
    assert s_h == 200
    assert b_h.get("clock_source") != "phone"

    http_post_json(f"{base_url}/api/v1/scan/stop", {})
    for _ in range(30):
        time.sleep(0.2)
        _, b_st, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b_st["state"] == "done":
            break

    assert b_st["state"] == "done"
    adv_id = b_st["advisory_id"]
    assert adv_id is not None
    s_a, b_a, _ = http_get(f"{base_url}/api/v1/advisory/{adv_id}")
    assert s_a == 200
    assert b_a["time_source"] != "phone"


def test_pod_ready_stays_true_after_scan_error(running_gateway, monkeypatch):
    """Test 7: pod_ready stays True after a scan ended in 'error' (mock mode)."""
    gw, base_url = running_gateway
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "exit1")

    s, b, _ = http_post_json(
        f"{base_url}/api/v1/scan/start",
        {"crop": "wheat", "field_id": "F01", "source": "replay"},
    )
    assert s == 202

    for _ in range(30):
        time.sleep(0.2)
        _, b_st, _ = http_get(f"{base_url}/api/v1/scan/status")
        if b_st["state"] == "error":
            break

    assert b_st["state"] == "error"

    s_h, b_h, _ = http_get(f"{base_url}/api/v1/health")
    assert s_h == 200
    assert b_h["scan_state"] == "error"
    assert b_h["pod_ready"] is True

