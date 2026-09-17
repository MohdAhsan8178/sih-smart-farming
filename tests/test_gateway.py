#!/usr/bin/env python3
"""
tests/test_gateway.py — Unit tests for Offline HTTP API Gateway (Steps 30–32).

Covers:
1. GET /api/v1/health (liveness, degraded GPS time, Subsystem 7 sync seam).
2. GET /api/v1/manifest (cursor by seq, cursor by advisory_id string, limit, truncation).
3. Error handling: 400 bad_request on invalid query params, 404 not_found on unknown advisory.
4. GET /api/v1/advisory/<id> (retrieval by advisory_id string and by integer seq).
5. GET /api/v1/media/<id> (410 gone retention pruned per §A5/§A10).
6. POST /api/v1/ack (courtesy ack, cursor advance, 400 on malformed JSON).
7. 503 boot window (Retry-After: 5 header and not_ready payload).
8. AP/STA mode-switch seam status toggling.
"""

import json
from pathlib import Path
import tempfile
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Dict, Tuple
import urllib.error
import urllib.request

import pytest

from edge.storage import EdgeStorage
from gateway.server import EdgeGateway


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


@pytest.fixture
def test_gateway():
    """Starts an EdgeGateway bound to 127.0.0.1 on an ephemeral port with an isolated temporary DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "gateway_test.db"
        storage = EdgeStorage(db_path=db_path)

        # Pre-seed 3 scans and advisories
        for i in range(1, 4):
            scan_id = f"scan_{i:03d}"
            storage.record_scan_start(scan_id)
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=0,
                timestamp_utc=f"2026-09-16T12:0{i}:00Z",
                cell_id="cell_test",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.95,
                tile_decisions=[],
            )
            storage.record_scan_end(scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)
            storage.create_advisory(scan_id=scan_id, advisory_id=f"adv_{i:03d}", replay=True)

        # Bind to localhost port 0 (ephemeral OS-assigned port)
        gateway = EdgeGateway(host="127.0.0.1", port=0, storage=storage, db_path=db_path)
        gateway.start_background()

        actual_port = gateway.server.server_address[1]
        base_url = f"http://127.0.0.1:{actual_port}"

        yield gateway, storage, base_url

        gateway.stop()


def test_health_endpoint(test_gateway):
    gateway, storage, base_url = test_gateway
    status, body, headers = http_get(f"{base_url}/api/v1/health")

    assert status == 200
    assert headers.get("Content-Type") == "application/json; charset=utf-8"
    assert body["device"] == "sih-pod-01"
    assert body["schema_version"] == "1.0"
    assert body["latest_seq"] == 3
    assert body["advisory_count"] == 3  # 3 unacked
    assert body["gps_time_valid"] is False  # Degraded state per current hardware
    assert body["clock_source"] == "filesystem"
    assert body["syncing"] is False
    assert body["sync_state"] == "IDLE"


def test_manifest_endpoint_pagination_and_cursors(test_gateway):
    gateway, storage, base_url = test_gateway

    # Page 1: limit=2 from since=0 (expect truncated=True)
    status, body, _ = http_get(f"{base_url}/api/v1/manifest?since=0&limit=2")
    assert status == 200
    assert body["schema_version"] == "1.0"
    assert len(body["advisories"]) == 2
    assert body["truncated"] is True
    assert body["advisories"][0]["seq"] == 1
    assert body["advisories"][1]["seq"] == 2

    # Page 2: since=2 (monotonic seq cursor)
    status, body, _ = http_get(f"{base_url}/api/v1/manifest?since=2&limit=2")
    assert status == 200
    assert len(body["advisories"]) == 1
    assert body["truncated"] is False
    assert body["advisories"][0]["seq"] == 3

    # Cursor compatibility: since=<advisory_id string> (§A3)
    status, body, _ = http_get(f"{base_url}/api/v1/manifest?since=adv_002&limit=10")
    assert status == 200
    assert len(body["advisories"]) == 1
    assert body["advisories"][0]["seq"] == 3


def test_manifest_bad_request_validation(test_gateway):
    gateway, storage, base_url = test_gateway

    # Invalid limit string
    status, body, _ = http_get(f"{base_url}/api/v1/manifest?limit=abc")
    assert status == 400
    assert body["error"] == "bad_request"

    # Invalid limit < 1
    status, body, _ = http_get(f"{base_url}/api/v1/manifest?limit=0")
    assert status == 400
    assert body["error"] == "bad_request"


def test_advisory_retrieval_by_id_and_seq(test_gateway):
    gateway, storage, base_url = test_gateway

    # Query by string id
    status, body, _ = http_get(f"{base_url}/api/v1/advisory/adv_002")
    assert status == 200
    assert body["schema_version"] == "1.0"
    assert body["advisory_id"] == "adv_002"
    assert body["seq"] == 2
    assert body["replay"] is True

    # Query by integer seq
    status, body, _ = http_get(f"{base_url}/api/v1/advisory/3")
    assert status == 200
    assert body["advisory_id"] == "adv_003"
    assert body["seq"] == 3


def test_advisory_not_found(test_gateway):
    gateway, storage, base_url = test_gateway
    status, body, _ = http_get(f"{base_url}/api/v1/advisory/unknown_advisory_999")
    assert status == 404
    assert body["error"] == "not_found"
    assert body["advisory_id"] == "unknown_advisory_999"


def test_media_endpoint_410_gone(test_gateway):
    gateway, storage, base_url = test_gateway
    status, body, _ = http_get(f"{base_url}/api/v1/media/1-crop-1")
    assert status == 410
    assert body["error"] == "gone"
    assert body["reason"] == "retention_pruned"


def test_ack_endpoint_and_health_count_advancement(test_gateway):
    gateway, storage, base_url = test_gateway

    # Initially 3 unacked
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert body["advisory_count"] == 3

    # Ack advisory adv_002
    status, body, _ = http_post_json(f"{base_url}/api/v1/ack", {"advisory_id": "adv_002"})
    assert status == 200
    assert body["status"] == "ok"
    assert body["acked"] == "adv_002"

    # Health unacked count now decremented to 2
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert body["advisory_count"] == 2

    # Courtesy ack of nonexistent ID (must succeed harmlessly per §A6)
    status, body, _ = http_post_json(f"{base_url}/api/v1/ack", {"upto": "adv_nonexistent"})
    assert status == 200
    assert body["status"] == "ok"

    # Malformed payload (missing upto or advisory_id)
    status, body, _ = http_post_json(f"{base_url}/api/v1/ack", {"wrong_key": 123})
    assert status == 400
    assert body["error"] == "bad_request"


def test_503_boot_window_with_retry_after(test_gateway):
    gateway, storage, base_url = test_gateway

    # Simulate boot window where database/AP is initializing
    gateway.set_ready(False)

    status, body, headers = http_get(f"{base_url}/api/v1/health")
    assert status == 503
    assert headers.get("Retry-After") == "5"
    assert body["error"] == "not_ready"

    # Restore ready state
    gateway.set_ready(True)
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert status == 200


def test_ap_sta_mode_switch_seam(test_gateway):
    gateway, storage, base_url = test_gateway

    # Normal AP hosting state
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert body["syncing"] is False
    assert body["sync_state"] == "IDLE"

    # Background radio switching to STA mode to pull mast data
    gateway.set_sync_state(syncing=True, state_name="MAST_SYNC")
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert body["syncing"] is True
    assert body["sync_state"] == "MAST_SYNC"

    # Switch back to AP hosting
    gateway.set_sync_state(syncing=False, state_name="AP_HOSTING")
    status, body, _ = http_get(f"{base_url}/api/v1/health")
    assert body["syncing"] is False
    assert body["sync_state"] == "AP_HOSTING"
