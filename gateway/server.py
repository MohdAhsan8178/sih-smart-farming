#!/usr/bin/env python3
"""
gateway/server.py — Offline HTTP API Gateway for Handheld Nano Pod (Steps 30–32).

Serves the offline mobile client (farmer's smartphone) over local WiFi AP (SIH-FIELD)
according to the wire contract in docs/PAYLOAD_CONTRACT.md.

Endpoints (9 total):
  GET  /api/v1/health                 -> Device liveness, advisory count, clock validity, sync state
  GET  /api/v1/sync/status            -> Ground mast telemetry synchronization status & metrics
  GET  /api/v1/manifest?since=&limit= -> Paginated advisory catalog ordered strictly by monotonic seq
  GET  /api/v1/advisory/<id_or_seq>   -> Complete v1.0 advisory document by UUID or sequence number
  GET  /api/v1/advisory/latest        -> Most recently synthesized advisory document
  GET  /api/v1/media/<id>             -> 410 Gone (media retention pruned per policy)
  POST /api/v1/ack                    -> Non-blocking phone cursor advancement
  POST /api/v1/trap/upload            -> Sticky trap card photo for Model B segmentation & classification
  POST /api/v1/sync/trigger           -> Triggers asynchronous ground mast collector pull

Key Architectural Properties:
1. Python 3.6 stdlib only (http.server + socketserver.ThreadingMixIn) — zero external dependencies.
2. Concurrent-safe with edge/pipeline.py writes via SQLite WAL mode and thread-isolated connections.
3. Binds to 192.168.4.1:8080 (plain HTTP, no TLS) by default.
4. Clean seam for Subsystem 7 AR9271 AP/STA mode-switching ("syncing" status flag).
5. 503 boot window support with Retry-After: 5 header during startup/initialization.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import socketserver
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union
import urllib.parse

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.storage import DEFAULT_DB_PATH, EdgeStorage, get_utc_iso_now

DEFAULT_GATEWAY_HOST = "192.168.4.1"
DEFAULT_GATEWAY_PORT = 8080



class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """
    Multi-threaded HTTP Server ensuring concurrent client connections are handled
    without blocking during disk I/O, SQLite operations, or Model B processing.
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: Any,
        storage: Optional[EdgeStorage] = None,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        allow_mock: bool = False,
    ):
        super(ThreadedHTTPServer, self).__init__(server_address, RequestHandlerClass)
        self.db_path = Path(db_path)
        self.storage = storage if storage is not None else EdgeStorage(db_path=self.db_path)
        self.is_ready = True
        self.allow_mock = bool(allow_mock)
        # Seam for Subsystem 7 AP/STA mode-switching (AR9271 WiFi) and sync management
        self.is_syncing = False
        self.sync_state = "IDLE"
        self.sync_in_progress = False
        self.last_sync_utc = None
        self.last_success_utc = None
        self.last_attempt_utc = None
        self.last_result = None  # "OK", "MAST_NOT_FOUND", "PARTIAL", "ERROR", "SKIPPED_CLIENT_CONNECTED"
        self.last_records_pulled = 0
        self.last_trap_images_pulled = 0
        self._sync_lock = threading.Lock()

    def set_sync_state(self, syncing: bool, state_name: str = "IDLE") -> None:
        """
        AP/STA mode-switch seam (PENDING_HARDWARE.md Subsystem 7).
        Allows a background radio sync worker (when AR9271 hardware arrives) to flag
        that the pod is currently in STA mode pulling data from the mast node, so that
        /api/v1/health exposes syncing=True to connected clients without modifying the server.
        """
        self.is_syncing = bool(syncing)
        self.sync_state = str(state_name)

    def set_ready(self, ready: bool) -> None:
        """Sets server boot readiness (controls 503 boot window)."""
        self.is_ready = bool(ready)

    def trigger_sync(self) -> bool:
        """
        Triggers an asynchronous mast synchronization pull cycle (M3.1, M3.3).
        Returns True if started, False if a sync is already running.
        """
        with self._sync_lock:
            if self.sync_in_progress:
                return False
            self.sync_in_progress = True
            self.last_attempt_utc = get_utc_iso_now()

        def _worker():
            try:
                from edge.mast_collector import MastCollector
                collector = MastCollector(storage=self.storage)
                res = collector.collect()
                if isinstance(res, dict) and res.get("status") == "ok":
                    self.last_result = "OK"
                    self.last_success_utc = get_utc_iso_now()
                    self.last_records_pulled = int(res.get("records_pulled", 0))
                    self.last_trap_images_pulled = int(res.get("traps_processed", 0))
                else:
                    self.last_result = res.get("result_code", "PARTIAL") if isinstance(res, dict) else "PARTIAL"
                    self.last_records_pulled = int(res.get("records_pulled", 0)) if isinstance(res, dict) else 0
                    self.last_trap_images_pulled = int(res.get("traps_processed", 0)) if isinstance(res, dict) else 0
            except Exception as ex:
                err_str = str(ex).lower()
                if "not found" in err_str or "unreachable" in err_str or "refused" in err_str or "timed out" in err_str or "no route" in err_str:
                    self.last_result = "MAST_NOT_FOUND"
                else:
                    self.last_result = "ERROR"
            finally:
                self.last_sync_utc = get_utc_iso_now()
                with self._sync_lock:
                    self.sync_in_progress = False

        th = threading.Thread(target=_worker, daemon=True, name="MastSyncWorker")
        th.start()
        return True


class GatewayRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request handler implementing docs/PAYLOAD_CONTRACT.md wire contract.
    """

    # Suppress default noisy per-request logging to stderr in quiet mode
    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super(GatewayRequestHandler, self).log_message(format, *args)

    def _send_json_response(self, status_code: int, data: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None) -> None:
        """Encodes and sends an application/json; charset=utf-8 HTTP response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status_code: int, error: str, detail: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> None:
        payload = {"error": error}
        if detail:
            payload["detail"] = detail
        self._send_json_response(status_code, payload, extra_headers=extra_headers)

    def do_GET(self) -> None:
        """Handles HTTP GET requests."""
        # 1. Check boot window readiness (§A5)
        if not getattr(self.server, "is_ready", True):
            self._send_error_json(503, "not_ready", extra_headers={"Retry-After": "5"})
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            # Route: GET /api/v1/health (§A9)
            if path == "/api/v1/health":
                health = self.server.storage.get_health()
                # Inject Subsystem 7 AP/STA mode-switch seam
                health["syncing"] = getattr(self.server, "is_syncing", False) or getattr(self.server, "sync_in_progress", False)
                health["sync_state"] = getattr(self.server, "sync_state", "IDLE")
                self._send_json_response(200, health)
                return

            # Route: GET /api/v1/sync/status (L7.3, M3.3)
            if path == "/api/v1/sync/status":
                conn = self.server.storage._get_connection()
                latest_mast = conn.execute("SELECT received_at FROM mast_telemetry ORDER BY id DESC LIMIT 1;").fetchone()
                mast_data_age_s = None
                if latest_mast and latest_mast["received_at"]:
                    try:
                        m_dt = _parse_iso_timestamp(latest_mast["received_at"])
                        if m_dt:
                            mast_data_age_s = int(abs((datetime.datetime.now(datetime.timezone.utc) - m_dt).total_seconds()))
                    except Exception:
                        pass

                status_payload = {
                    "sync_in_progress": getattr(self.server, "sync_in_progress", False),
                    "last_success_utc": getattr(self.server, "last_success_utc", None),
                    "last_attempt_utc": getattr(self.server, "last_attempt_utc", None),
                    "last_result": getattr(self.server, "last_result", None),
                    "mast_data_age_s": mast_data_age_s,
                    "records_pulled": getattr(self.server, "last_records_pulled", 0),
                    "trap_images_pulled": getattr(self.server, "last_trap_images_pulled", 0),
                }
                self._send_json_response(200, status_payload)
                return

            # Route: GET /api/v1/manifest (§A2, §A3, §A4)
            if path == "/api/v1/manifest":
                since_param = query.get("since", ["0"])[0]
                limit_param = query.get("limit", ["200"])[0]

                # Validate limit parameter
                try:
                    limit_val = int(limit_param)
                    if limit_val < 1:
                        self._send_error_json(400, "bad_request", "limit must be >= 1")
                        return
                except ValueError:
                    self._send_error_json(400, "bad_request", "limit must be an integer")
                    return

                # Validate since parameter (must be int or string advisory_id)
                since_val = since_param
                if since_param.isdigit():
                    since_val = int(since_param)

                manifest = self.server.storage.get_manifest(since=since_val, limit=limit_val)

                # L4.2: If allow_mock is False, omit mock advisories from manifest
                if not getattr(self.server, "allow_mock", False) and "advisories" in manifest:
                    manifest["advisories"] = [
                        a for a in manifest["advisories"]
                        if a.get("inference_backend") != "mock"
                    ]
                    manifest["count"] = len(manifest["advisories"])

                self._send_json_response(200, manifest)
                return

            # Route: GET /api/v1/advisory/<id_or_seq> (§A5, §A8, §8)
            advisory_match = re.match(r"^/api/v1/advisory/([^/]+)$", path)
            if advisory_match:
                advisory_id = urllib.parse.unquote(advisory_match.group(1))
                advisory = self.server.storage.get_advisory(advisory_id)
                if advisory is None:
                    self._send_json_response(404, {"error": "not_found", "advisory_id": advisory_id})
                else:
                    # L4.2: If allow_mock is False and advisory is mock, reject with 403 Forbidden
                    if not getattr(self.server, "allow_mock", False) and advisory.get("inference_backend") == "mock":
                        self._send_error_json(
                            403,
                            "mock_advisory_rejected",
                            "Gateway running in production mode rejecting mock advisory",
                        )
                        return
                    self._send_json_response(200, advisory)
                return

            # Route: GET /api/v1/media/<id> (§A5, §A10)
            media_match = re.match(r"^/api/v1/media/([^/]+)$", path)
            if media_match:
                # Per §A5/§A10: Media endpoint is out of demo critical path; retention pruned
                self._send_json_response(410, {"error": "gone", "reason": "retention_pruned"})
                return

            # Unmatched route
            self._send_json_response(404, {"error": "not_found", "path": path})

        except Exception as e:
            self._send_json_response(500, {"error": "internal", "detail": str(e)})

    def do_POST(self) -> None:
        """Handles HTTP POST requests."""
        # Check boot window readiness (§A5)
        if not getattr(self.server, "is_ready", True):
            self._send_error_json(503, "not_ready", extra_headers={"Retry-After": "5"})
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            # Route: POST /api/v1/ack (§A6, §D3)
            if path == "/api/v1/ack":
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    self._send_error_json(400, "bad_request", "Missing Content-Length header")
                    return

                try:
                    length = int(content_length)
                    raw_body = self.rfile.read(length)
                    body_json = json.loads(raw_body.decode("utf-8"))
                except Exception as ex:
                    self._send_error_json(400, "bad_request", f"Malformed JSON: {ex}")
                    return

                # Accepts {"upto": "<id>"}, {"advisory_id": "<id>"}, or {"seq": <int>}
                target_id = body_json.get("upto") or body_json.get("advisory_id") or body_json.get("seq")
                if not target_id:
                    self._send_error_json(400, "bad_request", "Missing 'upto' or 'advisory_id' field in payload")
                    return

                # Courtesy ack: advances last-acked cursor without deleting data
                self.server.storage.ack_advisory(str(target_id))
                self._send_json_response(200, {"status": "ok", "acked": target_id})
                return

            # Route: POST /api/v1/trap/upload (Step 31 / H5.2)
            if path == "/api/v1/trap/upload":
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    self._send_error_json(400, "bad_request", "Missing Content-Length header")
                    return

                try:
                    length = int(content_length)
                    raw_body = self.rfile.read(length)
                except Exception as ex:
                    self._send_error_json(400, "bad_request", "Failed to read body: %s" % ex)
                    return

                query = urllib.parse.parse_qs(parsed.query)

                # Metadata extraction from query params or HTTP headers
                trap_id = query.get("trap_id", [self.headers.get("X-Trap-Id", "TRAP_01")])[0]
                placed_at = query.get("placed_at", [self.headers.get("X-Placed-At", None)])[0]
                days_param = query.get("days", [self.headers.get("X-Days-Monitored", None)])[0]
                days_monitored = float(days_param) if days_param else None

                scale_param = query.get("scale", [self.headers.get("X-Scale", None)])[0]
                scale = float(scale_param) if scale_param else None

                allow_prov_param = query.get("allow_provisional", [self.headers.get("X-Allow-Provisional", "false")])[0]
                allow_provisional = str(allow_prov_param).lower() in ("true", "1", "yes")

                content_type = self.headers.get("Content-Type", "")
                image_bytes = raw_body

                # Multipart form parsing if uploaded via form
                if "multipart/form-data" in content_type:
                    boundary = None
                    for part in content_type.split(";"):
                        part = part.strip()
                        if part.startswith("boundary="):
                            boundary = part.split("=", 1)[1].strip().strip('"').encode("ascii")
                    if boundary:
                        parts = raw_body.split(b"--" + boundary)
                        for p in parts:
                            if b"Content-Disposition:" in p:
                                header_data, body_data = p.split(b"\r\n\r\n", 1)
                                header_str = header_data.decode("latin1", errors="replace")
                                body_data = body_data.rstrip(b"\r\n--")
                                if 'name="image"' in header_str or 'filename=' in header_str:
                                    image_bytes = body_data
                                elif 'name="trap_id"' in header_str:
                                    trap_id = body_data.decode("utf-8", errors="replace").strip()
                                elif 'name="placed_at"' in header_str:
                                    placed_at = body_data.decode("utf-8", errors="replace").strip()
                                elif 'name="days"' in header_str:
                                    days_monitored = float(body_data.decode("utf-8", errors="replace").strip())
                                elif 'name="scale"' in header_str:
                                    scale = float(body_data.decode("utf-8", errors="replace").strip())
                                elif 'name="allow_provisional"' in header_str:
                                    allow_provisional = body_data.decode("utf-8", errors="replace").strip().lower() in ("true", "1", "yes")

                # Persist incoming card to inbox
                inbox_dir = ROOT / "data" / "trap_inbox"
                inbox_dir.mkdir(parents=True, exist_ok=True)
                now_ts = int(time.time() * 1000)
                job_id = "trap_job_%s_%d" % (trap_id, now_ts)
                saved_path = inbox_dir / ("%s.jpg" % job_id)
                with open(str(saved_path), "wb") as f:
                    f.write(image_bytes)

                # Process trap card through Model B job
                from edge.trap_job import process_trap_image
                try:
                    job_res = process_trap_image(
                        image_path=saved_path,
                        scale=scale,
                        allow_provisional=allow_provisional,
                        trap_id=trap_id,
                        placed_at=placed_at,
                        days_monitored=days_monitored,
                        storage=self.server.storage,
                        job_id=job_id,
                    )
                except ValueError as scale_err:
                    self._send_error_json(400, "scale_uncalibrated", str(scale_err))
                    return
                except Exception as proc_err:
                    self._send_error_json(500, "processing_failure", str(proc_err))
                    return

                # Synthesize / update latest advisory with the new trap result
                conn = self.server.storage._get_connection()
                latest_scan = conn.execute("SELECT scan_id FROM scans ORDER BY started_utc DESC LIMIT 1;").fetchone()
                advisory_id = None
                if latest_scan:
                    scan_id = latest_scan["scan_id"]
                    try:
                        adv = self.server.storage.create_advisory(scan_id=scan_id)
                        advisory_id = adv["advisory_id"]
                    except Exception:
                        pass

                resp = {
                    "status": "ok",
                    "job_id": job_id,
                    "trap_id": trap_id,
                    "record_id": job_res["record_id"],
                    "advisory_id": advisory_id,
                    "total_blobs_counted": job_res["total_blobs_counted"],
                    "scale_status": job_res["scale_status"],
                    "scale_mm_per_pixel": job_res["scale_mm_per_pixel"],
                    "etl_status": job_res["etl_status"],
                    "pest": job_res["pest"],
                }
                self._send_json_response(200, resp)
                return

            # Route: POST /api/v1/sync/trigger (L7.3, M3.3)
            if path == "/api/v1/sync/trigger":
                started = self.server.trigger_sync()
                if not started:
                    self._send_error_json(
                        409,
                        "sync_in_progress",
                        "Mast synchronization is already running",
                    )
                    return
                self._send_json_response(202, {
                    "status": "accepted",
                    "timestamp": get_utc_iso_now(),
                    "expected_ap_downtime_s": 30,
                })
                return

            self._send_json_response(404, {"error": "not_found", "path": path})

        except Exception as e:
            self._send_json_response(500, {"error": "internal", "detail": str(e)})


class EdgeGateway(object):
    """
    Lifecycle manager for the HTTP API Gateway server on the Handheld Nano Pod.
    """

    def __init__(
        self,
        host: str = DEFAULT_GATEWAY_HOST,
        port: int = DEFAULT_GATEWAY_PORT,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        storage: Optional[EdgeStorage] = None,
        verbose: bool = False,
        allow_mock: bool = False,
    ):
        self.host = host
        self.port = int(port)
        self.db_path = Path(db_path)
        self.storage = storage if storage is not None else EdgeStorage(db_path=self.db_path)
        self.verbose = verbose
        self.allow_mock = bool(allow_mock)

        self.server = ThreadedHTTPServer(
            (self.host, self.port),
            GatewayRequestHandler,
            storage=self.storage,
            db_path=self.db_path,
            allow_mock=self.allow_mock,
        )
        self.server.verbose = self.verbose
        self._thread: Optional[threading.Thread] = None

    @property
    def server_address(self) -> Tuple[str, int]:
        return self.server.server_address

    def start_background(self) -> None:
        """Starts the gateway server in a background daemon thread."""
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="EdgeGatewayServer")
        self._thread.start()

    def stop(self) -> None:
        """Stops the gateway server and joins thread."""
        self.server.shutdown()
        self.server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def set_sync_state(self, syncing: bool, state_name: str = "IDLE") -> None:
        """Exposes AP/STA mode-switch status to /health (Subsystem 7 seam)."""
        self.server.set_sync_state(syncing, state_name)

    def set_ready(self, ready: bool) -> None:
        """Sets boot readiness flag (controls 503 boot window)."""
        self.server.set_ready(ready)


def main():
    parser = argparse.ArgumentParser(description="Offline HTTP Gateway for Handheld Nano Pod (Steps 30–32)")
    parser.add_argument("--host", type=str, default=DEFAULT_GATEWAY_HOST, help=f"Bind host (default: {DEFAULT_GATEWAY_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_GATEWAY_PORT, help=f"Bind port (default: {DEFAULT_GATEWAY_PORT})")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--verbose", action="store_true", help="Enable per-request HTTP access logging")
    parser.add_argument("--allow-mock", action="store_true", default=False, help="Allow serving mock advisories in non-production environments")
    args = parser.parse_args()

    gateway = EdgeGateway(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        verbose=args.verbose,
        allow_mock=args.allow_mock,
    )

    print("=" * 70)
    print("HANDHELD NANO POD OFFLINE HTTP API GATEWAY (Steps 30–32)")
    print("=" * 70)
    print("Binding Address     : http://%s:%d" % (args.host, args.port))
    print("Storage Database    : %s" % args.db_path)
    print("Wire Contract       : docs/PAYLOAD_CONTRACT.md (v1.0)")
    print("Endpoints (9 total) : GET  /api/v1/health")
    print("                      GET  /api/v1/sync/status")
    print("                      GET  /api/v1/manifest?since=&limit=")
    print("                      GET  /api/v1/advisory/<id_or_seq>")
    print("                      GET  /api/v1/advisory/latest")
    print("                      GET  /api/v1/media/<id>")
    print("                      POST /api/v1/ack")
    print("                      POST /api/v1/trap/upload")
    print("                      POST /api/v1/sync/trigger")
    print("AP/STA Seam Status  : IDLE (Subsystem 7 ready)")
    print("Press Ctrl+C to terminate.")
    print("=" * 70)

    try:
        gateway.server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down gateway server...")
        gateway.stop()


if __name__ == "__main__":
    main()
