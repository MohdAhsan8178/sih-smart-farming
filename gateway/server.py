#!/usr/bin/env python3
"""
gateway/server.py — Offline HTTP API Gateway for Handheld Nano Pod (Steps 30–32).

Serves the offline mobile client (farmer's smartphone) over local WiFi AP (SIH-FIELD)
according to the wire contract in ans_for_vitthal.md (§2, §3, §8).

Endpoints:
  GET  /api/v1/health              -> Device liveness, unacked count, clock validity, sync seam
  GET  /api/v1/manifest?since=&limit= -> Paginated advisory catalog ordered strictly by monotonic seq
  GET  /api/v1/advisory/<id>       -> Complete v1.0 advisory document
  POST /api/v1/ack                 -> Non-blocking phone cursor advancement
  GET  /api/v1/media/<id>          -> 410 Gone (media retention pruned per §A5/§A10)

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
    Multi-threaded HTTP server handling each request in a separate thread.
    Allows concurrent reads while edge/pipeline.py writes to the WAL-mode database.
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass,
        storage: Optional[EdgeStorage] = None,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
    ):
        super(ThreadedHTTPServer, self).__init__(server_address, RequestHandlerClass)
        self.db_path = Path(db_path)
        self.storage = storage if storage is not None else EdgeStorage(db_path=self.db_path)
        self.is_ready = True
        # Seam for Subsystem 7 AP/STA mode-switching (AR9271 WiFi)
        self.is_syncing = False
        self.sync_state = "IDLE"

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


class GatewayRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request handler implementing ans_for_vitthal.md wire contract (§2, §3, §8).
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
                health["syncing"] = getattr(self.server, "is_syncing", False)
                health["sync_state"] = getattr(self.server, "sync_state", "IDLE")
                self._send_json_response(200, health)
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
                self._send_json_response(200, manifest)
                return

            # Route: GET /api/v1/advisory/<id_or_seq> (§A5, §A8, §8)
            advisory_match = re.match(r"^/api/v1/advisory/([^/]+)$", path)
            if advisory_match:
                advisory_id = advisory_match.group(1)
                advisory = self.server.storage.get_advisory(advisory_id)
                if advisory is None:
                    self._send_json_response(404, {"error": "not_found", "advisory_id": advisory_id})
                else:
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
    ):
        self.host = host
        self.port = int(port)
        self.db_path = Path(db_path)
        self.storage = storage if storage is not None else EdgeStorage(db_path=self.db_path)
        self.verbose = verbose

        self.server = ThreadedHTTPServer(
            (self.host, self.port),
            GatewayRequestHandler,
            storage=self.storage,
            db_path=self.db_path,
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
    args = parser.parse_args()

    gateway = EdgeGateway(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        verbose=args.verbose,
    )

    print("=" * 70)
    print("HANDHELD NANO POD OFFLINE HTTP API GATEWAY (Steps 30–32)")
    print("=" * 70)
    print("Binding Address     : http://%s:%d" % (args.host, args.port))
    print("Storage Database    : %s" % args.db_path)
    print("Wire Contract       : ans_for_vitthal.md (§2, §3, §8)")
    print("Endpoints           : /api/v1/health, /api/v1/manifest, /api/v1/advisory/<id>, /api/v1/ack")
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
