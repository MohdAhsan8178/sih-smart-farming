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
3. Binds to 0.0.0.0:8080 (plain HTTP, no TLS) by default.
4. Clean seam for Subsystem 7 AR9271 AP/STA mode-switching ("syncing" status flag).
5. 503 boot window support with Retry-After: 5 header during startup/initialization.
"""

import argparse
import concurrent.futures
from http.server import BaseHTTPRequestHandler, HTTPServer
import datetime
import json
import os
from pathlib import Path
import re
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union
import urllib.parse
import urllib.request

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.storage import DEFAULT_DB_PATH, EdgeStorage, _parse_iso_timestamp, get_utc_iso_now

DEFAULT_GATEWAY_HOST = "0.0.0.0"
DEFAULT_GATEWAY_PORT = 8080


def _get_wlan_subnet_prefix():
    """Finds the local /24 subnet prefix for wlan0 or active network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3])
    except Exception:
        pass
    return "192.168.43"


def check_field_station_health(base_url, timeout=0.5):
    """Checks GET /api/v1/health on base_url and returns True if node_id is SIH-NODE-01."""
    if not base_url:
        return False
    try:
        url = "%s/health" % base_url.rstrip("/")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("node_id") == "SIH-NODE-01":
                    return True
    except Exception:
        pass
    return False


def probe_for_field_station(subnet_prefix=None, timeout_per_host=0.5, max_workers=32, total_timeout=8.0, candidate_hosts=None, probe_port=80):
    """Probes /24 subnet (or candidate hosts) in parallel for ESP32 Field Station node_id == 'SIH-NODE-01'."""
    if candidate_hosts is not None:
        hosts = list(candidate_hosts)
    else:
        prefix = subnet_prefix or _get_wlan_subnet_prefix()
        hosts = ["%s.%d" % (prefix, i) for i in range(1, 255)]

    def _probe_host(host_entry):
        if str(host_entry).startswith("http://") or str(host_entry).startswith("https://"):
            url = "%s/health" % str(host_entry).rstrip("/")
            if "/api/v1" not in url:
                url = "%s/api/v1/health" % str(host_entry).rstrip("/")
            ret_base = str(host_entry).rstrip("/")
            if not ret_base.endswith("/api/v1"):
                ret_base = "%s/api/v1" % ret_base
        elif ":" in str(host_entry):
            url = "http://%s/api/v1/health" % host_entry
            ret_base = "http://%s/api/v1" % host_entry
        else:
            if probe_port != 80:
                url = "http://%s:%d/api/v1/health" % (host_entry, probe_port)
                ret_base = "http://%s:%d/api/v1" % (host_entry, probe_port)
            else:
                url = "http://%s/api/v1/health" % host_entry
                ret_base = "http://%s/api/v1" % host_entry

        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=timeout_per_host) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, dict) and data.get("node_id") == "SIH-NODE-01":
                        return ret_base
        except Exception:
            pass
        return None

    found_url = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_host = {executor.submit(_probe_host, h): h for h in hosts}
        try:
            for fut in concurrent.futures.as_completed(future_to_host, timeout=total_timeout):
                res = fut.result()
                if res:
                    found_url = res
                    break
        except Exception:
            pass
    return found_url


def discover_field_station_url(data_dir, mast_base_url_override=None, candidate_hosts=None, probe_port=80):
    """
    Discovery fallback order (Spec §1.2 / Fix 1 & Fix 3):
      a) mast_base_url_override (if set)
      b) /etc/sih/field_station.json (if exists, never overwritten)
      c) <data_dir>/field_station.json (cache)
         - If healthy, return it.
         - If health check fails, run /24 probe once; if found, update cache and return it.
      d) parallel wlan0 /24 subnet probe (if no cache)
         - If found, save to <data_dir>/field_station.json and return it.
    """
    if mast_base_url_override:
        return mast_base_url_override

    # a) /etc/sih/field_station.json (configuration, never overwritten)
    etc_path = Path("/etc/sih/field_station.json")
    if etc_path.exists():
        try:
            with open(str(etc_path), "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict) and d.get("base_url"):
                    return d["base_url"].rstrip("/")
        except Exception:
            pass

    # b) <data_dir>/field_station.json (cache)
    data_path = Path(data_dir) / "field_station.json"
    cached_url = None
    if data_path.exists():
        try:
            with open(str(data_path), "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict) and d.get("base_url"):
                    cached_url = d["base_url"].rstrip("/")
        except Exception:
            pass

    prefix = _get_wlan_subnet_prefix()

    if cached_url:
        if check_field_station_health(cached_url, timeout=0.5):
            return cached_url
        # Cached URL failed health check: run /24 probe once
        found = probe_for_field_station(
            subnet_prefix=prefix,
            timeout_per_host=0.5,
            max_workers=32,
            total_timeout=8.0,
            candidate_hosts=candidate_hosts,
            probe_port=probe_port,
        )
        if found:
            # Update cache (<data_dir>/field_station.json); /etc/sih/field_station.json is NEVER overwritten
            try:
                data_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(data_path), "w", encoding="utf-8") as f:
                    json.dump({"base_url": found}, f)
            except Exception:
                pass
            return found
        return None

    # c) probe subnet in parallel
    found = probe_for_field_station(
        subnet_prefix=prefix,
        timeout_per_host=0.5,
        max_workers=32,
        total_timeout=8.0,
        candidate_hosts=candidate_hosts,
        probe_port=probe_port,
    )
    if found:
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(data_path), "w", encoding="utf-8") as f:
                json.dump({"base_url": found}, f)
        except Exception:
            pass
        return found

    return None


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

        # Scan controller state (Stage 1 / spec §1.2, §1.3, §1.5)
        self.engine_path = ROOT / "artifacts" / "engines" / "model_a_fp16.engine"
        self.camera_device = Path("/dev/video0")
        self.replay_source = ROOT / "test_video_from_dataset_images.mp4"
        self.pipeline_script = str(ROOT / "edge" / "pipeline.py")
        self.startup_timeout_s = 60.0
        self.watchdog_timeout_s = 30.0
        self.clock_source = "filesystem"
        self.init_utc = get_utc_iso_now()
        self.current_scan_state = "idle"
        self.current_scan_id = None
        self.current_scan_proc = None
        self.current_scan_monitor_thread = None
        self.current_scan_info = None
        self.current_scan_status_file = None
        self.current_scan_started_mono = 0.0
        self.last_scan_status = None
        self.current_stop_reason = None
        self.scan_lock = threading.Lock()
        self.field_station_status = {"reachable": None, "readings_collected": None, "last_reading_utc": None}
        self.field_station_warning = False

        # Interrupted recovery on startup (Point 5)
        try:
            self.storage.recover_interrupted_scans()
        except Exception:
            pass

    def is_pod_ready(self) -> bool:
        """
        pod_ready = gateway up AND (allow_mock OR (engine file exists AND /dev/video0 exists)).
        It is cheap: never open the camera or load the engine for health.
        """
        if not getattr(self, "is_ready", True):
            return False
        if getattr(self, "allow_mock", False):
            return True
        return self.engine_path.exists() and self.camera_device.exists()

    def get_scan_status(self) -> Dict[str, Any]:
        """Returns full spec §1.3 scan status shape with all required keys."""
        status_file = getattr(self, "current_scan_status_file", None)
        st = None
        if status_file and status_file.exists():
            try:
                with open(str(status_file), "r", encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:
                st = None

        if self.current_scan_state in ("done", "error") and self.last_scan_status is not None:
            status = dict(self.last_scan_status)
        elif st is not None:
            if self.current_scan_state not in ("done", "error", "finalizing") and "state" in st:
                self.current_scan_state = st["state"]
            status = dict(st)
            if self.current_scan_state == "finalizing":
                status["state"] = "finalizing"
        else:
            scan_id = self.current_scan_id
            elapsed_s = 0
            if self.current_scan_started_mono > 0:
                elapsed_s = int(time.monotonic() - self.current_scan_started_mono)
            status = {
                "state": self.current_scan_state,
                "scan_id": scan_id,
                "field_id": self.current_scan_info.get("field_id") if self.current_scan_info else None,
                "crop": self.current_scan_info.get("crop") if self.current_scan_info else None,
                "replay": self.current_scan_info.get("replay", False) if self.current_scan_info else False,
                "started_utc": self.current_scan_info.get("started_utc") if self.current_scan_info else None,
                "elapsed_s": elapsed_s,
                "max_duration_s": 1800,
                "counts": {
                    "frames_seen": 0,
                    "frames_used": 0,
                    "stretches": 0,
                    "healthy": 0,
                    "need_look": 0,
                    "unclear": 0,
                    "not_crop": 0,
                },
                "thermal_c_latest": None,
                "field_station": {
                    "reachable": None,
                    "readings_collected": None,
                    "last_reading_utc": None,
                },
                "warnings": [],
                "alerts": [],
                "advisory_id": None,
                "stop_reason": self.current_stop_reason,
            }

        # Guarantee all spec §1.3 keys exist
        required_counts = ["frames_seen", "frames_used", "stretches", "healthy", "need_look", "unclear", "not_crop"]
        if "counts" not in status or not isinstance(status["counts"], dict):
            status["counts"] = {}
        for k in required_counts:
            if k not in status["counts"]:
                status["counts"][k] = 0

        if "field_station" not in status or not isinstance(status["field_station"], dict):
            status["field_station"] = {
                "reachable": None,
                "readings_collected": None,
                "last_reading_utc": None,
            }
        # If status file has null field_station or missing, check server's live status
        if status["field_station"].get("reachable") is None:
            fs_live = getattr(self, "field_station_status", None)
            if fs_live and fs_live.get("reachable") is not None:
                status["field_station"] = dict(fs_live)
            else:
                for k in ("reachable", "readings_collected", "last_reading_utc"):
                    if k not in status["field_station"]:
                        status["field_station"][k] = None
        else:
            for k in ("reachable", "readings_collected", "last_reading_utc"):
                if k not in status["field_station"]:
                    status["field_station"][k] = None

        if "warnings" not in status or not isinstance(status["warnings"], list):
            status["warnings"] = []
        if "alerts" not in status or not isinstance(status["alerts"], list):
            status["alerts"] = []

        # Add FIELD_STATION_NOT_FOUND warning if active
        if getattr(self, "field_station_warning", False):
            has_fs_warn = any(w.get("code") == "FIELD_STATION_NOT_FOUND" for w in status["warnings"] if isinstance(w, dict))
            if not has_fs_warn:
                status["warnings"].append({
                    "code": "FIELD_STATION_NOT_FOUND",
                    "since_utc": self.current_scan_info.get("started_utc", self.init_utc) if getattr(self, "current_scan_info", None) else self.init_utc,
                })

        # Add STORAGE_CARD_MISSING warning if storage is internal
        if getattr(self.storage, "storage_location", "internal") == "internal":
            has_storage_warn = any(w.get("code") == "STORAGE_CARD_MISSING" for w in status["warnings"] if isinstance(w, dict))
            if not has_storage_warn:
                status["warnings"].append({"code": "STORAGE_CARD_MISSING", "since_utc": self.init_utc})

        if "advisory_id" not in status:
            status["advisory_id"] = None
        if "stop_reason" not in status:
            status["stop_reason"] = self.current_stop_reason

        return status

    def start_scan(
        self,
        field_id: str,
        crop: str,
        source: str = "camera",
        phone_utc: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        with self.scan_lock:
            # 1. Concurrency check (409)
            if self.current_scan_state in ("starting", "scanning", "finalizing"):
                return 409, {"error": "scan_in_progress", "scan_id": self.current_scan_id}

            # 2. Camera & Engine check (503)
            if not self.allow_mock:
                if not self.engine_path.exists():
                    return 503, {"error": "camera_unavailable", "detail": "model_a_fp16.engine missing"}
                if source == "camera" and not self.camera_device.exists():
                    return 503, {"error": "camera_unavailable", "detail": "/dev/video0 missing"}

            # 3. Adopt phone time if applicable
            if phone_utc and self.clock_source != "gps":
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", phone_utc):
                    try:
                        year = int(phone_utc[:4])
                        if 2025 <= year <= 2035:
                            phone_dt = _parse_iso_timestamp(phone_utc)
                            if phone_dt:
                                drift = abs((datetime.datetime.now(datetime.timezone.utc) - phone_dt).total_seconds())
                                if drift <= 2.0:
                                    self.clock_source = "phone"
                                    self.storage.clock_source = "phone"
                                else:
                                    try:
                                        res = subprocess.run(
                                            ["sudo", "-n", "/usr/local/sbin/aegis-set-time", phone_utc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            timeout=10,
                                        )
                                        print("[Gateway] aegis-set-time returncode=%d" % res.returncode)
                                        if res.returncode == 0:
                                            self.clock_source = "phone"
                                            self.storage.clock_source = "phone"
                                        else:
                                            print("[Gateway] WARNING: aegis-set-time failed (rc=%d), clock_source remains '%s'" % (res.returncode, self.clock_source))
                                    except Exception as exc:
                                        print("[Gateway] WARNING: aegis-set-time exception: %s" % exc)
                    except Exception:
                        pass

            # 4. Clear any previous error state & prepare scan
            now_iso = get_utc_iso_now()
            scan_id = "%s_%s" % (now_iso, field_id)
            replay_flag = (source == "replay")
            self.current_scan_id = scan_id
            self.current_scan_state = "starting"
            self.current_stop_reason = None
            self.last_scan_status = None
            self.current_scan_started_mono = time.monotonic()
            status_file_path = self.storage.data_dir / "status" / ("scan_status_%s.json" % scan_id)
            status_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_scan_status_file = status_file_path

            self.current_scan_info = {
                "scan_id": scan_id,
                "field_id": field_id,
                "crop": crop,
                "source": source,
                "replay": replay_flag,
                "started_utc": now_iso,
                "status_file": str(status_file_path),
            }

            # 5. Build pipeline subprocess command
            pipeline_script = getattr(self, "pipeline_script", str(ROOT / "edge" / "pipeline.py"))
            cmd = [
                sys.executable,
                pipeline_script,
                "--source", "0" if source == "camera" else str(self.replay_source),
                "--until-stopped",
                "--scan-id", scan_id,
                "--field-id", field_id,
                "--crop", crop,
                "--status-file", str(status_file_path),
                "--db-path", str(self.storage.db_path),
                "--time-source", self.clock_source,
            ]
            if source == "camera":
                backend_type = "mock" if self.allow_mock else "trt"
                cmd.extend(["--backend", backend_type])
                if not self.allow_mock:
                    cmd.extend(["--engine", str(self.engine_path)])
            else:
                backend_type = "mock" if self.allow_mock else "trt"
                cmd.extend(["--backend", backend_type, "--no-realtime"])
                if not self.allow_mock:
                    cmd.extend(["--engine", str(self.engine_path)])

            log_dir = self.storage.data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / ("%s.log" % scan_id)
            log_f = open(str(log_file_path), "a", encoding="utf-8")

            try:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
                self.current_scan_proc = proc
            except Exception as e:
                try:
                    log_f.close()
                except Exception:
                    pass
                self.current_scan_state = "error"
                self.current_stop_reason = "error"
                return 503, {"error": "camera_unavailable", "detail": str(e)}

            # Record scan start in SQLite with PID
            self.storage.record_scan_start(
                scan_id=scan_id,
                started_utc=now_iso,
                source=source,
                mode="walk",
                pid=proc.pid,
                status="running",
                crop=crop,
                field_id=field_id,
                time_source=self.clock_source,
                replay=1 if replay_flag else 0,
            )

            # Background field station discovery & telemetry poll (Spec §1.2 / Item G / Fix 1)
            self.field_station_warning = False

            def _poll_fs_on_start():
                try:
                    from edge.mast_collector import MastCollector
                    override = getattr(self, "mast_base_url", None)
                    base_url = discover_field_station_url(self.storage.data_dir, mast_base_url_override=override)
                    if not base_url:
                        self.field_station_status = {
                            "reachable": False,
                            "readings_collected": 0,
                            "last_reading_utc": None,
                        }
                        self.field_station_warning = True
                        return

                    collector = MastCollector(base_url=base_url, storage=self.storage, timeout=1.5, max_retries=0)
                    try:
                        res = collector.collect()
                    except Exception:
                        res = None

                    # If collect failed and base_url came from cache, run /24 probe once
                    if not isinstance(res, dict) or res.get("status") != "ok":
                        data_path = Path(self.storage.data_dir) / "field_station.json"
                        is_cached = False
                        if data_path.exists():
                            try:
                                with open(str(data_path), "r", encoding="utf-8") as f:
                                    d = json.load(f)
                                    if isinstance(d, dict) and d.get("base_url") == base_url:
                                        is_cached = True
                            except Exception:
                                pass
                        if is_cached:
                            prefix = _get_wlan_subnet_prefix()
                            found = probe_for_field_station(
                                subnet_prefix=prefix,
                                timeout_per_host=0.5,
                                max_workers=32,
                                total_timeout=8.0,
                            )
                            if found and found != base_url:
                                try:
                                    with open(str(data_path), "w", encoding="utf-8") as f:
                                        json.dump({"base_url": found}, f)
                                except Exception:
                                    pass
                                base_url = found
                                collector = MastCollector(base_url=base_url, storage=self.storage, timeout=1.5, max_retries=0)
                                try:
                                    res = collector.collect()
                                except Exception:
                                    res = None

                    if isinstance(res, dict) and res.get("status") == "ok":
                        latest = self.storage.get_latest_mast_telemetry()
                        last_utc = latest.get("utc") if latest else None
                        self.field_station_status = {
                            "reachable": True,
                            "readings_collected": int(res.get("records_pulled", 0)),
                            "last_reading_utc": last_utc,
                        }
                        self.field_station_warning = False

                        # Set ESP32 clock if phone or gps (Fix 1)
                        if self.clock_source in ("phone", "gps"):
                            try:
                                collector.set_time(int(time.time()))
                            except Exception:
                                pass
                    else:
                        self.field_station_status = {
                            "reachable": False,
                            "readings_collected": 0,
                            "last_reading_utc": None,
                        }
                        self.field_station_warning = True
                except Exception:
                    self.field_station_status = {
                        "reachable": False,
                        "readings_collected": 0,
                        "last_reading_utc": None,
                    }
                    self.field_station_warning = True

            t_fs = threading.Thread(target=_poll_fs_on_start, daemon=True, name="FieldStationStartPoll")
            t_fs.start()

            # Spawn background monitor thread
            t = threading.Thread(
                target=self._monitor_scan_process,
                args=(scan_id, proc, status_file_path, self.current_scan_started_mono, log_f),
            )
            t.daemon = True
            self.current_scan_monitor_thread = t
            t.start()

            return 202, {"scan_id": scan_id, "state": "starting", "replay": replay_flag}

    def _has_advisory_for_scan(self, scan_id):
        """Check if an advisory already exists for a scan_id. Must be called under scan_lock."""
        try:
            conn = self.storage._get_connection()
            row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchone()
            return row is not None
        except Exception:
            return False

    def _poll_fs_on_finalize(self):
        """Collect once more at stop/finalize (max 5s, non-blocking) so advisory gets latest readings."""
        def _bg_stop_poll():
            try:
                from edge.mast_collector import MastCollector
                override = getattr(self, "mast_base_url", None)
                base_url = discover_field_station_url(self.storage.data_dir, mast_base_url_override=override)
                if base_url:
                    collector = MastCollector(base_url=base_url, storage=self.storage, timeout=1.5, max_retries=0)
                    collector.collect()
            except Exception:
                pass
        t = threading.Thread(target=_bg_stop_poll, daemon=True, name="FieldStationStopPoll")
        t.start()

    def _build_final_status(self, final_st, scan_id, state, stop_reason, adv_id):
        """Build the final status dict, preferring pipeline's own status file."""
        if final_st and final_st.get("scan_id") == scan_id:
            st = dict(final_st)
        else:
            st = self.get_scan_status()
        st["state"] = state
        st["stop_reason"] = stop_reason
        st["advisory_id"] = adv_id
        st["scan_id"] = scan_id
        return st

    def _monitor_scan_process(
        self,
        scan_id: str,
        proc: subprocess.Popen,
        status_file: Path,
        started_mono: float,
        log_f: Any = None,
    ) -> None:
        """Background thread monitoring the pipeline subprocess."""
        status_file_seen = False
        try:
            while True:
                time.sleep(0.5)

                with self.scan_lock:
                    if self.current_scan_proc is not proc:
                        break

                    # Check if process exited
                    ret = proc.poll()
                    if ret is not None:
                        # Read the pipeline's final status file for this scan
                        final_st = None
                        if status_file.exists():
                            try:
                                with open(str(status_file), "r", encoding="utf-8") as f:
                                    final_st = json.load(f)
                                if final_st.get("scan_id") != scan_id:
                                    final_st = None
                            except Exception:
                                final_st = None

                        if self.current_scan_state == "finalizing":
                            eff_reason = self.current_stop_reason if self.current_stop_reason == "interrupted" else "user"
                            self.current_scan_state = "done"
                            self.current_stop_reason = eff_reason
                            adv_id = final_st.get("advisory_id") if final_st else None
                            if not adv_id and not self._has_advisory_for_scan(scan_id):
                                try:
                                    adv = self.storage.create_advisory(scan_id=scan_id, stop_reason=eff_reason)
                                    adv_id = adv.get("advisory_id")
                                except Exception:
                                    pass
                            if not adv_id:
                                try:
                                    conn = self.storage._get_connection()
                                    row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchone()
                                    if row:
                                        adv_id = row["advisory_id"]
                                except Exception:
                                    pass
                            try:
                                status_name = "interrupted" if eff_reason == "interrupted" else "complete"
                                self.storage.record_scan_end(scan_id=scan_id, status=status_name, stop_reason=eff_reason)
                            except Exception:
                                pass
                            st = self._build_final_status(final_st, scan_id, "done", eff_reason, adv_id)
                            self.last_scan_status = st
                        elif ret != 0:
                            self.current_scan_state = "error"
                            self.current_stop_reason = "error"
                            adv_id = None
                            if not self._has_advisory_for_scan(scan_id):
                                try:
                                    adv = self.storage.create_advisory(scan_id=scan_id, stop_reason="error")
                                    adv_id = adv.get("advisory_id")
                                except Exception:
                                    pass
                            if not adv_id:
                                try:
                                    conn = self.storage._get_connection()
                                    row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchone()
                                    if row:
                                        adv_id = row["advisory_id"]
                                except Exception:
                                    pass
                            try:
                                self.storage.record_scan_end(scan_id=scan_id, status="error", stop_reason="error")
                            except Exception:
                                pass
                            st = self._build_final_status(final_st, scan_id, "error", "error", adv_id)
                            self.last_scan_status = st
                        else:
                            # Exit 0: prefer pipeline's own final status file
                            if final_st and final_st.get("stop_reason"):
                                eff_reason = final_st["stop_reason"]
                            elif self.current_stop_reason:
                                eff_reason = self.current_stop_reason
                            else:
                                eff_reason = "user"
                            self.current_scan_state = "done"
                            self.current_stop_reason = eff_reason
                            adv_id = final_st.get("advisory_id") if final_st else None
                            if not adv_id and not self._has_advisory_for_scan(scan_id):
                                try:
                                    adv = self.storage.create_advisory(scan_id=scan_id, stop_reason=eff_reason)
                                    adv_id = adv.get("advisory_id")
                                except Exception:
                                    pass
                            if not adv_id:
                                try:
                                    conn = self.storage._get_connection()
                                    row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchone()
                                    if row:
                                        adv_id = row["advisory_id"]
                                except Exception:
                                    pass
                            try:
                                self.storage.record_scan_end(scan_id=scan_id, status="complete", stop_reason=eff_reason)
                            except Exception:
                                pass
                            st = self._build_final_status(final_st, scan_id, "done", eff_reason, adv_id)
                            self.last_scan_status = st

                        self.current_scan_proc = None
                        break

                    # Check status file
                    if status_file.exists():
                        status_file_seen = True
                        try:
                            with open(str(status_file), "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("state") == "scanning":
                                self.current_scan_state = "scanning"
                            elif data.get("state") in ("done", "error"):
                                self.current_scan_state = data.get("state")
                        except Exception:
                            pass

                    # Timeout check: if no status file written within startup_timeout_s of spawn
                    now_mono = time.monotonic()
                    timeout_limit = getattr(self, "startup_timeout_s", 60.0)
                    if not status_file_seen and (now_mono - started_mono > timeout_limit):
                        try:
                            proc.terminate()
                            time.sleep(1.0)
                            if proc.poll() is None:
                                proc.kill()
                        except Exception:
                            pass
                        self.current_scan_state = "error"
                        self.current_stop_reason = "error"
                        adv_id = None
                        if not self._has_advisory_for_scan(scan_id):
                            try:
                                adv = self.storage.create_advisory(scan_id=scan_id, stop_reason="error")
                                adv_id = adv.get("advisory_id")
                            except Exception:
                                pass
                        try:
                            self.storage.record_scan_end(scan_id=scan_id, status="error", stop_reason="error")
                        except Exception:
                            pass
                        st = self.get_scan_status()
                        st["state"] = "error"
                        st["stop_reason"] = "error"
                        st["advisory_id"] = adv_id
                        self.last_scan_status = st
                        self.current_scan_proc = None
                        break
        finally:
            if log_f is not None:
                try:
                    log_f.close()
                except Exception:
                    pass

    def stop_scan(self, reason: str = "user", wait_timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
        with self.scan_lock:
            # Fix 12: Ignore client reason; always "user"
            effective_reason = "user"

            if self.current_scan_state in ("idle", "done", "error"):
                return 200, self.get_scan_status()
            if self.current_scan_state == "finalizing":
                return 200, self.get_scan_status()

            proc = self.current_scan_proc
            active_scan_id = self.current_scan_id

            if proc is None or proc.poll() is not None:
                # Process is already gone! Finalize immediately, never leave state at finalizing without a process
                ret = proc.poll() if proc is not None else -1
                if ret != 0:
                    self.current_scan_state = "error"
                    self.current_stop_reason = "error"
                    adv_id = None
                    if not self._has_advisory_for_scan(active_scan_id):
                        try:
                            adv = self.storage.create_advisory(scan_id=active_scan_id, stop_reason="error")
                            adv_id = adv.get("advisory_id")
                        except Exception:
                            pass
                    try:
                        self.storage.record_scan_end(scan_id=active_scan_id, status="error", stop_reason="error")
                    except Exception:
                        pass
                    st = self.get_scan_status()
                    st["state"] = "error"
                    st["stop_reason"] = "error"
                    st["advisory_id"] = adv_id
                    self.last_scan_status = st
                else:
                    self.current_scan_state = "done"
                    self.current_stop_reason = effective_reason
                    adv_id = None
                    if not self._has_advisory_for_scan(active_scan_id):
                        try:
                            adv = self.storage.create_advisory(scan_id=active_scan_id, stop_reason=effective_reason)
                            adv_id = adv.get("advisory_id")
                        except Exception:
                            pass
                    try:
                        self.storage.record_scan_end(scan_id=active_scan_id, status="complete", stop_reason=effective_reason)
                    except Exception:
                        pass
                    st = self.get_scan_status()
                    st["state"] = "done"
                    st["stop_reason"] = effective_reason
                    st["advisory_id"] = adv_id
                    self.last_scan_status = st

                self.current_scan_proc = None
                return 200, self.get_scan_status()

            self.current_scan_state = "finalizing"
            self.current_stop_reason = effective_reason
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

            effective_wait = getattr(self, "watchdog_timeout_s", wait_timeout)

            def _watchdog():
                t0 = time.monotonic()
                while time.monotonic() - t0 < effective_wait:
                    if proc.poll() is not None:
                        return
                    time.sleep(0.1)
                with self.scan_lock:
                    if self.current_scan_proc == proc and proc.poll() is None:
                        self.current_stop_reason = "interrupted"
                        try:
                            proc.kill()
                            proc.wait(timeout=1.0)
                        except Exception:
                            pass
                        self.current_scan_state = "done"
                        adv_id = None
                        if not self._has_advisory_for_scan(active_scan_id):
                            try:
                                adv = self.storage.create_advisory(scan_id=active_scan_id, stop_reason="interrupted")
                                adv_id = adv.get("advisory_id")
                            except Exception:
                                pass
                        if not adv_id:
                            try:
                                conn = self.storage._get_connection()
                                row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (active_scan_id,)).fetchone()
                                if row:
                                    adv_id = row["advisory_id"]
                            except Exception:
                                pass
                        try:
                            self.storage.record_scan_end(scan_id=active_scan_id, status="interrupted", stop_reason="interrupted")
                        except Exception:
                            pass
                        st = self.get_scan_status()
                        st["state"] = "done"
                        st["stop_reason"] = "interrupted"
                        st["advisory_id"] = adv_id
                        st["scan_id"] = active_scan_id
                        self.last_scan_status = st
                        self.current_scan_proc = None

            wd = threading.Thread(target=_watchdog)
            wd.daemon = True
            wd.start()

            return 202, {"state": "finalizing"}

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
                health["pod_ready"] = self.server.is_pod_ready()
                health["scan_state"] = getattr(self.server, "current_scan_state", "idle")
                self._send_json_response(200, health)
                return

            # Route: GET /api/v1/scan/status (Stage 1 / Spec §1.3)
            if path == "/api/v1/scan/status":
                status = self.server.get_scan_status()
                self._send_json_response(200, status)
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
                    self._send_error_json(400, "bad_request", "Malformed JSON: %s" % ex)
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
            # Route: POST /api/v1/scan/start (Spec §1.2)
            if path == "/api/v1/scan/start":
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    self._send_error_json(400, "bad_request", "Missing Content-Length header")
                    return
                try:
                    length = int(content_length)
                    raw_body = self.rfile.read(length)
                    body_json = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception as ex:
                    self._send_error_json(400, "bad_request", "Malformed JSON: %s" % ex)
                    return

                crop = body_json.get("crop")
                field_id = body_json.get("field_id")
                source = body_json.get("source", "camera")
                phone_utc = body_json.get("phone_utc")

                if not crop or crop not in ("wheat", "rice", "sugarcane"):
                    self._send_error_json(400, "bad_request", "crop is required and must be one of: wheat, rice, sugarcane")
                    return
                if not field_id or not isinstance(field_id, str) or not field_id.strip():
                    self._send_error_json(400, "bad_request", "field_id is required and must be a non-empty string")
                    return
                if source not in ("camera", "replay"):
                    self._send_error_json(400, "bad_request", "source must be 'camera' or 'replay'")
                    return
                if phone_utc is not None:
                    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", phone_utc):
                        self._send_error_json(400, "bad_request", "phone_utc must be ISO-8601 UTC with Z suffix (YYYY-MM-DDTHH:MM:SSZ)")
                        return
                    try:
                        yr = int(phone_utc[:4])
                        if not (2025 <= yr <= 2035):
                            self._send_error_json(400, "bad_request", "phone_utc year out of valid range (2025-2035)")
                            return
                    except Exception:
                        self._send_error_json(400, "bad_request", "invalid phone_utc year")
                        return

                code, resp = self.server.start_scan(
                    field_id=field_id.strip(),
                    crop=crop,
                    source=source,
                    phone_utc=phone_utc,
                )
                self._send_json_response(code, resp)
                return

            # Route: POST /api/v1/scan/track (Spec §1.4 / Item E)
            if path == "/api/v1/scan/track":
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    self._send_error_json(400, "bad_request", "Missing Content-Length header")
                    return
                try:
                    length = int(content_length)
                    raw_body = self.rfile.read(length)
                    body_json = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception as ex:
                    self._send_error_json(400, "bad_request", "Malformed JSON: %s" % ex)
                    return

                if not isinstance(body_json, dict):
                    self._send_error_json(400, "bad_request", "JSON object payload expected")
                    return

                scan_id = body_json.get("scan_id")
                fixes = body_json.get("fixes")
                if not scan_id or not isinstance(scan_id, str):
                    self._send_error_json(400, "bad_request", "scan_id is required")
                    return
                if fixes is None or not isinstance(fixes, list):
                    self._send_error_json(400, "bad_request", "fixes list is required")
                    return
                if len(fixes) > 500:
                    self._send_error_json(400, "bad_request", "Batch size exceeds maximum limit of 500 fixes")
                    return

                # Check if scan exists in DB (Fix 8: 404 unknown_scan)
                conn = self.server.storage._get_connection()
                srow = conn.execute("SELECT scan_id FROM scans WHERE scan_id = ?;", (scan_id.strip(),)).fetchone()
                if not srow:
                    self._send_error_json(404, "unknown_scan", "Scan ID '%s' not found" % scan_id)
                    return

                res = self.server.storage.record_scan_track(scan_id=scan_id.strip(), fixes=fixes)
                self._send_json_response(200, {"accepted": res["accepted"]})
                return

            # Route: POST /api/v1/scan/stop (Spec §1.5)
            if path == "/api/v1/scan/stop":
                reason = "user"
                content_length = self.headers.get("Content-Length")
                if content_length:
                    try:
                        length = int(content_length)
                        raw_body = self.rfile.read(length)
                        if raw_body:
                            body_json = json.loads(raw_body.decode("utf-8"))
                            if isinstance(body_json, dict) and "reason" in body_json:
                                reason = str(body_json["reason"])
                    except Exception:
                        pass

                code, resp = self.server.stop_scan(reason=reason)
                self._send_json_response(code, resp)
                return

            # Route: POST /api/v1/pod/shutdown (Spec §1.6 / Point F)
            if path == "/api/v1/pod/shutdown":
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    self._send_error_json(400, "missing_confirm", "confirm: true is required")
                    return
                try:
                    length = int(content_length)
                    raw_body = self.rfile.read(length)
                    body_json = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception as ex:
                    self._send_error_json(400, "bad_request", "Malformed JSON: %s" % ex)
                    return

                if not isinstance(body_json, dict) or body_json.get("confirm") is not True:
                    self._send_error_json(400, "missing_confirm", "confirm: true is required")
                    return

                self._send_json_response(202, {"state": "shutting_down", "shutting_down_in_s": 5})

                def _async_shutdown():
                    if getattr(self.server, "current_scan_proc", None) is not None:
                        self.server.stop_scan(reason="user")
                        t0 = time.monotonic()
                        while time.monotonic() - t0 < 30.0:
                            if self.server.current_scan_proc is None or self.server.current_scan_proc.poll() is not None:
                                break
                            time.sleep(0.5)
                        with self.server.scan_lock:
                            if self.server.current_scan_proc is not None and self.server.current_scan_proc.poll() is None:
                                try:
                                    self.server.current_scan_proc.kill()
                                    self.server.current_scan_proc.wait(timeout=1.0)
                                except Exception:
                                    pass
                                active_id = self.server.current_scan_id
                                self.server.current_stop_reason = "interrupted"
                                self.server.current_scan_state = "done"
                                adv_id = None
                                if not self.server._has_advisory_for_scan(active_id):
                                    try:
                                        adv = self.server.storage.create_advisory(scan_id=active_id, stop_reason="interrupted")
                                        adv_id = adv.get("advisory_id")
                                    except Exception:
                                        pass
                                try:
                                    self.server.storage.record_scan_end(scan_id=active_id, status="interrupted", stop_reason="interrupted")
                                except Exception:
                                    pass
                                st = self.server.get_scan_status()
                                st["state"] = "done"
                                st["stop_reason"] = "interrupted"
                                if adv_id:
                                    st["advisory_id"] = adv_id
                                st["scan_id"] = active_id
                                self.server.last_scan_status = st
                                self.server.current_scan_proc = None
                    time.sleep(1.0)
                    try:
                        res = subprocess.run(
                            ["sudo", "-n", "/usr/local/sbin/aegis-shutdown"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=10,
                        )
                        print("[Gateway] aegis-shutdown returncode=%d" % res.returncode)
                    except Exception as exc:
                        print("[Gateway] WARNING: aegis-shutdown exception: %s" % exc)

                t = threading.Thread(target=_async_shutdown)
                t.daemon = True
                t.start()
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
        proc = getattr(self.server, "current_scan_proc", None)
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        t_mon = getattr(self.server, "current_scan_monitor_thread", None)
        if t_mon is not None and t_mon.is_alive():
            try:
                t_mon.join(timeout=3.0)
            except Exception:
                pass
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
