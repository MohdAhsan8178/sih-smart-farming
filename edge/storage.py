#!/usr/bin/env python3
"""
edge/storage.py — SQLite Persistence Layer for the Handheld Nano Pod (Step 33).

Provides thread-safe local storage for scan sessions, frame-level events,
cell-level spatial verdicts, and synthesized advisory documents conforming to the
API contract in ans_for_vitthal.md (§2, §3, §8).

Key Architectural Properties:
1. Stdlib sqlite3 only (zero external dependencies, strictly Python 3.6 compatible).
2. WAL mode enabled (PRAGMA journal_mode=WAL) with 5000ms busy timeout for safe concurrent
   reading by gateway/ while DecisionAggregateStoreThread writes.
3. Thread-isolated connections (threading.local) preventing cross-thread lock contention.
4. Retention policy: enforces MAX_RETAINED_FRAMES (default 50,000 frames, empirically measured
   at 121 MB / 5.9% of free eMMC) on granular frame events, while preserving advisory documents permanently
   without risk to the 2GB eMMC free space.
5. Direct query helpers matching gateway endpoints:
   - get_manifest(since_seq, limit) -> GET /api/v1/manifest
   - get_advisory(advisory_id_or_seq) -> GET /api/v1/advisory/<id>
   - ack_advisory(advisory_id) -> POST /api/v1/ack
   - get_health() -> GET /api/v1/health
"""

import collections
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.reliability import get_cross_source_reliability
import configs.train_config as train_config
import numpy as np

# SD Card mount and storage path definitions
SD_MOUNT_POINT = Path("/mnt/aegisdata")


def resolve_data_directory(
    mount_point: Optional[Union[str, Path]] = None,
    internal_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str, bool]:
    """
    Resolves active data directory based on SD card mount status and writability.

    Rule:
      Use SD only if os.path.ismount(mount_point) AND a real write test succeeds
      (create + delete a temp file in <mount_point>/aegis/).
      If not mounted or write test fails, falls back to internal_dir (repo ROOT / 'data')
      and reports card_missing=True.

    Returns:
      (resolved_data_dir: Path, location_str: 'sd'|'internal', card_missing: bool)
    """
    mnt = Path(mount_point) if mount_point is not None else SD_MOUNT_POINT
    repo_root = Path(__file__).resolve().parent.parent
    internal = Path(internal_dir) if internal_dir is not None else (repo_root / "data")

    # 1. Check if mount_point is an actual mount
    if os.path.ismount(str(mnt)):
        sd_aegis = mnt / "aegis"
        try:
            sd_aegis.mkdir(parents=True, exist_ok=True)
            # Real write test: create + delete temp file
            test_file = sd_aegis / (".write_test_%d_%d" % (os.getpid(), int(time.time())))
            with open(str(test_file), "w") as f:
                f.write("aegis_sd_test\n")
            test_file.unlink()

            # Ensure subdirectories exist
            (sd_aegis / "archive").mkdir(parents=True, exist_ok=True)
            (sd_aegis / "logs").mkdir(parents=True, exist_ok=True)
            (sd_aegis / "status").mkdir(parents=True, exist_ok=True)

            # One-time migration: if SD dir has no edge.db and internal data/edge.db exists, copy it
            sd_db = sd_aegis / "edge.db"
            internal_db = internal / "edge.db"
            if not sd_db.exists() and internal_db.exists():
                shutil.copy2(str(internal_db), str(sd_db))

            return sd_aegis, "sd", False
        except Exception:
            pass

    # Fallback to internal storage
    internal.mkdir(parents=True, exist_ok=True)
    (internal / "archive").mkdir(parents=True, exist_ok=True)
    (internal / "logs").mkdir(parents=True, exist_ok=True)
    (internal / "status").mkdir(parents=True, exist_ok=True)
    return internal, "internal", True


def get_storage_free_mb(target_dir: Union[str, Path]) -> int:
    """Returns free filesystem space in megabytes for the given directory."""
    try:
        st = os.statvfs(str(target_dir))
        return int((st.f_bavail * st.f_frsize) / (1024 * 1024))
    except Exception:
        return 0


DEFAULT_DATA_DIR, DEFAULT_STORAGE_LOCATION, DEFAULT_STORAGE_CARD_MISSING = resolve_data_directory()
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "edge.db"

# Retention ceiling: 50,000 frames (empirically measured at 121 MB on disk: ~2.1 KB/row including 9 tile decisions, metadata, B-tree indexes, and SQLite freelist page allocation)
DEFAULT_MAX_RETAINED_FRAMES = 50000


def get_utc_iso_now() -> str:
    """Returns current UTC time formatted as ISO-8601 with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_timestamp(ts_str: Optional[str]) -> Optional[datetime.datetime]:
    """Parses ISO timestamp string into timezone-aware datetime in a Python 3.6 compatible manner."""
    if not ts_str:
        return None
    clean = str(ts_str).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.datetime.strptime(clean.replace("+00:00", "").replace("Z", ""), fmt)
            return naive.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def _extract_iso_from_scan_id(scan_id: Optional[str]) -> Optional[str]:
    """Extracts ISO timestamp from standard scan_id prefix if present (e.g. 2026-10-03T10:00:00Z_F01)."""
    if not scan_id:
        return None
    part = str(scan_id).split("_")[0]
    dt = _parse_iso_timestamp(part)
    if dt:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


HEALTHY_CLASSES_BY_CROP: Dict[str, Set[str]] = {
    "rice": {"rice__normal"},
    "sugarcane": {"sugarcane__healthy", "sugarcane__dried_leaf"},
    "wheat": {"wheat__healthy"},
}


class EdgeStorage(object):
    """
    Thread-safe SQLite storage engine for edge pipeline telemetry, detections,
    spatial cell aggregation, and advisory generation.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        max_retained_frames: int = DEFAULT_MAX_RETAINED_FRAMES,
    ):
        if db_path is None or Path(db_path) == DEFAULT_DB_PATH:
            resolved_dir, loc, missing = resolve_data_directory()
            self.data_dir = resolved_dir
            self.storage_location = loc
            self.storage_card_missing = missing
            self.db_path = self.data_dir / "edge.db"
        else:
            self.db_path = Path(db_path)
            self.data_dir = self.db_path.parent
            if "/mnt/aegisdata" in str(self.db_path):
                self.storage_location = "sd"
                self.storage_card_missing = False
            else:
                self.storage_location = "internal"
                self.storage_card_missing = True

        self.max_retained_frames = int(max_retained_frames)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock_source = "filesystem"

        self._local = threading.local()
        self._init_lock = threading.Lock()

        # Initialize schema and pragmas
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection with WAL mode and row factory."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=5.0,  # 5000ms busy timeout
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Enforce concurrency and performance pragmas
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_database(self) -> None:
        """Initializes tables and indexes under an exclusive initialization lock."""
        with self._init_lock:
            conn = self._get_connection()
            with conn:
                # 1. Scan sessions
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scans (
                        scan_id TEXT PRIMARY KEY,
                        started_utc TEXT NOT NULL,
                        ended_utc TEXT,
                        mode TEXT DEFAULT 'handheld_pod',
                        source TEXT,
                        frames_captured INTEGER DEFAULT 0,
                        frames_evaluated INTEGER DEFAULT 0,
                        tiles_classified INTEGER DEFAULT 0,
                        distance_walked_m REAL,
                        distance_reason TEXT,
                        metadata_json TEXT,
                        pid INTEGER,
                        status TEXT DEFAULT 'complete',
                        crop TEXT,
                        field_id TEXT,
                        stop_reason TEXT,
                        duration_s REAL,
                        time_source TEXT,
                        replay INTEGER DEFAULT 0,
                        duration_reason TEXT,
                        replay_loops INTEGER DEFAULT 0
                    );
                    """
                )

                # Safe column migrations for existing databases
                try:
                    scan_cols = set(r["name"] for r in conn.execute("PRAGMA table_info(scans);").fetchall())
                    for col_name, col_type in [
                        ("pid", "INTEGER"),
                        ("status", "TEXT DEFAULT 'complete'"),
                        ("crop", "TEXT"),
                        ("field_id", "TEXT"),
                        ("stop_reason", "TEXT"),
                        ("duration_s", "REAL"),
                        ("time_source", "TEXT"),
                        ("replay", "INTEGER DEFAULT 0"),
                        ("duration_reason", "TEXT"),
                        ("replay_loops", "INTEGER DEFAULT 0"),
                    ]:
                        if scan_cols and col_name not in scan_cols:
                            conn.execute("ALTER TABLE scans ADD COLUMN %s %s;" % (col_name, col_type))
                except Exception:
                    pass

                # 2. Granular frame events (replaces JSONL stream)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS frame_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id TEXT NOT NULL,
                        frame_idx INTEGER NOT NULL,
                        timestamp_utc TEXT NOT NULL,
                        source_image TEXT,
                        cell_id TEXT NOT NULL,
                        gate_passed INTEGER NOT NULL DEFAULT 1,
                        blur_score REAL,
                        dark_fraction REAL,
                        bright_fraction REAL,
                        displacement REAL,
                        canopy_cover REAL,
                        vari REAL,
                        exg REAL,
                        tgi REAL,
                        dgci REAL,
                        dgci_ood_frac REAL,
                        indices_status TEXT,
                        n_valid_tiles INTEGER NOT NULL,
                        frame_state TEXT NOT NULL,
                        class_id INTEGER,
                        class_name TEXT,
                        confidence REAL NOT NULL,
                        lat REAL,
                        lon REAL,
                        gps_fix_quality INTEGER,
                        gps_hdop REAL,
                        tile_decisions_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
                    );
                    """
                )

                # 3. Spatial GPS cell verdicts
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cell_verdicts (
                        scan_id TEXT NOT NULL,
                        cell_id TEXT NOT NULL,
                        state TEXT NOT NULL,
                        class_id INTEGER,
                        class_name TEXT,
                        score REAL NOT NULL,
                        n_frames INTEGER NOT NULL,
                        n_agree INTEGER NOT NULL,
                        canopy_cover REAL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (scan_id, cell_id),
                        FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
                    );
                    """
                )

                # Safe column migrations for existing databases
                try:
                    fe_cols = set(r["name"] for r in conn.execute("PRAGMA table_info(frame_events);").fetchall())
                    if fe_cols and "canopy_cover" not in fe_cols:
                        conn.execute("ALTER TABLE frame_events ADD COLUMN canopy_cover REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN vari REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN exg REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN tgi REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN dgci REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN dgci_ood_frac REAL;")
                        conn.execute("ALTER TABLE frame_events ADD COLUMN indices_status TEXT;")
                except Exception:
                    pass

                try:
                    cv_cols = set(r["name"] for r in conn.execute("PRAGMA table_info(cell_verdicts);").fetchall())
                    if cv_cols and "canopy_cover" not in cv_cols:
                        conn.execute("ALTER TABLE cell_verdicts ADD COLUMN canopy_cover REAL;")
                except Exception:
                    pass

                # 4. Synthesized Advisories for phone sync (ans_for_vitthal.md §2, §8)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS advisories (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        advisory_id TEXT UNIQUE NOT NULL,
                        scan_id TEXT NOT NULL,
                        generated_at_utc TEXT NOT NULL,
                        bytes INTEGER NOT NULL,
                        replay INTEGER NOT NULL DEFAULT 1,
                        acked INTEGER NOT NULL DEFAULT 0,
                        acked_at_utc TEXT,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
                    );
                    """
                )

                # 5. Key-value state (last acked sequence, sync cursor)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_sync_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                # 6. Sticky-Trap Pest Records (Model B Gateway Job, Step 31 / H5)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trap_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trap_id TEXT NOT NULL,
                        job_id TEXT UNIQUE NOT NULL,
                        image_path TEXT NOT NULL,
                        placed_at_utc TEXT,
                        processed_at_utc TEXT NOT NULL,
                        days_monitored REAL,
                        total_blobs_counted INTEGER NOT NULL,
                        scale_mm_per_pixel REAL,
                        scale_status TEXT NOT NULL,
                        etl_status TEXT NOT NULL,
                        pest_json TEXT NOT NULL
                    );
                    """
                )

                # 7. Ground Mast Telemetry (ESP32 Station SIH-NODE-01 / Guide §6)
                try:
                    mt_cols = set(r["name"] for r in conn.execute("PRAGMA table_info(mast_telemetry);").fetchall())
                    if mt_cols and "log_epoch" not in mt_cols:
                        conn.execute("DROP TABLE mast_telemetry;")
                except Exception:
                    pass

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mast_telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_epoch INTEGER NOT NULL,
                        seq INTEGER NOT NULL,
                        node_id TEXT NOT NULL,
                        field_id TEXT,
                        utc TEXT,
                        rtc_valid INTEGER NOT NULL DEFAULT 0,
                        uptime_s INTEGER,
                        air_temp_c REAL,
                        rh_pct REAL,
                        ir_object_c REAL,
                        ir_ambient_c REAL,
                        lux REAL,
                        soil1_v REAL,
                        soil2_v REAL,
                        battery_v REAL,
                        status_json TEXT,
                        received_at TEXT NOT NULL,
                        UNIQUE(log_epoch, seq)
                    );
                    """
                )

                # 8. Ground Mast Sync Cursor (Guide §6)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mast_sync_state (
                        node_id TEXT PRIMARY KEY,
                        log_epoch INTEGER NOT NULL,
                        last_seq INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                # 9. Sentinel-2 Satellite NDVI Cache (M4.5)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS satellite_ndvi (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        field_id TEXT,
                        fetched_utc TEXT NOT NULL,
                        scene_date TEXT NOT NULL,
                        ndvi_mean REAL,
                        ndvi_std REAL,
                        valid_pixel_count INTEGER,
                        cloud_masked_fraction REAL,
                        raw_response_json TEXT,
                        source TEXT NOT NULL DEFAULT 'SENTINEL2_L2A_CDSE'
                    );
                    """
                )

                # Performance indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_satellite_ndvi_date ON satellite_ndvi(scene_date);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_events_scan ON frame_events(scan_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_events_cell ON frame_events(cell_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_advisories_seq ON advisories(seq);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_advisories_acked ON advisories(acked);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trap_records_trap ON trap_records(trap_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trap_records_job ON trap_records(job_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mast_telemetry_epoch_seq ON mast_telemetry(log_epoch, seq);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mast_telemetry_rec ON mast_telemetry(received_at);")

    # -------------------------------------------------------------------------
    # Scan Session Lifecycle
    # -------------------------------------------------------------------------

    def record_scan_start(
        self,
        scan_id: str,
        started_utc: Optional[str] = None,
        source: Optional[str] = None,
        mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pid: Optional[int] = None,
        status: str = "running",
        crop: Optional[str] = None,
        field_id: Optional[str] = None,
        time_source: Optional[str] = None,
        replay: Optional[Union[bool, int]] = None,
    ) -> None:
        """Records the beginning of a scanning session safely without wiping gateway columns."""
        started_utc = started_utc or _extract_iso_from_scan_id(scan_id) or get_utc_iso_now()
        conn = self._get_connection()
        initial_mode = mode or "handheld_pod"
        replay_int = int(replay) if replay is not None else 0

        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO scans (
                    scan_id, started_utc, mode, source, metadata_json,
                    pid, status, crop, field_id, time_source, replay
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    scan_id,
                    started_utc,
                    initial_mode,
                    str(source) if source is not None else None,
                    json.dumps(metadata or {}) if metadata is not None else None,
                    int(pid) if pid is not None else None,
                    str(status),
                    str(crop) if crop is not None else None,
                    str(field_id) if field_id is not None else None,
                    str(time_source) if time_source is not None else None,
                    replay_int,
                ),
            )
            # If the row already exists, update ONLY columns explicitly provided (not None)
            updates: List[str] = []
            vals: List[Any] = []
            if mode is not None:
                updates.append("mode = ?")
                vals.append(str(mode))
            if source is not None:
                updates.append("source = ?")
                vals.append(str(source))
            if metadata is not None:
                updates.append("metadata_json = ?")
                vals.append(json.dumps(metadata))
            if pid is not None:
                updates.append("pid = ?")
                vals.append(int(pid))
            if status is not None:
                updates.append("status = ?")
                vals.append(str(status))
            if crop is not None:
                updates.append("crop = ?")
                vals.append(str(crop))
            if field_id is not None:
                updates.append("field_id = ?")
                vals.append(str(field_id))
            if time_source is not None:
                updates.append("time_source = ?")
                vals.append(str(time_source))
            if replay is not None:
                updates.append("replay = ?")
                vals.append(replay_int)

            if updates:
                sql = "UPDATE scans SET " + ", ".join(updates) + " WHERE scan_id = ?;"
                vals.append(scan_id)
                conn.execute(sql, tuple(vals))

    def update_scan_progress(
        self,
        scan_id: str,
        frames_seen: int,
        frames_evaluated: int,
        tiles_classified: int,
        status: str = "running",
    ) -> None:
        """Incrementally updates scan row metrics during an ongoing scan."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE scans SET
                    frames_captured = ?,
                    frames_evaluated = ?,
                    tiles_classified = ?,
                    status = ?
                WHERE scan_id = ?;
                """,
                (
                    int(frames_seen),
                    int(frames_evaluated),
                    int(tiles_classified),
                    str(status),
                    scan_id,
                ),
            )

    def record_scan_end(
        self,
        scan_id: str,
        ended_utc: Optional[str] = None,
        frames_captured: int = 0,
        frames_evaluated: int = 0,
        tiles_classified: int = 0,
        distance_walked_m: Optional[float] = None,
        distance_reason: Optional[str] = None,
        status: str = "complete",
        stop_reason: Optional[str] = "user",
        duration_s: Optional[float] = None,
        replay_loops: int = 0,
    ) -> None:
        """Records completion metrics for a scanning session."""
        ended_utc = ended_utc or get_utc_iso_now()
        if distance_walked_m is None and distance_reason is None:
            distance_reason = "GPS_TRACK_NOT_RECORDED"

        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE scans SET
                    ended_utc = ?,
                    frames_captured = ?,
                    frames_evaluated = ?,
                    tiles_classified = ?,
                    distance_walked_m = ?,
                    distance_reason = ?,
                    status = ?,
                    stop_reason = ?,
                    duration_s = ?,
                    replay_loops = ?
                WHERE scan_id = ?;
                """,
                (
                    ended_utc,
                    int(frames_captured),
                    int(frames_evaluated),
                    int(tiles_classified),
                    distance_walked_m,
                    distance_reason,
                    str(status),
                    str(stop_reason) if stop_reason else None,
                    float(duration_s) if duration_s is not None else None,
                    int(replay_loops),
                    scan_id,
                ),
            )

    # -------------------------------------------------------------------------
    # Event & Verdict Recording
    # -------------------------------------------------------------------------

    def record_frame_event(
        self,
        scan_id: str,
        frame_idx: int,
        timestamp_utc: str,
        cell_id: str,
        gate_passed: bool,
        gate_metrics: Dict[str, Any],
        n_valid_tiles: int,
        frame_state: str,
        class_id: Optional[int],
        confidence: float,
        tile_decisions: List[Dict[str, Any]],
        source_image: Optional[str] = None,
        gps: Optional[Dict[str, Any]] = None,
        canopy_cover: Optional[float] = None,
        vari: Optional[float] = None,
        exg: Optional[float] = None,
        tgi: Optional[float] = None,
        dgci: Optional[float] = None,
        dgci_ood_frac: Optional[float] = None,
        indices_status: Optional[str] = None,
        indices: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Inserts a single evaluated frame event into SQLite."""
        conn = self._get_connection()
        class_name = CLASS_NAMES[class_id] if class_id is not None and 0 <= class_id < len(CLASS_NAMES) else None
        gps = gps or {}

        if indices:
            if canopy_cover is None:
                canopy_cover = indices.get("canopy_cover")
            if vari is None:
                vari = indices.get("vari")
            if exg is None:
                exg = indices.get("exg")
            if tgi is None:
                tgi = indices.get("tgi")
            if dgci is None:
                dgci = indices.get("dgci")
            if dgci_ood_frac is None:
                dgci_ood_frac = indices.get("dgci_ood_frac")
            if indices_status is None:
                indices_status = indices.get("status")

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO frame_events (
                    scan_id, frame_idx, timestamp_utc, source_image, cell_id,
                    gate_passed, blur_score, dark_fraction, bright_fraction, displacement,
                    canopy_cover, vari, exg, tgi, dgci, dgci_ood_frac, indices_status,
                    n_valid_tiles, frame_state, class_id, class_name, confidence,
                    lat, lon, gps_fix_quality, gps_hdop,
                    tile_decisions_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    scan_id,
                    int(frame_idx),
                    str(timestamp_utc),
                    source_image,
                    str(cell_id),
                    1 if gate_passed else 0,
                    gate_metrics.get("blur_score"),
                    gate_metrics.get("dark_fraction"),
                    gate_metrics.get("bright_fraction"),
                    gate_metrics.get("displacement"),
                    float(canopy_cover) if canopy_cover is not None else None,
                    float(vari) if vari is not None else None,
                    float(exg) if exg is not None else None,
                    float(tgi) if tgi is not None else None,
                    float(dgci) if dgci is not None else None,
                    float(dgci_ood_frac) if dgci_ood_frac is not None else None,
                    str(indices_status) if indices_status is not None else None,
                    int(n_valid_tiles),
                    str(frame_state),
                    int(class_id) if class_id is not None else None,
                    class_name,
                    float(confidence),
                    gps.get("latitude") or gps.get("lat"),
                    gps.get("longitude") or gps.get("lon"),
                    gps.get("fix_quality"),
                    gps.get("hdop"),
                    json.dumps(tile_decisions),
                    get_utc_iso_now(),
                ),
            )
            return cursor.lastrowid

    def record_frame_events_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> int:
        """Batch inserts multiple evaluated frame events in a single transaction."""
        if not events:
            return 0
        conn = self._get_connection()
        now_utc = get_utc_iso_now()
        rows = []
        for e in events:
            class_id = e.get("class_id")
            class_name = CLASS_NAMES[class_id] if class_id is not None and 0 <= class_id < len(CLASS_NAMES) else None
            gps = e.get("gps") or {}
            gate_metrics = e.get("gate_metrics") or {}
            ind = e.get("indices") or {}
            rows.append((
                e["scan_id"],
                int(e["frame_idx"]),
                str(e["timestamp_utc"]),
                e.get("source_image"),
                str(e["cell_id"]),
                1 if e.get("gate_passed", True) else 0,
                gate_metrics.get("blur_score"),
                gate_metrics.get("dark_fraction"),
                gate_metrics.get("bright_fraction"),
                gate_metrics.get("displacement"),
                e.get("canopy_cover", ind.get("canopy_cover")),
                e.get("vari", ind.get("vari")),
                e.get("exg", ind.get("exg")),
                e.get("tgi", ind.get("tgi")),
                e.get("dgci", ind.get("dgci")),
                e.get("dgci_ood_frac", ind.get("dgci_ood_frac")),
                e.get("indices_status", ind.get("status")),
                int(e.get("n_valid_tiles", 9)),
                str(e["frame_state"]),
                int(class_id) if class_id is not None else None,
                class_name,
                float(e.get("confidence", 0.0)),
                gps.get("latitude") or gps.get("lat"),
                gps.get("longitude") or gps.get("lon"),
                gps.get("fix_quality"),
                gps.get("hdop"),
                json.dumps(e.get("tile_decisions", [])),
                now_utc,
            ))
        with conn:
            conn.executemany(
                """
                INSERT INTO frame_events (
                    scan_id, frame_idx, timestamp_utc, source_image, cell_id,
                    gate_passed, blur_score, dark_fraction, bright_fraction, displacement,
                    canopy_cover, vari, exg, tgi, dgci, dgci_ood_frac, indices_status,
                    n_valid_tiles, frame_state, class_id, class_name, confidence,
                    lat, lon, gps_fix_quality, gps_hdop,
                    tile_decisions_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
        return len(rows)

    def record_cell_verdict(
        self,
        scan_id: str,
        cell_id: str,
        state: str,
        class_id: Optional[int],
        score: float,
        n_frames: int,
        n_agree: int,
        canopy_cover: Optional[float] = None,
    ) -> None:
        """Inserts or updates a cell-level spatial verdict for a scan."""
        conn = self._get_connection()
        class_name = CLASS_NAMES[class_id] if class_id is not None and 0 <= class_id < len(CLASS_NAMES) else None

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cell_verdicts (
                    scan_id, cell_id, state, class_id, class_name,
                    score, n_frames, n_agree, canopy_cover, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    scan_id,
                    str(cell_id),
                    str(state),
                    int(class_id) if class_id is not None else None,
                    class_name,
                    float(score),
                    int(n_frames),
                    int(n_agree),
                    float(canopy_cover) if canopy_cover is not None else None,
                    get_utc_iso_now(),
                ),
            )

    # -------------------------------------------------------------------------
    # Advisory Document Generation & Assembly (§8 / §7.3)
    # -------------------------------------------------------------------------

    def create_advisory(
        self,
        scan_id: str,
        advisory_id: Optional[str] = None,
        replay: Optional[bool] = None,
        field_id: Optional[str] = None,
        days_since_planting: Optional[int] = None,
        total_cycle_days: Optional[int] = None,
        pest_data: Optional[Dict[str, Any]] = None,
        inference_backend: str = "trt",
        thermal_frame_data: Optional[Dict[str, Any]] = None,
        stop_reason: Optional[str] = None,
        crop_declared: Optional[str] = None,
        duration_s: Optional[float] = None,
        time_source: Optional[str] = None,
        mode: Optional[str] = None,
        ended_utc: Optional[str] = None,
        duration_reason: Optional[str] = None,
        replay_loops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete frozen Advisory document (schema v1.0) from recorded
        scan metadata, frame events, and cell verdicts. Writes to SQLite advisories table.
        """
        conn = self._get_connection()

        # Fetch scan metadata
        scan_row = conn.execute("SELECT * FROM scans WHERE scan_id = ?;", (scan_id,)).fetchone()
        if not scan_row:
            raise ValueError(f"Scan '{scan_id}' not found in database")

        resolved_field_id = field_id or (scan_row["field_id"] if "field_id" in scan_row.keys() and scan_row["field_id"] else "F01")
        resolved_replay = replay if replay is not None else (bool(scan_row["replay"]) if "replay" in scan_row.keys() and scan_row["replay"] is not None else False)
        resolved_mode = mode or (scan_row["mode"] if "mode" in scan_row.keys() and scan_row["mode"] else "handheld_pod")
        declared_crop = crop_declared if crop_declared is not None else (scan_row["crop"] if "crop" in scan_row.keys() and scan_row["crop"] else None)
        is_walk_mode = (resolved_mode == "walk") or (declared_crop is not None)
        resolved_replay_loops = replay_loops if replay_loops is not None else (scan_row["replay_loops"] if "replay_loops" in scan_row.keys() and scan_row["replay_loops"] is not None else 0)

        # Resolve planting date and variety cycle days from scan metadata if omitted
        if scan_row["metadata_json"]:
            try:
                meta = json.loads(scan_row["metadata_json"])
                if isinstance(meta, dict):
                    if days_since_planting is None and "days_since_planting" in meta:
                        days_since_planting = meta.get("days_since_planting")
                    if total_cycle_days is None and "total_cycle_days" in meta:
                        total_cycle_days = meta.get("total_cycle_days")
            except Exception:
                pass

        now_utc = get_utc_iso_now()
        resolved_ended_utc = ended_utc or scan_row["ended_utc"] or now_utc
        resolved_stop_reason = (stop_reason if stop_reason in ("user", "time_limit", "error", "interrupted") else (scan_row["stop_reason"] if "stop_reason" in scan_row.keys() and scan_row["stop_reason"] in ("user", "time_limit", "error", "interrupted") else None))

        resolved_duration_s = None
        resolved_duration_reason = duration_reason or (scan_row["duration_reason"] if "duration_reason" in scan_row.keys() and scan_row["duration_reason"] else None)

        if duration_s is not None:
            resolved_duration_s = float(duration_s)
        elif "duration_s" in scan_row.keys() and scan_row["duration_s"] is not None:
            resolved_duration_s = float(scan_row["duration_s"])
        else:
            s_dt = _parse_iso_timestamp(scan_row["started_utc"])
            e_dt = _parse_iso_timestamp(resolved_ended_utc)
            if s_dt and e_dt:
                diff = (e_dt - s_dt).total_seconds()
                if diff >= 0:
                    resolved_duration_s = float(round(diff, 2))
                else:
                    resolved_duration_s = None
                    if not resolved_duration_reason:
                        resolved_duration_reason = "CLOCK_RESET_AFTER_POWER_LOSS"
            else:
                resolved_duration_s = None
                if not resolved_duration_reason:
                    resolved_duration_reason = "CLOCK_RESET_AFTER_POWER_LOSS"

        if not advisory_id:
            candidate_suffix = ("_%s" % resolved_field_id) if resolved_field_id else ""
            candidate_id = "%s%s" % (now_utc, candidate_suffix)
            existing = conn.execute("SELECT 1 FROM advisories WHERE advisory_id = ?;", (candidate_id,)).fetchone()
            if existing:
                ms = datetime.datetime.now(datetime.timezone.utc).strftime("%f")[:3]
                candidate_id = "%s.%sZ%s" % (now_utc[:-1], ms, candidate_suffix)
                idx = 2
                while conn.execute("SELECT 1 FROM advisories WHERE advisory_id = ?;", (candidate_id,)).fetchone():
                    candidate_id = "%s.%sZ%s_%d" % (now_utc[:-1], ms, candidate_suffix, idx)
                    idx += 1
            advisory_id = candidate_id

        # Aggregate frame events
        events = conn.execute(
            "SELECT * FROM frame_events WHERE scan_id = ? ORDER BY frame_idx ASC;",
            (scan_id,),
        ).fetchall()

        # Aggregate cell verdicts
        cells = conn.execute(
            "SELECT * FROM cell_verdicts WHERE scan_id = ?;",
            (scan_id,),
        ).fetchall()

        # Count states across frames
        state_counts = collections.defaultdict(int)
        for e in events:
            state_counts[e["frame_state"]] += 1

        # Point-event detections list with GPS (ans_for_vitthal.md §8 S3)
        detections = []
        gps_points = 0
        crop_counts = collections.defaultdict(int)

        for e in events:
            if e["lat"] is not None and e["lon"] is not None:
                gps_points += 1

            if e["frame_state"] in ("DISEASE", "HEALTHY"):
                cname = e["class_name"] or "unknown"
                if "__" in cname:
                    crop = cname.split("__")[0]
                    crop_counts[crop] += 1

                detections.append({
                    "class": cname,
                    "confidence": round(float(e["confidence"]), 4),
                    "cross_source_reliability": get_cross_source_reliability(cname),
                    "lat": e["lat"],
                    "lon": e["lon"],
                    "fix_quality": e["gps_fix_quality"],
                    "hdop": e["gps_hdop"],
                    "captured_utc": e["timestamp_utc"],
                    "source": "measured",
                })

        # Overall dominant crop derivation
        dominant_crop = None
        crop_health_reason = None
        crop_source = None

        if is_walk_mode and declared_crop:
            dominant_crop = declared_crop
            crop_source = "declared"

        max_agree = max([c["n_agree"] for c in cells] or [0])
        cell_states = set(c["state"] for c in cells)
        overall_state = "NO_DATA"

        if not events:
            overall_state = "NO_DATA"
            crop_health_reason = "NO_USABLE_FRAMES"
        else:
            if is_walk_mode and declared_crop:
                healthy_set = HEALTHY_CLASSES_BY_CROP.get(declared_crop, set())
                disease_events = []
                healthy_events = []
                not_crop_events = []
                uncertain_events = []
                for e in events:
                    cname = e["class_name"] or ""
                    f_state = e["frame_state"]
                    if cname == "not_crop" or f_state == "NOT_CROP":
                        not_crop_events.append(e)
                    elif cname.startswith(declared_crop + "__"):
                        if cname in healthy_set or f_state == "HEALTHY":
                            healthy_events.append(e)
                        else:
                            disease_events.append(e)
                    else:
                        uncertain_events.append(e)

                if cells:
                    if "DISEASE" in cell_states and len(disease_events) > 0:
                        overall_state = "DISEASE"
                    elif "HEALTHY" in cell_states:
                        overall_state = "HEALTHY"
                    elif len(not_crop_events) == len(events):
                        overall_state = "NOT_CROP"
                    else:
                        overall_state = "UNCERTAIN"
                        if not crop_health_reason:
                            crop_health_reason = "HIGH_UNCERTAINTY"
                else:
                    if len(disease_events) > 0:
                        overall_state = "DISEASE"
                    elif len(healthy_events) > 0:
                        overall_state = "HEALTHY"
                    elif len(not_crop_events) == len(events):
                        overall_state = "NOT_CROP"
                    else:
                        overall_state = "UNCERTAIN"
                        if not crop_health_reason:
                            crop_health_reason = "HIGH_UNCERTAINTY"
            else:
                if crop_counts:
                    total_crop_detections = sum(crop_counts.values())
                    sorted_crops = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)
                    top_crop, top_count = sorted_crops[0]

                    if len(sorted_crops) == 1:
                        dominant_crop = top_crop
                    else:
                        if (top_count / float(total_crop_detections)) >= 0.85:
                            dominant_crop = top_crop
                        else:
                            dominant_crop = None
                            crop_health_reason = "MULTIPLE_CROPS_DETECTED"

                if cells:
                    if "DISEASE" in cell_states:
                        overall_state = "DISEASE"
                    elif "HEALTHY" in cell_states:
                        overall_state = "HEALTHY"
                    elif state_counts["NOT_CROP"] == len(events):
                        overall_state = "NOT_CROP"
                    else:
                        overall_state = "UNCERTAIN"
                        if not crop_health_reason:
                            if max_agree == 0 and (state_counts["DISEASE"] > 0 or state_counts["HEALTHY"] > 0):
                                crop_health_reason = "UNCONFIRMED_DETECTIONS"
                            else:
                                crop_health_reason = "HIGH_UNCERTAINTY"
                else:
                    if state_counts["DISEASE"] > 0:
                        overall_state = "DISEASE"
                    elif state_counts["HEALTHY"] > 0:
                        overall_state = "HEALTHY"
                    elif state_counts["NOT_CROP"] == len(events):
                        overall_state = "NOT_CROP"
                    else:
                        overall_state = "UNCERTAIN"
                        if not crop_health_reason:
                            crop_health_reason = "HIGH_UNCERTAINTY"

        # Actions block: Evaluated deterministically by Step 27 (edge/rules_engine.py)
        from edge.rules_engine import evaluate_rules
        actions = evaluate_rules(
            crop=dominant_crop,
            state=overall_state,
            reason=crop_health_reason,
            detections=detections,
            cells=cells,
            events=events,
        )

        # Vegetation block (§7.3 reconciled with ans_for_vitthal.md §B6, §B7)
        canopy_covers = [float(e["canopy_cover"]) for e in events if e["canopy_cover"] is not None]
        varis = [float(e["vari"]) for e in events if e["vari"] is not None]
        exgs = [float(e["exg"]) for e in events if e["exg"] is not None]
        tgis = [float(e["tgi"]) for e in events if e["tgi"] is not None]
        dgcis = [float(e["dgci"]) for e in events if e["dgci"] is not None]
        dgci_oods = [float(e["dgci_ood_frac"]) for e in events if e["dgci_ood_frac"] is not None]

        if not canopy_covers:
            mean_canopy = 0.0
            canopy_cover_block = {
                "mean": None,
                "p10": None,
                "p50": None,
                "p90": None,
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "status": "NO_DATA",
                "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = %d" % train_config.PROVISIONAL_EXG_VEG_THRESHOLD,
                "threshold_confirmed": False,
                "source": "measured",
            }
        else:
            mean_canopy = float(np.mean(canopy_covers))
            p10_canopy = float(np.percentile(canopy_covers, 10))
            p50_canopy = float(np.percentile(canopy_covers, 50))
            p90_canopy = float(np.percentile(canopy_covers, 90))

            canopy_cover_block = {
                "mean": round(mean_canopy, 4),
                "p10": round(p10_canopy, 4),
                "p50": round(p50_canopy, 4),
                "p90": round(p90_canopy, 4),
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "status": "OK" if mean_canopy >= train_config.PROVISIONAL_MIN_CANOPY_FRACTION else "INSUFFICIENT_CANOPY",
                "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = %d" % train_config.PROVISIONAL_EXG_VEG_THRESHOLD,
                "threshold_confirmed": False,
                "source": "measured",
            }

        if (not canopy_covers) or (mean_canopy < train_config.PROVISIONAL_MIN_CANOPY_FRACTION) or not varis:
            vari_block = {
                "mean": None,
                "reason": "INSUFFICIENT_CANOPY_FRACTION",
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "threshold_source": "PROVISIONAL — within-scan relative, minimum canopy floor",
                "threshold_confirmed": False,
                "source": "measured",
            }
            exg_block = {
                "mean": None,
                "reason": "INSUFFICIENT_CANOPY_FRACTION",
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = %d" % train_config.PROVISIONAL_EXG_VEG_THRESHOLD,
                "threshold_confirmed": False,
                "source": "measured",
            }
            tgi_block = {
                "mean": None,
                "reason": "INSUFFICIENT_CANOPY_FRACTION",
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "threshold_source": "PROVISIONAL — uncalibrated RGB TGI regression",
                "threshold_confirmed": False,
                "source": "measured",
            }
            dgci_block = {
                "mean": None,
                "reason": "INSUFFICIENT_CANOPY_FRACTION",
                "min_fraction_threshold": train_config.PROVISIONAL_MIN_CANOPY_FRACTION,
                "threshold_source": "PROVISIONAL — narrowed foliage domain [60, 120] deg",
                "threshold_confirmed": False,
                "source": "measured",
            }
        else:
            vari_mean = float(np.mean(varis))
            vari_p10 = float(np.percentile(varis, 10))
            vari_p25 = float(np.percentile(varis, 25))
            vari_p50 = float(np.percentile(varis, 50))
            vari_p75 = float(np.percentile(varis, 75))
            vari_p90 = float(np.percentile(varis, 90))

            if vari_mean < vari_p10:
                band = "LOWER_TAIL"
            elif vari_mean < vari_p25:
                band = "BELOW_TYPICAL"
            elif vari_mean <= vari_p75:
                band = "TYPICAL"
            else:
                band = "ABOVE_TYPICAL"

            vari_block = {
                "mean": round(vari_mean, 4),
                "p10": round(vari_p10, 4),
                "p50": round(vari_p50, 4),
                "p90": round(vari_p90, 4),
                "field_median": round(vari_p50, 4),
                "band": band,
                "band_basis": "within_scan_percentile",
                "threshold_source": "PROVISIONAL — within-scan relative, no published absolute band exists for uncalibrated RGB VARI",
                "threshold_confirmed": False,
                "source": "measured",
            }

            exg_block = {
                "mean": round(float(np.mean(exgs)), 2),
                "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = %d" % train_config.PROVISIONAL_EXG_VEG_THRESHOLD,
                "threshold_confirmed": False,
                "source": "measured",
            }

            tgi_block = {
                "mean": round(float(np.mean(tgis)), 2),
                "threshold_source": "PROVISIONAL — uncalibrated RGB TGI regression",
                "threshold_confirmed": False,
                "source": "measured",
            }

            mean_ood = float(np.mean(dgci_oods)) if dgci_oods else 0.0
            if mean_ood > train_config.PROVISIONAL_DGCI_MAX_OOD_FRACTION:
                dgci_block = {
                    "mean": None,
                    "out_of_domain_fraction": round(mean_ood, 4),
                    "reason": "OUT_OF_DOMAIN_FRACTION_EXCEEDED",
                    "threshold": train_config.PROVISIONAL_DGCI_MAX_OOD_FRACTION,
                    "threshold_source": "PROVISIONAL — engineering judgement, not measured",
                    "threshold_confirmed": False,
                    "source": "measured",
                }
            else:
                dgci_block = {
                    "mean": round(float(np.mean(dgcis)), 4),
                    "out_of_domain_fraction": round(mean_ood, 4),
                    "threshold_source": "PROVISIONAL — narrowed foliage domain [60, 120] deg",
                    "threshold_confirmed": False,
                    "source": "measured",
                }

        vegetation = {
            "interpretation_mode": "relative",
            "canopy_cover": canopy_cover_block,
            "vari": vari_block,
            "exg": exg_block,
            "tgi": tgi_block,
            "dgci": dgci_block,
            "ndvi": None,
            "ndvi_status": "PENDING_HARDWARE_FINALIZATION",
            "ndvi_reason": "Optical path and calib_matrix in progress. Field reserved; not estimated.",
        }

        # Growth Stage block: Deterministic lookup via core/growth_stage.py (Step 28 / FAO-56)
        from core.growth_stage import estimate_growth_stage
        growth_stage = estimate_growth_stage(
            crop=dominant_crop,
            days_since_planting=days_since_planting,
            canopy_cover=mean_canopy,
            total_cycle_days=total_cycle_days,
        )

        # J5 / L4 / M2 / M4 Hardware-Gated Blocks: Thermal, NDVI, Satellite NDVI, Irrigation
        from edge.sensors import MastTelemetryReader
        from edge.hardware_detect import detect_mlx90640, detect_noir_camera
        reader = MastTelemetryReader(self)
        mast_reading = reader.get_latest()

        # 1. Thermal block (MLX90640 runtime I2C probe + Reference CWSI M2.3/M2.5)
        if replay:
            thermal_block = {
                "available": False,
                "reason": "REPLAY_THERMAL_NOT_OF_SCENE",
                "tc_c": None,
                "twet_c": None,
                "tdry_c": None,
                "cwsi": None,
                "flag": None,
                "thermal_source": str(thermal_frame_data.get("thermal_source", "hardware")) if thermal_frame_data else "hardware",
                "frame_utc": None,
            }
        elif thermal_frame_data and thermal_frame_data.get("available") and thermal_frame_data.get("temperature_array") is not None:
            from edge.thermal_capture import load_thermal_refs
            from core.thermal import evaluate_reference_cwsi_from_frame
            refs_cfg = load_thermal_refs()
            cwsi_eval = evaluate_reference_cwsi_from_frame(thermal_frame_data["temperature_array"], refs_cfg)
            thermal_block = {
                "available": cwsi_eval["available"],
                "reason": cwsi_eval["reason"],
                "tc_c": cwsi_eval["tc_c"],
                "twet_c": cwsi_eval["twet_c"],
                "tdry_c": cwsi_eval["tdry_c"],
                "cwsi": cwsi_eval["cwsi"],
                "flag": cwsi_eval["flag"],
                "thermal_source": str(thermal_frame_data.get("thermal_source", "hardware")),
                "frame_utc": thermal_frame_data.get("timestamp_utc", scan_row["started_utc"]),
            }
        else:
            mlx_ok, mlx_reason = detect_mlx90640()
            if mlx_ok:
                try:
                    from edge.thermal_capture import MLX90640, load_thermal_refs
                    from core.thermal import evaluate_reference_cwsi_from_frame
                    sensor = MLX90640(mock=False)
                    tf = sensor.capture_frame()
                    if tf.get("available") and tf.get("temperature_array") is not None:
                        refs_cfg = load_thermal_refs()
                        cwsi_eval = evaluate_reference_cwsi_from_frame(tf["temperature_array"], refs_cfg)
                        thermal_block = {
                            "available": cwsi_eval["available"],
                            "reason": cwsi_eval["reason"],
                            "tc_c": cwsi_eval["tc_c"],
                            "twet_c": cwsi_eval["twet_c"],
                            "tdry_c": cwsi_eval["tdry_c"],
                            "cwsi": cwsi_eval["cwsi"],
                            "flag": cwsi_eval["flag"],
                            "thermal_source": "hardware",
                            "frame_utc": tf.get("timestamp_utc", scan_row["started_utc"]),
                        }
                    else:
                        from edge.thermal_capture import load_thermal_refs
                        refs_cfg = load_thermal_refs()
                        reason = "THERMAL_REFS_NOT_CONFIGURED" if refs_cfg.get("status") != "MEASURED" else tf.get("reason", "NO_THERMAL_FRAME_IN_SCAN")
                        thermal_block = {
                            "available": False,
                            "reason": reason,
                            "tc_c": None,
                            "twet_c": None,
                            "tdry_c": None,
                            "cwsi": None,
                            "flag": None,
                            "thermal_source": "hardware",
                            "frame_utc": None,
                        }
                except Exception as ex:
                    from edge.thermal_capture import load_thermal_refs
                    refs_cfg = load_thermal_refs()
                    reason = "THERMAL_REFS_NOT_CONFIGURED" if refs_cfg.get("status") != "MEASURED" else "HARDWARE_CAPTURE_FAILED: %s" % ex
                    thermal_block = {
                        "available": False,
                        "reason": reason,
                        "tc_c": None,
                        "twet_c": None,
                        "tdry_c": None,
                        "cwsi": None,
                        "flag": None,
                        "thermal_source": "hardware",
                        "frame_utc": None,
                    }
            else:
                thermal_block = {
                    "available": False,
                    "reason": mlx_reason,
                    "tc_c": None,
                    "twet_c": None,
                    "tdry_c": None,
                    "cwsi": None,
                    "flag": None,
                    "thermal_source": "hardware",
                    "frame_utc": None,
                }

        # 2. NDVI block (IMX219-77IR NoIR camera CSI probe)
        noir_ok, noir_reason = detect_noir_camera()
        ndvi_block = {
            "available": noir_ok,
            "reason": noir_reason,
        }

        # 3. Sentinel-2 Satellite NDVI block (M4.6)
        sat_record = self.get_latest_satellite_ndvi(field_id=resolved_field_id)
        if sat_record:
            ndvi_satellite_block = {
                "available": True,
                "reason": None,
                "source": "SENTINEL2_L2A_CDSE",
                "scene_date": sat_record["scene_date"],
                "age_days": sat_record.get("age_days"),
                "ndvi_mean": sat_record["ndvi_mean"],
                "ndvi_std": sat_record["ndvi_std"],
                "valid_pixel_count": sat_record["valid_pixel_count"],
                "cloud_masked_fraction": sat_record["cloud_masked_fraction"],
                "pixel_size_m": 10,
                "reliability_note": sat_record.get("reliability_note"),
            }
        else:
            ndvi_satellite_block = {
                "available": False,
                "reason": "NO_SATELLITE_DATA_RECORDED",
                "source": "SENTINEL2_L2A_CDSE",
                "scene_date": None,
                "age_days": None,
                "ndvi_mean": None,
                "ndvi_std": None,
                "valid_pixel_count": None,
                "cloud_masked_fraction": None,
                "pixel_size_m": 10,
                "reliability_note": None,
            }

        # 3. Irrigation block (FAO-56 Hargreaves-Samani, Eq 52 & Dynamic Ra Eqs 21-25)
        history_24h = self.get_mast_readings_history(hours=24.0)
        valid_clock_readings = []
        invalid_clock_count = 0
        for r in history_24h:
            if r.get("rtc_valid") and r.get("utc"):
                dt = _parse_iso_timestamp(r["utc"])
                if dt is not None:
                    valid_clock_readings.append((dt, r))
                else:
                    invalid_clock_count += 1
            else:
                invalid_clock_count += 1

        # Sort valid readings chronologically by their valid UTC timestamp
        valid_clock_readings.sort(key=lambda x: x[0])
        valid_temps = [float(r["air_temp_c"]) for dt, r in valid_clock_readings if r.get("air_temp_c") is not None]

        # Minimum coverage rule:
        # At least 6 samples spanning >= 6.0 hours (to capture diurnal day/night spread)
        has_min_coverage = False
        time_span_h = 0.0
        if len(valid_clock_readings) >= 2:
            dt0 = valid_clock_readings[0][0]
            dt1 = valid_clock_readings[-1][0]
            time_span_h = abs((dt1 - dt0).total_seconds()) / 3600.0

        if len(valid_temps) >= 6 and time_span_h >= 6.0:
            has_min_coverage = True

        if mast_reading is not None and mast_reading.get("air_temp_c") is not None:
            if not has_min_coverage:
                excl_suffix = ""
                if invalid_clock_count > 0:
                    excl_suffix = ", %d reading%s excluded for invalid clock" % (
                        invalid_clock_count,
                        "s" if invalid_clock_count != 1 else "",
                    )
                irrigation_block = {
                    "available": False,
                    "reason": "INSUFFICIENT_24H_HISTORY (need >=6 readings spanning >=6h in last 24h for Tmin/Tmax, found %d readings spanning %.1fh%s)" % (
                        len(valid_temps),
                        time_span_h,
                        excl_suffix,
                    ),
                }
            else:
                t_min = min(valid_temps)
                t_max = max(valid_temps)
                t_mean = sum(valid_temps) / float(len(valid_temps))
                delta_t = max(0.0, t_max - t_min)
                kc = float(growth_stage.get("kc") if growth_stage and growth_stage.get("kc") is not None else 1.0)

                # Day of year J
                scan_dt = _parse_iso_timestamp(scan_row["started_utc"]) if scan_row["started_utc"] else None
                if not scan_dt:
                    scan_dt = datetime.datetime.now(datetime.timezone.utc)
                day_of_year = scan_dt.timetuple().tm_yday

                # Latitude resolution: GPS first, then config fallback
                gps_lats = [float(e["lat"]) for e in events if e["lat"] is not None]
                if gps_lats:
                    resolved_lat = float(gps_lats[0])
                    ra_source = "GPS"
                else:
                    config_lat = None
                    if scan_row["metadata_json"]:
                        try:
                            s_meta = json.loads(scan_row["metadata_json"])
                            if isinstance(s_meta, dict) and "latitude" in s_meta:
                                config_lat = float(s_meta["latitude"])
                        except Exception:
                            pass
                    if config_lat is None:
                        config_lat = 28.5  # default field config latitude
                    resolved_lat = float(config_lat)
                    ra_source = "CONFIG_LATITUDE"

                # Dynamic FAO-56 Extraterrestrial Radiation (Eqs. 21-25)
                from edge.irrigation_model import calculate_extraterrestrial_radiation_fao56
                ra_dict = calculate_extraterrestrial_radiation_fao56(day_of_year, resolved_lat)
                ra_mm_day = ra_dict["ra_mm_day"]
                ra_mj_m2_day = ra_dict["ra_mj_m2_day"]

                et0_est = round(0.0023 * (t_mean + 17.8) * (delta_t ** 0.5) * ra_mm_day, 2)
                et0_est = max(0.0, min(15.0, et0_est))
                crop_et = round(et0_est * kc, 2)

                irrigation_block = {
                    "available": True,
                    "method": "fao56_hargreaves_samani",
                    "air_temp_c": mast_reading["air_temp_c"],
                    "rh_pct": mast_reading.get("rh_pct"),
                    "t_min_24h_c": round(t_min, 2),
                    "t_max_24h_c": round(t_max, 2),
                    "t_mean_24h_c": round(t_mean, 2),
                    "ra_mm_day": ra_mm_day,
                    "ra_mj_m2_day": ra_mj_m2_day,
                    "ra_source": ra_source,
                    "ra_latitude_deg": resolved_lat,
                    "day_of_year": day_of_year,
                    "et0_mm_day": et0_est,
                    "kc": kc,
                    "crop_et_mm_day": crop_et,
                    "soil1_v": mast_reading.get("soil1_v"),
                    "soil2_v": mast_reading.get("soil2_v"),
                    "battery_v": mast_reading.get("battery_v"),
                    "samples_24h": len(valid_temps),
                    "source": "derived_fao56",
                }
        else:
            irrigation_block = {
                "available": False,
                "reason": "HARDWARE_NOT_CONNECTED (Ground mast SHT40 weather telemetry absent or stale)",
            }

        # Pest block: Explicitly passed or resolved from latest trap record
        resolved_pest = pest_data
        if resolved_pest is None:
            latest_trap = self.get_latest_trap_record()
            if latest_trap and "pest" in latest_trap:
                resolved_pest = latest_trap["pest"]
        if resolved_pest is None:
            resolved_pest = []

        # Construct stretches, summary, alerts
        if is_walk_mode:
            start_dt = _parse_iso_timestamp(scan_row["started_utc"])
            if events:
                first_event_dt = _parse_iso_timestamp(events[0]["timestamp_utc"])
                if first_event_dt and (not start_dt or first_event_dt < start_dt):
                    start_dt = first_event_dt
            start_ts = start_dt.timestamp() if start_dt else 0.0
            dur = resolved_duration_s if resolved_duration_s is not None else 0.0
            if dur > 0:
                num_stretches = max(1, min(int(math.ceil(dur / 20.0)), 90))
            elif len(events) > 0:
                last_dt = _parse_iso_timestamp(events[-1]["timestamp_utc"])
                last_ts = last_dt.timestamp() if last_dt else start_ts
                num_stretches = max(1, min(int(math.ceil(max(1.0, last_ts - start_ts) / 20.0)), 90))
            else:
                num_stretches = 1

            stretches = []
            healthy_set = HEALTHY_CLASSES_BY_CROP.get(declared_crop, set()) if declared_crop else set()
            for i in range(num_stretches):
                s_start_ts = start_ts + 20.0 * i
                s_end_ts = start_ts + 20.0 * (i + 1)
                s_start_utc = datetime.datetime.fromtimestamp(s_start_ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                s_end_utc = datetime.datetime.fromtimestamp(s_end_ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                s_events = []
                for e in events:
                    e_dt = _parse_iso_timestamp(e["timestamp_utc"])
                    e_ts = e_dt.timestamp() if e_dt else 0.0
                    if i == num_stretches - 1:
                        if s_start_ts <= e_ts:
                            s_events.append(e)
                    else:
                        if s_start_ts <= e_ts < s_end_ts:
                            s_events.append(e)

                if len(s_events) == 0:
                    s_verdict = "NO_DATA"
                    s_top_class = None
                    s_agreeing = 0
                else:
                    nc_cnt = sum(1 for e in s_events if (e["class_name"] == "not_crop" or e["frame_state"] == "NOT_CROP"))
                    if nc_cnt / float(len(s_events)) >= 0.80:
                        s_verdict = "NOT_CROP"
                        s_top_class = None
                        s_agreeing = nc_cnt
                    else:
                        dis_counts = collections.defaultdict(int)
                        hl_cnt = 0
                        for e in s_events:
                            cname = e["class_name"] or ""
                            if declared_crop:
                                if not cname.startswith(declared_crop + "__"):
                                    continue
                                if cname in healthy_set or e["frame_state"] == "HEALTHY":
                                    hl_cnt += 1
                                elif cname != "not_crop":
                                    dis_counts[cname] += 1
                            else:
                                if "normal" in cname or "healthy" in cname or "dried" in cname or e["frame_state"] == "HEALTHY":
                                    hl_cnt += 1
                                elif cname != "not_crop":
                                    dis_counts[cname] += 1

                        s_verdict = None
                        s_top_class = None
                        s_agreeing = 0
                        if dis_counts:
                            top_d, top_c = max(dis_counts.items(), key=lambda x: x[1])
                            if top_c >= 3:
                                s_verdict = "DISEASE"
                                s_top_class = top_d
                                s_agreeing = top_c
                        if s_verdict is None:
                            if hl_cnt >= 3:
                                s_verdict = "HEALTHY"
                                s_top_class = None
                                s_agreeing = hl_cnt
                            else:
                                s_verdict = "UNCERTAIN"
                                s_top_class = None
                                s_agreeing = 0

                stretches.append({
                    "index": i,
                    "start_utc": s_start_utc,
                    "end_utc": s_end_utc,
                    "frames_used": len(s_events),
                    "verdict": s_verdict,
                    "top_class": s_top_class,
                    "frames_agreeing": s_agreeing,
                    "thermal_median_c": None,
                    "lat": None,
                    "lon": None,
                    "pos_accuracy_m": None,
                    "pos_source": None,
                })

            summary = {
                "stretches_total": len(stretches),
                "healthy": sum(1 for s in stretches if s["verdict"] == "HEALTHY"),
                "need_look": sum(1 for s in stretches if s["verdict"] == "DISEASE"),
                "unclear": sum(1 for s in stretches if s["verdict"] == "UNCERTAIN"),
                "not_crop": sum(1 for s in stretches if s["verdict"] == "NOT_CROP"),
                "no_data": sum(1 for s in stretches if s["verdict"] == "NO_DATA"),
            }

            all_alerts = []
            last_alert_time = {}
            parsed_events = []
            for e in events:
                dt = _parse_iso_timestamp(e["timestamp_utc"])
                t = dt.timestamp() if dt else 0.0
                parsed_events.append((t, e))
            parsed_events.sort(key=lambda x: x[0])

            for t_curr, e in parsed_events:
                cname = e["class_name"] or ""
                if declared_crop:
                    if not cname.startswith(declared_crop + "__"):
                        continue
                    if cname in healthy_set:
                        continue
                else:
                    if "healthy" in cname or "normal" in cname or "dried" in cname or cname == "not_crop":
                        continue
                if cname:
                    agreeing = [ev for t_prev, ev in parsed_events if (t_curr - 6.0 <= t_prev <= t_curr and ev["class_name"] == cname)]
                    if len(agreeing) >= 3:
                        if t_curr - last_alert_time.get(cname, -999.0) >= 15.0:
                            last_alert_time[cname] = t_curr
                            alert_id = len(all_alerts) + 1
                            all_alerts.append({
                                "alert_id": alert_id,
                                "utc": e["timestamp_utc"],
                                "class": cname,
                                "frames_agreeing": len(agreeing),
                                "lat": e["lat"],
                                "lon": e["lon"],
                                "pos_accuracy_m": None,
                            })
            alerts = all_alerts
        else:
            stretches = []
            summary = {
                "stretches_total": 0,
                "healthy": 0,
                "need_look": 0,
                "unclear": 0,
                "not_crop": 0,
                "no_data": 0,
            }
            alerts = []

        if overall_state == "NO_DATA" and not crop_health_reason:
            crop_health_reason = "NO_USABLE_FRAMES"

        crop_health_block = {
            "state": overall_state,
            "reason": crop_health_reason if (overall_state not in ("HEALTHY", "DISEASE") or crop_health_reason) else None,
            "crop": dominant_crop,
            "frames_evaluated": len(events),
            "frames_agreeing": max_agree if len(events) > 0 else 0,
            "frames_rejected_ood": state_counts["UNCERTAIN"] if len(events) > 0 else 0,
            "frames_rejected_not_crop": state_counts["NOT_CROP"] if len(events) > 0 else 0,
            "frames_uncertain": state_counts["UNCERTAIN"] if len(events) > 0 else 0,
            "source": "measured",
        }
        if crop_source:
            crop_health_block["crop_source"] = crop_source

        scan_payload = {
            "started_utc": scan_row["started_utc"],
            "ended_utc": resolved_ended_utc,
            "mode": resolved_mode,
            "crop_declared": declared_crop,
            "duration_s": resolved_duration_s,
            "duration_reason": resolved_duration_reason,
            "stop_reason": resolved_stop_reason,
            "frames_captured": int(scan_row["frames_captured"] or len(events)),
            "frames_evaluated": len(events),
            "tiles_classified": int(scan_row["tiles_classified"] or len(events) * train_config.N_TILES),
            "distance_walked_m": scan_row["distance_walked_m"],
            "distance_reason": scan_row["distance_reason"] or ("GPS_TRACK_NOT_RECORDED" if gps_points == 0 else None),
        }

        # Construct payload strictly according to ans_for_vitthal.md §8 and data_flow_architecture.md §7.3
        payload = {
            "schema_version": "1.0",
            "advisory_id": advisory_id,
            "seq": 0,  # initial value, updated to monotonic rowid during SQLite insert
            "generated_at_utc": now_utc,
            "inference_backend": str(inference_backend),
            "replay": bool(resolved_replay),
            "scan": scan_payload,
            "time_source": (scan_row["time_source"] if "time_source" in scan_row.keys() and scan_row["time_source"] else time_source) or getattr(self, "clock_source", "filesystem"),
            "summary": summary,
            "stretches": stretches,
            "alerts": alerts,
            "field_conditions": {
                "available": False,
                "reason": "NO_VALID_MAST_READING",
                "node_id": None,
                "reading_utc": None,
                "age_minutes": None,
                "air_temp_c": None,
                "rh_pct": None,
                "lux": None,
                "soil1_v": None,
                "soil2_v": None,
                "battery_v": None,
                "soil_units": "raw_volts_uncalibrated",
                "source": None,
            },
            "crop_health": crop_health_block,
            "growth_stage": growth_stage,
            "vegetation": vegetation,
            "thermal": thermal_block,
            "ndvi": ndvi_block,
            "ndvi_satellite": ndvi_satellite_block,
            "irrigation": irrigation_block,
            "detections": detections,
            "gps": {
                "status": "OK" if gps_points > 0 else "ABSENT",
                "point_count": gps_points,
                "accuracy_note": "Point tagging only, approximately 2.5 m CEP. Not a survey-grade position.",
            },
            "disease": [
                {
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "media_ids": [],  # Always present empty array per §8 S4
                    "source": "measured",
                }
                for d in detections if "__" in d["class"] and "normal" not in d["class"] and "healthy" not in d["class"]
            ],
            "pest": resolved_pest,
            "inputs": [
                {
                    "name": "pod_camera_rgb",
                    "source_node": "POD",
                    "status": "OK" if int(scan_row["frames_captured"] or len(events)) > 0 else "ABSENT",
                },
                {
                    "name": "pod_gps",
                    "source_node": "POD",
                    "status": "OK" if gps_points > 0 else "ABSENT",
                },
                {
                    "name": "pod_thermal",
                    "source_node": "POD",
                    "status": (
                        "MOCK_PROVISIONAL"
                        if thermal_block.get("thermal_source") == "mock"
                        else (
                            "OK"
                            if thermal_block.get("available")
                            else (
                                "PENDING_CALIBRATION"
                                if (
                                    not replay
                                    and (
                                        (thermal_frame_data and thermal_frame_data.get("available") and thermal_frame_data.get("thermal_source") == "hardware")
                                        or (thermal_block.get("tc_c") is not None)
                                        or (
                                            thermal_block.get("reason")
                                            and not str(thermal_block.get("reason")).startswith("HARDWARE_NOT_CONNECTED")
                                            and not str(thermal_block.get("reason")).startswith("HARDWARE_CAPTURE_FAILED")
                                        )
                                    )
                                )
                                else "ABSENT"
                            )
                        )
                    ),
                },
            ],
            "actions": actions,
        }

        payload_bytes = len(json.dumps(payload).encode("utf-8"))

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO advisories (
                    advisory_id, scan_id, generated_at_utc, bytes, replay, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    advisory_id,
                    scan_id,
                    now_utc,
                    payload_bytes,
                    1 if replay else 0,
                    json.dumps(payload),
                ),
            )
            assigned_seq = cursor.lastrowid
            # Update seq in payload JSON to match assigned SQLite rowid
            payload["seq"] = assigned_seq
            conn.execute(
                "UPDATE advisories SET payload_json = ? WHERE seq = ?;",
                (json.dumps(payload), assigned_seq),
            )

        return payload

    # -------------------------------------------------------------------------
    # Gateway Endpoint Query Helpers (§2)
    # -------------------------------------------------------------------------

    def get_manifest(
        self,
        since: Union[int, str] = 0,
        limit: int = 200,
        since_seq: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Serves GET /api/v1/manifest?since=<seq_or_id>&limit=<limit>.
        Orders strictly ascending by monotonic seq (SQLite rowid).
        Resolves string advisory_id cursors to seq internally for backward compatibility (§A3).
        """
        conn = self._get_connection()
        limit = min(max(int(limit), 1), 500)

        cursor_val = since_seq if since_seq is not None else since
        resolved_seq = 0
        if isinstance(cursor_val, int):
            resolved_seq = max(0, cursor_val)
        elif isinstance(cursor_val, str):
            if cursor_val.isdigit():
                resolved_seq = max(0, int(cursor_val))
            elif cursor_val.strip():
                row = conn.execute("SELECT seq FROM advisories WHERE advisory_id = ?;", (cursor_val.strip(),)).fetchone()
                if row:
                    resolved_seq = row["seq"]
                else:
                    resolved_seq = 0

        # Query 1 extra row to determine truncation
        rows = conn.execute(
            """
            SELECT advisory_id, seq, generated_at_utc, bytes, replay, payload_json
            FROM advisories
            WHERE seq > ?
            ORDER BY seq ASC
            LIMIT ?;
            """,
            (resolved_seq, limit + 1),
        ).fetchall()

        truncated = len(rows) > limit
        items = rows[:limit]

        advisories_list = []
        for r in items:
            backend = "trt"
            if r["payload_json"]:
                try:
                    p = json.loads(r["payload_json"])
                    backend = p.get("inference_backend", "trt")
                except Exception:
                    pass
            advisories_list.append({
                "advisory_id": r["advisory_id"],
                "seq": r["seq"],
                "generated_at_utc": r["generated_at_utc"],
                "bytes": r["bytes"],
                "replay": bool(r["replay"]),
                "inference_backend": backend,
            })

        return {
            "schema_version": "1.0",
            "count": len(advisories_list),
            "advisories": advisories_list,
            "truncated": truncated,
        }

    def get_advisory(self, advisory_id_or_seq: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Serves GET /api/v1/advisory/<id> or by monotonic integer seq.
        Returns parsed advisory dictionary or None if not found.
        """
        conn = self._get_connection()
        if str(advisory_id_or_seq).lower() == "latest":
            row = conn.execute(
                "SELECT payload_json FROM advisories ORDER BY seq DESC LIMIT 1;"
            ).fetchone()
        elif isinstance(advisory_id_or_seq, int) or (isinstance(advisory_id_or_seq, str) and advisory_id_or_seq.isdigit()):
            row = conn.execute(
                "SELECT payload_json FROM advisories WHERE seq = ?;",
                (int(advisory_id_or_seq),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT payload_json FROM advisories WHERE advisory_id = ?;",
                (str(advisory_id_or_seq),),
            ).fetchone()

        if not row:
            return None
        return json.loads(row["payload_json"])

    def ack_advisory(self, advisory_id: str) -> bool:
        """
        Serves POST /api/v1/ack. Marks the advisory as acked.
        Advances the unacked count displayed on /health without pruning advisory data.
        """
        conn = self._get_connection()
        now_utc = get_utc_iso_now()
        with conn:
            cursor = conn.execute(
                """
                UPDATE advisories
                SET acked = 1, acked_at_utc = ?
                WHERE advisory_id = ?;
                """,
                (now_utc, str(advisory_id)),
            )
            # Record last acked timestamp in sync state
            conn.execute(
                """
                INSERT OR REPLACE INTO app_sync_state (key, value, updated_at)
                VALUES ('last_acked_advisory', ?, ?);
                """,
                (str(advisory_id), now_utc),
            )
            return cursor.rowcount > 0

    def get_health(self) -> Dict[str, Any]:
        """
        Serves GET /api/v1/health.
        Returns device liveness, unacked advisory count, and free eMMC storage.
        """
        conn = self._get_connection()

        # Unacked count
        unacked_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM advisories WHERE acked = 0;"
        ).fetchone()
        advisory_count = unacked_row["cnt"] if unacked_row else 0

        # Latest seq
        latest_row = conn.execute("SELECT MAX(seq) AS max_seq FROM advisories;").fetchone()
        latest_seq = latest_row["max_seq"] or 0

        # Storage free KB
        stat = os.statvfs(str(self.db_path.parent))
        storage_free_kb = int((stat.f_bavail * stat.f_frsize) / 1024)
        storage_free_mb = int(storage_free_kb / 1024)

        return {
            "device": "sih-pod-01",
            "advisory_count": advisory_count,
            "storage_free_kb": storage_free_kb,
            "gps_time_valid": False,  # True once real GPS NMEA lock is confirmed
            "schema_version": "1.0",
            "latest_seq": latest_seq,
            "server_time_utc": get_utc_iso_now(),
            "clock_source": getattr(self, "clock_source", "filesystem"),
            "storage": {
                "location": getattr(self, "storage_location", "internal"),
                "free_mb": storage_free_mb,
            },
        }

    def recover_interrupted_scans(self) -> List[str]:
        """
        On gateway start: any scan with status 'running' and no live pipeline process is finalized
        into a normal advisory from its saved events, with scan.stop_reason = 'interrupted'.
        Nothing saved before the cut may be lost.
        Returns list of recovered advisory_ids.
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM scans WHERE status = 'running';").fetchall()
        recovered_ids: List[str] = []

        for row in rows:
            scan_id = row["scan_id"]
            pid = row["pid"] if "pid" in row.keys() else None
            alive = False
            if pid and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                    # Check if process is actually pipeline.py
                    try:
                        cmdline_file = Path("/proc") / str(pid) / "cmdline"
                        if cmdline_file.exists():
                            cmd_str = cmdline_file.read_text()
                            if "pipeline.py" not in cmd_str:
                                alive = False
                        else:
                            ps_out = subprocess.check_output(
                                ["ps", "-p", str(pid), "-o", "command="],
                                stderr=subprocess.DEVNULL,
                            )
                            if b"pipeline.py" not in ps_out:
                                alive = False
                    except Exception:
                        pass
                except OSError:
                    alive = False

            if alive:
                continue

            # Check if advisory already exists
            adv_row = conn.execute("SELECT advisory_id FROM advisories WHERE scan_id = ?;", (scan_id,)).fetchone()
            if adv_row:
                with conn:
                    conn.execute("UPDATE scans SET status = 'interrupted', stop_reason = 'interrupted' WHERE scan_id = ?;", (scan_id,))
                continue

            # Find last event or status timestamp
            last_event = conn.execute("SELECT timestamp_utc FROM frame_events WHERE scan_id = ? ORDER BY id DESC LIMIT 1;", (scan_id,)).fetchone()
            last_event_utc = last_event["timestamp_utc"] if last_event else None

            # Also check if status file exists and has newer timestamp or status
            status_file = self.data_dir / "status" / (scan_id + ".json")
            status_utc = None
            if status_file.exists():
                try:
                    with open(str(status_file), "r", encoding="utf-8") as sf:
                        st_data = json.load(sf)
                        st_started = st_data.get("started_utc")
                        st_elapsed = st_data.get("elapsed_s", 0)
                        if st_started and st_elapsed > 0:
                            s_dt = _parse_iso_timestamp(st_started)
                            if s_dt:
                                status_utc = datetime.datetime.fromtimestamp(s_dt.timestamp() + st_elapsed, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass

            ended_utc = None
            if last_event_utc and status_utc:
                dt_ev = _parse_iso_timestamp(last_event_utc)
                dt_st = _parse_iso_timestamp(status_utc)
                if dt_ev and dt_st:
                    ended_utc = last_event_utc if dt_ev >= dt_st else status_utc
                else:
                    ended_utc = last_event_utc
            elif last_event_utc:
                ended_utc = last_event_utc
            elif status_utc:
                ended_utc = status_utc
            else:
                ended_utc = row["started_utc"]

            dur_s = None
            dur_reason = None
            s_dt = _parse_iso_timestamp(row["started_utc"])
            e_dt = _parse_iso_timestamp(ended_utc)
            if s_dt and e_dt:
                diff = (e_dt - s_dt).total_seconds()
                if diff >= 0:
                    dur_s = float(round(diff, 2))
                else:
                    dur_s = None
                    dur_reason = "CLOCK_RESET_AFTER_POWER_LOSS"
            else:
                dur_reason = "CLOCK_RESET_AFTER_POWER_LOSS"

            try:
                is_replay = bool(row["replay"]) if ("replay" in row.keys() and row["replay"] is not None) else False
                advisory = self.create_advisory(
                    scan_id=scan_id,
                    stop_reason="interrupted",
                    replay=is_replay,
                    ended_utc=ended_utc,
                    duration_s=dur_s,
                    duration_reason=dur_reason,
                )
                with conn:
                    conn.execute(
                        "UPDATE scans SET status = 'interrupted', stop_reason = 'interrupted', ended_utc = ?, duration_s = ?, duration_reason = ? WHERE scan_id = ?;",
                        (ended_utc, dur_s, dur_reason, scan_id),
                    )
                recovered_ids.append(advisory["advisory_id"])
            except Exception:
                with conn:
                    conn.execute(
                        "UPDATE scans SET status = 'interrupted', stop_reason = 'interrupted', ended_utc = ?, duration_s = ?, duration_reason = ? WHERE scan_id = ?;",
                        (ended_utc, dur_s, dur_reason, scan_id),
                    )
        return recovered_ids

    # -------------------------------------------------------------------------
    # Retention & Disk Space Policy
    # -------------------------------------------------------------------------

    def prune_retained_data(self, max_frames: Optional[int] = None) -> int:
        """
        Prunes oldest frame events exceeding max_retained_frames.
        Preserves all advisory JSON documents permanently (they are small ~5KB).
        Performs WAL checkpoint to reclaim pages and guard against eMMC exhaustion.
        """
        max_frames = max_frames if max_frames is not None else self.max_retained_frames
        conn = self._get_connection()

        with conn:
            # Count frame events
            cnt_row = conn.execute("SELECT COUNT(*) AS total FROM frame_events;").fetchone()
            total_events = cnt_row["total"] if cnt_row else 0

            pruned = 0
            if total_events > max_frames:
                excess = total_events - max_frames
                # Find cutoff ID
                cutoff_row = conn.execute(
                    "SELECT id FROM frame_events ORDER BY id ASC LIMIT 1 OFFSET ?;",
                    (excess,),
                ).fetchone()
                if cutoff_row:
                    cutoff_id = cutoff_row["id"]
                    cursor = conn.execute(
                        "DELETE FROM frame_events WHERE id < ?;",
                        (cutoff_id,),
                    )
                    pruned = cursor.rowcount

        # WAL checkpoint truncate to ensure WAL log file doesn't grow unbounded
        # Executed outside transaction so WAL pages can be transferred and file truncated
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except sqlite3.OperationalError:
            pass

        return pruned

    # -------------------------------------------------------------------------
    # Sticky-Trap Job Storage (Model B Gateway Job, Step 31 / H5)
    # -------------------------------------------------------------------------

    def record_trap_job(
        self,
        trap_id: str,
        job_id: str,
        image_path: str,
        placed_at_utc: Optional[str],
        days_monitored: Optional[float],
        total_blobs_counted: int,
        scale_mm_per_pixel: Optional[float],
        scale_status: str,
        etl_status: str,
        pest_payload: List[Dict[str, Any]],
    ) -> int:
        """Records a completed Model B trap processing job to SQLite."""
        conn = self._get_connection()
        now_utc = get_utc_iso_now()

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO trap_records (
                    trap_id, job_id, image_path, placed_at_utc, processed_at_utc,
                    days_monitored, total_blobs_counted, scale_mm_per_pixel,
                    scale_status, etl_status, pest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(trap_id),
                    str(job_id),
                    str(image_path),
                    str(placed_at_utc) if placed_at_utc is not None else None,
                    now_utc,
                    float(days_monitored) if days_monitored is not None else None,
                    int(total_blobs_counted),
                    float(scale_mm_per_pixel) if scale_mm_per_pixel is not None else None,
                    str(scale_status),
                    str(etl_status),
                    json.dumps(pest_payload),
                ),
            )
            return cursor.lastrowid

    def get_latest_trap_record(self, trap_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches the most recent trap job record, returning parsed pest dictionary."""
        conn = self._get_connection()
        if trap_id:
            row = conn.execute(
                "SELECT * FROM trap_records WHERE trap_id = ? ORDER BY id DESC LIMIT 1;",
                (str(trap_id),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM trap_records ORDER BY id DESC LIMIT 1;"
            ).fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "trap_id": row["trap_id"],
            "job_id": row["job_id"],
            "image_path": row["image_path"],
            "placed_at_utc": row["placed_at_utc"],
            "processed_at_utc": row["processed_at_utc"],
            "days_monitored": row["days_monitored"],
            "total_blobs_counted": row["total_blobs_counted"],
            "scale_mm_per_pixel": row["scale_mm_per_pixel"],
            "scale_status": row["scale_status"],
            "etl_status": row["etl_status"],
            "pest": json.loads(row["pest_json"]),
        }

    def record_mast_reading(
        self,
        record: Dict[str, Any],
        log_epoch: int,
        received_at: Optional[str] = None,
    ) -> int:
        """Records a single ground mast reading (Guide §6) into SQLite."""
        conn = self._get_connection()
        now_utc = received_at or record.get("received_at") or record.get("utc") or get_utc_iso_now()

        seq = int(record["seq"])
        node_id = str(record.get("node_id", "N01"))
        field_id = record.get("field_id")
        rtc_valid = 1 if record.get("rtc_valid") is True else 0
        utc_val = record.get("utc")
        uptime_s = int(record["uptime_s"]) if record.get("uptime_s") is not None else None

        air_temp = float(record["air_temp_c"]) if record.get("air_temp_c") is not None else None
        rh_pct = float(record["rh_pct"]) if record.get("rh_pct") is not None else None
        ir_obj = float(record["ir_object_c"]) if record.get("ir_object_c") is not None else None
        ir_amb = float(record["ir_ambient_c"]) if record.get("ir_ambient_c") is not None else None
        lux = float(record["lux"]) if record.get("lux") is not None else None
        soil1_v = float(record["soil1_v"]) if record.get("soil1_v") is not None else None
        soil2_v = float(record["soil2_v"]) if record.get("soil2_v") is not None else None
        battery_v = float(record["battery_v"]) if record.get("battery_v") is not None else None

        status_val = record.get("status")
        status_json = json.dumps(status_val) if isinstance(status_val, dict) else (str(status_val) if status_val is not None else None)

        with conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO mast_telemetry (
                    log_epoch, seq, node_id, field_id, utc, rtc_valid, uptime_s,
                    air_temp_c, rh_pct, ir_object_c, ir_ambient_c, lux,
                    soil1_v, soil2_v, battery_v, status_json, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    int(log_epoch),
                    seq,
                    node_id,
                    str(field_id) if field_id is not None else None,
                    str(utc_val) if utc_val is not None else None,
                    rtc_valid,
                    uptime_s,
                    air_temp,
                    rh_pct,
                    ir_obj,
                    ir_amb,
                    lux,
                    soil1_v,
                    soil2_v,
                    battery_v,
                    status_json,
                    now_utc,
                ),
            )
            return cursor.lastrowid

    def record_mast_readings(
        self,
        records: List[Dict[str, Any]],
        log_epoch: int,
        received_at: Optional[str] = None,
    ) -> int:
        """Records a batch of ground mast readings atomically."""
        count = 0
        for rec in records:
            if isinstance(rec, dict) and "seq" in rec:
                rec_ts = rec.get("received_at") or received_at
                self.record_mast_reading(rec, log_epoch=log_epoch, received_at=rec_ts)
                count += 1
        return count

    def get_mast_cursor(self, node_id: str = "N01") -> Tuple[int, int]:
        """Returns (log_epoch, last_seq) stored for the node, defaulting to (0, 0)."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT log_epoch, last_seq FROM mast_sync_state WHERE node_id = ?;",
            (str(node_id),),
        ).fetchone()
        if row:
            return (int(row["log_epoch"]), int(row["last_seq"]))
        return (0, 0)

    def update_mast_cursor(self, node_id: str, log_epoch: int, last_seq: int) -> None:
        """Persists the sync cursor for a mast node."""
        conn = self._get_connection()
        now_utc = get_utc_iso_now()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mast_sync_state (node_id, log_epoch, last_seq, updated_at)
                VALUES (?, ?, ?, ?);
                """,
                (str(node_id), int(log_epoch), int(last_seq), now_utc),
            )

    def get_latest_mast_telemetry(self, node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches the latest mast telemetry row from SQLite."""
        conn = self._get_connection()
        if node_id:
            row = conn.execute(
                "SELECT * FROM mast_telemetry WHERE node_id = ? ORDER BY id DESC LIMIT 1;",
                (str(node_id),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM mast_telemetry ORDER BY id DESC LIMIT 1;"
            ).fetchone()

        if not row:
            return None

        status_obj = None
        if row["status_json"]:
            try:
                status_obj = json.loads(row["status_json"])
            except Exception:
                status_obj = None

        return {
            "id": row["id"],
            "log_epoch": row["log_epoch"],
            "seq": row["seq"],
            "node_id": row["node_id"],
            "field_id": row["field_id"],
            "utc": row["utc"],
            "rtc_valid": bool(row["rtc_valid"]),
            "uptime_s": row["uptime_s"],
            "air_temp_c": row["air_temp_c"],
            "rh_pct": row["rh_pct"],
            "ir_object_c": row["ir_object_c"],
            "ir_ambient_c": row["ir_ambient_c"],
            "lux": row["lux"],
            "soil1_v": row["soil1_v"],
            "soil2_v": row["soil2_v"],
            "battery_v": row["battery_v"],
            "status": status_obj,
            "received_at": row["received_at"],
        }

    def get_mast_readings_history(
        self,
        hours: float = 24.0,
        node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches mast readings recorded over the last `hours` window based on valid UTC timestamps.
        """
        conn = self._get_connection()
        query = "SELECT * FROM mast_telemetry "
        params = []
        if node_id:
            query += "WHERE node_id = ? "
            params.append(str(node_id))
        query += "ORDER BY id ASC;"

        rows = conn.execute(query, tuple(params)).fetchall()
        all_readings = []
        valid_utc_list = []

        for row in rows:
            status_obj = None
            if row["status_json"]:
                try:
                    status_obj = json.loads(row["status_json"])
                except Exception:
                    status_obj = None
            rec = {
                "id": row["id"],
                "log_epoch": row["log_epoch"],
                "seq": row["seq"],
                "node_id": row["node_id"],
                "field_id": row["field_id"],
                "utc": row["utc"],
                "rtc_valid": bool(row["rtc_valid"]),
                "uptime_s": row["uptime_s"],
                "air_temp_c": row["air_temp_c"],
                "rh_pct": row["rh_pct"],
                "ir_object_c": row["ir_object_c"],
                "ir_ambient_c": row["ir_ambient_c"],
                "lux": row["lux"],
                "soil1_v": row["soil1_v"],
                "soil2_v": row["soil2_v"],
                "battery_v": row["battery_v"],
                "status": status_obj,
                "received_at": row["received_at"],
            }
            if rec["rtc_valid"] and rec["utc"]:
                dt = _parse_iso_timestamp(rec["utc"])
                if dt is not None:
                    valid_utc_list.append(dt)
            all_readings.append(rec)

        if not valid_utc_list:
            return all_readings

        latest_utc_dt = max(valid_utc_list)
        cutoff_dt = latest_utc_dt - datetime.timedelta(hours=hours)

        filtered_result = []
        for r in all_readings:
            if r["rtc_valid"] and r["utc"]:
                dt = _parse_iso_timestamp(r["utc"])
                if dt is not None:
                    if dt >= cutoff_dt:
                        filtered_result.append(r)
                else:
                    filtered_result.append(r)
            else:
                filtered_result.append(r)

        return filtered_result

    def record_satellite_ndvi(
        self,
        ndvi_data: Dict[str, Any],
        field_id: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Records a Sentinel-2 satellite NDVI observation into SQLite (M4.5)."""
        conn = self._get_connection()
        now_utc = get_utc_iso_now()
        raw_json = json.dumps(raw_response) if raw_response is not None else None

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO satellite_ndvi (
                    field_id, fetched_utc, scene_date, ndvi_mean, ndvi_std,
                    valid_pixel_count, cloud_masked_fraction, raw_response_json, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(field_id) if field_id is not None else None,
                    now_utc,
                    str(ndvi_data["scene_date"]),
                    float(ndvi_data["ndvi_mean"]) if ndvi_data.get("ndvi_mean") is not None else None,
                    float(ndvi_data["ndvi_std"]) if ndvi_data.get("ndvi_std") is not None else None,
                    int(ndvi_data["valid_pixel_count"]) if ndvi_data.get("valid_pixel_count") is not None else None,
                    float(ndvi_data["cloud_masked_fraction"]) if ndvi_data.get("cloud_masked_fraction") is not None else None,
                    raw_json,
                    str(ndvi_data.get("source", "SENTINEL2_L2A_CDSE")),
                ),
            )
            return cursor.lastrowid

    def get_latest_satellite_ndvi(self, field_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches the most recent cached satellite NDVI record."""
        conn = self._get_connection()
        if field_id:
            row = conn.execute(
                "SELECT * FROM satellite_ndvi WHERE field_id = ? ORDER BY scene_date DESC, id DESC LIMIT 1;",
                (str(field_id),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM satellite_ndvi ORDER BY scene_date DESC, id DESC LIMIT 1;"
            ).fetchone()

        if not row:
            return None

        # Compute age in days
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        age_days = None
        try:
            s_dt = datetime.datetime.strptime(row["scene_date"], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            age_days = round(abs((now_dt - s_dt).total_seconds()) / 86400.0, 1)
        except Exception:
            pass

        valid_count = row["valid_pixel_count"]
        reliability_note = None
        if valid_count is not None and valid_count < 9:
            reliability_note = "UNRELIABLE_SMALL_FIELD (<9 pixels / ~30x30m footprint)"

        return {
            "id": row["id"],
            "field_id": row["field_id"],
            "fetched_utc": row["fetched_utc"],
            "scene_date": row["scene_date"],
            "age_days": age_days,
            "ndvi_mean": row["ndvi_mean"],
            "ndvi_std": row["ndvi_std"],
            "valid_pixel_count": valid_count,
            "cloud_masked_fraction": row["cloud_masked_fraction"],
            "source": row["source"],
            "reliability_note": reliability_note,
        }

    def close(self) -> None:
        """Closes the current thread's connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

