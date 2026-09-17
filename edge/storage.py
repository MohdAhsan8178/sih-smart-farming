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
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.reliability import get_cross_source_reliability
import configs.train_config as train_config
import numpy as np

# Default database storage location on the Jetson Nano filesystem
DEFAULT_DB_PATH = Path("data/edge.db")

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


class EdgeStorage(object):
    """
    Thread-safe SQLite storage engine for edge pipeline telemetry, detections,
    spatial cell aggregation, and advisory generation.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        max_retained_frames: int = DEFAULT_MAX_RETAINED_FRAMES,
    ):
        self.db_path = Path(db_path)
        self.max_retained_frames = int(max_retained_frames)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

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
                        metadata_json TEXT
                    );
                    """
                )

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

                # Performance indexes
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
        mode: str = "handheld_pod",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records the beginning of a scanning session."""
        started_utc = started_utc or get_utc_iso_now()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scans (
                    scan_id, started_utc, mode, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (
                    scan_id,
                    started_utc,
                    mode,
                    str(source) if source else None,
                    json.dumps(metadata or {}),
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
                    distance_reason = ?
                WHERE scan_id = ?;
                """,
                (
                    ended_utc,
                    int(frames_captured),
                    int(frames_evaluated),
                    int(tiles_classified),
                    distance_walked_m,
                    distance_reason,
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
        replay: bool = True,
        field_id: str = "F01",
        days_since_planting: Optional[int] = None,
        total_cycle_days: Optional[int] = None,
        pest_data: Optional[Dict[str, Any]] = None,
        inference_backend: str = "trt",
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
        if not advisory_id:
            candidate_id = f"{now_utc}_{field_id}"
            existing = conn.execute("SELECT 1 FROM advisories WHERE advisory_id = ?;", (candidate_id,)).fetchone()
            if existing:
                ms = datetime.datetime.now(datetime.timezone.utc).strftime("%f")[:3]
                candidate_id = f"{now_utc[:-1]}.{ms}Z_{field_id}"
                idx = 2
                while conn.execute("SELECT 1 FROM advisories WHERE advisory_id = ?;", (candidate_id,)).fetchone():
                    candidate_id = f"{now_utc[:-1]}.{ms}Z_{field_id}_{idx}"
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

        if crop_counts:
            total_crop_detections = sum(crop_counts.values())
            sorted_crops = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)
            top_crop, top_count = sorted_crops[0]

            if len(sorted_crops) == 1:
                dominant_crop = top_crop
            else:
                # Detections span multiple distinct crops.
                # Only assert a dominant crop if it represents a clear supermajority (>=85%)
                if (top_count / float(total_crop_detections)) >= 0.85:
                    dominant_crop = top_crop
                else:
                    # Degrade honestly: do not assert a dominant crop when detections are mixed/conflicting
                    dominant_crop = None
                    crop_health_reason = "MULTIPLE_CROPS_DETECTED"

        # Overall crop_health state derivation
        max_agree = max([c["n_agree"] for c in cells] or [0])
        cell_states = set(c["state"] for c in cells)

        overall_state = "NO_DATA"
        if events:
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
                # Fallback when no cell verdicts exist (e.g. short test scans)
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

        mean_canopy = float(np.mean(canopy_covers)) if canopy_covers else 0.0
        p10_canopy = float(np.percentile(canopy_covers, 10)) if canopy_covers else 0.0
        p50_canopy = float(np.percentile(canopy_covers, 50)) if canopy_covers else 0.0
        p90_canopy = float(np.percentile(canopy_covers, 90)) if canopy_covers else 0.0

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

        if mean_canopy < train_config.PROVISIONAL_MIN_CANOPY_FRACTION or not varis:
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

        # J5 Hardware-Gated Blocks: Thermal, NDVI, Irrigation
        from edge.sensors import MastTelemetryReader
        reader = MastTelemetryReader(self)
        mast_reading = reader.get_latest()

        # 1. Thermal block (PENDING_HARDWARE.md Subsystem 1)
        thermal_block = {
            "available": False,
            "reason": "HARDWARE_NOT_CONNECTED (MLX90640 handheld thermal array absent, see PENDING_HARDWARE.md Subsystem 1)",
        }

        # 2. NDVI block (PENDING_HARDWARE.md Subsystem 4)
        ndvi_block = {
            "available": False,
            "reason": "HARDWARE_NOT_CONNECTED (IMX219-77IR NoIR camera and DB660/850 filter absent, see PENDING_HARDWARE.md Subsystem 4)",
        }

        # 3. Irrigation block (FAO-56 Hargreaves-Samani, Eq 52)
        history_24h = self.get_mast_readings_history(hours=24.0)
        valid_temps = [float(r["air_temp_c"]) for r in history_24h if r.get("air_temp_c") is not None]

        # Minimum coverage rule:
        # At least 6 samples spanning >= 6.0 hours (to capture diurnal day/night spread)
        has_min_coverage = False
        time_span_h = 0.0
        if len(valid_temps) >= 6 and len(history_24h) >= 2:
            dt0 = _parse_iso_timestamp(history_24h[0]["received_at"])
            dt1 = _parse_iso_timestamp(history_24h[-1]["received_at"])
            if dt0 and dt1:
                time_span_h = abs((dt1 - dt0).total_seconds()) / 3600.0
                if time_span_h >= 6.0:
                    has_min_coverage = True


        if mast_reading is not None and mast_reading.get("air_temp_c") is not None:
            if not has_min_coverage:
                irrigation_block = {
                    "available": False,
                    "reason": "INSUFFICIENT_24H_HISTORY (need >=6 readings spanning >=6h in last 24h for Tmin/Tmax, found %d readings spanning %.1fh)" % (len(valid_temps), time_span_h),
                }
            else:
                t_min = min(valid_temps)
                t_max = max(valid_temps)
                t_mean = sum(valid_temps) / float(len(valid_temps))
                delta_t = max(0.0, t_max - t_min)
                kc = float(growth_stage.get("kc") if growth_stage and growth_stage.get("kc") is not None else 1.0)
                
                # FAO-56 Hargreaves-Samani Equation 52:
                # ET0 = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * Ra
                # Ra = 15.0 mm/day equivalent (benchmark for 20°N-28°N latitude, FAO-56 Table 2.6)
                ra_mm_day = 15.0
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

        # Construct payload strictly according to ans_for_vitthal.md §8 and data_flow_architecture.md §7.3
        payload = {
            "schema_version": "1.0",
            "advisory_id": advisory_id,
            "seq": 0,  # placeholder, replaced upon SQLite insert
            "generated_at_utc": now_utc,
            "inference_backend": str(inference_backend),
            "replay": bool(replay),
            "scan": {
                "started_utc": scan_row["started_utc"],
                "ended_utc": scan_row["ended_utc"] or now_utc,
                "mode": scan_row["mode"] or "handheld_pod",
                "frames_captured": int(scan_row["frames_captured"] or len(events)),
                "frames_evaluated": len(events),
                "tiles_classified": int(scan_row["tiles_classified"] or len(events) * train_config.N_TILES),
                "distance_walked_m": scan_row["distance_walked_m"],
                "distance_reason": scan_row["distance_reason"] or ("GPS_TRACK_NOT_RECORDED" if gps_points == 0 else None),
            },
            "crop_health": {
                "state": overall_state,
                "reason": crop_health_reason if (overall_state not in ("HEALTHY", "DISEASE") or crop_health_reason) else None,
                "crop": dominant_crop,
                "frames_evaluated": len(events),
                "frames_agreeing": max_agree,
                "frames_rejected_ood": state_counts["UNCERTAIN"],
                "frames_rejected_not_crop": state_counts["NOT_CROP"],
                "frames_uncertain": state_counts["UNCERTAIN"],
                "source": "measured",
            },
            "growth_stage": growth_stage,
            "vegetation": vegetation,
            "thermal": thermal_block,
            "ndvi": ndvi_block,
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
                {"name": "pod_camera_rgb", "source_node": "POD", "status": "OK"},
                {"name": "pod_gps", "source_node": "POD", "status": "OK" if gps_points > 0 else "ABSENT"},
                {"name": "pod_thermal", "source_node": "POD", "status": "MOCK_PROVISIONAL"},
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
            SELECT advisory_id, seq, generated_at_utc, bytes, replay
            FROM advisories
            WHERE seq > ?
            ORDER BY seq ASC
            LIMIT ?;
            """,
            (resolved_seq, limit + 1),
        ).fetchall()

        truncated = len(rows) > limit
        items = rows[:limit]

        advisories_list = [
            {
                "advisory_id": r["advisory_id"],
                "seq": r["seq"],
                "generated_at_utc": r["generated_at_utc"],
                "bytes": r["bytes"],
                "replay": bool(r["replay"]),
            }
            for r in items
        ]

        return {
            "schema_version": "1.0",
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

        return {
            "device": "sih-pod-01",
            "advisory_count": advisory_count,
            "storage_free_kb": storage_free_kb,
            "gps_time_valid": False,  # True once real GPS NMEA lock is confirmed
            "schema_version": "1.0",
            "latest_seq": latest_seq,
            "server_time_utc": get_utc_iso_now(),
            "clock_source": "filesystem",  # "gps" once GPS time sync is live
        }

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
        Fetches mast readings recorded over the last `hours` window.
        """
        conn = self._get_connection()
        cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
        cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        query = "SELECT * FROM mast_telemetry WHERE received_at >= ? "
        params = [cutoff_iso]
        if node_id:
            query += "AND node_id = ? "
            params.append(str(node_id))
        query += "ORDER BY id ASC;"

        rows = conn.execute(query, tuple(params)).fetchall()
        result = []
        for row in rows:
            status_obj = None
            if row["status_json"]:
                try:
                    status_obj = json.loads(row["status_json"])
                except Exception:
                    status_obj = None
            result.append({
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
            })
        return result

    def close(self) -> None:
        """Closes the current thread's connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

