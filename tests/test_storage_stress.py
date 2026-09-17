#!/usr/bin/env python3
"""
tests/test_storage_stress.py — Stress test for EdgeStorage retention policy.

Exercises the 50,000-row retention ceiling with 60,000+ realistic frame events:
1. Ingests 60,000 full-payload frame events (with 9 tile decisions each).
2. Verifies retention pruning triggers and reduces row count to exactly 50,000.
3. Verifies WAL checkpointing truncates the -wal log.
4. Simulates post-prune ongoing pipeline streaming (inserting an additional 5,000 rows).
5. Verifies page reuse on SQLite freelist (file does not balloon continuously).
6. Confirms zero corruption (PRAGMA integrity_check == 'ok') and no deadlocks.
7. Reports exact measured file sizes on disk.
"""

import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Dict, List

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from edge.storage import EdgeStorage, get_utc_iso_now


def generate_realistic_frame_events(scan_id: str, start_idx: int, count: int) -> List[Dict[str, Any]]:
    """Generates realistic frame event payloads with 9 tile decisions each."""
    events = []
    base_time = "2026-09-16T00:00:00Z"
    for i in range(count):
        idx = start_idx + i
        tile_decisions = [
            {
                "tile_idx": t,
                "status": "ACCEPTED" if t < 7 else "REJECTED_ENERGY",
                "reason": None if t < 7 else "ENERGY_ABOVE_TAU",
                "class_id": 1 if (idx % 2 == 0 and t < 5) else 0,
                "class_name": "rice__bacterial_leaf_blight" if (idx % 2 == 0 and t < 5) else "rice__normal",
                "energy": -2.45 if t < 7 else -0.85,
                "confidence": 0.985 if t < 7 else 0.421,
            }
            for t in range(9)
        ]
        events.append({
            "scan_id": scan_id,
            "frame_idx": idx,
            "timestamp_utc": base_time,
            "source_image": f"frame_{idx:06d}.jpg",
            "cell_id": "cell_28.6139_77.2090",
            "gate_passed": True,
            "gate_metrics": {
                "blur_score": 245.8,
                "dark_fraction": 0.012,
                "bright_fraction": 0.005,
                "displacement": 142.3,
            },
            "n_valid_tiles": 9,
            "frame_state": "DISEASE" if idx % 2 == 0 else "HEALTHY",
            "class_id": 1 if idx % 2 == 0 else 0,
            "confidence": 0.985,
            "tile_decisions": tile_decisions,
            "gps": {
                "latitude": 28.6139,
                "longitude": 77.2090,
                "fix_quality": 1,
                "hdop": 1.2,
            },
        })
    return events


def test_storage_retention_stress_60k_rows():
    """
    Stress-tests retention policy with 60,000 inserted rows, pruning down to 50,000,
    WAL truncation, post-prune ingestion, and SQLite page freelist verification.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "edge_stress.db"
        storage = EdgeStorage(db_path=db_path, max_retained_frames=50000)

        scan_id = "stress_scan_20260916"
        storage.record_scan_start(scan_id=scan_id, mode="handheld_pod")

        print("\n--- PHASE 1: Ingesting 60,000 frame events in 2,500-row transactions ---")
        t0 = time.time()
        batch_size = 2500
        total_to_insert = 60000

        for offset in range(0, total_to_insert, batch_size):
            batch = generate_realistic_frame_events(scan_id, offset, batch_size)
            storage.record_frame_events_batch(batch)

        elapsed_ingest = time.time() - t0
        print(f"Ingested {total_to_insert} rows in {elapsed_ingest:.2f}s ({total_to_insert / elapsed_ingest:.0f} rows/s)")

        # Verify initial row count
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        count_pre = cur.execute("SELECT COUNT(*) FROM frame_events;").fetchone()[0]
        conn.commit()
        assert count_pre == 60000, f"Expected 60000 rows pre-prune, got {count_pre}"

        wal_path = Path(str(db_path) + "-wal")
        shm_path = Path(str(db_path) + "-shm")

        print("Disk listing at 60,000 rows (pre-prune):")
        subprocess.run(["ls", "-lh", str(db_path), str(wal_path), str(shm_path)], check=False)

        # Measure disk sizes at 60k rows (before pruning)
        db_size_60k = db_path.stat().st_size if db_path.exists() else 0
        wal_size_60k = wal_path.stat().st_size if wal_path.exists() else 0
        shm_size_60k = shm_path.stat().st_size if shm_path.exists() else 0

        # --- PHASE 2: Execute Retention Pruning ---
        print("\n--- PHASE 2: Triggering prune_retained_data(max_frames=50000) ---")
        t_prune_start = time.time()
        pruned_count = storage.prune_retained_data(max_frames=50000)
        elapsed_prune = time.time() - t_prune_start

        print(f"Pruned {pruned_count} rows in {elapsed_prune:.4f}s")
        assert pruned_count == 10000, f"Expected 10000 pruned rows, got {pruned_count}"

        # Verify capped row count
        count_post = cur.execute("SELECT COUNT(*) FROM frame_events;").fetchone()[0]
        conn.commit()
        assert count_post == 50000, f"Expected exactly 50000 rows post-prune, got {count_post}"

        print("Disk listing after pruning down to 50,000 rows & WAL truncate:")
        subprocess.run(["ls", "-lh", str(db_path), str(wal_path), str(shm_path)], check=False)

        # Measure disk sizes after prune and WAL truncate checkpoint
        db_size_post = db_path.stat().st_size if db_path.exists() else 0
        wal_size_post = wal_path.stat().st_size if wal_path.exists() else 0
        shm_size_post = shm_path.stat().st_size if shm_path.exists() else 0

        # Verify WAL checkpoint bounded the -wal file
        assert wal_size_post == 0, f"WAL file should be truncated to 0 bytes by checkpoint, got {wal_size_post}"

        # Check SQLite freelist page count (pages ready for immediate reuse)
        freelist_pages = cur.execute("PRAGMA freelist_count;").fetchone()[0]
        page_size = cur.execute("PRAGMA page_size;").fetchone()[0]
        conn.commit()
        freelist_bytes = freelist_pages * page_size
        print(f"SQLite Freelist (reusable free pages): {freelist_pages} pages ({freelist_bytes / (1024 * 1024):.2f} MB)")
        assert freelist_pages > 0, "Freelist should have recovered pages from the 10,000 deleted rows"

        # --- PHASE 3: Stream another 5,000 rows post-prune ---
        print("\n--- PHASE 3: Ingesting 5,000 subsequent rows to verify freelist reuse ---")
        post_batch = generate_realistic_frame_events(scan_id, 60000, 5000)
        storage.record_frame_events_batch(post_batch)

        # Re-run prune to maintain 50,000 ceiling
        pruned_second = storage.prune_retained_data(max_frames=50000)
        assert pruned_second == 5000, f"Expected 5000 rows pruned in second pass, got {pruned_second}"

        count_final = cur.execute("SELECT COUNT(*) FROM frame_events;").fetchone()[0]
        assert count_final == 50000, f"Row count should stay capped at 50,000, got {count_final}"

        db_size_final = db_path.stat().st_size if db_path.exists() else 0
        wal_size_final = wal_path.stat().st_size if wal_path.exists() else 0
        print(f"Disk sizes after streaming post-prune rows and re-pruning:")
        print(f"  .db  : {db_size_final / (1024 * 1024):.2f} MB ({db_size_final} bytes)")
        print(f"  -wal : {wal_size_final / (1024 * 1024):.2f} MB ({wal_size_final} bytes)")

        # Verify main database file did NOT expand because it reused freelist pages
        assert db_size_final <= db_size_60k * 1.05, (
            f"Database file expanded unexpectedly: {db_size_final} vs {db_size_60k}"
        )

        # --- PHASE 4: Database Integrity Check ---
        integrity = cur.execute("PRAGMA integrity_check;").fetchall()
        assert len(integrity) == 1 and integrity[0][0] == "ok", f"Integrity check failed: {integrity}"
        print(f"\nDatabase Integrity: {integrity[0][0].upper()} (zero corruption)")

        # --- PHASE 5: Verify Advisory Creation & Persistence ---
        storage.record_scan_end(scan_id=scan_id, frames_evaluated=count_final, tiles_classified=count_final * 9)
        adv = storage.create_advisory(scan_id=scan_id, replay=True)
        assert adv is not None and adv["seq"] == 1
        print(f"Advisory successfully synthesized post-stress: seq={adv['seq']}, advisory_id={adv['advisory_id']}")

        conn.close()


if __name__ == "__main__":
    test_storage_retention_stress_60k_rows()
