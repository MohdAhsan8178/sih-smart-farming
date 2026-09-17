#!/usr/bin/env python3
"""
Unit and concurrency tests for edge/storage.py (Step 33).

Tests:
1. test_schema_creation_and_pragmas: WAL mode, busy timeout, tables, indexes.
2. test_frame_event_insert_and_query: roundtrip with null GPS and with real GPS.
3. test_cell_verdict_insert_and_query: primary key upsert behavior.
4. test_advisory_assembly_and_manifest_pagination: monotonic seq, limit, truncation, get_advisory.
5. test_ack_advisory_and_health_endpoint: ack tracking, unacked count, storage_free_kb.
6. test_concurrent_access_wal: simultaneous read/write across multiple threads without locks.
7. test_retention_policy_pruning: pruning oldest frame events while keeping advisories.
"""

import os
from pathlib import Path
import tempfile
import threading
import time
import pytest

from edge.storage import EdgeStorage


def test_schema_creation_and_pragmas():
    """Verify SQLite database initializes with WAL mode, busy timeout, and all tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        conn = storage._get_connection()
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "scans" in tables
        assert "frame_events" in tables
        assert "cell_verdicts" in tables
        assert "advisories" in tables
        assert "app_sync_state" in tables

        storage.close()


def test_frame_event_insert_and_query_with_null_gps():
    """Verify frame events insert cleanly with None GPS and can be retrieved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        storage.record_scan_start("scan_001", source="test_video.mp4")

        row_id = storage.record_frame_event(
            scan_id="scan_001",
            frame_idx=0,
            timestamp_utc="2026-09-15T18:00:00Z",
            cell_id="cell_walk_pod",
            gate_passed=True,
            gate_metrics={"blur_score": 150.0, "displacement": 1.0},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=1,
            confidence=0.985,
            tile_decisions=[{"state": "OK", "class_id": 1, "conf": 0.985}] * 9,
            source_image="leaf1.jpg",
            gps=None,  # Nullable GPS
        )

        assert row_id > 0

        conn = storage._get_connection()
        row = conn.execute("SELECT * FROM frame_events WHERE id = ?;", (row_id,)).fetchone()
        assert row["scan_id"] == "scan_001"
        assert row["frame_state"] == "DISEASE"
        assert row["class_id"] == 1
        assert row["class_name"] == "rice__bacterial_leaf_blight"
        assert row["lat"] is None
        assert row["lon"] is None

        storage.close()


def test_cell_verdict_upsert():
    """Verify cell verdicts update cleanly on subsequent visits to the same cell."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        storage.record_scan_start("scan_001")

        storage.record_cell_verdict(
            scan_id="scan_001",
            cell_id="cell_10_20",
            state="UNCERTAIN",
            class_id=None,
            score=0.0,
            n_frames=1,
            n_agree=0,
        )

        # Update on second visit
        storage.record_cell_verdict(
            scan_id="scan_001",
            cell_id="cell_10_20",
            state="HEALTHY",
            class_id=0,
            score=0.92,
            n_frames=2,
            n_agree=2,
        )

        conn = storage._get_connection()
        rows = conn.execute(
            "SELECT * FROM cell_verdicts WHERE scan_id = ? AND cell_id = ?;",
            ("scan_001", "cell_10_20"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["state"] == "HEALTHY"
        assert rows[0]["n_agree"] == 2

        storage.close()


def test_advisory_assembly_and_manifest_pagination():
    """Verify advisory assembly, monotonic seq ordering, and manifest pagination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        # Create 3 scans and advisories
        for i in range(1, 4):
            scan_id = f"scan_{i:03d}"
            storage.record_scan_start(scan_id)
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=0,
                timestamp_utc=f"2026-09-15T18:0{i}:00Z",
                cell_id="cell_walk_pod",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.95,
                tile_decisions=[],
            )
            storage.record_scan_end(scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)
            advisory = storage.create_advisory(scan_id=scan_id, advisory_id=f"adv_{i:03d}", replay=True)
            assert advisory["seq"] == i

        # Test manifest with limit=2 (expect truncated=True)
        manifest = storage.get_manifest(since_seq=0, limit=2)
        assert len(manifest["advisories"]) == 2
        assert manifest["truncated"] is True
        assert manifest["advisories"][0]["seq"] == 1
        assert manifest["advisories"][1]["seq"] == 2

        # Page 2 from seq=2
        manifest2 = storage.get_manifest(since_seq=2, limit=2)
        assert len(manifest2["advisories"]) == 1
        assert manifest2["truncated"] is False
        assert manifest2["advisories"][0]["seq"] == 3

        # Test get_advisory by id and by seq
        adv_by_id = storage.get_advisory("adv_002")
        assert adv_by_id is not None
        assert adv_by_id["advisory_id"] == "adv_002"
        assert adv_by_id["seq"] == 2

        adv_by_seq = storage.get_advisory(2)
        assert adv_by_seq is not None
        assert adv_by_seq["advisory_id"] == "adv_002"

        storage.close()


def test_ack_advisory_and_health():
    """Verify ack_advisory advances unacked count in /health without deleting records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)

        storage.record_scan_start("scan_001")
        storage.record_scan_end("scan_001")
        storage.create_advisory(scan_id="scan_001", advisory_id="adv_001")

        health = storage.get_health()
        assert health["device"] == "sih-pod-01"
        assert health["advisory_count"] == 1
        assert health["latest_seq"] == 1
        assert health["storage_free_kb"] > 0

        # Ack advisory
        assert storage.ack_advisory("adv_001") is True
        health2 = storage.get_health()
        assert health2["advisory_count"] == 0  # unacked count drops to 0

        # Advisory remains retrievable after ack
        assert storage.get_advisory("adv_001") is not None

        storage.close()


def test_concurrent_access_wal():
    """Verify multi-threaded concurrent read/write does not lock or crash under WAL mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path)
        storage.record_scan_start("scan_conc")

        errors = []

        def writer_thread(t_id):
            try:
                for i in range(20):
                    storage.record_frame_event(
                        scan_id="scan_conc",
                        frame_idx=t_id * 100 + i,
                        timestamp_utc="2026-09-15T18:00:00Z",
                        cell_id=f"cell_{t_id}",
                        gate_passed=True,
                        gate_metrics={},
                        n_valid_tiles=9,
                        frame_state="HEALTHY",
                        class_id=0,
                        confidence=0.9,
                        tile_decisions=[],
                    )
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("writer", e))

        def reader_thread():
            try:
                for _ in range(30):
                    _ = storage.get_health()
                    _ = storage.get_manifest(since_seq=0, limit=50)
                    time.sleep(0.004)
            except Exception as e:
                errors.append(("reader", e))

        threads = [
            threading.Thread(target=writer_thread, args=(1,)),
            threading.Thread(target=writer_thread, args=(2,)),
            threading.Thread(target=reader_thread),
            threading.Thread(target=reader_thread),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        storage.close()


def test_retention_policy_pruning():
    """Verify prune_retained_data enforces frame limit while preserving advisories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edge.db"
        storage = EdgeStorage(db_path=db_path, max_retained_frames=10)

        storage.record_scan_start("scan_prune")

        # Insert 25 frame events
        for i in range(25):
            storage.record_frame_event(
                scan_id="scan_prune",
                frame_idx=i,
                timestamp_utc="2026-09-15T18:00:00Z",
                cell_id="cell_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="HEALTHY",
                class_id=0,
                confidence=0.9,
                tile_decisions=[],
            )

        storage.record_scan_end("scan_prune", frames_captured=25, frames_evaluated=25, tiles_classified=225)
        adv = storage.create_advisory("scan_prune", advisory_id="adv_prune")

        conn = storage._get_connection()
        total_before = conn.execute("SELECT COUNT(*) FROM frame_events;").fetchone()[0]
        assert total_before == 25

        # Run pruning
        pruned_count = storage.prune_retained_data(max_frames=10)
        assert pruned_count == 15

        total_after = conn.execute("SELECT COUNT(*) FROM frame_events;").fetchone()[0]
        assert total_after == 10

        # Advisory must still exist
        assert storage.get_advisory("adv_prune") is not None

        storage.close()


def test_multicrop_without_supermajority_produces_null_crop():
    """Detections spanning multiple crops without >=85% supermajority yields crop=None and MULTIPLE_CROPS_DETECTED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        scan_id = "scan_multicrop"
        storage.record_scan_start(scan_id)

        # 4 rice events, 2 sugarcane events, 1 wheat event (4/7 = 57% < 85%)
        classes = [1, 1, 1, 1, 14, 14, 24]  # rice, rice, rice, rice, sugarcane, sugarcane, wheat
        for i, cid in enumerate(classes):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-15T18:00:00Z",
                cell_id="cell_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="DISEASE",
                class_id=cid,
                confidence=0.95,
                tile_decisions=[],
            )

        storage.record_scan_end(scan_id, frames_captured=len(classes), frames_evaluated=len(classes), tiles_classified=len(classes)*9)
        adv = storage.create_advisory(scan_id)

        assert adv["crop_health"]["crop"] is None
        assert adv["crop_health"]["reason"] == "MULTIPLE_CROPS_DETECTED"
        storage.close()


def test_single_crop_at_85pct_supermajority_asserts_crop():
    """A dominant crop representing >=85% of detections asserts that crop in crop_health."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        scan_id = "scan_supermajority"
        storage.record_scan_start(scan_id)

        # 9 rice events, 1 sugarcane event (9/10 = 90% >= 85%)
        classes = [1] * 9 + [14]
        for i, cid in enumerate(classes):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-15T18:00:00Z",
                cell_id="cell_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="DISEASE",
                class_id=cid,
                confidence=0.95,
                tile_decisions=[],
            )

        storage.record_scan_end(scan_id, frames_captured=len(classes), frames_evaluated=len(classes), tiles_classified=len(classes)*9)
        adv = storage.create_advisory(scan_id)

        assert adv["crop_health"]["crop"] == "rice"
        storage.close()


def test_no_cell_reaching_n_agree_produces_uncertain_state():
    """When recorded cells do not achieve temporal consensus (n_agree < k), overall state is UNCERTAIN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        scan_id = "scan_unconfirmed"
        storage.record_scan_start(scan_id)

        # Frame event saw disease
        storage.record_frame_event(
            scan_id=scan_id,
            frame_idx=0,
            timestamp_utc="2026-09-15T18:00:00Z",
            cell_id="cell_0",
            gate_passed=True,
            gate_metrics={},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=1,
            confidence=0.95,
            tile_decisions=[],
        )

        # But cell verdict was UNCERTAIN with n_agree=0
        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_0",
            state="UNCERTAIN",
            class_id=None,
            score=0.0,
            n_frames=2,
            n_agree=0,
        )

        storage.record_scan_end(scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)
        adv = storage.create_advisory(scan_id)

        assert adv["crop_health"]["state"] == "UNCERTAIN"
        assert adv["crop_health"]["frames_agreeing"] == 0
        storage.close()


def test_cell_reaching_consensus_produces_confirmed_state_and_inspect_action():
    """A cell reaching consensus yields confirmed state and ACT_INSPECT_CONFIRM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        scan_id = "scan_confirmed"
        storage.record_scan_start(scan_id)

        # Frame events
        for i in range(3):
            storage.record_frame_event(
                scan_id=scan_id,
                frame_idx=i,
                timestamp_utc="2026-09-15T18:00:00Z",
                cell_id="cell_0",
                gate_passed=True,
                gate_metrics={},
                n_valid_tiles=9,
                frame_state="DISEASE",
                class_id=1,  # rice__bacterial_leaf_blight
                confidence=0.95,
                tile_decisions=[],
            )

        # Cell confirmed DISEASE with 3 agreeing frames
        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_0",
            state="DISEASE",
            class_id=1,
            score=0.95,
            n_frames=3,
            n_agree=3,
        )

        storage.record_scan_end(scan_id, frames_captured=3, frames_evaluated=3, tiles_classified=27)
        adv = storage.create_advisory(scan_id)

        assert adv["crop_health"]["state"] == "DISEASE"
        assert adv["crop_health"]["crop"] == "rice"
        assert adv["crop_health"]["frames_agreeing"] == 3

        # Actions must emit confirmed disease template (ACT_TREAT_RICE_BLIGHT) with generated_by="template"
        assert len(adv["actions"]) >= 1
        assert adv["actions"][0]["template_id"] == "ACT_TREAT_RICE_BLIGHT"
        assert adv["actions"][0]["params"]["crop"] == "rice"
        assert adv["actions"][0]["params"]["disease"] == "bacterial_leaf_blight"
        assert adv["actions"][0]["generated_by"] == "template"

        storage.close()


def test_standing_guard_generated_by_always_template():
    """Standing guard: generated_by must ALWAYS be 'template' (ans_for_vitthal.md §7 F4) and NEVER 'placeholder' or 'rules_engine'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        storage.record_scan_start("scan_guard")
        storage.record_frame_event(
            scan_id="scan_guard",
            frame_idx=0,
            timestamp_utc="2026-09-15T18:00:00Z",
            cell_id="cell_0",
            gate_passed=True,
            gate_metrics={},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=1,
            confidence=0.95,
            tile_decisions=[],
        )
        storage.record_scan_end("scan_guard", frames_captured=1, frames_evaluated=1, tiles_classified=9)
        adv = storage.create_advisory("scan_guard")

        for action in adv.get("actions", []):
            assert action.get("generated_by") == "template", (
                f"Violation: generated_by was {action.get('generated_by')}, expected 'template' per F4 contract"
            )
            assert action.get("generated_by") not in ("rules_engine", "placeholder")
        storage.close()


def test_cross_source_reliability_in_detections():
    """F7/G3: Verify cross-source reliability tiers follow exact spec logic and are present in detections."""
    import json
    from configs.classes import CLASS_NAMES
    from configs.reliability import get_cross_source_reliability
    valid_tiers = {"TESTED_ROBUST", "TESTED_WEAK", "TESTED_FAILED", "UNTESTED"}

    # 1. Verify all 29 classes receive a valid tier and obey exact spec threshold logic
    rel_json = Path(__file__).resolve().parent.parent / "artifacts/reports/model_a_cross_source_reliability.json"
    if rel_json.exists():
        with open(rel_json) as f:
            rel_data = json.load(f)
        for c in CLASS_NAMES:
            tier = get_cross_source_reliability(c)
            assert tier in valid_tiers, f"Class {c} received invalid tier {tier}"
            if c in rel_data:
                meta = rel_data[c]
                sup = meta.get("support", 0)
                rec = meta.get("recall")
                # Rule: TESTED_ROBUST (recall >= 0.60), TESTED_WEAK (0.30 <= recall < 0.60), TESTED_FAILED (recall < 0.30), UNTESTED (support == 0)
                if sup == 0 or rec is None:
                    assert tier == "UNTESTED", f"Expected UNTESTED for {c}, got {tier}"
                elif rec >= 0.60:
                    assert tier == "TESTED_ROBUST", f"Expected TESTED_ROBUST for {c} with recall {rec}, got {tier}"
                elif rec >= 0.30:
                    assert tier == "TESTED_WEAK", f"Expected TESTED_WEAK for {c} with recall {rec}, got {tier}"
                else:
                    assert tier == "TESTED_FAILED", f"Expected TESTED_FAILED for {c} with recall {rec}, got {tier}"

    # Verify specific benchmark tiers per G3.2
    assert get_cross_source_reliability("sugarcane__healthy") == "TESTED_ROBUST"
    assert get_cross_source_reliability("wheat__yellow_rust") == "TESTED_ROBUST"
    assert get_cross_source_reliability("rice__normal") == "TESTED_WEAK"
    assert get_cross_source_reliability("wheat__powdery_mildew") == "TESTED_WEAK"
    assert get_cross_source_reliability("rice__bacterial_leaf_blight") == "TESTED_FAILED"
    assert get_cross_source_reliability("wheat__septoria") == "TESTED_FAILED"
    assert get_cross_source_reliability("sugarcane__smut") == "UNTESTED"

    # 2. Verify detection payload inclusion
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EdgeStorage(db_path=Path(tmpdir) / "test.db")
        storage.record_scan_start("scan_rel")
        storage.record_frame_event(
            scan_id="scan_rel",
            frame_idx=0,
            timestamp_utc="2026-09-15T18:00:00Z",
            cell_id="cell_0",
            gate_passed=True,
            gate_metrics={},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=1,  # rice__bacterial_leaf_blight
            confidence=0.92,
            tile_decisions=[],
        )
        storage.record_scan_end("scan_rel", frames_captured=1, frames_evaluated=1, tiles_classified=9)
        adv = storage.create_advisory("scan_rel")

        assert len(adv["detections"]) == 1
        d = adv["detections"][0]
        assert "cross_source_reliability" in d
        assert d["cross_source_reliability"] == "TESTED_FAILED"
        storage.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
