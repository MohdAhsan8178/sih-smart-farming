"""
Test payload contract synchronization and real schema conformance (J1.4).
Validates that docs/PAYLOAD_CONTRACT.md and docs/APP_TEAM_CHANGES.md stay strictly
synchronized with code constants and real synthesized advisory payloads emitted by edge/storage.py.
"""
import json
from pathlib import Path
import tempfile
import pytest

from configs.classes import CLASS_NAMES
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from configs.reliability import MODEL_A_CROSS_SOURCE_RELIABILITY
from edge.rules_engine import TEMPLATES
from core.trap_segmentation import TRAP_ETL_REGISTRY
from edge.storage import EdgeStorage
from gateway.server import EdgeGateway

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_payload_contract_contains_all_classes():
    contract_file = REPO_ROOT / "docs" / "PAYLOAD_CONTRACT.md"
    assert contract_file.exists()
    content = contract_file.read_text(encoding="utf-8")

    # 1. Model A classes
    for c in CLASS_NAMES:
        assert c in content, f"Class {c} missing from PAYLOAD_CONTRACT.md"

    # 2. Model B classes
    for mb in MODEL_B_CLASSES:
        assert mb in content, f"Model B class {mb} missing from PAYLOAD_CONTRACT.md"

    # 3. Action templates
    for tid in TEMPLATES.keys():
        assert tid in content, f"Template ID {tid} missing from PAYLOAD_CONTRACT.md"

    # 4. Reliability tiers
    for tier in {"TESTED_ROBUST", "TESTED_WEAK", "TESTED_FAILED", "UNTESTED"}:
        assert tier in content, f"Reliability tier {tier} missing from PAYLOAD_CONTRACT.md"

    # 5. Verification status frozen enum
    for status in {"VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED"}:
        assert status in content, f"Verification status {status} missing from PAYLOAD_CONTRACT.md"

    # 6. Established ETL comparison statuses
    for etl_s in {"BELOW_ETL", "AT_ETL", "ABOVE_ETL"}:
        assert etl_s in content, f"ETL status {etl_s} missing from PAYLOAD_CONTRACT.md"


def test_app_team_changes_contains_all_payload_contract_fields():
    import re
    contract_file = REPO_ROOT / "docs" / "PAYLOAD_CONTRACT.md"
    app_file = REPO_ROOT / "docs" / "APP_TEAM_CHANGES.md"
    assert contract_file.exists(), "docs/PAYLOAD_CONTRACT.md missing"
    assert app_file.exists(), "docs/APP_TEAM_CHANGES.md missing"

    contract_content = contract_file.read_text(encoding="utf-8")
    app_content = app_file.read_text(encoding="utf-8")

    # Extract all table column fields | `field_name` | from PAYLOAD_CONTRACT.md
    fields = re.findall(r"\|\s*\`([a-zA-Z0-9_\.]+)\`\s*\|", contract_content)
    assert len(fields) > 50, f"Expected > 50 fields, extracted {len(fields)}"

    missing_fields = []
    for f in fields:
        if f not in app_content:
            missing_fields.append(f)

    assert not missing_fields, (
        f"The following fields from docs/PAYLOAD_CONTRACT.md are missing from "
        f"docs/APP_TEAM_CHANGES.md: {missing_fields}"
    )


def test_real_advisory_payload_schema_conformance():
    """
    Rigorously validates that a real synthesized advisory payload emitted by
    EdgeStorage.create_advisory() exactly matches the documented wire contract schema.
    Prevents silent schema drift between storage.py, gateway/server.py, and docs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_contract_edge.db"
        storage = EdgeStorage(db_path=db_path)

        scan_id = "scan_contract_001"
        storage.record_scan_start(
            scan_id=scan_id,
            started_utc="2026-09-19T10:00:00Z",
            source="test_walk.mp4",
            metadata={"days_since_planting": 75, "total_cycle_days": 150, "latitude": 28.52},
        )

        # 1. Record healthy frame event with GPS
        storage.record_frame_event(
            scan_id=scan_id,
            frame_idx=0,
            timestamp_utc="2026-09-19T10:00:01Z",
            cell_id="cell_001",
            gate_passed=True,
            gate_metrics={"blur_score": 140.0, "displacement": 1.2},
            n_valid_tiles=9,
            frame_state="HEALTHY",
            class_id=0,
            confidence=0.98,
            tile_decisions=[{"state": "OK", "class_id": 0, "conf": 0.98}] * 9,
            source_image="frame_0.jpg",
            gps={"latitude": 28.5201, "longitude": 77.5801, "fix_quality": 1, "hdop": 1.2},
            canopy_cover=0.82,
            vari=0.24,
            exg=45.0,
            tgi=18.0,
            dgci=0.58,
            dgci_ood_frac=0.0,
            indices_status="OK",
        )

        # 2. Record disease frame event with GPS
        storage.record_frame_event(
            scan_id=scan_id,
            frame_idx=1,
            timestamp_utc="2026-09-19T10:00:02Z",
            cell_id="cell_001",
            gate_passed=True,
            gate_metrics={"blur_score": 135.0, "displacement": 1.1},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=4,  # rice__blast
            confidence=0.96,
            tile_decisions=[{"state": "OK", "class_id": 4, "conf": 0.96}] * 9,
            source_image="frame_1.jpg",
            gps={"latitude": 28.5202, "longitude": 77.5802, "fix_quality": 1, "hdop": 1.3},
            canopy_cover=0.80,
            vari=0.22,
            exg=42.0,
            tgi=17.5,
            dgci=0.55,
            dgci_ood_frac=0.0,
            indices_status="OK",
        )

        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_001",
            state="DISEASE",
            class_id=4,
            score=0.96,
            n_frames=2,
            n_agree=2,
            canopy_cover=0.81,
        )

        storage.record_scan_end(
            scan_id=scan_id,
            ended_utc="2026-09-19T10:00:10Z",
            frames_captured=2,
            frames_evaluated=2,
            tiles_classified=18,
            distance_walked_m=12.5,
        )

        # 3. Create advisory
        payload = storage.create_advisory(
            scan_id=scan_id,
            advisory_id="adv_contract_001",
            replay=False,
            field_id="F01",
            days_since_planting=75,
            total_cycle_days=150,
            inference_backend="trt",
        )

        # --- Top-Level Schema Keys ---
        expected_top_keys = {
            "schema_version",
            "advisory_id",
            "seq",
            "generated_at_utc",
            "inference_backend",
            "replay",
            "scan",
            "crop_health",
            "growth_stage",
            "vegetation",
            "thermal",
            "ndvi",
            "ndvi_satellite",
            "irrigation",
            "detections",
            "gps",
            "disease",
            "pest",
            "inputs",
            "actions",
        }
        assert set(payload.keys()) == expected_top_keys, f"Top-level keys mismatch: {set(payload.keys()) ^ expected_top_keys}"
        assert payload["schema_version"] == "1.0"
        assert payload["inference_backend"] == "trt"
        assert payload["replay"] is False

        # --- Scan Block Schema Keys ---
        expected_scan_keys = {
            "started_utc",
            "ended_utc",
            "mode",
            "frames_captured",
            "frames_evaluated",
            "tiles_classified",
            "distance_walked_m",
            "distance_reason",
        }
        assert set(payload["scan"].keys()) == expected_scan_keys
        assert payload["scan"]["frames_evaluated"] == 2
        assert payload["scan"]["tiles_classified"] == 18

        # --- Crop Health Block Schema Keys ---
        expected_crop_health_keys = {
            "state",
            "reason",
            "crop",
            "frames_evaluated",
            "frames_agreeing",
            "frames_rejected_ood",
            "frames_rejected_not_crop",
            "frames_uncertain",
            "source",
        }
        assert set(payload["crop_health"].keys()) == expected_crop_health_keys
        assert payload["crop_health"]["state"] == "DISEASE"
        assert payload["crop_health"]["crop"] == "rice"

        # --- Detections Block Schema Keys (Item 1) ---
        expected_detection_keys = {
            "class",
            "confidence",
            "cross_source_reliability",
            "lat",
            "lon",
            "fix_quality",
            "hdop",
            "captured_utc",
            "source",
        }
        assert len(payload["detections"]) == 2
        for det in payload["detections"]:
            assert set(det.keys()) == expected_detection_keys, f"Detection keys mismatch: {set(det.keys()) ^ expected_detection_keys}"
            assert isinstance(det["class"], str)
            assert isinstance(det["confidence"], float)
            assert isinstance(det["lat"], float)
            assert isinstance(det["lon"], float)
            assert det["source"] == "measured"

        # --- GPS Block Schema Keys (Item 2) ---
        expected_gps_keys = {"status", "point_count", "accuracy_note"}
        assert set(payload["gps"].keys()) == expected_gps_keys, f"GPS keys mismatch: {set(payload['gps'].keys()) ^ expected_gps_keys}"
        assert payload["gps"]["status"] == "OK"
        assert payload["gps"]["point_count"] == 2
        assert isinstance(payload["gps"]["accuracy_note"], str)

        # --- Disease Block Schema Keys (Item 3) ---
        expected_disease_keys = {"class", "confidence", "media_ids", "source"}
        assert len(payload["disease"]) == 1  # 1 blast disease detection
        for dis in payload["disease"]:
            assert set(dis.keys()) == expected_disease_keys, f"Disease keys mismatch: {set(dis.keys()) ^ expected_disease_keys}"
            assert dis["class"] == "rice__blast"
            assert isinstance(dis["confidence"], float)
            assert isinstance(dis["media_ids"], list)
            assert dis["source"] == "measured"

        # --- Actions Block Schema Keys (Item 4) ---
        expected_action_keys = {
            "rank",
            "template_id",
            "action",
            "rationale",
            "params",
            "verification_status",
            "url",
            "document_reference",
            "offline_source_file",
            "confidence",
            "advisory_only",
            "generated_by",
            "source",
        }
        assert len(payload["actions"]) >= 1
        for act in payload["actions"]:
            assert set(act.keys()) == expected_action_keys, f"Action keys mismatch: {set(act.keys()) ^ expected_action_keys}"
            assert isinstance(act["rank"], int)
            assert isinstance(act["template_id"], str)
            assert isinstance(act["action"], str)
            assert isinstance(act["rationale"], str)
            assert isinstance(act["params"], dict)
            assert isinstance(act["confidence"], str)
            assert act["confidence"] in ("high", "medium", "low")
            assert act["advisory_only"] is True
            assert act["generated_by"] == "template"
            assert act["source"] == "derived"

        # --- Inputs Block Schema Keys ---
        expected_input_keys = {"name", "source_node", "status"}
        assert len(payload["inputs"]) == 3
        for inp in payload["inputs"]:
            assert set(inp.keys()) == expected_input_keys
            assert inp["status"] in ("OK", "PENDING_CALIBRATION", "MOCK_PROVISIONAL", "ABSENT")

        # --- Verify GET /api/v1/advisory/latest against EdgeStorage ---
        latest_adv = storage.get_advisory("latest")
        assert latest_adv is not None
        assert latest_adv["advisory_id"] == "adv_contract_001"
        assert latest_adv["seq"] == payload["seq"]

        storage.close()


def test_gateway_latest_endpoint_retrieval():
    """
    Verifies that GET /api/v1/advisory/latest returns the most recent advisory (Item 6).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "gateway_latest_test.db"
        storage = EdgeStorage(db_path=db_path)

        for i in range(1, 4):
            scan_id = f"scan_{i:03d}"
            storage.record_scan_start(scan_id)
            storage.record_scan_end(scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=9)
            storage.create_advisory(scan_id=scan_id, advisory_id=f"adv_{i:03d}", replay=True)

        gateway = EdgeGateway(host="127.0.0.1", port=0, storage=storage, db_path=db_path, allow_mock=True)
        gateway.start_background()
        actual_port = gateway.server.server_address[1]
        base_url = f"http://127.0.0.1:{actual_port}"

        try:
            import urllib.request
            req = urllib.request.Request(f"{base_url}/api/v1/advisory/latest")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["advisory_id"] == "adv_003"
                assert data["seq"] == 3
        finally:
            gateway.stop()
            storage.close()
