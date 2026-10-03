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

    # Extract all table column 1 fields | `field_name` | from PAYLOAD_CONTRACT.md
    fields = re.findall(r"^\s*\|\s*\`([a-zA-Z0-9_\.]+)\`\s*\|", contract_content, re.MULTILINE)
    assert len(fields) > 50, f"Expected > 50 fields, extracted {len(fields)}"

    missing_fields = []
    for f in fields:
        if f not in app_content:
            missing_fields.append(f)

    assert not missing_fields, (
        f"The following fields from docs/PAYLOAD_CONTRACT.md are missing from "
        f"docs/APP_TEAM_CHANGES.md: {missing_fields}"
    )


def assert_advisory_payload_schema_recursive(payload: dict):
    """
    Recursively validates exact key sets and value constraints across every nested
    block, sub-block, and list element in an Advisory document against the wire contract.
    """
    # 1. Top-Level Keys
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
        "time_source",
        "summary",
        "stretches",
        "alerts",
        "field_conditions",
    }
    assert set(payload.keys()) == expected_top_keys, f"Top-level keys mismatch: {set(payload.keys()) ^ expected_top_keys}"
    assert payload["schema_version"] == "1.0"
    assert isinstance(payload["advisory_id"], str)
    assert isinstance(payload["seq"], int) and payload["seq"] >= 1
    assert isinstance(payload["generated_at_utc"], str)
    assert payload["inference_backend"] in ("trt", "onnx", "mock")
    assert isinstance(payload["replay"], bool)
    assert payload["time_source"] in ("gps", "phone", "filesystem")
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["stretches"], list)
    assert isinstance(payload["alerts"], list)
    assert isinstance(payload["field_conditions"], dict)

    # 2. Scan Block
    expected_scan_keys = {
        "started_utc",
        "ended_utc",
        "mode",
        "frames_captured",
        "frames_evaluated",
        "tiles_classified",
        "distance_walked_m",
        "distance_reason",
        "crop_declared",
        "duration_s",
        "duration_reason",
        "stop_reason",
    }
    assert set(payload["scan"].keys()) == expected_scan_keys, f"Scan keys mismatch: {set(payload['scan'].keys()) ^ expected_scan_keys}"
    assert isinstance(payload["scan"]["started_utc"], str)
    assert isinstance(payload["scan"]["ended_utc"], str)
    assert payload["scan"]["mode"] in ("walk", "handheld_pod")
    assert isinstance(payload["scan"]["frames_captured"], int)
    assert isinstance(payload["scan"]["frames_evaluated"], int)
    assert isinstance(payload["scan"]["tiles_classified"], int)

    # 3. Crop Health Block
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
    assert set(payload["crop_health"].keys()) == expected_crop_health_keys, f"Crop health keys mismatch: {set(payload['crop_health'].keys()) ^ expected_crop_health_keys}"
    assert payload["crop_health"]["state"] in ("HEALTHY", "DISEASE", "UNCERTAIN", "NOT_CROP", "NO_DATA")
    assert payload["crop_health"]["source"] == "measured"

    # 4. Growth Stage Block
    if payload["growth_stage"].get("status") == "OK":
        expected_growth_stage_keys = {
            "crop",
            "stage",
            "stage_code",
            "days_since_planting",
            "total_cycle_days",
            "cycle_source",
            "cycle_verification_status",
            "stage_lengths_days",
            "canopy_cover_measured",
            "canopy_cover_expected_range",
            "kc",
            "status",
            "verification_status",
            "source",
            "document_reference",
        }
    elif payload["growth_stage"].get("status") == "AWAITING_PLANTING_DATE":
        expected_growth_stage_keys = {
            "crop",
            "stage",
            "stage_code",
            "reason",
            "status",
            "days_since_planting",
            "total_cycle_days",
            "cycle_source",
            "cycle_verification_status",
            "canopy_cover_measured",
            "canopy_cover_expected_range",
            "kc",
            "verification_status",
            "source",
            "document_reference",
        }
    else:
        expected_growth_stage_keys = {
            "crop",
            "stage",
            "stage_code",
            "reason",
            "status",
            "days_since_planting",
            "total_cycle_days",
            "canopy_cover_measured",
            "canopy_cover_expected_range",
            "kc",
            "verification_status",
            "source",
        }
    assert set(payload["growth_stage"].keys()) == expected_growth_stage_keys, f"Growth stage keys mismatch: {set(payload['growth_stage'].keys()) ^ expected_growth_stage_keys}"
    assert payload["growth_stage"]["verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")
    assert payload["growth_stage"]["source"] == "derived"

    # 5. Vegetation Block & Nested Sub-Blocks
    expected_veg_top_keys = {
        "interpretation_mode",
        "canopy_cover",
        "vari",
        "exg",
        "tgi",
        "dgci",
        "ndvi",
        "ndvi_status",
        "ndvi_reason",
    }
    assert set(payload["vegetation"].keys()) == expected_veg_top_keys, f"Vegetation top keys mismatch: {set(payload['vegetation'].keys()) ^ expected_veg_top_keys}"
    assert payload["vegetation"]["interpretation_mode"] == "relative"
    assert payload["vegetation"]["ndvi"] is None
    assert payload["vegetation"]["ndvi_status"] in ("GATED_HARDWARE_CALIBRATION", "PENDING_HARDWARE_FINALIZATION")

    # 5.1 Canopy Cover Sub-Block
    expected_canopy_keys = {
        "mean",
        "p10",
        "p50",
        "p90",
        "min_fraction_threshold",
        "status",
        "threshold_source",
        "threshold_confirmed",
        "source",
    }
    assert set(payload["vegetation"]["canopy_cover"].keys()) == expected_canopy_keys, f"Canopy cover keys mismatch: {set(payload['vegetation']['canopy_cover'].keys()) ^ expected_canopy_keys}"
    assert payload["vegetation"]["canopy_cover"]["status"] in ("OK", "INSUFFICIENT_CANOPY")
    assert isinstance(payload["vegetation"]["canopy_cover"]["threshold_confirmed"], bool)
    assert payload["vegetation"]["canopy_cover"]["source"] == "measured"

    # 5.2 VARI Sub-Block
    if payload["vegetation"]["canopy_cover"]["status"] == "OK":
        expected_vari_keys = {
            "band",
            "band_basis",
            "field_median",
            "mean",
            "p10",
            "p50",
            "p90",
            "source",
            "threshold_confirmed",
            "threshold_source",
        }
        assert set(payload["vegetation"]["vari"].keys()) == expected_vari_keys, f"VARI keys mismatch: {set(payload['vegetation']['vari'].keys()) ^ expected_vari_keys}"
        assert payload["vegetation"]["vari"]["band"] in ("LOWER_TAIL", "BELOW_TYPICAL", "TYPICAL", "ABOVE_TYPICAL")
        assert payload["vegetation"]["vari"]["band_basis"] == "within_scan_percentile"
        assert isinstance(payload["vegetation"]["vari"]["threshold_confirmed"], bool)
        assert payload["vegetation"]["vari"]["source"] == "measured"
    else:
        expected_vari_fallback_keys = {
            "mean",
            "reason",
            "min_fraction_threshold",
            "threshold_source",
            "threshold_confirmed",
            "source",
        }
        assert set(payload["vegetation"]["vari"].keys()) == expected_vari_fallback_keys

    # 5.3 ExG Sub-Block
    if payload["vegetation"]["canopy_cover"]["status"] == "OK":
        expected_exg_keys = {"mean", "source", "threshold_confirmed", "threshold_source"}
        assert set(payload["vegetation"]["exg"].keys()) == expected_exg_keys, f"ExG keys mismatch: {set(payload['vegetation']['exg'].keys()) ^ expected_exg_keys}"
    else:
        expected_exg_fallback_keys = {"mean", "reason", "min_fraction_threshold", "threshold_source", "threshold_confirmed", "source"}
        assert set(payload["vegetation"]["exg"].keys()) == expected_exg_fallback_keys

    # 5.4 TGI Sub-Block
    if payload["vegetation"]["canopy_cover"]["status"] == "OK":
        expected_tgi_keys = {"mean", "source", "threshold_confirmed", "threshold_source"}
        assert set(payload["vegetation"]["tgi"].keys()) == expected_tgi_keys, f"TGI keys mismatch: {set(payload['vegetation']['tgi'].keys()) ^ expected_tgi_keys}"
    else:
        expected_tgi_fallback_keys = {"mean", "reason", "min_fraction_threshold", "threshold_source", "threshold_confirmed", "source"}
        assert set(payload["vegetation"]["tgi"].keys()) == expected_tgi_fallback_keys

    # 5.5 DGCI Sub-Block
    if payload["vegetation"]["canopy_cover"]["status"] == "OK":
        if payload["vegetation"]["dgci"].get("reason") == "OUT_OF_DOMAIN_FRACTION_EXCEEDED":
            expected_dgci_keys = {"mean", "out_of_domain_fraction", "reason", "threshold", "threshold_source", "threshold_confirmed", "source"}
        else:
            expected_dgci_keys = {"mean", "out_of_domain_fraction", "source", "threshold_confirmed", "threshold_source"}
        assert set(payload["vegetation"]["dgci"].keys()) == expected_dgci_keys, f"DGCI keys mismatch: {set(payload['vegetation']['dgci'].keys()) ^ expected_dgci_keys}"
    else:
        expected_dgci_fallback_keys = {"mean", "reason", "min_fraction_threshold", "threshold_source", "threshold_confirmed", "source"}
        assert set(payload["vegetation"]["dgci"].keys()) == expected_dgci_fallback_keys

    # 6. Thermal Block
    expected_thermal_keys = {
        "available",
        "reason",
        "tc_c",
        "twet_c",
        "tdry_c",
        "cwsi",
        "flag",
        "thermal_source",
        "frame_utc",
    }
    assert set(payload["thermal"].keys()) == expected_thermal_keys, f"Thermal keys mismatch: {set(payload['thermal'].keys()) ^ expected_thermal_keys}"
    assert isinstance(payload["thermal"]["available"], bool)
    assert payload["thermal"]["thermal_source"] in ("hardware", "mock")

    # 7. NDVI Hardware Probe Block
    expected_ndvi_keys = {"available", "reason"}
    assert set(payload["ndvi"].keys()) == expected_ndvi_keys, f"NDVI keys mismatch: {set(payload['ndvi'].keys()) ^ expected_ndvi_keys}"
    assert isinstance(payload["ndvi"]["available"], bool)

    # 8. Sentinel-2 Satellite NDVI Block
    expected_ndvi_sat_keys = {
        "available",
        "reason",
        "source",
        "scene_date",
        "age_days",
        "ndvi_mean",
        "ndvi_std",
        "valid_pixel_count",
        "cloud_masked_fraction",
        "pixel_size_m",
        "reliability_note",
    }
    assert set(payload["ndvi_satellite"].keys()) == expected_ndvi_sat_keys, f"NDVI satellite keys mismatch: {set(payload['ndvi_satellite'].keys()) ^ expected_ndvi_sat_keys}"
    assert isinstance(payload["ndvi_satellite"]["available"], bool)
    assert payload["ndvi_satellite"]["source"] == "SENTINEL2_L2A_CDSE"

    # 9. Irrigation Block
    if payload["irrigation"]["available"]:
        expected_irrigation_keys = {
            "available",
            "method",
            "air_temp_c",
            "rh_pct",
            "t_min_24h_c",
            "t_max_24h_c",
            "t_mean_24h_c",
            "ra_mj_m2_day",
            "ra_mm_day",
            "ra_source",
            "ra_latitude_deg",
            "day_of_year",
            "et0_mm_day",
            "kc",
            "crop_et_mm_day",
            "soil1_v",
            "soil2_v",
            "battery_v",
            "samples_24h",
            "source",
        }
        assert set(payload["irrigation"].keys()) == expected_irrigation_keys, f"Irrigation keys mismatch: {set(payload['irrigation'].keys()) ^ expected_irrigation_keys}"
        assert payload["irrigation"]["source"] == "derived_fao56"
        assert payload["irrigation"]["method"] == "fao56_hargreaves_samani"
    else:
        expected_irrigation_keys = {"available", "reason"}
        assert set(payload["irrigation"].keys()) == expected_irrigation_keys, f"Irrigation keys mismatch: {set(payload['irrigation'].keys()) ^ expected_irrigation_keys}"
        assert isinstance(payload["irrigation"]["reason"], str)

    # 10. GPS Scan Summary Block
    expected_gps_keys = {"status", "point_count", "accuracy_note"}
    assert set(payload["gps"].keys()) == expected_gps_keys, f"GPS keys mismatch: {set(payload['gps'].keys()) ^ expected_gps_keys}"
    assert payload["gps"]["status"] in ("OK", "ABSENT")
    assert isinstance(payload["gps"]["point_count"], int)
    assert isinstance(payload["gps"]["accuracy_note"], str)

    # 11. Detections List Elements
    expected_det_keys = {
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
    for det in payload["detections"]:
        assert set(det.keys()) == expected_det_keys, f"Detection keys mismatch: {set(det.keys()) ^ expected_det_keys}"
        assert isinstance(det["class"], str)
        assert isinstance(det["confidence"], float)
        assert det["cross_source_reliability"] in ("TESTED_ROBUST", "TESTED_WEAK", "TESTED_FAILED", "UNTESTED")
        assert det["source"] == "measured"

    # 12. Disease List Elements
    expected_disease_keys = {"class", "confidence", "media_ids", "source"}
    for dis in payload["disease"]:
        assert set(dis.keys()) == expected_disease_keys, f"Disease keys mismatch: {set(dis.keys()) ^ expected_disease_keys}"
        assert isinstance(dis["class"], str)
        assert isinstance(dis["confidence"], float)
        assert isinstance(dis["media_ids"], list)
        assert dis["source"] == "measured"

    # 13. Pest List Elements
    expected_pest_keys = {
        "target_pest_context",
        "count_basis",
        "count_observed",
        "days_monitored",
        "daily_rate",
        "threshold_value",
        "threshold_unit",
        "threshold_available",
        "status",
        "threshold_verification_status",
        "classification_verification_status",
        "total_blobs_counted",
        "classification_source",
    }
    for p in payload["pest"]:
        assert set(p.keys()) == expected_pest_keys, f"Pest keys mismatch: {set(p.keys()) ^ expected_pest_keys}"
        assert isinstance(p["threshold_available"], bool)
        assert p["threshold_verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")

    # 14. Inputs List Elements
    expected_input_keys = {"name", "source_node", "status"}
    for inp in payload["inputs"]:
        assert set(inp.keys()) == expected_input_keys, f"Input keys mismatch: {set(inp.keys()) ^ expected_input_keys}"
        assert inp["source_node"] in ("POD", "MAST")
        assert inp["status"] in ("OK", "PENDING_CALIBRATION", "MOCK_PROVISIONAL", "ABSENT")

    # 15. Actions List Elements
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
    for act in payload["actions"]:
        assert set(act.keys()) == expected_action_keys, f"Action keys mismatch: {set(act.keys()) ^ expected_action_keys}"
        assert isinstance(act["rank"], int) and act["rank"] >= 1
        assert isinstance(act["template_id"], str)
        assert isinstance(act["action"], str)
        assert isinstance(act["rationale"], str)
        assert isinstance(act["params"], dict)
        assert act["verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")
        assert act["confidence"] in ("high", "medium", "low")
        assert act["advisory_only"] is True
        assert act["generated_by"] == "template"
        assert act["source"] == "derived"


def test_real_advisory_payload_schema_conformance():
    """
    Rigorously validates that a real synthesized advisory payload emitted by
    EdgeStorage.create_advisory() exactly matches the documented wire contract schema
    at all nesting levels.
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

        # Recursively validate the entire advisory document structure
        assert_advisory_payload_schema_recursive(payload)

        # Specific value checks
        assert payload["scan"]["frames_evaluated"] == 2
        assert payload["scan"]["tiles_classified"] == 18
        assert payload["crop_health"]["state"] == "DISEASE"
        assert payload["crop_health"]["crop"] == "rice"
        assert len(payload["detections"]) == 2
        assert payload["gps"]["status"] == "OK"
        assert payload["gps"]["point_count"] == 2
        assert len(payload["disease"]) == 1
        assert payload["disease"][0]["class"] == "rice__blast"
        assert len(payload["inputs"]) == 3
        assert len(payload["actions"]) >= 1

        # Verify GET /api/v1/advisory/latest against EdgeStorage
        latest_adv = storage.get_advisory("latest")
        assert latest_adv is not None
        assert latest_adv["advisory_id"] == "adv_contract_001"
        assert latest_adv["seq"] == payload["seq"]
        assert_advisory_payload_schema_recursive(latest_adv)

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


def test_gateway_all_endpoints_schema_conformance():
    """
    Rigorously validates response key-sets and types for every HTTP gateway endpoint:
    - GET  /api/v1/health
    - GET  /api/v1/sync/status
    - GET  /api/v1/manifest (including count field)
    - GET  /api/v1/advisory/latest (recursive validation)
    - GET  /api/v1/advisory/<id_or_seq> (recursive validation)
    - POST /api/v1/ack
    - GET  /api/v1/media/<id>
    """
    import urllib.request
    import urllib.error

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "gateway_conformance_test.db"
        storage = EdgeStorage(db_path=db_path)

        # Seed realistic scan and advisory
        scan_id = "scan_conf_001"
        storage.record_scan_start(
            scan_id=scan_id,
            started_utc="2026-09-21T05:00:00Z",
            source="test_cam.mp4",
        )
        storage.record_frame_event(
            scan_id=scan_id,
            frame_idx=0,
            timestamp_utc="2026-09-21T05:00:01Z",
            cell_id="cell_001",
            gate_passed=True,
            gate_metrics={"blur_score": 150.0, "displacement": 1.0},
            n_valid_tiles=9,
            frame_state="DISEASE",
            class_id=4,  # rice__blast
            confidence=0.95,
            tile_decisions=[{"state": "OK", "class_id": 4, "conf": 0.95}] * 9,
            source_image="frame_0.jpg",
            gps={"latitude": 28.52, "longitude": 77.58, "fix_quality": 1, "hdop": 1.1},
            canopy_cover=0.85,
            vari=0.25,
            exg=48.0,
            tgi=20.0,
            dgci=0.60,
            dgci_ood_frac=0.0,
            indices_status="OK",
        )
        storage.record_cell_verdict(
            scan_id=scan_id,
            cell_id="cell_001",
            state="DISEASE",
            class_id=4,
            score=0.95,
            n_frames=1,
            n_agree=1,
            canopy_cover=0.85,
        )
        storage.record_scan_end(
            scan_id=scan_id,
            ended_utc="2026-09-21T05:00:10Z",
            frames_captured=1,
            frames_evaluated=1,
            tiles_classified=9,
        )
        created_adv = storage.create_advisory(
            scan_id=scan_id,
            advisory_id="adv_conf_001",
            replay=False,
            field_id="F01",
            inference_backend="trt",
        )

        gateway = EdgeGateway(host="127.0.0.1", port=0, storage=storage, db_path=db_path, allow_mock=True)
        gateway.start_background()
        actual_port = gateway.server.server_address[1]
        base_url = f"http://127.0.0.1:{actual_port}"

        try:
            # 1. GET /api/v1/health
            req = urllib.request.Request(f"{base_url}/api/v1/health")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                health = json.loads(resp.read().decode("utf-8"))
                expected_health_keys = {
                    "device",
                    "schema_version",
                    "server_time_utc",
                    "gps_time_valid",
                    "clock_source",
                    "advisory_count",
                    "latest_seq",
                    "storage_free_kb",
                    "syncing",
                    "sync_state",
                    "storage",
                    "pod_ready",
                    "scan_state",
                }
                assert set(health.keys()) == expected_health_keys, f"Health keys mismatch: {set(health.keys()) ^ expected_health_keys}"
                assert isinstance(health["device"], str)
                assert health["schema_version"] == "1.0"
                assert isinstance(health["server_time_utc"], str)
                assert isinstance(health["gps_time_valid"], bool)
                assert health["clock_source"] in ("gps", "rtc", "filesystem", "phone")
                assert isinstance(health["advisory_count"], int) and health["advisory_count"] >= 1
                assert isinstance(health["latest_seq"], int) and health["latest_seq"] >= 1
                assert isinstance(health["storage_free_kb"], int) and health["storage_free_kb"] >= 0
                assert isinstance(health["syncing"], bool)
                assert health["sync_state"] in ("IDLE", "STA_SYNC")
                assert isinstance(health["storage"], dict)
                assert isinstance(health["pod_ready"], bool)
                assert isinstance(health["scan_state"], str)

            # 2. GET /api/v1/sync/status
            req = urllib.request.Request(f"{base_url}/api/v1/sync/status")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                sync_status = json.loads(resp.read().decode("utf-8"))
                expected_sync_keys = {
                    "sync_in_progress",
                    "last_success_utc",
                    "last_attempt_utc",
                    "last_result",
                    "mast_data_age_s",
                    "records_pulled",
                    "trap_images_pulled",
                }
                assert set(sync_status.keys()) == expected_sync_keys, f"Sync status keys mismatch: {set(sync_status.keys()) ^ expected_sync_keys}"
                assert isinstance(sync_status["sync_in_progress"], bool)
                assert sync_status["last_success_utc"] is None or isinstance(sync_status["last_success_utc"], str)
                assert sync_status["last_attempt_utc"] is None or isinstance(sync_status["last_attempt_utc"], str)
                assert sync_status["last_result"] is None or sync_status["last_result"] in ("OK", "MAST_NOT_FOUND", "PARTIAL", "ERROR")
                assert sync_status["mast_data_age_s"] is None or (isinstance(sync_status["mast_data_age_s"], int) and sync_status["mast_data_age_s"] >= 0)
                assert isinstance(sync_status["records_pulled"], int) and sync_status["records_pulled"] >= 0
                assert isinstance(sync_status["trap_images_pulled"], int) and sync_status["trap_images_pulled"] >= 0

            # 3. GET /api/v1/manifest (with count)
            req = urllib.request.Request(f"{base_url}/api/v1/manifest?since=0&limit=10")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                manifest = json.loads(resp.read().decode("utf-8"))
                expected_manifest_keys = {
                    "schema_version",
                    "count",
                    "advisories",
                    "truncated",
                }
                assert set(manifest.keys()) == expected_manifest_keys, f"Manifest keys mismatch: {set(manifest.keys()) ^ expected_manifest_keys}"
                assert manifest["schema_version"] == "1.0"
                assert isinstance(manifest["count"], int)
                assert manifest["count"] == len(manifest["advisories"])
                assert isinstance(manifest["truncated"], bool)
                assert isinstance(manifest["advisories"], list)
                assert len(manifest["advisories"]) >= 1

                expected_adv_item_keys = {
                    "advisory_id",
                    "seq",
                    "generated_at_utc",
                    "bytes",
                    "replay",
                    "inference_backend",
                }
                for item in manifest["advisories"]:
                    assert set(item.keys()) == expected_adv_item_keys, f"Manifest advisory item keys mismatch: {set(item.keys()) ^ expected_adv_item_keys}"
                    assert isinstance(item["advisory_id"], str)
                    assert isinstance(item["seq"], int) and item["seq"] >= 1
                    assert isinstance(item["generated_at_utc"], str)
                    assert isinstance(item["bytes"], int) and item["bytes"] >= 0
                    assert isinstance(item["replay"], bool)
                    assert item["inference_backend"] in ("trt", "onnx", "mock")

            # 4. GET /api/v1/advisory/latest (recursive)
            req = urllib.request.Request(f"{base_url}/api/v1/advisory/latest")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                latest_adv = json.loads(resp.read().decode("utf-8"))
                assert latest_adv["advisory_id"] == "adv_conf_001"
                assert latest_adv["seq"] == created_adv["seq"]
                assert_advisory_payload_schema_recursive(latest_adv)

            # 5. GET /api/v1/advisory/<id_or_seq> (recursive)
            # By ID
            req = urllib.request.Request(f"{base_url}/api/v1/advisory/adv_conf_001")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                adv_by_id = json.loads(resp.read().decode("utf-8"))
                assert adv_by_id["advisory_id"] == "adv_conf_001"
                assert_advisory_payload_schema_recursive(adv_by_id)
            # By Seq
            req = urllib.request.Request(f"{base_url}/api/v1/advisory/{created_adv['seq']}")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                adv_by_seq = json.loads(resp.read().decode("utf-8"))
                assert adv_by_seq["advisory_id"] == "adv_conf_001"
                assert_advisory_payload_schema_recursive(adv_by_seq)

            # 6. POST /api/v1/ack
            ack_body = json.dumps({"advisory_id": "adv_conf_001"}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/v1/ack",
                data=ack_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                ack_resp = json.loads(resp.read().decode("utf-8"))
                assert set(ack_resp.keys()) == {"status", "acked"}
                assert ack_resp["status"] == "ok"
                assert ack_resp["acked"] == "adv_conf_001"

            # 7. GET /api/v1/media/<id> -> 410 Gone
            req = urllib.request.Request(f"{base_url}/api/v1/media/img_test_001")
            try:
                urllib.request.urlopen(req)
                pytest.fail("Expected HTTP 410 Gone for GET /api/v1/media/<id>")
            except urllib.error.HTTPError as e:
                assert e.code == 410
                media_resp = json.loads(e.read().decode("utf-8"))
                assert set(media_resp.keys()) == {"error", "reason"}
                assert media_resp["error"] == "gone"
                assert media_resp["reason"] == "retention_pruned"

        finally:
            gateway.stop()
            storage.close()


