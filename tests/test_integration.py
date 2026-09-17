"""
tests/test_integration.py — End-to-End Edge Pipeline Integration Test Suite (BUILD_CHECKLIST Step 35 / J6).

Execution Flow:
  1. Real Video Source: `data/video/test_video_from_dataset_images.mp4` (compiled from real dataset images).
  2. Edge Pipeline: 4-thread runtime (`edge/pipeline.py`) running real inference with ONNX Runtime (`artifacts/onnx/model_a_fused.onnx`)
     standing in for NVIDIA TensorRT on development/macOS hosts.
  3. Rejection & Consensus: Multi-tile quality gating, open-set energy gate, prior adjustment, spatial & temporal consensus.
  4. Rules Engine: Deterministic agronomic action synthesis.
  5. Ground Mast Telemetry: HTTP `POST /api/v1/mast/telemetry` with weather and water level data.
  6. Sticky-Trap Node: HTTP `POST /api/v1/trap/upload` with sticky trap card image evaluated by Model B.
  7. Mobile App Gateway: HTTP `GET /api/v1/advisory/<id>` fetching authoritative advisory JSON.
  8. Contract Validation: Validates the entire payload against `docs/PAYLOAD_CONTRACT.md`.
"""
import datetime
import json
import socket
import tempfile
import time
import urllib.request
from pathlib import Path
import numpy as np
import pytest
import cv2

from configs.classes import CLASS_NAMES
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from edge.pipeline import EdgePipeline
from edge.storage import EdgeStorage
from gateway.server import EdgeGateway

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_j6_end_to_end_pipeline_integration():
    """
    J6.1 & J6.2: End-to-end integration test running real pipeline, gateway HTTP endpoints,
    trap card upload, mast telemetry POST, and schema validation against PAYLOAD_CONTRACT.md.
    """
    video_path = REPO_ROOT / "data/video/test_video_from_dataset_images.mp4"
    onnx_path = REPO_ROOT / "artifacts/onnx/model_a_fused.onnx"

    assert video_path.exists(), f"Real video source missing at {video_path}"
    assert onnx_path.exists(), f"Model A fused ONNX missing at {onnx_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "integration_edge.db"
        jsonl_path = Path(tmpdir) / "integration_events.jsonl"
        storage = EdgeStorage(db_path=db_path)

        # 1. Initialize and start HTTP Gateway
        port = _find_free_port()
        gw = EdgeGateway(host="127.0.0.1", port=port, storage=storage)
        gw.start_background()

        try:
            # 2. Ingest Mast Telemetry via HTTP POST
            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            telemetry_payload = {
                "node_id": "SIH-NODE-01",
                "recorded_at_utc": now_iso,
                "firmware_version": "1.0.0",
                "air_temp_c": 29.5,
                "humidity_pct": 68.0,
                "canopy_temp_c": 27.2,
                "soil_moisture_v": [1.85, 1.88, 1.83],
                "water_level_mm": -30.0,
            }
            req_telemetry = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/mast/telemetry",
                data=json.dumps(telemetry_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_telemetry, timeout=5.0) as resp:
                assert resp.status == 200
                res_tel = json.loads(resp.read().decode("utf-8"))
                assert res_tel["status"] == "ok"

            # 3. Ingest Sticky-Trap Image via HTTP POST /api/v1/trap/upload
            # Generate a synthetic trap image with 10 blobs for Model B
            dummy_trap = np.ones((400, 400, 3), dtype=np.uint8) * 200
            for pt in [(50, 50), (100, 150), (200, 200), (300, 120), (150, 300)]:
                cv2.circle(dummy_trap, pt, 6, (20, 20, 20), -1)
            _, trap_bytes = cv2.imencode(".jpg", dummy_trap)

            req_trap = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/trap/upload?trap_id=TRAP_FIELD_01&days=3.0&allow_provisional=true",
                data=trap_bytes.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                method="POST",
            )
            with urllib.request.urlopen(req_trap, timeout=10.0) as resp:
                assert resp.status == 200
                res_trap = json.loads(resp.read().decode("utf-8"))
                assert res_trap["status"] == "ok"
                assert "pest" in res_trap

            # 4. Run Edge Pipeline on real video with ONNX Runtime backend
            scan_id = "integration_scan_001"
            pipeline = EdgePipeline(
                source=str(video_path),
                onnx_path=onnx_path,
                dry_run=False,          # Non-dry-run real inference!
                max_frames=6,           # Process 6 video frames (54 tiles)
                output_jsonl=jsonl_path,
                db_path=db_path,
                scan_id=scan_id,
                realtime=False,
                days_since_planting=40,
                total_cycle_days=150,
            )

            metrics = pipeline.run()
            assert metrics["frames_seen"] >= 6
            assert metrics["tiles_classified"] >= 9

            # 5. Fetch Synthesized Advisory via HTTP GET /api/v1/advisory/<id>
            adv_id = pipeline.t4_decision.last_advisory["advisory_id"]
            req_adv = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/advisory/{adv_id}",
                method="GET",
            )
            with urllib.request.urlopen(req_adv, timeout=5.0) as resp:
                assert resp.status == 200
                advisory = json.loads(resp.read().decode("utf-8"))

            # 6. Authoritative PAYLOAD_CONTRACT.md Schema Validations
            assert advisory["schema_version"] == "1.0"
            assert advisory["advisory_id"] == adv_id
            assert advisory["seq"] >= 1
            assert "generated_at_utc" in advisory

            # Scan block
            scan = advisory["scan"]
            assert scan["frames_evaluated"] >= 1
            assert scan["tiles_classified"] >= 9

            # Crop Health block
            ch = advisory["crop_health"]
            assert ch["state"] in ("HEALTHY", "DISEASE", "UNCERTAIN", "NOT_CROP", "NO_DATA")
            assert ch["source"] == "measured"

            # Growth Stage block
            gs = advisory["growth_stage"]
            assert gs["crop"] in ("rice", "wheat", "sugarcane", None)
            assert gs["verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")

            # Hardware-Gated Blocks
            assert advisory["thermal"]["available"] is False
            assert advisory["ndvi"]["available"] is False
            
            # Irrigation block (enabled because fresh mast telemetry was POSTed)
            irr = advisory["irrigation"]
            assert irr["available"] is True
            assert irr["method"] == "fao56_hargreaves_samani"
            assert irr["air_temp_c"] == 29.5
            assert irr["crop_et_mm_day"] > 0.0

            # Pest block (populated from uploaded trap card)
            assert len(advisory["pest"]) >= 1
            for p in advisory["pest"]:
                assert p["status"] in ("BELOW_ETL", "AT_ETL", "ABOVE_ETL", "NO_PUBLISHED_ETL", "NOT_SAMPLED_BY_STICKY_TRAP", "UNKNOWN_PEST", "CARD_SATURATED", "INVALID_MONITORING_WINDOW", "MISSING_DEPLOYMENT_TIMESTAMP")
                assert "threshold_available" in p
                assert p["threshold_verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")

            # Actions block
            assert len(advisory["actions"]) >= 1
            for act in advisory["actions"]:
                assert act["verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")
                assert act["advisory_only"] is True

        finally:
            gw.stop()
