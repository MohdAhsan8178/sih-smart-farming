#!/usr/bin/env python3
"""
scripts/test_gateway_abstention_curl.py

D3: Trace UNCERTAIN_NON_TARGET and abstention semantics through edge storage, rules engine,
and offline HTTP API gateway, capturing real curl requests and JSON responses.
"""
import sys
import os
import json
import time
import tempfile
import subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.storage import EdgeStorage
from gateway.server import EdgeGateway
from core.trap_segmentation import classify_trap_blobs, evaluate_trap_counts_against_etl

print("=================================================================")
print("PART D3: TRACING UNCERTAIN_NON_TARGET THROUGH STORAGE & GATEWAY")
print("=================================================================")

# 1. Run trap classification on sample crops (generating both confident and abstained blobs)
print("\n--- STEP 1: CLASSIFY TRAP BLOBS WITH ABSTENTION PATH ---")
blobs = []
# Blob 1: real small_pale_winged crop (confident)
p1 = ROOT / "data" / "processed" / "model_b_patches_v4" / "small_pale_winged" / "small_pale_winged_000646.png"
if p1.exists():
    img1 = cv2.imread(str(p1))
    blobs.append((img1, (32, 32), 25.0, 1.0))

# Blob 2: flat yellow debris crop (confident debris)
p2 = ROOT / "data" / "processed" / "model_b_patches_v4" / "debris" / "debris_000000.png"
if p2.exists():
    img2 = cv2.imread(str(p2))
    blobs.append((img2, (64, 64), 10.0, 1.0))

# Blob 3: synthetic noise/random pattern (forces UNCERTAIN_NON_TARGET via energy or confidence)
np.random.seed(42)
synthetic_odd = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
blobs.append((synthetic_odd, (100, 100), 50.0, 1.0))

# Blob 4: another unusual pattern (checkerboard/edge)
checker = np.zeros((64, 64, 3), dtype=np.uint8)
checker[::8, :] = 255
checker[:, ::8] = 255
blobs.append((checker, (120, 120), 40.0, 1.0))

onnx_target = ROOT / "artifacts" / "onnx" / "model_b.onnx"
if not onnx_target.exists():
    onnx_target = ROOT / "artifacts" / "onnx" / "model_b_calibrated.onnx"

trap_result = classify_trap_blobs(blobs, onnx_path=str(onnx_target))

print(f"Total blobs submitted to classifier : {len(blobs)}")
print(f"Total blobs counted (watershed)     : {trap_result['total_blobs_counted']}")
print("Morphological distribution:")
for k, v in trap_result["morphological_distribution"].items():
    print(f"  {k:<22}: count={v['count']}, fraction={v['fraction']:.4f}")
print(f"Predictions list: {trap_result['predictions']}")

# 2. Evaluate against ETL
print("\n--- STEP 2: EVALUATE COUNTS AGAINST ICAR/NIPHM ETL (WATERSHED-PRIMARY) ---")
etl_results = evaluate_trap_counts_against_etl(
    pest_counts={"sugarcane_whitefly": trap_result["total_blobs_counted"]},
    days_monitored=3.0,
    card_saturated=False,
    card_coverage=0.08,
    total_blobs_counted=trap_result["total_blobs_counted"],
    morphological_distribution=trap_result["morphological_distribution"],
)
print("ETL Evaluation Results (Watershed-Primary):")
for r in etl_results:
    print(f"  Target Pest: {r['target_pest_context']:<22} | Basis: {r['count_basis']} | Count: {r['count_observed']} | Status: {r['status']:<15} | Daily Rate: {r.get('daily_rate')} | Threshold: {r.get('threshold_value')}")

# 3. Store into EdgeStorage and serve through Gateway
print("\n--- STEP 3: STORE OBSERVATION AND LAUNCH GATEWAY SERVER ---")
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "field_pod.db"
    storage = EdgeStorage(db_path=db_path)
    
    # Store advisory containing trap distribution and ETL status
    scan_id = "trap_scan_20260916_01"
    advisory_id = "adv_trap_001"
    storage.record_scan_start(scan_id)
    storage.record_frame_event(
        scan_id=scan_id,
        frame_idx=0,
        timestamp_utc="2026-09-16T17:25:00Z",
        cell_id="sticky_trap_card_01",
        gate_passed=True,
        gate_metrics={"total_blobs": trap_result["total_blobs_counted"]},
        n_valid_tiles=1,
        frame_state="MONITORED",
        class_id=0,
        confidence=0.95,
        tile_decisions=[]
    )
    storage.record_scan_end(scan_id, frames_captured=1, frames_evaluated=1, tiles_classified=len(blobs))
    
    pest_block = etl_results
    
    # Create structured advisory with trap telemetry
    adv = storage.create_advisory(
        scan_id=scan_id,
        advisory_id=advisory_id,
        pest_data=pest_block,
        replay=True
    )
    
    # Start EdgeGateway on ephemeral port
    gateway = EdgeGateway(host="127.0.0.1", port=0, storage=storage, db_path=db_path)
    gateway.start_background()
    time.sleep(0.5)
    
    actual_port = gateway.server.server_address[1]
    base_url = f"http://127.0.0.1:{actual_port}"
    print(f"Gateway listening at {base_url}")
    
    # 4. Perform real curl commands
    print("\n--- STEP 4: REAL CURL VERIFICATION ---")
    
    print(f"\n>>> curl -s -i http://127.0.0.1:{actual_port}/api/v1/health")
    res_health = subprocess.run(["curl", "-s", "-i", f"{base_url}/api/v1/health"], capture_output=True, text=True)
    print(res_health.stdout)
    
    print(f"\n>>> curl -s -i http://127.0.0.1:{actual_port}/api/v1/manifest")
    res_manifest = subprocess.run(["curl", "-s", "-i", f"{base_url}/api/v1/manifest"], capture_output=True, text=True)
    print(res_manifest.stdout)
    
    print(f"\n>>> curl -s -i http://127.0.0.1:{actual_port}/api/v1/advisory/{advisory_id}")
    res_advisory = subprocess.run(["curl", "-s", "-i", f"{base_url}/api/v1/advisory/{advisory_id}"], capture_output=True, text=True)
    print(res_advisory.stdout)
    
    # Stop gateway
    gateway.stop()
    print("\nGateway stopped cleanly. Part D3 trace complete!")
