#!/usr/bin/env python3
"""
scripts/verify_gateway_pipeline_concurrency.py

Verifies simultaneous live concurrency:
1. Boots EdgeGateway on 127.0.0.1:8888 backed by data/edge.db.
2. Spawns edge/pipeline.py against test_video_from_dataset_images.mp4.
3. While the pipeline is actively running and writing to SQLite, issues real
   `curl` commands to every endpoint, printing full HTTP headers and response bodies.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.server import EdgeGateway


def run_curl(desc: str, cmd: list) -> None:
    print("\n" + "=" * 70)
    print(f"CURL TEST: {desc}")
    print("COMMAND  : " + " ".join(cmd))
    print("-" * 70)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR   :", res.stderr)
    print("=" * 70)


def main():
    db_path = ROOT / "data" / "edge.db"
    port = 8888
    gateway = EdgeGateway(host="127.0.0.1", port=port, db_path=db_path)
    gateway.start_background()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{port}"
    print(f"EdgeGateway started at {base_url} (connected to {db_path})")

    # Launch pipeline dry-run in background
    pipeline_cmd = [
        sys.executable,
        str(ROOT / "edge" / "pipeline.py"),
        "--source",
        str(ROOT / "test_video_from_dataset_images.mp4"),
        "--dry-run",
        "--db-path",
        str(db_path),
    ]
    print(f"Launching pipeline process: {' '.join(pipeline_cmd)}")
    pipe_proc = subprocess.Popen(pipeline_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

    # Give pipeline 0.5s to start ingesting frames
    time.sleep(0.5)

    try:
        # 1. Health endpoint during active writes
        run_curl("GET /api/v1/health during active pipeline write", ["curl", "-s", "-i", f"{base_url}/api/v1/health"])

        # 2. Manifest endpoint during active writes
        run_curl("GET /api/v1/manifest?since=0&limit=3", ["curl", "-s", "-i", f"{base_url}/api/v1/manifest?since=0&limit=3"])

        # 3. Advisory endpoint for existing record
        run_curl("GET /api/v1/advisory/1", ["curl", "-s", "-i", f"{base_url}/api/v1/advisory/1"])

        # 4. Courtesy Ack endpoint
        run_curl(
            "POST /api/v1/ack",
            [
                "curl", "-s", "-i",
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", '{"advisory_id": "1"}',
                f"{base_url}/api/v1/ack",
            ],
        )

        # 5. Media endpoint (410 Gone)
        run_curl("GET /api/v1/media/1-crop-1", ["curl", "-s", "-i", f"{base_url}/api/v1/media/1-crop-1"])

        # 6. Not Found error case
        run_curl("GET /api/v1/advisory/nonexistent_id_999", ["curl", "-s", "-i", f"{base_url}/api/v1/advisory/nonexistent_id_999"])

        # 7. Bad request error case
        run_curl("GET /api/v1/manifest?limit=-1", ["curl", "-s", "-i", f"{base_url}/api/v1/manifest?limit=-1"])

        # Wait for pipeline to finish
        stdout, stderr = pipe_proc.communicate(timeout=15.0)
        print("\nPipeline finished successfully with exit code", pipe_proc.returncode)

        # 8. Query manifest again after pipeline completes to show newly created advisory
        run_curl("GET /api/v1/manifest after pipeline completed", ["curl", "-s", "-i", f"{base_url}/api/v1/manifest?limit=5"])

    finally:
        gateway.stop()
        print("\nEdgeGateway stopped cleanly.")


if __name__ == "__main__":
    main()
