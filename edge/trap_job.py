#!/usr/bin/env python3
"""
edge/trap_job.py — Offline Sticky-Trap Processing Job for Model B (Jetson Nano / Gateway).

Adheres to Python 3.6 compatibility for Jetson Nano (JetPack 4.6.4).
Pipeline:
  1. Validates physical scale (mm_per_pixel). Fails loudly if uncalibrated without --allow-provisional.
  2. Watershed Segmentation (core/trap_segmentation.py:segment_trap_blobs).
  3. Model B Inference (core/trap_segmentation.py:classify_trap_blobs).
  4. Cumulative ETL Evaluation (core/trap_segmentation.py:evaluate_trap_counts_against_etl).
  5. Persistence to SQLite edge database (edge/storage.py:record_trap_job).

Callable as:
  - CLI on a single file: python3 -m edge.trap_job --image card.jpg --scale 0.125
  - Watcher on an inbox directory: python3 -m edge.trap_job --watch data/trap_inbox --scale 0.125
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# 1. Loud Standing Guard for ONNX Runtime with exact wheel recommendation
try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required for Model B inference (%s). "
        "For Jetson Nano (JetPack 4.6.4 / Python 3.6 aarch64), install: "
        "pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl" % e
    )

import cv2
import numpy as np

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trap_segmentation import (
    NOMINAL_MM_PER_PIXEL_DESIGN_TARGET,
    classify_trap_blobs,
    evaluate_trap_counts_against_etl,
    segment_trap_blobs,
)
from edge.storage import DEFAULT_DB_PATH, EdgeStorage, get_utc_iso_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edge.trap_job")


def validate_scale_parameters(
    scale: Optional[float] = None,
    allow_provisional: bool = False,
) -> Tuple[float, str]:
    """
    Validates physical scale (mm/pixel).
    Fails loudly if scale is None and allow_provisional is False.
    Returns (validated_scale, scale_status).
    """
    if scale is None:
        if not allow_provisional:
            raise ValueError(
                "Physical scale MM_PER_PIXEL is uncalibrated and no explicit --scale was provided. "
                "Pass an explicit --scale or use --allow-provisional for unconfirmed prototype testing."
            )
        resolved_scale = float(NOMINAL_MM_PER_PIXEL_DESIGN_TARGET)
        scale_status = "PROVISIONAL"
        logger.warning(
            "JOB WARNING: Running with PROVISIONAL scale %.4f mm/pixel. "
            "Real field card calibration pending physical measurement.",
            resolved_scale,
        )
    else:
        if float(scale) <= 0:
            raise ValueError("Scale must be positive, got %s" % scale)
        resolved_scale = float(scale)
        if allow_provisional:
            scale_status = "PROVISIONAL"
            logger.warning(
                "JOB WARNING: Explicit scale %.4f mm/pixel tagged as PROVISIONAL.",
                resolved_scale,
            )
        else:
            scale_status = "CALIBRATED"
            logger.info("Using calibrated scale %.4f mm/pixel.", resolved_scale)

    return resolved_scale, scale_status


def process_trap_image(
    image_path: Union[str, Path],
    scale: Optional[float] = None,
    allow_provisional: bool = False,
    trap_id: str = "TRAP_01",
    placed_at: Optional[Union[str, float]] = None,
    days_monitored: Optional[float] = None,
    storage: Optional[EdgeStorage] = None,
    onnx_path: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end processing of a single sticky trap card image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError("Trap image not found: %s" % image_path)

    # 1. Scale validation
    resolved_scale, scale_status = validate_scale_parameters(
        scale=scale,
        allow_provisional=allow_provisional,
    )

    # 2. Image ingestion
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError("Failed to decode image from %s" % image_path)

    # 3. Deterministic Watershed Segmentation
    blobs = segment_trap_blobs(
        bgr,
        abs_floor_px=None,
        mm_per_pixel=resolved_scale,
    )
    total_blobs = len(blobs)
    logger.info(
        "Segmented %d blobs from %s (scale=%.4f mm/px, scale_status=%s)",
        total_blobs,
        image_path.name,
        resolved_scale,
        scale_status,
    )

    # 4. Model B Patch Classification
    class_results = classify_trap_blobs(blobs, onnx_path=onnx_path)
    morph_dist = class_results.get("morphological_distribution", {})

    # 5. ICAR/NIPHM Cumulative ETL Evaluation
    # Default to 1.0 day if no deployment timestamp or days passed
    effective_days = days_monitored
    effective_placed_at = placed_at
    if effective_days is None and effective_placed_at is None:
        effective_days = 1.0
        effective_placed_at = get_utc_iso_now()

    pest_counts = {"sugarcane_whitefly": total_blobs}
    etl_results = evaluate_trap_counts_against_etl(
        pest_counts=pest_counts,
        card_replaced_at=effective_placed_at,
        days_monitored=effective_days,
        total_blobs_counted=total_blobs,
        morphological_distribution=morph_dist,
    )

    if not job_id:
        now_ts = int(time.time() * 1000)
        job_id = "trap_job_%s_%d" % (trap_id, now_ts)

    # Enrich each pest result with job metadata and scale status
    for item in etl_results:
        item["scale_status"] = scale_status
        item["scale_mm_per_pixel"] = resolved_scale
        item["trap_id"] = str(trap_id)
        item["job_id"] = str(job_id)

    primary_etl = etl_results[0] if etl_results else {}
    etl_status = primary_etl.get("status", "UNKNOWN")

    # 6. Persistence to SQLite edge storage
    if storage is None:
        storage = EdgeStorage()

    row_id = storage.record_trap_job(
        trap_id=str(trap_id),
        job_id=str(job_id),
        image_path=str(image_path),
        placed_at_utc=str(effective_placed_at) if effective_placed_at else None,
        days_monitored=effective_days,
        total_blobs_counted=total_blobs,
        scale_mm_per_pixel=resolved_scale,
        scale_status=scale_status,
        etl_status=etl_status,
        pest_payload=primary_etl,
    )

    summary = {
        "status": "ok",
        "job_id": job_id,
        "record_id": row_id,
        "trap_id": trap_id,
        "image_path": str(image_path),
        "total_blobs_counted": total_blobs,
        "scale_mm_per_pixel": resolved_scale,
        "scale_status": scale_status,
        "etl_status": etl_status,
        "morphological_distribution": morph_dist,
        "pest": primary_etl,
    }
    logger.info(
        "Job %s completed: %d blobs, ETL=%s, scale_status=%s",
        job_id,
        total_blobs,
        etl_status,
        scale_status,
    )
    return summary


def run_watcher(
    inbox_dir: Union[str, Path],
    scale: Optional[float] = None,
    allow_provisional: bool = False,
    trap_id: str = "TRAP_01",
    poll_interval: float = 2.0,
    db_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Watches an inbox directory for incoming sticky-trap card images and processes them.
    """
    inbox_path = Path(inbox_dir)
    inbox_path.mkdir(parents=True, exist_ok=True)
    processed_path = inbox_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    storage = EdgeStorage(db_path=db_path) if db_path else EdgeStorage()
    logger.info("Starting trap inbox watcher on %s (poll_interval=%.1fs)", inbox_path, poll_interval)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    iterations = 0

    while True:
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1

        try:
            candidates = [
                p for p in inbox_path.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ]
            for img_file in sorted(candidates):
                logger.info("Watcher picked up %s", img_file.name)
                sidecar = img_file.with_suffix(".json")
                item_trap_id = trap_id
                item_placed_at = None
                item_days = None
                item_scale = scale
                item_allow_prov = allow_provisional

                if sidecar.exists():
                    try:
                        with open(str(sidecar), "r") as sf:
                            meta = json.load(sf)
                            item_trap_id = meta.get("trap_id", item_trap_id)
                            item_placed_at = meta.get("placed_at", item_placed_at)
                            item_days = meta.get("days_monitored", item_days)
                            if "scale" in meta:
                                item_scale = float(meta["scale"])
                            if "allow_provisional" in meta:
                                item_allow_prov = bool(meta["allow_provisional"])
                    except Exception as ex:
                        logger.warning("Failed to parse sidecar %s: %s", sidecar.name, ex)

                try:
                    process_trap_image(
                        image_path=img_file,
                        scale=item_scale,
                        allow_provisional=item_allow_prov,
                        trap_id=item_trap_id,
                        placed_at=item_placed_at,
                        days_monitored=item_days,
                        storage=storage,
                        onnx_path=onnx_path,
                    )
                    dest_img = processed_path / img_file.name
                    if dest_img.exists():
                        dest_img.unlink()
                    shutil.move(str(img_file), str(dest_img))
                    if sidecar.exists():
                        dest_sidecar = processed_path / sidecar.name
                        if dest_sidecar.exists():
                            dest_sidecar.unlink()
                        shutil.move(str(sidecar), str(dest_sidecar))
                except Exception as proc_ex:
                    logger.error("Error processing %s: %s", img_file.name, proc_ex)

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Inbox watcher stopped by user.")
            break
        except Exception as e:
            logger.error("Watcher iteration error: %s", e)
            time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model B Sticky Trap Processing Job")
    parser.add_argument("--image", type=str, help="Path to a single trap image file")
    parser.add_argument("--watch", type=str, help="Directory to watch for incoming images")
    parser.add_argument("--scale", type=float, default=None, help="Explicit scale in mm/pixel")
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Permit provisional scale evaluation when physical calibration is unconfirmed",
    )
    parser.add_argument("--trap-id", type=str, default="TRAP_01", help="Trap identifier")
    parser.add_argument("--placed-at", type=str, default=None, help="Deployment timestamp ISO string")
    parser.add_argument("--days", type=float, default=None, help="Days monitored")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to SQLite edge.db")
    parser.add_argument("--onnx-path", type=str, default=None, help="Path to Model B ONNX model")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Watcher poll interval in seconds")

    args = parser.parse_args()

    if args.image:
        try:
            res = process_trap_image(
                image_path=args.image,
                scale=args.scale,
                allow_provisional=args.allow_provisional,
                trap_id=args.trap_id,
                placed_at=args.placed_at,
                days_monitored=args.days,
                storage=EdgeStorage(db_path=args.db_path),
                onnx_path=args.onnx_path,
            )
            print(json.dumps(res, indent=2))
            return 0
        except Exception as e:
            logger.error("Fatal error processing %s: %s", args.image, e)
            return 1

    elif args.watch:
        run_watcher(
            inbox_dir=args.watch,
            scale=args.scale,
            allow_provisional=args.allow_provisional,
            trap_id=args.trap_id,
            poll_interval=args.poll_interval,
            db_path=args.db_path,
            onnx_path=args.onnx_path,
        )
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
