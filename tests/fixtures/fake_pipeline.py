#!/usr/bin/env python3
"""
Test fixture: Standalone fake pipeline subprocess for testing gateway process-monitoring,
signal-handling, and recovery without hardware or full neural network inference.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.storage import EdgeStorage, get_utc_iso_now


def write_status(
    status_file: Path,
    state: str,
    scan_id: str,
    field_id: str,
    crop: str,
    replay: bool,
    started_utc: str,
    elapsed_s: int,
    max_duration_s: int,
    frames_seen: int,
    frames_used: int,
    advisory_id=None,
    stop_reason=None,
):
    payload = {
        "state": state,
        "scan_id": scan_id,
        "field_id": field_id,
        "crop": crop,
        "replay": replay,
        "started_utc": started_utc,
        "elapsed_s": int(elapsed_s),
        "max_duration_s": int(max_duration_s),
        "counts": {
            "frames_seen": int(frames_seen),
            "frames_used": int(frames_used),
            "stretches": 0,
            "healthy": 0,
            "need_look": 0,
            "unclear": 0,
            "not_crop": 0,
        },
        "thermal_c_latest": None,
        "field_station": {
            "reachable": None,
            "readings_collected": None,
            "last_reading_utc": None,
        },
        "warnings": [],
        "alerts": [],
        "advisory_id": advisory_id,
        "stop_reason": stop_reason,
    }
    tmp = Path(str(status_file) + (".tmp.%d" % os.getpid()))
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(str(tmp), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(status_file))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Fake Pipeline for Gateway Testing")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--until-stopped", action="store_true")
    parser.add_argument("--scan-id", type=str, default=None)
    parser.add_argument("--field-id", type=str, default=None)
    parser.add_argument("--crop", type=str, default=None)
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="mock")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--max-duration-s", type=int, default=1800)
    parser.add_argument("--time-source", type=str, default="filesystem")
    args = parser.parse_args()

    fake_mode = os.environ.get("FAKE_PIPELINE_MODE", "")
    if args.crop == "trigger-error" or fake_mode == "exit1":
        sys.exit(1)

    if fake_mode == "no_status":
        while True:
            time.sleep(1.0)

    if fake_mode == "ignore_sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if args.until_stopped:
        missing = []
        if not args.scan_id:
            missing.append("--scan-id")
        if not args.field_id:
            missing.append("--field-id")
        if not args.crop:
            missing.append("--crop")
        if missing:
            parser.error("The following arguments are required with --until-stopped: %s" % ", ".join(missing))

    storage = EdgeStorage(db_path=Path(args.db_path)) if args.db_path else None
    scan_id = args.scan_id or ("scan_%d" % int(time.time()))
    field_id = args.field_id or "F01"
    crop = args.crop or "wheat"
    replay = (args.source != "0")
    status_file = Path(args.status_file) if args.status_file else None
    started_utc = get_utc_iso_now()
    start_mono = time.monotonic()

    if storage:
        storage.record_scan_start(
            scan_id=scan_id,
            source=args.source,
            mode="walk" if args.until_stopped else "handheld_pod",
            pid=os.getpid(),
            status="running",
            crop=crop,
            field_id=field_id,
            time_source=args.time_source,
            replay=1 if replay else 0,
        )

    if status_file:
        write_status(
            status_file=status_file,
            state="starting",
            scan_id=scan_id,
            field_id=field_id,
            crop=crop,
            replay=replay,
            started_utc=started_utc,
            elapsed_s=0,
            max_duration_s=args.max_duration_s,
            frames_seen=0,
            frames_used=0,
        )

    running = True
    stop_reason = "user"

    def _sig_handler(signum, frame):
        nonlocal running, stop_reason
        stop_reason = "user"
        running = False

    if fake_mode != "ignore_sigterm":
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)

    frames_count = 0
    while running:
        time.sleep(0.2)
        now_mono = time.monotonic()
        elapsed_s = int(now_mono - start_mono)
        frames_count += 1

        if args.max_duration_s > 0 and (now_mono - start_mono >= args.max_duration_s):
            stop_reason = "time_limit"
            break

        if status_file:
            write_status(
                status_file=status_file,
                state="scanning",
                scan_id=scan_id,
                field_id=field_id,
                crop=crop,
                replay=replay,
                started_utc=started_utc,
                elapsed_s=elapsed_s,
                max_duration_s=args.max_duration_s,
                frames_seen=frames_count,
                frames_used=frames_count,
            )

    adv_id = None
    if storage:
        storage.record_scan_end(
            scan_id=scan_id,
            frames_captured=frames_count,
            frames_evaluated=frames_count,
            tiles_classified=frames_count * 9,
            status="complete",
            stop_reason=stop_reason,
            duration_s=round(time.monotonic() - start_mono, 2),
        )
        adv = storage.create_advisory(
            scan_id=scan_id,
            replay=replay,
            field_id=field_id,
            stop_reason=stop_reason,
            crop_declared=crop,
            mode="walk" if args.until_stopped else "handheld_pod",
            time_source=args.time_source,
        )
        adv_id = adv.get("advisory_id")

    if status_file:
        write_status(
            status_file=status_file,
            state="done",
            scan_id=scan_id,
            field_id=field_id,
            crop=crop,
            replay=replay,
            started_utc=started_utc,
            elapsed_s=int(time.monotonic() - start_mono),
            max_duration_s=args.max_duration_s,
            frames_seen=frames_count,
            frames_used=frames_count,
            advisory_id=adv_id,
            stop_reason=stop_reason,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
