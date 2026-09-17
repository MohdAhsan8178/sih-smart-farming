#!/usr/bin/env python3
"""
edge/mast_collector.py -- Pull-Style Ground Mast Telemetry & Trap Ingestion (K1.3).

Connects to the ESP32 Ground Mast node (AP SIH-NODE-01 at 192.168.9.1)
and pulls sensor readings and sticky-trap images per Guide §6 wire contract.

Execution Flow:
  1. GET /api/v1/health: Check mast node_id and log_epoch. Reset cursor to 0 if log_epoch changed.
  2. POST /api/v1/time: Synchronize DS3231 RTC IF AND ONLY IF Nano has valid GPS fix and time.
  3. GET /api/v1/readings: Page sensor records from stored cursor until truncated=False.
     Persist batches to SQLite; partial failures do NOT corrupt the cursor.
  4. GET /api/v1/trap/list & /trap/image: Fetch new trap photos and trigger Model B job.

Python 3.6 compatible.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.sensors import GPS
from edge.storage import DEFAULT_DB_PATH, EdgeStorage, get_utc_iso_now
from edge.trap_job import process_trap_card

logger = logging.getLogger("edge.mast_collector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MAST_BASE_URL = "http://192.168.9.1/api/v1"
DEFAULT_TRAP_DIR = ROOT / "data/traps"


class MastCollector(object):
    """
    Client for pulling readings and trap images from ESP32 Ground Mast Node.
    """

    def __init__(
        self,
        base_url=DEFAULT_MAST_BASE_URL,
        storage=None,
        gps_reader=None,
        trap_dir=DEFAULT_TRAP_DIR,
        timeout=5.0,
        max_retries=2,
    ):
        self.base_url = base_url.rstrip("/")
        self.storage = storage or EdgeStorage()
        self.gps_reader = gps_reader or GPS()
        self.trap_dir = Path(trap_dir)
        self.trap_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    def _http_request(self, endpoint, method="GET", params=None, body_bytes=None, headers=None):
        """Executes HTTP request with retry logic."""
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = self.base_url + endpoint
        if params:
            query = urllib.parse.urlencode(params)
            url += ("?" + query) if "?" not in url else ("&" + query)

        hdrs = headers or {}
        req = urllib.request.Request(url, data=body_bytes, headers=hdrs, method=method)

        last_err = None
        for attempt in range(1, self.max_retries + 2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status_code = resp.status
                    data = resp.read()
                    return status_code, data
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as ex:
                last_err = ex
                logger.warning("HTTP %s %s attempt %d/%d failed: %s", method, url, attempt, self.max_retries + 1, ex)
                if attempt <= self.max_retries:
                    time.sleep(0.5)

        raise last_err

    def get_health(self):
        """Fetches health status from mast node."""
        code, raw = self._http_request("/health")
        if code != 200:
            raise RuntimeError("Health check returned status %d" % code)
        return json.loads(raw.decode("utf-8"))

    def sync_time_from_gps(self):
        """
        Synchronizes the DS3231 RTC on the mast node if Nano has a valid GPS fix.
        Per Guide §6: Only call /time when Nano's own GPS fix and time are valid.
        """
        fix = self.gps_reader.get_current_fix()
        if not fix or not fix.get("valid") or not fix.get("utc_timestamp"):
            logger.info("Nano GPS fix not valid/available; skipping mast clock synchronization.")
            return False, "GPS_FIX_INVALID_OR_ABSENT"

        utc_ts = int(fix["utc_timestamp"])
        logger.info("Nano GPS fix valid (UTC timestamp %d). Syncing mast DS3231 clock...", utc_ts)
        try:
            code, raw = self._http_request("/time", method="POST", params={"utc": utc_ts})
            if code == 200:
                logger.info("Mast RTC synchronized successfully to %d", utc_ts)
                return True, "SYNCED"
            return False, "HTTP_%d" % code
        except Exception as e:
            logger.error("Failed to sync mast time: %s", e)
            return False, str(e)

    def pull_readings(self, node_id, log_epoch):
        """
        Pages sensor readings from the mast using the stored cursor.
        """
        stored_epoch, stored_seq = self.storage.get_mast_cursor(node_id)
        if stored_epoch != log_epoch:
            logger.warning("Mast log_epoch changed (%d -> %d). Resetting cursor seq to 0.", stored_epoch, log_epoch)
            stored_seq = 0

        current_cursor = stored_seq
        total_records_pulled = 0

        while True:
            logger.info("Requesting mast readings: node=%s since=%d limit=500...", node_id, current_cursor)
            code, raw = self._http_request("/readings", params={"since": current_cursor, "limit": 500})
            if code != 200:
                raise RuntimeError("Readings endpoint returned status %d" % code)

            data = json.loads(raw.decode("utf-8"))
            records = data.get("records", [])
            count = data.get("count", len(records))
            next_since = data.get("next_since", current_cursor)
            truncated = data.get("truncated", False)

            if records:
                now_utc = get_utc_iso_now()
                self.storage.record_mast_readings(records, log_epoch=log_epoch, received_at=now_utc)
                total_records_pulled += len(records)
                # Only update stored cursor on successful DB persistence
                current_cursor = int(next_since)
                self.storage.update_mast_cursor(node_id, log_epoch, current_cursor)
                logger.info("Stored %d readings (cursor advanced to %d)", len(records), current_cursor)

            if not truncated or count == 0:
                break

        return total_records_pulled, current_cursor

    def pull_trap_images(self, node_id):
        """
        Fetches new sticky trap images from the mast and triggers Model B processing.
        """
        try:
            code, raw = self._http_request("/trap/list")
            if code != 200:
                logger.warning("Failed to list trap images: HTTP %d", code)
                return 0
            list_data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.warning("Could not query /trap/list: %s", e)
            return 0

        images = list_data.get("images", []) if isinstance(list_data, dict) else list_data
        if not isinstance(images, list):
            images = []

        processed_count = 0
        for item in images:
            trap_id = item.get("trap_id") or item.get("id")
            if trap_id is None:
                continue

            dest_filename = "T%s.jpg" % trap_id
            dest_path = self.trap_dir / dest_filename

            if not dest_path.exists():
                logger.info("Fetching new trap image T%s from mast...", trap_id)
                try:
                    code, img_bytes = self._http_request("/trap/image", params={"id": trap_id})
                    if code == 200 and img_bytes:
                        dest_path.write_bytes(img_bytes)
                        logger.info("Saved trap image to %s (%d bytes)", dest_path, len(img_bytes))

                        # Process with Model B trap job
                        rec_utc = item.get("received_utc")
                        job_res = process_trap_card(
                            image_path=str(dest_path),
                            placed_at=rec_utc,
                            days_monitored=1.0,
                            storage=self.storage,
                            allow_provisional=True,
                            trap_id="T%s" % trap_id,
                        )
                        logger.info("Processed trap job: %s (blobs=%d, ETL=%s)", job_res.get("job_id"), job_res.get("total_blobs_counted", 0), job_res.get("etl_status"))
                        processed_count += 1
                except Exception as ex:
                    logger.error("Failed to download/process trap image T%s: %s", trap_id, ex)

        return processed_count

    def collect(self):
        """
        Executes a full pull cycle from the mast node.
        """
        t0 = time.time()
        logger.info("Connecting to ground mast node at %s...", self.base_url)

        # 1. Health check & Epoch validation
        health = self.get_health()
        node_id = str(health.get("node_id", "N01"))
        log_epoch = int(health.get("log_epoch", 1))
        logger.info("Mast online: node=%s, epoch=%d, rtc_valid=%s, battery=%.2fV", node_id, log_epoch, health.get("rtc_valid"), float(health.get("battery_v") or 0.0))

        # 2. Time synchronization (if GPS fix is valid)
        time_synced, time_reason = self.sync_time_from_gps()

        # 3. Pull sensor readings
        records_pulled, final_seq = self.pull_readings(node_id, log_epoch)

        # 4. Pull trap images
        traps_processed = self.pull_trap_images(node_id)

        elapsed = time.time() - t0
        summary = {
            "status": "ok",
            "node_id": node_id,
            "log_epoch": log_epoch,
            "time_synced": time_synced,
            "time_reason": time_reason,
            "records_pulled": records_pulled,
            "cursor_seq": final_seq,
            "traps_processed": traps_processed,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info("Mast collection cycle complete in %.2fs: %s", elapsed, summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Pull telemetry and trap cards from Ground Mast node (K1.3)")
    parser.add_argument("--base-url", default=DEFAULT_MAST_BASE_URL, help="Mast base URL")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--trap-dir", default=str(DEFAULT_TRAP_DIR), help="Sticky trap image directory")
    args = parser.parse_args()

    storage = EdgeStorage(db_path=args.db_path)
    collector = MastCollector(
        base_url=args.base_url,
        storage=storage,
        trap_dir=args.trap_dir,
    )
    res = collector.collect()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
