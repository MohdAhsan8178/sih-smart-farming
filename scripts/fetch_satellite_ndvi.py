#!/usr/bin/env python3
"""
scripts/fetch_satellite_ndvi.py -- CLI tool for Sentinel-2 NDVI Retrieval & Caching (M4.5).

Checks internet connectivity, requests Sentinel-2 L2A Statistical API data from CDSE,
and caches the results in edge.db.

Python 3.6 compatible.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.ndvi_satellite import (
    load_cdse_credentials,
    load_field_polygon,
    check_internet_connectivity,
    fetch_cdse_token,
    build_statistics_request_payload,
    parse_statistical_api_response,
    CDSE_STATS_URL,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_FIELD_CONFIG_PATH,
    DEFAULT_DAYS_HISTORY,
)
from edge.storage import EdgeStorage


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 L2A Satellite NDVI Fetcher (M4.5)")
    parser.add_argument("--db-path", default="edge.db", help="Path to SQLite edge.db")
    parser.add_argument("--credentials", default=DEFAULT_CREDENTIALS_PATH, help="Path to CDSE credentials JSON")
    parser.add_argument("--field-config", default=DEFAULT_FIELD_CONFIG_PATH, help="Path to field geometry JSON")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_HISTORY, help="Days of history to query")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and print without saving to DB")
    args = parser.parse_args()

    print("[SAT_NDVI] Checking internet connectivity...")
    if not check_internet_connectivity():
        print("[WARNING] Internet unreachable. Skipping satellite NDVI fetch.")
        sys.exit(0)

    print("[SAT_NDVI] Loading credentials from %s..." % args.credentials)
    client_id, client_secret, cred_err = load_cdse_credentials(args.credentials)
    if cred_err:
        print("[ERROR] Failed loading credentials: %s" % cred_err)
        sys.exit(1)

    print("[SAT_NDVI] Loading field geometry from %s..." % args.field_config)
    geometry, geom_err = load_field_polygon(args.field_config)
    if geom_err:
        print("[ERROR] Failed loading field polygon: %s" % geom_err)
        sys.exit(1)

    print("[SAT_NDVI] Authenticating with Copernicus Data Space Ecosystem...")
    token, tok_err = fetch_cdse_token(client_id, client_secret)
    if tok_err or not token:
        print("[ERROR] OAuth authentication failed: %s" % tok_err)
        sys.exit(1)

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    from_dt = now_dt - datetime.timedelta(days=args.days)
    from_iso = from_dt.strftime("%Y-%m-%dT00:00:00Z")
    to_iso = now_dt.strftime("%Y-%m-%dT23:59:59Z")

    print("[SAT_NDVI] Querying Sentinel Hub Statistical API (%s to %s)..." % (from_iso, to_iso))
    payload = build_statistics_request_payload(geometry, from_iso, to_iso)

    req = urllib.request.Request(
        CDSE_STATS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % token,
            "User-Agent": "SIH-Smart-Farming-Edge",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            stats_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        print("[ERROR] Statistical API HTTP %d: %s" % (he.code, he.reason))
        sys.exit(1)
    except Exception as ex:
        print("[ERROR] Statistical API request error: %s" % ex)
        sys.exit(1)

    result = parse_statistical_api_response(stats_json)
    print("\n[RESULT]")
    print(json.dumps(result, indent=2))

    if not args.dry_run and result.get("available"):
        storage = EdgeStorage(args.db_path)
        storage.record_satellite_ndvi(result, raw_response=stats_json)
        print("\n[SUCCESS] Cached satellite NDVI record into %s" % args.db_path)


if __name__ == "__main__":
    main()
