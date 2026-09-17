#!/usr/bin/env python3
"""
tests/test_ndvi_satellite.py -- Unit Tests for Sentinel-2 Satellite NDVI Fallback (M4).
All tests run with synthetic/mocked responses; zero live network calls.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import urllib.error

from edge.ndvi_satellite import (
    load_cdse_credentials,
    load_field_polygon,
    fetch_cdse_token,
    parse_statistical_api_response,
    build_statistics_request_payload,
    CDSE_TOKEN_URL,
    CDSE_STATS_URL,
    PROVISIONAL_MIN_VALID_PIXEL_FRACTION,
)
from edge.storage import EdgeStorage


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_m4_1_cdse_endpoints_and_evalscript():
    """
    M4.1: Verify official CDSE endpoints and evalscript structure.
    """
    assert CDSE_TOKEN_URL == "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    assert CDSE_STATS_URL == "https://sh.dataspace.copernicus.eu/api/v1/statistics"

    geom = {"type": "Polygon", "coordinates": [[[77.5, 28.5], [77.6, 28.5], [77.6, 28.6], [77.5, 28.6], [77.5, 28.5]]]}
    payload = build_statistics_request_payload(geom, "2026-08-20T00:00:00Z", "2026-09-20T23:59:59Z")
    assert payload["input"]["bounds"]["geometry"] == geom
    assert payload["aggregation"]["evalscript"].startswith("//VERSION=3")


def test_m4_2_missing_credentials_fails_gracefully():
    """
    M4.2: Missing credentials file returns available=False and reason CREDENTIALS_MISSING.
    """
    cid, sec, err = load_cdse_credentials("/nonexistent/path/cdse.json")
    assert cid is None
    assert sec is None
    assert err == "CREDENTIALS_MISSING"


def test_m4_2_secrets_check_and_gitignore():
    """
    M4.2: Verify .gitignore contains cdse.json and secret patterns, and no credentials leaked in repo.
    """
    gitignore_path = REPO_ROOT / ".gitignore"
    assert gitignore_path.exists()
    content = gitignore_path.read_text(encoding="utf-8")
    assert "cdse.json" in content
    assert "*.secret" in content


def test_m4_3_not_configured_field_polygon():
    """
    M4.3: When configs/field.json has status NOT_CONFIGURED, load_field_polygon returns FIELD_POLYGON_NOT_CONFIGURED.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"status": "NOT_CONFIGURED", "polygon_coordinates": None}, tf)
        temp_path = tf.name

    try:
        geom, err = load_field_polygon(temp_path)
        assert geom is None
        assert err == "FIELD_POLYGON_NOT_CONFIGURED"
    finally:
        os.unlink(temp_path)


def test_m4_4_synthetic_all_cloudy_response():
    """
    M4.4: Synthetic Statistical API response where all scenes have valid pixel fraction < 50%
    (e.g., cloudy/shadow) must return NO_CLEAR_SCENE.
    """
    synthetic_cloudy = {
        "data": [
            {
                "interval": {"from": "2026-09-01T00:00:00Z", "to": "2026-09-02T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 100,
                                    "noDataCount": 85,  # 85% cloud masked -> valid fraction = 15% < 50%
                                    "mean": 0.45,
                                    "stDev": 0.05,
                                }
                            }
                        }
                    }
                }
            },
            {
                "interval": {"from": "2026-09-10T00:00:00Z", "to": "2026-09-11T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 100,
                                    "noDataCount": 90,  # 90% cloud masked -> valid fraction = 10% < 50%
                                    "mean": 0.40,
                                    "stDev": 0.04,
                                }
                            }
                        }
                    }
                }
            }
        ]
    }

    res = parse_statistical_api_response(synthetic_cloudy)
    assert res["available"] is False
    assert "NO_CLEAR_SCENE" in res["reason"]


def test_m4_4_synthetic_mixed_dates_response():
    """
    M4.4: Synthetic response with mixed cloud/clear dates. Must pick the most recent
    clear date (valid fraction >= 50%) and parse mean, std, counts.
    """
    synthetic_mixed = {
        "data": [
            {
                "interval": {"from": "2026-09-05T00:00:00Z", "to": "2026-09-06T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 100,
                                    "noDataCount": 10,  # 90% clear
                                    "mean": 0.6823,
                                    "stDev": 0.0412,
                                }
                            }
                        }
                    }
                }
            },
            {
                "interval": {"from": "2026-09-15T00:00:00Z", "to": "2026-09-16T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 100,
                                    "noDataCount": 20,  # 80% clear
                                    "mean": 0.7245,
                                    "stDev": 0.0388,
                                }
                            }
                        }
                    }
                }
            },
            {
                "interval": {"from": "2026-09-18T00:00:00Z", "to": "2026-09-19T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 100,
                                    "noDataCount": 95,  # 5% clear -> rejected by cloud gate
                                    "mean": 0.50,
                                    "stDev": 0.05,
                                }
                            }
                        }
                    }
                }
            }
        ]
    }

    res = parse_statistical_api_response(synthetic_mixed)
    assert res["available"] is True
    assert res["scene_date"] == "2026-09-15"
    assert res["ndvi_mean"] == 0.7245
    assert res["ndvi_std"] == 0.0388
    assert res["valid_pixel_count"] == 80
    assert res["cloud_masked_fraction"] == 0.2
    assert res["pixel_size_m"] == 10
    assert res["reliability_note"] is None


def test_m4_4_small_field_reliability_warning():
    """
    M4.6: Field with fewer than 9 valid pixels (< 30x30m footprint) includes a reliability warning.
    """
    synthetic_small = {
        "data": [
            {
                "interval": {"from": "2026-09-15T00:00:00Z", "to": "2026-09-16T00:00:00Z"},
                "outputs": {
                    "ndvi": {
                        "bands": {
                            "B0": {
                                "stats": {
                                    "sampleCount": 8,
                                    "noDataCount": 2,  # 6 valid pixels (< 9)
                                    "mean": 0.6500,
                                    "stDev": 0.0200,
                                }
                            }
                        }
                    }
                }
            }
        ]
    }

    res = parse_statistical_api_response(synthetic_small)
    assert res["available"] is True
    assert res["valid_pixel_count"] == 6
    assert "UNRELIABLE_SMALL_FIELD" in res["reliability_note"]


def test_m4_5_token_failure_handling():
    """
    M4.7: Mock OAuth token HTTP 401 error returns clean error tuple.
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
        token, err = fetch_cdse_token("dummy_id", "dummy_secret")
        assert token is None
        assert "OAUTH_HTTP_ERROR_401" in err


def test_m4_6_storage_satellite_ndvi_caching_and_advisory(tmp_path):
    """
    M4.5 / M4.6: Verify recording satellite NDVI to EdgeStorage caches it and includes
    it in create_advisory top-level payload.
    """
    db_file = tmp_path / "test_edge.db"
    storage = EdgeStorage(db_file)

    # 1. Initially empty -> get_latest_satellite_ndvi returns None
    assert storage.get_latest_satellite_ndvi() is None

    # 2. Record satellite NDVI record
    sample_sat = {
        "scene_date": "2026-09-15",
        "ndvi_mean": 0.7421,
        "ndvi_std": 0.0350,
        "valid_pixel_count": 45,
        "cloud_masked_fraction": 0.10,
        "source": "SENTINEL2_L2A_CDSE",
    }
    rec_id = storage.record_satellite_ndvi(sample_sat, field_id="F01")
    assert rec_id > 0

    latest = storage.get_latest_satellite_ndvi(field_id="F01")
    assert latest is not None
    assert latest["scene_date"] == "2026-09-15"
    assert latest["ndvi_mean"] == 0.7421

    # 3. Create advisory and verify ndvi_satellite block
    storage.record_scan_start("scan_sat_01")
    storage.record_frame_event(
        scan_id="scan_sat_01",
        frame_idx=0,
        timestamp_utc="2026-09-17T10:00:00Z",
        cell_id="c0",
        gate_passed=True,
        gate_metrics={"blur_score": 120.0, "dark_fraction": 0.05, "bright_fraction": 0.05, "displacement": 5.0},
        n_valid_tiles=9,
        frame_state="HEALTHY",
        class_id=0,
        confidence=0.95,
        tile_decisions=[],
    )
    adv = storage.create_advisory("scan_sat_01", field_id="F01")

    assert "ndvi_satellite" in adv
    sat_block = adv["ndvi_satellite"]
    assert sat_block["available"] is True
    assert sat_block["source"] == "SENTINEL2_L2A_CDSE"
    assert sat_block["scene_date"] == "2026-09-15"
    assert sat_block["ndvi_mean"] == 0.7421
    assert sat_block["valid_pixel_count"] == 45
