#!/usr/bin/env python3
"""
edge/ndvi_satellite.py -- Sentinel-2 L2A Satellite NDVI Fallback Module (M4.1-M4.6).

Provides cloud-masked field-level NDVI from Copernicus Data Space Ecosystem (CDSE)
via Sentinel Hub Statistical API over plain HTTPS (urllib, Python 3.6 compatible, no GDAL).

Endpoints (authoritative from CDSE documentation):
  - Token: https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  - Statistics: https://sh.dataspace.copernicus.eu/api/v1/statistics
"""
import datetime
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional, Tuple, Union

# Official CDSE endpoints (Status: UNVERIFIED ON LIVE NETWORK; unit tested with synthetic responses in CI)
# Documentation URLs:
#   - OAuth2 / Token: https://documentation.dataspace.copernicus.eu/APIs/Token.html
#   - Statistical API: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Statistical.html
#   - Evalscript V3: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Evalscript/V3.html
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

DEFAULT_CREDENTIALS_PATH = "/etc/sih/cdse.json"
DEFAULT_FIELD_CONFIG_PATH = "configs/field.json"

# Provisional validation threshold: minimum 50% clear / valid pixels in field polygon
PROVISIONAL_MIN_VALID_PIXEL_FRACTION = 0.50
DEFAULT_DAYS_HISTORY = 30


# Evalscript (VERSION=3) returning NDVI and dataMask masked by SCL (cloud, shadow, snow, no-data)
S2_NDVI_SCL_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B08", "SCL", "dataMask"]
    }],
    output: [{
      id: "ndvi",
      bands: 1,
      sampleType: "FLOAT32"
    }, {
      id: "dataMask",
      bands: 1
    }]
  };
}

function evaluatePixel(samples) {
  // SCL classes to exclude: 0=no_data, 1=saturated/defective, 3=cloud_shadow, 8=med_cloud, 9=high_cloud, 10=cirrus, 11=snow
  const isCloudOrMasked = [0, 1, 3, 8, 9, 10, 11].indexOf(samples.SCL) !== -1;
  const isNoData = (samples.dataMask === 0);

  const denom = samples.B08 + samples.B04;
  const ndvi = (denom > 0.0) ? ((samples.B08 - samples.B04) / denom) : 0.0;

  return {
    ndvi: [ndvi],
    dataMask: [(isCloudOrMasked || isNoData) ? 0 : 1]
  };
}
"""


def load_cdse_credentials(credentials_path: str = DEFAULT_CREDENTIALS_PATH) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Reads client_id and client_secret from external file (M4.2).
    Returns (client_id, client_secret, error_reason).
    """
    if not os.path.exists(credentials_path):
        return None, None, "CREDENTIALS_MISSING"

    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cid = data.get("client_id")
        secret = data.get("client_secret")
        if not cid or not secret:
            return None, None, "INVALID_CREDENTIALS_FORMAT"
        return str(cid), str(secret), None
    except Exception as ex:
        return None, None, "CREDENTIALS_READ_ERROR: %s" % ex


def load_field_polygon(field_config_path: str = DEFAULT_FIELD_CONFIG_PATH) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads field geometry polygon from configs/field.json (M4.3).
    Returns (geojson_geometry, error_reason).
    """
    if not os.path.exists(field_config_path):
        return None, "FIELD_POLYGON_NOT_CONFIGURED"

    try:
        with open(field_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("status") != "MEASURED" or not cfg.get("polygon_coordinates"):
            return None, "FIELD_POLYGON_NOT_CONFIGURED"

        coords = cfg.get("polygon_coordinates")
        geometry = {
            "type": "Polygon",
            "coordinates": coords,
        }
        return geometry, None
    except Exception as ex:
        return None, "FIELD_CONFIG_READ_ERROR: %s" % ex


def check_internet_connectivity(test_url: str = "https://identity.dataspace.copernicus.eu", timeout_s: float = 3.0) -> bool:
    """Quick HTTPS connectivity check before attempting fetch."""
    try:
        req = urllib.request.Request(test_url, headers={"User-Agent": "SIH-Smart-Farming-Edge"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status < 500
    except Exception:
        return False


def fetch_cdse_token(client_id: str, client_secret: str, token_url: str = CDSE_TOKEN_URL, timeout_s: float = 10.0) -> Tuple[Optional[str], Optional[str]]:
    """Performs OAuth2 client_credentials token request."""
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode("utf-8")

    req = urllib.request.Request(
        token_url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "SIH-Smart-Farming-Edge",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            token = body.get("access_token")
            if not token:
                return None, "TOKEN_RESPONSE_MISSING_ACCESS_TOKEN"
            return token, None
    except urllib.error.HTTPError as he:
        return None, "OAUTH_HTTP_ERROR_%d: %s" % (he.code, he.reason)
    except Exception as ex:
        return None, "OAUTH_NETWORK_ERROR: %s" % ex


def build_statistics_request_payload(
    geometry: Dict[str, Any],
    from_iso: str,
    to_iso: str,
    evalscript: str = S2_NDVI_SCL_EVALSCRIPT,
) -> Dict[str, Any]:
    """Constructs Sentinel Hub Statistical API POST payload."""
    return {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4325"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "mosaickingOrder": "leastCC"
                }
            }]
        },
        "aggregation": {
            "timeRange": {
                "from": from_iso,
                "to": to_iso,
            },
            "aggregationInterval": {
                "of": "P1D"
            },
            "evalscript": evalscript,
            "resx": 10.0,
            "resy": 10.0,
        },
        "calculations": {
            "default": {}
        }
    }


def parse_statistical_api_response(
    response_data: Dict[str, Any],
    min_valid_fraction: float = PROVISIONAL_MIN_VALID_PIXEL_FRACTION,
) -> Dict[str, Any]:
    """
    Parses daily aggregation entries from Sentinel Hub Statistical API.
    Picks the most recent date with valid_pixel_fraction >= min_valid_fraction.
    """
    data_entries = response_data.get("data", [])
    if not data_entries:
        return {
            "available": False,
            "reason": "EMPTY_STATISTICS_RESPONSE",
        }

    # Sort descending by date (most recent first)
    valid_scenes = []
    all_scenes = []

    for entry in data_entries:
        interval = entry.get("interval", {})
        from_ts = interval.get("from", "")
        scene_date = from_ts[:10] if len(from_ts) >= 10 else from_ts

        outputs = entry.get("outputs", {})
        ndvi_out = outputs.get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})

        # Total sample count vs valid (unmasked) count
        sample_count = int(ndvi_out.get("sampleCount", 0)) if ndvi_out.get("sampleCount") is not None else 0
        no_data_count = int(ndvi_out.get("noDataCount", 0)) if ndvi_out.get("noDataCount") is not None else 0
        valid_count = max(0, sample_count - no_data_count)

        if sample_count > 0:
            valid_frac = float(valid_count) / float(sample_count)
            cloud_frac = 1.0 - valid_frac
        else:
            valid_frac = 0.0
            cloud_frac = 1.0

        mean_val = ndvi_out.get("mean")
        std_val = ndvi_out.get("stDev")

        scene_info = {
            "scene_date": scene_date,
            "valid_pixel_count": valid_count,
            "total_pixel_count": sample_count,
            "valid_fraction": valid_frac,
            "cloud_masked_fraction": round(cloud_frac, 4),
            "ndvi_mean": round(float(mean_val), 4) if mean_val is not None else None,
            "ndvi_std": round(float(std_val), 4) if std_val is not None else None,
        }
        all_scenes.append(scene_info)

        if valid_frac >= min_valid_fraction and mean_val is not None:
            valid_scenes.append(scene_info)

    if not valid_scenes:
        return {
            "available": False,
            "reason": "NO_CLEAR_SCENE (All scenes in 30-day window had valid pixel fraction < %.0f%%)" % (min_valid_fraction * 100),
            "scenes_checked": len(all_scenes),
            "last_checked_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # Pick latest scene
    latest = valid_scenes[-1]  # entries are chronological, last is most recent
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    try:
        s_dt = datetime.datetime.strptime(latest["scene_date"], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        age_days = round(abs((now_dt - s_dt).total_seconds()) / 86400.0, 1)
    except Exception:
        age_days = None

    reliability_note = None
    if latest["valid_pixel_count"] < 9:
        reliability_note = "UNRELIABLE_SMALL_FIELD (<9 pixels / ~30x30m footprint)"

    return {
        "available": True,
        "reason": None,
        "source": "SENTINEL2_L2A_CDSE",
        "scene_date": latest["scene_date"],
        "age_days": age_days,
        "ndvi_mean": latest["ndvi_mean"],
        "ndvi_std": latest["ndvi_std"],
        "valid_pixel_count": latest["valid_pixel_count"],
        "cloud_masked_fraction": latest["cloud_masked_fraction"],
        "pixel_size_m": 10,
        "reliability_note": reliability_note,
    }
