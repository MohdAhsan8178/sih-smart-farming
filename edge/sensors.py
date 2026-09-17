#!/usr/bin/env python3
"""
edge/sensors.py -- Hardware-agnostic sensor interfaces for the Jetson Nano Smart Farming Pod.

Provides:
  - GPS: Serial GPS reader (u-blox / NMEA) with simulation-mode fallback when hardware
    is absent.  pipeline.py imports this class at runtime; missing hardware results in
    GPS = None records rather than fabricated coordinates.
  - MastTelemetryReader: Thin wrapper over EdgeStorage.get_latest_mast_telemetry()
    that enforces a 3600-second staleness window and returns explicit None+reason
    rather than stale or fabricated values.

Python 3.6 compatible -- no walrus, no f-strings with = specifier, no dataclasses.
"""

import datetime
import os
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now():
    """Returns the current UTC time as a timezone-aware datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


def _iso_now():
    """Returns current UTC time as ISO-8601 string with Z suffix."""
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(ts_str):
    """
    Parses an ISO-8601 UTC string of the form 'YYYY-MM-DDTHH:MM:SSZ' into a
    timezone-aware datetime.  Returns None on any parse failure.

    Uses manual strptime -- avoids datetime.fromisoformat (Python 3.7+).
    """
    if not ts_str:
        return None
    clean = ts_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.datetime.strptime(clean, fmt)
            return naive.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------

class GPS(object):
    """
    Hardware-agnostic GPS reader.

    When the configured serial port exists, this class is intended to be
    extended (in a future hardware-unlock step) with pynmea2 / pyserial
    NMEA parsing.  Until then -- or whenever the port is absent on the
    current host -- the class operates in *simulation mode* and read()
    returns None with an explicit error code rather than fabricating
    coordinates.

    Design constraints:
      - MUST NOT fabricate latitude/longitude values.
      - Must be instantiable even if the port does not exist.
      - read() signature is consumed by pipeline.py lines 264-273.
    """

    def __init__(self, port='/dev/ttyUSB0', baud=9600):
        """
        Stores configuration and detects hardware availability.

        Args:
            port: Serial device path (default /dev/ttyUSB0 for u-blox on Jetson).
            baud: Serial baud rate (default 9600, standard for NMEA GPS receivers).
        """
        self.port = port
        self.baud = int(baud)
        # Detect whether the device node exists at construction time.
        # On Mac / CI / Jetson without the dongle this will be False.
        self._simulation_mode = not os.path.exists(self.port)

    # ------------------------------------------------------------------
    # Public API -- called by pipeline.py
    # ------------------------------------------------------------------

    def read(self):
        """
        Attempts to obtain a fresh GPS fix.

        Returns:
            On success (hardware present and fix obtained):
                {
                    "latitude": float,
                    "longitude": float,
                    "timestamp_utc": str,   # ISO-8601 Z-suffix
                    "staleness_seconds": float,
                }
            On failure / simulation mode:
                None

        Rationale for returning None (not a dict with zeroed coords):
            pipeline.py lines 264-273 gates on `reading and "latitude" in reading`.
            Returning None skips GPS tagging without injecting bogus coordinates
            into frame events or advisory documents.
        """
        if self._simulation_mode:
            # Hardware absent -- do not fabricate coordinates.
            return None

        # Hardware path: pyserial + pynmea2 would go here.
        # Left as a stub until PENDING_HARDWARE.md GPS dongle arrives.
        # Return None rather than partially-parsed or defaulted data.
        return None

    # ------------------------------------------------------------------
    # Diagnostic helpers (not called by pipeline.py)
    # ------------------------------------------------------------------

    def is_simulation_mode(self):
        """Returns True if running without GPS hardware."""
        return self._simulation_mode

    def get_config(self):
        """Returns the current configuration dict (useful for health endpoints)."""
        return {
            "port": self.port,
            "baud": self.baud,
            "simulation_mode": self._simulation_mode,
        }


# ---------------------------------------------------------------------------
# MastTelemetryReader
# ---------------------------------------------------------------------------

_MAST_STALE_SECONDS = 3600  # 1 hour -- readings older than this are treated as stale

# Sensor field names that the ESP32 mast node (SIH-NODE-01) may populate.
_MAST_SENSOR_FIELDS = (
    "air_temp_c",
    "humidity_pct",
    "canopy_temp_c",
    "soil_moisture_v",
    "water_level_mm",
)


class MastTelemetryReader(object):
    """
    Thin staleness-aware wrapper over EdgeStorage.get_latest_mast_telemetry().

    The mast node (SIH-NODE-01) POSTs telemetry via the gateway every few
    minutes.  This reader is used by the pipeline or health endpoints to
    obtain the most recent valid reading without re-querying the DB directly.

    Staleness policy:
      - If the latest stored row is older than _MAST_STALE_SECONDS (3600 s),
        get_latest() returns None with reason "MAST_DATA_STALE".
      - If no rows exist, returns None with reason "NO_MAST_DATA".
      - Sensor fields that are absent in the stored row are returned as None
        (never filled with defaults or fabricated values).
    """

    def __init__(self, storage):
        """
        Args:
            storage: An EdgeStorage instance (must expose get_latest_mast_telemetry()).
        """
        self._storage = storage

    def get_latest(self):
        """
        Fetches the latest mast telemetry row and validates its staleness.

        Returns:
            A dict with keys:
                recorded_at_utc   (str)
                received_at_utc   (str)
                staleness_seconds (float)
                air_temp_c        (float or None)
                humidity_pct      (float or None)
                canopy_temp_c     (float or None)
                soil_moisture_v   (list or None)
                water_level_mm    (float or None)
                node_id           (str)
                firmware_version  (str or None)
            ... or None if no data or data is stale.
        """
        row = self._storage.get_latest_mast_telemetry()

        if row is None:
            return None  # reason: NO_MAST_DATA (caller checks None)

        # Determine staleness from received_at_utc (when Jetson got it)
        reference_ts = row.get("received_at_utc") or row.get("recorded_at_utc")
        if not reference_ts:
            return None  # malformed row

        dt_received = _parse_iso_utc(reference_ts)
        if dt_received is None:
            return None  # unparseable timestamp

        now = _utc_now()
        staleness = (now - dt_received).total_seconds()

        if staleness > _MAST_STALE_SECONDS:
            return None  # reason: MAST_DATA_STALE

        result = {
            "node_id": row.get("node_id"),
            "firmware_version": row.get("firmware_version"),
            "recorded_at_utc": row.get("recorded_at_utc"),
            "received_at_utc": row.get("received_at_utc"),
            "staleness_seconds": round(staleness, 2),
        }

        # Sensor fields: explicitly None if absent -- no defaults, no fabrication.
        for field in _MAST_SENSOR_FIELDS:
            result[field] = row.get(field)  # None if not present in stored row

        return result
