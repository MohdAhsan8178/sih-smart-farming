#!/usr/bin/env python3
"""
edge/sensors.py -- Hardware-agnostic sensor interfaces for the Jetson Nano Smart Farming Pod (K1.5, K2.1).

Provides:
  - GPS: 40-pin UART (pins 8/10 -> /dev/ttyTHS1, 9600 baud) reader with pure stdlib
    NMEA parser (GGA/RMC) and simulation-mode fallback when hardware is absent.
  - MastTelemetryReader: Thin wrapper over EdgeStorage.get_latest_mast_telemetry()
    that enforces a 3600-second staleness window (using received_at when rtc_valid=false)
    and returns explicit None rather than stale or fabricated values.

Python 3.6 compatible -- no walrus :=, no union types |, no dataclasses.
"""

import datetime
import os
import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

try:
    import serial
except ImportError:
    serial = None


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
    timezone-aware datetime. Returns None on any parse failure.
    """
    if not ts_str:
        return None
    clean = str(ts_str).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.datetime.strptime(clean, fmt)
            return naive.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def parse_nmea_coordinate(coord_str, direction):
    """
    Converts NMEA latitude/longitude (ddmm.mmmmm / dddmm.mmmmm) to decimal degrees.
    """
    if not coord_str or not direction:
        return None
    try:
        val = float(coord_str)
        deg_digits = 3 if direction.upper() in ('E', 'W') else 2
        deg_part = int(val / 100)
        min_part = val - (deg_part * 100)
        decimal = deg_part + (min_part / 60.0)
        if direction.upper() in ('S', 'W'):
            decimal = -decimal
        return decimal
    except Exception:
        return None


def parse_nmea_sentence(sentence):
    """
    Parses a single NMEA sentence (GPGGA, GPRMC, GNGGA, GNRMC) using pure stdlib.
    Returns a dict with parsed fields, or None if invalid checksum or unrecognised.
    """
    if not sentence or not isinstance(sentence, str):
        return None
    raw = sentence.strip()
    if not raw.startswith("$"):
        return None

    # Checksum validation if present
    if "*" in raw:
        body, chk = raw[1:].split("*", 1)
        try:
            expected_chk = int(chk[:2], 16)
            calc_chk = 0
            for char in body:
                calc_chk ^= ord(char)
            if calc_chk != expected_chk:
                return None
        except ValueError:
            return None
    else:
        body = raw[1:]

    parts = body.split(",")
    talker = parts[0]
    if len(talker) < 3:
        return None
    sent_type = talker[-3:]

    if sent_type == "GGA":
        # $GPGGA,hhmmss.sss,ddmm.mmmm,N/S,dddmm.mmmm,E/W,fix_quality,num_sat,hdop,alt,M,...
        if len(parts) < 10:
            return None
        time_raw = parts[1]
        lat = parse_nmea_coordinate(parts[2], parts[3])
        lon = parse_nmea_coordinate(parts[4], parts[5])
        try:
            fix_quality = int(parts[6]) if parts[6] else 0
        except ValueError:
            fix_quality = 0
        try:
            satellites = int(parts[7]) if parts[7] else 0
        except ValueError:
            satellites = 0
        try:
            alt = float(parts[9]) if parts[9] else 0.0
        except ValueError:
            alt = 0.0

        return {
            "type": "GGA",
            "time_raw": time_raw,
            "latitude": lat,
            "longitude": lon,
            "fix_quality": fix_quality,
            "satellites": satellites,
            "altitude_m": alt,
            "valid": (fix_quality > 0 and satellites >= 3 and lat is not None and lon is not None),
        }

    elif sent_type == "RMC":
        # $GPRMC,hhmmss.sss,status,ddmm.mmmm,N/S,dddmm.mmmm,E/W,speed,track,ddmmyy,...
        if len(parts) < 10:
            return None
        time_raw = parts[1]
        status = parts[2].upper() if parts[2] else "V"
        lat = parse_nmea_coordinate(parts[3], parts[4])
        lon = parse_nmea_coordinate(parts[5], parts[6])
        date_raw = parts[9]

        utc_iso = None
        utc_ts = None
        if len(date_raw) == 6 and len(time_raw) >= 6 and status == "A":
            try:
                day = int(date_raw[0:2])
                month = int(date_raw[2:4])
                year = 2000 + int(date_raw[4:6])
                hour = int(time_raw[0:2])
                minute = int(time_raw[2:4])
                second = int(time_raw[4:6])
                dt = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)
                utc_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                utc_ts = int(dt.timestamp())
            except Exception:
                pass

        return {
            "type": "RMC",
            "time_raw": time_raw,
            "date_raw": date_raw,
            "status": status,
            "latitude": lat,
            "longitude": lon,
            "utc_iso": utc_iso,
            "utc_timestamp": utc_ts,
            "valid": (status == "A" and lat is not None and lon is not None),
        }

    return None


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------

DEFAULT_GPS_UART = "/dev/ttyTHS1"  # Jetson Nano 40-pin header UART (pins 8/10)


class GPS(object):
    """
    NEO-6M GPS receiver wired to Jetson Nano 40-pin header UART (/dev/ttyTHS1 at 9600 baud).

    Features:
      - Small pure stdlib NMEA parser for GGA and RMC sentences (no external pynmea2).
      - Strict fix validity checks (GGA fix quality > 0, sat count >= 3, RMC status A).
      - UTC timestamp extracted directly from RMC sentence.
      - Simulation-mode fallback when /dev/ttyTHS1 does not exist.
      - Configurable device path.
      - Safe pyserial import check with clear error diagnostics.
    """

    def __init__(self, port=DEFAULT_GPS_UART, baud=9600, timeout=1.0):
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self._simulation_mode = not os.path.exists(self.port)
        self._last_fix = None
        self._last_fix_time = 0.0

    def is_simulation_mode(self):
        """Returns True if running without GPS hardware or port absent."""
        return self._simulation_mode

    def get_config(self):
        """Returns diagnostic configuration dictionary."""
        return {
            "port": self.port,
            "baud": self.baud,
            "simulation_mode": self._simulation_mode,
            "serial_available": serial is not None,
        }

    def parse_sentence(self, sentence):
        """Public helper to parse a raw NMEA sentence."""
        return parse_nmea_sentence(sentence)

    def read_stream(self, lines):
        """
        Processes a sequence of NMEA lines (e.g. from recorded file or generator),
        updating the current fix state.
        """
        current_lat = None
        current_lon = None
        current_alt = 0.0
        current_sats = 0
        current_fix_quality = 0
        current_utc_iso = None
        current_utc_ts = None
        is_valid = False

        for line in lines:
            parsed = parse_nmea_sentence(line)
            if not parsed:
                continue
            if parsed["type"] == "GGA":
                current_fix_quality = parsed["fix_quality"]
                current_sats = parsed["satellites"]
                current_alt = parsed["altitude_m"]
                if parsed["valid"]:
                    current_lat = parsed["latitude"]
                    current_lon = parsed["longitude"]
            elif parsed["type"] == "RMC":
                if parsed["valid"]:
                    current_lat = parsed["latitude"]
                    current_lon = parsed["longitude"]
                    current_utc_iso = parsed["utc_iso"]
                    current_utc_ts = parsed["utc_timestamp"]

        if (current_fix_quality > 0 or current_utc_iso is not None) and current_lat is not None and current_lon is not None:
            is_valid = True
            now_iso = current_utc_iso or _iso_now()
            self._last_fix = {
                "latitude": round(current_lat, 6),
                "longitude": round(current_lon, 6),
                "altitude_m": round(current_alt, 2),
                "satellites": current_sats,
                "fix_quality": current_fix_quality,
                "utc_iso": now_iso,
                "utc_timestamp": current_utc_ts or int(time.time()),
                "timestamp_utc": now_iso,
                "staleness_seconds": 0.0,
                "valid": True,
            }
            self._last_fix_time = time.time()
            return self._last_fix

        return None

    def read(self):
        """
        Obtains a fresh GPS fix from /dev/ttyTHS1.

        Returns:
            Dict {"latitude": float, "longitude": float, "timestamp_utc": str, "staleness_seconds": float}
            or None if hardware is absent or fix is invalid.
        """
        if self._simulation_mode:
            return None

        if serial is None:
            raise RuntimeError(
                "pyserial package is required for GPS hardware on %s but is not installed. "
                "Install via 'pip install pyserial==3.5'." % self.port
            )

        try:
            with serial.Serial(self.port, self.baud, timeout=self.timeout) as ser:
                lines = []
                start_t = time.time()
                while time.time() - start_t < 2.0:
                    raw_line = ser.readline()
                    if raw_line:
                        try:
                            lines.append(raw_line.decode("ascii", errors="replace"))
                        except Exception:
                            continue
                    if len(lines) >= 10:
                        break
                return self.read_stream(lines)
        except Exception:
            return None

    def get_current_fix(self):
        """
        Returns full fix details including UTC timestamp for mast time synchronization.
        """
        fix = self.read()
        if fix and fix.get("valid") and fix.get("utc_timestamp"):
            return fix
        return None


# ---------------------------------------------------------------------------
# MastTelemetryReader (Guide §6 compliant)
# ---------------------------------------------------------------------------

_MAST_STALE_SECONDS = 3600  # 1 hour staleness window

_MAST_SENSOR_FIELDS = (
    "seq",
    "node_id",
    "field_id",
    "utc",
    "rtc_valid",
    "uptime_s",
    "air_temp_c",
    "rh_pct",
    "ir_object_c",
    "ir_ambient_c",
    "lux",
    "soil1_v",
    "soil2_v",
    "battery_v",
    "status",
    "received_at",
    "log_epoch",
)


class MastTelemetryReader(object):
    """
    Staleness-aware reader over EdgeStorage.get_latest_mast_telemetry() (Guide §6).

    Rules:
      - Staleness is computed from `utc` if `rtc_valid` is true and `utc` is parseable.
      - Staleness is computed from `received_at` when `rtc_valid` is false.
      - If older than 3600 s (1 hour) or no records exist, returns None.
      - Null sensor fields remain None -- zero defaults and fabrication forbidden.
    """

    def __init__(self, storage):
        self._storage = storage

    def get_latest(self):
        """
        Fetches the latest mast telemetry reading and validates staleness.
        """
        row = self._storage.get_latest_mast_telemetry()
        if row is None:
            return None

        rtc_valid = row.get("rtc_valid", False)
        if rtc_valid and row.get("utc"):
            ref_ts = row["utc"]
        else:
            ref_ts = row.get("received_at")

        if not ref_ts:
            return None

        dt_ref = _parse_iso_utc(ref_ts)
        if dt_ref is None:
            return None

        now = _utc_now()
        staleness = (now - dt_ref).total_seconds()
        if staleness > _MAST_STALE_SECONDS:
            return None

        result = {
            "staleness_seconds": round(staleness, 2),
        }
        for field in _MAST_SENSOR_FIELDS:
            result[field] = row.get(field)

        return result
