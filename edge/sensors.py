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
import threading
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
      - Background asynchronous daemon thread reading UART continuously off the capture path.
      - Thread-safe non-blocking get_latest_fix() query with configurable staleness timeout.
      - Strict fix validity checks (GGA fix quality > 0, sat count >= 3, RMC status A).
      - UTC timestamp extracted directly from RMC sentence.
      - Simulation-mode fallback when /dev/ttyTHS1 does not exist.
      - Configurable device path.
      - Safe pyserial import check with clear error diagnostics.
    """

    def __init__(self, port=DEFAULT_GPS_UART, baud=9600, timeout=1.0, auto_start=False):
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self._simulation_mode = not os.path.exists(self.port)
        self._latest_fix = None
        self._last_fix_time = 0.0
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self._state_lat = None
        self._state_lon = None
        self._state_alt = 0.0
        self._state_satellites = 0
        self._state_fix_quality = 0
        self._state_utc_iso = None
        self._state_utc_ts = None

        if auto_start:
            self.start()

    def start(self):
        """Starts the background GPS reader thread if hardware port is present."""
        with self._lock:
            if self._running:
                return self
            if self._simulation_mode:
                return self
            if serial is None:
                return self

            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, name="GPSReaderThread")
            self._thread.daemon = True
            self._thread.start()
            return self

    def stop(self, timeout=1.0):
        """Stops the background GPS reader thread and waits for termination."""
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def is_running(self):
        """Returns True if the background reader thread is running."""
        with self._lock:
            return bool(self._running and self._thread is not None and self._thread.is_alive())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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
            "running": self.is_running(),
        }

    def parse_sentence(self, sentence):
        """Public helper to parse a raw NMEA sentence."""
        return parse_nmea_sentence(sentence)

    def _update_from_parsed(self, parsed):
        """Internal helper to update state and latest fix from a single parsed sentence."""
        if not parsed:
            return

        with self._lock:
            if parsed["type"] == "GGA":
                self._state_fix_quality = parsed.get("fix_quality", 0)
                self._state_satellites = parsed.get("satellites", 0)
                self._state_alt = parsed.get("altitude_m", 0.0)
                if parsed.get("valid") and parsed.get("latitude") is not None and parsed.get("longitude") is not None:
                    self._state_lat = parsed["latitude"]
                    self._state_lon = parsed["longitude"]
                else:
                    self._state_lat = None
                    self._state_lon = None

            elif parsed["type"] == "RMC":
                if parsed.get("valid") and parsed.get("latitude") is not None and parsed.get("longitude") is not None:
                    self._state_lat = parsed["latitude"]
                    self._state_lon = parsed["longitude"]
                    self._state_utc_iso = parsed.get("utc_iso")
                    self._state_utc_ts = parsed.get("utc_timestamp")
                else:
                    self._state_lat = None
                    self._state_lon = None

            if self._state_lat is not None and self._state_lon is not None:
                now_iso = self._state_utc_iso or _iso_now()
                now_ts = self._state_utc_ts or int(time.time())
                self._latest_fix = {
                    "latitude": round(self._state_lat, 6),
                    "longitude": round(self._state_lon, 6),
                    "altitude_m": round(self._state_alt, 2),
                    "satellites": self._state_satellites,
                    "fix_quality": self._state_fix_quality,
                    "utc_iso": now_iso,
                    "utc_timestamp": now_ts,
                    "timestamp_utc": now_iso,
                    "staleness_seconds": 0.0,
                    "valid": True,
                }
                self._last_fix_time = time.time()

    def _reader_loop(self):
        """
        Background daemon thread that continuously reads NMEA sentences from UART.
        """
        if serial is None:
            return

        while self._running:
            ser = None
            try:
                ser = serial.Serial(self.port, self.baud, timeout=0.2)
                while self._running:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    try:
                        line = raw_line.decode("ascii", errors="replace").strip()
                    except Exception:
                        continue
                    if not line or not line.startswith("$"):
                        continue
                    parsed = parse_nmea_sentence(line)
                    if parsed:
                        self._update_from_parsed(parsed)
            except Exception:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
                for _ in range(10):
                    if not self._running:
                        break
                    time.sleep(0.1)
            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass

    def get_latest_fix(self, max_staleness_s=5.0):
        """
        Non-blocking atomic retrieval of the most recent valid GPS fix.
        Returns None if no fix has been acquired or if the fix is older than max_staleness_s.
        """
        with self._lock:
            if self._latest_fix is None:
                return None
            staleness = time.time() - self._last_fix_time
            if max_staleness_s is not None and staleness > float(max_staleness_s):
                return None
            fix_copy = dict(self._latest_fix)
            fix_copy["staleness_seconds"] = round(staleness, 2)
            return fix_copy

    def read_stream(self, lines):
        """
        Processes a sequence of NMEA lines (e.g. from recorded file or generator),
        updating the current fix state.
        """
        for line in lines:
            parsed = parse_nmea_sentence(line)
            if parsed:
                self._update_from_parsed(parsed)

        return self.get_latest_fix(max_staleness_s=None)

    def read(self, max_staleness_s=5.0):
        """
        Obtains a GPS fix. If the background reader is active, returns the latest
        cached fix non-blockingly. If not active, performs a short one-shot read
        or returns the latest valid fix.

        Returns:
            Dict {"latitude": float, "longitude": float, "timestamp_utc": str, "staleness_seconds": float, ...}
            or None if hardware is absent or fix is invalid / stale.
        """
        if self._simulation_mode:
            return None

        if self.is_running():
            return self.get_latest_fix(max_staleness_s=max_staleness_s)

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

    def get_current_fix(self, max_staleness_s=5.0):
        """
        Returns full fix details including UTC timestamp for mast time synchronization.
        """
        fix = self.read(max_staleness_s=max_staleness_s)
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
