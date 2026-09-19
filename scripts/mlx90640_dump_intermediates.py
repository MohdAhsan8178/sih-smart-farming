#!/usr/bin/env python3
"""
scripts/mlx90640_dump_intermediates.py -- MLX90640 Calibration & Intermediate Diagnostics Tool.

Dumps raw 16-bit signed registers and computed intermediate calibration variables:
  - Raw RAM words: gain_ram (0x070A), vdd_pix (0x072A), vptat (0x0720), vbe (0x0700), cp0 (0x0708), cp1 (0x0728)
  - Computed intermediates: k_gain, delta_vdd, vdd, vptat_art, vptat_comp, delta_ta, Ta, To statistics

Python 3.6+ compatible.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.thermal_capture import MLX90640


def dump_mlx90640_diagnostics(bus_num=1, address=0x33, mock=False, emissivity=0.95, as_json=False):
    # type: (int, int, bool, float, bool) -> Dict[str, Any]
    """
    Initializes MLX90640 and dumps raw register words and computed intermediate values.
    """
    sensor = MLX90640(bus_num=bus_num, address=address, mock=mock)
    sensor.emissivity = emissivity

    if not mock and sensor._init_error is not None:
        print("[ERROR] Failed to initialize MLX90640 hardware: %s" % sensor._init_error, file=sys.stderr)
        sys.exit(1)

    # Capture a live frame / read RAM words
    if mock:
        ram_words = [0] * 834
        ram_words[768] = 16000  # Vbe (0x0700)
        ram_words[776] = 0      # CP0 (0x0708)
        ram_words[778] = 6000   # Gain (0x070A)
        ram_words[800] = 1350   # PTAT (0x0720)
        ram_words[808] = 0      # CP1 (0x0728)
        ram_words[810] = -13000 # Vdd (0x072A)
        ram_words[832] = 0x0800 # Control register (Res=2)
        subpage = 0
    else:
        try:
            status_words = sensor._read_words(sensor.STATUS_REG, 1)
            subpage = (status_words[0] & 0x01) if status_words else 0
            ram_words = sensor._read_words(sensor.RAM_START, sensor.RAM_WORDS)
            ctrl_words = sensor._read_words(sensor.CTRL_REG, 1)
            if ctrl_words:
                ram_words.append(ctrl_words[0])
        except Exception as ex:
            print("[ERROR] Failed to read RAM from MLX90640: %s" % ex, file=sys.stderr)
            sys.exit(1)

    to_arr, ta, intermediates = sensor._calculate_temperatures(ram_words, subpage=subpage, return_intermediates=True)

    report = {
        "status": "OK",
        "mode": "mock" if mock else "hardware",
        "bus_num": bus_num,
        "address": hex(address),
        "subpage": subpage,
        "emissivity": emissivity,
        "eeprom_params": {
            "gainEE": sensor.params.get("gain"),
            "kVdd": sensor.params.get("kVdd"),
            "vdd25": sensor.params.get("vdd25"),
            "kvPTAT": sensor.params.get("kvPTAT"),
            "ktPTAT": sensor.params.get("ktPTAT"),
            "vPTAT25": sensor.params.get("vPTAT25"),
            "alphaPTAT": sensor.params.get("alphaPTAT"),
            "tgc": sensor.params.get("tgc"),
            "ksTa": sensor.params.get("ksTa"),
            "ksTo": sensor.params.get("ksTo"),
            "offCP": sensor.params.get("offCP"),
            "ktaCP": sensor.params.get("ktaCP"),
            "kvCP": sensor.params.get("kvCP"),
        },
        "raw_ram_words": {
            "gain_ram (0x070A / index 778)": intermediates["gain_ram"],
            "vdd_pix (0x072A / index 810)": intermediates["vdd_pix"],
            "vptat (0x0720 / index 800)": intermediates["vptat"],
            "vbe (0x0700 / index 768)": intermediates["vbe"],
            "cp0 (0x0708 / index 776)": intermediates["cp0"],
            "cp1 (0x0728 / index 808)": intermediates["cp1"],
        },
        "computed_intermediates": {
            "k_gain": round(intermediates["k_gain"], 6),
            "delta_vdd": round(intermediates["delta_vdd"], 6),
            "vdd (V)": round(intermediates["vdd"], 4),
            "vptat_art": round(intermediates["vptat_art"], 4),
            "vptat_comp": round(intermediates["vptat_comp"], 4),
            "delta_ta": round(intermediates["delta_ta"], 4),
            "Ta (ambient °C)": round(intermediates["ta"], 2),
        },
        "object_temperature_stats": {
            "to_min (°C)": round(intermediates["to_min"], 2),
            "to_max (°C)": round(intermediates["to_max"], 2),
            "to_mean (°C)": round(intermediates["to_mean"], 2),
            "to_median (°C)": round(intermediates["to_median"], 2),
        },
    }

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print(" MLX90640 Thermal Array Intermediate Diagnostics")
        print("=" * 70)
        print("  Mode         : %s" % report["mode"].upper())
        print("  I2C Address  : Bus %d, %s" % (bus_num, hex(address)))
        print("  Subpage      : %d" % subpage)
        print("  Emissivity   : %.2f" % emissivity)
        print("-" * 70)
        print("1. Raw Signed 16-Bit RAM Words:")
        for k, v in report["raw_ram_words"].items():
            print("   - %-32s : %d (0x%04X)" % (k, v, v & 0xFFFF))
        print("-" * 70)
        print("2. Computed Intermediate Calibration Variables:")
        for k, v in report["computed_intermediates"].items():
            print("   - %-32s : %s" % (k, str(v)))
        print("-" * 70)
        print("3. Object Temperature (To) Statistics (24x32 Array):")
        for k, v in report["object_temperature_stats"].items():
            print("   - %-32s : %s" % (k, str(v)))
        print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="MLX90640 Calibration & Diagnostics Dump Tool")
    parser.add_argument("--bus-num", type=int, default=1, help="I2C bus number (default: 1)")
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x33, help="I2C hex address (default: 0x33)")
    parser.add_argument("--emissivity", type=float, default=0.95, help="Target surface emissivity (default: 0.95)")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic / mock sensor values")
    parser.add_argument("--json", action="store_true", help="Output diagnostic results as JSON")
    args = parser.parse_args()

    dump_mlx90640_diagnostics(
        bus_num=args.bus_num,
        address=args.address,
        mock=args.mock,
        emissivity=args.emissivity,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
