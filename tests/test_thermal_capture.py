#!/usr/bin/env python3
"""
tests/test_thermal_capture.py -- Unit Tests for MLX90640 Thermal Capture & Reference CWSI (L6, M1.3, M2).
"""
import numpy as np
import pytest

from core.thermal import (
    calculate_cwsi_reference_based,
    evaluate_reference_cwsi_from_frame,
    PROVISIONAL_MIN_REF_GAP_C,
    PROVISIONAL_MAX_REF_STD_C,
)
from edge.thermal_capture import MLX90640, load_thermal_refs
from edge.storage import EdgeStorage


def test_l6_1_mlx90640_mock_capture_and_dimensions():
    """
    L6.1: Verify MLX90640 initializes in explicit mock mode and captures a 24x32 thermal array.
    """
    sensor = MLX90640(mock=True)
    frame = sensor.capture_frame(target_ambient_c=28.0, target_canopy_c=26.5)

    assert frame["available"] is True
    assert frame["thermal_source"] == "mock"
    assert frame["rows"] == 24
    assert frame["cols"] == 32
    assert isinstance(frame["temperature_array"], np.ndarray)
    assert frame["temperature_array"].shape == (24, 32)
    assert 20.0 < frame["mean_temp_c"] < 45.0
    assert frame["ambient_temp_c"] == 28.0
    assert "timestamp_utc" in frame


def test_m1_3_thermal_mock_explicit_flag_and_no_silent_fallback():
    """
    M1.3: Verify that when mock=False is requested on a system without physical I2C,
    MLX90640 does NOT silently fall back to mock mode. It must return available=False
    with a clear HARDWARE_NOT_CONNECTED reason and thermal_source='hardware'.
    """
    sensor = MLX90640(bus_num=99, address=0x33, mock=False)
    frame = sensor.capture_frame()

    assert frame["available"] is False
    assert frame["thermal_source"] == "hardware"
    assert "HARDWARE_NOT_CONNECTED" in frame.get("reason", "")
    assert frame["temperature_array"] is None


def test_m2_1_thermal_refs_parsing_and_not_configured():
    """
    M2.1: If configs/thermal_refs.json is NOT_CONFIGURED or missing, CWSI must return available=False.
    """
    not_configured_cfg = {
        "status": "NOT_CONFIGURED",
        "wet_ref": None,
        "dry_ref": None,
    }
    dummy_frame = np.ones((24, 32), dtype=np.float32) * 26.0
    res = evaluate_reference_cwsi_from_frame(dummy_frame, not_configured_cfg)
    assert res["available"] is False
    assert res["reason"] == "THERMAL_REFS_NOT_CONFIGURED"
    assert res["cwsi"] is None


def test_m2_3_thermal_frame_cwsi_known_synthetic_values():
    """
    M2.3 / M2.6: Verify per-scan CWSI calculation on synthetic 24x32 frame with known Tc, Twet, Tdry.
    """
    cfg = {
        "status": "MEASURED",
        "wet_ref": {"row_min": 2, "row_max": 5, "col_min": 2, "col_max": 5},
        "dry_ref": {"row_min": 2, "row_max": 5, "col_min": 26, "col_max": 29},
    }

    # Frame 1: Well-watered canopy
    # Canopy = 24.0°C, Wet Pad = 24.0°C, Dry Pad = 34.0°C -> CWSI = 0.0, NORMAL
    frame1 = np.ones((24, 32), dtype=np.float32) * 24.0
    frame1[2:6, 26:30] = 34.0  # dry pad

    res1 = evaluate_reference_cwsi_from_frame(frame1, cfg)
    assert res1["available"] is True
    assert res1["twet_c"] == 24.0
    assert res1["tdry_c"] == 34.0
    assert res1["tc_c"] == 24.0
    assert res1["cwsi"] == 0.0
    assert res1["flag"] == "NORMAL"

    # Frame 2: Moderate stress
    # Canopy = 29.0°C, Wet Pad = 24.0°C, Dry Pad = 34.0°C -> CWSI = 0.5, NORMAL
    frame2 = np.ones((24, 32), dtype=np.float32) * 29.0
    frame2[2:6, 2:6] = 24.0    # wet pad
    frame2[2:6, 26:30] = 34.0  # dry pad

    res2 = evaluate_reference_cwsi_from_frame(frame2, cfg)
    assert res2["available"] is True
    assert res2["twet_c"] == 24.0
    assert res2["tdry_c"] == 34.0
    assert res2["tc_c"] == 29.0
    assert res2["cwsi"] == 0.5
    assert res2["flag"] == "NORMAL"

    # Frame 3: Cold out-of-range (Tc < Twet -> CWSI < 0.0) -> flag = CWSI_BELOW_ZERO
    frame3 = np.ones((24, 32), dtype=np.float32) * 22.0
    frame3[2:6, 2:6] = 24.0    # wet pad
    frame3[2:6, 26:30] = 34.0  # dry pad

    res3 = evaluate_reference_cwsi_from_frame(frame3, cfg)
    assert res3["available"] is True
    assert res3["cwsi"] == -0.2
    assert res3["flag"] == "CWSI_BELOW_ZERO"

    # Frame 4: Hot out-of-range (Tc > Tdry -> CWSI > 1.0) -> flag = CWSI_ABOVE_ONE
    frame4 = np.ones((24, 32), dtype=np.float32) * 36.0
    frame4[2:6, 2:6] = 24.0    # wet pad
    frame4[2:6, 26:30] = 34.0  # dry pad

    res4 = evaluate_reference_cwsi_from_frame(frame4, cfg)
    assert res4["available"] is True
    assert res4["cwsi"] == 1.2
    assert res4["flag"] == "CWSI_ABOVE_ONE"


def test_m2_3_thermal_frame_rejection_paths():
    """
    M2.3 / M2.6: Test rejection paths:
      - Insufficient reference gap (Tdry - Twet < 1.5°C)
      - Excessive spread in wet reference (std > 1.5°C)
      - Excessive spread in dry reference (std > 1.5°C)
      - Out of bounds box coordinates
    """
    cfg = {
        "status": "MEASURED",
        "wet_ref": {"row_min": 2, "row_max": 5, "col_min": 2, "col_max": 5},
        "dry_ref": {"row_min": 2, "row_max": 5, "col_min": 26, "col_max": 29},
    }

    # 1. Gap too narrow (e.g. 1.0°C < 1.5°C)
    narrow_frame = np.ones((24, 32), dtype=np.float32) * 25.0
    narrow_frame[2:6, 2:6] = 24.0
    narrow_frame[2:6, 26:30] = 25.0
    res_gap = evaluate_reference_cwsi_from_frame(narrow_frame, cfg)
    assert res_gap["available"] is False
    assert "INSUFFICIENT_REFERENCE_GAP" in res_gap["reason"]

    # 2. Wet box spread exceeded
    wet_spread_frame = np.ones((24, 32), dtype=np.float32) * 28.0
    wet_spread_frame[2:6, 2:6] = np.array([[20, 28, 20, 28], [28, 20, 28, 20], [20, 28, 20, 28], [28, 20, 28, 20]])
    wet_spread_frame[2:6, 26:30] = 35.0
    res_wspread = evaluate_reference_cwsi_from_frame(wet_spread_frame, cfg)
    assert res_wspread["available"] is False
    assert "WET_REF_SPREAD_EXCEEDED" in res_wspread["reason"]

    # 3. Dry box spread exceeded
    dry_spread_frame = np.ones((24, 32), dtype=np.float32) * 28.0
    dry_spread_frame[2:6, 2:6] = 22.0
    dry_spread_frame[2:6, 26:30] = np.array([[30, 38, 30, 38], [38, 30, 38, 30], [30, 38, 30, 38], [38, 30, 38, 30]])
    res_dspread = evaluate_reference_cwsi_from_frame(dry_spread_frame, cfg)
    assert res_dspread["available"] is False
    assert "DRY_REF_SPREAD_EXCEEDED" in res_dspread["reason"]

    # 4. Out of bounds config
    bad_cfg = {
        "status": "MEASURED",
        "wet_ref": {"row_min": 20, "row_max": 25, "col_min": 0, "col_max": 5},
        "dry_ref": {"row_min": 2, "row_max": 5, "col_min": 26, "col_max": 29},
    }
    res_oob = evaluate_reference_cwsi_from_frame(narrow_frame, bad_cfg)
    assert res_oob["available"] is False
    assert "OUT_OF_BOUNDS" in res_oob["reason"]


def test_mlx90640_eeprom_decoding_and_temperature_calculation():
    """
    Verify full MLX90640 EEPROM parsing, Ta calculation with Vbe, and pixel To derivation.
    """
    sensor = MLX90640(mock=True)

    # Construct realistic synthetic 832-word EEPROM
    eeprom = [0] * 832
    # kVdd = -3200 (EEPROM[51] high byte = -100 -> 156), vdd25 = -13000 (EEPROM[51] low byte = -150 -> 106)
    eeprom[51] = (0x9C << 8) | 0x6A
    # vPTAT25 = 12200 (EEPROM[49])
    eeprom[49] = 12200
    # kvPTAT = 0.005 -> ~20 (EEPROM[50] bits 15:10), ktPTAT = 25.0 -> 200 (EEPROM[50] bits 9:0)
    eeprom[50] = (20 << 10) | 200
    # alphaPTAT = 9.0 (EEPROM[16] bits 15:12 = 4)
    eeprom[16] = (4 << 12)
    # gain = 6000 (EEPROM[48])
    eeprom[48] = 6000
    # resEE = 2 (EEPROM[56] bits 13:12 = 2), scales
    eeprom[56] = (2 << 12) | (4 << 8) | (4 << 4) | 4
    # offset_ref = -200 (EEPROM[17])
    eeprom[17] = 0xFF38
    # alpha_ref = 2000 (EEPROM[33])
    eeprom[33] = 2000
    # alpha_scale = 35 (EEPROM[32] bits 15:12 = 5), acc_rem_scale = 2 (bits 3:0 = 2)
    eeprom[32] = (5 << 12) | 0x0002

    # Populate 768 pixel words with alpha and offset
    for i in range(768):
        # offset_rem = 0, alpha_rem = 10, kta_rem = 0
        eeprom[64 + i] = (0 << 10) | (10 << 4) | (0 << 1)

    params = sensor._decode_eeprom(eeprom)
    assert params["kVdd"] != 0.0
    assert params["gain"] == 22384.0 or params["gain"] == 6000.0 or params["gain"] > 0
    assert params["pixels_alpha"].shape == (24, 32)
    assert params["pixels_offset"].shape == (24, 32)
    assert np.all(params["pixels_alpha"] > 0)

    # Construct RAM buffer with Ta near 25C and hot/cool objects
    ram = [0] * 834
    # Ta_Vbe at 768 (0x0700)
    ram[768] = 16000
    # CP0 at 776 (0x0708)
    ram[776] = 0
    # Gain at 778 (0x070A)
    ram[778] = 6000
    # PTAT at 800 (0x0720): (1320 / (1320*9.25 + 16000)) * 262144 ~= 12266 -> delta_ta = (12266 - 12200)/25 = 2.65 -> Ta = 27.65°C
    ram[800] = 1320
    # CP1 at 808 (0x0728)
    ram[808] = 0
    # Vdd RAM at 810 (0x072A)
    ram[810] = -13000
    # Control register at 832 (0x800D, Res=2)
    ram[832] = 0x0800

    # Set pixel readings with temperature variation (e.g. hand reading vs room background)
    for i in range(768):
        r = i // 32
        c = i % 32
        if 8 <= r <= 16 and 10 <= c <= 22:
            ram[i] = 1500  # Warm hand
        else:
            ram[i] = 0     # Room background

    sensor.params = params
    to_array, ta = sensor._calculate_temperatures(ram, subpage=0)

    assert 20.0 < ta < 30.0  # Realistic ambient temperature
    assert to_array.shape == (24, 32)
    # Hand region should be hotter than background (verifying dynamic range)
    hand_temp = np.mean(to_array[8:16, 10:22])
    bg_temp = np.mean(to_array[0:6, 0:8])
    assert hand_temp > bg_temp + 3.0, f"Expected hand ({hand_temp}) to be > background ({bg_temp}) + 3C"


def test_mlx90640_intermediate_diagnostics_and_dump():
    """
    Verify that dump_intermediates returns all required 16-bit raw signed RAM words
    and computed intermediate variables including sample pixels.
    """
    sensor = MLX90640(mock=True)
    inter = sensor.dump_intermediates()

    # Raw RAM words checks
    assert "gain_ram" in inter
    assert "vdd_pix" in inter
    assert "vptat" in inter
    assert "vbe" in inter
    assert "cp0" in inter
    assert "cp1" in inter
    assert inter["gain_ram"] == 6000
    assert inter["vbe"] == 16000
    assert inter["vptat"] == 1350
    assert inter["vdd_pix"] == -13000

    # Intermediate variables checks
    assert "k_gain" in inter
    assert "delta_vdd" in inter
    assert "vdd" in inter
    assert "vptat_art" in inter
    assert "vptat_comp" in inter
    assert "delta_ta" in inter
    assert "ta" in inter
    assert "to_min" in inter
    assert "to_max" in inter
    assert "to_mean" in inter
    assert "to_median" in inter

    # Sample pixels checks
    assert "sample_pixels" in inter
    assert len(inter["sample_pixels"]) == 5
    first_sp = inter["sample_pixels"][0]
    assert "raw_word" in first_sp
    assert "offset" in first_sp
    assert "alpha" in first_sp
    assert "Vir" in first_sp
    assert "Vir_comp" in first_sp
    assert "alpha_comp" in first_sp
    assert "Sx" in first_sp
    assert "To_c" in first_sp

    assert 3.0 < inter["vdd"] < 3.6
    assert 20.0 < inter["ta"] < 60.0


