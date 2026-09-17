#!/usr/bin/env python3
"""
tests/test_thermal_capture.py -- Unit Tests for MLX90640 Thermal Capture & Reference CWSI (L6.4).
"""
import numpy as np
import pytest

from core.thermal import calculate_cwsi_reference_based
from edge.thermal_capture import MLX90640, calculate_cwsi_reference_based as edge_calc_cwsi


def test_l6_1_mlx90640_mock_capture_and_dimensions():
    """
    L6.1: Verify MLX90640 initializes in mock mode and captures a 24x32 thermal array.
    """
    sensor = MLX90640(mock=True)
    frame = sensor.capture_frame(target_ambient_c=28.0, target_canopy_c=26.5)

    assert frame["available"] is True
    assert frame["mock"] is True
    assert frame["rows"] == 24
    assert frame["cols"] == 32
    assert isinstance(frame["temperature_array"], np.ndarray)
    assert frame["temperature_array"].shape == (24, 32)
    assert 20.0 < frame["mean_temp_c"] < 45.0
    assert frame["ambient_temp_c"] == 28.0
    assert "timestamp_utc" in frame


def test_l6_3_cwsi_reference_based_calculations():
    """
    L6.3: Verify CWSI calculation using wet and dry physical references:
      CWSI = (Tc - Twet) / (Tdry - Twet)
    """
    # Case 1: Well-watered canopy at wet reference temperature (Tc = Twet = 24.0°C, Tdry = 34.0°C) -> CWSI = 0.0
    cwsi_val, status = calculate_cwsi_reference_based(tc=24.0, t_wet=24.0, t_dry=34.0)
    assert status == "OK"
    assert cwsi_val == 0.0

    # Case 2: Fully stressed canopy at dry reference temperature (Tc = Tdry = 34.0°C, Twet = 24.0°C) -> CWSI = 1.0
    cwsi_val, status = calculate_cwsi_reference_based(tc=34.0, t_wet=24.0, t_dry=34.0)
    assert status == "OK"
    assert cwsi_val == 1.0

    # Case 3: Moderate stress (Tc = 29.0°C, Twet = 24.0°C, Tdry = 34.0°C) -> CWSI = 0.5
    cwsi_val, status = calculate_cwsi_reference_based(tc=29.0, t_wet=24.0, t_dry=34.0)
    assert status == "OK"
    assert abs(cwsi_val - 0.5) < 1e-4

    # Case 4: Clamping outside bounds
    cwsi_cold, _ = calculate_cwsi_reference_based(tc=22.0, t_wet=24.0, t_dry=34.0)
    assert cwsi_cold == 0.0
    cwsi_hot, _ = calculate_cwsi_reference_based(tc=36.0, t_wet=24.0, t_dry=34.0)
    assert cwsi_hot == 1.0


def test_l6_3_cwsi_missing_references_rejected():
    """
    L6.3: When wet/dry references are missing (None) or invalid, CWSI MUST NEVER
    compute from defaults and MUST return WET_DRY_REFERENCES_NOT_CONFIGURED.
    """
    # Missing both
    val, status = calculate_cwsi_reference_based(tc=28.0, t_wet=None, t_dry=None)
    assert val is None
    assert status == "WET_DRY_REFERENCES_NOT_CONFIGURED"

    # Missing dry
    val, status = calculate_cwsi_reference_based(tc=28.0, t_wet=24.0, t_dry=None)
    assert val is None
    assert status == "WET_DRY_REFERENCES_NOT_CONFIGURED"

    # Missing wet
    val, status = calculate_cwsi_reference_based(tc=28.0, t_wet=None, t_dry=34.0)
    assert val is None
    assert status == "WET_DRY_REFERENCES_NOT_CONFIGURED"

    # Inverted or zero gap references (Tdry <= Twet)
    val, status = calculate_cwsi_reference_based(tc=28.0, t_wet=34.0, t_dry=34.0)
    assert val is None
    assert "INSUFFICIENT_REFERENCE_TEMPERATURE_GAP" in status
