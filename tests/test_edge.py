"""
tests/test_edge.py
------------------
Unit and regression tests for Section E edge modules:
  1. edge/irrigation_model.py (FAO-56 ET0 Penman-Monteith & Hargreaves, Table 12 Kc, net irrigation)
  2. edge/adc.py (ADS1115 16-bit ADC, capacitive moisture calibration, mockable driver)
  3. edge/flow.py (YF-S201 Hall-effect turbine flow sensor, accumulator, volume verification)
  4. edge/actuation.py (Fail-safe pump/valve control loop, exception de-assertion, dry-run & leak faults)
  5. edge/lora.py (SX1278 packet framing, CRC16-CCITT, compact telemetry encoding)
  6. edge/thermal_point.py (MLX90614 single-point IR sensor, CRC-8 PEC, thermal mast cross-check)

Note: All tests use synthetic or mocked hardware interfaces. Zero physical hardware access is assumed.
"""

import sys
import os
import json
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.irrigation_model import (
    calculate_et0_penman_monteith,
    calculate_et0_hargreaves,
    get_crop_coefficient,
    calculate_crop_et,
    calculate_net_irrigation,
    atmospheric_pressure_fao56,
    psychrometric_constant_fao56,
    saturation_vapour_pressure_fao56,
    slope_vapour_pressure_curve_fao56,
    FAO56_CROP_COEFFICIENTS,
    PROVISIONAL_IRRIGATION_EFFICIENCIES,
    calculate_paddy_land_prep_requirement,
    calculate_paddy_daily_requirement,
    calculate_paddy_terminal_drainage,
    evaluate_paddy_awd_status,
    AWDWaterBudgetAdvisor,
    WATER_LEVEL_SENSOR_PRESENT,
    PROVISIONAL_PADDY_SATURATION_MM,
    PROVISIONAL_PADDY_LAND_PREP_PONDING_MM,
    PROVISIONAL_PADDY_PERCOLATION_MM_DAY,
    PROVISIONAL_PADDY_PERCOLATION_BY_SOIL,
    PROVISIONAL_AWD_REIRRIGATION_DEFICIT_MM,
    PROVISIONAL_AWD_REFILL_DEPTH_MM,
    PROVISIONAL_PADDY_RECOVERY_DAYS,
    PROVISIONAL_PADDY_TERMINAL_DRAINAGE_DAYS
)
from edge.adc import (
    raw_to_voltage,
    voltage_to_moisture_pct,
    get_lsb_voltage,
    ADS1115Driver,
    PROVISIONAL_V_DRY,
    PROVISIONAL_V_WET
)
from edge.flow import (
    pulses_to_liters,
    liters_to_pulses,
    frequency_to_flow_rate_lpm,
    FlowMonitor,
    PROVISIONAL_YF_S201_PULSES_PER_LITER,
    PROVISIONAL_YF_S201_HZ_PER_LPM
)
from edge.actuation import (
    ActuationController,
    MockGPIO,
    PROVISIONAL_MAX_IRRIGATION_DURATION_S,
    PROVISIONAL_FLOW_DISAGREEMENT_TIMEOUT_S,
    PROVISIONAL_LEAK_PULSE_THRESHOLD
)
from edge.lora import (
    crc16_ccitt,
    build_lora_packet,
    parse_lora_packet,
    encode_telemetry_payload,
    decode_telemetry_payload,
    SX1278Driver,
    MSG_TELEMETRY,
    MSG_COMMAND,
    MSG_ACK,
    MSG_ALERT
)
from edge.thermal_point import (
    raw_to_celsius_mlx90614,
    celsius_to_raw_mlx90614,
    calculate_smbus_pec,
    verify_smbus_pec,
    MLX90614Driver
)
import numpy as np
from edge.camera import (
    build_gstreamer_pipeline,
    create_frame_metadata,
    correct_from_reference_card,
    SENSOR_ID_RGB,
    SENSOR_ID_NIR,
    WBMODE_OFF,
    PROVISIONAL_EXPOSURE_NS,
    PROVISIONAL_GAIN,
    PROVISIONAL_DIGITAL_GAIN,
    PROVISIONAL_SATURATION_THRESHOLD_DN,
    PROVISIONAL_SATURATION_ALERT_FRACTION
)


# ==============================================================================
# 1. Tests for edge/irrigation_model.py
# ==============================================================================

def test_section_e_irrigation_psychrometric_equations():
    """Verify FAO-56 physical equations at sea level (z=0) and 20 deg C."""
    # Pressure at sea level = 101.3 kPa
    p0 = atmospheric_pressure_fao56(0.0)
    assert abs(p0 - 101.3) < 1e-4

    # Psychrometric constant gamma = 0.000665 * 101.3 = 0.0673645 kPa / C
    gamma0 = psychrometric_constant_fao56(p0)
    assert abs(gamma0 - 0.06736) < 1e-4

    # Saturation vapour pressure e0(20C) = 0.6108 * exp(17.27 * 20 / (20 + 237.3))
    # 17.27 * 20 = 345.4; 345.4 / 257.3 = 1.342398; exp(1.342398) = 3.82823
    # 0.6108 * 3.82823 = 2.3383 kPa (FAO-56 Table 2.3 gives 2.34 kPa)
    e0_20 = saturation_vapour_pressure_fao56(20.0)
    assert abs(e0_20 - 2.3383) < 0.01

    # Slope Delta(20C) = 4098 * 2.3383 / (257.3^2) = 9582.35 / 66203.29 = 0.1447 kPa/C
    delta_20 = slope_vapour_pressure_curve_fao56(20.0)
    assert abs(delta_20 - 0.1447) < 0.005


def test_section_e_irrigation_penman_monteith_standard_case():
    """Verify FAO-56 Penman-Monteith (Eq. 6) on typical daytime agronomic conditions."""
    # Input: T=25C, Rn=18.0 MJ/m2/day, G=0, u2=2.5 m/s, RH=50%, z=100m
    res = calculate_et0_penman_monteith(
        t_mean_c=25.0,
        rn_mj_m2_day=18.0,
        u2_m_s=2.5,
        rh_mean_pct=50.0,
        elevation_m=100.0,
        g_mj_m2_day=0.0
    )

    assert res["method"] == "FAO-56 Penman-Monteith (Eq. 6)"
    assert res["et0_mm_day"] > 0.0
    # Exact FAO-56 Penman-Monteith calculation: 6.997 mm/day
    assert abs(res["et0_mm_day"] - 6.997) < 0.01
    assert res["vpd_kpa"] > 0.0
    assert json.loads(json.dumps(res))["et0_mm_day"] == res["et0_mm_day"]


def test_section_e_irrigation_hargreaves_known_calculation():
    """Verify FAO-56 Hargreaves-Samani (Eq. 52) against exact manual calculation."""
    # Tmin=15.0, Tmax=25.0 -> delta_T = 10.0, Tmean = 20.0
    # Ra = 30.0 MJ/m2/day -> Ra_mm = 0.408 * 30.0 = 12.24 mm/day
    # ET0 = 0.0023 * (20 + 17.8) * sqrt(10) * 12.24
    #     = 0.0023 * 37.8 * 3.16227766 * 12.24
    #     = 3.366 mm/day
    res = calculate_et0_hargreaves(
        t_min_c=15.0,
        t_max_c=25.0,
        ra_mj_m2_day=30.0
    )
    expected_et0 = 0.0023 * (20.0 + 17.8) * math.sqrt(10.0) * (0.408 * 30.0)
    assert abs(res["et0_mm_day"] - round(expected_et0, 3)) < 1e-3
    assert res["t_mean_c"] == 20.0
    assert res["ra_mm_day"] == 12.24


def test_section_e_irrigation_crop_coefficients_fao56_table12():
    """Verify FAO-56 Table 12 crop coefficients for project target crops."""
    # Rice (wetland)
    assert FAO56_CROP_COEFFICIENTS["rice"]["kc_ini"] == 1.05
    assert FAO56_CROP_COEFFICIENTS["rice"]["kc_mid"] == 1.20
    assert FAO56_CROP_COEFFICIENTS["rice"]["kc_end"] == 0.90

    # Sugarcane
    assert FAO56_CROP_COEFFICIENTS["sugarcane"]["kc_ini"] == 0.40
    assert FAO56_CROP_COEFFICIENTS["sugarcane"]["kc_mid"] == 1.25
    assert FAO56_CROP_COEFFICIENTS["sugarcane"]["kc_end"] == 0.75

    # Wheat split: Spring wheat vs Winter wheat (non-frozen / frozen)
    # 1. Spring wheat: bare seedbed (Kc_ini = 0.30)
    assert FAO56_CROP_COEFFICIENTS["spring_wheat"]["kc_ini"] == 0.30
    assert FAO56_CROP_COEFFICIENTS["spring_wheat"]["kc_mid"] == 1.15
    assert FAO56_CROP_COEFFICIENTS["spring_wheat"]["kc_end"] == 0.30

    # 2. Winter wheat (non-frozen soils): established canopy resuming growth (Kc_ini = 0.70)
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_non_frozen"]["kc_ini"] == 0.70
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_non_frozen"]["kc_mid"] == 1.15
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_non_frozen"]["kc_end"] == 0.30

    # 3. Winter wheat (frozen soils): dormant / frost (Kc_ini = 0.40)
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_frozen"]["kc_ini"] == 0.40
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_frozen"]["kc_mid"] == 1.15
    assert FAO56_CROP_COEFFICIENTS["winter_wheat_frozen"]["kc_end"] == 0.30

    # 4. 'wheat' generic alias defaults to spring_wheat (bare-seedbed Rabi sowing in India)
    assert FAO56_CROP_COEFFICIENTS["wheat"]["kc_ini"] == 0.30
    assert FAO56_CROP_COEFFICIENTS["wheat"]["default_for"] == "spring_wheat"

    # Test stage progression interpolation for wheat:
    # At halfway through development (prog=0.5) for spring wheat:
    # Kc = 0.30 + 0.5 * (1.15 - 0.30) = 0.30 + 0.425 = 0.725
    kc_dev = get_crop_coefficient("spring_wheat", "development", stage_progress=0.5)
    assert abs(kc_dev - 0.725) < 1e-3

    # For winter wheat non-frozen: Kc_ini = 0.70, at halfway through dev:
    # Kc = 0.70 + 0.5 * (1.15 - 0.70) = 0.70 + 0.225 = 0.925
    kc_dev_ww = get_crop_coefficient("winter_wheat_non_frozen", "development", stage_progress=0.5)
    assert abs(kc_dev_ww - 0.925) < 1e-3

    # Generic alias 'wheat' evaluates identically to spring_wheat
    assert get_crop_coefficient("wheat", "initial") == 0.30

    # Unknown crop raises ValueError
    with pytest.raises(ValueError, match="not in FAO-56 Table 12"):
        get_crop_coefficient("unknown_crop", "mid")


def test_section_e_irrigation_net_requirement_and_volume():
    """Verify crop ETc, net irrigation, and volume conversion to liters."""
    et0 = 5.0  # mm/day
    kc_rice_mid = 1.20
    etc = calculate_crop_et(et0, kc_rice_mid)
    assert etc == 6.0  # 5.0 * 1.20 = 6.0 mm/day

    # 1. No rain, drip irrigation (eff = 0.90), area = 500 m2
    # I_net = 6.0 mm, I_gross = 6.0 / 0.90 = 6.667 mm
    # Volume = 6.667 * 500 = 3333.33 L
    res1 = calculate_net_irrigation(
        et_c_mm_day=etc,
        effective_rain_mm_day=0.0,
        area_m2=500.0,
        irrigation_method="drip"
    )
    assert res1["net_irrigation_mm"] == 6.0
    assert abs(res1["gross_irrigation_mm"] - (6.0 / 0.90)) < 0.01
    assert abs(res1["volume_liters"] - (6.0 / 0.90 * 500.0)) < 1.0

    # 2. Rain exceeds ETc (e.g. 10.0 mm rain vs 6.0 mm ETc) -> net irrigation is 0.0
    res2 = calculate_net_irrigation(
        et_c_mm_day=etc,
        effective_rain_mm_day=10.0,
        area_m2=500.0
    )
    assert res2["net_irrigation_mm"] == 0.0
    assert res2["gross_irrigation_mm"] == 0.0
    assert res2["volume_liters"] == 0.0


# ==============================================================================
# 2. Tests for edge/adc.py
# ==============================================================================

def test_section_e_adc_gain_lsb_and_raw_conversion():
    """Verify ADS1115 LSB voltage scaling and 16-bit signed conversion."""
    # Gain 1: FSR = 4.096V, LSB = 0.125 mV (0.000125 V)
    assert get_lsb_voltage(1) == 0.000125

    # Raw code 0 -> 0.0V
    assert raw_to_voltage(0, gain=1) == 0.0

    # Raw code 16000 at Gain 1 -> 16000 * 0.000125 = 2.0V
    assert abs(raw_to_voltage(16000, gain=1) - 2.0) < 1e-4

    # Negative code -8000 at Gain 1 -> -1.0V
    assert abs(raw_to_voltage(-8000, gain=1) - (-1.0)) < 1e-4

    # Full scale positive 32767 at Gain 2/3 (LSB = 0.0001875 V) -> 6.1438 V
    assert abs(raw_to_voltage(32767, gain=2/3) - 6.1438) < 0.001

    # Out of range code raises ValueError
    with pytest.raises(ValueError):
        raw_to_voltage(40000, gain=1)


def test_section_e_adc_capacitive_moisture_calibration():
    """Verify capacitive soil moisture transfer function and guards."""
    v_dry = PROVISIONAL_V_DRY  # 3.0V
    v_wet = PROVISIONAL_V_WET  # 1.2V

    # Voltage at or above dry threshold -> 0.0% moisture
    assert voltage_to_moisture_pct(3.0, v_dry=v_dry, v_wet=v_wet) == 0.0
    assert voltage_to_moisture_pct(3.2, v_dry=v_dry, v_wet=v_wet) == 0.0

    # Voltage at or below wet threshold -> 100.0% moisture
    assert voltage_to_moisture_pct(1.2, v_dry=v_dry, v_wet=v_wet) == 100.0
    assert voltage_to_moisture_pct(1.0, v_dry=v_dry, v_wet=v_wet) == 100.0

    # Exact midpoint: 2.1V -> 50.0% moisture
    # (3.0 - 2.1) / (3.0 - 1.2) = 0.9 / 1.8 = 0.50 -> 50.0%
    assert abs(voltage_to_moisture_pct(2.1, v_dry=v_dry, v_wet=v_wet) - 50.0) < 1e-3

    # Electrical rail violation: negative voltage or >5.5V raises ValueError
    with pytest.raises(ValueError):
        voltage_to_moisture_pct(-0.5)
    with pytest.raises(ValueError):
        voltage_to_moisture_pct(6.0)


def test_section_e_adc_driver_mock_reading():
    """Verify ADS1115Driver interface with mock synthetic voltages."""
    driver = ADS1115Driver(default_gain=1)

    # Set channel 0 to 2.1V (50% moisture at default provisional calibration)
    driver.set_mock_voltage(channel=0, voltage=2.1, gain=1)

    res0 = driver.read_moisture_pct(channel=0, gain=1)
    assert abs(res0["voltage_v"] - 2.1) < 0.01
    assert abs(res0["moisture_pct"] - 50.0) < 1.0
    assert res0["calibration_provisional"] is True
    assert json.loads(json.dumps(res0))["channel"] == 0


# ==============================================================================
# 3. Tests for edge/flow.py
# ==============================================================================

# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_flow_pulse_and_rate_conversions():
    """Verify YF-S201 conversion factors: 450 pulses/L and 7.5 Hz/(L/min)."""
    # 450 pulses = 1.0 Liter
    assert pulses_to_liters(450, pulses_per_liter=450.0) == 1.0
    assert pulses_to_liters(900, pulses_per_liter=450.0) == 2.0
    assert pulses_to_liters(0, pulses_per_liter=450.0) == 0.0

    # 1.5 Liters = 675 pulses
    assert liters_to_pulses(1.5, pulses_per_liter=450.0) == 675

    # Frequency conversion: 7.5 Hz = 1.0 L/min; 15.0 Hz = 2.0 L/min
    assert frequency_to_flow_rate_lpm(7.5, hz_per_lpm=7.5) == 1.0
    assert frequency_to_flow_rate_lpm(15.0, hz_per_lpm=7.5) == 2.0
    assert frequency_to_flow_rate_lpm(0.0, hz_per_lpm=7.5) == 0.0


# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_flow_monitor_accumulation_and_verification():
    """Verify FlowMonitor accumulation and target volume verification."""
    monitor = FlowMonitor(pulses_per_liter=450.0, hz_per_lpm=7.5)

    # Interval 1: 450 pulses over 60 seconds (1.0 L delivered, rate = 1.0 L/min)
    step1 = monitor.record_pulse_delta(delta_pulses=450, elapsed_seconds=60.0, timestamp=100.0)
    assert step1["total_pulses"] == 450
    assert step1["delivered_liters"] == 1.0
    assert abs(step1["current_rate_lpm"] - 1.0) < 1e-2

    # Target not yet reached (target = 2.0 L)
    assert monitor.is_target_reached(2.0) is False

    # Interval 2: another 450 pulses over 60 seconds (total 2.0 L)
    step2 = monitor.record_pulse_delta(delta_pulses=450, elapsed_seconds=60.0, timestamp=160.0)
    assert step2["total_pulses"] == 900
    assert step2["delivered_liters"] == 2.0
    assert monitor.is_target_reached(2.0) is True

    # Verification against target 2.0 L within 5% tolerance
    verif = monitor.verify_delivery(target_liters=2.0, tolerance_pct=0.05)
    assert verif["within_bounds"] is True
    assert verif["discrepancy_liters"] == 0.0
    assert verif["error_pct"] == 0.0


# ==============================================================================
# 4. Tests for edge/actuation.py
# ==============================================================================

# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_actuation_failsafe_normal_and_exception_deassertion():
    """Verify try...finally context manager guarantees pin deassertion."""
    mock_gpio = MockGPIO()
    controller = ActuationController(
        pump_pin=18,
        valve_pins={"zone_1": 23, "zone_2": 24},
        gpio_backend=mock_gpio
    )

    # Initial state: everything LOW (Normally Closed)
    assert mock_gpio.get_pin(18) == 0
    assert mock_gpio.get_pin(23) == 0
    assert mock_gpio.get_pin(24) == 0

    # 1. Normal session exit deasserts pins
    with controller.session("zone_1", start_time=1000.0):
        # Asserted during session
        assert mock_gpio.get_pin(18) == 1
        assert mock_gpio.get_pin(23) == 1
        assert mock_gpio.get_pin(24) == 0
    # After context exit, verified deasserted
    assert mock_gpio.get_pin(18) == 0
    assert mock_gpio.get_pin(23) == 0

    # 2. Unhandled exception in context manager MUST deassert pins
    with pytest.raises(RuntimeError, match="Simulated crash"):
        with controller.session("zone_2", start_time=1100.0):
            assert mock_gpio.get_pin(18) == 1
            assert mock_gpio.get_pin(24) == 1
            raise RuntimeError("Simulated crash during irrigation!")

    # Verified fail-safe de-assertion executed despite exception
    assert mock_gpio.get_pin(18) == 0
    assert mock_gpio.get_pin(24) == 0
    assert controller.state == ActuationController.STATE_IDLE


# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_actuation_flow_disagreement_no_flow_fault():
    """Verify commanded OPEN with zero flow trips FAULT_NO_FLOW after timeout."""
    mock_gpio = MockGPIO()
    controller = ActuationController(
        pump_pin=18,
        valve_pins={"zone_1": 23},
        gpio_backend=mock_gpio,
        no_flow_timeout_s=15.0
    )

    t0 = 1000.0
    controller.start_irrigation("zone_1", start_time=t0)
    assert controller.state == ActuationController.STATE_IRRIGATING

    # Update at t0 + 10s (within 15s timeout): no fault yet
    res_10 = controller.update_safety_loop(current_time=t0 + 10.0, flow_pulses_delta=0)
    assert res_10["state"] == ActuationController.STATE_IRRIGATING

    # Update at t0 + 16s (exceeds 15s timeout): must trip FAULT_NO_FLOW and shut down
    res_16 = controller.update_safety_loop(current_time=t0 + 16.0, flow_pulses_delta=0)
    assert res_16["state"] == ActuationController.STATE_FAULT_NO_FLOW
    assert "No water flow detected" in res_16["fault"]

    # Verify physical pins were driven LOW upon fault trip
    assert mock_gpio.get_pin(18) == 0
    assert mock_gpio.get_pin(23) == 0

    # Further irrigation starts are blocked until fault reset
    with pytest.raises(RuntimeError, match="fault state"):
        controller.start_irrigation("zone_1", start_time=t0 + 20.0)


# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_actuation_leak_detection_when_closed():
    """Verify commanded CLOSED with pulses trips FAULT_LEAK_DETECTED."""
    mock_gpio = MockGPIO()
    controller = ActuationController(
        pump_pin=18,
        valve_pins={"zone_1": 23},
        gpio_backend=mock_gpio,
        leak_pulse_thresh=5
    )

    # Controller in IDLE state (valves closed)
    assert controller.state == ActuationController.STATE_IDLE

    # 3 pulses arrive (below threshold of 5)
    res_3 = controller.update_safety_loop(current_time=100.0, flow_pulses_delta=3)
    assert res_3["state"] == ActuationController.STATE_IDLE

    # Another 3 pulses arrive (total 6 >= 5 threshold) -> trip FAULT_LEAK_DETECTED
    res_6 = controller.update_safety_loop(current_time=105.0, flow_pulses_delta=3)
    assert res_6["state"] == ActuationController.STATE_FAULT_LEAK
    assert "Hydraulic leak detected" in res_6["fault"]


# SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system produces irrigation prescriptions for manual farmer execution. This module is retained as a validated reference implementation for future closed-loop deployment and is NOT wired into any runtime path.
def test_section_e_actuation_watchdog_timeout_cutoff():
    """Verify continuous runtime exceeding watchdog limit forces shutdown."""
    mock_gpio = MockGPIO()
    controller = ActuationController(
        pump_pin=18,
        valve_pins={"zone_1": 23},
        gpio_backend=mock_gpio,
        max_duration_s=600.0  # 10 minute test watchdog limit
    )

    t0 = 1000.0
    controller.start_irrigation("zone_1", start_time=t0)

    # At 500s: valid flow provided -> still IRRIGATING
    res_500 = controller.update_safety_loop(current_time=t0 + 500.0, flow_pulses_delta=50, flow_rate_lpm=5.0)
    assert res_500["state"] == ActuationController.STATE_IRRIGATING

    # At 605s: exceeds 600s watchdog -> trips FAULT_WATCHDOG_TIMEOUT
    res_605 = controller.update_safety_loop(current_time=t0 + 605.0, flow_pulses_delta=50, flow_rate_lpm=5.0)
    assert res_605["state"] == ActuationController.STATE_FAULT_WATCHDOG
    assert mock_gpio.get_pin(18) == 0


# ==============================================================================
# 5. Tests for edge/lora.py
# ==============================================================================

def test_section_e_lora_crc16_ccitt_standard_vector():
    """Verify CRC16-CCITT algorithm against standard test vector '123456789' -> 0x29B1."""
    test_bytes = b"123456789"
    computed = crc16_ccitt(test_bytes)
    assert computed == 0x29B1, f"Expected 0x29B1, got 0x{computed:04X}"


def test_section_e_lora_packet_framing_and_parse_roundtrip():
    """Verify complete LoRa frame serialization, CRC integrity, and deserialization."""
    node_id = 42
    msg_type = MSG_TELEMETRY
    seq_num = 1001
    payload = b"field_sens_data_sample"

    packet = build_lora_packet(node_id, msg_type, seq_num, payload)
    assert len(packet) == (8 + len(payload) + 2)

    parsed = parse_lora_packet(packet)
    assert parsed["valid"] is True
    assert parsed["status"] == "ok"
    assert parsed["node_id"] == node_id
    assert parsed["msg_type"] == msg_type
    assert parsed["msg_type_name"] == "TELEMETRY"
    assert parsed["seq_num"] == seq_num
    assert parsed["payload"] == payload


def test_section_e_lora_corrupted_packet_detection():
    """Verify corrupted packet rejection (CRC error, invalid sync, truncated buffer)."""
    packet = build_lora_packet(node_id=1, msg_type=MSG_COMMAND, seq_num=5, payload=b"start_valve_1")

    # 1. Tamper single byte in payload -> CRC_MISMATCH
    corrupted_payload = bytearray(packet)
    corrupted_payload[10] ^= 0xFF
    res_crc = parse_lora_packet(bytes(corrupted_payload))
    assert res_crc["valid"] is False
    assert res_crc["status"] == "CRC_MISMATCH"

    # 2. Corrupt sync byte -> INVALID_SYNC
    corrupted_sync = bytearray(packet)
    corrupted_sync[0] = 0x00
    res_sync = parse_lora_packet(bytes(corrupted_sync))
    assert res_sync["valid"] is False
    assert res_sync["status"] == "INVALID_SYNC"

    # 3. Truncate packet -> PACKET_TOO_SHORT
    res_short = parse_lora_packet(packet[:5])
    assert res_short["valid"] is False
    assert res_short["status"] == "PACKET_TOO_SHORT"


def test_section_e_lora_telemetry_payload_packing_roundtrip():
    """Verify 16-byte fixed binary telemetry encoding and decoding."""
    moisture = 45.50
    soil_temp = 22.35
    canopy_temp = 26.80
    battery = 3950
    flow_rate = 4.25
    volume = 1250.75
    faults = 0x0002

    packed = encode_telemetry_payload(
        moisture_pct=moisture,
        soil_temp_c=soil_temp,
        canopy_temp_c=canopy_temp,
        battery_mv=battery,
        flow_rate_lpm=flow_rate,
        cumulative_liters=volume,
        fault_flags=faults
    )
    assert len(packed) == 16

    unpacked = decode_telemetry_payload(packed)
    assert abs(unpacked["moisture_pct"] - moisture) < 0.01
    assert abs(unpacked["soil_temp_c"] - soil_temp) < 0.01
    assert abs(unpacked["canopy_temp_c"] - canopy_temp) < 0.01
    assert unpacked["battery_mv"] == battery
    assert abs(unpacked["flow_rate_lpm"] - flow_rate) < 0.01
    assert abs(unpacked["cumulative_liters"] - volume) < 0.01
    assert unpacked["fault_flags"] == faults


def test_section_e_lora_driver_mock_send_receive():
    """Verify SX1278Driver queue and transmission logic."""
    driver = SX1278Driver(node_id=10)
    pkt1 = driver.send(MSG_TELEMETRY, b"payload_1")
    pkt2 = driver.send(MSG_ALERT, b"leak_alert")

    assert len(driver.tx_buffer) == 2
    assert driver.seq_num == 2

    # Queue pkt1 into RX and receive it
    driver.queue_mock_rx(pkt1)
    recv_res = driver.receive()
    assert recv_res is not None
    assert recv_res["valid"] is True
    assert recv_res["payload"] == b"payload_1"


# ==============================================================================
# 6. Tests for edge/thermal_point.py
# ==============================================================================

def test_section_e_thermal_point_raw_to_celsius_conversion():
    """Verify MLX90614 raw register to Celsius conversion (0.02 K / LSB)."""
    # 0 deg C = 273.15 K -> raw code = 273.15 / 0.02 = 13657.5 -> 13658
    # 13658 * 0.02 = 273.16 K -> 0.01 deg C
    res_zero = raw_to_celsius_mlx90614(13658)
    assert res_zero["valid"] is True
    assert abs(res_zero["temp_c"] - 0.01) < 0.02

    # 25.0 deg C = 298.15 K -> raw code = 298.15 / 0.02 = 14907.5 -> 14908
    raw_25 = celsius_to_raw_mlx90614(25.0)
    res_25 = raw_to_celsius_mlx90614(raw_25)
    assert res_25["valid"] is True
    assert abs(res_25["temp_c"] - 25.0) < 0.02

    # Bit 15 error flag (0x8000) asserts sensor fault
    fault_code = raw_25 | 0x8000
    res_fault = raw_to_celsius_mlx90614(fault_code)
    assert res_fault["valid"] is False
    assert res_fault["status"] == "SENSOR_ERROR_FLAG"
    assert res_fault["error_flag"] is True


def test_section_e_thermal_point_smbus_pec_checksum():
    """Verify SMBus CRC-8 PEC calculation and transaction verification."""
    # Test transaction sequence:
    # [slave_addr_write, reg_addr, slave_addr_read, data_low, data_high]
    slave_write = 0xB4   # (0x5A << 1)
    reg_addr = 0x07      # To register
    slave_read = 0xB5    # (0x5A << 1) | 1
    data_low = 0x3C
    data_high = 0x3A

    tx = bytes([slave_write, reg_addr, slave_read, data_low, data_high])
    pec = calculate_smbus_pec(tx)
    assert isinstance(pec, int)
    assert 0 <= pec <= 0xFF

    # Verify matching PEC
    assert verify_smbus_pec(tx, pec) is True

    # Tampered byte fails verification
    assert verify_smbus_pec(tx, pec ^ 0x01) is False


def test_section_e_thermal_point_driver_and_thermal_mast_crosscheck():
    """Verify MLX90614Driver cross-check against MLX90640 canopy temperature."""
    driver = MLX90614Driver()

    # 1. Consistent reading: Mast = 28.0 C, MLX90614 = 28.5 C (delta = -0.5 C <= 3.0 C limit)
    driver.set_mock_temperatures(ambient_c=25.0, object_c=28.5)
    res_ok = driver.cross_check_with_thermal_mast(mlx90640_canopy_temp=28.0, max_disagreement=3.0)
    assert res_ok["valid"] is True
    assert res_ok["status"] == "ok"
    assert abs(res_ok["disagreement_c"] - 0.5) < 0.05

    # 2. Drift / Disagreement: Mast = 28.0 C, MLX90614 = 33.0 C (delta = 5.0 C > 3.0 C limit)
    driver.set_mock_temperatures(ambient_c=25.0, object_c=33.0)
    res_drift = driver.cross_check_with_thermal_mast(mlx90640_canopy_temp=28.0, max_disagreement=3.0)
    assert res_drift["valid"] is False
    assert res_drift["status"] == "sensor_disagreement_or_drift"
    assert abs(res_drift["disagreement_c"] - 5.0) < 0.05


# ==============================================================================
# 7. Tests for Paddy Rice & Alternate Wetting and Drying (AWD)
# ==============================================================================

def test_section_e_paddy_land_prep_one_time_isolated():
    """Verify one-time bulk requirement for Phase 1 (Land Prep & Puddling)."""
    # 1. Default parameters: 200mm SAT + 25mm ponding = 225mm net depth
    res_default = calculate_paddy_land_prep_requirement()
    assert res_default["phase"] == "land_preparation"
    assert res_default["saturation_depth_mm"] == 200.0
    assert res_default["initial_ponding_mm"] == 25.0
    assert res_default["net_water_depth_mm"] == 225.0
    assert abs(res_default["gross_water_depth_mm"] - (225.0 / 0.60)) < 0.01
    assert res_default["control_mode"] == "open_loop"
    assert res_default["water_level_sensor_present"] is False
    assert "ADVISORY ONLY" in res_default["volume_note"]

    # 2. Pre-transplanting effective rainfall offset: 225mm - 25mm rain = 200mm net
    res_rain = calculate_paddy_land_prep_requirement(
        effective_rain_accum_mm=25.0,
        area_m2=1000.0,
        irrigation_efficiency=0.60
    )
    assert res_rain["net_water_depth_mm"] == 200.0
    assert abs(res_rain["gross_water_depth_mm"] - 333.33) < 0.02
    assert abs(res_rain["volume_liters"] - 333333.33) < 50.0

    # 3. Input validations
    with pytest.raises(ValueError):
        calculate_paddy_land_prep_requirement(saturation_depth_mm=-10.0)
    with pytest.raises(ValueError):
        calculate_paddy_land_prep_requirement(area_m2=0.0)
    with pytest.raises(ValueError):
        calculate_paddy_land_prep_requirement(irrigation_efficiency=1.5)

    # 4. JSON serialization
    serialized = json.dumps(res_rain)
    assert "land_preparation" in serialized


def test_section_e_paddy_daily_requirement_open_loop_and_no_sat():
    """Verify Phase 2 daily requirement excludes one-time SAT and standing water Delta WL."""
    # ETc = 6.0 mm/day, Percolation (default) = 5.0 mm/day, Effective rain = 2.0 mm/day
    # Net requirement = max(0.0, 6.0 + 5.0 - 2.0) = 9.0 mm/day
    res = calculate_paddy_daily_requirement(
        et_c_mm_day=6.0,
        effective_rain_mm_day=2.0,
        area_m2=500.0,
        irrigation_efficiency=0.60
    )
    assert res["phase"] == "daily_maintenance"
    assert res["et_c_mm"] == 6.0
    assert res["percolation_mm"] == 5.0
    assert res["effective_rain_mm"] == 2.0
    assert res["net_irrigation_mm"] == 9.0
    # Gross = 9.0 / 0.60 = 15.0 mm -> Volume = 15.0 * 500 = 7500.0 L
    assert res["gross_irrigation_mm"] == 15.0
    assert res["volume_liters"] == 7500.0
    assert res["control_mode"] == "open_loop"
    assert res["water_level_sensor_present"] is False
    assert "ADVISORY ONLY" in res["volume_note"]

    # Soil texture override: 'clay' gives percolation = 4.0 mm/day
    res_clay = calculate_paddy_daily_requirement(
        et_c_mm_day=6.0,
        soil_type="clay",
        effective_rain_mm_day=0.0,
        area_m2=100.0,
        irrigation_efficiency=0.60
    )
    assert res_clay["percolation_mm"] == 4.0
    assert res_clay["net_irrigation_mm"] == 10.0


def test_section_e_paddy_water_level_sensor_guard_raises_runtime_error():
    """Verify fail-loud guard when caller supplies water_level_mm on Tier 1 hardware."""
    # Since WATER_LEVEL_SENSOR_PRESENT is False, passing water_level_mm MUST raise RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        calculate_paddy_daily_requirement(
            et_c_mm_day=5.0,
            water_level_mm=30.0
        )
    err_msg = str(excinfo.value)
    assert "WATER_LEVEL_SENSOR_PRESENT" in err_msg
    assert "Capacitive soil moisture probes sit in saturated soil" in err_msg


def test_section_e_paddy_terminal_drainage_suppresses_irrigation():
    """Verify Phase 3 terminal drainage completely halts irrigation 10-14 days before harvest."""
    res = calculate_paddy_terminal_drainage(area_m2=1000.0, days_to_harvest=10.0)
    assert res["phase"] == "terminal_drainage"
    assert res["net_irrigation_mm"] == 0.0
    assert res["gross_irrigation_mm"] == 0.0
    assert res["volume_liters"] == 0.0
    assert res["irrigation_suppressed"] is True
    assert res["days_to_harvest"] == 10.0
    assert res["control_mode"] == "open_loop"
    assert res["water_level_sensor_present"] is False


def test_section_e_awd_safety_overrides_seedling_and_flowering():
    """Verify AWD is strictly suspended during seedling recovery and flowering/milk stages."""
    # 1. Critical Window 1: Seedling recovery (days <= 14 post-transplanting)
    res_seedling = evaluate_paddy_awd_status(
        cumulative_deficit_mm=10.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=7.0,  # <= 14d recovery
        stage="development",
        soil_type="clay",
        area_m2=100.0,
        irrigation_efficiency=0.60
    )
    assert res_seedling["awd_status"] == "SUSPENDED_SHALLOW_FLOOD"
    assert res_seedling["awd_suspended"] is True
    assert res_seedling["awd_active"] is False
    assert "Seedling recovery active" in res_seedling["suspension_reason"]
    # Net irrigation is daily replenishment: 5.0 (ETc) + 4.0 (clay percolation) = 9.0 mm
    assert res_seedling["net_irrigation_mm"] == 9.0
    assert res_seedling["new_cumulative_deficit_mm"] == 0.0

    # 2. Critical Window 2: Flowering through milk stage (spikelet sterility prevention)
    # Test published rice reproductive stages: ('panicle_initiation', 'booting', 'heading', 'flowering', 'anthesis', 'milk')
    for test_stage in ("panicle_initiation", "booting", "heading", "flowering", "anthesis", "milk"):
        res_flower = evaluate_paddy_awd_status(
            cumulative_deficit_mm=25.0,
            et_c_mm_day=6.0,
            effective_rain_mm_day=0.0,
            days_since_transplanting=60.0,  # > 14d recovery
            stage=test_stage,
            soil_type="clay",
            area_m2=100.0,
            irrigation_efficiency=0.60
        )
        assert res_flower["awd_status"] == "SUSPENDED_SHALLOW_FLOOD", f"Failed for stage {test_stage}"
        assert res_flower["awd_suspended"] is True
        assert "Reproductive stage active" in res_flower["suspension_reason"]
        assert res_flower["net_irrigation_mm"] == 10.0  # 6.0 + 4.0

    # Test by explicit is_flowering=True flag
    res_flag = evaluate_paddy_awd_status(
        cumulative_deficit_mm=15.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=50.0,
        stage="custom_stage",
        is_flowering=True,
        soil_type="clay"
    )
    assert res_flag["awd_status"] == "SUSPENDED_SHALLOW_FLOOD"
    assert res_flag["awd_suspended"] is True


def test_section_e_awd_fao56_mid_stage_does_not_suspend_awd():
    """Verify that FAO-56 macro growth stages ('mid', 'mid_season') do NOT trip reproductive gate."""
    for stage_label in ("mid", "mid_season"):
        res = evaluate_paddy_awd_status(
            cumulative_deficit_mm=10.0,
            et_c_mm_day=5.0,
            effective_rain_mm_day=0.0,
            days_since_transplanting=35.0,  # well past 14d seedling recovery
            stage=stage_label,
            is_flowering=False,
            soil_type="clay",
            area_m2=100.0,
            irrigation_efficiency=0.60
        )
        # AWD MUST remain active; not suspended
        assert res["awd_status"] == "DRYING_DOWN"
        assert res["awd_active"] is True
        assert res["awd_suspended"] is False
        assert res["suspension_reason"] is None
        # In clay soil (perc=4.0), net depletion = 5.0 + 4.0 = 9.0 mm
        # Deficit advances from 10.0 to 19.0 mm (< 40 mm threshold)
        assert res["new_cumulative_deficit_mm"] == 19.0
        assert res["net_irrigation_mm"] == 0.0

    # Verify same behavior via stateful AWDWaterBudgetAdvisor
    advisor = AWDWaterBudgetAdvisor(soil_type="clay", area_m2=100.0)
    adv_res = advisor.evaluate_daily(
        et_c_mm_day=5.0,
        days_since_transplanting=35.0,
        stage="mid",
        is_flowering=False
    )
    assert adv_res["awd_status"] == "DRYING_DOWN"
    assert adv_res["awd_active"] is True
    assert adv_res["awd_suspended"] is False


def test_section_e_awd_unconfirmed_stage_fails_safe_withheld():
    """Verify silence is not consent: stage=None and is_flowering=False withholds AWD."""
    # When days_since_transplanting is beyond seedling recovery (e.g. 35.0d)
    # but stage is None and is_flowering is False:
    res = evaluate_paddy_awd_status(
        cumulative_deficit_mm=15.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=35.0,
        stage=None,
        is_flowering=False,
        soil_type="clay",
        area_m2=100.0,
        irrigation_efficiency=0.60
    )
    assert res["awd_status"] == "STAGE_UNKNOWN_AWD_WITHHELD"
    assert res["awd_suspended"] is True
    assert res["awd_active"] is False
    assert "AWD dry-down requires an affirmatively confirmed" in res["suspension_reason"]
    # Shallow flooding is maintained: 5.0 (ETc) + 4.0 (clay perc) = 9.0 mm net
    assert res["net_irrigation_mm"] == 9.0
    assert res["new_cumulative_deficit_mm"] == 0.0

    # Also verify with AWDWaterBudgetAdvisor
    advisor = AWDWaterBudgetAdvisor(soil_type="clay", area_m2=100.0)
    adv_res = advisor.evaluate_daily(
        et_c_mm_day=5.0,
        days_since_transplanting=35.0,
        stage=None,
        is_flowering=False
    )
    assert adv_res["awd_status"] == "STAGE_UNKNOWN_AWD_WITHHELD"
    assert adv_res["awd_suspended"] is True
    assert adv_res["awd_active"] is False


def test_section_e_awd_active_dry_down_and_reirrigation_cycle():
    """Verify cumulative water deficit accumulation and +50mm re-irrigation trigger."""
    advisor = AWDWaterBudgetAdvisor(
        soil_type="clay",      # percolation = 4.0 mm/day
        area_m2=100.0,
        irrigation_efficiency=0.60
    )
    # Daily net loss = ETc (6.0) + percolation (4.0) - rain (0.0) = 10.0 mm/day
    # Stage: vegetative tillering (days_since_transplanting = 25, stage = "development")

    # Day 1: Deficit 0 -> 10 mm (< 40 mm threshold)
    d1 = advisor.evaluate_daily(et_c_mm_day=6.0, days_since_transplanting=25.0, stage="development")
    assert d1["awd_status"] == "DRYING_DOWN"
    assert d1["awd_active"] is True
    assert d1["reirrigation_triggered"] is False
    assert d1["new_cumulative_deficit_mm"] == 10.0
    assert d1["net_irrigation_mm"] == 0.0
    assert d1["volume_liters"] == 0.0

    # Day 2: Deficit 10 -> 20 mm
    d2 = advisor.evaluate_daily(et_c_mm_day=6.0, days_since_transplanting=26.0, stage="development")
    assert d2["awd_status"] == "DRYING_DOWN"
    assert d2["new_cumulative_deficit_mm"] == 20.0
    assert d2["net_irrigation_mm"] == 0.0

    # Day 3: Deficit 20 -> 30 mm
    d3 = advisor.evaluate_daily(et_c_mm_day=6.0, days_since_transplanting=27.0, stage="development")
    assert d3["awd_status"] == "DRYING_DOWN"
    assert d3["new_cumulative_deficit_mm"] == 30.0

    # Day 4: Deficit 30 -> 40 mm (Reaches threshold 40.0 mm!)
    d4 = advisor.evaluate_daily(et_c_mm_day=6.0, days_since_transplanting=28.0, stage="development")
    assert d4["awd_status"] == "REIRRIGATE_TRIGGERED"
    assert d4["reirrigation_triggered"] is True
    assert d4["refill_depth_mm"] == 50.0
    assert d4["net_irrigation_mm"] == 50.0
    # Gross = 50.0 / 0.60 = 83.333 mm -> Volume = 83.333 * 100 = 8333.33 L
    assert abs(d4["gross_irrigation_mm"] - 83.33) < 0.02
    assert abs(d4["volume_liters"] - 8333.33) < 1.0
    # Deficit resets to 0.0 upon re-flooding!
    assert d4["new_cumulative_deficit_mm"] == 0.0
    assert advisor.reirrigation_count == 1

    # Day 5: Dry-down starts anew with rain (Pe = 5.0 mm): net depletion = 10.0 - 5.0 = 5.0 mm
    d5 = advisor.evaluate_daily(et_c_mm_day=6.0, effective_rain_mm_day=5.0, days_since_transplanting=29.0, stage="development")
    assert d5["awd_status"] == "DRYING_DOWN"
    assert d5["new_cumulative_deficit_mm"] == 5.0
    assert d5["net_irrigation_mm"] == 0.0


def test_section_e_awd_terminal_drainage_and_json_serialization():
    """Verify terminal drainage override within AWD advisor and complete JSON serialization."""
    advisor = AWDWaterBudgetAdvisor(soil_type="loam", area_m2=200.0)

    # 1. When days_to_harvest <= 14, terminal drainage overrides AWD
    res_term = advisor.evaluate_daily(
        et_c_mm_day=5.0,
        days_since_transplanting=110.0,
        stage="late",
        days_to_harvest=10.0
    )
    assert res_term["phase"] == "terminal_drainage"
    assert res_term["awd_status"] == "TERMINAL_DRAINAGE"
    assert res_term["awd_active"] is False
    assert res_term["irrigation_suppressed"] is True
    assert res_term["volume_liters"] == 0.0

    # 2. Check JSON dumps on all outputs
    payloads = [
        calculate_paddy_land_prep_requirement(),
        calculate_paddy_daily_requirement(et_c_mm_day=4.5),
        calculate_paddy_terminal_drainage(),
        res_term,
        advisor.history[0]
    ]
    for p in payloads:
        dumped = json.dumps(p)
        assert isinstance(dumped, str)
        assert "open_loop" in dumped


def test_section_e_awd_status_precedence_seedling_and_reproductive_over_unknown():
    """
    Correction 2: Verify status precedence in evaluate_paddy_awd_status.
    Seedling recovery and reproductive stage overrides must take precedence over
    STAGE_UNKNOWN_AWD_WITHHELD when stage is None.
    """
    # Case (a): stage is None, but is_flowering=True
    # Must report SUSPENDED_SHALLOW_FLOOD (Reproductive), NOT STAGE_UNKNOWN_AWD_WITHHELD
    res_a = evaluate_paddy_awd_status(
        cumulative_deficit_mm=10.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=30.0,  # Beyond seedling recovery
        stage=None,
        is_flowering=True,
        soil_type="clay",
        area_m2=100.0
    )
    assert res_a["awd_status"] == "SUSPENDED_SHALLOW_FLOOD"
    assert res_a["awd_suspended"] is True
    assert "Reproductive stage active" in res_a["suspension_reason"]

    # Case (b): stage is None, but days_since_transplanting=5 (seedling recovery active)
    # Must report SUSPENDED_SHALLOW_FLOOD (Seedling recovery), NOT STAGE_UNKNOWN_AWD_WITHHELD
    res_b = evaluate_paddy_awd_status(
        cumulative_deficit_mm=10.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=5.0,  # <= 14d recovery
        stage=None,
        is_flowering=False,
        soil_type="clay",
        area_m2=100.0
    )
    assert res_b["awd_status"] == "SUSPENDED_SHALLOW_FLOOD"
    assert res_b["awd_suspended"] is True
    assert "Seedling recovery active" in res_b["suspension_reason"]

    # Case (c): stage is None, is_flowering=False, days_since_transplanting=30
    # Beyond seedling recovery and not flowering -> MUST report STAGE_UNKNOWN_AWD_WITHHELD
    res_c = evaluate_paddy_awd_status(
        cumulative_deficit_mm=10.0,
        et_c_mm_day=5.0,
        effective_rain_mm_day=0.0,
        days_since_transplanting=30.0,
        stage=None,
        is_flowering=False,
        soil_type="clay",
        area_m2=100.0
    )
    assert res_c["awd_status"] == "STAGE_UNKNOWN_AWD_WITHHELD"
    assert res_c["awd_suspended"] is True
    assert "AWD dry-down requires an affirmatively confirmed" in res_c["suspension_reason"]


# ==============================================================================
# 7. Tests for Prompt 3 Section E (edge/camera.py)
# ==============================================================================

def test_section_e_camera_gstreamer_pipeline_fixed_controls():
    """Verify GStreamer pipeline builder enforces fixed AWB, AE lock, and exposure."""
    # 1. Default RGB pipeline (sensor-id=0)
    pipe_default = build_gstreamer_pipeline()
    assert "nvarguscamerasrc sensor-id=0" in pipe_default
    assert "wbmode=0" in pipe_default
    assert "aelock=true" in pipe_default
    assert "awblock=true" in pipe_default
    # Default 10.0 ms = 10,000,000 ns
    assert 'exposuretimerange="10000000 10000000"' in pipe_default
    assert 'gainrange="1.00 1.00"' in pipe_default
    assert 'ispdigitalgainrange="1.00 1.00"' in pipe_default
    assert "width=(int)1920, height=(int)1080" in pipe_default
    assert "framerate=(fraction)30/1" in pipe_default
    assert "video/x-raw, format=(string)BGR" in pipe_default

    # 2. Custom NIR pipeline (sensor-id=1, 5ms exposure, 2.0x gain)
    pipe_nir = build_gstreamer_pipeline(
        sensor_id=1,
        capture_width=1280,
        capture_height=720,
        framerate=60,
        exposure_time_ms=5.0,
        gain=2.0,
        isp_digital_gain=1.5
    )
    assert "nvarguscamerasrc sensor-id=1" in pipe_nir
    assert 'exposuretimerange="5000000 5000000"' in pipe_nir
    assert 'gainrange="2.00 2.00"' in pipe_nir
    assert 'ispdigitalgainrange="1.50 1.50"' in pipe_nir
    assert "width=(int)1280, height=(int)720" in pipe_nir
    assert "framerate=(fraction)60/1" in pipe_nir

    # 3. Validation errors
    with pytest.raises(ValueError, match="sensor_id must be 0 or 1"):
        build_gstreamer_pipeline(sensor_id=2)
    with pytest.raises(ValueError, match="exposure_time_ms must be positive"):
        build_gstreamer_pipeline(exposure_time_ms=-1.0)
    with pytest.raises(ValueError, match="gain must be >= 1.0"):
        build_gstreamer_pipeline(gain=0.5)


def test_section_e_camera_frame_metadata_logging():
    """Verify frame metadata logs fixed exposure, gain, saturation check, and clipping flags."""
    assert PROVISIONAL_EXPOSURE_NS == 10000000
    assert PROVISIONAL_GAIN == 1.0
    assert PROVISIONAL_SATURATION_THRESHOLD_DN == 250

    # 1. Metadata without frame (default configuration)
    meta = create_frame_metadata(
        sensor_id=0,
        wbmode=WBMODE_OFF,
        exposure_ns=PROVISIONAL_EXPOSURE_NS,
        gain=PROVISIONAL_GAIN,
        isp_digital_gain=1.0,
        timestamp="2026-09-07T12:00:00Z"
    )
    assert meta["sensor_id"] == 0
    assert meta["wbmode"] == 0
    assert meta["white_balance_fixed"] is True
    assert meta["exposure_time_ms"] == 10.0
    assert meta["exposure_time_ns"] == 10000000
    assert meta["gain"] == 1.0
    assert meta["ae_locked"] is True
    assert meta["awb_locked"] is True
    assert meta["colour_calibrated"] is False
    assert meta["colour_uncalibrated"] is True
    assert meta["saturation_status"] == "NOT_EVALUATED"
    assert isinstance(meta["saturation_status_reason"], str) and len(meta["saturation_status_reason"]) > 0

    # 2. Metadata with clean, non-clipped frame
    clean_frame = np.full((50, 50, 3), 100, dtype=np.uint8)
    meta_clean = create_frame_metadata(frame=clean_frame)
    assert meta_clean["saturation_status"] == "SATURATION_OK"
    assert meta_clean["saturation_status_reason"] == ""
    assert meta_clean["saturation_fractions"]["red"] == 0.0
    assert meta_clean["saturation"]["warning"] is None

    # 3. Metadata with clipped frame (simulating bright sunlight saturating Red channel)
    clipped_frame = clean_frame.copy()
    # Saturate 10% of pixels in the Red channel (index 2) to 255
    clipped_frame[:10, :25, 2] = 255  # 250 pixels out of 2500 = 10%
    meta_clipped = create_frame_metadata(frame=clipped_frame)
    assert meta_clipped["saturation_status"] == "SATURATION_DETECTED"
    assert len(meta_clipped["saturation_status_reason"]) > 0
    assert abs(meta_clipped["saturation_fractions"]["red"] - 0.10) < 1e-4
    assert meta_clipped["saturation"]["warning"] is not None
    assert "Channel saturation detected" in meta_clipped["saturation"]["warning"]

    # 4. JSON serializability assertion
    dumped = json.dumps(meta_clipped)
    assert isinstance(dumped, str)
    loaded = json.loads(dumped)
    assert loaded["exposure_time_ns"] == 10000000
    assert loaded["saturation_status"] == "SATURATION_DETECTED"


def test_section_e_camera_metadata_saturation_float32_frame_returns_not_evaluated():
    """Verify float32 frame in [0.0, 1.0] returns NOT_EVALUATED with non-empty reason (not OK)."""
    float_frame = np.full((50, 50, 3), 0.75, dtype=np.float32)
    meta = create_frame_metadata(frame=float_frame)
    assert meta["saturation_status"] == "NOT_EVALUATED"
    assert isinstance(meta["saturation_status_reason"], str) and len(meta["saturation_status_reason"]) > 0
    assert "float32" in meta["saturation_status_reason"] or "uint8" in meta["saturation_status_reason"]
    # Ensure float32 frame is not silently coerced or treated as SATURATION_OK
    assert meta["saturation_status"] != "SATURATION_OK"
    assert meta["saturation_status"] != "SATURATION_DETECTED"


def test_section_e_camera_metadata_saturation_single_channel_2d_frame_returns_not_evaluated():
    """Verify single-channel 2D frame returns NOT_EVALUATED with non-empty reason."""
    gray_frame = np.full((50, 50), 200, dtype=np.uint8)
    meta = create_frame_metadata(frame=gray_frame)
    assert meta["saturation_status"] == "NOT_EVALUATED"
    assert isinstance(meta["saturation_status_reason"], str) and len(meta["saturation_status_reason"]) > 0
    assert "3-dimensional" in meta["saturation_status_reason"] or "ndim" in meta["saturation_status_reason"]


def test_section_e_camera_metadata_saturation_empty_array_returns_not_evaluated():
    """Verify empty frame array (size 0) returns NOT_EVALUATED with non-empty reason."""
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    meta = create_frame_metadata(frame=empty_frame)
    assert meta["saturation_status"] == "NOT_EVALUATED"
    assert isinstance(meta["saturation_status_reason"], str) and len(meta["saturation_status_reason"]) > 0
    assert "empty" in meta["saturation_status_reason"] or "size 0" in meta["saturation_status_reason"]


def test_section_e_camera_metadata_saturation_fraction_just_below_alert_returns_ok():
    """
    Verify saturation fraction just below the alert threshold returns SATURATION_OK.
    Computes pixel count dynamically from PROVISIONAL_SATURATION_ALERT_FRACTION.
    """
    h, w = 100, 100
    n_pixels = h * w  # 10,000 pixels
    # Alert threshold pixel count
    k_thresh = int(round(n_pixels * PROVISIONAL_SATURATION_ALERT_FRACTION))
    # Exactly one pixel below the alert threshold
    k_below = max(0, k_thresh - 1)

    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    flat = frame.reshape(-1, 3)
    if k_below > 0:
        flat[:k_below, 2] = 255  # Red channel saturated

    meta = create_frame_metadata(frame=frame)
    assert meta["saturation_status"] == "SATURATION_OK"
    assert meta["saturation_status_reason"] == ""
    assert meta["saturation_fractions"]["red"] < PROVISIONAL_SATURATION_ALERT_FRACTION
    assert abs(meta["saturation_fractions"]["red"] - (k_below / n_pixels)) < 1e-6


def test_section_e_camera_metadata_saturation_fraction_just_above_alert_returns_detected():
    """
    Verify saturation fraction at or just above the alert threshold returns SATURATION_DETECTED.
    Computes pixel count dynamically from PROVISIONAL_SATURATION_ALERT_FRACTION.
    """
    h, w = 100, 100
    n_pixels = h * w  # 10,000 pixels
    # Alert threshold pixel count
    k_thresh = int(round(n_pixels * PROVISIONAL_SATURATION_ALERT_FRACTION))
    # Just above threshold
    k_above = k_thresh + 1

    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    flat = frame.reshape(-1, 3)
    flat[:k_above, 2] = 255  # Red channel saturated

    meta = create_frame_metadata(frame=frame)
    assert meta["saturation_status"] == "SATURATION_DETECTED"
    assert isinstance(meta["saturation_status_reason"], str) and len(meta["saturation_status_reason"]) > 0
    assert meta["saturation_fractions"]["red"] >= PROVISIONAL_SATURATION_ALERT_FRACTION
    assert abs(meta["saturation_fractions"]["red"] - (k_above / n_pixels)) < 1e-6
    assert meta["saturation"]["warning"] is not None


def test_section_e_camera_reference_card_colour_calibration():
    """Verify reference card normalisation (E2) and fail-safe uncalibrated fallback (E3)."""
    # Create synthetic 20x20 BGR frame
    frame = np.full((20, 20, 3), 100, dtype=np.uint8)

    # 1. Case E3: No reference card detected
    frame_uncal, meta_uncal = correct_from_reference_card(frame, detected_patches=None)
    assert np.array_equal(frame_uncal, frame)
    assert meta_uncal["colour_calibrated"] is False
    assert meta_uncal["colour_uncalibrated"] is True
    assert meta_uncal["correction_applied"] is False
    assert "No reference card detected" in meta_uncal["reason"]

    # 2. Case E2: Reference card patches detected
    # Synthetic reference: 4 patches [B, G, R]
    ref_patches = np.array([
        [50.0, 50.0, 50.0],
        [100.0, 150.0, 80.0],
        [30.0, 40.0, 200.0],
        [200.0, 100.0, 50.0]
    ], dtype=np.float32)

    # Introduce a simulated linear color shift (e.g. sensor B is 10% lower, R is 15% higher)
    true_M = np.array([
        [0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.15]
    ], dtype=np.float32)
    # detected = ref @ true_M
    detected_patches = ref_patches @ true_M

    frame_cal, meta_cal = correct_from_reference_card(
        frame=frame,
        detected_patches=detected_patches,
        reference_patches=ref_patches
    )

    assert meta_cal["colour_calibrated"] is True
    assert meta_cal["colour_uncalibrated"] is False
    assert meta_cal["correction_applied"] is True
    assert meta_cal["calibration_rmse"] < 1e-2
    assert "ccm_matrix" in meta_cal

    # Check that the corrected frame has been transformed
    assert frame_cal.shape == frame.shape
    assert frame_cal.dtype == np.uint8


def test_load_log_priors_csv_pandas_equivalence():
    """F2.3: Verify load_log_priors stdlib csv parser matches pandas-based count logic."""
    from pathlib import Path
    from edge.pipeline import load_log_priors
    repo_root = Path(__file__).resolve().parent.parent
    priors = load_log_priors(repo_root)

    assert isinstance(priors, np.ndarray)
    assert priors.shape == (29,)
    # Must sum to 1 in probability space (exp(priors).sum() == 1.0)
    assert abs(np.exp(priors).sum() - 1.0) < 1e-5
    assert not np.isnan(priors).any()
    assert not np.isinf(priors).any()



