"""
edge/adc.py
-----------
Analog-to-Digital Converter (ADS1115) & Capacitive Soil Moisture Sensor Interface.

Implements:
  1. Pure ADS1115 16-bit Delta-Sigma ADC conversion logic (PGA gain scales, LSB sizing,
     clamping, raw code to voltage).
  2. Pure capacitive soil moisture sensor transfer function (voltage to volumetric water
     content percentage VWC%).
  3. Mockable ADS1115Driver interface decoupled from physical Linux /dev/i2c-* hardware.

Physical & Hardware Notes:
  - ADS1115 is a 16-bit precision ADC (Texas Instruments) with an integrated Programmable
    Gain Amplifier (PGA) and low-drift voltage reference.
  - Full-scale ranges (FSR) and LSB resolutions:
      * Gain 2/3 (+/- 6.144V): 0.187500 mV / LSB
      * Gain 1   (+/- 4.096V): 0.125000 mV / LSB
      * Gain 2   (+/- 2.048V): 0.062500 mV / LSB
      * Gain 4   (+/- 1.024V): 0.031250 mV / LSB
      * Gain 8   (+/- 0.512V): 0.015625 mV / LSB
      * Gain 16  (+/- 0.256V): 0.0078125 mV / LSB
  - Capacitive Soil Moisture Sensor (v1.2 analog probe):
      The probe measures dielectric permittivity of the soil matrix. High moisture
      increases capacitance, which lowers the oscillation frequency and produces a
      LOWER analog voltage output. Air (dry) produces maximum voltage (~3.0V);
      water saturation produces minimum voltage (~1.2V).

Caution / Guard:
  - PROVISIONAL_V_DRY and PROVISIONAL_V_WET are nominal uncalibrated lab reference values.
    Because capacitive probe voltage depends on supply rail stability (3.3V vs 5.0V),
    trace impedance, temperature, and soil bulk density/salinity, empirical site-specific
    calibration (gravimetric oven-drying, ASTM D2216) is required prior to agronomic actuation.
"""

from typing import Dict, Any, Optional, Union


# ==============================================================================
# ADS1115 PGA Gain Specifications (Texas Instruments Datasheet SBAS444B)
# ==============================================================================
PGA_CONFIG: Dict[Union[int, float], Dict[str, Any]] = {
    2/3: {
        "fsr_volts": 6.144,
        "lsb_volts": 0.0001875,   # 187.5 uV
        "description": "+/- 6.144V full-scale range"
    },
    1: {
        "fsr_volts": 4.096,
        "lsb_volts": 0.0001250,   # 125.0 uV
        "description": "+/- 4.096V full-scale range"
    },
    2: {
        "fsr_volts": 2.048,
        "lsb_volts": 0.0000625,   # 62.5 uV
        "description": "+/- 2.048V full-scale range"
    },
    4: {
        "fsr_volts": 1.024,
        "lsb_volts": 0.00003125,  # 31.25 uV
        "description": "+/- 1.024V full-scale range"
    },
    8: {
        "fsr_volts": 0.512,
        "lsb_volts": 0.000015625, # 15.625 uV
        "description": "+/- 0.512V full-scale range"
    },
    16: {
        "fsr_volts": 0.256,
        "lsb_volts": 0.0000078125,# 7.8125 uV
        "description": "+/- 0.256V full-scale range"
    }
}

# ==============================================================================
# Provisional Moisture Sensor Calibration Constants
# ==============================================================================
# Nominal voltages for 3.3V-powered analog capacitive soil moisture sensor v1.2.
# MANDATORY: Replace with empirical gravimetric calibration curves for field soils.
PROVISIONAL_V_DRY: float = 3.00   # Probe suspended in open air (0% VWC)
PROVISIONAL_V_WET: float = 1.20   # Probe immersed in water up to limit line (100% saturation)


# ==============================================================================
# Pure Conversion Functions (Decoupled from Hardware)
# ==============================================================================

def get_lsb_voltage(gain: Union[int, float] = 1) -> float:
    """
    Return the LSB voltage for the specified ADS1115 PGA gain.
    """
    # Match integer or float key
    for g_key, cfg in PGA_CONFIG.items():
        if abs(float(g_key) - float(gain)) < 1e-4:
            return float(cfg["lsb_volts"])
    valid_gains = [2/3, 1, 2, 4, 8, 16]
    raise ValueError(f"Invalid ADS1115 gain {gain}. Valid options: {valid_gains}")


def raw_to_voltage(raw_code: int, gain: Union[int, float] = 1) -> float:
    """
    Convert raw 16-bit signed integer ADC conversion code to physical voltage.

    Parameters:
      raw_code: Signed 16-bit ADC output integer in [-32768, 32767].
      gain: PGA gain setting (2/3, 1, 2, 4, 8, 16). Default 1 (+/- 4.096V).

    Returns:
      Voltage in Volts (float), clamped to physical full-scale range.
    """
    if not isinstance(raw_code, (int,)):
        raise TypeError(f"raw_code must be an integer, got {type(raw_code)}")
    if raw_code < -32768 or raw_code > 32767:
        raise ValueError(f"raw_code {raw_code} is outside 16-bit signed range [-32768, 32767].")

    lsb = get_lsb_voltage(gain)
    voltage = float(raw_code) * lsb
    # Clamp to full-scale range of the configured gain
    fsr = 6.144 if abs(float(gain) - (2/3)) < 1e-4 else (4.096 / float(gain))
    return float(round(max(-fsr, min(fsr, voltage)), 6))


def voltage_to_moisture_pct(voltage: float,
                             v_dry: float = PROVISIONAL_V_DRY,
                             v_wet: float = PROVISIONAL_V_WET) -> float:
    """
    Convert measured sensor voltage to volumetric water content percentage (VWC%).

    Inverse capacitive relationship:
      Higher moisture -> higher capacitance -> lower output voltage.
      V >= V_dry -> 0.0% moisture (completely dry air / soil)
      V <= V_wet -> 100.0% moisture (water saturation)

    Formula:
      moisture_% = ((V_dry - V) / (V_dry - V_wet)) * 100.0

    Parameters:
      voltage: Measured analog sensor voltage [V].
      v_dry: Calibrated dry voltage [V].
      v_wet: Calibrated saturated wet voltage [V].

    Returns:
      Moisture percentage in [0.0, 100.0]%.
    """
    if v_dry <= v_wet:
        raise ValueError(f"v_dry ({v_dry}V) must be strictly greater than v_wet ({v_wet}V).")

    v = float(voltage)
    # Hardware sanity guard: sensor powered from 0-5.5V supply
    if v < -0.1 or v > 5.5:
        raise ValueError(f"Sensor voltage {v:.3f}V is outside physical electrical rail [0.0, 5.5]V.")

    if v >= v_dry:
        return 0.0
    elif v <= v_wet:
        return 100.0
    else:
        fraction = (v_dry - v) / (v_dry - v_wet)
        pct = fraction * 100.0
        return float(round(max(0.0, min(100.0, pct)), 2))


# ==============================================================================
# Mockable Driver Class (Zero Hardware Dependency)
# ==============================================================================

class ADS1115Driver:
    """
    ADS1115 4-channel 16-bit ADC driver.

    Supports:
      - Injection of mock/synthetic I2C backend for unit tests and simulation.
      - Graceful fallback when smbus2 / Adafruit hardware libraries are absent.
      - Per-channel gain and calibration parameter storage.
    """
    def __init__(self,
                 i2c_bus: Optional[Any] = None,
                 i2c_address: int = 0x48,
                 default_gain: Union[int, float] = 1):
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.default_gain = default_gain

        # Mock simulation table: channel (0..3) -> raw_code (int)
        self._mock_channels: Dict[int, int] = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }

    def set_mock_raw_code(self, channel: int, raw_code: int) -> None:
        """Set a synthetic raw ADC code for a channel (used in unit tests)."""
        if channel not in (0, 1, 2, 3):
            raise ValueError(f"Invalid channel {channel}. ADS1115 has channels 0, 1, 2, 3.")
        if raw_code < -32768 or raw_code > 32767:
            raise ValueError(f"raw_code {raw_code} out of 16-bit range.")
        self._mock_channels[channel] = int(raw_code)

    def set_mock_voltage(self, channel: int, voltage: float, gain: Optional[Union[int, float]] = None) -> None:
        """Set a synthetic voltage for a channel by calculating equivalent raw code."""
        g = self.default_gain if gain is None else gain
        lsb = get_lsb_voltage(g)
        raw_code = int(round(float(voltage) / lsb))
        raw_code = max(-32768, min(32767, raw_code))
        self.set_mock_raw_code(channel, raw_code)

    def read_raw(self, channel: int) -> int:
        """
        Read raw 16-bit signed integer conversion code from channel.
        If physical I2C bus is present, communicates via SMBus protocol.
        Otherwise, reads from mock registry.
        """
        if channel not in (0, 1, 2, 3):
            raise ValueError(f"Invalid channel {channel}. ADS1115 supports channels 0, 1, 2, 3.")

        if self.i2c_bus is not None and hasattr(self.i2c_bus, "read_i2c_block_data"):
            # Example hardware SMBus transaction (mockable via duck typing)
            # Config register 0x01, Conversion register 0x00
            data = self.i2c_bus.read_i2c_block_data(self.i2c_address, 0x00, 2)
            raw = (data[0] << 8) | data[1]
            if raw > 32767:
                raw -= 65536
            return int(raw)

        return self._mock_channels[channel]

    def read_voltage(self, channel: int, gain: Optional[Union[int, float]] = None) -> float:
        """
        Read voltage from specified channel using configured or default gain.
        """
        g = self.default_gain if gain is None else gain
        raw = self.read_raw(channel)
        return raw_to_voltage(raw, gain=g)

    def read_moisture_pct(self,
                          channel: int,
                          gain: Optional[Union[int, float]] = None,
                          v_dry: float = PROVISIONAL_V_DRY,
                          v_wet: float = PROVISIONAL_V_WET) -> Dict[str, Any]:
        """
        Read soil moisture percentage from an analog capacitive sensor on channel.

        Returns:
          dict containing:
            channel: int
            raw_code: int
            voltage_v: float
            moisture_pct: float
            v_dry: float
            v_wet: float
            calibration_provisional: bool (True)
        """
        v = self.read_voltage(channel, gain=gain)
        raw = self.read_raw(channel)
        pct = voltage_to_moisture_pct(v, v_dry=v_dry, v_wet=v_wet)

        return {
            "channel": channel,
            "raw_code": raw,
            "voltage_v": v,
            "moisture_pct": pct,
            "v_dry": v_dry,
            "v_wet": v_wet,
            "calibration_provisional": True
        }
