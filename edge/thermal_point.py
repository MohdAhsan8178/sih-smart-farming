"""
edge/thermal_point.py
---------------------
MLX90614 Single-Point Infrared Thermometer Interface & PEC Checksum.

Implements:
  1. Pure raw register to Celsius conversion logic (0.02 K/LSB).
  2. SMBus Packet Error Code (PEC) CRC-8 checksum calculation and validation
     (polynomial 0x07, init 0x00).
  3. Continuous cross-check integration with MLX90640 thermal mast array
     (delegates to core.thermal.cross_check_mlx90614).
  4. Mockable MLX90614Driver interface decoupled from physical Linux SMBus/I2C.

Hardware & Physical Specifications:
  - Sensor: Melexis MLX90614 non-contact infrared thermometer.
  - Interface: SMBus / I2C (standard 7-bit address 0x5A).
  - Registers (RAM):
      * 0x06: Ta (ambient sensor temperature)
      * 0x07: To (object 1 infrared temperature)
  - Temperature transfer function:
      T_kelvin = raw_code * 0.02
      T_celsius = T_kelvin - 273.15
  - Operational bounds:
      * Ambient: -40.0 C to +125.0 C
      * Object:  -70.0 C to +382.2 C
      * Bit 15: Error flag in raw register. If (raw_code & 0x8000) is set,
        the reading represents an internal sensor fault.

Role in Edge Architecture:
  - The MLX90614 has a broad conical field-of-view (typically 35 to 90 degrees)
    and zero spatial resolution. It cannot separate canopy foliage from soil.
  - It is deployed strictly as a ground-truth sanity check and continuous drift
    monitor for the MLX90640 focal-plane array. It CANNOT drive CWSI independently.
"""

from typing import Dict, Any, Optional, Union, Tuple
import struct

from core.thermal import cross_check_mlx90614, MAX_MLX90614_DISAGREEMENT


# ==============================================================================
# Hardware Register & Address Constants
# ==============================================================================
MLX90614_DEFAULT_I2C_ADDR: int = 0x5A
MLX90614_RAM_TA: int = 0x06     # Ambient temperature register
MLX90614_RAM_TOBJ1: int = 0x07  # Object 1 temperature register

# Physical measurement limits
MIN_OBJECT_TEMP_C: float = -70.0
MAX_OBJECT_TEMP_C: float = 382.2
MIN_AMBIENT_TEMP_C: float = -40.0
MAX_AMBIENT_TEMP_C: float = 125.0


# ==============================================================================
# Pure Temperature Conversion & Boundary Checking
# ==============================================================================

def raw_to_celsius_mlx90614(raw_code: int, is_ambient: bool = False) -> Dict[str, Any]:
    """
    Convert 16-bit raw MLX90614 register value to temperature in degrees Celsius.

    Melexis Datasheet Section 8.4:
      Resolution = 0.02 K / LSB
      T_kelvin = raw_code * 0.02
      T_celsius = T_kelvin - 273.15

    Parameters:
      raw_code: 16-bit unsigned register value (0x0000 to 0xFFFF).
      is_ambient: True if converting Ta (0x06), False if object To (0x07).

    Returns:
      Dict containing:
        valid: bool
        temp_c: float or None
        raw_code: int
        error_flag: bool (bit 15 set)
        status: 'ok', 'SENSOR_ERROR_FLAG', 'OUT_OF_BOUNDS', etc.
    """
    if not isinstance(raw_code, int):
        raise TypeError(f"raw_code must be an integer, got {type(raw_code)}")
    if raw_code < 0 or raw_code > 0xFFFF:
        raise ValueError(f"raw_code 0x{raw_code:04X} outside 16-bit unsigned range [0x0000, 0xFFFF].")

    # Bit 15 error flag check
    has_error_flag = bool(raw_code & 0x8000)
    if has_error_flag:
        return {
            "valid": False,
            "temp_c": None,
            "raw_code": raw_code,
            "error_flag": True,
            "status": "SENSOR_ERROR_FLAG",
            "detail": "MLX90614 internal error flag (bit 15) is asserted."
        }

    # Standard conversion
    t_kelvin = float(raw_code) * 0.02
    t_celsius = float(round(t_kelvin - 273.15, 2))

    min_t = MIN_AMBIENT_TEMP_C if is_ambient else MIN_OBJECT_TEMP_C
    max_t = MAX_AMBIENT_TEMP_C if is_ambient else MAX_OBJECT_TEMP_C

    if t_celsius < min_t or t_celsius > max_t:
        return {
            "valid": False,
            "temp_c": t_celsius,
            "raw_code": raw_code,
            "error_flag": False,
            "status": "OUT_OF_BOUNDS",
            "detail": f"Temperature {t_celsius}C outside physical sensor range [{min_t}, {max_t}]C."
        }

    return {
        "valid": True,
        "temp_c": t_celsius,
        "raw_code": raw_code,
        "error_flag": False,
        "status": "ok",
        "detail": "Measurement nominal."
    }


def celsius_to_raw_mlx90614(temp_c: float) -> int:
    """
    Synthesize raw 16-bit register code from target Celsius temperature.
    Used for unit testing and mock sensor injection.
    """
    t_kelvin = float(temp_c) + 273.15
    raw = int(round(t_kelvin / 0.02))
    return max(0, min(0x7FFF, raw))


# ==============================================================================
# Pure SMBus Packet Error Code (PEC) Calculation (CRC-8)
# ==============================================================================

def calculate_smbus_pec(data: bytes) -> int:
    """
    Calculate SMBus Packet Error Code (PEC) using CRC-8.
    Polynomial: x^8 + x^2 + x^1 + 1 (0x07), Initial value: 0x00.

    Standard SMBus read word transaction:
      Bytes: [ (slave_addr << 1) | 0 (write), reg_addr, (slave_addr << 1) | 1 (read), data_low, data_high ]
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def verify_smbus_pec(transaction_bytes: bytes, expected_pec: int) -> bool:
    """
    Verify whether the computed PEC over transaction_bytes matches expected_pec.
    """
    computed = calculate_smbus_pec(transaction_bytes)
    return bool(computed == (int(expected_pec) & 0xFF))


# ==============================================================================
# Mockable Driver Class
# ==============================================================================

class MLX90614Driver:
    """
    Mockable MLX90614 driver supporting simulated and real SMBus transactions.
    """
    def __init__(self,
                 i2c_bus: Optional[Any] = None,
                 i2c_addr: int = MLX90614_DEFAULT_I2C_ADDR):
        self.i2c_bus = i2c_bus
        self.i2c_addr = int(i2c_addr)

        # Mock registers
        self._mock_ambient_raw: int = celsius_to_raw_mlx90614(25.0)
        self._mock_object_raw: int = celsius_to_raw_mlx90614(28.0)

    def set_mock_temperatures(self, ambient_c: float, object_c: float) -> None:
        """Inject synthetic temperatures for unit testing."""
        self._mock_ambient_raw = celsius_to_raw_mlx90614(ambient_c)
        self._mock_object_raw = celsius_to_raw_mlx90614(object_c)

    def set_mock_raw_registers(self, ambient_raw: int, object_raw: int) -> None:
        """Inject synthetic raw 16-bit register codes."""
        self._mock_ambient_raw = int(ambient_raw)
        self._mock_object_raw = int(object_raw)

    def read_register(self, reg_addr: int) -> Tuple[int, int]:
        """
        Read 16-bit word and PEC byte from register.
        Returns: (raw_word, pec_byte)
        """
        if self.i2c_bus is not None and hasattr(self.i2c_bus, "read_i2c_block_data"):
            # Real SMBus read transaction: 3 bytes (data_low, data_high, pec)
            data = self.i2c_bus.read_i2c_block_data(self.i2c_addr, reg_addr, 3)
            raw = (data[1] << 8) | data[0]
            pec = data[2]
            return raw, pec

        # Mock simulation
        raw = self._mock_ambient_raw if reg_addr == MLX90614_RAM_TA else self._mock_object_raw
        # Synthesize valid PEC
        slave_write = (self.i2c_addr << 1) & 0xFE
        slave_read = (self.i2c_addr << 1) | 0x01
        data_low = raw & 0xFF
        data_high = (raw >> 8) & 0xFF
        tx_bytes = bytes([slave_write, reg_addr, slave_read, data_low, data_high])
        pec = calculate_smbus_pec(tx_bytes)

        return raw, pec

    def read_ambient_temperature(self, verify_checksum: bool = True) -> Dict[str, Any]:
        """Read and validate ambient temperature (Ta, 0x06)."""
        raw, pec = self.read_register(MLX90614_RAM_TA)
        if verify_checksum:
            slave_write = (self.i2c_addr << 1) & 0xFE
            slave_read = (self.i2c_addr << 1) | 0x01
            data_low = raw & 0xFF
            data_high = (raw >> 8) & 0xFF
            tx_bytes = bytes([slave_write, MLX90614_RAM_TA, slave_read, data_low, data_high])
            if not verify_smbus_pec(tx_bytes, pec):
                return {
                    "valid": False,
                    "temp_c": None,
                    "status": "PEC_CHECKSUM_ERROR",
                    "detail": "SMBus CRC-8 PEC mismatch."
                }
        return raw_to_celsius_mlx90614(raw, is_ambient=True)

    def read_object_temperature(self, verify_checksum: bool = True) -> Dict[str, Any]:
        """Read and validate object infrared temperature (To, 0x07)."""
        raw, pec = self.read_register(MLX90614_RAM_TOBJ1)
        if verify_checksum:
            slave_write = (self.i2c_addr << 1) & 0xFE
            slave_read = (self.i2c_addr << 1) | 0x01
            data_low = raw & 0xFF
            data_high = (raw >> 8) & 0xFF
            tx_bytes = bytes([slave_write, MLX90614_RAM_TOBJ1, slave_read, data_low, data_high])
            if not verify_smbus_pec(tx_bytes, pec):
                return {
                    "valid": False,
                    "temp_c": None,
                    "status": "PEC_CHECKSUM_ERROR",
                    "detail": "SMBus CRC-8 PEC mismatch."
                }
        return raw_to_celsius_mlx90614(raw, is_ambient=False)

    def cross_check_with_thermal_mast(self,
                                      mlx90640_canopy_temp: float,
                                      max_disagreement: float = MAX_MLX90614_DISAGREEMENT) -> Dict[str, Any]:
        """
        Perform continuous sanity cross-check against MLX90640 thermal mast array.
        Delegates to core.thermal.cross_check_mlx90614.
        """
        obj_res = self.read_object_temperature()
        if not obj_res["valid"]:
            return {
                "valid": False,
                "status": f"mlx90614_read_failure: {obj_res['status']}",
                "tc_mlx90640": float(mlx90640_canopy_temp),
                "mlx90614_temp": None,
                "delta_c": None
            }

        return cross_check_mlx90614(
            tc_mlx90640=mlx90640_canopy_temp,
            temp_mlx90614=obj_res["temp_c"],
            max_disagreement=max_disagreement
        )
