#!/usr/bin/env python3
"""
edge/hardware_detect.py -- Runtime Hardware Probing for MLX90640 & NoIR Camera (L4.1).

Detects connected sensors on Jetson Nano / edge platforms at runtime:
  1. detect_mlx90640: Probes I2C bus (default /dev/i2c-1 at address 0x33).
  2. detect_noir_camera: Probes CSI camera interfaces (IMX219 /dev/video* or nvargus).

Python 3.6 compatible.
"""

import os
from typing import Tuple


def detect_mlx90640(bus_num=1, address=0x33):
    # type: (int, int) -> Tuple[bool, str]
    """
    Probes I2C bus for the Melexis MLX90640 32x24 thermal array at address 0x33.
    """
    i2c_dev = "/dev/i2c-%d" % bus_num
    if not os.path.exists(i2c_dev):
        return False, "HARDWARE_NOT_CONNECTED (I2C device %s does not exist)" % i2c_dev

    try:
        import smbus2
    except ImportError:
        return False, "HARDWARE_NOT_CONNECTED (smbus2 library not installed)"

    try:
        with smbus2.SMBus(bus_num) as bus:
            bus.read_byte(address)
            return True, "I2C_DEVICE_DETECTED at %s:0x%02x" % (i2c_dev, address)
    except Exception as ex:
        return False, "HARDWARE_NOT_CONNECTED (MLX90640 probe failed at %s:0x%02x: %s)" % (i2c_dev, address, ex)


def detect_noir_camera(sensor_id=1):
    # type: (int) -> Tuple[bool, str]
    """
    Probes CSI camera interface for IMX219 NoIR camera.
    """
    v4l2_devs = ["/dev/video%d" % i for i in range(4) if os.path.exists("/dev/video%d" % i)]
    if not v4l2_devs:
        return False, "HARDWARE_NOT_CONNECTED (No V4L2 video devices found for CSI camera sensor_id=%d)" % sensor_id

    target_dev = "/dev/video%d" % sensor_id
    if os.path.exists(target_dev):
        return True, "CSI_CAMERA_DETECTED at %s (sensor_id=%d)" % (target_dev, sensor_id)

    return False, "HARDWARE_NOT_CONNECTED (Target CSI camera %s not found; available: %s)" % (target_dev, v4l2_devs)
