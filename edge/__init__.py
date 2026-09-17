"""
edge package
------------
Edge sensing, agronomic modeling, actuation, and LoRa telemetry for Jetson Nano / node gateways.
"""

# Active edge package initialisation.
# Note: Descoped hardware modules (edge.lora, edge.actuation, edge.flow, edge.adc)
# are not imported here to maintain clean import boundaries for the pod runtime.

from edge.thermal_point import (
    MLX90614Driver,
    raw_to_celsius_mlx90614,
    calculate_smbus_pec,
    verify_smbus_pec
)
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

__all__ = [
    "MLX90614Driver",
    "raw_to_celsius_mlx90614",
    "calculate_smbus_pec",
    "verify_smbus_pec",
    "build_gstreamer_pipeline",
    "create_frame_metadata",

    "correct_from_reference_card",
    "SENSOR_ID_RGB",
    "SENSOR_ID_NIR",
    "WBMODE_OFF",
    "PROVISIONAL_EXPOSURE_NS",
    "PROVISIONAL_GAIN",
    "PROVISIONAL_DIGITAL_GAIN",
    "PROVISIONAL_SATURATION_THRESHOLD_DN",
    "PROVISIONAL_SATURATION_ALERT_FRACTION",
]
