"""
Edge camera capture pipeline, fixed white balance/exposure control,
per-frame metadata logging, and reference card colour calibration.

Prompt 3 Section E:
  - CIELAB b* nutrient index and visible vegetation indices assume stable colour.
  - Auto white balance (AWB) shifts colour temperature and b* between frames,
    silently corrupting nutrient estimation.
  - Auto exposure (AE) causes fluctuating brightness, invalidating unnormalized
    channel differences (such as GMR).

HARDWARE VALIDATION STATUS:
  WARNING: Exposure and gain settings (PROVISIONAL_EXPOSURE_NS = 10,000,000 ns / 10 ms,
  PROVISIONAL_GAIN = 1.0) HAVE NOT BEEN VALIDATED ON HARDWARE under field lighting conditions.
  In bright Indian midday sun (~80,000-100,000 lux), 10ms at gain 1.0 is expected to severely
  overexpose and saturate the Red Bayer channel. Saturation invalidates any index that
  divides by red (such as NDVI and VARI).
  Empirical field calibration against a diffuse 18% reflectance gray card under solar noon
  is mandatory to determine the non-saturating operational exposure.

Requirements:
  E1. Camera capture MUST use FIXED white balance and FIXED exposure.
      Builds explicit, configurable GStreamer pipeline for Jetson Nano nvarguscamerasrc.
      Logs actual capture parameters with every captured frame.
  E2. Reference card colour correction (correct_from_reference_card) accepting
      detected colour reference patches (e.g. 24-patch chart) and normalising
      the frame against ground truth reference reflectance.
  E3. Fail-safe flag: if no reference card is detected, pipeline proceeds but sets
      colour_uncalibrated=True. Never silently assumes calibration occurred.
"""
from typing import Dict, Any, Tuple, Optional, List, Union
import datetime
import numpy as np
import cv2

# Default sensor IDs on Jetson Nano dual CSI header (J13)
SENSOR_ID_RGB = 0       # Standard IMX219-77 (inspection pass: disease, b*, RGB indices)
SENSOR_ID_NIR = 1       # IMX219-77IR + MidOpt DB660/850 (survey pass: NDVI)

# White balance mode in nvarguscamerasrc:
# 0 = Off (manual white balance / fixed colour gains)
# 1 = Auto (DISABLED by design - shifts b* nutrient index)
WBMODE_OFF = 0

# Provisional camera exposure and gain defaults - UNTESTED defaults requiring field measurement
# against reference target under bright midday sunlight.
PROVISIONAL_EXPOSURE_NS = 10000000    # 10 milliseconds (10,000,000 ns) - UNTESTED default
PROVISIONAL_GAIN = 1.0                # 1.0x analog gain - UNTESTED default
PROVISIONAL_DIGITAL_GAIN = 1.0         # 1.0x ISP digital gain - UNTESTED default

# Saturation monitoring thresholds for 8-bit frames
PROVISIONAL_SATURATION_THRESHOLD_DN = 250   # DN >= 250 considered near-saturation clipping in 8-bit range
PROVISIONAL_SATURATION_ALERT_FRACTION = 0.01 # >1% saturated pixels triggers clipping alert


def build_gstreamer_pipeline(sensor_id: int = SENSOR_ID_RGB,
                             capture_width: int = 1920,
                             capture_height: int = 1080,
                             framerate: int = 30,
                             wbmode: int = WBMODE_OFF,
                             exposure_ns: Optional[int] = PROVISIONAL_EXPOSURE_NS,
                             exposure_time_ms: Optional[float] = None,
                             gain: Optional[float] = PROVISIONAL_GAIN,
                             isp_digital_gain: Optional[float] = PROVISIONAL_DIGITAL_GAIN,
                             aelock: bool = True,
                             awblock: bool = True) -> str:
    """
    Build a GStreamer pipeline string for Jetson Nano nvarguscamerasrc.

    Enforces fixed exposure and fixed white balance to guarantee radiometric
    and colorimetric repeatability across flight passes.

    Parameters:
      sensor_id: CSI camera port (0 for RGB inspection, 1 for NIR survey).
      capture_width: Raw sensor capture width (default 1920).
      capture_height: Raw sensor capture height (default 1080).
      framerate: Target capture framerate in fps (default 30).
      wbmode: White balance mode (0 = Off / manual, 1 = Auto). Must be 0 for calibrated passes.
      exposure_ns: Fixed exposure time in nanoseconds (default PROVISIONAL_EXPOSURE_NS = 10ms).
      exposure_time_ms: Optional exposure time in ms (for backward compatibility).
      gain: Fixed analog gain multiplier (1.0 to 16.0).
      isp_digital_gain: Fixed ISP digital gain multiplier (1.0 to 256.0).
      aelock: True to lock auto exposure.
      awblock: True to lock auto white balance.

    Returns:
      GStreamer pipeline string suitable for cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER).
    """
    if sensor_id not in (0, 1):
        raise ValueError(f"sensor_id must be 0 or 1 on Jetson Nano dual CSI port, got {sensor_id}.")
    if capture_width <= 0 or capture_height <= 0:
        raise ValueError(f"Capture dimensions must be positive, got {capture_width}x{capture_height}.")
    if framerate <= 0:
        raise ValueError(f"Framerate must be positive, got {framerate}.")

    # Resolve exposure nanoseconds
    if exposure_time_ms is not None:
        if exposure_time_ms <= 0.0:
            raise ValueError(f"exposure_time_ms must be positive, got {exposure_time_ms}.")
        target_exp_ns = int(round(float(exposure_time_ms) * 1e6))
    elif exposure_ns is not None:
        if exposure_ns <= 0:
            raise ValueError(f"exposure_ns must be positive, got {exposure_ns}.")
        target_exp_ns = int(exposure_ns)
    else:
        target_exp_ns = PROVISIONAL_EXPOSURE_NS

    elements = [f"nvarguscamerasrc sensor-id={int(sensor_id)}"]

    # White balance mode: 0 = Off
    elements.append(f"wbmode={int(wbmode)}")

    # Lock flags
    elements.append(f"aelock={'true' if aelock else 'false'}")
    elements.append(f"awblock={'true' if awblock else 'false'}")

    # Fixed exposure time range: nvarguscamerasrc accepts nanoseconds
    elements.append(f'exposuretimerange="{target_exp_ns} {target_exp_ns}"')

    # Fixed gain range
    if gain is not None:
        if gain < 1.0:
            raise ValueError(f"gain must be >= 1.0, got {gain}.")
        gain_val = float(gain)
        elements.append(f'gainrange="{gain_val:.2f} {gain_val:.2f}"')

    # Fixed ISP digital gain range
    if isp_digital_gain is not None:
        if isp_digital_gain < 1.0:
            raise ValueError(f"isp_digital_gain must be >= 1.0, got {isp_digital_gain}.")
        isp_val = float(isp_digital_gain)
        elements.append(f'ispdigitalgainrange="{isp_val:.2f} {isp_val:.2f}"')

    # Camera caps
    elements.append(
        f"! video/x-raw(memory:NVMM), width=(int){int(capture_width)}, "
        f"height=(int){int(capture_height)}, format=(string)NV12, "
        f"framerate=(fraction){int(framerate)}/1"
    )

    # Conversion elements to BGR OpenCV format
    elements.append("! nvvidconv")
    elements.append("! video/x-raw, format=(string)BGRx")
    elements.append("! videoconvert")
    elements.append("! video/x-raw, format=(string)BGR")
    elements.append("! appsink")

    return " ".join(elements)


def create_frame_metadata(frame: Optional[np.ndarray] = None,
                          sensor_id: int = SENSOR_ID_RGB,
                          wbmode: int = WBMODE_OFF,
                          exposure_ns: int = PROVISIONAL_EXPOSURE_NS,
                          exposure_time_ms: Optional[float] = None,
                          gain: float = PROVISIONAL_GAIN,
                          isp_digital_gain: float = PROVISIONAL_DIGITAL_GAIN,
                          aelock: bool = True,
                          awblock: bool = True,
                          timestamp: Optional[str] = None,
                          saturation_threshold_dn: int = PROVISIONAL_SATURATION_THRESHOLD_DN) -> Dict[str, Any]:
    """
    Log actual capture parameters alongside each captured frame.
    Guarantees full traceability of illumination, sensor gain states, and channel saturation.

    Saturation Check:
      Evaluates the fraction of pixels at or near channel maximum (DN >= saturation_threshold_dn)
      per channel (Blue, Green, Red) for 3-channel uint8 frames.
      Emits saturation_status:
        - "SATURATION_OK": evaluated, all channels below alert fraction.
        - "SATURATION_DETECTED": evaluated, one or more channels at or above alert fraction.
        - "NOT_EVALUATED": evaluation did not run (e.g. frame is None, non-3D, non-3-channel,
          empty, or non-uint8 dtype).
      Emits saturation_status_reason: non-empty string whenever status is NOT_EVALUATED
      or SATURATION_DETECTED.
    """
    ts = timestamp if timestamp is not None else datetime.datetime.now().isoformat()

    if exposure_time_ms is not None:
        exp_ns = int(round(float(exposure_time_ms) * 1e6))
        exp_ms = float(exposure_time_ms)
    else:
        exp_ns = int(exposure_ns)
        exp_ms = float(exp_ns) / 1e6

    sat_b: Optional[float] = None
    sat_g: Optional[float] = None
    sat_r: Optional[float] = None
    warning_msg: Optional[str] = None

    if frame is None:
        saturation_status = "NOT_EVALUATED"
        saturation_status_reason = "Frame is None; saturation not evaluated"
    else:
        arr = np.asarray(frame)
        if arr.size == 0:
            saturation_status = "NOT_EVALUATED"
            saturation_status_reason = "Frame array is empty (size 0); saturation not evaluated"
        elif arr.ndim != 3:
            saturation_status = "NOT_EVALUATED"
            saturation_status_reason = f"Frame ndim {arr.ndim} != 3; expected 3-dimensional image"
        elif arr.shape[2] != 3:
            saturation_status = "NOT_EVALUATED"
            saturation_status_reason = f"Frame channels {arr.shape[2]} != 3; expected 3-channel image"
        elif arr.dtype != np.uint8:
            saturation_status = "NOT_EVALUATED"
            saturation_status_reason = f"Frame dtype {arr.dtype} is not uint8 (scaling unknown); saturation not evaluated"
        else:
            # OpenCV BGR: 0=Blue, 1=Green, 2=Red
            sat_b = float(np.mean(arr[..., 0] >= saturation_threshold_dn))
            sat_g = float(np.mean(arr[..., 1] >= saturation_threshold_dn))
            sat_r = float(np.mean(arr[..., 2] >= saturation_threshold_dn))
            if (sat_b >= PROVISIONAL_SATURATION_ALERT_FRACTION or
                sat_g >= PROVISIONAL_SATURATION_ALERT_FRACTION or
                sat_r >= PROVISIONAL_SATURATION_ALERT_FRACTION):
                saturation_status = "SATURATION_DETECTED"
                warning_msg = (
                    f"Channel saturation detected (>= {PROVISIONAL_SATURATION_ALERT_FRACTION:.1%} pixels >= {saturation_threshold_dn}): "
                    f"Blue={sat_b:.2%}, Green={sat_g:.2%}, Red={sat_r:.2%}. "
                    "Vegetation indices dividing by red (NDVI, VARI) may be corrupted."
                )
                saturation_status_reason = warning_msg
            else:
                saturation_status = "SATURATION_OK"
                saturation_status_reason = ""

    sat_info = {
        "blue_fraction": sat_b,
        "green_fraction": sat_g,
        "red_fraction": sat_r,
        "threshold_dn": int(saturation_threshold_dn),
        "status": saturation_status,
        "reason": saturation_status_reason,
        "warning": warning_msg
    }

    return {
        "timestamp": ts,
        "sensor_id": int(sensor_id),
        "wbmode": int(wbmode),
        "white_balance_fixed": bool(wbmode == WBMODE_OFF),
        "exposure_time_ms": exp_ms,
        "exposure_time_ns": exp_ns,
        "gain": float(gain),
        "isp_digital_gain": float(isp_digital_gain),
        "ae_locked": bool(aelock),
        "awb_locked": bool(awblock),
        "saturation": sat_info,
        "saturation_fractions": {
            "blue": sat_b,
            "green": sat_g,
            "red": sat_r
        },
        "saturation_status": saturation_status,
        "saturation_status_reason": saturation_status_reason,
        "colour_calibrated": False,
        "colour_uncalibrated": True
    }


def correct_from_reference_card(frame: np.ndarray,
                                detected_patches: Optional[np.ndarray] = None,
                                reference_patches: Optional[np.ndarray] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prompt 3 Section E2 & E3: Reference card colour normalisation.

    Accepts detected colour reference patch averages (e.g. from a 24-patch Macbeth chart)
    and maps them to standard reference reflectance via least-squares Colour Correction Matrix (CCM).

    Parameters:
      frame: BGR input image (H, W, 3) uint8.
      detected_patches: Optional (N, 3) array of detected patch RGB/BGR values.
      reference_patches: Optional (N, 3) array of known reference patch RGB/BGR values.
      metadata: Optional frame metadata dictionary to update.

    Returns:
      Tuple of (corrected_frame, updated_metadata).

    Fail-Safe Behavior (E3):
      - If detected_patches is None or empty:
        Returns frame untouched, with colour_uncalibrated=True and colour_calibrated=False.
        Never silently assumes calibration happened.
      - If detected_patches and reference_patches are valid:
        Fits least-squares 3x3 transformation matrix M, applies to frame, clamps to [0, 255] uint8.
        Sets colour_calibrated=True, colour_uncalibrated=False, and logs calibration RMSE.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a 3-channel 2D image (H, W, 3).")

    out_meta = dict(metadata) if metadata is not None else create_frame_metadata(frame=frame)

    # E3: Reference card not detected -> Fail-safe uncalibrated fallback
    if detected_patches is None or len(detected_patches) == 0 or reference_patches is None or len(reference_patches) == 0:
        out_meta["colour_calibrated"] = False
        out_meta["colour_uncalibrated"] = True
        out_meta["correction_applied"] = False
        out_meta["reason"] = "No reference card detected; frame colour uncalibrated"
        return frame.copy(), out_meta

    det = np.asarray(detected_patches, dtype=np.float32)
    ref = np.asarray(reference_patches, dtype=np.float32)

    if det.ndim != 2 or det.shape[1] != 3:
        raise ValueError(f"detected_patches must be shaped (N, 3), got {det.shape}.")
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"reference_patches must be shaped (N, 3), got {ref.shape}.")
    if len(det) != len(ref):
        raise ValueError(f"Length mismatch: {len(det)} detected patches vs {len(ref)} reference patches.")

    # Fit 3x3 Colour Correction Matrix M such that det @ M ~ ref
    # det: (N, 3), ref: (N, 3) -> M: (3, 3)
    try:
        M, residuals, rank, s = np.linalg.lstsq(det, ref, rcond=None)
    except Exception as e:
        out_meta["colour_calibrated"] = False
        out_meta["colour_uncalibrated"] = True
        out_meta["correction_applied"] = False
        out_meta["reason"] = f"Least-squares CCM solve failed: {str(e)}"
        return frame.copy(), out_meta

    # Apply 3x3 transformation to image: (H, W, 3) @ (3, 3) -> (H, W, 3)
    h, w, c = frame.shape
    frame_flat = frame.reshape(-1, 3).astype(np.float32)
    corrected_flat = frame_flat @ M
    corrected_frame = np.clip(corrected_flat, 0, 255).astype(np.uint8).reshape(h, w, c)

    # Compute root-mean-square error (RMSE) on calibration patches
    pred_ref = det @ M
    rmse = float(np.sqrt(np.mean((pred_ref - ref) ** 2)))

    out_meta["colour_calibrated"] = True
    out_meta["colour_uncalibrated"] = False
    out_meta["correction_applied"] = True
    out_meta["ccm_matrix"] = M.tolist()
    out_meta["calibration_rmse"] = rmse
    out_meta["n_patches"] = len(det)

    return corrected_frame, out_meta
