"""
Dual-bandpass NDVI computation, channel response cross-talk correction,
empirical line calibration, and dual-camera survey/inspection architecture.

Prompt 3 Section D (Committed Hardware Path):
  - Sensor: Waveshare IMX219-77IR (NoIR, 79.3 deg FOV, matched to RGB camera).
  - Filter: MidOpt DB660/850 dual-bandpass filter mounted on lens.
    Passbands: 660nm (visible red) and 850nm (near-infrared).

Counterintuitive Channel Mapping Derivation (D1):
  - In a standard Bayer CFA without an IR-cut filter (NoIR), silicon photodiodes
    are sensitive to near-infrared light across all pixels.
  - The MidOpt DB660/850 filter blocks all light EXCEPT ~660nm and ~850nm.
  - The Red Bayer micro-filter transmits 660nm (visible red) and leaks 850nm (NIR).
  - The Blue Bayer micro-filter blocks 660nm, but transmits 850nm (NIR).
  - The Green Bayer micro-filter receives negligible light in both windows and is unused.
  - Consequently:
      * 660nm (Red) is captured in the RED Bayer channel (with NIR leakage).
      * 850nm (NIR) is captured in the BLUE Bayer channel.
  - The normalized difference vegetation index formula is:
      NDVI = (NIR - Red) / (NIR + Red)
           = (Blue - Red) / (Blue + Red)
  - WARNING: Do NOT attempt to "correct" this channel mapping. In this dual-bandpass
    configuration, BLUE is NIR and RED is Visible Red.

Silicon Cross-Talk and Channel Leakage (D2):
  - Because organic Bayer dyes leak NIR, raw channel Digital Numbers (DN) represent:
      DN_Red  = k_RR * L_660 + k_RNIR * L_850
      DN_Blue = k_BR * L_660 + k_BNIR * L_850
  - True spectral radiances require an empirical cross-talk calibration matrix K^-1
    measured on the assembled camera.
  - Until bench calibration exists, apply_channel_response_correction() MUST raise
    NotImplementedError and ndvi_from_dual_bandpass() must fail loud by default.
    A silently incorrect NDVI is catastrophic for agronomic decision-making.

Empirical Line Calibration Across Flights (D3):
  - To make NDVI comparable across different flights and changing solar irradiance,
    reflectance panels (e.g. 5%, 50%, 84%) are imaged pre- and post-flight.
  - If panels are missing or unmeasured, NDVI is flagged within_flight_relative_only=True.
    Never silently assume radiometric calibration occurred.

Dual Camera Hardware Architecture - NEVER FUSED (D4):
  - CSI-0 (sensor_id=0): Standard IMX219-77 RGB for disease diagnosis, nutrient b*,
    and RGB indices during low-altitude inspection passes.
  - CSI-1 (sensor_id=1): Waveshare IMX219-77IR + DB660/850 for NDVI during broad
    high-altitude survey passes.
  - NEVER FUSED: The Jetson Nano CSI ports lack hardware sync/genlock for IMX219
    rolling-shutter sensors. Combining them at pixel level introduces severe rolling-shutter
    shear, temporal mismatch, and parallax errors in flight. Operating them on separate
    passes eliminates this entire failure domain.
"""
from typing import Mapping, Dict, Any, Tuple, Optional, Union, List
import numpy as np

# Hardware sensor port mapping on Jetson Nano dual CSI header
SENSOR_ID_RGB_INSPECTION = 0
SENSOR_ID_NIR_SURVEY = 1

# Hardware assembly status:
# Waveshare IMX219-77IR + MidOpt DB660/850 is a committed BOM component currently on import lead time.
# Physical camera assembly is tracked in the project Bill of Materials (BOM) rather than in code;
# the runtime data validity of NDVI computation is gated strictly by the presence of an empirical
# bench calibration matrix (calib_matrix is not None), which is the sole software gate.

# Provisional ground panel reflectances (recalled nominal values for commercial 3-panel targets, unverified).
# Actual physical panels procured must be measured or sourced from the vendor calibration certificate.
PROVISIONAL_PANEL_REFLECTANCES = (0.05, 0.50, 0.84)
PANEL_REFLECTANCES_CONFIRMED = False

BandMap = Mapping[str, np.ndarray]


def apply_channel_response_correction(raw_red: np.ndarray,
                                     raw_blue: np.ndarray,
                                     calib_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prompt 3 Section D2: Per-channel spectral response cross-talk correction.

    Without an IR-cut filter, raw Bayer channels suffer from NIR leakage.
    Unmixing requires an empirical 2x2 response calibration matrix K^-1
    measured on the assembled IMX219-77IR + DB660/850 camera on an optical bench:
      [L_660, L_850]^T = K^-1 * [DN_Red, DN_Blue]^T

    Parameters:
      raw_red: Raw Digital Numbers from Red Bayer channel.
      raw_blue: Raw Digital Numbers from Blue Bayer channel.
      calib_matrix: Optional 2x2 unmixing matrix (K^-1).

    WARNING (Calibration Validity):
      Passing an identity matrix disables cross-talk correction and produces NDVI
      from raw uncorrected Bayer DN, which is valid for unit testing and invalid
      for any agronomic output.

    Fails loud by default:
      If calib_matrix is None, raises NotImplementedError. Do NOT use a plausible-looking
      placeholder matrix. Run scripts/calibrate_dual_bandpass.py once the camera arrives.

    Returns:
      Tuple of (corrected_red_660nm, corrected_nir_850nm) as float32 arrays.
    """
    if calib_matrix is None:
        raise NotImplementedError(
            "Bench response calibration on assembled IMX219-77IR + DB660/850 camera has not "
            "been performed. Raw Bayer channels leak NIR across silicon photodiodes and cannot "
            "be used for valid NDVI without an empirical cross-talk calibration matrix. "
            "Run scripts/calibrate_dual_bandpass.py once the physical camera arrives."
        )

    k_inv = np.asarray(calib_matrix, dtype=np.float32)
    if k_inv.shape != (2, 2):
        raise ValueError(f"Cross-talk calibration matrix must be shaped (2, 2), got {k_inv.shape}.")

    r = np.asarray(raw_red, dtype=np.float32)
    b = np.asarray(raw_blue, dtype=np.float32)

    # Channel unmixing:
    # L_red = k_inv[0, 0] * r + k_inv[0, 1] * b
    # L_nir = k_inv[1, 0] * r + k_inv[1, 1] * b
    l_red = np.clip(k_inv[0, 0] * r + k_inv[0, 1] * b, 0.0, None)
    l_nir = np.clip(k_inv[1, 0] * r + k_inv[1, 1] * b, 0.0, None)

    return l_red, l_nir


def ndvi_from_dual_bandpass(image_or_bands: Union[np.ndarray, BandMap],
                            calib_matrix: Optional[np.ndarray] = None,
                            eps: float = 1e-6) -> np.ndarray:
    """
    Prompt 3 Section D1: NDVI computation from IMX219-77IR + MidOpt DB660/850 dual-bandpass filter.

    Channel Derivation:
      - 660nm visible red lands in the RED Bayer channel (plus NIR leakage).
      - 850nm NIR lands in the BLUE Bayer channel.
      - Green Bayer channel is unused.
      - NDVI = (NIR - Red) / (NIR + Red) = (Blue - Red) / (Blue + Red)

    Parameters:
      image_or_bands: Either a (H, W, 3) BGR array or a mapping with "red" and "blue" keys.
      calib_matrix: 2x2 response calibration matrix. Defaults to None, which causes
                    apply_channel_response_correction() to raise NotImplementedError.
      eps: Division-by-zero protection.

    WARNING (Calibration Validity):
      Passing an identity matrix disables cross-talk correction and produces NDVI
      from raw uncorrected Bayer DN, which is valid for unit testing and invalid
      for any agronomic output.

    Returns:
      2D float32 array of NDVI values clamped to [-1.0, 1.0].
    """
    if isinstance(image_or_bands, dict) or (hasattr(image_or_bands, "__getitem__") and "blue" in image_or_bands):
        raw_red = np.asarray(image_or_bands["red"], dtype=np.float32)
        raw_blue = np.asarray(image_or_bands["blue"], dtype=np.float32)
    else:
        arr = np.asarray(image_or_bands)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Image array must be (H, W, 3), got shape {arr.shape}.")
        # OpenCV standard BGR: index 0 is Blue, index 2 is Red
        raw_blue = arr[..., 0].astype(np.float32)
        raw_red = arr[..., 2].astype(np.float32)

    # D2: Mandatory spectral response correction (fails loud by default if calib_matrix is None)
    corr_red, corr_nir = apply_channel_response_correction(
        raw_red=raw_red,
        raw_blue=raw_blue,
        calib_matrix=calib_matrix
    )

    denom = corr_nir + corr_red
    safe_denom = np.where(np.abs(denom) < eps, eps, denom)

    ndvi = (corr_nir - corr_red) / safe_denom
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def apply_empirical_line_calibration(raw_index_or_dn: np.ndarray,
                                     panel_observations: Optional[Union[List[float], np.ndarray]] = None,
                                     panel_reflectances: Optional[Union[List[float], np.ndarray]] = None,
                                     allow_provisional: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prompt 3 Section D3: Empirical Line Method (ELM) reflectance calibration.

    Fits a linear relationship (Reflectance = m * DN + c) using standard ground panels
    imaged pre- and post-flight.
    Enables radiometric comparability ACROSS flights under differing solar irradiance.

    Parameters:
      raw_index_or_dn: 2D or 1D array of uncalibrated DN or NDVI values.
      panel_observations: Observed sensor Digital Numbers or raw values over reference panels.
      panel_reflectances: Explicit, verified ground-truth fractional reflectance values
                          sourced from the physical panel datasheet (e.g. from vendor
                          calibration certificate). Required whenever panel_observations is provided.
      allow_provisional: Set to True to allow fallback to PROVISIONAL_PANEL_REFLECTANCES
                         for unit testing only. Raises RuntimeError if False and panel_reflectances is unconfirmed.

    Fail-Safe Behavior (D3 & P1):
      - If panel observations are missing:
        Returns raw data untouched with metadata:
          radiometrically_calibrated=False
          within_flight_relative_only=True
          reason="Panel observations missing or insufficient (< 2 targets); NDVI is within-flight-relative only."
        Never silently assumes calibration happened.
      - If panel_reflectances is unprovided (None) and PANEL_REFLECTANCES_CONFIRMED is False:
        Raises RuntimeError unless allow_provisional=True.
      - If valid panel data is provided:
        Fits slope (gain) and intercept (bias) via linear regression.
        Applies radiometric calibration and returns calibrated array with quality metrics (R^2).
    """
    arr = np.asarray(raw_index_or_dn, dtype=np.float32)

    # Fail-safe check for missing panel observations
    if panel_observations is None:
        meta = {
            "radiometrically_calibrated": False,
            "within_flight_relative_only": True,
            "reason": "Calibration panel observations missing; NDVI is within-flight-relative only",
            "gain_m": None,
            "offset_c": None,
            "r_squared": None,
            "n_panels": 0,
            "provisional_reflectances_used": False
        }
        return arr.copy(), meta

    # Guard unconfirmed panel reflectances (P1):
    if panel_reflectances is None:
        if not PANEL_REFLECTANCES_CONFIRMED and not allow_provisional:
            raise RuntimeError(
                "Ground panel reflectances are unconfirmed (panel_reflectances is None). "
                "Supply verified panel reflectances from the physical panel datasheet, "
                "or pass allow_provisional=True to use PROVISIONAL_PANEL_REFLECTANCES for testing only."
            )
        panel_reflectances = PROVISIONAL_PANEL_REFLECTANCES
        provisional_used = True
    else:
        provisional_used = False

    obs = np.asarray(panel_observations, dtype=np.float32).ravel()
    ref = np.asarray(panel_reflectances, dtype=np.float32).ravel()

    if len(obs) != len(ref):
        raise ValueError(f"Panel count mismatch: {len(obs)} observations vs {len(ref)} reflectances.")

    if len(obs) < 2:
        meta = {
            "radiometrically_calibrated": False,
            "within_flight_relative_only": True,
            "reason": f"Insufficient calibration panels ({len(obs)} < 2); cannot fit empirical line",
            "gain_m": None,
            "offset_c": None,
            "r_squared": None,
            "n_panels": len(obs)
        }
        return arr.copy(), meta

    # Check for zero variance in observations
    if np.allclose(obs, obs[0]):
        meta = {
            "radiometrically_calibrated": False,
            "within_flight_relative_only": True,
            "reason": "Panel observations have zero variance; cannot fit empirical line",
            "gain_m": None,
            "offset_c": None,
            "r_squared": None,
            "n_panels": len(obs)
        }
        return arr.copy(), meta

    # Linear regression: Ref = m * Obs + c
    poly = np.polyfit(obs, ref, deg=1)
    gain_m = float(poly[0])
    offset_c = float(poly[1])

    # Compute coefficient of determination R^2
    pred_ref = gain_m * obs + offset_c
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    ss_res = float(np.sum((ref - pred_ref) ** 2))
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else 1.0

    calibrated = gain_m * arr + offset_c

    meta = {
        "radiometrically_calibrated": True,
        "within_flight_relative_only": False,
        "gain_m": gain_m,
        "offset_c": offset_c,
        "r_squared": r_squared,
        "n_panels": len(obs),
        "provisional_reflectances_used": provisional_used
    }
    return calibrated, meta
