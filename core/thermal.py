"""
Canopy temperature extraction for CWSI, and the RGB vegetation mask.

v4 fixes (red-team round 3):
  * The upper gate no longer rejects severely stressed canopies.
    v3 used T_air + 7 degC, conflating the CWSI *upper baseline* (where
    CWSI = 1.0, the maximum stress the index measures) with the *physical
    limit of plant tissue*. Those are different quantities. A fully
    non-transpiring canopy under high radiation sits well above +7 degC,
    so v3 discarded exactly the drought it was deployed to detect.
    The gate now separates PLANT TISSUE from BARE SOIL, which is a much
    wider margin: soil at 55-65 degC with T_air = 38 is T_air + 17..27.
  * vegetation_mask uses an ABSOLUTE ExG threshold, not Otsu.
    Otsu assumes bimodality; on a 100% closed canopy the ExG histogram is
    unimodal and Otsu bisects it near the mean, marking half a pure green
    field as non-vegetation.
"""
import numpy as np
import cv2

# Soil/canopy separation, not health assessment. See module docstring.
MAX_ABOVE_AIR = 15.0     # above this, the cool population is not plant tissue
MAX_BELOW_AIR = 15.0     # below this, sky / open water / sensor fault
EXG_VEG_THRESHOLD = 20   # absolute, on the clipped 0..255 ExG scale


def excess_green(bgr):
    """ExG = 2G - R - B, clipped to [0, 255]. Negative values are non-green."""
    b, g, r = cv2.split(np.asarray(bgr, dtype=np.float32))
    return np.clip(2.0 * g - r - b, 0, 255).astype(np.uint8)


def vegetation_mask(bgr, thresh=EXG_VEG_THRESHOLD):
    """
    Absolute-threshold vegetation mask. Returns (mask_uint8, veg_fraction).

    Deliberately NOT Otsu. Otsu on a unimodal histogram splits near the mean,
    so a fully vegetated frame comes back ~50% vegetation and fails any
    downstream purity gate.
    """
    exg = excess_green(bgr)
    mask = (exg > thresh).astype(np.uint8) * 255
    return mask, float((mask > 0).mean())


def canopy_temperature(thermal, air_temp_c,
                       veg_fraction=None,
                       max_above_air=MAX_ABOVE_AIR,
                       max_below_air=MAX_BELOW_AIR,
                       min_frac=0.15, min_pixels=40, bimodal_gap=4.0):
    """
    Extract canopy temperature from a thermal array that may contain soil.

    thermal      : 2D array of degrees C
    air_temp_c   : ambient air temperature, needed for CWSI anyway
    veg_fraction : optional vegetation fraction from a co-registered RGB frame.
                   When supplied it is the PRIMARY discriminator; temperature
                   gates then only catch sensor faults.

    Returns (canopy_temp_c, canopy_fraction) on success,
            (None, reason_string) on rejection.
    """
    t = np.asarray(thermal, dtype=np.float32).ravel()
    t = t[np.isfinite(t)]
    if t.size < min_pixels:
        return None, 'too_few_pixels'

    lo, hi = float(t.min()), float(t.max())

    if hi - lo < bimodal_gap:
        # One population. Could be all canopy or all soil - temperature decides.
        cool = t
        frac = 1.0
    else:
        norm = ((t - lo) / (hi - lo) * 255.0).astype(np.uint8)
        thr, _ = cv2.threshold(norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sel = norm <= thr
        cool = t[sel]
        frac = float(sel.mean())
        if cool.size < 8:
            return None, 'cool_mode_too_small'

    if veg_fraction is not None and veg_fraction < min_frac:
        return None, 'canopy_fraction_too_low'
    if veg_fraction is None and frac < min_frac:
        return None, 'canopy_fraction_too_low'

    tc = float(np.median(cool))

    # Gate separates PLANT TISSUE from BARE SOIL, not healthy from stressed.
    # A fully non-transpiring canopy is a valid, important reading - it is
    # CWSI = 1.0, the drought alarm. It must never be discarded here.
    if tc > air_temp_c + max_above_air:
        return None, 'no_vegetation_bare_soil'
    if tc < air_temp_c - max_below_air:
        return None, 'implausibly_cold'

    return tc, frac


def cwsi(tc, ta, vpd_kpa, ll_slope, ll_intercept, ul_offset):
    """
    Empirical CWSI (Idso et al. 1981).
      (Tc-Ta)_LL = ll_slope * VPD + ll_intercept   (non-water-stressed baseline)
      (Tc-Ta)_UL = ul_offset                       (non-transpiring baseline)
    Baselines are CROP AND REGION SPECIFIC. Use published values and say so.
    """
    d = tc - ta
    ll = ll_slope * vpd_kpa + ll_intercept
    ul = ul_offset
    if ul - ll <= 0:
        return None
    return float(np.clip((d - ll) / (ul - ll), 0.0, 1.0))
