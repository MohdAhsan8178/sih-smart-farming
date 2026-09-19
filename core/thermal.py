"""
Canopy temperature extraction for CWSI, RGB vegetation mask, and fixed ground mast station.

Architectural Rescope (Ground Mast):
    Thermal sensing (MLX90640 array and MLX90614 single-point IR) is rescoped from
    the drone payload to a fixed ground mast facing the canopy.
    REASON: Rotor downwash at 1-2 m forces convection over the canopy, pulling
    sunlit leaf temperature toward air temperature. That is precisely the quantity
    CWSI measures, and the error biases stressed plants to look healthy. Flying
    the measurement destroys it.
    Fixed mount geometry (fixed distance, fixed field of view) eliminates altitude/GSD
    scaling variations.

Sensors and Roles:
    - MLX90640 (32x24 thermal array): DRIVES canopy temperature extraction and reported CWSI.
      Spatial resolution enables soil-pixel rejection via Otsu cool-mode thresholding and
      vegetation masking.
    - MLX90614 (single-point IR thermometer): CONTINUOUS CROSS-CHECK and drift trace ONLY.
      Lacks soil rejection capability and CANNOT report canopy temperature or CWSI.
      Disagreements with MLX90640 are logged as a data-quality / drift signal.

Non-Water-Stressed Baseline (NWSB):
    Empirical CWSI requires site-specific baselines (Tc - Ta vs VPD) collected near solar noon
    under well-watered conditions (Idso et al. 1981).
    A minimum of 14 clear-sky solar noon observations (MIN_BASELINE_OBSERVATIONS = 14)
    is required to fit the linear regression (2 parameters with adequate degrees of freedom
    across varying VPD). Until 14 observations exist, baseline fitting returns None with
    reason "baseline_insufficient" and CWSI is NOT computed.

v4 Biophysical Gates (Retained Unchanged):
    - MAX_ABOVE_AIR = 15.0 degC: Separates plant tissue from bare soil (which can reach 55-65 degC).
      Does NOT reject severely drought-stressed canopies (CWSI = 1.0).
    - PROVISIONAL_EXG_VEG_THRESHOLD = 20: Absolute ExG threshold (not Otsu) to prevent bisecting 100% green canopies.
    - Otsu cool-mode extraction on thermal pixels within the canopy mask.
"""
import numpy as np
import cv2

from core.indices import excess_green, vegetation_mask, PROVISIONAL_EXG_VEG_THRESHOLD

# Soil/canopy separation, not health assessment. See module docstring.
MAX_ABOVE_AIR = 15.0     # above this, the cool population is not plant tissue
MAX_BELOW_AIR = 15.0     # below this, sky / open water / sensor fault
MIN_BASELINE_OBSERVATIONS = 14  # minimum solar noon observations required to fit NWSB
MAX_MLX90614_DISAGREEMENT = 3.5 # degC threshold for cross-check / drift warning

# Provisional thermal segmentation constants - UNMEASURED defaults requiring field calibration
PROVISIONAL_BIMODAL_GAP_C = 4.0  # UNMEASURED default: minimum temperature span (hi - lo in deg C) to consider bimodality

# UNMEASURED default: minimum Otsu between-class variance ratio (eta = sigma_B^2 / sigma_T^2) to accept bimodal split.
# Empirical sweep envelope (80 parameter combinations across soil fraction, delta T, canopy spread):
#   - Unimodal baseline: Gaussian unimodal splits yield eta ~ 0.61-0.66 (mean ~0.64); uniform canopy yields eta ~ 0.73-0.77 (mean ~0.75);
#     skewed/heavy-tailed distributions yield eta <= 0.77. No unimodal canopy distribution exceeds 0.85.
#     * Thin margin on U-shaped distributions: U-shaped Beta(0.5, 0.5) reaches eta = 0.835, a margin of only 0.015 below 0.85.
#       Real canopy sun/shade at low solar elevation is genuinely U-shaped (sunlit leaf peak and shaded leaf peak), so
#       the guard's safety margin against a strongly bimodal pure canopy is thin.
#   - Boundary where genuine canopy+soil mixture falls below 0.85 and is kept whole:
#     Occurs when delta T <= 2 degC across all soil fractions (eta <= 0.78); or delta T = 5 degC when f_soil <= 0.05
#     or when canopy spread >= 4 degC (eta <= 0.79); or delta T = 10 degC when f_soil <= 0.02 (eta <= 0.84) or canopy spread >= 6 degC.
#   - Failure direction and sensitivity:
#     When kept whole despite soil presence, Tc is biased UPWARD by warm soil and CWSI is biased HIGH, producing a FALSE POSITIVE
#     drought alarm. This conservative trade-off guarantees that pure canopy is never falsely bisected (which would suppress drought alarms).
#     * Critical sensitivity boundary: The false-alarm risk at low soil fraction (f_soil <= 0.10) is bounded in practice because
#       np.median(cool) is robust to a small hot minority. The genuinely dangerous cells that cause large Tc upward bias are
#       HIGH soil fraction with LOW separation (f_soil >= 0.25, delta T <= 10 degC, canopy spread >= 6 degC), where soil pixels
#       heavily infiltrate the un-split population and shift the median, not low soil fraction.
PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO = 0.85


def canopy_temperature(thermal, air_temp_c,
                       veg_fraction=None,
                       max_above_air=MAX_ABOVE_AIR,
                       max_below_air=MAX_BELOW_AIR,
                       min_frac=0.15, min_pixels=40,
                       bimodal_gap=PROVISIONAL_BIMODAL_GAP_C,
                       min_interclass_ratio=PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO):
    """
    Extract canopy temperature from a thermal array that may contain soil.

    thermal      : 2D array of degrees C
    air_temp_c   : ambient air temperature, needed for CWSI anyway
    veg_fraction : Optional float representing scene vegetation fraction.
                   NOTE (Mast Architecture Rescope): Under the fixed ground mast
                   deployment, veg_fraction is PERMANENTLY None because the RGB camera
                   is on the aerial drone while the MLX90640 thermal array sits on a
                   static mast with no co-located RGB sensor or spatial co-registration.
                   Thermal canopy extraction in the mast path is strictly temperature-only.
                   When passed (e.g. in test rigs or future co-located payloads), it acts
                   only as an external whole-frame quality gate; spatial masking is not
                   performed here because t is flattened.

    Bimodal Split & Unimodality Guard (T1):
      When temperature span hi - lo >= bimodal_gap, Otsu thresholding is evaluated.
      CRITICAL DEFECT GUARD: A pure canopy with sunlit and shaded leaves routinely
      spans more than 4 C. Bisecting a pure canopy biases Tc downward toward the
      shaded leaf temperature, which biases CWSI low and suppresses drought alarms.
      To prevent this, an explicit unimodality test evaluates Otsu's inter-class
      variance ratio:
          eta = sigma_B^2 / sigma_T^2
      Otsu (1979) formulated eta as the normalized criterion measure of class separability
      in [0, 1]. For a unimodal Gaussian or uniform canopy distribution, eta <= 0.76.
      For a true bimodal mixture of cool canopy and hot sunlit soil, eta >= 0.90.
      If eta < min_interclass_ratio (default PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO = 0.85),
      the split is rejected as unimodal, and the entire array is retained as a single
      canopy population (cool = t, frac = 1.0).

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
        warm = t[~sel]

        # Explicit unimodality test (T1):
        # Prevent pure-canopy bisection where sun/shade leaf variance spans > bimodal_gap.
        # Criterion: Otsu's normalized inter-class variance ratio eta = sigma_B^2 / sigma_T^2.
        w0 = float(len(cool)) / float(t.size)
        w1 = float(len(warm)) / float(t.size)
        mu0 = float(cool.mean()) if cool.size > 0 else 0.0
        mu1 = float(warm.mean()) if warm.size > 0 else 0.0
        mu_total = float(t.mean())
        var_between = w0 * (mu0 - mu_total) ** 2 + w1 * (mu1 - mu_total) ** 2
        var_total = float(np.var(t))
        eta = (var_between / var_total) if var_total > 1e-6 else 0.0

        if cool.size < 8 or warm.size < 8 or eta < min_interclass_ratio:
            # Reject bimodal split as unimodal; treat whole array as one canopy population
            cool = t
            frac = 1.0
        else:
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


# Provisional reference CWSI constants (Jones 1999) - UNMEASURED defaults requiring field verification
PROVISIONAL_MIN_REF_GAP_C = 1.5  # degC minimum (Tdry - Twet) required to compute CWSI
PROVISIONAL_MAX_REF_STD_C = 1.5  # degC maximum standard deviation within wet/dry reference box


def calculate_cwsi_reference_based(tc, t_wet, t_dry, allow_unclamped=False):
    """
    Computes Crop Water Stress Index (CWSI) using physical wet and dry reference surfaces (L6.3, M2.3).

    Formula (Jones, 1999; Idso et al., 1981):
      CWSI = (Tc - T_wet) / (T_dry - T_wet)

    Constraints:
      1. T_wet and T_dry MUST be explicitly configured / measured.
      2. If missing or invalid, NEVER guess or use uncalibrated defaults.
      3. Returns (None, "WET_DRY_REFERENCES_NOT_CONFIGURED") on missing references.
      4. If allow_unclamped=True, returns (raw_cwsi, flag, "OK").
    """
    if t_wet is None or t_dry is None:
        return None, "WET_DRY_REFERENCES_NOT_CONFIGURED"

    try:
        t_wet_val = float(t_wet)
        t_dry_val = float(t_dry)
        tc_val = float(tc)
    except (ValueError, TypeError):
        return None, "INVALID_REFERENCE_TEMPERATURES"

    denom = t_dry_val - t_wet_val
    if denom < PROVISIONAL_MIN_REF_GAP_C:
        return None, "INSUFFICIENT_REFERENCE_TEMPERATURE_GAP (T_dry - T_wet < %.1fC)" % PROVISIONAL_MIN_REF_GAP_C

    cwsi_raw = (tc_val - t_wet_val) / denom
    if allow_unclamped:
        if cwsi_raw < 0.0:
            flag = "CWSI_BELOW_ZERO"
        elif cwsi_raw > 1.0:
            flag = "CWSI_ABOVE_ONE"
        else:
            flag = "NORMAL"
        return float(round(cwsi_raw, 4)), flag, "OK"

    cwsi_clamped = max(0.0, min(1.0, cwsi_raw))
    return float(round(cwsi_clamped, 4)), "OK"


def evaluate_reference_cwsi_from_frame(thermal_array, refs_config, air_temp_c=None):
    """
    Evaluates per-scan reference CWSI on a 24x32 thermal array using reference region config (M2.3).

    Excludes wet_ref and dry_ref boxes from canopy pixel pool.
    Twet and Tdry are medians of their respective pixel boxes.
    Evaluates PROVISIONAL_MIN_REF_GAP_C and PROVISIONAL_MAX_REF_STD_C.
    Returns dictionary conforming to M2.5 advisory thermal block schema.
    """
    arr = np.asarray(thermal_array, dtype=np.float32)
    if arr.shape != (24, 32):
        return {
            "available": False,
            "reason": "INVALID_FRAME_DIMENSIONS (expected 24x32, got %s)" % str(arr.shape),
            "tc_c": None,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    valid_pixels = arr[np.isfinite(arr)]
    if valid_pixels.size == 0:
        return {
            "available": False,
            "reason": "INVALID_THERMAL_FRAME (no finite pixels)",
            "tc_c": None,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    raw_tc = round(float(np.median(valid_pixels)), 2)

    if not isinstance(refs_config, dict) or refs_config.get("status") != "MEASURED":
        return {
            "available": False,
            "reason": "THERMAL_REFS_NOT_CONFIGURED",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    wet_box = refs_config.get("wet_ref")
    dry_box = refs_config.get("dry_ref")
    if not isinstance(wet_box, dict) or not isinstance(dry_box, dict):
        return {
            "available": False,
            "reason": "THERMAL_REFS_NOT_CONFIGURED",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    try:
        rw0, rw1 = int(wet_box["row_min"]), int(wet_box["row_max"])
        cw0, cw1 = int(wet_box["col_min"]), int(wet_box["col_max"])
        rd0, rd1 = int(dry_box["row_min"]), int(dry_box["row_max"])
        cd0, cd1 = int(dry_box["col_min"]), int(dry_box["col_max"])
    except (KeyError, ValueError, TypeError):
        return {
            "available": False,
            "reason": "INVALID_REFERENCE_BOX_COORDINATES",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    # Validate bounds
    if not (0 <= rw0 <= rw1 < 24 and 0 <= cw0 <= cw1 < 32):
        return {
            "available": False,
            "reason": "WET_REF_OUT_OF_BOUNDS",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }
    if not (0 <= rd0 <= rd1 < 24 and 0 <= cd0 <= cd1 < 32):
        return {
            "available": False,
            "reason": "DRY_REF_OUT_OF_BOUNDS",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    wet_pixels = arr[rw0 : rw1 + 1, cw0 : cw1 + 1]
    dry_pixels = arr[rd0 : rd1 + 1, cd0 : cd1 + 1]

    if wet_pixels.size == 0 or dry_pixels.size == 0:
        return {
            "available": False,
            "reason": "EMPTY_REFERENCE_REGION",
            "tc_c": raw_tc,
            "twet_c": None,
            "tdry_c": None,
            "cwsi": None,
            "flag": None,
        }

    t_wet = float(np.median(wet_pixels))
    t_dry = float(np.median(dry_pixels))
    std_wet = float(np.std(wet_pixels))
    std_dry = float(np.std(dry_pixels))

    # Canopy mask: exclude wet and dry reference boxes
    canopy_mask = np.ones((24, 32), dtype=bool)
    canopy_mask[rw0 : rw1 + 1, cw0 : cw1 + 1] = False
    canopy_mask[rd0 : rd1 + 1, cd0 : cd1 + 1] = False

    canopy_pixels = arr[canopy_mask]
    canopy_pixels = canopy_pixels[np.isfinite(canopy_pixels)]
    if canopy_pixels.size < 10:
        return {
            "available": False,
            "reason": "INSUFFICIENT_CANOPY_PIXELS",
            "tc_c": raw_tc,
            "twet_c": round(t_wet, 2),
            "tdry_c": round(t_dry, 2),
            "cwsi": None,
            "flag": None,
        }

    t_c = float(np.median(canopy_pixels))
    t_c_rounded = round(t_c, 2)

    # Spread checks
    if std_wet > PROVISIONAL_MAX_REF_STD_C:
        return {
            "available": False,
            "reason": "WET_REF_SPREAD_EXCEEDED (std=%.2fC > %.1fC)" % (std_wet, PROVISIONAL_MAX_REF_STD_C),
            "tc_c": t_c_rounded,
            "twet_c": round(t_wet, 2),
            "tdry_c": round(t_dry, 2),
            "cwsi": None,
            "flag": None,
        }
    if std_dry > PROVISIONAL_MAX_REF_STD_C:
        return {
            "available": False,
            "reason": "DRY_REF_SPREAD_EXCEEDED (std=%.2fC > %.1fC)" % (std_dry, PROVISIONAL_MAX_REF_STD_C),
            "tc_c": t_c_rounded,
            "twet_c": round(t_wet, 2),
            "tdry_c": round(t_dry, 2),
            "cwsi": None,
            "flag": None,
        }

    # Minimum gap check
    gap = t_dry - t_wet
    if gap < PROVISIONAL_MIN_REF_GAP_C:
        return {
            "available": False,
            "reason": "INSUFFICIENT_REFERENCE_GAP (Tdry - Twet = %.2fC < %.1fC)" % (gap, PROVISIONAL_MIN_REF_GAP_C),
            "tc_c": t_c_rounded,
            "twet_c": round(t_wet, 2),
            "tdry_c": round(t_dry, 2),
            "cwsi": None,
            "flag": None,
        }

    # Raw CWSI calculation (unclamped)
    cwsi_raw = (t_c - t_wet) / gap
    if cwsi_raw < 0.0:
        flag = "CWSI_BELOW_ZERO"
    elif cwsi_raw > 1.0:
        flag = "CWSI_ABOVE_ONE"
    else:
        flag = "NORMAL"

    return {
        "available": True,
        "reason": None,
        "tc_c": t_c_rounded,
        "twet_c": round(t_wet, 2),
        "tdry_c": round(t_dry, 2),
        "cwsi": round(float(cwsi_raw), 4),
        "flag": flag,
    }


def compute_vpd(air_temp_c, relative_humidity_pct):
    """
    Calculate Vapour Pressure Deficit (VPD in kPa) using the standard Tetens equation (FAO-56).

    e_s(Ta) = 0.61078 * exp(17.27 * Ta / (Ta + 237.3))  [saturation vapour pressure in kPa]
    e_a = e_s * (RH / 100.0)                            [actual vapour pressure in kPa]
    VPD = e_s - e_a
    """
    es = 0.61078 * np.exp((17.27 * air_temp_c) / (air_temp_c + 237.3))
    ea = es * (float(relative_humidity_pct) / 100.0)
    return float(max(0.0, es - ea))


def fit_non_water_stressed_baseline(observations, min_obs=MIN_BASELINE_OBSERVATIONS):
    """
    Fit empirical non-water-stressed lower baseline: (Tc - Ta) = slope * VPD + intercept.

    observations: list of items, each either:
                  - tuple/list: (delta_t, vpd_kpa) or (tc, ta, vpd_kpa)
                  - dict: {'delta_t': float, 'vpd_kpa': float} or {'tc': float, 'ta': float, 'vpd_kpa': float}
    min_obs     : minimum observation count required (default MIN_BASELINE_OBSERVATIONS=14).

    Rationale for min_obs = 14:
      A 2-parameter OLS regression requires sufficient degrees of freedom across a range
      of atmospheric vapor pressure deficits (typically 1.0 to 3.5 kPa). With fewer than 14
      solar-noon clear-sky observations, outlier days (e.g. passing clouds, intermittent gusts)
      heavily distort the slope and intercept, leading to false stress classifications.
      Standard agricultural literature (Idso et al. 1981, Gardner et al. 1992) mandates
      multi-day clear-sky replication.

    Returns:
      (params_dict, "ok") on success, where params_dict contains:
          ll_slope: float
          ll_intercept: float
          r_squared: float
          n_obs: int
      (None, "baseline_insufficient") if len(observations) < min_obs.
      (None, reason_str) on numerical invalidity.
    """
    if len(observations) < min_obs:
        return None, "baseline_insufficient"

    deltas = []
    vpds = []

    for obs in observations:
        if isinstance(obs, dict):
            if 'delta_t' in obs:
                dt = float(obs['delta_t'])
            elif 'tc' in obs and 'ta' in obs:
                dt = float(obs['tc']) - float(obs['ta'])
            else:
                continue
            vpd = float(obs.get('vpd_kpa', obs.get('vpd', 0.0)))
        elif isinstance(obs, (tuple, list)):
            if len(obs) == 2:
                dt, vpd = float(obs[0]), float(obs[1])
            elif len(obs) >= 3:
                dt, vpd = float(obs[0]) - float(obs[1]), float(obs[2])
            else:
                continue
        else:
            continue

        if np.isfinite(dt) and np.isfinite(vpd) and vpd >= 0.0:
            deltas.append(dt)
            vpds.append(vpd)

    if len(deltas) < min_obs:
        return None, "baseline_insufficient"

    vpds = np.asarray(vpds, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    vpd_var = float(np.var(vpds))
    if vpd_var < 0.05:
        return None, "vpd_range_too_narrow"

    # Ordinary Least Squares regression: deltas = slope * vpds + intercept
    slope, intercept = np.polyfit(vpds, deltas, 1)

    pred = slope * vpds + intercept
    ss_res = np.sum((deltas - pred) ** 2)
    ss_tot = np.sum((deltas - np.mean(deltas)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    return {
        "ll_slope": float(slope),
        "ll_intercept": float(intercept),
        "r_squared": float(r2),
        "n_obs": int(len(deltas)),
        "min_obs": int(min_obs),
        "status": "calibrated"
    }, "ok"


def cross_check_mlx90614(tc_mlx90640, temp_mlx90614, max_disagreement=MAX_MLX90614_DISAGREEMENT):
    """
    MLX90614 continuous cross-check against MLX90640 canopy temperature.

    MLX90614 is a single-point IR sensor with broad FOV (~35-90 deg) and NO soil rejection.
    It serves strictly as a continuous validation and drift monitor. It CANNOT drive CWSI.

    Returns:
      dict containing:
        valid: bool (True if disagreement <= max_disagreement)
        tc_mlx90640: float
        mlx90614_temp: float
        delta_c: float (tc_mlx90640 - temp_mlx90614)
        disagreement_c: float abs(delta_c)
        status: 'ok' or 'sensor_disagreement_or_drift'
    """
    delta = float(tc_mlx90640 - temp_mlx90614)
    disagreement = abs(delta)
    valid = disagreement <= max_disagreement
    return {
        "valid": valid,
        "tc_mlx90640": float(tc_mlx90640),
        "mlx90614_temp": float(temp_mlx90614),
        "delta_c": delta,
        "disagreement_c": disagreement,
        "status": "ok" if valid else "sensor_disagreement_or_drift"
    }


class ThermalMastStation:
    """
    Continuous ground-mast thermal monitoring station.

    Mounted at a fixed position overlooking the canopy (replacing drone flights
    to eliminate rotor downwash convective bias).

    Roles:
      - MLX90640 (32x24 array): Primary sensor. Uses spatial soil rejection to compute Tc and CWSI.
      - MLX90614 (single-point IR): Secondary validation sensor. Monitors drift/faults.
        CANNOT calculate CWSI independently.
    """
    def __init__(self, station_id="mast_01", crop_name="rice",
                 min_baseline_obs=MIN_BASELINE_OBSERVATIONS,
                 max_disagreement=MAX_MLX90614_DISAGREEMENT):
        self.station_id = station_id
        self.crop_name = crop_name
        self.min_baseline_obs = min_baseline_obs
        self.max_disagreement = max_disagreement

        self.baseline_observations = []
        self.baseline_params = None
        self.history = []

    def record_baseline_observation(self, delta_t, vpd_kpa, timestamp=None):
        """
        Record a solar noon clear-sky observation for empirical baseline calibration.
        """
        obs = {
            "delta_t": float(delta_t),
            "vpd_kpa": float(vpd_kpa),
            "timestamp": timestamp
        }
        self.baseline_observations.append(obs)
        return len(self.baseline_observations)

    def fit_baseline(self):
        """
        Fit baseline from accumulated observations.
        Returns (params_dict, status_str).
        """
        params, status = fit_non_water_stressed_baseline(
            self.baseline_observations, min_obs=self.min_baseline_obs
        )
        if status == "ok":
            self.baseline_params = params
        return params, status

    def process_reading(self, thermal_array, air_temp_c,
                        relative_humidity_pct=None, vpd_kpa=None,
                        mlx90614_temp=None, veg_fraction=None,
                        ul_offset=5.0, timestamp=None):
        """
        Process a mast reading.

        thermal_array: 2D array from MLX90640 (32x24). Single-point values rejected.
        air_temp_c   : Ambient air temperature in degrees C.
        """
        # Enforce that single-point readings cannot drive canopy extraction or CWSI
        if isinstance(thermal_array, (int, float)) or (isinstance(thermal_array, np.ndarray) and thermal_array.size <= 1):
            raise TypeError(
                "MLX90614 single-point reading cannot drive canopy temperature or CWSI. "
                "MLX90640 2D thermal array is required for soil-pixel rejection."
            )

        tc, frac = canopy_temperature(thermal_array, air_temp_c, veg_fraction=veg_fraction)
        if tc is None:
            record = {
                "timestamp": timestamp,
                "canopy_temp_c": None,
                "canopy_fraction": frac,
                "cwsi": None,
                "status": f"tc_rejected_{frac}",
                "cross_check": None
            }
            self.history.append(record)
            return record

        # Cross-check against MLX90614 if present
        cross_check = None
        if mlx90614_temp is not None:
            cross_check = cross_check_mlx90614(tc, mlx90614_temp, max_disagreement=self.max_disagreement)

        # Determine VPD
        if vpd_kpa is None and relative_humidity_pct is not None:
            vpd = compute_vpd(air_temp_c, relative_humidity_pct)
        elif vpd_kpa is not None:
            vpd = float(vpd_kpa)
        else:
            vpd = None

        # Compute CWSI only if baseline is calibrated and VPD available
        cwsi_val = None
        if self.baseline_params is None:
            cwsi_status = "baseline_insufficient"
        elif vpd is None:
            cwsi_status = "missing_vpd"
        else:
            cwsi_val = cwsi(
                tc=tc,
                ta=air_temp_c,
                vpd_kpa=vpd,
                ll_slope=self.baseline_params["ll_slope"],
                ll_intercept=self.baseline_params["ll_intercept"],
                ul_offset=ul_offset
            )
            cwsi_status = "ok"

        record = {
            "timestamp": timestamp,
            "canopy_temp_c": tc,
            "canopy_fraction": frac,
            "vpd_kpa": vpd,
            "cwsi": cwsi_val,
            "status": cwsi_status,
            "cross_check": cross_check
        }
        self.history.append(record)
        return record
