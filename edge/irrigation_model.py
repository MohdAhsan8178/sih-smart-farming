"""
edge/irrigation_model.py
------------------------
FAO-56 Evapotranspiration and Crop Water Requirement Model.

Implements reference evapotranspiration (ET0) using:
  1. FAO-56 Penman-Monteith combination equation (FAO-56 Eq. 6)
  2. FAO-56 Hargreaves-Samani temperature equation (FAO-56 Eq. 52)
as recommended by the Food and Agriculture Organization (FAO) of the
United Nations.

Crop coefficients (Kc) are drawn directly from FAO-56 Table 12 for the
project's target crops: Wetland Rice, Sugarcane, Spring Wheat, and Winter Wheat.

References:
  Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
  'Crop evapotranspiration - Guidelines for computing crop water requirements'.
  FAO Irrigation and drainage paper 56. Food and Agriculture Organization of
  the United Nations, Rome, Italy.

Citation Notice:
  All equation and table citations reference FAO-56 by equation/table number
  as defined in the official FAO document. Specific printed page numbers vary
  across publication formats/editions and are intentionally omitted to avoid
  unverified page references.

Constraints:
  - Pure algorithmic implementation: zero hardware dependencies.
"""

import math
from typing import Dict, Any, Optional, Tuple, List


# ==============================================================================
# FAO-56 Table 12 Published Crop Coefficients (Kc)
# Allen et al. (1998), FAO Irrigation and Drainage Paper 56, Table 12.
# ==============================================================================

FAO56_CROP_COEFFICIENTS: Dict[str, Dict[str, Any]] = {
    "rice": {
        # FAO-56 Table 12: Rice, wetland (paddy)
        "kc_ini": 1.05,
        "kc_mid": 1.20,
        "kc_end": 0.90,
        "citation": "FAO-56 Table 12 (Rice, wetland / paddy)"
    },
    "sugarcane": {
        # FAO-56 Table 12: Sugar cane
        "kc_ini": 0.40,
        "kc_mid": 1.25,
        "kc_end": 0.75,
        "citation": "FAO-56 Table 12 (Sugar cane)"
    },
    "spring_wheat": {
        # FAO-56 Table 12: Cereals - Spring Wheat
        # Sown in bare soil: Kc_ini = 0.30 (from Cereals group header), Kc_mid = 1.15, Kc_end = 0.30 (0.25-0.40 range)
        "kc_ini": 0.30,
        "kc_mid": 1.15,
        "kc_end": 0.30,
        "citation": "FAO-56 Table 12 (Cereals: Spring Wheat)"
    },
    "winter_wheat_non_frozen": {
        # FAO-56 Table 12: Cereals - Winter Wheat with non-frozen soils
        # Established canopy resuming growth in non-freezing climate: Kc_ini = 0.70
        "kc_ini": 0.70,
        "kc_mid": 1.15,
        "kc_end": 0.30,
        "citation": "FAO-56 Table 12 (Cereals: Winter Wheat with non-frozen soils)"
    },
    "winter_wheat_frozen": {
        # FAO-56 Table 12: Cereals - Winter Wheat with frozen soils (dormancy / snow cover)
        "kc_ini": 0.40,
        "kc_mid": 1.15,
        "kc_end": 0.30,
        "citation": "FAO-56 Table 12 (Cereals: Winter Wheat with frozen soils)"
    },
    # Generic 'wheat' alias defaults to 'spring_wheat'
    # RATIONALE: In Indian agriculture (the operational context of SIH), wheat is grown as a Rabi crop
    # sown from seed in October/November into a prepared, bare seedbed (0% canopy cover).
    # FAO-56 Table 11 explicitly lists Central India wheat planting in November with a 15-day initial stage.
    # At sowing and early emergence, the surface is predominantly bare soil, matching the bare-soil initial
    # condition of spring wheat (Kc_ini = 0.30). Defaulting to winter_wheat_non_frozen (Kc_ini = 0.70) would
    # falsely assume 70% reference evapotranspiration on a bare pre-emergence field, causing 2.3x over-irrigation.
    # winter_wheat_non_frozen (0.70) is available as an explicit choice when monitoring an established canopy.
    "wheat": {
        "kc_ini": 0.30,
        "kc_mid": 1.15,
        "kc_end": 0.30,
        "citation": "FAO-56 Table 12 (Cereals: Spring Wheat; default for Indian Rabi wheat sown into bare seedbed)",
        "default_for": "spring_wheat"
    }
}

# Provisional irrigation system application efficiencies (fraction)
# Used when converting net irrigation requirement to gross applied volume.
# These values require site-specific distribution uniformity auditing.
PROVISIONAL_IRRIGATION_EFFICIENCIES: Dict[str, float] = {
    "drip": 0.90,       # High-efficiency localized micro-irrigation
    "sprinkler": 0.75,  # Moderate overhead sprinkler distribution
    "furrow": 0.60      # Surface furrow / basin flooding with percolation losses
}

# ==============================================================================
# Paddy Rice & AWD Provisional Constants and Hardware Flags
# ==============================================================================

# Hardware availability flag: In Tier 1, no surface ponding water level sensor exists
# (capacitive soil probe reads 100% permanently under ponding).
# Set to True only if a dedicated ultrasonic/hydrostatic water level sensor is installed.
WATER_LEVEL_SENSOR_PRESENT: bool = False

# Origin Note: The following percolation and saturation constants originate from
# secondary search recall (FAO Training Manual 3 and Paper 24 references), NOT from
# primary in-situ soil measurements on the target field. Site-specific infiltration
# testing (double-ring infiltrometer) is mandatory before agronomic field deployment.
PROVISIONAL_PADDY_SATURATION_MM: float = 200.0          # One-time soil soaking & puddling depth (mm)
PROVISIONAL_PADDY_LAND_PREP_PONDING_MM: float = 25.0    # One-time pre-transplanting water layer (mm)
PROVISIONAL_PADDY_PERCOLATION_MM_DAY: float = 5.0       # Daily percolation + seepage rate (mm/day)
PROVISIONAL_PADDY_PERCOLATION_BY_SOIL: Dict[str, float] = {
    "clay": 4.0,   # Heavy puddled clay with plow sole
    "loam": 6.0,   # Silty / clay loam
    "sand": 8.0    # Coarse / sandy loam
}

# Alternate Wetting and Drying (AWD) Constants
# Provenance Notice: The following figures originate from secondary search recall;
# primary sources are not verified on this system.
#
# Specific Deficit Note:
# IRRI's safe AWD standard is defined as a water table depth of -15 cm inside a
# perforated field water tube ("pani pipe"), NOT as a cumulative millimeter deficit.
# Converting a -15 cm perched water table drop into millimeters of surface deficit requires
# multiplying by soil drainable porosity / specific yield (mu): Delta D = mu * 150 mm.
# Specific yield has NOT been measured for the target field; PROVISIONAL_AWD_REIRRIGATION_DEFICIT_MM (40.0 mm)
# is our own engineering conversion under an assumed, unmeasured drainable porosity (~0.27).
PROVISIONAL_AWD_REIRRIGATION_DEFICIT_MM: float = 40.0   # Engineering conversion to mm under unmeasured porosity assumption
PROVISIONAL_AWD_REFILL_DEPTH_MM: float = 50.0           # Secondary search recall (IRRI 5cm re-flooding depth; primary source unverified)
PROVISIONAL_PADDY_RECOVERY_DAYS: float = 14.0           # Secondary search recall (seedling recovery window; primary source unverified)
PROVISIONAL_PADDY_TERMINAL_DRAINAGE_DAYS: float = 14.0  # Secondary search recall (pre-harvest cutoff; primary source unverified)


# ==============================================================================
# Pure Psychrometric & Vapour Pressure Equations (FAO-56 Chapter 3)
# ==============================================================================

def atmospheric_pressure_fao56(elevation_m: float) -> float:
    """
    Atmospheric pressure as a function of elevation above sea level.
    FAO-56 Eq. 7:
      P = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
    Returns pressure P in kPa.
    """
    if elevation_m < -500.0 or elevation_m > 9000.0:
        raise ValueError(f"Elevation {elevation_m}m is outside physical terrestrial bounds [-500, 9000]m.")
    base = (293.0 - 0.0065 * float(elevation_m)) / 293.0
    if base <= 0:
        raise ValueError(f"Elevation {elevation_m}m produces invalid atmospheric base.")
    return float(101.3 * (base ** 5.26))


def psychrometric_constant_fao56(pressure_kpa: float) -> float:
    """
    Psychrometric constant gamma as a function of atmospheric pressure.
    FAO-56 Eq. 8:
      gamma = 0.000665 * P
    Returns gamma in kPa / deg C.
    """
    if pressure_kpa <= 0.0:
        raise ValueError(f"Atmospheric pressure must be positive, got {pressure_kpa} kPa.")
    return float(0.000665 * float(pressure_kpa))


def saturation_vapour_pressure_fao56(temp_c: float) -> float:
    """
    Saturation vapour pressure at air temperature temp_c.
    FAO-56 Eq. 11, Tetens (1930):
      e0(T) = 0.6108 * exp((17.27 * T) / (T + 237.3))
    Returns e0(T) in kPa.
    """
    if temp_c < -50.0 or temp_c > 70.0:
        raise ValueError(f"Air temperature {temp_c}C is outside typical terrestrial bounds [-50, 70]C.")
    return float(0.6108 * math.exp((17.27 * float(temp_c)) / (float(temp_c) + 237.3)))


def slope_vapour_pressure_curve_fao56(temp_c: float) -> float:
    """
    Slope of the saturation vapour pressure curve (Delta) at air temperature temp_c.
    FAO-56 Eq. 13:
      Delta = (4098.0 * (0.6108 * exp((17.27 * T) / (T + 237.3)))) / ((T + 237.3) ** 2)
    Returns Delta in kPa / deg C.
    """
    e0 = saturation_vapour_pressure_fao56(temp_c)
    denom = (float(temp_c) + 237.3) ** 2
    return float((4098.0 * e0) / denom)


def actual_vapour_pressure_from_rh_fao56(temp_c: float, rh_pct: float) -> float:
    """
    Actual vapour pressure (ea) derived from relative humidity and air temperature.
    FAO-56 Eq. 17:
      ea = e0(T) * (RH / 100.0)
    Returns ea in kPa.
    """
    if rh_pct < 0.0 or rh_pct > 100.0:
        raise ValueError(f"Relative humidity must be in [0, 100]%, got {rh_pct}%.")
    e0 = saturation_vapour_pressure_fao56(temp_c)
    return float(e0 * (float(rh_pct) / 100.0))


# ==============================================================================
# Reference Evapotranspiration: Penman-Monteith (FAO-56 Eq. 6)
# ==============================================================================

def calculate_et0_penman_monteith(t_mean_c: float,
                                  rn_mj_m2_day: float,
                                  u2_m_s: float,
                                  rh_mean_pct: float,
                                  elevation_m: float = 0.0,
                                  g_mj_m2_day: float = 0.0,
                                  t_max_c: Optional[float] = None,
                                  t_min_c: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate daily Reference Evapotranspiration (ET0) using the standardized
    FAO-56 Penman-Monteith equation (FAO-56 Eq. 6).

    Equation:
      ET0 = (0.408 * Delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)) /
            (Delta + gamma * (1 + 0.34 * u2))

    Parameters:
      t_mean_c: Mean daily air temperature at 2m height [deg C].
      rn_mj_m2_day: Net radiation at crop surface [MJ / m^2 / day].
      u2_m_s: Wind speed at 2m height [m / s].
      rh_mean_pct: Mean relative humidity [%].
      elevation_m: Elevation above sea level [m] (default 0.0m).
      g_mj_m2_day: Soil heat flux density [MJ / m^2 / day] (0.0 for daily timesteps per FAO-56 Box 6).
      t_max_c: Optional daily maximum temperature [deg C] (improves es estimation if present).
      t_min_c: Optional daily minimum temperature [deg C].

    Returns:
      Dict containing:
        et0_mm_day: float, reference ET in mm/day (clamped >= 0.0)
        rn_mj_m2_day: float
        g_mj_m2_day: float
        t_mean_c: float
        u2_m_s: float
        es_kpa: float, saturation vapour pressure
        ea_kpa: float, actual vapour pressure
        vpd_kpa: float, vapour pressure deficit (es - ea)
        delta_kpa_c: float, slope of vapour pressure curve
        gamma_kpa_c: float, psychrometric constant
        elevation_m: float
        method: 'FAO-56 Penman-Monteith (Eq. 6)'
    """
    if u2_m_s < 0.0:
        raise ValueError(f"Wind speed u2 must be non-negative, got {u2_m_s} m/s.")
    if rh_mean_pct < 0.0 or rh_mean_pct > 100.0:
        raise ValueError(f"Relative humidity must be in [0, 100]%, got {rh_mean_pct}%.")

    # 1. Atmospheric pressure and psychrometric constant (FAO-56 Eqs. 7 & 8)
    p_kpa = atmospheric_pressure_fao56(elevation_m)
    gamma = psychrometric_constant_fao56(p_kpa)

    # 2. Slope of saturation vapour pressure curve (FAO-56 Eq. 13)
    delta = slope_vapour_pressure_curve_fao56(t_mean_c)

    # 3. Saturation vapour pressure (FAO-56 Eq. 11 or 12)
    if t_max_c is not None and t_min_c is not None:
        if t_max_c < t_min_c:
            raise ValueError(f"t_max ({t_max_c}C) cannot be less than t_min ({t_min_c}C).")
        e0_max = saturation_vapour_pressure_fao56(t_max_c)
        e0_min = saturation_vapour_pressure_fao56(t_min_c)
        es = (e0_max + e0_min) / 2.0  # FAO-56 Eq. 12
    else:
        es = saturation_vapour_pressure_fao56(t_mean_c)

    # 4. Actual vapour pressure (FAO-56 Eq. 17)
    ea = es * (float(rh_mean_pct) / 100.0)
    vpd = max(0.0, es - ea)

    # 5. Penman-Monteith combination equation (FAO-56 Eq. 6)
    rad_term = 0.408 * delta * (float(rn_mj_m2_day) - float(g_mj_m2_day))
    wind_term = gamma * (900.0 / (float(t_mean_c) + 273.0)) * float(u2_m_s) * vpd
    denominator = delta + gamma * (1.0 + 0.34 * float(u2_m_s))

    if denominator <= 0.0:
        raise RuntimeError("Denominator in Penman-Monteith equation must be strictly positive.")

    et0_raw = (rad_term + wind_term) / denominator
    et0_val = max(0.0, float(round(et0_raw, 3)))

    return {
        "et0_mm_day": et0_val,
        "rn_mj_m2_day": float(rn_mj_m2_day),
        "g_mj_m2_day": float(g_mj_m2_day),
        "t_mean_c": float(t_mean_c),
        "u2_m_s": float(u2_m_s),
        "es_kpa": float(round(es, 4)),
        "ea_kpa": float(round(ea, 4)),
        "vpd_kpa": float(round(vpd, 4)),
        "delta_kpa_c": float(round(delta, 4)),
        "gamma_kpa_c": float(round(gamma, 4)),
        "elevation_m": float(elevation_m),
        "method": "FAO-56 Penman-Monteith (Eq. 6)"
    }


# ==============================================================================
# Reference Evapotranspiration: Hargreaves-Samani (FAO-56 Eq. 52)
# ==============================================================================

def calculate_et0_hargreaves(t_min_c: float,
                             t_max_c: float,
                             ra_mj_m2_day: float,
                             t_mean_c: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate Reference Evapotranspiration (ET0) using the Hargreaves-Samani
    temperature method (FAO-56 Eq. 52).

    Used as an operational fallback when solar radiation, wind speed, or humidity
    sensors are missing or offline on the edge node.

    Equation:
      ET0 = 0.0023 * (T_mean + 17.8) * (T_max - T_min) ** 0.5 * Ra
    Where Ra is extraterrestrial radiation expressed in mm/day.
    If Ra is supplied in MJ / m^2 / day, it is multiplied by 0.408
    (1 MJ / m^2 / day = 0.408 mm / day equivalent water evaporation).

    Parameters:
      t_min_c: Daily minimum temperature [deg C].
      t_max_c: Daily maximum temperature [deg C].
      ra_mj_m2_day: Extraterrestrial radiation [MJ / m^2 / day].
      t_mean_c: Optional mean temperature [deg C]; defaults to (t_max + t_min) / 2.

    Returns:
      Dict containing:
        et0_mm_day: float
        t_min_c: float
        t_max_c: float
        t_mean_c: float
        ra_mj_m2_day: float
        ra_mm_day: float
        method: 'FAO-56 Hargreaves-Samani (Eq. 52)'
    """
    if t_max_c < t_min_c:
        raise ValueError(f"t_max ({t_max_c}C) cannot be less than t_min ({t_min_c}C).")
    if ra_mj_m2_day < 0.0:
        raise ValueError(f"Extraterrestrial radiation Ra must be non-negative, got {ra_mj_m2_day} MJ/m^2/day.")

    if t_mean_c is None:
        t_mean = (float(t_max_c) + float(t_min_c)) / 2.0
    else:
        t_mean = float(t_mean_c)

    delta_t = max(0.0, float(t_max_c) - float(t_min_c))
    # Convert Ra from MJ/m^2/day to equivalent water evaporation in mm/day:
    # 0.408 factor (FAO-56 Chapter 3)
    ra_mm_day = 0.408 * float(ra_mj_m2_day)

    et0_raw = 0.0023 * (t_mean + 17.8) * (delta_t ** 0.5) * ra_mm_day
    et0_val = max(0.0, float(round(et0_raw, 3)))

    return {
        "et0_mm_day": et0_val,
        "t_min_c": float(t_min_c),
        "t_max_c": float(t_max_c),
        "t_mean_c": float(round(t_mean, 2)),
        "ra_mj_m2_day": float(ra_mj_m2_day),
        "ra_mm_day": float(round(ra_mm_day, 3)),
        "method": "FAO-56 Hargreaves-Samani (Eq. 52)"
    }


# ==============================================================================
# Crop Evapotranspiration (ETc) & Net Irrigation Sizing
# ==============================================================================

def get_crop_coefficient(crop: str, stage: str, stage_progress: float = 0.0) -> float:
    """
    Retrieve or interpolate crop coefficient (Kc) from FAO-56 Table 12.

    Stages:
      'initial': Returns Kc_ini.
      'development': Linearly interpolates Kc from Kc_ini to Kc_mid over stage_progress in [0.0, 1.0].
      'mid': Returns Kc_mid.
      'late' or 'end': Linearly interpolates Kc from Kc_mid to Kc_end over stage_progress in [0.0, 1.0].

    Parameters:
      crop: 'rice', 'sugarcane', 'wheat', 'spring_wheat', 'winter_wheat_non_frozen', or 'winter_wheat_frozen'.
      stage: Growth stage ('initial', 'development', 'mid', 'late', 'end').
      stage_progress: Fraction through stage [0.0, 1.0] for transition stages.

    Returns:
      Kc value (float).
    """
    norm_crop = str(crop).strip().lower()
    if norm_crop not in FAO56_CROP_COEFFICIENTS:
        valid_crops = list(FAO56_CROP_COEFFICIENTS.keys())
        raise ValueError(f"Crop '{crop}' not in FAO-56 Table 12 registry. Supported crops: {valid_crops}")

    params = FAO56_CROP_COEFFICIENTS[norm_crop]
    kc_ini = params["kc_ini"]
    kc_mid = params["kc_mid"]
    kc_end = params["kc_end"]

    norm_stage = str(stage).strip().lower()
    prog = max(0.0, min(1.0, float(stage_progress)))

    if norm_stage == "initial":
        return float(kc_ini)
    elif norm_stage == "development":
        # Linear interpolation from initial to mid-season
        return float(round(kc_ini + prog * (kc_mid - kc_ini), 3))
    elif norm_stage in ("mid", "mid_season"):
        return float(kc_mid)
    elif norm_stage in ("late", "end", "late_season"):
        # Linear interpolation from mid-season to harvest end
        return float(round(kc_mid + prog * (kc_end - kc_mid), 3))
    else:
        raise ValueError(f"Unknown stage '{stage}'. Supported: 'initial', 'development', 'mid', 'late', 'end'.")


def calculate_crop_et(et0_mm_day: float, kc: float) -> float:
    """
    Calculate Crop Evapotranspiration under standard conditions (ETc).
    FAO-56 Eq. 56:
      ETc = Kc * ET0
    Returns ETc in mm/day.
    """
    if et0_mm_day < 0.0:
        raise ValueError(f"ET0 must be non-negative, got {et0_mm_day} mm/day.")
    if kc < 0.0:
        raise ValueError(f"Kc must be non-negative, got {kc}.")
    return float(round(float(et0_mm_day) * float(kc), 3))


def calculate_net_irrigation(et_c_mm_day: float,
                             effective_rain_mm_day: float = 0.0,
                             area_m2: float = 1.0,
                             irrigation_method: str = "drip",
                             custom_efficiency: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate net and gross irrigation requirements and equivalent volume in liters.

    Agronomic equations (for upland field crops):
      I_net (mm) = max(0.0, ETc - P_eff)
      I_gross (mm) = I_net / efficiency
      Volume (L) = I_gross (mm) * Area (m^2)
      (Since 1 mm of depth over 1 m^2 equals exactly 1.0 Liter).

    Parameters:
      et_c_mm_day: Crop evapotranspiration [mm / day].
      effective_rain_mm_day: Effective precipitation reaching root zone [mm / day].
      area_m2: Field or plot surface area in square meters.
      irrigation_method: 'drip', 'sprinkler', or 'furrow' (picks default efficiency).
      custom_efficiency: Optional override for application efficiency in (0.0, 1.0].

    Returns:
      Dict containing:
        et_c_mm: float
        effective_rain_mm: float
        net_irrigation_mm: float
        gross_irrigation_mm: float
        volume_liters: float
        area_m2: float
        irrigation_efficiency: float
        irrigation_method: str
    """
    if et_c_mm_day < 0.0:
        raise ValueError(f"ETc must be non-negative, got {et_c_mm_day}.")
    if effective_rain_mm_day < 0.0:
        raise ValueError(f"Effective rain must be non-negative, got {effective_rain_mm_day}.")
    if area_m2 <= 0.0:
        raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")

    if custom_efficiency is not None:
        eff = float(custom_efficiency)
    else:
        norm_method = str(irrigation_method).strip().lower()
        eff = PROVISIONAL_IRRIGATION_EFFICIENCIES.get(norm_method, 0.80)

    if eff <= 0.0 or eff > 1.0:
        raise ValueError(f"Irrigation efficiency must be in (0.0, 1.0], got {eff}.")

    net_mm = max(0.0, float(et_c_mm_day) - float(effective_rain_mm_day))
    gross_mm = net_mm / eff if net_mm > 0 else 0.0
    volume_liters = gross_mm * float(area_m2)

    return {
        "et_c_mm": float(round(et_c_mm_day, 3)),
        "effective_rain_mm": float(round(effective_rain_mm_day, 3)),
        "net_irrigation_mm": float(round(net_mm, 3)),
        "gross_irrigation_mm": float(round(gross_mm, 3)),
        "volume_liters": float(round(volume_liters, 2)),
        "area_m2": float(area_m2),
        "irrigation_efficiency": float(eff),
        "irrigation_method": irrigation_method
    }


# ==============================================================================
# Paddy Rice Phase-Specific Water Requirement & Alternate Wetting and Drying (AWD)
# ==============================================================================

def calculate_paddy_land_prep_requirement(saturation_depth_mm: float = PROVISIONAL_PADDY_SATURATION_MM,
                                         initial_ponding_mm: float = PROVISIONAL_PADDY_LAND_PREP_PONDING_MM,
                                         effective_rain_accum_mm: float = 0.0,
                                         area_m2: float = 1.0,
                                         irrigation_efficiency: float = 0.60) -> Dict[str, Any]:
    """
    Calculate the one-time bulk water requirement for Phase 1: Land Preparation & Puddling.
    This function is structurally isolated from the daily irrigation scheduler and must
    NEVER be called on a daily basis.

    Equation:
      Net Depth (mm) = max(0.0, SAT + WL_initial - P_effective_accum)
      Gross Depth (mm) = Net Depth / irrigation_efficiency
      Volume (L) = Gross Depth * area_m2
      (1 mm of depth over 1 m^2 equals exactly 1.0 Liter).

    Parameters:
      saturation_depth_mm: Water depth required to saturate dry soil and establish puddle [mm]
                           (default PROVISIONAL_PADDY_SATURATION_MM = 200.0 mm).
      initial_ponding_mm: Standing water depth required at transplanting [mm]
                          (default PROVISIONAL_PADDY_LAND_PREP_PONDING_MM = 25.0 mm).
      effective_rain_accum_mm: Cumulative effective rainfall received during land prep [mm].
      area_m2: Surface area of the paddy field [m^2].
      irrigation_efficiency: Application efficiency for basin / flood irrigation (default 0.60).

    Returns:
      Dict with phase, net/gross depths, advisory volume, and hardware boundary metadata.
    """
    if saturation_depth_mm < 0.0:
        raise ValueError(f"Saturation depth must be non-negative, got {saturation_depth_mm} mm.")
    if initial_ponding_mm < 0.0:
        raise ValueError(f"Initial ponding depth must be non-negative, got {initial_ponding_mm} mm.")
    if effective_rain_accum_mm < 0.0:
        raise ValueError(f"Effective rain must be non-negative, got {effective_rain_accum_mm} mm.")
    if area_m2 <= 0.0:
        raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")
    if irrigation_efficiency <= 0.0 or irrigation_efficiency > 1.0:
        raise ValueError(f"Irrigation efficiency must be in (0.0, 1.0], got {irrigation_efficiency}.")

    net_mm = max(0.0, float(saturation_depth_mm) + float(initial_ponding_mm) - float(effective_rain_accum_mm))
    gross_mm = net_mm / float(irrigation_efficiency) if net_mm > 0.0 else 0.0
    volume_liters = gross_mm * float(area_m2)

    return {
        "phase": "land_preparation",
        "saturation_depth_mm": float(round(saturation_depth_mm, 2)),
        "initial_ponding_mm": float(round(initial_ponding_mm, 2)),
        "effective_rain_accum_mm": float(round(effective_rain_accum_mm, 2)),
        "net_water_depth_mm": float(round(net_mm, 2)),
        "gross_water_depth_mm": float(round(gross_mm, 2)),
        "volume_liters": float(round(volume_liters, 2)),
        "area_m2": float(area_m2),
        "irrigation_efficiency": float(irrigation_efficiency),
        "control_mode": "open_loop",
        "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
        "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
    }


def calculate_paddy_daily_requirement(et_c_mm_day: float,
                                      percolation_mm_day: Optional[float] = None,
                                      effective_rain_mm_day: float = 0.0,
                                      soil_type: Optional[str] = None,
                                      area_m2: float = 1.0,
                                      irrigation_efficiency: float = 0.60,
                                      water_level_mm: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate daily water requirement for Phase 2: Ponded vegetative maintenance.

    Equation:
      I_net (mm/day) = max(0.0, ETc + PERC - P_eff)
      I_gross (mm/day) = I_net / irrigation_efficiency
      Volume (L/day) = I_gross * area_m2

    Hardware Boundary Guard:
      Standing water depth (Delta WL) CANNOT be measured on Tier 1 hardware because capacitive
      soil moisture probes read 100% saturation permanently under ponding and cannot distinguish
      between 5mm and 50mm of ponded water. Therefore, Delta WL is excluded from the daily rate.
      If a caller attempts to supply water_level_mm while WATER_LEVEL_SENSOR_PRESENT is False,
      this function fails loud by raising RuntimeError.

    Parameters:
      et_c_mm_day: Daily crop evapotranspiration [mm/day].
      percolation_mm_day: Optional daily percolation rate [mm/day]. If None, looked up from soil_type.
      effective_rain_mm_day: Effective precipitation received on that day [mm/day].
      soil_type: Optional soil texture ('clay', 'loam', 'sand') used to pick provisional percolation rate.
      area_m2: Surface area of the paddy field [m^2].
      irrigation_efficiency: Application efficiency for basin / flood irrigation (default 0.60).
      water_level_mm: Measured ponding depth [mm]. MUST be None on Tier 1 hardware.

    Returns:
      Dict with daily water budget, advisory volume, and hardware boundary metadata.
    """
    if water_level_mm is not None and not WATER_LEVEL_SENSOR_PRESENT:
        raise RuntimeError(
            "water_level_mm cannot be evaluated because WATER_LEVEL_SENSOR_PRESENT is False. "
            "Capacitive soil moisture probes sit in saturated soil and cannot distinguish "
            "5mm of ponded water from 50mm. Standing water depth cannot be measured on "
            "Tier 1 hardware without a dedicated water level sensor."
        )

    if et_c_mm_day < 0.0:
        raise ValueError(f"ETc must be non-negative, got {et_c_mm_day} mm/day.")
    if effective_rain_mm_day < 0.0:
        raise ValueError(f"Effective rain must be non-negative, got {effective_rain_mm_day} mm/day.")
    if area_m2 <= 0.0:
        raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")
    if irrigation_efficiency <= 0.0 or irrigation_efficiency > 1.0:
        raise ValueError(f"Irrigation efficiency must be in (0.0, 1.0], got {irrigation_efficiency}.")

    if percolation_mm_day is not None:
        perc = float(percolation_mm_day)
    elif soil_type is not None:
        norm_soil = str(soil_type).strip().lower()
        perc = PROVISIONAL_PADDY_PERCOLATION_BY_SOIL.get(norm_soil, PROVISIONAL_PADDY_PERCOLATION_MM_DAY)
    else:
        perc = PROVISIONAL_PADDY_PERCOLATION_MM_DAY

    if perc < 0.0:
        raise ValueError(f"Percolation rate must be non-negative, got {perc} mm/day.")

    net_mm = max(0.0, float(et_c_mm_day) + float(perc) - float(effective_rain_mm_day))
    gross_mm = net_mm / float(irrigation_efficiency) if net_mm > 0.0 else 0.0
    volume_liters = gross_mm * float(area_m2)

    return {
        "phase": "daily_maintenance",
        "et_c_mm": float(round(et_c_mm_day, 3)),
        "percolation_mm": float(round(perc, 3)),
        "effective_rain_mm": float(round(effective_rain_mm_day, 3)),
        "net_irrigation_mm": float(round(net_mm, 3)),
        "gross_irrigation_mm": float(round(gross_mm, 3)),
        "volume_liters": float(round(volume_liters, 2)),
        "area_m2": float(area_m2),
        "irrigation_efficiency": float(irrigation_efficiency),
        "soil_type": soil_type,
        "control_mode": "open_loop",
        "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
        "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
    }


def calculate_paddy_terminal_drainage(area_m2: float = 1.0,
                                     days_to_harvest: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate water requirement for Phase 3: Terminal Drainage prior to harvest.
    Irrigation is completely suppressed (net requirement = 0.0 mm) 10-14 days before harvest
    to accelerate uniform grain ripening, promote senescence, and dry the soil for combine
    harvesters or manual field trafficability.

    Parameters:
      area_m2: Surface area of the paddy field [m^2].
      days_to_harvest: Optional days remaining until harvest.

    Returns:
      Dict with net_irrigation_mm = 0.0, irrigation_suppressed = True, and advisory metadata.
    """
    if area_m2 <= 0.0:
        raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")
    if days_to_harvest is not None and days_to_harvest < 0.0:
        raise ValueError(f"Days to harvest must be non-negative, got {days_to_harvest}.")

    return {
        "phase": "terminal_drainage",
        "net_irrigation_mm": 0.0,
        "gross_irrigation_mm": 0.0,
        "volume_liters": 0.0,
        "area_m2": float(area_m2),
        "days_to_harvest": float(days_to_harvest) if days_to_harvest is not None else None,
        "irrigation_suppressed": True,
        "reason": "Pre-harvest terminal drainage active to promote uniform ripening and field trafficability.",
        "control_mode": "open_loop",
        "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
        "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
    }


def evaluate_paddy_awd_status(cumulative_deficit_mm: float,
                              et_c_mm_day: float,
                              effective_rain_mm_day: float = 0.0,
                              days_since_transplanting: Optional[float] = None,
                              stage: Optional[str] = None,
                              is_flowering: bool = False,
                              days_to_harvest: Optional[float] = None,
                              soil_type: Optional[str] = None,
                              percolation_mm_day: Optional[float] = None,
                              reirrigation_deficit_threshold_mm: float = PROVISIONAL_AWD_REIRRIGATION_DEFICIT_MM,
                              refill_depth_mm: float = PROVISIONAL_AWD_REFILL_DEPTH_MM,
                              area_m2: float = 1.0,
                              irrigation_efficiency: float = 0.60) -> Dict[str, Any]:
    """
    Evaluate Alternate Wetting and Drying (AWD) status and daily water recommendation.

    Hard Safety Overrides (Mandatory Continuous Shallow Flooding):
      1. Seedling recovery: days_since_transplanting <= PROVISIONAL_PADDY_RECOVERY_DAYS (14 days).
         AWD is strictly suspended to allow root establishment and prevent seedling desiccation.
      2. Reproductive phase (panicle initiation through milk stage):
         is_flowering is True or stage in:
           ('panicle_initiation', 'booting', 'heading', 'flowering', 'anthesis', 'milk')
         AWD is strictly suspended because moisture stress during booting (microsporogenesis)
         and heading/anthesis causes irreversible spikelet sterility (chaffy grain) and severe
         yield collapse.
         IMPORTANT: Macro FAO-56 stage strings 'mid' or 'mid_season' are intentionally EXCLUDED.
         In FAO-56, 'mid' spans from full ground cover through tillering, heading, and grain fill (~50%
         of total season). Tripping on 'mid' would disable AWD for the entire mid-season.
      3. Unconfirmed crop growth stage fail-safe:
         When stage is None and is_flowering is False (and beyond seedling recovery window),
         AWD is WITHHELD (status: STAGE_UNKNOWN_AWD_WITHHELD, awd_suspended: True).
         AWD dry-down runs ONLY when the caller has affirmatively supplied a non-reproductive stage.
         Silence is not consent. Shallow flooding is maintained pending farmer confirmation.

    Terminal Drainage Cutoff:
      days_to_harvest <= PROVISIONAL_PADDY_TERMINAL_DRAINAGE_DAYS (14 days).
      All irrigation is suppressed.

    Active AWD Cycle (Vegetative Tillering):
      When safety overrides are inactive, cumulative water deficit tracks daily net loss:
        daily_loss = ETc + PERC - P_eff
        cumulative_deficit += daily_loss
      If cumulative_deficit >= reirrigation_deficit_threshold_mm (40 mm):
        Re-irrigation is triggered with advisory refill depth (50 mm).
        Cumulative deficit resets to 0.0.
      Else:
        Field continues drying down. Advisory irrigation is 0.0 mm.

    Parameters:
      cumulative_deficit_mm: Current accumulated water deficit [mm].
      et_c_mm_day: Daily crop evapotranspiration [mm/day].
      effective_rain_mm_day: Daily effective rainfall [mm/day].
      days_since_transplanting: Optional days elapsed since transplanting.
      stage: Optional phenological stage ('initial', 'development', 'mid', 'flowering', 'heading', 'anthesis', 'milk', 'late', 'end').
      is_flowering: Boolean flag indicating active flowering/heading.
      days_to_harvest: Optional days remaining until harvest.
      soil_type: Soil texture ('clay', 'loam', 'sand').
      percolation_mm_day: Optional percolation override [mm/day].
      reirrigation_deficit_threshold_mm: Threshold deficit triggering re-flooding [mm].
      refill_depth_mm: Depth of water layer applied upon re-irrigation [mm].
      area_m2: Field surface area [m^2].
      irrigation_efficiency: Application efficiency for basin flooding (default 0.60).

    Returns:
      Dict with AWD status, suspension flags, net/gross requirements, advisory volume, and metadata.
    """
    if cumulative_deficit_mm < 0.0:
        raise ValueError(f"Cumulative deficit must be non-negative, got {cumulative_deficit_mm} mm.")
    if et_c_mm_day < 0.0:
        raise ValueError(f"ETc must be non-negative, got {et_c_mm_day} mm/day.")
    if effective_rain_mm_day < 0.0:
        raise ValueError(f"Effective rain must be non-negative, got {effective_rain_mm_day} mm/day.")
    if area_m2 <= 0.0:
        raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")
    if irrigation_efficiency <= 0.0 or irrigation_efficiency > 1.0:
        raise ValueError(f"Irrigation efficiency must be in (0.0, 1.0], got {irrigation_efficiency}.")
    if reirrigation_deficit_threshold_mm <= 0.0:
        raise ValueError(f"Re-irrigation deficit threshold must be strictly positive, got {reirrigation_deficit_threshold_mm} mm.")
    if refill_depth_mm <= 0.0:
        raise ValueError(f"Refill depth must be strictly positive, got {refill_depth_mm} mm.")

    norm_stage = str(stage).strip().lower() if stage is not None else ""

    # 1. Check Terminal Drainage Cutoff
    if days_to_harvest is not None and days_to_harvest <= PROVISIONAL_PADDY_TERMINAL_DRAINAGE_DAYS:
        term_res = calculate_paddy_terminal_drainage(area_m2=area_m2, days_to_harvest=days_to_harvest)
        return {
            "phase": "terminal_drainage",
            "awd_status": "TERMINAL_DRAINAGE",
            "awd_active": False,
            "awd_suspended": True,
            "irrigation_suppressed": True,
            "suspension_reason": f"Pre-harvest terminal drainage active ({days_to_harvest:.1f} <= {PROVISIONAL_PADDY_TERMINAL_DRAINAGE_DAYS} days to harvest).",
            "reirrigation_triggered": False,
            "prior_deficit_mm": float(round(cumulative_deficit_mm, 3)),
            "daily_depletion_mm": 0.0,
            "new_cumulative_deficit_mm": 0.0,
            "deficit_threshold_mm": float(reirrigation_deficit_threshold_mm),
            "refill_depth_mm": 0.0,
            "net_irrigation_mm": 0.0,
            "gross_irrigation_mm": 0.0,
            "volume_liters": 0.0,
            "area_m2": float(area_m2),
            "irrigation_efficiency": float(irrigation_efficiency),
            "control_mode": "open_loop",
            "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
            "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
        }

    # 2. Hard Safety Override: Seedling Recovery Window (First 14 days post-transplanting)
    is_seedling_recovery = (
        days_since_transplanting is not None and
        days_since_transplanting <= PROVISIONAL_PADDY_RECOVERY_DAYS
    )
    if is_seedling_recovery:
        daily_res = calculate_paddy_daily_requirement(
            et_c_mm_day=et_c_mm_day,
            percolation_mm_day=percolation_mm_day,
            effective_rain_mm_day=effective_rain_mm_day,
            soil_type=soil_type,
            area_m2=area_m2,
            irrigation_efficiency=irrigation_efficiency
        )
        return {
            "phase": "daily_maintenance",
            "awd_status": "SUSPENDED_SHALLOW_FLOOD",
            "awd_active": False,
            "awd_suspended": True,
            "suspension_reason": (
                f"Seedling recovery active ({days_since_transplanting:.1f} <= {PROVISIONAL_PADDY_RECOVERY_DAYS} days post-transplanting): "
                "AWD suspended; continuous shallow flooding mandatory for root establishment."
            ),
            "reirrigation_triggered": bool(daily_res["net_irrigation_mm"] > 0.0),
            "prior_deficit_mm": float(round(cumulative_deficit_mm, 3)),
            "daily_depletion_mm": float(round(daily_res["net_irrigation_mm"], 3)),
            "new_cumulative_deficit_mm": 0.0,
            "deficit_threshold_mm": float(reirrigation_deficit_threshold_mm),
            "refill_depth_mm": 0.0,
            "net_irrigation_mm": daily_res["net_irrigation_mm"],
            "gross_irrigation_mm": daily_res["gross_irrigation_mm"],
            "volume_liters": daily_res["volume_liters"],
            "area_m2": float(area_m2),
            "irrigation_efficiency": float(irrigation_efficiency),
            "control_mode": "open_loop",
            "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
            "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
        }

    # 3. Hard Safety Override: Reproductive Stages (Panicle initiation through milk stage)
    # Published rice reproductive stage terms (IRRI Rice Knowledge Bank / BBCH scale for rice):
    #   - 'panicle_initiation' (IRRI Stage 4 / BBCH 30): panicle primordial development
    #   - 'booting' (IRRI Stage 5 / BBCH 41-49): flag leaf sheath swelling, microsporogenesis
    #   - 'heading' (IRRI Stage 6 / BBCH 51-59): panicle emergence from boot
    #   - 'anthesis' (IRRI Stage 7 / BBCH 61-69): flowering / pollination
    #   - 'milk' (IRRI Stage 8 / BBCH 71-77): grain filling / caryopsis milky ripe
    # Common alias:
    #   - 'flowering': common synonym for heading/anthesis
    #
    # EXCLUDED: 'mid' and 'mid_season' are FAO-56 macro growth stages and must NOT be used here.
    # In FAO-56, 'mid' spans from full ground cover through tillering, heading, and grain fill (~50%
    # of total season). Tripping on 'mid' would disable AWD for the entire mid-season.
    RICE_REPRODUCTIVE_STAGES = (
        "panicle_initiation",
        "booting",
        "heading",
        "flowering",
        "anthesis",
        "milk"
    )
    is_reproductive = (
        is_flowering or
        norm_stage in RICE_REPRODUCTIVE_STAGES
    )
    if is_reproductive:
        daily_res = calculate_paddy_daily_requirement(
            et_c_mm_day=et_c_mm_day,
            percolation_mm_day=percolation_mm_day,
            effective_rain_mm_day=effective_rain_mm_day,
            soil_type=soil_type,
            area_m2=area_m2,
            irrigation_efficiency=irrigation_efficiency
        )
        return {
            "phase": "daily_maintenance",
            "awd_status": "SUSPENDED_SHALLOW_FLOOD",
            "awd_active": False,
            "awd_suspended": True,
            "suspension_reason": (
                f"Reproductive stage active (stage='{stage}', is_flowering={is_flowering}): "
                "AWD suspended; continuous shallow flooding mandatory to prevent spikelet sterility."
            ),
            "reirrigation_triggered": bool(daily_res["net_irrigation_mm"] > 0.0),
            "prior_deficit_mm": float(round(cumulative_deficit_mm, 3)),
            "daily_depletion_mm": float(round(daily_res["net_irrigation_mm"], 3)),
            "new_cumulative_deficit_mm": 0.0,
            "deficit_threshold_mm": float(reirrigation_deficit_threshold_mm),
            "refill_depth_mm": 0.0,
            "net_irrigation_mm": daily_res["net_irrigation_mm"],
            "gross_irrigation_mm": daily_res["gross_irrigation_mm"],
            "volume_liters": daily_res["volume_liters"],
            "area_m2": float(area_m2),
            "irrigation_efficiency": float(irrigation_efficiency),
            "control_mode": "open_loop",
            "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
            "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
        }

    # 4. Fail-Safe Guard: Unconfirmed Crop Growth Stage
    # Silence is NOT consent. If stage is not affirmatively supplied (stage is None or empty)
    # and is_flowering is False, the advisor MUST NOT enter active dry-down.
    # Withholding AWD costs a modest amount of water (~15-30%); running dry-down through
    # an unconfirmed flowering period risks catastrophic spikelet sterility (100% yield loss).
    # Until an autonomous phenology model is integrated, AWD is strictly a farmer-confirmed
    # advisory feature.
    if not norm_stage:
        daily_res = calculate_paddy_daily_requirement(
            et_c_mm_day=et_c_mm_day,
            percolation_mm_day=percolation_mm_day,
            effective_rain_mm_day=effective_rain_mm_day,
            soil_type=soil_type,
            area_m2=area_m2,
            irrigation_efficiency=irrigation_efficiency
        )
        return {
            "phase": "daily_maintenance",
            "awd_status": "STAGE_UNKNOWN_AWD_WITHHELD",
            "awd_active": False,
            "awd_suspended": True,
            "suspension_reason": (
                "AWD dry-down requires an affirmatively confirmed, non-reproductive crop growth stage. "
                "Because crop stage is unconfirmed (stage=None, is_flowering=False), AWD is withheld "
                "to eliminate the risk of irreversible spikelet sterility. "
                "Continuous shallow flooding is maintained pending farmer confirmation."
            ),
            "reirrigation_triggered": bool(daily_res["net_irrigation_mm"] > 0.0),
            "prior_deficit_mm": float(round(cumulative_deficit_mm, 3)),
            "daily_depletion_mm": float(round(daily_res["net_irrigation_mm"], 3)),
            "new_cumulative_deficit_mm": 0.0,
            "deficit_threshold_mm": float(reirrigation_deficit_threshold_mm),
            "refill_depth_mm": 0.0,
            "net_irrigation_mm": daily_res["net_irrigation_mm"],
            "gross_irrigation_mm": daily_res["gross_irrigation_mm"],
            "volume_liters": daily_res["volume_liters"],
            "area_m2": float(area_m2),
            "irrigation_efficiency": float(irrigation_efficiency),
            "control_mode": "open_loop",
            "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
            "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
        }

    # 4. Active AWD Cycle
    if percolation_mm_day is not None:
        perc = float(percolation_mm_day)
    elif soil_type is not None:
        norm_soil = str(soil_type).strip().lower()
        perc = PROVISIONAL_PADDY_PERCOLATION_BY_SOIL.get(norm_soil, PROVISIONAL_PADDY_PERCOLATION_MM_DAY)
    else:
        perc = PROVISIONAL_PADDY_PERCOLATION_MM_DAY

    if perc < 0.0:
        raise ValueError(f"Percolation rate must be non-negative, got {perc} mm/day.")

    daily_depletion = float(et_c_mm_day) + float(perc) - float(effective_rain_mm_day)
    new_deficit_calc = max(0.0, float(cumulative_deficit_mm) + daily_depletion)

    if new_deficit_calc >= float(reirrigation_deficit_threshold_mm):
        # Trigger AWD Re-irrigation to +50mm refill depth
        reirrigation_triggered = True
        awd_status = "REIRRIGATE_TRIGGERED"
        net_mm = float(refill_depth_mm)
        gross_mm = net_mm / float(irrigation_efficiency)
        volume_liters = gross_mm * float(area_m2)
        final_deficit_mm = 0.0  # Reset deficit upon re-irrigation
    else:
        # Field continues drying down; no irrigation needed
        reirrigation_triggered = False
        awd_status = "DRYING_DOWN"
        net_mm = 0.0
        gross_mm = 0.0
        volume_liters = 0.0
        final_deficit_mm = new_deficit_calc

    return {
        "phase": "awd_cycle",
        "awd_status": awd_status,
        "awd_active": True,
        "awd_suspended": False,
        "suspension_reason": None,
        "reirrigation_triggered": reirrigation_triggered,
        "prior_deficit_mm": float(round(cumulative_deficit_mm, 3)),
        "daily_depletion_mm": float(round(daily_depletion, 3)),
        "new_cumulative_deficit_mm": float(round(final_deficit_mm, 3)),
        "deficit_threshold_mm": float(reirrigation_deficit_threshold_mm),
        "refill_depth_mm": float(refill_depth_mm) if reirrigation_triggered else 0.0,
        "net_irrigation_mm": float(round(net_mm, 3)),
        "gross_irrigation_mm": float(round(gross_mm, 3)),
        "volume_liters": float(round(volume_liters, 2)),
        "area_m2": float(area_m2),
        "irrigation_efficiency": float(irrigation_efficiency),
        "control_mode": "open_loop",
        "water_level_sensor_present": WATER_LEVEL_SENSOR_PRESENT,
        "volume_note": "ADVISORY ONLY. Physical delivery must be metered and verified via YF-S201 flow sensor."
    }


class AWDWaterBudgetAdvisor:
    """
    Stateful agronomic advisor for paddy rice Alternate Wetting and Drying (AWD).

    Maintains the cumulative water deficit over consecutive dry-down days, enforces
    hard safety overrides during seedling recovery and flowering/milk stages, and
    triggers refilling recommendations when cumulative deficit reaches the threshold.
    """

    def __init__(self,
                 soil_type: str = "clay",
                 reirrigation_deficit_threshold_mm: float = PROVISIONAL_AWD_REIRRIGATION_DEFICIT_MM,
                 refill_depth_mm: float = PROVISIONAL_AWD_REFILL_DEPTH_MM,
                 area_m2: float = 1.0,
                 irrigation_efficiency: float = 0.60):
        if area_m2 <= 0.0:
            raise ValueError(f"Field area must be strictly positive, got {area_m2} m^2.")
        if irrigation_efficiency <= 0.0 or irrigation_efficiency > 1.0:
            raise ValueError(f"Irrigation efficiency must be in (0.0, 1.0], got {irrigation_efficiency}.")
        if reirrigation_deficit_threshold_mm <= 0.0:
            raise ValueError(f"Re-irrigation deficit threshold must be positive, got {reirrigation_deficit_threshold_mm} mm.")
        if refill_depth_mm <= 0.0:
            raise ValueError(f"Refill depth must be positive, got {refill_depth_mm} mm.")

        self.soil_type = str(soil_type).strip().lower()
        self.reirrigation_deficit_threshold_mm = float(reirrigation_deficit_threshold_mm)
        self.refill_depth_mm = float(refill_depth_mm)
        self.area_m2 = float(area_m2)
        self.irrigation_efficiency = float(irrigation_efficiency)

        self.cumulative_deficit_mm: float = 0.0
        self.reirrigation_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_daily(self,
                       et_c_mm_day: float,
                       effective_rain_mm_day: float = 0.0,
                       days_since_transplanting: Optional[float] = None,
                       stage: Optional[str] = None,
                       is_flowering: bool = False,
                       days_to_harvest: Optional[float] = None,
                       percolation_mm_day: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate daily water requirement, update stateful cumulative deficit, and log cycle history.
        """
        result = evaluate_paddy_awd_status(
            cumulative_deficit_mm=self.cumulative_deficit_mm,
            et_c_mm_day=et_c_mm_day,
            effective_rain_mm_day=effective_rain_mm_day,
            days_since_transplanting=days_since_transplanting,
            stage=stage,
            is_flowering=is_flowering,
            days_to_harvest=days_to_harvest,
            soil_type=self.soil_type,
            percolation_mm_day=percolation_mm_day,
            reirrigation_deficit_threshold_mm=self.reirrigation_deficit_threshold_mm,
            refill_depth_mm=self.refill_depth_mm,
            area_m2=self.area_m2,
            irrigation_efficiency=self.irrigation_efficiency
        )

        self.cumulative_deficit_mm = float(result["new_cumulative_deficit_mm"])
        if result.get("reirrigation_triggered") and result.get("awd_active"):
            self.reirrigation_count += 1

        self.history.append(dict(result))
        return result

    def reset_deficit(self) -> None:
        """Manually reset the cumulative deficit to 0.0 (e.g. after manual flooding)."""
        self.cumulative_deficit_mm = 0.0
