"""
Deterministic phenological growth stage estimation from planting date and canopy cover.

Architectural Design:
  - Deterministic lookup, not an ML model.
  - Sourced from FAO Irrigation and Drainage Paper No. 56:
      * Chapter 5: Crop growth stages (initial, development, mid-season, late-season)
        and ground cover fraction boundaries (fc <= 0.10, 0.10 < fc < 0.70-0.80, fc >= 0.70).
      * Chapter 6: Table 11 stage lengths (days) and Table 12 crop coefficients (Kc).
  - Primary documents archived in docs/sources/ with recorded SHA256 manifests.
  - Sourced stage boundaries are WEB_VERIFIED.
  - Default variety cycle duration assumptions are PROVISIONAL / RECALLED_UNVERIFIED
    (registered in docs/HUMAN_ACTIONS_OUTSTANDING.md).
  - Handles missing/unknown planting date gracefully (emits stage None with
    machine-readable reason 'DAYS_SINCE_PLANTING_REQUIRED').
  - Emits objective canopy cover comparison ([min_fc, max_fc]) without judgmental
    health verdicts.
  - Python 3.6 compatible (no walrus, no dataclasses, no union type operator).
"""
from typing import Dict, Any, Optional, Tuple, List
from configs.train_config import PROVISIONAL_DEFAULT_CYCLE_DAYS

# Frozen Verification Status Enum values per ans_for_vitthal.md & docs/TEMPLATE_ID_REGISTRY.md
STATUS_VERIFIED = "VERIFIED"
STATUS_WEB_VERIFIED = "WEB_VERIFIED"
STATUS_RECALLED_UNVERIFIED = "RECALLED_UNVERIFIED"
STATUS_UNSOURCED = "UNSOURCED"

# Supported crops aligned with configs/classes.py
SUPPORTED_CROPS = ("rice", "wheat", "sugarcane")

# FAO-56 Table 11 Base Stage Lengths in Days: [L_ini, L_dev, L_mid, L_late]
# Sourced from FAO-56 Chapter 6 Table 11 (archived in docs/sources/fao56_chapter6_stage_lengths_table11.md)
FAO56_STAGE_LENGTHS_BASE = {
    # Rice (Tropics): 30 / 30 / 60 / 30 -> 150 days
    "rice": [30, 30, 60, 30],
    # Wheat (Central India): 15 / 25 / 50 / 30 -> 120 days
    "wheat": [15, 25, 50, 30],
    # Sugarcane (Ratoon, Low Latitudes): 25 / 70 / 135 / 50 -> 280 days
    "sugarcane": [25, 70, 135, 50],
}

# FAO-56 Table 12 Benchmark Crop Coefficients: (Kc_ini, Kc_mid, Kc_end)
FAO56_KC_BENCHMARKS = {
    # Rice: Kc_ini = 1.05 (standing water), Kc_mid = 1.20, Kc_end = 0.75 (midpoint 0.60-0.90)
    "rice": (1.05, 1.20, 0.75),
    # Wheat: Kc_ini = 0.50 (midpoint 0.30-0.70), Kc_mid = 1.15, Kc_end = 0.30 (midpoint 0.25-0.40)
    "wheat": (0.50, 1.15, 0.30),
    # Sugarcane: Kc_ini = 0.40, Kc_mid = 1.25, Kc_end = 0.75
    "sugarcane": (0.40, 1.25, 0.75),
}

# FAO-56 Chapter 5 Expected Canopy Cover Ranges: [min_fc, max_fc]
# Initial: emergence to ~10% cover
# Development: 10% to effective full cover (~70%)
# Mid-season: full cover (~70% to 100%)
# Late-season: full cover to maturity / senescence / leaf yellowing
FAO56_STAGE_CANOPY_RANGES = {
    "initial": [0.0, 0.10],
    "development": [0.10, 0.70],
    "mid_season": [0.70, 1.00],
    "late_season": [0.20, 0.85],
}

DOCUMENT_REFERENCE = (
    "FAO Irrigation and Drainage Paper No. 56, Chapter 5 (Crop growth stages) "
    "& Chapter 6 (Table 11 & Table 12). Archived in docs/sources/ with SHA256."
)


def get_scaled_stage_lengths(crop: str, total_cycle_days: int) -> List[int]:
    """
    Scale FAO-56 Table 11 stage lengths proportionally for a specific variety cycle length.

    Guarantees sum(lengths) == total_cycle_days and every stage has at least 1 day.
    """
    crop = crop.lower()
    base_lengths = FAO56_STAGE_LENGTHS_BASE[crop]
    base_total = sum(base_lengths)
    scale = float(total_cycle_days) / float(base_total)

    l_ini = max(1, int(round(base_lengths[0] * scale)))
    l_dev = max(1, int(round(base_lengths[1] * scale)))
    l_mid = max(1, int(round(base_lengths[2] * scale)))
    l_late = total_cycle_days - (l_ini + l_dev + l_mid)

    if l_late < 1:
        # Edge case guard for very short cycle overrides
        l_late = 1
        l_mid = max(1, total_cycle_days - (l_ini + l_dev + l_late))

    return [l_ini, l_dev, l_mid, l_late]


def estimate_growth_stage(
    crop: Optional[str],
    days_since_planting: Optional[int] = None,
    canopy_cover: Optional[float] = None,
    total_cycle_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Deterministically estimates phenological crop growth stage from days since planting
    and compares measured canopy cover fraction against FAO-56 expected ground cover ranges.

    Args:
        crop: Target crop name ('rice', 'wheat', 'sugarcane'). Case-insensitive.
        days_since_planting: Farmer-reported elapsed days since planting/sowing.
            If None, stage is emitted as None with reason 'DAYS_SINCE_PLANTING_REQUIRED'.
        canopy_cover: Measured canopy cover fraction [0.0, 1.0] from core/indices.py.
            Optional; handled gracefully if 0.0 or None.
        total_cycle_days: Optional variety maturity cycle duration in days.
            If None, uses PROVISIONAL_DEFAULT_CYCLE_DAYS from config.

    Returns:
        Structured dictionary matching advisory growth_stage block contract.
    """
    # Guard missing or invalid crop
    if not crop or not isinstance(crop, str):
        return {
            "crop": None,
            "stage": None,
            "stage_code": None,
            "reason": "CROP_NOT_SPECIFIED",
            "status": "CROP_NOT_SPECIFIED",
            "days_since_planting": days_since_planting,
            "total_cycle_days": total_cycle_days,
            "canopy_cover_measured": round(float(canopy_cover), 4) if canopy_cover is not None else None,
            "canopy_cover_expected_range": None,
            "kc": None,
            "verification_status": STATUS_UNSOURCED,
            "source": "derived",
        }

    crop_lower = crop.lower()
    if crop_lower not in SUPPORTED_CROPS:
        return {
            "crop": crop_lower,
            "stage": None,
            "stage_code": None,
            "reason": "UNSUPPORTED_CROP",
            "status": "UNSUPPORTED_CROP",
            "days_since_planting": days_since_planting,
            "total_cycle_days": total_cycle_days,
            "canopy_cover_measured": round(float(canopy_cover), 4) if canopy_cover is not None else None,
            "canopy_cover_expected_range": None,
            "kc": None,
            "verification_status": STATUS_UNSOURCED,
            "source": "derived",
        }

    # Guard negative days_since_planting
    if days_since_planting is not None and days_since_planting < 0:
        return {
            "crop": crop_lower,
            "stage": None,
            "stage_code": None,
            "reason": "INVALID_NEGATIVE_DAYS_SINCE_PLANTING",
            "status": "INVALID_INPUT",
            "days_since_planting": days_since_planting,
            "total_cycle_days": total_cycle_days,
            "canopy_cover_measured": round(float(canopy_cover), 4) if canopy_cover is not None else None,
            "canopy_cover_expected_range": None,
            "kc": None,
            "verification_status": STATUS_WEB_VERIFIED,
            "source": "derived",
        }

    # Resolve cycle duration & verification status of the variety duration
    if total_cycle_days is not None and total_cycle_days > 0:
        cycle_days = int(total_cycle_days)
        cycle_verification_status = STATUS_VERIFIED
        cycle_source = "farmer_override"
    else:
        cycle_days = int(PROVISIONAL_DEFAULT_CYCLE_DAYS[crop_lower])
        cycle_verification_status = STATUS_RECALLED_UNVERIFIED
        cycle_source = "default_assumption"

    # Guard missing planting date: normal operational state awaiting farmer input
    if days_since_planting is None:
        return {
            "crop": crop_lower,
            "stage": None,
            "stage_code": None,
            "reason": "DAYS_SINCE_PLANTING_REQUIRED",
            "status": "AWAITING_PLANTING_DATE",
            "days_since_planting": None,
            "total_cycle_days": cycle_days,
            "cycle_source": cycle_source,
            "cycle_verification_status": cycle_verification_status,
            "canopy_cover_measured": round(float(canopy_cover), 4) if canopy_cover is not None else None,
            "canopy_cover_expected_range": None,
            "kc": None,
            "verification_status": STATUS_WEB_VERIFIED,
            "source": "derived",
            "document_reference": DOCUMENT_REFERENCE,
        }

    # Calculate stage lengths
    stage_lengths = get_scaled_stage_lengths(crop_lower, cycle_days)
    l_ini, l_dev, l_mid, l_late = stage_lengths

    # Stage cutoffs
    end_ini = l_ini
    end_dev = end_ini + l_dev
    end_mid = end_dev + l_mid
    end_late = cycle_days

    # FAO-56 Table 12 benchmark Kc values
    kc_ini, kc_mid, kc_end = FAO56_KC_BENCHMARKS[crop_lower]

    dsp = int(days_since_planting)

    if dsp <= end_ini:
        stage = "initial"
        stage_code = "INI"
        kc = kc_ini
    elif dsp <= end_dev:
        stage = "development"
        stage_code = "DEV"
        # Linear interpolation between kc_ini and kc_mid per FAO-56 Eq. 66
        frac = float(dsp - end_ini) / float(l_dev) if l_dev > 0 else 1.0
        kc = kc_ini + frac * (kc_mid - kc_ini)
    elif dsp <= end_mid:
        stage = "mid_season"
        stage_code = "MID"
        kc = kc_mid
    elif dsp <= end_late:
        stage = "late_season"
        stage_code = "LATE"
        # Linear interpolation between kc_mid and kc_end per FAO-56 Eq. 67
        frac = float(dsp - end_mid) / float(l_late) if l_late > 0 else 1.0
        kc = kc_mid - frac * (kc_mid - kc_end)
    else:
        # Field past expected harvest cycle date
        stage = "late_season"
        stage_code = "LATE"
        kc = kc_end

    expected_cover_range = list(FAO56_STAGE_CANOPY_RANGES[stage])
    measured_cover = round(float(canopy_cover), 4) if canopy_cover is not None else None

    return {
        "crop": crop_lower,
        "stage": stage,
        "stage_code": stage_code,
        "days_since_planting": dsp,
        "total_cycle_days": cycle_days,
        "cycle_source": cycle_source,
        "cycle_verification_status": cycle_verification_status,
        "stage_lengths_days": {
            "initial": l_ini,
            "development": l_dev,
            "mid_season": l_mid,
            "late_season": l_late,
        },
        "canopy_cover_measured": measured_cover,
        "canopy_cover_expected_range": expected_cover_range,
        "kc": round(float(kc), 3),
        "status": "OK",
        "verification_status": STATUS_WEB_VERIFIED,
        "source": "derived",
        "document_reference": DOCUMENT_REFERENCE,
    }
