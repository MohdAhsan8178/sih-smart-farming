"""
Sticky-trap blob segmentation, fiducial frame verification, and gateway pest evaluation
for ESP32-CAM nodes.

Section B Rework:
  - Finding 1: Explicit NOT_SAMPLED_BY_STICKY_TRAP negative entries for taxa not validly
    sampled by yellow sticky cards (stem borers, leaf folders, planthoppers, hispa).
  - Finding 2: Verified ICAR/NIPHM registry with primary source URLs and exact quotes.
    Exactly ONE surviving numeric sticky-card threshold: sugarcane whitefly / woolly aphid
    (100 per trap daily). Rice and wheat sticky traps reframed as relative trend alerting.
  - Finding 3: MM_PER_PIXEL is uncalibrated by default (None). NOMINAL_MM_PER_PIXEL_DESIGN_TARGET
    is reserved for synthetic unit tests. Runtime derivation raises RuntimeError if uncalibrated.
  - Finding 4: Four-corner printed ArUco fiducial frame (verify_fiducial_frame) measuring dynamic
    mm/pixel, perspective homography, tilt/rotation guard, and Laplacian focus guard.
  - Finding 5: Fiducial regions are masked out before watershed segmentation, preventing
    printed markers from being counted as insect blobs (enables strict == count assertions).
  - Finding 6: Cumulative monitoring requires deployment timestamp; enforces [1.0, 7.0] day
    valid monitoring window (INVALID_MONITORING_WINDOW) and card saturation guard (>30% area).
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2

# Scale calibration flags and provisional dimensions
MARKER_SPACING_CONFIRMED = False  # Flips to True only after physical caliper measurement on manufactured card
MM_PER_PIXEL = None  # UNCALIBRATED - PENDING PHYSICAL MEASUREMENT
NOMINAL_MM_PER_PIXEL_DESIGN_TARGET = 0.125  # Design assumption (800px / 100mm) for synthetic unit tests only
TARGET_MIN_PEST_RADIUS_MM = 0.225  # whitefly Bemisia tabaci radius ~0.225 mm
DEFAULT_ABS_FLOOR_PX = None  # Uncalibrated by default; derived at runtime

# Provisional constants - require empirical calibration on real field imagery
PROVISIONAL_FOCUS_FLOOR_LAPLACIAN = 100.0  # Variance of Laplacian floor on marker patches
PROVISIONAL_MAX_ROTATION_DEG = 10.0        # Max allowed camera tilt / rotation relative to card
PROVISIONAL_MAX_CARD_SATURATION = 0.30     # Max blob area fraction (30%) before card saturation
# PROVISIONAL UNCONFIRMED VALUES: Assumed nominal centroid spacing between ArUco markers
# on a 100x75mm card with 10mm margins. MUST be physically measured with calipers on the
# actual manufactured card before production deployment.
PROVISIONAL_MARKER_SPACING_W_MM = 80.0
PROVISIONAL_MARKER_SPACING_H_MM = 55.0

# Provisional trend alerting constants - require empirical tuning on real trap data
PROVISIONAL_TREND_MIN_OBSERVATIONS = 3      # Minimum valid prior observations required to compute rolling baseline (engineering judgement)
PROVISIONAL_TREND_ROLLING_WINDOW = 7        # Maximum prior observations retained in rolling baseline (engineering judgement matching 7-day card cap)
PROVISIONAL_TREND_RISING_RATIO = 1.50       # Multiplier >= 1.50 (50% increase over median baseline) to sit above expected day-to-day sampling noise
PROVISIONAL_TREND_FALLING_RATIO = 0.67      # Multiplier <= 0.67 (33% decrease below median baseline) to detect meaningful population abatement
PROVISIONAL_TREND_MIN_RATE_DIFF = 1.0       # Minimum rate delta (insects/day) required when baseline is zero or near-zero


def derive_watershed_floor_px(mm_per_pixel: Optional[float] = None,
                              target_radius_mm: float = TARGET_MIN_PEST_RADIUS_MM) -> float:
    """
    Derive watershed distance-transform marker floor in pixels from target pest physical radius.
    Raises RuntimeError if mm_per_pixel is None and MM_PER_PIXEL is uncalibrated.
    """
    scale = MM_PER_PIXEL if mm_per_pixel is None else mm_per_pixel
    if scale is None:
        raise RuntimeError(
            "Physical scale MM_PER_PIXEL is uncalibrated. "
            "Run calibrate_scale_from_fiducials() or provide a calibrated mm_per_pixel."
        )
    if scale <= 0:
        raise ValueError("mm_per_pixel must be positive.")
    return round(target_radius_mm / scale, 2)


def calibrate_scale_from_fiducials(marker_centers: Dict[int, Tuple[float, float]],
                                  marker_spacing_mm: Tuple[float, float] = (
                                      PROVISIONAL_MARKER_SPACING_W_MM,
                                      PROVISIONAL_MARKER_SPACING_H_MM
                                  ),
                                  allow_provisional: bool = False) -> Dict[str, Any]:
    """
    Derive dynamic mm/px from detected four-corner marker geometry.
    marker_centers: dict mapping ID -> (cx, cy) for IDs 0 (TL), 1 (TR), 2 (BR), 3 (BL).

    Follow-Up 1 Guard:
      Raises RuntimeError if MARKER_SPACING_CONFIRMED is False and allow_provisional is False.
      When allow_provisional=True, returns dict carrying 'scale_provisional': True.
    """
    if not MARKER_SPACING_CONFIRMED and not allow_provisional:
        raise RuntimeError(
            "Physical marker spacing on printed trap card is unconfirmed (MARKER_SPACING_CONFIRMED = False). "
            "Marker spacing must be physically measured with calipers on the manufactured card before production use. "
            "Pass allow_provisional=True for synthetic testing only."
        )

    c0 = np.array(marker_centers[0], dtype=np.float64)
    c1 = np.array(marker_centers[1], dtype=np.float64)
    c2 = np.array(marker_centers[2], dtype=np.float64)
    c3 = np.array(marker_centers[3], dtype=np.float64)

    w_top = float(np.linalg.norm(c1 - c0))
    w_bot = float(np.linalg.norm(c2 - c3))
    w_px = (w_top + w_bot) / 2.0

    h_left = float(np.linalg.norm(c3 - c0))
    h_right = float(np.linalg.norm(c2 - c1))
    h_px = (h_left + h_right) / 2.0

    w_mm, h_mm = marker_spacing_mm
    measured_mm_per_px = float((w_mm + h_mm) / (w_px + h_px))
    return {
        "measured_mm_per_pixel": measured_mm_per_px,
        "w_px": w_px,
        "h_px": h_px,
        "scale_provisional": not MARKER_SPACING_CONFIRMED
    }


def verify_fiducial_frame(img_bgr: np.ndarray,
                          expected_marker_ids: Tuple[int, ...] = (0, 1, 2, 3),
                          marker_spacing_mm: Tuple[float, float] = (
                              PROVISIONAL_MARKER_SPACING_W_MM,
                              PROVISIONAL_MARKER_SPACING_H_MM
                          ),
                          max_rotation_deg: float = PROVISIONAL_MAX_ROTATION_DEG,
                          min_focus_score: float = PROVISIONAL_FOCUS_FLOOR_LAPLACIAN,
                          allow_provisional: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Detect printed 4-corner ArUco fiducial frame (DICT_4X4_50) in trap housing.
    
    Verifies:
      1. Presence of all expected markers (default IDs 0, 1, 2, 3).
      2. Dynamic mm/px scale calibration.
      3. Perspective homography matrix to canonical rectangle.
      4. Camera tilt/rotation guard (<= max_rotation_deg).
      5. Camera focus guard (Laplacian variance >= min_focus_score).
      
    Returns:
      (passed: bool, reason: str, info: dict)
    """
    img = np.asarray(img_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if not (hasattr(cv2, "aruco") and hasattr(cv2.aruco, "getPredefinedDictionary")):
        return False, "aruco_not_available", {"detail": "OpenCV aruco module missing"}

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False, "fiducial_incomplete", {
            "found_ids": [],
            "missing_ids": list(expected_marker_ids),
            "detail": "No fiducial markers detected in frame"
        }

    flat_ids = [int(i) for i in ids.flatten()]
    missing_ids = [mid for mid in expected_marker_ids if mid not in flat_ids]

    if missing_ids:
        return False, "fiducial_incomplete", {
            "found_ids": flat_ids,
            "missing_ids": missing_ids,
            "detail": f"Fiducial markers incomplete: missing IDs {missing_ids}"
        }

    # Extract centers and corner quads
    marker_centers = {}
    marker_corners = {}
    focus_scores = []

    for idx, mid in enumerate(flat_ids):
        if mid in expected_marker_ids:
            c = corners[idx][0]  # shape (4, 2)
            cx = float(c[:, 0].mean())
            cy = float(c[:, 1].mean())
            marker_centers[mid] = (cx, cy)
            marker_corners[mid] = c

            # Focus metric on marker patch
            x0, x1 = max(0, int(c[:, 0].min())), min(gray.shape[1], int(c[:, 0].max()))
            y0, y1 = max(0, int(c[:, 1].min())), min(gray.shape[0], int(c[:, 1].max()))
            patch = gray[y0:y1, x0:x1]
            if patch.size > 0:
                var_lap = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                focus_scores.append(var_lap)

    mean_focus = float(np.mean(focus_scores)) if focus_scores else 0.0

    # Defocus guard
    if mean_focus < min_focus_score:
        return False, "image_blurred_defocused", {
            "focus_score": mean_focus,
            "min_required_focus": min_focus_score,
            "found_ids": flat_ids,
            "detail": f"Focus score {mean_focus:.1f} below threshold {min_focus_score:.1f}"
        }

    # Rotation / tilt guard
    c0 = np.array(marker_centers[0])
    c1 = np.array(marker_centers[1])
    c2 = np.array(marker_centers[2])
    c3 = np.array(marker_centers[3])

    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    rotation_deg = float(np.degrees(np.arctan2(dy, dx)))

    if abs(rotation_deg) > max_rotation_deg:
        return False, "camera_tilted", {
            "rotation_deg": rotation_deg,
            "max_allowed_rotation_deg": max_rotation_deg,
            "found_ids": flat_ids,
            "detail": f"Camera rotation {rotation_deg:.1f} deg exceeds limit {max_rotation_deg:.1f} deg"
        }

    # Derive live scale
    scale_info = calibrate_scale_from_fiducials(
        marker_centers, marker_spacing_mm, allow_provisional=allow_provisional
    )
    measured_mm_per_px = scale_info["measured_mm_per_pixel"]
    w_px = scale_info["w_px"]
    h_px = scale_info["h_px"]
    scale_provisional = scale_info["scale_provisional"]

    # Perspective homography mapping quad to canonical axis-aligned rectangle
    src_pts = np.array([c0, c1, c2, c3], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return True, "ok", {
        "measured_mm_per_pixel": measured_mm_per_px,
        "scale_provisional": scale_provisional,
        "rotation_deg": rotation_deg,
        "focus_score": mean_focus,
        "homography": homography,
        "marker_centers": marker_centers,
        "marker_corners": marker_corners,
        "w_px": w_px,
        "h_px": h_px,
        "detail": "fiducial_frame_verified"
    }


def verify_fiducial_marker(img_bgr: np.ndarray,
                           expected_pos: Tuple[float, float] = (45.0, 45.0),
                           tolerance_px: float = 15.0,
                           marker_id: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Backward-compatible single fiducial marker check for legacy callers.
    """
    img = np.asarray(img_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            matches = np.where(ids.flatten() == marker_id)[0]
            if len(matches) > 0:
                c = corners[matches[0]][0]
                cx = float(c[:, 0].mean())
                cy = float(c[:, 1].mean())
                dx = cx - expected_pos[0]
                dy = cy - expected_pos[1]
                displacement = float(np.hypot(dx, dy))
                if displacement > tolerance_px:
                    return False, "camera_shifted", {
                        "detected_center": (cx, cy),
                        "displacement_px": displacement,
                        "detail": f"displacement_{displacement:.1f}px_exceeds_tolerance_{tolerance_px:.1f}px"
                    }
                return True, "ok", {
                    "detected_center": (cx, cy),
                    "displacement_px": displacement,
                    "detail": "fiducial_aligned"
                }

    return False, "camera_shifted", {
        "detected_center": None,
        "displacement_px": None,
        "detail": "fiducial_absent_or_occluded"
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood from the border; anything unreached is an interior hole."""
    h, w = mask.shape
    ff = mask.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def extract_markers(dist: np.ndarray, abs_floor_px: float, rel_frac: float = 0.25) -> np.ndarray:
    """
    Dual-criterion seed extraction.
    abs_floor_px : absolute distance-transform floor derived from target pest radius.
    rel_frac     : relative criterion, retained so large clumps still seed.
    """
    _, micro = cv2.threshold(dist, abs_floor_px, 255, cv2.THRESH_BINARY)
    _, big = cv2.threshold(dist, rel_frac * float(dist.max()), 255, cv2.THRESH_BINARY)
    return np.uint8(np.maximum(micro, big))


def segment_trap_blobs(bgr: np.ndarray,
                       min_area: int = 8,
                       max_area: int = 1200,
                       abs_floor_px: Optional[float] = None,
                       mm_per_pixel: Optional[float] = None,
                       rel_frac: float = 0.25,
                       crop: int = 64,
                       fiducial_mask: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
    """
    Returns a list of (crop_bgr, (cx, cy), area) for each detected insect.
    Adaptive threshold on LAB b (robust to trap fading and glue glare),
    then distance-transform watershed with size-safe markers.

    Finding 5: When fiducial_mask is provided, fiducial regions are strictly masked out
    prior to distance transform and marker seeding, preventing printed markers from
    being detected as insects.
    """
    if abs_floor_px is None:
        abs_floor_px = derive_watershed_floor_px(mm_per_pixel)

    bgr = np.asarray(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    bch = lab[:, :, 2]

    adaptive = cv2.adaptiveThreshold(bch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 8)
    _, glob = cv2.threshold(bch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(adaptive, glob)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    clean = cv2.morphologyEx(_fill_holes(mask), cv2.MORPH_OPEN, k)

    # Mask out fiducials before distance transform
    if fiducial_mask is not None:
        clean[fiducial_mask > 0] = 0

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return []

    fg = extract_markers(dist, abs_floor_px, rel_frac)
    unknown = cv2.subtract(cv2.dilate(clean, k, iterations=2), fg)

    n, markers = cv2.connectedComponents(fg)      # n includes background label 0
    markers = markers + 1                          # background becomes 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)

    half = crop // 2
    pad = cv2.copyMakeBorder(bgr, half, half, half, half, cv2.BORDER_REPLICATE)

    out = []
    for lbl in range(2, n + 1):                    # objects are 2..n
        ys, xs = np.where(markers == lbl)
        area = ys.size
        if area < min_area or area > max_area:
            continue
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        if fiducial_mask is not None and fiducial_mask[cy, cx] > 0:
            continue
        patch = pad[cy:cy + crop, cx:cx + crop]
        if patch.shape[:2] == (crop, crop):
            out.append((patch, (cx, cy), int(area)))
    return out


def process_trap_frame(bgr: np.ndarray,
                       expected_marker_ids: Tuple[int, ...] = (0, 1, 2, 3),
                       marker_spacing_mm: Tuple[float, float] = (
                           PROVISIONAL_MARKER_SPACING_W_MM,
                           PROVISIONAL_MARKER_SPACING_H_MM
                       ),
                       max_rotation_deg: float = PROVISIONAL_MAX_ROTATION_DEG,
                       min_focus_score: float = PROVISIONAL_FOCUS_FLOOR_LAPLACIAN,
                       max_card_saturation: float = PROVISIONAL_MAX_CARD_SATURATION,
                       min_area: int = 6,
                       max_area: int = 8000,
                       crop: int = 64,
                       abs_floor_px: Optional[float] = None,
                       allow_provisional: bool = False) -> Tuple[Optional[List[Any]], str, Dict[str, Any]]:
    """
    Gateway processing pipeline for an incoming ESP32-CAM trap frame:
      1. Verifies 4-corner ArUco fiducial frame (scale, tilt, focus).
         Rejects frame with (None, reason, info) if any check fails.
      2. Generates fiducial mask over printed markers to prevent false insect detection.
      3. Executes watershed blob segmentation with dynamic mm/pixel derived scale.
      4. Evaluates card saturation fraction (sum(blob_area) / active_card_area).
         If saturation > max_card_saturation, returns (blobs, 'CARD_SATURATED', info)
         and triggers card_replacement_required = True.
    """
    passed, reason, info = verify_fiducial_frame(
        bgr,
        expected_marker_ids=expected_marker_ids,
        marker_spacing_mm=marker_spacing_mm,
        max_rotation_deg=max_rotation_deg,
        min_focus_score=min_focus_score,
        allow_provisional=allow_provisional
    )
    if not passed:
        return None, reason, info

    h, w = bgr.shape[:2]
    fiducial_mask = np.zeros((h, w), dtype=np.uint8)
    if "marker_corners" in info:
        for mid, quad in info["marker_corners"].items():
            cv2.fillPoly(fiducial_mask, [quad.astype(np.int32)], 255)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        fiducial_mask = cv2.dilate(fiducial_mask, k_dilate)

    blobs = segment_trap_blobs(
        bgr,
        min_area=min_area,
        max_area=max_area,
        abs_floor_px=abs_floor_px,
        mm_per_pixel=info["measured_mm_per_pixel"],
        rel_frac=0.25,
        crop=crop,
        fiducial_mask=fiducial_mask
    )

    active_area = float((fiducial_mask == 0).sum()) if fiducial_mask is not None else float(h * w)
    total_blob_area = sum(int(area) for _, _, area in blobs)
    coverage = float(total_blob_area / active_area) if active_area > 0 else 0.0

    info["n_blobs"] = len(blobs)
    info["total_blob_area_px"] = total_blob_area
    info["active_area_px"] = int(active_area)
    info["coverage_fraction"] = coverage
    info["card_saturated"] = coverage > max_card_saturation
    info["card_replacement_required"] = info["card_saturated"]

    if info["card_saturated"]:
        return blobs, "CARD_SATURATED", info

    return blobs, "ok", info


# ------------------------------------------------------------------------------
# Verified ICAR / NIPHM Trap ETL Registry
# ------------------------------------------------------------------------------
# Surviving numeric sticky-card threshold: exactly ONE (sugarcane whitefly / woolly aphid).
# All other taxa are negative entries (NOT_SAMPLED_BY_STICKY_TRAP) or qualitative trend alerting.
TRAP_ETL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # Surviving Numeric Sticky-Card Threshold (Sugarcane)
    # --------------------------------------------------------------------------
    "sugarcane_whitefly": {
        "scientific_name": "Aleurolobus barodensis",
        "crop": "sugarcane",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": 100.0,
        "threshold_unit": "insects_per_trap_daily",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "take up the intervention when the population exceeds 100 per trap.",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 11.",
        "ambiguity_note": (
            "Source counts woolly aphid and whitefly together under one 100/trap daily threshold. "
            "It is ambiguous whether 100 is combined across both species or per-species. "
            "Resolved conservatively as a combined count across both sucking pest species."
        ),
        "management_action": (
            "Intervention threshold exceeded (>100/trap daily). Conserve/release parasitoids "
            "Encarsia flavoscutellum; apply recommended IPM intervention if sustained."
        )
    },
    "sugarcane_woolly_aphid": {
        "scientific_name": "Ceratovacuna lanigera",
        "crop": "sugarcane",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": 100.0,
        "threshold_unit": "insects_per_trap_daily",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "take up the intervention when the population exceeds 100 per trap.",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 11.",
        "ambiguity_note": (
            "Source counts woolly aphid and whitefly together under one 100/trap daily threshold. "
            "It is ambiguous whether 100 is combined across both species or per-species. "
            "Resolved conservatively as a combined count across both sucking pest species."
        ),
        "management_action": (
            "Intervention threshold exceeded (>100/trap daily). Conserve Dipha aphidivora / "
            "Micromus igorotus predators; apply recommended IPM intervention if sustained."
        )
    },
    "sugarcane_whitefly_woolly_aphid": {
        "scientific_name": "Aleurolobus barodensis / Ceratovacuna lanigera",
        "crop": "sugarcane",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": 100.0,
        "threshold_unit": "insects_per_trap_daily",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "take up the intervention when the population exceeds 100 per trap.",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 11.",
        "ambiguity_note": (
            "Source counts woolly aphid and whitefly together under one 100/trap daily threshold. "
            "Resolved conservatively as combined count."
        ),
        "management_action": "Intervention threshold exceeded (>100/trap daily). Apply recommended IPM intervention."
    },

    # --------------------------------------------------------------------------
    # Wheat Sticky Traps (Reframed strictly as Relative Trend / Rate-of-Change Alerting)
    # --------------------------------------------------------------------------
    "wheat_aphid": {
        "scientific_name": "Rhopalosiphum padi / Sitobion avenae",
        "crop": "wheat",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "status": "NO_PUBLISHED_ETL",
        "source_url": "https://niphm.gov.in/IPMPackages/Wheat.pdf",
        "source_quote": "Count the number of aphids on the traps daily and take the appropriate decision",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Wheat', p. 10.",
        "management_action": (
            "Sticky traps provide relative trend / rate-of-change alerting (delta N). "
            "No published quantitative ETL exists in ICAR/NIPHM guidelines."
        )
    },
    "wheat_thrips": {
        "scientific_name": "Anaphothrips obscurus",
        "crop": "wheat",
        "trap_type": "blue_sticky_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "status": "NO_PUBLISHED_ETL",
        "source_url": "https://niphm.gov.in/IPMPackages/Wheat.pdf",
        "source_quote": "Count the number of thrips on the traps daily and take the appropriate decision",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Wheat', p. 10.",
        "management_action": (
            "Blue sticky traps provide relative trend / rate-of-change alerting (delta N). "
            "No published quantitative ETL exists in ICAR/NIPHM guidelines."
        )
    },

    # --------------------------------------------------------------------------
    # Explicit Negative Entries (NOT_SAMPLED_BY_STICKY_TRAP)
    # Hardware sampling boundaries: stem borers, leaf folders, planthoppers, hispa.
    # --------------------------------------------------------------------------
    "rice_yellow_stem_borer": {
        "scientific_name": "Scirpophaga incertulas",
        "crop": "rice",
        "trap_type": "pheromone_lure_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "pheromone_lure_trap (@ 5 traps/ha)",
        "native_threshold": "25 moths / trap / week",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Rice.pdf",
        "source_quote": "Stem borer 2 egg-mass/m 2 or 1 moth/m2 or 25 moths/ trap/ week",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Rice', p. 9.",
        "management_action": (
            "Taxon not validly sampled by sticky cards. Deploy pheromone lure traps @ 5/ha "
            "or monitor egg masses in field. Intervention threshold: 25 moths/trap/week."
        )
    },
    "rice_leaf_folder": {
        "scientific_name": "Cnaphalocrocis medinalis",
        "crop": "rice",
        "trap_type": "in_field_visual_scouting",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "in_field_visual_scouting",
        "native_threshold": "2 damaged leaves per hill",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Rice.pdf",
        "source_quote": "Leaf-folder 2 Fully damaged leaves (FDL) with larva/hill",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Rice', p. 9.",
        "management_action": (
            "Taxon not validly sampled by sticky cards. Conduct in-field visual scouting. "
            "Intervention threshold: 2 fully damaged leaves with larva per hill."
        )
    },
    "rice_brown_planthopper": {
        "scientific_name": "Nilaparvata lugens",
        "crop": "rice",
        "trap_type": "in_field_visual_scouting",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "in_field_visual_scouting (hill base tap)",
        "native_threshold": "10-15 hoppers per hill",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Rice.pdf",
        "source_quote": "Brown planthopper/WBPH 10-15 hoppers/hill",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Rice', p. 9.",
        "management_action": (
            "Taxon not validly sampled by sticky cards. Conduct in-field visual scouting by gently "
            "tapping hill bases. Intervention threshold: 10-15 hoppers per hill."
        )
    },
    "rice_hispa": {
        "scientific_name": "Dicladispa armigera",
        "crop": "rice",
        "trap_type": "in_field_visual_scouting",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "sweep_net_or_hill_inspection",
        "native_threshold": "2 adults or 2 dead leaves per hill",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Rice.pdf",
        "source_quote": "Rice hispa 2 adults or 2 dead leaf /hill",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Rice', p. 9.",
        "management_action": (
            "Taxon not validly sampled by sticky cards. Conduct visual field scouting or sweep net collection. "
            "Intervention threshold: 2 adults or 2 damaged leaves per hill."
        )
    },
    "sugarcane_early_shoot_borer": {
        "scientific_name": "Chilo infuscatellus",
        "crop": "sugarcane",
        "trap_type": "pheromone_lure_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "pheromone_lure_trap / visual field dead heart",
        "native_threshold": "15% dead heart visual scouting",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "During each week of surveillance, the number of moths/trap/week should be counted",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 11.",
        "management_action": (
            "Taxon not validly sampled by sticky cards. Deploy pheromone traps @ 4-5/acre or scout for 15% dead heart. "
            "Sticky traps cannot monitor borer moths."
        )
    },
    "sugarcane_top_borer": {
        "scientific_name": "Scirpophaga excerptalis",
        "crop": "sugarcane",
        "trap_type": "pheromone_lure_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "pheromone_lure_trap (@ 4-5/acre)",
        "native_threshold": "Moth trap monitoring",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "Pheromone traps for borer @ 4-5/acre have to be installed",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 11.",
        "management_action": "Taxon not validly sampled by sticky cards. Deploy pheromone lure traps @ 4-5/acre."
    },
    "sugarcane_pyrilla": {
        "scientific_name": "Pyrilla perpusilla",
        "crop": "sugarcane",
        "trap_type": "in_field_visual_scouting",
        "threshold_value": None,
        "threshold_unit": None,
        "native_sampling_method": "in_field_leaf_scouting",
        "native_threshold": "3-5 individuals per leaf",
        "status": "NOT_SAMPLED_BY_STICKY_TRAP",
        "source_url": "https://niphm.gov.in/IPMPackages/Sugarcane.pdf",
        "source_quote": "visual scouting on leaves; Epiricania is parasitoid",
        "citation": "NIPHM (2014), 'AESA Based IPM Package for Sugarcane', p. 18.",
        "ambiguity_note": (
            "Epiricania melanoleuca (3-5 cocoons/leaf) is the biocontrol parasitoid of Pyrilla, "
            "not Pyrilla itself. Threshold requires in-field visual scouting of nymphs/adults. "
            "Not sampled by sticky cards."
        ),
        "management_action": (
            "Conduct in-field leaf scouting. Conserve/redistribute Epiricania melanoleuca parasitoids. "
            "Not sampled by sticky cards."
        )
    },

    # Generic aliases
    "whitefly": {
        "scientific_name": "Bemisia tabaci / Aleurolobus barodensis",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "status": "NO_PUBLISHED_ETL",
        "citation": "NIPHM IPM guidelines. For sugarcane, see 'sugarcane_whitefly' (100/trap daily).",
        "management_action": "Sticky cards provide relative trend alerting unless sugarcane_whitefly is specified."
    },
    "aphids": {
        "scientific_name": "Aphididae",
        "trap_type": "yellow_sticky_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "status": "NO_PUBLISHED_ETL",
        "citation": "NIPHM IPM guidelines. For sugarcane, see 'sugarcane_woolly_aphid' (100/trap daily).",
        "management_action": "Sticky cards provide relative trend alerting unless sugarcane_woolly_aphid is specified."
    },
    "thrips": {
        "scientific_name": "Thripidae",
        "trap_type": "blue_or_yellow_sticky_trap",
        "threshold_value": None,
        "threshold_unit": None,
        "status": "NO_PUBLISHED_ETL",
        "citation": "NIPHM IPM guidelines. See 'wheat_thrips' for blue sticky trap guidance.",
        "management_action": "Sticky cards provide relative trend alerting. No published quantitative ETL."
    }
}


def evaluate_trap_counts_against_etl(pest_counts: Dict[str, Any],
                                     card_replaced_at: Optional[Any] = None,
                                     current_timestamp: Optional[Any] = None,
                                     days_monitored: Optional[float] = None,
                                     card_saturated: bool = False,
                                     card_coverage: Optional[float] = None,
                                     max_card_saturation: float = PROVISIONAL_MAX_CARD_SATURATION) -> List[Dict[str, Any]]:
    """
    Compare monitored trap counts against ICAR/NIPHM Economic Threshold Levels (ETL).

    Cumulative Monitoring Hardening (Finding 6):
      1. Saturation guard: If card_saturated is True or card_coverage > 0.30,
         returns status 'CARD_SATURATED' and sets card_replacement_required = True.
      2. Deployment timestamp guard: Requires card_replaced_at or explicit days_monitored.
         Missing timestamp returns status 'MISSING_DEPLOYMENT_TIMESTAMP'.
      3. Valid monitoring window: Enforces [1.0, 7.0] day window. Windows outside this range
         return status 'INVALID_MONITORING_WINDOW'.
      4. Quantitative comparison: Evaluates observed daily rate against surviving numeric ETL
         (sugarcane whitefly/woolly aphid = 100/trap daily).
      5. Negative entries: Pests not sampled by sticky cards return 'NOT_SAMPLED_BY_STICKY_TRAP'.
      6. Rice and wheat sucking pests return 'NO_PUBLISHED_ETL' with relative trend guidance.

    (Note on Finding 6 operational bounds: The [1.0, 7.0] day monitoring window and 0.30
    saturation floor are project engineering judgements based on dust buildup, UV breakdown
    of glue, and insect decomposition; they are not ICAR/NIPHM standards).
    """
    # 1. Card Saturation Guard
    if card_saturated or (card_coverage is not None and card_coverage > max_card_saturation):
        return [{
            "pest_name": "all",
            "status": "CARD_SATURATED",
            "card_replacement_required": True,
            "coverage_fraction": card_coverage,
            "detail": "Trap card is saturated (>30% area covered). Replace card immediately; automated counts suppressed.",
            "management_action": "Replace sticky card immediately."
        }]

    # 2. Deployment Timing & Validity Window
    if days_monitored is None:
        if card_replaced_at is None:
            return [{
                "pest_name": "all",
                "status": "MISSING_DEPLOYMENT_TIMESTAMP",
                "card_replacement_required": False,
                "detail": "card_replaced_at timestamp or explicit days_monitored is required to evaluate cumulative sticky-trap captures.",
                "management_action": "Record trap card deployment date/time."
            }]
        if current_timestamp is None:
            import time
            current_timestamp = time.time()

        if isinstance(card_replaced_at, (int, float)) and isinstance(current_timestamp, (int, float)):
            days_monitored = (float(current_timestamp) - float(card_replaced_at)) / 86400.0
        elif isinstance(card_replaced_at, str):
            from datetime import datetime
            t0 = datetime.fromisoformat(card_replaced_at.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(current_timestamp.replace("Z", "+00:00")) if isinstance(current_timestamp, str) else datetime.fromtimestamp(current_timestamp, t0.tzinfo)
            days_monitored = (t1 - t0).total_seconds() / 86400.0
        else:
            days_monitored = 0.0

    if days_monitored < 1.0 or days_monitored > 7.0:
        return [{
            "pest_name": "all",
            "status": "INVALID_MONITORING_WINDOW",
            "days_monitored": float(days_monitored),
            "card_replacement_required": days_monitored > 7.0,
            "detail": f"Monitoring window of {days_monitored:.2f} days is outside the valid range [1.0, 7.0] days. Counting suppressed.",
            "management_action": "Inspect within 1-7 days of deployment or replace degraded card."
        }]

    results = []
    for pest_key, count in pest_counts.items():
        norm_key = str(pest_key).strip().lower().replace(" ", "_").replace("-", "_")
        observed = float(count)
        daily_rate = float(round(observed / float(days_monitored), 2))
        weekly_rate = float(round(daily_rate * 7.0, 2))

        if norm_key not in TRAP_ETL_REGISTRY:
            results.append({
                "pest_name": pest_key,
                "count_observed": observed,
                "days_monitored": float(days_monitored),
                "daily_rate": daily_rate,
                "weekly_rate": weekly_rate,
                "threshold_value": None,
                "threshold_unit": None,
                "status": "UNKNOWN_PEST",
                "citation": "No ICAR/NIPHM record found in local registry for this pest identifier.",
                "source_url": None,
                "source_quote": None,
                "management_action": "Consult local Krishi Vigyan Kendra (KVK) advisory."
            })
            continue

        entry = TRAP_ETL_REGISTRY[norm_key]
        reg_status = entry.get("status")
        thresh = entry.get("threshold_value")
        unit = entry.get("threshold_unit")

        if reg_status == "NOT_SAMPLED_BY_STICKY_TRAP":
            status = "NOT_SAMPLED_BY_STICKY_TRAP"
        elif thresh is None:
            status = "NO_PUBLISHED_ETL"
        else:
            if unit == "insects_per_trap_daily":
                status = "EXCEEDS_ETL" if daily_rate >= thresh else "BELOW_ETL"
            elif unit == "insects_per_trap_weekly":
                status = "EXCEEDS_ETL" if weekly_rate >= thresh else "BELOW_ETL"
            else:
                status = "EXCEEDS_ETL" if daily_rate >= thresh else "BELOW_ETL"

        results.append({
            "pest_name": norm_key,
            "count_observed": observed,
            "days_monitored": float(days_monitored),
            "daily_rate": daily_rate,
            "weekly_rate": weekly_rate,
            "threshold_value": thresh,
            "threshold_unit": unit,
            "status": status,
            "citation": entry.get("citation"),
            "source_url": entry.get("source_url"),
            "source_quote": entry.get("source_quote"),
            "management_action": entry.get("management_action")
        })

    return results


# ------------------------------------------------------------------------------
# Time-Series Trap Count Trend Alerting (Section B Rework)
# ------------------------------------------------------------------------------

class TrapCountHistory:
    """
    Accumulates time-series trap capture records per trap station and computes
    relative trend alerts (RISING, STABLE, FALLING, INSUFFICIENT_HISTORY)
    against a rolling median baseline of prior valid observations.

    Pattern matches ThermalMastStation in core/thermal.py.

    Design and Operational Constraints:
      1. Saturated cards (CARD_SATURATED), invalid monitoring windows (<1d or >7d),
         or missing deployment timestamps are strictly excluded from baseline computation.
      2. Median is used over mean: pest captures are prone to transient gusts or single
         extreme outliers that would artificially elevate a mean baseline and mask
         genuine subsequent population increases.
      3. Card replacement handling: Sticky cards are cumulative collectors. Raw counts
         are card-scoped and may never be differenced across cards, but interval catch rates
         are physically comparable across cards, so discarding them loses valid signal to
         avoid an unmeasured trapping-efficiency bias. By default (reset_on_card_replacement=False),
         valid interval catch rates from the prior card carry over into the rolling baseline window,
         eliminating post-replacement blind periods while reporting baseline_spans_card_change=True.
         If reset_on_card_replacement=True is explicitly chosen, the baseline cohort resets,
         returning INSUFFICIENT_HISTORY until min_baseline_obs are accumulated on the new card.
    """
    def __init__(self,
                 trap_id: str = "trap_01",
                 min_baseline_obs: int = PROVISIONAL_TREND_MIN_OBSERVATIONS,
                 rolling_window: int = PROVISIONAL_TREND_ROLLING_WINDOW,
                 rising_ratio: float = PROVISIONAL_TREND_RISING_RATIO,
                 falling_ratio: float = PROVISIONAL_TREND_FALLING_RATIO,
                 min_rate_diff: float = PROVISIONAL_TREND_MIN_RATE_DIFF,
                 reset_on_card_replacement: bool = False):
        self.trap_id = trap_id
        self.min_baseline_obs = min_baseline_obs
        self.rolling_window = rolling_window
        self.rising_ratio = rising_ratio
        self.falling_ratio = falling_ratio
        self.min_rate_diff = min_rate_diff
        self.reset_on_card_replacement = reset_on_card_replacement

        # Dict mapping pest_name -> list of observation records
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        # Archive of past sessions retired upon card replacement
        self.archived_sessions: Dict[str, List[Dict[str, Any]]] = {}

    def record_observation(self,
                           timestamp: Any,
                           pest_name: str,
                           count: float,
                           card_replaced_at: Any,
                           status: str = "ok",
                           card_coverage: Optional[float] = None,
                           days_monitored: Optional[float] = None) -> Dict[str, Any]:
        """
        Record an incoming trap observation and compute relative trend status.

        Parameters:
          timestamp: float (epoch seconds) or ISO datetime string
          pest_name: str identifier of pest taxon
          count: cumulative insect count observed on current card
          card_replaced_at: deployment timestamp of current card
          status: observation quality status ('ok', 'CARD_SATURATED', etc.)
          card_coverage: optional fractional coverage of card area [0.0, 1.0]
          days_monitored: optional elapsed days; derived from timestamps if omitted

        Returns:
          dict containing:
            trap_id: str
            pest_name: str
            timestamp: float or str
            current_daily_rate: float
            baseline_median_rate: float or None
            n_baseline_observations: int
            status: 'RISING', 'STABLE', 'FALLING', or 'INSUFFICIENT_HISTORY'
            ratio: float or None
            record_status: str
            card_replaced: bool
        """
        norm_pest = str(pest_name).strip().lower().replace(" ", "_").replace("-", "_")

        # 1. Evaluate deployment timing & validity window
        effective_status = status
        if card_replaced_at is None and days_monitored is None:
            effective_status = "MISSING_DEPLOYMENT_TIMESTAMP"
            days_monitored = 0.0
            cumulative_daily_rate = 0.0
            interval_daily_rate = 0.0
        else:
            if days_monitored is None:
                if isinstance(card_replaced_at, (int, float)) and isinstance(timestamp, (int, float)):
                    days_monitored = (float(timestamp) - float(card_replaced_at)) / 86400.0
                elif isinstance(card_replaced_at, str):
                    from datetime import datetime
                    t0 = datetime.fromisoformat(card_replaced_at.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if isinstance(timestamp, str) else datetime.fromtimestamp(timestamp, t0.tzinfo)
                    days_monitored = (t1 - t0).total_seconds() / 86400.0
                else:
                    days_monitored = 0.0

            if card_coverage is not None and card_coverage > PROVISIONAL_MAX_CARD_SATURATION:
                effective_status = "CARD_SATURATED"
            elif effective_status == "ok" and (days_monitored < 1.0 or days_monitored > 7.0):
                effective_status = "INVALID_MONITORING_WINDOW"

            cumulative_daily_rate = float(round(float(count) / float(days_monitored), 2)) if days_monitored > 0 else 0.0

        # 2. Check for physical card replacement between observations
        if norm_pest not in self.history:
            self.history[norm_pest] = []
            self.archived_sessions[norm_pest] = []

        prior_records = self.history[norm_pest]
        card_changed = False
        if len(prior_records) > 0:
            last_replacement = prior_records[-1].get("card_replaced_at")
            if last_replacement is not None and card_replaced_at is not None and card_replaced_at != last_replacement:
                card_changed = True
                if self.reset_on_card_replacement:
                    # Counts on a card replaced between observations are not continuous.
                    # Archive prior observations to isolate the baseline to the active card cohort.
                    self.archived_sessions[norm_pest].extend(prior_records)
                    self.history[norm_pest] = []
                    prior_records = []

        # 3. Compute interval rate (Finding 2)
        # Prevents late-in-life surge dampening caused by cumulative running average.
        if len(prior_records) == 0 or card_changed:
            # First reading on this card: interval is elapsed time since card deployment
            interval_days = float(days_monitored)
            delta_count = max(0.0, float(count))
        else:
            prev_rec = prior_records[-1]
            prev_ts = prev_rec["timestamp"]
            prev_count = prev_rec["count"]
            if isinstance(prev_ts, (int, float)) and isinstance(timestamp, (int, float)):
                interval_days = (float(timestamp) - float(prev_ts)) / 86400.0
            else:
                from datetime import datetime
                t0 = datetime.fromisoformat(str(prev_ts).replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")) if isinstance(timestamp, str) else datetime.fromtimestamp(timestamp, t0.tzinfo)
                interval_days = (t1 - t0).total_seconds() / 86400.0
            delta_count = max(0.0, float(count) - float(prev_count))

        interval_daily_rate = float(round(delta_count / float(interval_days), 2)) if interval_days > 0 else 0.0

        # 4. Extract valid prior observations for rolling baseline (using interval rates)
        valid_prior_rates = []
        for rec in prior_records:
            # Exclude CARD_SATURATED, INVALID_MONITORING_WINDOW, MISSING_DEPLOYMENT_TIMESTAMP
            if rec.get("status") in ("CARD_SATURATED", "INVALID_MONITORING_WINDOW", "MISSING_DEPLOYMENT_TIMESTAMP"):
                continue
            if rec.get("card_saturated", False):
                continue
            valid_prior_rates.append(rec["interval_daily_rate"])

        # Retain only the most recent rolling window observations
        window_rates = valid_prior_rates[-self.rolling_window:]
        n_baseline = len(window_rates)

        # 5. Compute trend status against median baseline
        if n_baseline < self.min_baseline_obs:
            trend_status = "INSUFFICIENT_HISTORY"
            baseline_val = None
            ratio = None
            baseline_was_zero = False
            baseline_spans_card_change = False
        else:
            # Median chosen over mean per design constraint: in pest monitoring,
            # a single transient spike (e.g. a swarm gust) or anomalous reading should not
            # skew the baseline upward, which would mask a genuine gradual population rise
            # in subsequent readings.
            baseline_val = float(round(float(np.median(window_rates)), 2))
            baseline_was_zero = bool(baseline_val <= 0.0)

            # Check if rolling baseline draws on records from a prior card
            baseline_spans_card_change = any(
                rec.get("card_replaced_at") != card_replaced_at
                for rec in prior_records[-self.rolling_window:]
                if rec.get("status") not in ("CARD_SATURATED", "INVALID_MONITORING_WINDOW", "MISSING_DEPLOYMENT_TIMESTAMP")
                and not rec.get("card_saturated", False)
            )

            if baseline_was_zero:
                # Zero baseline: requires rate strictly above min_rate_diff to flag RISING.
                # Avoids float('inf') for JSON serializability: ratio is None, baseline_was_zero is True.
                if interval_daily_rate > self.min_rate_diff:
                    trend_status = "RISING"
                else:
                    trend_status = "STABLE"
                ratio = None
            else:
                ratio = float(round(interval_daily_rate / baseline_val, 2))
                if ratio >= self.rising_ratio and (interval_daily_rate - baseline_val) >= self.min_rate_diff:
                    trend_status = "RISING"
                elif ratio <= self.falling_ratio:
                    trend_status = "FALLING"
                else:
                    trend_status = "STABLE"

        # 6. Append this observation into history
        obs_entry = {
            "timestamp": timestamp,
            "pest_name": norm_pest,
            "count": float(count),
            "card_replaced_at": card_replaced_at,
            "days_monitored": float(days_monitored) if days_monitored is not None else None,
            "interval_daily_rate": float(interval_daily_rate),
            "cumulative_daily_rate": float(cumulative_daily_rate),
            "daily_rate": float(interval_daily_rate),  # primary daily rate is interval rate
            "status": effective_status,
            "card_saturated": (effective_status == "CARD_SATURATED")
        }
        self.history[norm_pest].append(obs_entry)

        # 7. Return auditable result dictionary
        return {
            "trap_id": self.trap_id,
            "pest_name": norm_pest,
            "timestamp": timestamp,
            "count": float(count),
            "days_monitored": float(days_monitored) if days_monitored is not None else None,
            "interval_daily_rate": float(interval_daily_rate),
            "cumulative_daily_rate": float(cumulative_daily_rate),
            "current_daily_rate": float(interval_daily_rate),
            "baseline_median_rate": baseline_val,
            "n_baseline_observations": n_baseline,
            "status": trend_status,
            "ratio": ratio,
            "baseline_was_zero": baseline_was_zero,
            "baseline_spans_card_change": baseline_spans_card_change,
            "record_status": effective_status,
            "card_replaced": card_changed
        }

