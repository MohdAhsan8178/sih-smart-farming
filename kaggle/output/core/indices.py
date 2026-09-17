"""
Band-addressed vegetation indices, RGB-only NDVI proxies, and canopy masking.

Architectural Design (Section C):
  - Primitives accept a named band mapping `BandMap = Mapping[str, np.ndarray]`
    with keys "red", "green", "blue" (and future "nir").
  - Thin adapters (*_from_bgr, *_from_rgb) perform channel slicing and delegate
    directly to the band-addressed functions with zero intermediate arithmetic,
    guaranteeing bit-for-bit exact equivalence (np.array_equal).

Canopy Segmentation Prior to Index Computation (Section A):
  - Every vegetation index is computed over CANOPY PIXELS ONLY.
  - Reuses the absolute ExG threshold (PROVISIONAL_EXG_VEG_THRESHOLD = 20) without Otsu,
    preventing unimodal bisection on closed canopies.
  - Emits full distribution statistics (mean, median, std, n_canopy_pixels, vegetation_fraction).
  - Enforces PROVISIONAL_MIN_CANOPY_FRACTION (default 0.15); if canopy fraction is below
    this floor or if n_canopy_pixels == 0, returns None with status "insufficient_canopy".

Supported Indices (Section B):
  - ExG (Excess Green): 2G - R - B (clipped to [0, 255] uint8).
  - VARI: (G - R) / (G + R - B) (Gitelson et al. 2002).
  - TGI: G - 0.39*R - 0.61*B (Hunt et al. 2011).
  - NGRDI: (G - R) / (G + R) (Tucker 1979).
  - GMR: G - R (unnormalized channel difference, scale/illumination dependent, Section E fixed-AWB).
  - DGCI: Dark Green Colour Index (Karcher & Richardson 2003) via float32 HSV conversion.
"""
from typing import Mapping, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2

BandMap = Mapping[str, np.ndarray]

# Provisional constants - require empirical calibration on real field imagery
PROVISIONAL_MIN_CANOPY_FRACTION = 0.15  # minimum canopy fraction floor before rejecting index
# PROVISIONAL_EXG_VEG_THRESHOLD: Empirically tuned heuristic against synthetic and field tiles
# to prevent Otsu bisection on pure green canopies. Not from published literature.
PROVISIONAL_EXG_VEG_THRESHOLD = 20


def bgr_to_bandmap(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a BGR (H, W, 3) image into a band-addressed mapping.
    Performs NO arithmetic to guarantee exact delegating equivalence.
    """
    arr = np.asarray(bgr)
    return {
        "blue": arr[..., 0],
        "green": arr[..., 1],
        "red": arr[..., 2]
    }


def rgb_to_bandmap(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split an RGB (H, W, 3) image into a band-addressed mapping.
    Performs NO arithmetic to guarantee exact delegating equivalence.
    """
    arr = np.asarray(rgb)
    return {
        "red": arr[..., 0],
        "green": arr[..., 1],
        "blue": arr[..., 2]
    }


# ------------------------------------------------------------------------------
# Core Band-Addressed Index Functions
# ------------------------------------------------------------------------------

def exg(bands: BandMap) -> np.ndarray:
    """
    Excess Green (ExG = 2G - R - B), clipped to [0, 255] uint8.
    Matches historical thermal pipeline behavior.
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    b = np.asarray(bands["blue"], dtype=np.float32)
    return np.clip(2.0 * g - r - b, 0, 255).astype(np.uint8)


def vari(bands: BandMap, eps: float = 1e-6) -> np.ndarray:
    """
    Visible Atmospherically Resistant Index (Gitelson et al. 2002).
      VARI = (G - R) / (G + R - B)
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    b = np.asarray(bands["blue"], dtype=np.float32)
    denom = g + r - b
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + (denom == 0.0) * eps, denom)
    return (g - r) / denom


def tgi(bands: BandMap) -> np.ndarray:
    """
    Triangular Greenness Index (Hunt et al. 2011).
      TGI = G - 0.39 * R - 0.61 * B
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    b = np.asarray(bands["blue"], dtype=np.float32)
    return g - 0.39 * r - 0.61 * b


def ngrdi(bands: BandMap, eps: float = 1e-6) -> np.ndarray:
    """
    Normalized Green-Red Difference Index (Tucker 1979).
      NGRDI = (G - R) / (G + R)
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    denom = g + r
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return (g - r) / denom


def gmr(bands: BandMap) -> np.ndarray:
    """
    Green Minus Red (GMR = G - R).
    Unnormalized channel difference (Correction 1, Option a).
    Note: Highly scale/illumination dependent; valid only under fixed-exposure
    and fixed-AWB capture conditions (Section E).
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    return g - r


def dgci(bands: BandMap) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dark Green Colour Index (Karcher & Richardson 2003).
      DGCI = [( (H - 60) / 60 ) + (1 - S) + (1 - V)] / 3

    Implementation Details (Option a - Narrowed Foliage Domain):
      - Normalized float32 RGB converted to HSV via OpenCV cv2.COLOR_RGB2HSV:
          H in [0, 360) degrees directly (not [0, 179] as in uint8),
          S in [0, 1] directly,
          V in [0, 1] directly.
      - Narrowed Foliage Hue Domain [60.0, 120.0] degrees:
        Karcher & Richardson (2003) define DGCI strictly across the yellow-to-green spectrum
        H in [60, 120] degrees. Under this guard, DGCI over in-domain foliage pixels
        is strictly bounded to [0.0, 1.0].
      - Known Limitation (Shade Bias):
        Deeply shadowed leaves or foliage with high blue reflectance / waxy cuticles
        genuinely produce H > 120 degrees (bluish-green sector). Narrowing the domain to
        [60, 120] degrees excludes these pixels as out-of-domain. On frames with significant
        shade or bluish leaf varieties, this exclusion will bias the reported mean DGCI
        upward (toward lighter green/yellow-green). The aggregation pipeline tracks this
        via `out_of_domain_fraction` so excessive shade exclusion is explicitly visible.

    Returns:
      (dgci_array, in_domain_mask_bool)
    """
    r = np.asarray(bands["red"], dtype=np.float32)
    g = np.asarray(bands["green"], dtype=np.float32)
    b = np.asarray(bands["blue"], dtype=np.float32)

    max_val = max(float(r.max()), float(g.max()), float(b.max()), 1.0)
    scale = 255.0 if max_val > 1.0 else 1.0

    rgb_f = np.stack([r / scale, g / scale, b / scale], axis=-1).astype(np.float32)
    hsv = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2HSV)

    h_deg = hsv[..., 0]   # [0, 360)
    s = hsv[..., 1]       # [0, 1]
    v = hsv[..., 2]       # [0, 1]

    dgci_map = (((h_deg - 60.0) / 60.0) + (1.0 - s) + (1.0 - v)) / 3.0

    # Foliage hue domain check strictly bounded to [60.0, 120.0] degrees per Karcher & Richardson (2003)
    in_domain = (h_deg >= 60.0) & (h_deg <= 120.0)
    return dgci_map, in_domain


# ------------------------------------------------------------------------------
# Backward-Compatible Thin Array Adapters (Exact Channel-Slicing Delegation)
# ------------------------------------------------------------------------------

def excess_green(bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible ExG adapter from 3-channel BGR array."""
    return exg(bgr_to_bandmap(bgr))


def vari_from_bgr(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Backward-compatible VARI adapter from 3-channel BGR array."""
    return vari(bgr_to_bandmap(bgr), eps=eps)


def tgi_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible TGI adapter from 3-channel BGR array."""
    return tgi(bgr_to_bandmap(bgr))


def ngrdi_from_bgr(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Backward-compatible NGRDI adapter from 3-channel BGR array."""
    return ngrdi(bgr_to_bandmap(bgr), eps=eps)


def gmr_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible GMR adapter from 3-channel BGR array."""
    return gmr(bgr_to_bandmap(bgr))


def dgci_from_bgr(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible DGCI adapter from 3-channel BGR array."""
    return dgci(bgr_to_bandmap(bgr))


# ------------------------------------------------------------------------------
# Canopy Masking and Aggregation (Section A)
# ------------------------------------------------------------------------------

def vegetation_mask(bgr_or_bands: Union[np.ndarray, BandMap],
                    thresh: int = PROVISIONAL_EXG_VEG_THRESHOLD) -> Tuple[np.ndarray, float]:
    """
    Absolute-threshold vegetation mask. Returns (mask_uint8, veg_fraction).

    Moved from core/thermal.py per Correction 3.
    Deliberately NOT Otsu. Otsu on a unimodal histogram splits near the mean,
    so a fully vegetated frame comes back ~50% vegetation and fails any
    downstream purity gate.
    """
    if isinstance(bgr_or_bands, dict) or (hasattr(bgr_or_bands, "__getitem__") and "green" in bgr_or_bands):
        exg_arr = exg(bgr_or_bands)
    else:
        exg_arr = excess_green(bgr_or_bands)
    mask = (exg_arr > thresh).astype(np.uint8) * 255
    return mask, float((mask > 0).mean())


def aggregate_index(index_map: np.ndarray,
                    mask: np.ndarray,
                    min_fraction: float = PROVISIONAL_MIN_CANOPY_FRACTION,
                    in_domain_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Aggregate a 2D vegetation index array over canopy-masked pixels only.

    Correction 6: Returns full distribution statistics:
      - mean: float or None
      - median: float or None
      - std: float or None
      - n_canopy_pixels: int
      - vegetation_fraction: float
      - status: 'ok' or 'insufficient_canopy'
      - out_of_domain_fraction (optional): float if domain mask supplied (e.g. DGCI).

    Guards empty mask (n_canopy_pixels == 0) and low-canopy condition explicitly.
    """
    mask_bool = np.asarray(mask) > 0
    total_pixels = mask_bool.size
    n_canopy = int(np.count_nonzero(mask_bool))
    veg_fraction = float(n_canopy / total_pixels) if total_pixels > 0 else 0.0

    # Explicit guard: zero pixels or below configurable canopy floor
    if n_canopy == 0 or veg_fraction < min_fraction:
        res = {
            "mean": None,
            "median": None,
            "std": None,
            "n_canopy_pixels": n_canopy,
            "vegetation_fraction": veg_fraction,
            "status": "insufficient_canopy"
        }
        if in_domain_mask is not None:
            res["out_of_domain_fraction"] = 0.0
        return res

    combined_mask = mask_bool
    out_of_domain_frac = 0.0
    if in_domain_mask is not None:
        valid_in_canopy = in_domain_mask[mask_bool]
        if valid_in_canopy.size > 0:
            out_of_domain_frac = float(1.0 - valid_in_canopy.mean())
        combined_mask = mask_bool & in_domain_mask

    values = index_map[combined_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        res = {
            "mean": None,
            "median": None,
            "std": None,
            "n_canopy_pixels": n_canopy,
            "vegetation_fraction": veg_fraction,
            "status": "insufficient_canopy"
        }
        if in_domain_mask is not None:
            res["out_of_domain_fraction"] = out_of_domain_frac
        return res

    res = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "n_canopy_pixels": n_canopy,
        "vegetation_fraction": veg_fraction,
        "status": "ok"
    }
    if in_domain_mask is not None:
        res["out_of_domain_fraction"] = out_of_domain_frac
    return res


def compute_canopy_index(image_or_bands: Union[np.ndarray, BandMap],
                         index_name: str = "vari",
                         min_fraction: float = PROVISIONAL_MIN_CANOPY_FRACTION,
                         thresh: int = PROVISIONAL_EXG_VEG_THRESHOLD,
                         eps: float = 1e-6) -> Dict[str, Any]:
    """
    Unified evaluation of any vegetation index over canopy-segmented pixels.
    """
    if isinstance(image_or_bands, dict) or (hasattr(image_or_bands, "__getitem__") and "green" in image_or_bands):
        bands = image_or_bands
    else:
        bands = bgr_to_bandmap(image_or_bands)

    mask, veg_frac = vegetation_mask(bands, thresh=thresh)
    in_domain_mask = None
    idx_lower = index_name.strip().lower()

    if idx_lower == "exg":
        idx_map = exg(bands).astype(np.float32)
    elif idx_lower == "vari":
        idx_map = vari(bands, eps=eps)
    elif idx_lower == "tgi":
        idx_map = tgi(bands)
    elif idx_lower == "ngrdi":
        idx_map = ngrdi(bands, eps=eps)
    elif idx_lower == "gmr":
        idx_map = gmr(bands)
    elif idx_lower == "dgci":
        idx_map, in_domain_mask = dgci(bands)
    else:
        raise ValueError(f"Unknown vegetation index: {index_name}. "
                         f"Available: 'exg', 'vari', 'tgi', 'ngrdi', 'gmr', 'dgci'")

    stats = aggregate_index(idx_map, mask, min_fraction=min_fraction, in_domain_mask=in_domain_mask)
    stats["index_name"] = idx_lower
    return stats


def masked_index_mean(image_or_bands: Union[np.ndarray, BandMap],
                      index_name: str = "vari",
                      min_fraction: float = PROVISIONAL_MIN_CANOPY_FRACTION,
                      thresh: int = PROVISIONAL_EXG_VEG_THRESHOLD,
                      eps: float = 1e-6) -> Tuple[Optional[float], float, str]:
    """
    Prompt 3 Section A: Canopy masking integration for vegetation index computation.

    Computes index mean over masked canopy pixels only, using vegetation_mask().
    Emits vegetation_fraction and status string alongside the index.

    Returns:
      Tuple[Optional[float], float, str]: (mean, vegetation_fraction, status)
        - mean: float or None (if insufficient canopy / no valid pixels)
        - vegetation_fraction: float (always float in [0.0, 1.0])
        - status: str ('ok' or 'insufficient_canopy')
    """
    stats = compute_canopy_index(
        image_or_bands=image_or_bands,
        index_name=index_name,
        min_fraction=min_fraction,
        thresh=thresh,
        eps=eps
    )
    return stats["mean"], float(stats["vegetation_fraction"]), str(stats["status"])


def check_frame_usable_for_indices(metadata: Mapping[str, Any]) -> Tuple[bool, str]:
    """
    Evaluate whether a captured frame's metadata indicates it is usable for vegetation index calculation.

    Checks the frame's saturation_status:
      - "SATURATION_OK": Returns (True, "") - frame is within radiometric tolerances.
      - "SATURATION_DETECTED": Returns (False, reason) - near-saturation clipping (>1% DN >= 250)
        invalidates division-based vegetation indices (such as VARI, NDVI).
      - "NOT_EVALUATED": Returns (False, reason) - frame quality was not or could not be evaluated.

    Parameters:
      metadata: Frame metadata dictionary (as emitted by edge.camera.create_frame_metadata).

    Returns:
      Tuple[bool, str]: (is_usable, reason_string)
    """
    if not isinstance(metadata, (dict, Mapping)):
        return False, "Metadata must be a dictionary or Mapping"

    status = metadata.get("saturation_status")
    if status == "SATURATION_OK":
        return True, ""
    elif status == "SATURATION_DETECTED":
        reason = metadata.get("saturation_status_reason") or "Channel saturation detected exceeding alert threshold"
        return False, f"Frame unusable for indices: {reason}"
    elif status == "NOT_EVALUATED":
        reason = metadata.get("saturation_status_reason") or "Saturation was not evaluated"
        return False, f"Frame unusable for indices: {reason}"
    else:
        return False, f"Frame unusable for indices: unknown or missing saturation_status '{status}'"
