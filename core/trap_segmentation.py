"""
Sticky-trap blob segmentation for the gateway pest counter.

v4 fixes (red-team round 3):
  * Marker extraction no longer uses a single global `0.3 * dist.max()`.
    The distance transform peak of a large moth or a clump sets that
    threshold to ~12 px, while a whitefly or thrip peaks at 2-4 px. Every
    micro-pest lost its marker and was flooded as background - a 100%
    false-negative rate for exactly the pests yellow traps are FOR,
    triggered by the presence of one large insect anywhere on the board.
    Markers now come from max(absolute floor, relative), with the floor set
    from the KNOWN physical scale (the trap camera geometry is fixed).
  * Label loop range corrected: after `markers += 1`, valid object labels
    are 2..n, not 2..n+1.
  * The padded image is built once, not once per blob.

v4 ADDITIONAL fix, found by EXECUTING the code (not present in any audit):
  * adaptiveThreshold with blockSize=25 responds only near EDGES of regions
    larger than the block, so a large insect segments as a HOLLOW RING and
    its interior is never marked foreground. This both fragments large
    insects and (accidentally) suppressed the dist.max() inflation, which is
    why the micro-pest bug did not reproduce until the blob was solidified.
    The mask is now adaptive OR global, followed by hole filling.
"""
import numpy as np
import cv2


def _fill_holes(mask):
    """Flood from the border; anything unreached is an interior hole."""
    h, w = mask.shape
    ff = mask.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def extract_markers(dist, abs_floor_px=1.8, rel_frac=0.25):
    """
    Dual-criterion seed extraction.

    abs_floor_px : absolute distance-transform floor, in pixels. Set from the
                   smallest target pest's radius at YOUR camera geometry.
                   The trap node has a fixed camera-to-trap distance, so this
                   is a known physical constant, not a tuning knob.
    rel_frac     : relative criterion, retained so large clumps still seed.
    """
    _, micro = cv2.threshold(dist, abs_floor_px, 255, cv2.THRESH_BINARY)
    _, big = cv2.threshold(dist, rel_frac * float(dist.max()), 255,
                           cv2.THRESH_BINARY)
    return np.uint8(np.maximum(micro, big))


def segment_trap_blobs(bgr, min_area=8, max_area=1200,
                       abs_floor_px=1.8, rel_frac=0.25, crop=64):
    """
    Returns a list of (crop_bgr, (cx, cy), area) for each detected insect.
    Adaptive threshold on LAB b (robust to trap fading and glue glare),
    then distance-transform watershed with size-safe markers.
    """
    bgr = np.asarray(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    bch = lab[:, :, 2]

    # Adaptive handles trap fading and uneven lighting but goes HOLLOW on
    # regions larger than blockSize. Global Otsu fills large bodies but drifts
    # as the trap yellows. Use both, then fill any remaining interior holes.
    adaptive = cv2.adaptiveThreshold(bch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 8)
    _, glob = cv2.threshold(bch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(adaptive, glob)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    clean = cv2.morphologyEx(_fill_holes(mask), cv2.MORPH_OPEN, k)

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
        patch = pad[cy:cy + crop, cx:cx + crop]
        if patch.shape[:2] == (crop, crop):
            out.append((patch, (cx, cy), int(area)))
    return out
