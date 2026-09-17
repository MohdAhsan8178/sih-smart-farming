"""
Spatial (within-frame) and temporal (across-frame) aggregation.

Spatial  = per-tile decision, then MIL max-pool. A lesion ANYWHERE means the
           frame is diseased. Logical OR, never a vote, never a top-k average.
Temporal = k-of-n agreement PLUS a score floor, across repeated passes over
           the same GPS cell. This is where voting belongs.

v4 fixes (red-team round 3):
  * aggregate_cell returns an explicit state, never a bare None.
    v3 returned (None, 0.0, n) for healthy cells, all-uncertain cells and
    never-visited cells alike, so prescription mapping could not tell
    "do not spray" from "re-fly this cell".
"""
import numpy as np
from collections import Counter

FRAME_STATES = ('DISEASE', 'HEALTHY', 'NOT_CROP', 'UNCERTAIN')
CELL_STATES = ('DISEASE', 'HEALTHY', 'UNCERTAIN', 'NO_DATA')


def aggregate_frame(tile_probs, healthy_cols, notcrop_col,
                    tau_disease=0.55, tau_margin=0.10, min_tiles=2,
                    tau_healthy=0.50, notcrop_frac=0.50):
    """
    tile_probs : (n_tiles, n_classes) softmax probabilities, or None
    returns    : (state, class_id_or_None, score)
    """
    if tile_probs is None:
        return 'UNCERTAIN', None, 0.0
    p = np.asarray(tile_probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < min_tiles or p.shape[1] < 2:
        return 'UNCERTAIN', None, 0.0

    n_classes = p.shape[1]
    healthy_cols = np.asarray(healthy_cols, dtype=int)
    notcrop_col = int(notcrop_col)
    disease_cols = np.setdiff1d(np.arange(n_classes),
                                np.append(healthy_cols, notcrop_col))
    if disease_cols.size == 0:
        return 'UNCERTAIN', None, 0.0

    # --- frames dominated by non-crop content --------------------------
    nc_mean = float(p[:, notcrop_col].mean())
    if nc_mean > notcrop_frac:
        return 'NOT_CROP', notcrop_col, nc_mean

    # --- per-tile, like-for-like comparison ----------------------------
    d_best = p[:, disease_cols].max(axis=1)
    d_arg = disease_cols[p[:, disease_cols].argmax(axis=1)]
    h_best = p[:, healthy_cols].max(axis=1)

    votes = (d_best >= tau_disease) & ((d_best - h_best) >= tau_margin)

    # --- MIL max-pool: ANY qualifying tile carries the frame -----------
    if votes.any():
        i = int(np.argmax(np.where(votes, d_best, -np.inf)))
        return 'DISEASE', int(d_arg[i]), float(d_best[i])

    if float(h_best.mean()) >= tau_healthy:
        hc = int(healthy_cols[p[:, healthy_cols].mean(axis=0).argmax()])
        return 'HEALTHY', hc, float(h_best.mean())

    return 'UNCERTAIN', None, float(d_best.max())


def aggregate_cell(frame_results, k=2, n=3, min_score=0.55, min_frames=2):
    """
    frame_results : list of (state, class_id, score) from aggregate_frame,
                    oldest first. Only the last `n` are considered.
    returns       : dict with an EXPLICIT state, never a bare None.

        {'state': 'DISEASE'|'HEALTHY'|'UNCERTAIN'|'NO_DATA',
         'class_id': int|None, 'score': float,
         'n_frames': int, 'n_agree': int}

    'HEALTHY'   -> do not spray
    'UNCERTAIN' -> evidence conflicting or too weak; re-inspect
    'NO_DATA'   -> not enough usable frames; re-fly
    """
    recent = list(frame_results or [])[-n:]
    if len(recent) < min_frames:
        return {'state': 'NO_DATA', 'class_id': None, 'score': 0.0,
                'n_frames': len(recent), 'n_agree': 0}

    diseased = [r for r in recent if r[0] == 'DISEASE']
    healthy = [r for r in recent if r[0] == 'HEALTHY']

    if len(diseased) >= k:
        counts = Counter(int(cid) for _, cid, _ in diseased)
        cls, cnt = counts.most_common(1)[0]
        if cnt >= k:
            scores = [sc for _, cid, sc in diseased if int(cid) == cls]
            mean_score = float(np.mean(scores))
            if mean_score >= min_score:
                return {'state': 'DISEASE', 'class_id': cls,
                        'score': mean_score, 'n_frames': len(recent),
                        'n_agree': cnt}
            # agreement without evidence is not a detection
            return {'state': 'UNCERTAIN', 'class_id': cls, 'score': mean_score,
                    'n_frames': len(recent), 'n_agree': cnt}

    if len(healthy) >= k:
        cls = Counter(int(cid) for _, cid, _ in healthy).most_common(1)[0][0]
        return {'state': 'HEALTHY', 'class_id': cls,
                'score': float(np.mean([sc for _, _, sc in healthy])),
                'n_frames': len(recent), 'n_agree': len(healthy)}

    return {'state': 'UNCERTAIN', 'class_id': None, 'score': 0.0,
            'n_frames': len(recent), 'n_agree': 0}
