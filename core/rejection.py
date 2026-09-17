"""
Three-layer rejection: known negative -> open-set energy -> confidence abstain.

v4 fixes (red-team round 3):
  * open_set_energy takes crop_cols and handles 1D or 2D input.
    v3 shipped a stale all-logit `energy_np` in the runtime section that
    contradicted the spec and crashed on 1D vectors.
  * The confidence/abstain gate uses UNADJUSTED probabilities. Post-hoc
    logit adjustment is applied to the CLASS CHOICE only. v3 gated on
    adjusted probabilities, which lets a rare class with a large prior
    shift manufacture high confidence on a low-evidence (OOD) input.
"""
import numpy as np


def _as_2d(logits):
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 1:
        return a[None, :], True
    if a.ndim != 2:
        raise ValueError(f'logits must be 1D or 2D, got shape {a.shape}')
    return a, False


def stable_logsumexp(a, axis=None, keepdims=False):
    """
    Numerically stable pure NumPy logsumexp: avoids scipy dependency on Jetson Nano.
    Handles extreme logits (e.g. +/- 1000) without overflow or underflow.
    """
    a = np.asarray(a)
    max_a = np.max(a, axis=axis, keepdims=True)
    safe_max = np.where(np.isneginf(max_a), 0.0, max_a)
    shifted = a - safe_max
    exp_sum = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    res = np.log(exp_sum) + safe_max
    if not keepdims and axis is not None:
        res = np.squeeze(res, axis=axis)
    if np.any(np.isneginf(max_a)):
        if keepdims or axis is None:
            res = np.where(np.isneginf(max_a), -np.inf, res)
        else:
            res = np.where(np.isneginf(np.squeeze(max_a, axis=axis)), -np.inf, res)
    return res


def open_set_energy(logits, crop_cols, T=1.0):
    """
    E(x) = -T * logsumexp(logits[crop_cols] / T)

    LOW  energy = strong evidence for a KNOWN crop condition (in-distribution)
    HIGH energy = no crop evidence (out-of-distribution / unknown object)

    crop_cols MUST exclude not_crop. Including it makes a confidently-detected
    soil image look in-distribution, which defeats the detector.

    Accepts a 1D vector or a 2D (N, C) batch. Always returns a 1D array.
    Compute on RAW logits: before temperature scaling, before prior adjustment.
    """
    a, was_1d = _as_2d(logits)
    z = a[:, np.asarray(crop_cols, dtype=int)] / T
    m = z.max(axis=1, keepdims=True)
    e = -T * (m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))
    return e


def softmax(logits, T=1.0):
    a, _ = _as_2d(logits)
    z = a / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def posthoc_logit_adjust(logits, log_priors, tau=1.0):
    """
    Balanced scores from a model trained with plain cross-entropy.
    Menon et al. (ICLR 2021), post-hoc form. Inference-time only.

    NOTE the minus sign: we REMOVE the training prior.
    Use for the class CHOICE. Do not use for the confidence gate (see decide()).
    """
    a, _ = _as_2d(logits)
    return a - tau * np.asarray(log_priors, dtype=np.float64)[None, :]


# NOTE (Sep 2026 Step 13 D1): Stage 1/2 model trains with weighted loss (sqrt-inverse-frequency).
# Applying post-hoc logit adjustment on top double-corrects for class imbalance;
# log_priors and tau_prior must be re-calibrated in Step 15.
def decide(logits, crop_cols, notcrop_col, log_priors,
           tau_energy, T_cal=1.0, tau_conf=0.60, tau_prior=1.0):
    """
    Per-tile decision combining all three rejection layers, in the right order.

    Returns a list of dicts, one per row:
        {'state': 'NOT_CROP'|'UNKNOWN'|'ABSTAIN'|'OK',
         'class_id': int|None, 'conf': float, 'energy': float}

    Ordering rationale:
      1. energy on RAW logits           -> is this a known crop condition at all?
      2. confidence on UNADJUSTED probs -> is the model sure about anything?
      3. prior adjustment               -> WHICH class, given 1 and 2 passed
    Steps 1 and 2 answer "how much evidence"; step 3 answers "which label".
    Mixing them lets a +6 prior shift on a 43-image class fabricate confidence
    from a logit vector that contains no evidence at all.
    """
    a, _ = _as_2d(logits)
    energy = open_set_energy(a, crop_cols)
    probs_raw = softmax(a, T=T_cal)                       # UNADJUSTED
    probs_adj = softmax(posthoc_logit_adjust(a, log_priors, tau_prior), T=T_cal)

    out = []
    for i in range(a.shape[0]):
        if int(np.argmax(a[i])) == int(notcrop_col):
            out.append({'state': 'NOT_CROP', 'class_id': int(notcrop_col),
                        'conf': float(probs_raw[i, notcrop_col]),
                        'energy': float(energy[i])})
            continue
        if energy[i] > tau_energy:
            out.append({'state': 'UNKNOWN', 'class_id': None,
                        'conf': 0.0, 'energy': float(energy[i])})
            continue
        crop_conf = float(probs_raw[i, np.asarray(crop_cols, dtype=int)].max())
        if crop_conf < tau_conf:
            out.append({'state': 'ABSTAIN', 'class_id': None,
                        'conf': crop_conf, 'energy': float(energy[i])})
            continue
        cid = int(np.argmax(probs_adj[i]))                # prior-adjusted CHOICE
        out.append({'state': 'OK', 'class_id': cid,
                    'conf': crop_conf, 'energy': float(energy[i])})
    return out


def fit_energy_threshold(id_logits, openset_logits, crop_cols, tpr=0.95):
    """
    Choose tau_energy so that `tpr` of real crop images are accepted.

    openset_logits MUST come from categories the model has NEVER seen -
    not from the not_crop images used in training. Outlier-exposure data
    is in-distribution for the trained model and cannot calibrate this.
    """
    id_e = open_set_energy(id_logits, crop_cols)
    ood_e = open_set_energy(openset_logits, crop_cols)
    tau = float(np.percentile(id_e, tpr * 100))
    fpr = float((ood_e < tau).mean())
    return tau, fpr
