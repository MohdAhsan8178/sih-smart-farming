"""
tests/test_energy_parity.py — Energy parity test between runtime and calibration (J2.3, J2.4).
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from configs.classes import CROP_COLS
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from configs.train_config import MODEL_B_T_CAL, T_CAL
from core.rejection import open_set_energy as runtime_model_a_energy, stable_logsumexp as core_logsumexp


def test_j2_3_model_a_energy_parity():
    """J2.3: Take 50 val logits, compute energy through runtime function and calibration function, assert equal."""
    np.random.seed(42)
    sample_logits = np.random.randn(50, 29).astype(np.float64)

    # 1. Runtime function (core.rejection.open_set_energy with T=1.0 on raw logits)
    runtime_e = runtime_model_a_energy(sample_logits, CROP_COLS, T=1.0)

    # 2. Calibration formula (train/calibrate.py calling core.rejection.open_set_energy / direct formula)
    crop_cols_arr = np.asarray(CROP_COLS, dtype=int)
    z_crop = sample_logits[:, crop_cols_arr]
    m = z_crop.max(axis=1, keepdims=True)
    calibration_e = -1.0 * (m.squeeze(1) + np.log(np.exp(z_crop - m).sum(axis=1)))

    assert len(runtime_e) == 50
    assert len(calibration_e) == 50
    np.testing.assert_allclose(runtime_e, calibration_e, rtol=1e-12, atol=1e-12)


def test_j2_4_model_b_energy_parity():
    """J2.4: Take 50 sample logits for Model B, compute energy through runtime formula and calibration formula."""
    np.random.seed(42)
    sample_logits = np.random.randn(50, len(MODEL_B_CLASSES)).astype(np.float64)

    # 1. Runtime formula in core/trap_segmentation.py:595
    runtime_b_energy = -MODEL_B_T_CAL * core_logsumexp(sample_logits / MODEL_B_T_CAL, axis=1)

    # 2. Calibration formula in sih-model-b-training.ipynb:411
    calib_b_energy = -MODEL_B_T_CAL * logsumexp(sample_logits / MODEL_B_T_CAL, axis=1)

    assert len(runtime_b_energy) == 50
    assert len(calib_b_energy) == 50
    np.testing.assert_allclose(runtime_b_energy, calib_b_energy, rtol=1e-12, atol=1e-12)
