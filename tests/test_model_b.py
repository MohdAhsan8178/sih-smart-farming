"""
Unit and integration tests for Model B (Trap Pest Patch Classifier).

Tests:
  1. Taxonomy constants and unfixable gap metadata (configs/classes_model_b.py).
  2. Model B forward pass and parameter count (~153k) (train/model_b.py).
  3. Augmentation transforms output dimensions and value ranges (train/transforms_model_b.py).
  4. ONNX model inference and output consistency (artifacts/onnx/model_b.onnx).
  5. classify_trap_blobs integration in core/trap_segmentation.py.
  6. Advisory wire contract integration in edge/storage.py.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.classes_model_b import (
    CLASS_NAMES, NUM_CLASSES, IDX, NAME_FROM_IDX,
    TARGET_CLASS_NAME, KNOWN_UNFIXABLE_GAPS
)
from train.model_b import TrapPestCNN, build_model_b
from train.transforms_model_b import train_transform_model_b, eval_transform_model_b
from core.trap_segmentation import classify_trap_blobs
from configs.paths import ONNX_DIR


def test_classes_model_b():
    """Verify 3-class taxonomy configuration and gap disclosures."""
    assert NUM_CLASSES == 3
    assert CLASS_NAMES == ['small_pale_winged', 'larger_insect', 'debris']
    assert TARGET_CLASS_NAME == 'small_pale_winged'
    assert len(IDX) == 3
    assert len(NAME_FROM_IDX) == 3
    for i, name in enumerate(CLASS_NAMES):
        assert IDX[name] == i
        assert NAME_FROM_IDX[i] == name

    # Verify formal gap disclosures exist
    assert "sugarcane_woolly_aphid_wax" in KNOWN_UNFIXABLE_GAPS
    assert "soft_bodied_aphids_thrips" in KNOWN_UNFIXABLE_GAPS
    assert KNOWN_UNFIXABLE_GAPS["sugarcane_woolly_aphid_wax"]["status"] == "UNREPRESENTED_IN_PUBLIC_DATA"
    assert KNOWN_UNFIXABLE_GAPS["soft_bodied_aphids_thrips"]["status"] == "UNLABELLED_IN_PUBLIC_STICKY_TRAP_DATA"


def test_model_b_architecture():
    """Verify TrapPestCNN forward pass and compact parameter budget."""
    model = build_model_b(num_classes=3)
    model.eval()

    # Parameter count check: ~294k params (4-block dual-conv stem)
    n_params = sum(p.numel() for p in model.parameters())
    assert 280_000 < n_params < 310_000, f"Expected ~294k params, got {n_params}"

    # Forward pass on 64x64 batch
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, 3)

    # Feature extraction check
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)


def test_model_b_transforms():
    """Verify train and eval transforms preserve (3, 64, 64) dimensions."""
    train_tf = train_transform_model_b(size=64)
    eval_tf = eval_transform_model_b(size=64)

    dummy_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    t_res = train_tf(image=dummy_rgb)['image']
    assert isinstance(t_res, torch.Tensor)
    assert t_res.shape == (3, 64, 64)

    e_res = eval_tf(image=dummy_rgb)['image']
    assert isinstance(e_res, torch.Tensor)
    assert e_res.shape == (3, 64, 64)


def test_onnx_model_b_standing_guard():
    """Standing guard: Assert production ONNX artifact exists, loads, and produces (B, 3)."""
    onnx_file = ONNX_DIR / "model_b.onnx" if (ONNX_DIR / "model_b.onnx").exists() else ONNX_DIR / "model_b_calibrated.onnx"
    assert onnx_file.exists(), f"MANDATORY ONNX artifact missing: {onnx_file}"

    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_file))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_input = np.random.randn(4, 3, 64, 64).astype(np.float32)
    res = session.run([output_name], {input_name: dummy_input})[0]
    assert res.shape == (4, 3), f"Expected shape (4, 3), got {res.shape}"
    assert not np.isnan(res).any(), "NaN detected in ONNX output"
    assert not np.isinf(res).any(), "Inf detected in ONNX output"


def test_classify_trap_blobs_standing_guard():
    """Verify classify_trap_blobs fails loudly on missing ONNX artifact and never falls back."""
    fake_path = ONNX_DIR / "non_existent_model.onnx"
    blobs = [(np.zeros((64, 64, 3), dtype=np.uint8), (50, 50), 30)]

    with pytest.raises(FileNotFoundError, match="Model B ONNX artifact missing"):
        classify_trap_blobs(blobs, onnx_path=str(fake_path))


def test_classify_trap_blobs_calibrated_and_abstention():
    """Verify classify_trap_blobs adheres to claim discipline, output wire contract, and abstention path."""
    onnx_file = ONNX_DIR / "model_b.onnx" if (ONNX_DIR / "model_b.onnx").exists() else ONNX_DIR / "model_b_calibrated.onnx"
    assert onnx_file.exists()

    # Create dummy extracted blobs: [(crop_64x64, (cx, cy), area)]
    blobs = [
        (np.random.randint(180, 255, (64, 64, 3), dtype=np.uint8), (100, 100), 45),
        (np.random.randint(0, 50, (64, 64, 3), dtype=np.uint8), (200, 200), 120),
        (np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8), (300, 300), 20),
    ]

    # Test with default onnx_path=None (should auto-load model_b.onnx)
    res = classify_trap_blobs(blobs, onnx_path=None)

    assert res["total_blobs_counted"] == 3
    assert res["primary_count_source"] == "deterministic_watershed"
    assert res["verification_status"] == "RECALLED_UNVERIFIED"
    assert res["classification_source"] == "CROSS_DOMAIN_PRETRAINED"
    assert "known_gap_warning" in res
    assert "provenance_disclaimer" in res
    assert "calibration" in res
    assert res["calibration"]["T_cal"] == 0.9541
    assert res["calibration"]["tau_energy"] == -3.8054
    assert res["calibration"]["tau_conf"] == 0.60

    dist = res["morphological_distribution"]
    assert "small_pale_winged" in dist
    assert "larger_insect" in dist
    assert "debris" in dist
    assert "UNCERTAIN_NON_TARGET" in dist

    total_frac = sum(dist[k]["fraction"] for k in dist)
    assert 0.99 <= total_frac <= 1.01
    assert "predictions" in res
    assert len(res["predictions"]) == 3
    valid_categories = set(CLASS_NAMES + ["UNCERTAIN_NON_TARGET"])
    for p in res["predictions"]:
        assert p in valid_categories


def test_storage_advisory_pest_integration(tmp_path):
    """Verify create_advisory integrates the populated pest block."""
    from edge.storage import EdgeStorage
    db_path = tmp_path / "test_advisory.db"
    storage = EdgeStorage(str(db_path))

    # Register a scan
    storage.record_scan_start(scan_id="SCAN_TEST_PEST", metadata={"dominant_crop": "sugarcane", "field_id": "F01"})
    storage.record_scan_end(scan_id="SCAN_TEST_PEST")

    pest_data = {
        "total_blobs_counted": 85,
        "primary_count_source": "deterministic_watershed",
        "morphological_distribution": {
            "small_pale_winged": {"count": 60, "fraction": 0.7059},
            "larger_insect": {"count": 15, "fraction": 0.1765},
            "debris": {"count": 10, "fraction": 0.1176},
        },
        "verification_status": "RECALLED_UNVERIFIED",
        "classification_source": "CROSS_DOMAIN_PRETRAINED",
        "provenance_disclaimer": "European sticky-trap CNN.",
        "known_gap_warning": "Woolly aphid wax gap and aphid/thrips gap.",
    }

    advisory = storage.create_advisory("SCAN_TEST_PEST", pest_data=pest_data)
    payload = advisory

    assert "pest" in payload
    assert payload["pest"]["total_blobs_counted"] == 85
    assert payload["pest"]["verification_status"] == "RECALLED_UNVERIFIED"
    assert payload["pest"]["classification_source"] == "CROSS_DOMAIN_PRETRAINED"
    assert payload["pest"]["morphological_distribution"]["small_pale_winged"]["count"] == 60


def test_watershed_primary_etl_evaluation():
    """F1.4: Verify ETL evaluation uses deterministic watershed count as primary gate."""
    from core.trap_segmentation import evaluate_trap_counts_against_etl

    morph_dist = {
        "small_pale_winged": {"count": 1, "fraction": 0.25},
        "larger_insect": {"count": 0, "fraction": 0.0},
        "debris": {"count": 2, "fraction": 0.50},
        "UNCERTAIN_NON_TARGET": {"count": 1, "fraction": 0.25},
    }

    # Case 1: 4 watershed blobs total (1 small_pale_winged, 2 debris, 1 UNCERTAIN_NON_TARGET)
    # count_observed must equal 4, NOT 1
    res_below = evaluate_trap_counts_against_etl(
        pest_counts={"sugarcane_whitefly": 1},
        days_monitored=3.0,
        total_blobs_counted=4,
        morphological_distribution=morph_dist,
    )
    assert len(res_below) == 1
    item = res_below[0]
    assert item["target_pest_context"] == "sugarcane_whitefly"
    assert item["count_basis"] == "watershed_all_blobs"
    assert item["count_observed"] == 4.0, f"Expected 4.0, got {item['count_observed']}"
    assert item["daily_rate"] == 1.33
    assert item["status"] == "BELOW_ETL"
    assert item["threshold_value"] == 100.0
    assert "disclaimer" in item
    assert "watershed blob count is authoritative" in item["disclaimer"]
    assert item["abstention_count"] == 1
    assert item["total_blobs_counted"] == 4

    # Case 2: 350 watershed blobs total over 3 days (rate = 116.67 > 100 ETL)
    res_exceeds = evaluate_trap_counts_against_etl(
        pest_counts={"sugarcane_whitefly": 50},
        days_monitored=3.0,
        total_blobs_counted=350,
        morphological_distribution=morph_dist,
    )
    assert len(res_exceeds) == 1
    item_ex = res_exceeds[0]
    assert item_ex["count_observed"] == 350.0
    assert item_ex["daily_rate"] == 116.67
    assert item_ex["status"] == "EXCEEDS_ETL"

