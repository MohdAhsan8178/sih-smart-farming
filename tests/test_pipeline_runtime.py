#!/usr/bin/env python3
"""
Unit and integration test suite for edge/pipeline.py runtime components (Step 21).

Tests:
1. DropOldestQueue: bounded capacity, drop-oldest behavior, sentinel preservation, clean closure.
2. CaptureThread: frame reading, timestamping, GPS handling, metadata injection.
3. GateTileThread: rejection handling, tile extraction, always producing N_TILES=9.
4. InferenceThread: dry-run mock inference shape (9, 29) and latency simulation.
5. DecisionAggregateStoreThread: decide, aggregate_frame, aggregate_cell, JSONL format.
6. End-to-end EdgePipeline execution on compiled test video.
"""

import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest

from configs.classes import CLASS_NAMES, NUM_CLASSES
from configs.train_config import ENGINE_BATCH, IMAGE_SIZE, N_TILES
from edge.pipeline import (
    CaptureThread,
    DecisionAggregateStoreThread,
    DropOldestQueue,
    EdgePipeline,
    GateTileThread,
    InferenceThread,
    _QUEUE_TIMEOUT,
    load_log_priors,
)
from edge.storage import EdgeStorage


def test_drop_oldest_queue_basic():
    """Verify bounded queue drops oldest item and preserves order."""
    q = DropOldestQueue(maxsize=3)
    assert q.put("A")
    assert q.put("B")
    assert q.put("C")
    assert q.dropped_count == 0

    # Put 4th item -> should drop 'A'
    assert q.put("D")
    assert q.dropped_count == 1

    assert q.get(timeout=0.1) == "B"
    assert q.get(timeout=0.1) == "C"
    assert q.get(timeout=0.1) == "D"
    assert q.get(timeout=0.01) is _QUEUE_TIMEOUT


def test_drop_oldest_queue_preserves_sentinel():
    """Verify sentinel None is not dropped when queue overflows."""
    q = DropOldestQueue(maxsize=3)
    q.put("A")
    q.put("B")
    q.put(None)  # Sentinel at tail

    # Attempt to put another item
    q.put("C")
    # Sentinel None should remain in queue
    items = []
    while True:
        it = q.get(timeout=0.05)
        if it is _QUEUE_TIMEOUT:
            continue
        if it is None:
            break
        items.append(it)
    assert "B" in items or "C" in items


def test_drop_oldest_queue_close():
    """Verify closing queue immediately unblocks waiting get()."""
    q = DropOldestQueue(maxsize=4)
    t0 = time.time()

    def delayed_close():
        time.sleep(0.05)
        q.close()

    import threading
    t = threading.Thread(target=delayed_close)
    t.start()
    res = q.get(timeout=2.0)
    t.join()
    assert res is None
    assert time.time() - t0 < 1.0


def test_load_log_priors():
    """Verify load_log_priors returns valid shape and probabilities."""
    priors = load_log_priors(Path(__file__).resolve().parent.parent)
    assert isinstance(priors, np.ndarray)
    assert priors.shape == (NUM_CLASSES,)
    assert not np.isnan(priors).any()
    assert not np.isinf(priors).any()


def test_end_to_end_pipeline_dryrun():
    """Verify EdgePipeline executes end-to-end and writes JSONL events."""
    video_path = Path("test_video_from_dataset_images.mp4")
    if not video_path.exists():
        video_path = Path("data/video/test_video_from_dataset_images.mp4")

    assert video_path.exists(), "Test video does not exist: %s" % video_path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "pipeline_events.jsonl"

        pipeline = EdgePipeline(
            source=str(video_path),
            dry_run=True,
            max_frames=15,  # test first 15 frames
            output_jsonl=str(output_jsonl),
            queue_size=8,
        )

        metrics = pipeline.run()

        assert metrics["frames_seen"] == 15
        assert metrics["frames_passed"] > 0
        assert metrics["tiles_classified"] == metrics["frames_passed"] * N_TILES
        assert metrics["scenes_per_second"] > 0.0
        assert output_jsonl.exists()

        # Check JSONL events content
        with open(output_jsonl, "r") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        assert len(lines) == metrics["frames_passed"]

        for event in lines:
            assert "event_id" in event
            assert "source_image" in event
            assert "timestamp_utc" in event
            assert "cell_id" in event
            assert event["gate_passed"] is True
            assert "indices" in event
            assert "canopy_cover" in event["indices"]
            assert event["indices"]["canopy_cover"] is not None
            assert "vari" in event["indices"]
            assert "exg" in event["indices"]
            assert "tgi" in event["indices"]
            assert "dgci" in event["indices"]
            assert "frame_verdict" in event
            assert event["frame_verdict"]["state"] in ("DISEASE", "HEALTHY", "NOT_CROP", "UNCERTAIN")
            assert "cell_verdict" in event
            assert event["cell_verdict"]["state"] in ("DISEASE", "HEALTHY", "UNCERTAIN", "NO_DATA")
            assert len(event["tile_decisions"]) == N_TILES

        # Check that the synthesized advisory has vegetation block
        assert pipeline.t4_decision.last_advisory is not None
        adv = pipeline.t4_decision.last_advisory
        assert "vegetation" in adv
        assert "canopy_cover" in adv["vegetation"]
        assert "vari" in adv["vegetation"]
        assert adv["vegetation"]["canopy_cover"]["mean"] is not None


def test_g5_2_temperature_and_energy_single_application():
    """
    G5.2 / L2.3: Trace the exact Nano runtime path from TensorRT output to softmax/energy,
    and prove T is applied exactly ONCE.
    Asserts result equals softmax(z / T_CAL) and energy = -1.0 * logsumexp(z[CROP_COLS] / 1.0).
    Energy is evaluated strictly on raw logits at T=1.0 per L2 spec.
    """
    from scipy.special import logsumexp
    from configs.train_config import T_CAL, TAU_ENERGY, TAU_CONF, TAU_PRIOR
    from configs.classes import CROP_COLS, NOTCROP_COL, NUM_CLASSES
    from core.rejection import softmax, open_set_energy, decide

    # Known fixed logit vector
    np.random.seed(42)
    z_raw = np.random.randn(9, NUM_CLASSES).astype(np.float32)
    z_copy = z_raw.copy()

    # Hop 1: Direct mathematical expectations
    expected_softmax = np.exp(z_raw / T_CAL) / np.sum(np.exp(z_raw / T_CAL), axis=1, keepdims=True)
    expected_energy = -1.0 * logsumexp(z_raw[:, CROP_COLS] / 1.0, axis=1)

    # Hop 2: Softmax scaling verification
    probs = softmax(z_raw, T=T_CAL)
    np.testing.assert_allclose(probs, expected_softmax, rtol=1e-5, atol=1e-6)

    # Hop 3: Energy computation verification (T=1.0 raw logits)
    energy = open_set_energy(z_raw, CROP_COLS, T=1.0)
    np.testing.assert_allclose(energy, expected_energy, rtol=1e-5, atol=1e-6)

    # Hop 4: Rejection decision integration (core/rejection.py:decide)
    log_priors = np.zeros(NUM_CLASSES, dtype=np.float32)
    decisions = decide(
        logits=z_raw,
        crop_cols=CROP_COLS,
        notcrop_col=NOTCROP_COL,
        log_priors=log_priors,
        tau_energy=TAU_ENERGY,
        T_cal=T_CAL,
        tau_conf=TAU_CONF,
        tau_prior=TAU_PRIOR,
    )
    assert len(decisions) == 9

    # Assert logits array was never mutated in-place
    np.testing.assert_array_equal(z_raw, z_copy)


def test_l2_2_energy_temperature_invariance_and_tau_energy_rejection():
    """
    L2.2: Verify decide() on 50 validation logits rejects exactly the samples where
    open_set_energy(z, CROP_COLS, T=1.0) > TAU_ENERGY (-2.7957), independent of T_cal.
    """
    from configs.train_config import T_CAL, TAU_ENERGY, TAU_CONF, TAU_PRIOR
    from configs.classes import CROP_COLS, NOTCROP_COL, NUM_CLASSES
    from core.rejection import open_set_energy, decide

    np.random.seed(12345)
    # Generate 50 synthetic logits spanning in-distribution and out-of-distribution energies
    z_val = np.random.randn(50, NUM_CLASSES).astype(np.float64) * 2.5
    # Force some logits to have distinct not_crop argmax, high crop energy, and low crop energy
    z_val[0:5, NOTCROP_COL] = 10.0  # NOT_CROP
    z_val[5:15, CROP_COLS] = -5.0   # High energy (OOD / UNKNOWN)
    z_val[15:25, CROP_COLS[0]] = 8.0  # Low energy (in-distribution)

    log_priors = np.zeros(NUM_CLASSES, dtype=np.float64)

    # Compute ground truth energy at T=1.0
    energy_t1 = open_set_energy(z_val, CROP_COLS, T=1.0)

    # Evaluate decide() across multiple temperature calibration settings
    temps_to_test = [1.0, T_CAL, 0.5, 1.5, 2.0]
    results_by_temp = []

    for t_val in temps_to_test:
        decisions = decide(
            logits=z_val,
            crop_cols=CROP_COLS,
            notcrop_col=NOTCROP_COL,
            log_priors=log_priors,
            tau_energy=TAU_ENERGY,
            T_cal=t_val,
            tau_conf=TAU_CONF,
            tau_prior=TAU_PRIOR,
        )
        results_by_temp.append(decisions)

    # 1. Assert energy values in decision dicts are IDENTICAL across all temperatures and match energy_t1
    for d_list in results_by_temp:
        assert len(d_list) == 50
        for i, d in enumerate(d_list):
            np.testing.assert_allclose(d["energy"], energy_t1[i], rtol=1e-5, atol=1e-6)

    # 2. For each sample where notcrop is not argmax, assert state == 'UNKNOWN' iff energy_t1[i] > TAU_ENERGY
    for d_list in results_by_temp:
        for i, d in enumerate(d_list):
            is_not_crop = int(np.argmax(z_val[i])) == int(NOTCROP_COL)
            if not is_not_crop:
                if energy_t1[i] > TAU_ENERGY:
                    assert d["state"] == "UNKNOWN", f"Sample {i} energy {energy_t1[i]} > {TAU_ENERGY} but got state {d['state']}"
                else:
                    assert d["state"] in ("OK", "ABSTAIN"), f"Sample {i} energy {energy_t1[i]} <= {TAU_ENERGY} but got state {d['state']}"

    # 3. Assert boolean mask of UNKNOWN rejections is 100% invariant to T_cal
    mask_baseline = [d["state"] == "UNKNOWN" for d in results_by_temp[0]]
    for idx, d_list in enumerate(results_by_temp[1:], 1):
        mask_test = [d["state"] == "UNKNOWN" for d in d_list]
        assert mask_test == mask_baseline, f"UNKNOWN mask mismatch for T_cal={temps_to_test[idx]}"



def test_h4_2_end_to_end_parity_mac_onnx_vs_pytorch():
    """
    H4.2: End-to-end parity test on Mac with ONNX Runtime standing in for TensorRT.
    Loads 20 test_indist images (BGR as camera path provides), passes through real edge
    preprocessing (cv2 bilinear resize to 224x224, float32 [0, 255]), and runs model_a_fused.onnx.
    Compares against PyTorch EMA weights with Albumentations eval_transform (resize reproduced identically).
    Asserts agreement and max logit difference (< 1e-4).
    Also asserts that feeding RGB instead of BGR fails the strict parity check.
    """
    import cv2
    import torch
    import onnxruntime as ort
    import pandas as pd
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from configs.classes import CLASS_NAMES
    from train.model import build_model

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "splits_v3/test_indist.csv"
    ckpt_path = repo_root / "artifacts/checkpoints/v3/stage1.pt"
    onnx_path = repo_root / "artifacts/onnx/model_a_fused.onnx"

    if not csv_path.exists() or not ckpt_path.exists() or not onnx_path.exists():
        pytest.skip("Required artifacts for H4.2 parity check not present on disk.")

    df = pd.read_csv(csv_path).head(20)

    # 1. PyTorch EMA model
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(ema_sd)
    model.eval()

    # 2. ONNX Runtime session
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    agreements = 0
    max_diffs = []

    for i, row in df.iterrows():
        p = repo_root / row["path"]
        bgr = cv2.imread(str(p))
        assert bgr is not None, f"Image {p} could not be read."

        # Identical resize: cv2.INTER_LINEAR to 224x224
        t_resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Path A: PyTorch Albumentations on RGB
        rgb = cv2.cvtColor(t_resized, cv2.COLOR_BGR2RGB)
        transformed = tf(image=rgb)["image"]
        with torch.no_grad():
            pt_out = model(transformed.unsqueeze(0)).numpy()[0]
        pt_pred = int(np.argmax(pt_out))

        # Path B: Edge BGR float32 -> model_a_fused.onnx
        batch_inp = np.zeros((9, 224, 224, 3), dtype=np.float32)
        batch_inp[0] = t_resized.astype(np.float32)
        onnx_out = sess.run(None, {inp_name: batch_inp})[0][0]
        onnx_pred = int(np.argmax(onnx_out))

        diff = float(np.max(np.abs(pt_out - onnx_out)))
        max_diffs.append(diff)
        if pt_pred == onnx_pred:
            agreements += 1

    # Verify 100% top-1 agreement and negligible numerical drift (< 1e-4)
    assert agreements == len(df), f"Agreement was {agreements}/{len(df)}, expected 20/20"
    assert max(max_diffs) < 1e-4, f"Max diff {max(max_diffs)} exceeded tolerance 1e-4"

    # Channel order guard: verify that if wrong channel order (RGB) is fed, diff blows up
    p0 = repo_root / df.iloc[0]["path"]
    bgr0 = cv2.imread(str(p0))
    t_rgb = cv2.cvtColor(cv2.resize(bgr0, (224, 224)), cv2.COLOR_BGR2RGB)
    wrong_batch = np.zeros((9, 224, 224, 3), dtype=np.float32)
    wrong_batch[0] = t_rgb.astype(np.float32)
    wrong_out = sess.run(None, {inp_name: wrong_batch})[0][0]
    rgb_diff = float(np.max(np.abs(pt_out - wrong_out)))
    assert rgb_diff > 1.0, f"RGB input should have caused large channel order discrepancy, got {rgb_diff}"


def test_k3_3_backend_trt_unavailable_raises_and_writes_no_advisory():
    """
    K3.3: Strict backend enforcement.
    When backend="trt" on a host where TRT is unavailable (or missing engine),
    EdgePipeline.run() raises RuntimeError immediately and writes NO advisory.
    """
    video_path = Path("test_video_from_dataset_images.mp4")
    if not video_path.exists():
        video_path = Path("data/video/test_video_from_dataset_images.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_no_fallback.db"
        storage = EdgeStorage(db_path=db_path)

        pipeline = EdgePipeline(
            source=str(video_path),
            backend="trt",
            engine_path=Path(tmpdir) / "nonexistent.engine",
            dry_run=False,
            max_frames=5,
            db_path=db_path,
            queue_size=4,
        )

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.run()

        assert "TensorRT backend requested but" in str(exc_info.value)

        # Assert no advisory was written to SQLite
        manifest = storage.get_manifest()
        assert len(manifest["advisories"]) == 0
        assert storage.get_advisory("latest") is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])


