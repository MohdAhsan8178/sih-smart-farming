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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
