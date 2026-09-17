#!/usr/bin/env python3
"""
Regression test suite for calibrated and provisional threshold plumbing in edge/pipeline.py.

Guards against Failure Pattern 4 (Green tests hiding defects / silent signature default fallback).
Asserts that every threshold affecting classification, rejection, and spatial/temporal
aggregation is explicitly passed as a keyword argument sourced from configs/train_config.py.
"""

import ast
import inspect
from pathlib import Path
import tempfile
import numpy as np
import pytest

import configs.train_config as train_config
from configs.classes import NUM_CLASSES
from edge.pipeline import (
    DecisionAggregateStoreThread,
    DropOldestQueue,
    load_log_priors,
)
from edge.tiler import TileBatch


# The mandatory mapping from function parameter name -> config constant name in configs/train_config.py
EXPECTED_PLUMBING = {
    "decide": {
        "tau_energy": "TAU_ENERGY",
        "T_cal": "T_CAL",
        "tau_conf": "TAU_CONF",
        "tau_prior": "TAU_PRIOR",
    },
    "aggregate_frame": {
        "tau_disease": "TAU_DISEASE",
        "tau_margin": "TAU_MARGIN",
        "min_tiles": "PROVISIONAL_MIN_TILES",
        "tau_healthy": "PROVISIONAL_TAU_HEALTHY",
        "notcrop_frac": "PROVISIONAL_NOTCROP_FRAC",
    },
    "aggregate_cell": {
        "k": "CELL_K",
        "n": "CELL_N",
        "min_score": "CELL_MIN_SCORE",
        "min_frames": "PROVISIONAL_CELL_MIN_FRAMES",
    },
}


def test_static_ast_call_sites_explicitly_pass_config_constants():
    """
    Statically parses edge/pipeline.py AST to verify that calls to decide(),
    aggregate_frame(), and aggregate_cell() pass every required threshold as an
    explicit keyword argument bound to the exact expected config constant name.
    """
    pipeline_path = Path(__file__).resolve().parent.parent / "edge" / "pipeline.py"
    assert pipeline_path.exists(), "edge/pipeline.py does not exist: %s" % pipeline_path

    with open(pipeline_path, "r") as f:
        tree = ast.parse(f.read(), filename=str(pipeline_path))

    calls_found = {fn: [] for fn in EXPECTED_PLUMBING}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in EXPECTED_PLUMBING:
                # Extract keyword argument mappings: kwarg_name -> variable_name
                kwargs = {}
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Name):
                        kwargs[kw.arg] = kw.value.id
                    elif isinstance(kw.value, ast.Constant):
                        kwargs[kw.arg] = kw.value.value
                calls_found[func_name].append((node.lineno, kwargs))

    for fn_name, expected_params in EXPECTED_PLUMBING.items():
        calls = calls_found[fn_name]
        assert len(calls) > 0, (
            "Found no call to %s() in edge/pipeline.py!" % fn_name
        )

        for line_no, kwargs in calls:
            for param_name, expected_const in expected_params.items():
                assert param_name in kwargs, (
                    "Call to %s() at line %d is missing explicit keyword argument '%s'! "
                    "It must be explicitly passed as %s=%s, never falling back to signature default."
                    % (fn_name, line_no, param_name, param_name, expected_const)
                )
                assert kwargs[param_name] == expected_const, (
                    "Call to %s() at line %d passes '%s=%s', but expected '%s=%s' from configs/train_config.py."
                    % (fn_name, line_no, param_name, kwargs[param_name], param_name, expected_const)
                )


def test_runtime_spy_receives_exact_config_values(monkeypatch):
    """
    Dynamically spies on edge.pipeline's calls to decide, aggregate_frame, and aggregate_cell
    during worker thread execution to verify that runtime kwargs contain and equal the
    exact values defined in configs/train_config.py.
    """
    import edge.pipeline as pipeline_mod

    intercepted = {}

    orig_decide = pipeline_mod.decide
    orig_aggregate_frame = pipeline_mod.aggregate_frame
    orig_aggregate_cell = pipeline_mod.aggregate_cell

    def spy_decide(*args, **kwargs):
        intercepted["decide"] = kwargs.copy()
        return orig_decide(*args, **kwargs)

    def spy_aggregate_frame(*args, **kwargs):
        intercepted["aggregate_frame"] = kwargs.copy()
        return orig_aggregate_frame(*args, **kwargs)

    def spy_aggregate_cell(*args, **kwargs):
        intercepted["aggregate_cell"] = kwargs.copy()
        return orig_aggregate_cell(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "decide", spy_decide)
    monkeypatch.setattr(pipeline_mod, "aggregate_frame", spy_aggregate_frame)
    monkeypatch.setattr(pipeline_mod, "aggregate_cell", spy_aggregate_cell)

    in_queue = DropOldestQueue(maxsize=4)
    repo_root = Path(__file__).resolve().parent.parent
    priors = load_log_priors(repo_root)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_jsonl = Path(tmpdir) / "test_events.jsonl"
        test_db = Path(tmpdir) / "test_storage.db"
        worker = DecisionAggregateStoreThread(
            in_queue=in_queue,
            storage=test_db,
            output_jsonl=out_jsonl,
            log_priors=priors,
        )

        # Mock frame item
        tiles = np.zeros((9, 224, 224, 3), dtype=np.uint8)
        boxes = [(0, 0, 10, 10)] * 9
        veg_fractions = [0.8] * 9
        tile_batch = TileBatch(tiles=tiles, boxes=boxes, veg_fractions=veg_fractions, n_valid=9)
        logits = np.zeros((9, NUM_CLASSES), dtype=np.float32)
        gate_metrics = {"blur_score": 200.0, "displacement": 1.0}
        metadata = {"source_image": "test.jpg", "timestamp_utc": "2026-09-15T00:00:00Z"}

        in_queue.put((0, metadata, tile_batch, gate_metrics, logits))
        in_queue.put(None)  # Sentinel to stop worker

        worker.start()
        worker.join(timeout=3.0)

    # Verify decide kwargs
    assert "decide" in intercepted, "decide was not called during thread execution"
    d_kwargs = intercepted["decide"]
    for param, const_name in EXPECTED_PLUMBING["decide"].items():
        assert param in d_kwargs, f"decide() did not receive keyword argument '{param}'"
        expected_val = getattr(train_config, const_name)
        assert d_kwargs[param] == pytest.approx(expected_val), (
            f"decide() received {param}={d_kwargs[param]}, expected {const_name}={expected_val}"
        )

    # Verify aggregate_frame kwargs
    assert "aggregate_frame" in intercepted, "aggregate_frame was not called during thread execution"
    af_kwargs = intercepted["aggregate_frame"]
    for param, const_name in EXPECTED_PLUMBING["aggregate_frame"].items():
        assert param in af_kwargs, f"aggregate_frame() did not receive keyword argument '{param}'"
        expected_val = getattr(train_config, const_name)
        assert af_kwargs[param] == pytest.approx(expected_val), (
            f"aggregate_frame() received {param}={af_kwargs[param]}, expected {const_name}={expected_val}"
        )

    # Verify aggregate_cell kwargs
    assert "aggregate_cell" in intercepted, "aggregate_cell was not called during thread execution"
    ac_kwargs = intercepted["aggregate_cell"]
    for param, const_name in EXPECTED_PLUMBING["aggregate_cell"].items():
        assert param in ac_kwargs, f"aggregate_cell() did not receive keyword argument '{param}'"
        expected_val = getattr(train_config, const_name)
        assert ac_kwargs[param] == pytest.approx(expected_val), (
            f"aggregate_cell() received {param}={ac_kwargs[param]}, expected {const_name}={expected_val}"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
