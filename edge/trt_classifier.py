#!/usr/bin/env python3
"""
Step 19: TensorRT Runtime Wrapper for Edge Deployment on NVIDIA Jetson Nano.

Implements TRTClassifier adhering strictly to Rule R9:
- NO `import pycuda.autoinit` anywhere.
- Explicit CUDA context creation and thread binding.
- Pagelocked (pinned) host memory allocation for (B, 224, 224, 3) uint8 BGR.
- Raw unadjusted logit outputs (preserves energy OOD score semantics).
- Chunked infer_all() ensuring arbitrary tile counts are never truncated.
- Safe teardown: deallocates device/host buffers before context detachment.
- Python 3.6+ compatible (no walrus operators, no dataclasses, no f-string debugging).

Reference: docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 19 & Rule R9
"""

import argparse
import gc
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.classes import NUM_CLASSES
from configs.train_config import ENGINE_BATCH, IMAGE_SIZE

# Lazy import of TensorRT and PyCUDA to allow import/linting on non-Jetson hosts
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    trt = None
    HAS_TRT = False

try:
    import pycuda.driver as cuda
    HAS_PYCUDA = True
except ImportError:
    cuda = None
    HAS_PYCUDA = False


class TRTClassifier(object):
    """
    TensorRT inference classifier for Jetson Nano.
    Accepts raw uint8 BGR tiles (batch, 224, 224, 3) and outputs raw logits (batch, 29).
    """
    def __init__(
        self,
        engine_path: Union[str, Path],
        batch: int = ENGINE_BATCH,
        size: int = IMAGE_SIZE,
        num_classes: Optional[int] = None,
        device_id: int = 0,
    ):
        if not HAS_TRT or not HAS_PYCUDA:
            raise RuntimeError(
                "TRTClassifier requires 'tensorrt' and 'pycuda.driver' installed on "
                "an NVIDIA Linux environment (e.g. Jetson Nano). Missing: "
                + ("tensorrt " if not HAS_TRT else "")
                + ("pycuda " if not HAS_PYCUDA else "")
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError("TensorRT engine file not found: %s" % self.engine_path)

        self.batch = int(batch)
        self.size = int(size)
        self.nc = int(num_classes if num_classes is not None else NUM_CLASSES)
        self.device_id = int(device_id)
        self.is_closed = False

        # 1. Explicit CUDA context management (RULE R9: NEVER use pycuda.autoinit)
        cuda.init()
        self.device = cuda.Device(self.device_id)
        self.ctx = self.device.make_context()

        try:
            # 2. Load serialized engine
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(str(self.engine_path), "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine from %s" % self.engine_path)

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            # 3. Pinned (pagelocked) host buffers — float32 BGR in, float32 logits out
            # Engine input is float32 (compatible with TensorRT 8.2 nvonnxparser);
            # caller passes raw uint8 BGR, cast occurs into pinned host memory during assignment.
            self.h_in = cuda.pagelocked_empty((self.batch, self.size, self.size, 3), np.float32)
            self.h_out = cuda.pagelocked_empty((self.batch, self.nc), np.float32)

            # 4. Device buffers on GPU
            self.d_in = cuda.mem_alloc(self.h_in.nbytes)
            self.d_out = cuda.mem_alloc(self.h_out.nbytes)
            self.bindings = [int(self.d_in), int(self.d_out)]
            self.stream = cuda.Stream()

        except Exception:
            self.close()
            raise
        finally:
            self.ctx.pop()

    def infer(self, bgr_tiles: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Executes inference on a single batch of tiles (<= self.batch).
        Returns RAW UNADJUSTED LOGITS (N, NUM_CLASSES), float32.
        Does NOT return probabilities (probabilities destroy energy score semantics).
        """
        if self.is_closed:
            raise RuntimeError("TRTClassifier has already been closed.")

        n_tiles = len(bgr_tiles)
        if n_tiles == 0:
            return np.empty((0, self.nc), dtype=np.float32)
        if n_tiles > self.batch:
            raise ValueError("infer() received %d tiles, exceeding batch limit %d. Use infer_all()." % (n_tiles, self.batch))

        self.ctx.push()
        try:
            # Copy input tiles into pagelocked memory
            if isinstance(bgr_tiles, np.ndarray):
                self.h_in[:n_tiles] = bgr_tiles[:n_tiles]
            else:
                for i in range(n_tiles):
                    self.h_in[i] = bgr_tiles[i]

            # Async copy Host -> Device
            cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)

            # Asynchronous kernel execution
            if hasattr(self.context, "execute_async_v2"):
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            else:
                self.context.execute_async(batch_size=self.batch, bindings=self.bindings, stream_handle=self.stream.handle)

            # Async copy Device -> Host
            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()

            # Return exact slice of unadjusted logits
            return self.h_out[:n_tiles].copy()
        finally:
            self.ctx.pop()

    def infer_all(self, bgr_tiles: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Chunk-loop inference across arbitrary number of tiles.
        Guarantees that an unexpected tile count can NEVER silently truncate tiles.
        Handles padding for the terminal chunk if necessary.
        """
        n_total = len(bgr_tiles)
        if n_total == 0:
            return np.empty((0, self.nc), dtype=np.float32)

        results = []
        for i in range(0, n_total, self.batch):
            chunk = bgr_tiles[i:i + self.batch]
            actual_len = len(chunk)

            if actual_len == self.batch:
                out = self.infer(chunk)
            else:
                # Pad chunk to self.batch to satisfy static engine constraint
                if isinstance(chunk, np.ndarray):
                    padded = np.zeros((self.batch, self.size, self.size, 3), dtype=np.uint8)
                    padded[:actual_len] = chunk
                else:
                    padded = list(chunk)
                    dummy = np.zeros((self.size, self.size, 3), dtype=np.uint8)
                    while len(padded) < self.batch:
                        padded.append(dummy)
                out = self.infer(padded)[:actual_len]

            results.append(out)

        return np.concatenate(results, axis=0)

    def close(self) -> None:
        """
        Safe teardown sequence:
        Frees d_in, d_out, stream, context, engine, then gc.collect().
        Context is detached ONLY after all bound GPU memory allocations are freed,
        preventing PyCUDA LogicError during restart or exit.
        """
        if self.is_closed:
            return

        if hasattr(self, "ctx") and self.ctx is not None:
            try:
                self.ctx.push()
            except Exception:
                pass

        if hasattr(self, "d_in") and self.d_in is not None:
            try:
                self.d_in.free()
            except Exception:
                pass
            self.d_in = None

        if hasattr(self, "d_out") and self.d_out is not None:
            try:
                self.d_out.free()
            except Exception:
                pass
            self.d_out = None

        if hasattr(self, "stream") and self.stream is not None:
            self.stream = None

        if hasattr(self, "context") and self.context is not None:
            del self.context
            self.context = None

        if hasattr(self, "engine") and self.engine is not None:
            del self.engine
            self.engine = None

        gc.collect()

        if hasattr(self, "ctx") and self.ctx is not None:
            try:
                self.ctx.pop()
                self.ctx.detach()
            except Exception:
                pass
            self.ctx = None

        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not getattr(self, "is_closed", True):
            self.close()


def main():
    parser = argparse.ArgumentParser(description="TensorRT Edge Classifier Self-Test (Jetson Nano).")
    parser.add_argument("--engine", type=str, default="artifacts/engines/model_a_fp16.engine", help="Path to .engine file")
    parser.add_argument("--batch", type=int, default=ENGINE_BATCH, help="Batch size")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Tile size")
    parser.add_argument("--selftest", action="store_true", default=False, help="Run timing and shape self-test")
    args = parser.parse_args()

    if not HAS_TRT or not HAS_PYCUDA:
        print("ERROR: TensorRT / PyCUDA not available on this platform (%s)." % sys.platform, file=sys.stderr)
        print("This self-test must be executed directly on the NVIDIA Jetson Nano.", file=sys.stderr)
        sys.exit(1)

    print("================================================================================")
    print("TRTClassifier Self-Test on Jetson Nano")
    print("Engine: %s | Batch: %d | Size: %d" % (args.engine, args.batch, args.size))
    print("================================================================================")

    clf = TRTClassifier(engine_path=args.engine, batch=args.batch, size=args.size)
    try:
        # Generate dummy batch
        dummy_tiles = np.random.randint(0, 256, (args.batch, args.size, args.size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            _ = clf.infer(dummy_tiles)

        # Timing loop
        iterations = 50
        t0 = time.time()
        for _ in range(iterations):
            out = clf.infer(dummy_tiles)
        total_time = time.time() - t0
        latency_ms = (total_time / iterations) * 1000.0

        print("[selftest] Output shape: %s (dtype: %s)" % (out.shape, out.dtype))
        print("[selftest] Mean batch latency: %.2f ms (over %d iterations)" % (latency_ms, iterations))
        assert out.shape == (args.batch, NUM_CLASSES), "Shape mismatch: expected (%d, %d), got %s" % (args.batch, NUM_CLASSES, out.shape)

        # Test infer_all with non-multiple tile count (e.g. 14 tiles)
        test_tiles = [np.random.randint(0, 256, (args.size, args.size, 3), dtype=np.uint8) for _ in range(14)]
        out_all = clf.infer_all(test_tiles)
        print("[selftest] infer_all(14 tiles) output shape: %s" % (out_all.shape,))
        assert out_all.shape == (14, NUM_CLASSES), "infer_all truncated tiles! Got shape %s" % (out_all.shape,)

        print("================================================================================")
        print("[selftest] ALL CHECKS PASSED. Exiting cleanly.")
        print("================================================================================")
    finally:
        clf.close()


if __name__ == "__main__":
    main()
