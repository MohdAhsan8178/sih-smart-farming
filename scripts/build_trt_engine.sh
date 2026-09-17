#!/usr/bin/env bash
# ==============================================================================
# scripts/build_trt_engine.sh
#
# Step 18: Compile static-shape FP16 TensorRT engine on the Jetson Nano (Maxwell SM 5.3).
# Reference: docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 18
#
# CONSTRAINTS & RULES RESPECTED:
# - No --int8: Maxwell (Jetson Nano) has no INT8 tensor cores; FP16 is strictly required.
# - Jetson-only execution: This script guards against execution on macOS or non-Tegra hosts.
# - Max clocks: Configures nvpmodel -m 0 and jetson_clocks prior to trtexec compilation.
# - Piped verbose logging: All trtexec outputs are tee'd to build.log and benchmark.log.
# ==============================================================================

set -euo pipefail

# 1. Platform Guard — Jetson Hardware Verification
if [[ "$(uname -s)" != "Linux" ]] || [[ ! -f /etc/nv_tegra_release && ! -d /usr/src/tensorrt ]]; then
    echo "================================================================================" >&2
    echo "ERROR: Target architecture mismatch!" >&2
    echo "Current host: $(uname -s) $(uname -m)" >&2
    echo "This script compiles a hardware-specific serialized TensorRT engine for NVIDIA" >&2
    echo "Jetson (Maxwell SM 5.3, JetPack 4.6+ / TensorRT 8.2)." >&2
    echo "TensorRT engines are non-portable across architectures and CANNOT be built on macOS." >&2
    echo "Please scp the simplified ONNX model to the Jetson Nano and execute this script there:" >&2
    echo "  scp artifacts/onnx/model_a_sim.onnx nano@<jetson-ip>:~/sih-smart-farming/artifacts/onnx/" >&2
    echo "================================================================================" >&2
    exit 1
fi

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [[ ! -x "$TRTEXEC" ]]; then
    echo "ERROR: trtexec binary not found or not executable at: $TRTEXEC" >&2
    exit 1
fi

# 2. Paths and Parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ONNX_INPUT="${1:-${ROOT_DIR}/artifacts/onnx/model_a_sim.onnx}"
ENGINE_OUTPUT="${2:-${ROOT_DIR}/artifacts/engines/model_a_fp16.engine}"
LOG_DIR="${ROOT_DIR}/artifacts/reports"

mkdir -p "$(dirname "$ENGINE_OUTPUT")"
mkdir -p "$LOG_DIR"

BUILD_LOG="${LOG_DIR}/trt_build.log"
BENCHMARK_LOG="${LOG_DIR}/trt_benchmark.log"

if [[ ! -f "$ONNX_INPUT" ]]; then
    echo "ERROR: Input ONNX model not found at: $ONNX_INPUT" >&2
    exit 1
fi

echo "================================================================================"
echo "STEP 18: TensorRT FP16 Engine Compilation on Jetson Nano"
echo "Input ONNX:      $ONNX_INPUT ($(ls -lh "$ONNX_INPUT" | awk '{print $5}'))"
echo "Output Engine:   $ENGINE_OUTPUT"
echo "Build Log:       $BUILD_LOG"
echo "Benchmark Log:   $BENCHMARK_LOG"
echo "================================================================================"

# 3. Lock High-Performance Clocks
echo "[build_trt_engine] Setting 10W power mode (nvpmodel -m 0) and max clocks (jetson_clocks)..."
if command -v nvpmodel >/dev/null 2>&1; then
    sudo nvpmodel -m 0 || true
fi
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks || true
fi

# 4. Engine Compilation via trtexec
# STRICTLY FP16. NO --int8 (Maxwell lacks INT8 hardware).
echo "[build_trt_engine] Compiling FP16 TensorRT engine with 1024MB workspace..."
"$TRTEXEC" \
    --onnx="$ONNX_INPUT" \
    --saveEngine="$ENGINE_OUTPUT" \
    --fp16 \
    --workspace=1024 \
    --verbose 2>&1 | tee "$BUILD_LOG"

if [[ ! -f "$ENGINE_OUTPUT" || ! -s "$ENGINE_OUTPUT" ]]; then
    echo "ERROR: Engine file was not generated or is empty: $ENGINE_OUTPUT" >&2
    exit 1
fi

echo "[build_trt_engine] Successfully compiled engine: $ENGINE_OUTPUT ($(ls -lh "$ENGINE_OUTPUT" | awk '{print $5}'))"

# 5. Standalone Engine Benchmark
echo "[build_trt_engine] Running 200 iterations benchmark (100 avgRuns)..."
"$TRTEXEC" \
    --loadEngine="$ENGINE_OUTPUT" \
    --iterations=200 \
    --avgRuns=100 2>&1 | tee "$BENCHMARK_LOG"

echo "================================================================================"
echo "[build_trt_engine] Compilation and benchmarking complete."
echo "Benchmark summary written to: $BENCHMARK_LOG"
echo "================================================================================"
