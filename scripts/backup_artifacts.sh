#!/usr/bin/env bash
set -euo pipefail

# scripts/backup_artifacts.sh — Copy production artifacts and generate SHA256SUMS.
# Usage: ./scripts/backup_artifacts.sh <destination_dir>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <destination_dir>" >&2
    exit 1
fi

DEST_DIR="$1"
mkdir -p "$DEST_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Explicit list of production files to back up
FILES=(
    "artifacts/engines/model_a_fp16.engine"
    "artifacts/onnx/model_a_sim.onnx"
    "artifacts/onnx/model_a_fused.onnx"
    "artifacts/onnx/model_b.onnx"
    "artifacts/checkpoints/v3/last.pt"
    "artifacts/checkpoints/v3/stage1.pt"
    "artifacts/checkpoints/model_b_best.pt"
    "artifacts/checkpoints/model_b_v4_honest_baseline.pt"
    "artifacts/reports/ood_metrics.json"
    "artifacts/reports/model_a_cross_source_reliability.json"
    "artifacts/reports/eval_indist.json"
    "artifacts/reports/eval_crossdomain.json"
    "artifacts/reports/per_class_recall.csv"
    "artifacts/reports/bias_audit.json"
    "artifacts/reports/threshold_sweep.png"
    "artifacts/reports/trt_benchmark.txt"
    "splits/all_images.csv"
    "splits/train.csv"
    "splits/val.csv"
    "splits/test_indist.csv"
    "splits/test_crossdomain.csv"
    "splits/class_weights.json"
    "splits/class_mapping.csv"
    "splits/model_b_manifest_v4.csv"
    "splits/openset_categories.csv"
)

echo "Starting backup to $DEST_DIR..."
COPIED_COUNT=0

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        target="$DEST_DIR/$file"
        mkdir -p "$(dirname "$target")"
        cp "$file" "$target"
        COPIED_COUNT=$((COPIED_COUNT + 1))
        echo "  [OK] Copied $file"
    else
        echo "  [WARN] File not found: $file (skipping)" >&2
    fi
done

echo "Computing SHA256 checksums in $DEST_DIR..."
(
    cd "$DEST_DIR"
    find artifacts splits -type f | sort | while read -r f; do
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum "$f"
        elif command -v shasum >/dev/null 2>&1; then
            shasum -a 256 "$f"
        fi
    done > SHA256SUMS
)

echo "Backup complete. Successfully archived $COPIED_COUNT files to $DEST_DIR."
echo "Checksums written to $DEST_DIR/SHA256SUMS."
