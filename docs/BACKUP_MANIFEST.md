# Production Artifacts Backup Manifest (v1.0)

**Document:** `docs/BACKUP_MANIFEST.md`  
**Purpose:** Authoritative inventory of git-ignored and tracked production artifacts required to deploy, execute, and reproduce the Smart Farming Edge Runtime & Diagnostic Pipeline.  
**Backup Script:** `scripts/backup_artifacts.sh` (copies artifacts and writes `SHA256SUMS`).

---

## 1. Machine Learning Models & Inference Artifacts

| Category | Relative Path | Size | Description |
|---|---|---|---|
| **TensorRT Engine** | `artifacts/engines/model_a_fp16.engine` | 8.52 MB | Model A FP16 TensorRT Engine for Jetson Nano |
| **ONNX Model** | `artifacts/onnx/model_a_sim.onnx` | 12.96 MB | Simplified ONNX Model A (Diagnostic Classifier) |
| **ONNX Model** | `artifacts/onnx/model_a_fused.onnx` | 13.04 MB | Fused ONNX Model A with preprocessing |
| **ONNX Model** | `artifacts/onnx/model_b.onnx` | 5.81 MB | Model B Calibrated Sticky-Trap Classifier |
| **PyTorch Checkpoint** | `artifacts/checkpoints/v3/last.pt` | 52.65 MB | Model A Final Training Checkpoint (v3 dataset) |
| **PyTorch Checkpoint** | `artifacts/checkpoints/v3/stage1.pt` | 52.65 MB | Model A Stage 1 Backbone Feature Checkpoint |
| **PyTorch Checkpoint** | `artifacts/checkpoints/model_b_best.pt` | 5.93 MB | Model B Best Training Checkpoint |
| **PyTorch Checkpoint** | `artifacts/checkpoints/model_b_v4_honest_baseline.pt` | 1.14 MB | Model B Honest Baseline Checkpoint (v4) |

---

## 2. Calibration & Metrics Deliverables

| Relative Path | Size | Description |
|---|---|---|
| `artifacts/reports/ood_metrics.json` | 1.36 KB | Decision thresholds & OOD calibration metrics (`TAU_ENERGY`, `T_CAL`, etc.) |
| `artifacts/reports/model_a_cross_source_reliability.json` | 3.82 KB | Cross-source reliability classification metrics per class |
| `artifacts/reports/eval_indist.json` | 4.65 KB | In-distribution evaluation metrics and confusion statistics |
| `artifacts/reports/eval_crossdomain.json` | 6.25 KB | Cross-domain evaluation metrics |
| `artifacts/reports/per_class_recall.csv` | 1.03 KB | Per-class recall table for all 29 classes |
| `artifacts/reports/bias_audit.json` | 1.41 KB | Perimeter background bias audit metrics |
| `artifacts/reports/threshold_sweep.png` | 383.10 KB | 4-panel threshold calibration visualization |
| `artifacts/reports/trt_benchmark.txt` | 4.30 KB | Latency and throughput benchmarking on Jetson Nano |

---

## 3. Split Manifests & Taxonomy

| Relative Path | Size | Description |
|---|---|---|
| `splits/all_images.csv` | 2.97 MB | Master unified dataset image index |
| `splits/train.csv` | 1.62 MB | Stratified training split |
| `splits/val.csv` | 206.55 KB | Validation split for threshold calibration |
| `splits/test_indist.csv` | 208.05 KB | In-distribution test split |
| `splits/test_crossdomain.csv` | 965.39 KB | Held-out cross-domain source test split |
| `splits/class_weights.json` | 2.38 KB | Normalized inverse sqrt class weights |
| `splits/class_mapping.csv` | 4.50 KB | Authoritative class label to index mapping |
| `splits/model_b_manifest_v4.csv` | 5.47 MB | Model B sticky-trap patch dataset manifest (v4) |
| `splits/openset_categories.csv` | 1.01 KB | Disjoint open-set category registry |

---

## 4. Backup Execution Procedure

To create a verifiable offline archive of all production artifacts:

```bash
# Usage: ./scripts/backup_artifacts.sh <destination_dir>
./scripts/backup_artifacts.sh /path/to/backup/destination
```

The script copies all listed files, preserves relative directory hierarchy, and writes a cryptographic `SHA256SUMS` file at the root of the destination.
