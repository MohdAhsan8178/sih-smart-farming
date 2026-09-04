# Edge-Deployed Smart Farming Assistant — SIH 2026

An edge-deployed smart agricultural intelligence platform designed for high-resolution crop disease classification, pest counting via smart sticky-trap nodes, canopy water-stress assessment using thermal radiometry (CWSI), nutrient analysis, and local advisory generation under intermittent field connectivity. Engineered specifically for deployment on an NVIDIA Jetson Nano edge processor aboard agricultural scout rovers/drones alongside distributed ESP32-CAM gateway nodes.

## Strict Two-Environment Architecture

To prevent runtime failures and dependency conflicts on edge hardware, this codebase strictly isolates training from edge execution:

1. **Training Environment (`train/`)**:
   - Modern Python (3.10+) running on workstation / cloud GPU (Colab).
   - PyTorch 2.x, `timm`, Albumentations, ONNX export tools.
   - Used for dataset curation, bias audits, fine-tuning backbones, and exporting static FP16 ONNX models.

2. **Edge Runtime Environment (`edge/`)**:
   - Python 3.6-compatible running on Jetson Nano 4GB (Maxwell GM20B, compute capability 5.3, JetPack 4.6.4, TensorRT 8.2).
   - Uses pre-built TensorRT FP16 engines (`.engine`) and OpenCV with locked camera gains.
   - **NO PyTorch**, no INT8 (unsupported by Maxwell GPU), no Python 3.7+ syntax (no f-string `=`, no walrus, no `dataclasses`, no `multiprocessing.shared_memory`).

3. **Pre-tested Core Logic (`core/` & `configs/`)**:
   - Spatial aggregation, open-set energy-based out-of-distribution rejection, thermal canopy extraction, and dual-threshold insect blob segmentation.

## Quickstart

### Development & Training Environment (Host Machine)
```bash
# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt

# Run regression tests
pytest tests/ -v
```

### Edge Environment (Jetson Nano)
```bash
# Verify system environment on JetPack 4.6.4
python3 --version  # Python 3.6.9
nvcc --version     # CUDA 10.2

# Install edge dependencies
pip3 install -r requirements-edge.txt

# Build TensorRT engine and run edge pipeline
bash edge/build_engine.sh
python3 edge/pipeline.py --config configs/field_demo.yaml
```

## Implementation Plan & Documentation
Detailed step-by-step implementation milestones, architecture reports, and hardware audit logs are located in the `docs/` directory:
- [Implementation & Execution Plan](docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md)
- [Build Checklist & File Inventory](docs/BUILD_CHECKLIST_1.md)
- [AI Models & Training Handbook](docs/AI_Handbook_4.md)
- [System Architecture & Agronomy Report](docs/SIH_Smart_Farming_AI_Report_4.md)
