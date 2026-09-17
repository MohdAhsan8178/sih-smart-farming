# Jetson Nano On-Device Deployment & Verification Runbook

**Document ID:** `docs/NANO_DEPLOYMENT_RUNBOOK.md`  
**Target Platform:** NVIDIA Jetson Nano Developer Kit (4GB, B01/A02)  
**Target OS:** JetPack 4.6.4 (L4T R32.7.4) / JetPack 4.6.1 (L4T R32.7.1) — Ubuntu 18.04 LTS (aarch64)  
**Target Python:** Python 3.6.9 (System Python, `/usr/bin/python3`)  
**Target Accelerators:** NVIDIA Maxwell GPU (128 CUDA cores), CUDA 10.2, TensorRT 8.2.1.9  
**Scope:** Model A Handheld Pod Pipeline (`edge/pipeline.py`), Edge SQLite Storage (`edge/storage.py`), Rules Engine (`edge/rules_engine.py`), RGB Vegetation Indices (`core/indices.py`), Phenological Growth Stage Engine (`core/growth_stage.py`), and Offline Gateway API Server (`gateway/server.py`).

---

## 1. Prerequisites & Target Architecture

> [!IMPORTANT]
> **DO NOT INSTALL PIP VERSIONS OF OPENCV OR TENSORRT ON THE NANO.**  
> OpenCV (`4.1.1` with GStreamer/V4L2) and TensorRT (`8.2.1.9`) are pre-installed by JetPack into `/usr/lib/python3.6/dist-packages/`. Overwriting them via `pip install opencv-python` or `pip install tensorrt` will break system CUDA bindings and GStreamer hardware acceleration pipelines.

### Physical & Power Configuration
- **Power Supply:** 5V / 4A DC barrel jack (J41 jumper capped). **DO NOT** run TensorRT inference using Micro-USB power (2A limit triggers brownout resets under GPU load).
- **Power Mode:** Max performance 10W mode (MAXN):
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```
- **IP Address:** Jetson Nano static hotspot IP is `192.168.4.1` (or DHCP assigned on local LAN, e.g., `nvidia.local`).
- **Target Working Directory:** `/home/nvidia/sih-smart-farming/`

---

## 2. Exact Copy Manifests

To preserve the Nano's limited eMMC storage and prevent binary conflicts, **do not copy the full laptop workspace**. Copy only the deployment files.

### 2.1 File & Directory Manifest

| Path on Laptop | Destination on Nano | Size | Purpose |
| :--- | :--- | :--- | :--- |
| `edge/` | `~/sih-smart-farming/edge/` | ~160 KB | Pipeline (`pipeline.py`), storage (`storage.py`), rules engine (`rules_engine.py`), frame gate (`frame_gate.py`), tiler (`tiler.py`), TRT classifier (`trt_classifier.py`) |
| `gateway/` | `~/sih-smart-farming/gateway/` | ~35 KB | Offline HTTP API server (`server.py`) |
| `core/` | `~/sih-smart-farming/core/` | ~130 KB | Growth stage phenology (`growth_stage.py`), RGB vegetation indices (`indices.py`), rejection & energy (`rejection.py`), spatial/temporal aggregation (`aggregate.py`) |
| `configs/` | `~/sih-smart-farming/configs/` | ~40 KB | Class taxonomy (`classes.py`), paths (`paths.py`), calibration thresholds & default cycles (`train_config.py`) |
| `artifacts/engines/model_a_fp16.engine` | `~/sih-smart-farming/artifacts/engines/` | 8.5 MB | Calibrated FP16 TensorRT engine (Batch 9, 224x224x3) |
| `artifacts/onnx/model_b.onnx` | `~/sih-smart-farming/artifacts/onnx/` | 6.1 MB | Model B Trap Pest Classifier (OpSet 13, 64x64x3) |
| `test_video_from_dataset_images.mp4` | `~/sih-smart-farming/` | ~2.5 MB | Multi-class validation video clip for offline execution |
| `requirements-edge.txt` | `~/sih-smart-farming/` | 132 B | Edge runtime dependencies (`numpy==1.19.5`, `pycuda==2020.1`, `Pillow==8.4.0`) |
| `ans_for_vitthal.md` | `~/sih-smart-farming/` | ~20 KB | Mobile app API contract reference |
| `docs/TEMPLATE_ID_REGISTRY.md` | `~/sih-smart-farming/docs/` | ~40 KB | Master template registry and localization contract |
| `docs/sources/` (Markdown only) | `~/sih-smart-farming/docs/sources/` | ~260 KB | Agronomic source manifests & FAO-56 stage transcripts (`SOURCES_MANIFEST.md`, `fao56_*.md`, `dppqs_*.md`) |
| `docs/sources/*.pdf` (Optional) | `~/sih-smart-farming/docs/sources/` | 6.76 MB | 4 primary regulatory PDF publications (CIB&RC & DPPQS). See §2.2 for copy policy. |

### 2.2 Primary Regulatory PDFs Copy Policy (`docs/sources/`)

The `docs/sources/` directory contains 4 official PDF publications:
- `cibrc_fungicides_2026.pdf` (1.35 MB)
- `cibrc_insecticides_2026.pdf` (1.68 MB)
- `dppqs_ipm_rice.pdf` (1.52 MB)
- `dppqs_ipm_sugarcane.pdf` (2.19 MB)
- **Combined PDF Size:** 6.76 MB (Total directory with markdown: 7.02 MB)

**Runtime Dependency Analysis:**
- The Python modules (`edge/rules_engine.py` and `core/growth_stage.py`) have all web-verified doses, trade formulations, active ingredients, and FAO-56 stage boundary thresholds compiled directly into Python source code.
- **The edge runtime does NOT open or parse these PDFs at runtime.**

**Copy Decision:**
1. **Recommended for Field Demos / Evaluation Audits:** Copy `docs/sources/` completely (7.02 MB). At 7 MB, it consumes only **~0.4%** of the Nano's 1.5 GB safety buffer, allowing agronomists and evaluation judges to verify original regulatory citations offline directly from the device.
2. **Minimalist / Low-Write Option:** If flash write cycles or absolute minimalism is desired, omit `*.pdf` and copy only `docs/sources/*.md` (~260 KB). The pipeline and gateway run identically under both options.

### 2.3 Excluded Files (DO NOT COPY)
- `.git/` (hundreds of MBs of Git history)
- `.venv/` (macOS Darwin binaries will fail on Linux aarch64)
- `data/raw/`, `data/downloads_v3/`, `data/processed/` (multi-GB training images)
- `artifacts/checkpoints/` (PyTorch `.pt` model weights, 150+ MB)
- `__pycache__/`, `.pytest_cache/`, `.DS_Store`

### 2.4 Exact `rsync` Command (Run from Laptop Terminal)

From the laptop repo root (`/Users/mohdahsan/Downloads/SIH/sih-smart-farming`):

```bash
# Define Nano SSH destination (adjust IP if not nvidia.local)
NANO_USER="nvidia"
NANO_HOST="192.168.4.1" # or "nvidia.local"
NANO_DEST="/home/nvidia/sih-smart-farming"

# Create target directories on the Nano
ssh ${NANO_USER}@${NANO_HOST} "mkdir -p ${NANO_DEST}/artifacts/engines ${NANO_DEST}/artifacts/reports ${NANO_DEST}/data ${NANO_DEST}/docs/sources"

# Synchronize edge codebase, gateway, core algorithms, and configs
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  edge/ ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/edge/

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  gateway/ ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/gateway/

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  core/ ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/core/

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  configs/ ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/configs/

# Synchronize contracts and documentation
rsync -avz --progress \
  ans_for_vitthal.md requirements-edge.txt \
  ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/

rsync -avz --progress \
  docs/TEMPLATE_ID_REGISTRY.md \
  ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/docs/

# Synchronize primary agronomic sources (including PDFs, total ~7.0 MB)
rsync -avz --progress \
  docs/sources/ \
  ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/docs/sources/

# Synchronize TensorRT engine and validation video clip
rsync -avz --progress \
  artifacts/engines/model_a_fp16.engine ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/artifacts/engines/

rsync -avz --progress \
  test_video_from_dataset_images.mp4 \
  ${NANO_USER}@${NANO_HOST}:${NANO_DEST}/
```

*(Note: If choosing the minimalist option without PDFs, add `--exclude '*.pdf'` to the `docs/sources/` rsync command).*

### 2.5 Model B Sticky-Trap Deployment & ONNX Runtime Wheel Policy

Sticky-trap pest classification (Model B) has fundamentally different operational characteristics than Model A:
1. **Separation of Execution Contexts:** Model A runs continuously at 30 fps on the real-time handheld video stream (`edge/pipeline.py`) using TensorRT. In contrast, Model B runs intermittently (once daily or per card inspection) on sticky-trap images captured by station nodes (ESP32-CAM) via `core/trap_segmentation.py` (`classify_trap_blobs`).
2. **Runtime Engine Recommendation:**
   - **Recommendation:** Deploy Model B using **ONNX Runtime (Python 3.6 aarch64 wheel)** rather than compiling a TensorRT engine.
   - **Rationale:** Sticky cards yield dynamic batches of arbitrary numbers of segmented insect crops (ranging from 0 to 500+ blobs per card). ONNX Runtime executes dynamic batch sizes out of the box with zero compilation overhead. Building a dynamic TensorRT profile for sub-millisecond throughput on a task evaluated once per day introduces needless driver and memory profile fragility without operational benefit.
3. **Official JetPack 4.6 Wheel for Python 3.6 aarch64:**
   - Pre-built wheel matching JetPack 4.6.4 / L4T R32.7.4:
     ```bash
     # Install official Microsoft/NVIDIA aarch64 Python 3.6 wheel supporting OpSet 13:
     pip3 install onnxruntime-1.10.0-cp36-cp36m-linux_aarch64.whl
     ```
   - Model B ONNX artifact: `artifacts/onnx/model_b.onnx` (OpSet 13, 6.1 MB, outputs raw logits; temperature scaling is performed in Python).

---

## 3. Exhaustive Python 3.6 Compatibility Audit

The Jetson Nano runs **Python 3.6.9**. Python 3.6 has strict syntax and standard library limitations. An automated AST analysis and test guard ([`tests/test_python36_ast_guard.py`](file:///Users/mohdahsan/Downloads/SIH/sih-smart-farming/tests/test_python36_ast_guard.py)) validates every production module.

### 3.1 Syntax Hazard Audit Matrix

| Language Feature | Minimum Version | Present in Deployment Files? | Resolution / Evidence |
| :--- | :--- | :--- | :--- |
| **Walrus Operator (`:=`)** | Python 3.8 | **None (0 occurrences)** | Standard assignment and explicit `while` loop checks used. |
| **Debug f-strings (`f"{x=}"`)** | Python 3.8 | **None (0 occurrences)** | Traditional format strings `%s` or `f"{x}"` used. |
| **Positional-only args (`/`)** | Python 3.8 | **None (0 occurrences)** | Standard function signatures across all modules. |
| **Pattern Matching (`match/case`)** | Python 3.10 | **None (0 occurrences)** | Traditional `if/elif/else` blocks used. |
| **PEP 585 Generics (`list[...]`, `dict[...]`)** | Python 3.9 | **None (0 occurrences)** | Standard `typing.List`, `typing.Dict`, `typing.Tuple` used throughout. |
| **PEP 604 Union Types (`X \| Y`)** | Python 3.10 | **None (0 occurrences)** | Standard `typing.Union[X, Y]` and `typing.Optional[X]` used. |
| **Dataclasses (`@dataclass`)** | Python 3.7 | **None (0 occurrences)** | Plain Python objects and `collections.namedtuple` used (e.g. `TileBatch`). |
| **Subprocess (`text=True`, `capture_output=True`)**| Python 3.7 | **None (0 occurrences)** | Subprocess calls use `universal_newlines=True` or `stdout=subprocess.PIPE`. |
| **`datetime.fromisoformat()`** | Python 3.7 | **Resolved (0 occurrences)** | Custom helper `get_utc_iso_now()` with `strftime("%Y-%m-%dT%H:%M:%SZ")`. |
| **`http.server.ThreadingHTTPServer`** | Python 3.7 | **Resolved (0 occurrences)** | Implemented via `class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer)` in `gateway/server.py`. |
| **Dict Ordering Guarantees** | Python 3.7 spec | **Safe** | CPython 3.6.9 maintains insertion order as an implementation detail. Critical ordering uses `collections.deque` and SQLite monotonic `seq`. |

### 3.2 Audit Verification Command (Run on Nano over SSH)
Verify that all deployment files parse cleanly under the Nano's Python 3.6 AST parser:
```bash
python3 -c "
import ast, glob
files = glob.glob('edge/**/*.py', recursive=True) + glob.glob('gateway/**/*.py', recursive=True) + glob.glob('core/**/*.py', recursive=True) + glob.glob('configs/**/*.py', recursive=True)
for f in files:
    with open(f, 'r') as fp:
        ast.parse(fp.read(), filename=f)
print('Successfully parsed %d deployment files under Python 3.6 AST!' % len(files))
"
```

---

## 4. Nano Environment Verification & Dependencies

Log in to the Jetson Nano via SSH:
```bash
ssh nvidia@192.168.4.1
cd /home/nvidia/sih-smart-farming
```

### 4.1 System & Hardware Checks
```bash
# 1. Verify OS and Python version
python3 -V
# Expected: Python 3.6.9

# 2. Verify JetPack TensorRT and OpenCV bindings
python3 -c "import tensorrt as trt; print('TensorRT Version:', trt.__version__)"
# Expected: TensorRT Version: 8.2.1.9 (or 8.2.x)

python3 -c "import cv2; print('OpenCV Version:', cv2.__version__)"
# Expected: OpenCV Version: 4.1.1

# 3. Verify CUDA runtime and Maxwell GPU
python3 -c "import pycuda.driver as cuda; cuda.init(); print('Device:', cuda.Device(0).name())"
# Expected: Device: NVIDIA Tegra X1
```

### 4.2 Edge Dependencies Installation
If `pycuda` or `Pillow` are not already installed for user `nvidia`:
```bash
pip3 install --user -r requirements-edge.txt
```
> [!CAUTION]
> If `pip3` attempts to install `opencv-python` or `tensorrt`, abort immediately (`Ctrl+C`). Check `requirements-edge.txt` — it must contain only `numpy==1.19.5`, `pycuda==2020.1`, `Pillow==8.4.0`, and `requests==2.27.1`.

---

## 5. eMMC Headroom & Storage Checks (Pre-Execution)

The Jetson Nano eMMC is typically 16 GB, with 12–14 GB consumed by the OS and JetPack components.

### 5.1 Free Space Rule & Itemized Storage Ledger
> [!IMPORTANT]
> **The 1.5 GB Headroom Rule:** The Nano requires at least **1.5 GB to 2.0 GB** of free eMMC space. Dropping below 500 MB can freeze the OS, stall the Linux memory swapper (ZRAM), or corrupt SQLite WAL journal commits.

#### Complete SIH Application Disk Footprint Ledger:
| Component | Disk Allocation | Notes |
| :--- | :--- | :--- |
| **Python Codebase** (`edge/`, `gateway/`, `core/`, `configs/`) | ~450 KB | Plain Python source files |
| **Model A TensorRT Engine** (`model_a_fp16.engine`) | 8.5 MB | Compiled FP16 Maxwell engine |
| **Test Video Clip** (`test_video_from_dataset_images.mp4`) | 2.5 MB | Offline evaluation video |
| **Contracts & Registries** (`ans_for_vitthal.md`, `TEMPLATE_ID_REGISTRY.md`) | ~60 KB | Contract specifications |
| **Primary Regulatory Documents** (`docs/sources/`) | 7.02 MB | 4 PDFs (6.76 MB) + Markdown manifests (0.26 MB) |
| **Edge SQLite Database Ceiling** (`data/edge.db` + WAL) | **121 MB** | Strictly enforced by 50,000 frame / 500 advisory FIFO pruning |
| **TOTAL WORST-CASE DISK FOOTPRINT** | **~140 MB** | **< 9.4% of the 1.5 GB safety buffer** |

Even with all primary PDFs and SQLite at maximum retained capacity (50,000 frames), the system leaves **> 1.35 GB** of completely unallocated headroom.

Run the headroom check:
```bash
df -h /
```
- **Minimum required available space:** `>= 1.5 GB`.
- If available space is `< 1.0 GB`, clean system caches:
  ```bash
  sudo apt clean
  rm -rf ~/.cache/pip
  rm -rf /tmp/*
  ```

---

## 5. Configuring the 40-Pin Header UART for NEO-6M GPS (`/dev/ttyTHS1`)

**Status:** UNVERIFIED ON HARDWARE (Bench verified in simulation; hardware validation pending dongle/wiring test on physical Nano).

The NEO-6M GPS receiver connects directly to the Jetson Nano 40-pin expansion header UART (Pin 8 = UART2 TXD -> GPS RX, Pin 10 = UART2 RXD -> GPS TX, Pin 2 = 5V VCC, Pin 6 = GND). The serial device node is `/dev/ttyTHS1` operating at 9600 baud.

> [!WARNING]
> On default JetPack 4.6 installations, `/dev/ttyTHS1` is bound to the `nvgetty.service` system serial console. If `nvgetty` is running, it competes for incoming serial bytes, causing NMEA sentence fragmentation and framing errors.

Execute the following commands on the Jetson Nano to free `/dev/ttyTHS1` and set permissions:

```bash
# 1. Stop and disable the nvgetty serial console service
sudo systemctl stop nvgetty.service
sudo systemctl disable nvgetty.service

# 2. Add current user to dialout group for non-root serial port access
sudo usermod -a -G dialout $USER

# 3. Add udev rule for persistent device permissions
sudo tee /etc/udev/rules.d/99-nv-gps.rules << 'EOF'
KERNEL=="ttyTHS1", MODE="0666", GROUP="dialout"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# 4. Install pyserial (listed in requirements-edge.txt)
pip3 install pyserial==3.5

# 5. Reboot the Nano to ensure service and group changes take effect
sudo reboot
```

---

## 6. Running `edge/pipeline.py` on Jetson Nano (Real Inference)

Run the edge processing pipeline **WITHOUT** `--dry-run`, executing real TensorRT inference on the Maxwell GPU alongside real vegetation indices and phenology lookup.

### 6.1 Set Clock & Power Mode
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 6.2 Execute Pipeline on Stored Test Video with Phenology Inputs

#### Case A: Standard Rice Crop (Day 75 post-planting, default 150d cycle)
```bash
python3 edge/pipeline.py \
  --source test_video_from_dataset_images.mp4 \
  --engine artifacts/engines/model_a_fp16.engine \
  --days-since-planting 75 \
  --db-path data/edge.db \
  --report \
  --output artifacts/reports/pipeline_nano_events.jsonl
```

#### Case B: Short-Duration Cultivar Override (Day 45 post-planting, 110d cycle override)
```bash
python3 edge/pipeline.py \
  --source test_video_from_dataset_images.mp4 \
  --engine artifacts/engines/model_a_fp16.engine \
  --days-since-planting 45 \
  --total-cycle-days 110 \
  --db-path data/edge.db \
  --report \
  --output artifacts/reports/pipeline_nano_events.jsonl
```

#### Case C: Missing Planting Date (Normal operational state before farmer input)
```bash
python3 edge/pipeline.py \
  --source test_video_from_dataset_images.mp4 \
  --engine artifacts/engines/model_a_fp16.engine \
  --db-path data/edge.db \
  --report \
  --output artifacts/reports/pipeline_nano_events.jsonl
```

### 6.3 Execute Pipeline on Live Pod Camera (USB / CSI V4L2)
```bash
# Assuming USB or CSI camera enumerates at /dev/video0
python3 edge/pipeline.py \
  --source 0 \
  --engine artifacts/engines/model_a_fp16.engine \
  --days-since-planting 75 \
  --db-path data/edge.db \
  --report
```

### 6.4 Revised Pipeline Throughput & Latency Profile

Adding the full band-addressed vegetation indices suite ([`core/indices.py`](file:///Users/mohdahsan/Downloads/SIH/sih-smart-farming/core/indices.py)) into Thread 2 (`GateTileThread`) added ~5–12 ms of CPU processing per accepted frame on the Cortex-A57 CPU cores.

#### Detailed Stage Breakdown per Scene (9 Tiles):
| Thread / Stage | Hardware | Time per Scene | Notes |
| :--- | :--- | :--- | :--- |
| **Thread 1: Capture** | Hardware V4L2 / Video | ~33 ms | Paced at 30 FPS camera capture |
| **Thread 2: Gate + Indices + Tile** | 4x ARM Cortex-A57 CPU | **~16–27 ms** | • FrameGate blur & clipping: ~3 ms<br>• Indices (ExG mask, VARI, TGI, DGCI): **~5–12 ms**<br>• Tiler 3x3 slicing & resize to 224x224: ~8–12 ms |
| **Thread 3: Inference** | **NVIDIA Maxwell GPU** | **~117 ms** | Batch 9 FP16 EfficientNet-Lite0 (~13 ms/tile) |
| **Thread 4: Decide + Aggregate + Store** | ARM Cortex-A57 CPU | ~2–4 ms | Logit scaling, temporal consensus, phenology lookup, SQLite WAL insert |

#### Concurrency & Throughput Analysis:
1. **Parallel Producer-Consumer Decoupling:**
   - Thread 2 (CPU) and Thread 3 (GPU) run concurrently in pipelined stages separated by bounded `DropOldestQueue(maxsize=8)`.
   - Because Thread 2's total execution time (**~16–27 ms**) is significantly less than Thread 3's GPU execution time (**~117 ms**), Thread 2 finishes its work well before the GPU completes inference.
2. **GPU Remains the Rate-Limiting Bottleneck:**
   - The ~5–12 ms CPU indices computation is completely masked behind the ~117 ms GPU inference window.
   - **Throughput is NOT degraded by indices computation:** Peak sustained GPU throughput remains **~7.0 to 7.5 scenes/sec** (governed by the ~117 ms GPU batch inference).
3. **Single-Frame Transit Latency:**
   - The end-to-end transit latency for a single scene traversing from camera capture to final advisory storage increases by ~10 ms (from **~132 ms** to **~142–147 ms** total transit latency).
4. **Operational Scene Rate:**
   - At normal field walking speed (1.0–1.5 m/s) with the 60% scene displacement novelty filter active, only 3–8% of captured camera frames pass gating. The operational compute load is **2.0 to 4.0 scenes/sec**, operating well inside the 10W thermal budget.

---

## 7. On-Device SQLite Storage Verification

After the pipeline finishes, inspect the stored scan and advisory in `data/edge.db`.

### 7.1 Verify Scan Record & Metadata
```bash
sqlite3 data/edge.db "SELECT scan_id, started_utc, ended_utc, frames_evaluated, tiles_classified, metadata_json FROM scans;"
```
*Expected Output:*
```
scan_2026...|2026-09...|2026-09...|13|117|{"days_since_planting": 75, "total_cycle_days": 150}
```

### 7.2 Verify Complete Advisory Document & Provenance Guard
```bash
sqlite3 data/edge.db "SELECT seq, advisory_id, json_extract(payload_json, '$.crop_health.crop'), json_extract(payload_json, '$.crop_health.state'), json_extract(payload_json, '$.growth_stage.stage'), json_extract(payload_json, '$.growth_stage.kc'), json_extract(payload_json, '$.vegetation.canopy_cover.mean'), json_extract(payload_json, '$.actions[0].template_id'), json_extract(payload_json, '$.actions[0].generated_by'), json_extract(payload_json, '$.actions[0].verification_status') FROM advisories;"
```
*Expected Output for test clip:*
```
1|2026-09..._F01||UNCERTAIN||||ACT_MULTICROP_INVESTIGATE|template|VERIFIED
```
*(Notice: `generated_by` is strictly `template`, NEVER `placeholder` or `rules_engine`).*

### 7.3 Inspect Real Numerical Values in `growth_stage` and `vegetation`

Extract the complete `growth_stage` and `vegetation` blocks using Python's built-in JSON tool:
```bash
sqlite3 data/edge.db "SELECT payload_json FROM advisories ORDER BY seq DESC LIMIT 1;" | python3 -c "
import sys, json
doc = json.load(sys.stdin)
print('--- GROWTH STAGE BLOCK ---')
print(json.dumps(doc.get('growth_stage'), indent=2))
print('--- VEGETATION BLOCK ---')
print(json.dumps(doc.get('vegetation'), indent=2))
print('--- ACTIONS BLOCK (TOP ACTION) ---')
print(json.dumps(doc.get('actions', [{}])[0], indent=2))
"
```

*Expected Structure of `growth_stage` (when planting date is provided):*
```json
{
  "crop": "rice",
  "stage": "mid_season",
  "stage_code": "MID",
  "days_since_planting": 75,
  "total_cycle_days": 150,
  "cycle_source": "default_assumption",
  "cycle_verification_status": "RECALLED_UNVERIFIED",
  "stage_lengths_days": {
    "initial": 30,
    "development": 30,
    "mid_season": 60,
    "late_season": 30
  },
  "canopy_cover_measured": 0.7821,
  "canopy_cover_expected_range": [0.70, 1.00],
  "kc": 1.20,
  "status": "OK",
  "verification_status": "WEB_VERIFIED",
  "source": "derived",
  "document_reference": "FAO Irrigation and Drainage Paper No. 56, Chapter 5 (Crop growth stages) & Chapter 6 (Table 11 & Table 12). Archived in docs/sources/ with SHA256."
}
```

*Expected Structure of `growth_stage` (when planting date is omitted):*
```json
{
  "crop": "rice",
  "stage": null,
  "stage_code": null,
  "reason": "DAYS_SINCE_PLANTING_REQUIRED",
  "status": "AWAITING_PLANTING_DATE",
  "days_since_planting": null,
  "total_cycle_days": 150,
  "cycle_source": "default_assumption",
  "cycle_verification_status": "RECALLED_UNVERIFIED",
  "canopy_cover_measured": 0.7821,
  "canopy_cover_expected_range": null,
  "kc": null,
  "verification_status": "WEB_VERIFIED",
  "source": "derived",
  "document_reference": "FAO Irrigation and Drainage Paper No. 56, Chapter 5 (Crop growth stages) & Chapter 6 (Table 11 & Table 12). Archived in docs/sources/ with SHA256."
}
```

*Expected Structure of `vegetation` (measured real values):*
```json
{
  "canopy_cover": {
    "mean": 0.7821,
    "p10": 0.7104,
    "p50": 0.7852,
    "p90": 0.8410,
    "status": "OK",
    "source": "measured"
  },
  "vari": {
    "mean": 0.2451,
    "band": "TYPICAL",
    "status": "OK",
    "source": "measured"
  },
  "exg": {
    "mean": 42.15,
    "status": "OK",
    "source": "measured"
  },
  "tgi": {
    "mean": 18.32,
    "status": "OK",
    "source": "measured"
  },
  "dgci": {
    "mean": 0.582,
    "status": "OK",
    "source": "measured"
  },
  "ndvi": null,
  "ndvi_status": "PENDING_HARDWARE_FINALIZATION",
  "ndvi_reason": "Optical path and calib_matrix in progress. Field reserved; not estimated."
}
```

---

## 8. Running & Verifying the API Gateway on Jetson Nano

The mobile phone connects to the Nano's local WiFi AP (`SIH-FIELD`, `192.168.4.1`) to pull advisories. The gateway server and the pipeline execute concurrently without database locking issues due to SQLite WAL mode.

### 8.1 Launch Gateway Server in Background
```bash
nohup python3 gateway/server.py --host 0.0.0.0 --port 8080 --db-path data/edge.db > artifacts/reports/gateway.log 2>&1 &
```
Verify the server is listening:
```bash
curl -s http://127.0.0.1:8080/api/v1/health
```

### 8.2 Local Endpoint Verification Commands (`curl`)

Run these curl commands to test all 5 endpoints:

```bash
# 1. Health Check
curl -s http://127.0.0.1:8080/api/v1/health
# Expected: {"status": "ok", "subsystems": {"storage": "ok", "clock": "valid", "syncing": false}, "unacked_advisories": 1}

# 2. Advisory Manifest Catalog
curl -s "http://127.0.0.1:8080/api/v1/manifest?since=0&limit=10"
# Expected: {"schema_version": "1.0", "count": 1, "has_more": false, "advisories": [{"seq": 1, "advisory_id": "...", ...}]}

# 3. Retrieve Advisory Document (seq 1)
curl -s http://127.0.0.1:8080/api/v1/advisory/1 | python3 -m json.tool | head -n 35
# Expected: Complete v1.0 advisory JSON document matching ans_for_vitthal.md schema

# 4. Acknowledge Advisory Receipt
curl -s -X POST http://127.0.0.1:8080/api/v1/ack \
  -H "Content-Type: application/json" \
  -d '{"last_acked_seq": 1}'
# Expected: {"status": "ok", "acked_count": 1, "unacked_remaining": 0}

# 5. Media Pruning Verification
curl -s -i http://127.0.0.1:8080/api/v1/media/img_test.jpg
# Expected: HTTP/1.0 410 Gone (confirming §A5/§A10 media pruning contract)
```

### 8.3 Concurrent Pipeline Write & Gateway Read Test
To verify that the mobile app polling the gateway does not block or crash the scanning pipeline:
1. Start the gateway in background.
2. In terminal 1, launch the pipeline:
   ```bash
   python3 edge/pipeline.py --source test_video_from_dataset_images.mp4 --engine artifacts/engines/model_a_fp16.engine --db-path data/edge.db
   ```
3. In terminal 2, loop curls while the pipeline is actively writing:
   ```bash
   for i in {1..20}; do curl -s http://127.0.0.1:8080/api/v1/health | grep '"status": "ok"' || echo "GATEWAY LOCK ERROR"; sleep 0.5; done
   ```
4. Verify that 0 errors occur and all requests return HTTP 200 instantly (SQLite WAL concurrency guarantee).

### 8.4 Model B Fallback: Compiling TensorRT Engine with trtexec
Model B defaults to ONNX Runtime execution (`onnxruntime-gpu` / CPU wheel). In environments where TensorRT acceleration is required for batch patch classification on the Jetson Nano, compile the dynamic batch Model B ONNX artifact (opset 13) into an FP16 TensorRT engine using `trtexec`:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/model_b.onnx \
  --saveEngine=artifacts/engines/model_b.engine \
  --fp16 \
  --minShapes=input:1x3x64x64 \
  --optShapes=input:64x3x64x64 \
  --maxShapes=input:256x3x64x64 \
  --workspace=256
```

> [!NOTE]
> - **Input Contract:** Tensor name `input`, shape `[batch_size, 3, 64, 64]`, dtype `float32`, RGB channel order normalized with ImageNet mean/std.
> - **Batch Profile:** Dynamic batching with minimum shape `1`, nominal optimization shape `64`, and upper ceiling `256` blobs per inference pass.
> - **Fallback Scope:** Currently retained as documentation and engine build specification; edge runtime defaults to ONNX Runtime per H5.2.

---


## 9. Post-Execution eMMC Headroom Verification

After executing the pipeline and gateway:
```bash
df -h /
```
Confirm:
1. Available eMMC storage is within 5–10 MB of pre-execution reading.
2. No orphaned swap files or core dumps exist in `/tmp/` or the working directory.

---

## 10. Rollback & Emergency Remediation Procedures

### 10.1 CUDA Context Stall / Zombie Process
If the pipeline exits abnormally or CUDA throws `CUDA_ERROR_OUT_OF_MEMORY`:
```bash
# Identify any lingering python or CUDA processes
fuser -v /dev/nvhost-ctrl-gpu

# Force terminate any stalled process
killall -9 python3

# Reset Jetson clocks
sudo jetson_clocks --restore
```

### 10.2 Corrupt SQLite Database Recovery / Reset
If an ungraceful power outage occurs while SQLite is writing:
```bash
# Check database integrity
sqlite3 data/edge.db "PRAGMA integrity_check;"

# If corrupt or reset desired, cleanly re-initialize:
rm -f data/edge.db data/edge.db-wal data/edge.db-shm

# Re-run a dry-run or live scan to rebuild fresh schema
python3 edge/pipeline.py --source test_video_from_dataset_images.mp4 --dry-run
```

### 10.3 Brownout or System Freeze Remediation
If the Jetson Nano abruptly reboots when inference starts:
1. Check power supply: Micro-USB power cannot supply the 3.5A transient spikes of Maxwell GPU inference. Switch to 5V/4A DC barrel jack.
2. Throttle to 5W mode if on battery:
   ```bash
   sudo nvpmodel -m 1
   ```
   *(5W mode disables 2 CPU cores and throttles GPU clock to 640 MHz; tile latency increases to ~24 ms).*

---

## 11. Mast Node Clock Synchronization Rules (Guide §6 / L1.4)

When connecting to the ESP32 Ground Mast node (`SIH-NODE-01` at `192.168.9.1`), the Nano pod evaluates whether to issue `POST /api/v1/time?utc=<unix_seconds>`:

### 11.1 Synchronization Gate Conditions
The Nano will send `POST /time` **IF AND ONLY IF**:
1. The Nano's onboard GPS fix is valid (`valid: true`, non-null UTC timestamp in range `[1735689600, 4102444800]`), **AND**
2. At least one of the following drift conditions is met:
   - The mast's `rtc_valid` flag is `false` (RTC lost power / uninitialized).
   - The absolute time drift between the mast's DS3231 RTC and the Nano's GPS timestamp exceeds 120 seconds:
     $$\left| t_{\text{mast, UTC}} - t_{\text{GPS, UTC}} \right| > 120\text{ s}$$

If `rtc_valid` is `true` and $\left| t_{\text{mast}} - t_{\text{GPS}} \right| \le 120\text{ s}$, `POST /time` is skipped to avoid unnecessary non-volatile flash writes on the DS3231.

---

## 12. MLX90640 I2C 400 kHz Bus Configuration (L6.2)

> [!NOTE]
> **Status:** `UNVERIFIED ON EAGLE-101`  
> Physical I2C timing must be validated with an oscilloscope or logic analyzer on the bench before production field flight.

The Melexis MLX90640 32x24 thermal sensor requires I2C Fast Mode (400 kHz) to sustain 2–4 Hz frame refresh rates without bus choking (each full subpage transfer reads 832 16-bit words = 1664 bytes).

### 12.1 Configuring Jetson Nano I2C Bus Speed to 400 kHz
To configure `/dev/i2c-1` (pins 3 & 5 on the 40-pin header) for 400 kHz:
```bash
# Verify current I2C bus clock frequency (default 100 kHz)
sudo cat /sys/bus/i2c/devices/i2c-1/bus_clk_rate || true

# Set clock rate to 400000 Hz in device tree / runtime sysfs:
sudo sh -c 'echo 400000 > /sys/bus/i2c/devices/i2c-1/bus_clk_rate' 2>/dev/null || true
```
Ensure $4.7\text{ k}\Omega$ pull-up resistors to 3.3V are populated on both SDA (Pin 3) and SCL (Pin 5).

---

## 13. Systemd Services & Collector Trigger Policy (L7.1, M3.1–M3.4)

Production deployment on the Jetson Nano utilizes systemd unit files installed by `scripts/setup_nano_services.sh`:

### 13.1 Service Inventory & Trigger Hierarchy

| Unit Name | Type | Purpose | Policy & Dependencies |
| :--- | :--- | :--- | :--- |
| `sih-gateway.service` | `simple` | Offline HTTP API Gateway (`0.0.0.0:8080`) | `After=network-online.target` |
| `sih-pipeline.service` | `simple` | Model A Edge Video Inference Daemon | `After=sih-gateway.service`, `Requires=sih-gateway.service` |
| `sih-collector-boot.service` | `oneshot` | Initial single sync after gateway boots | `After=sih-gateway.service`, `Requires=sih-gateway.service` |
| `sih-collector.service`| `oneshot` | Ground Mast Pull Collector | Mutex guarded via `/usr/bin/flock -n /run/lock/sih-collector.lock` |
| `sih-collector.timer`  | `timer` | Background collector polling (Disabled by default) | Configurable interval (min 60m), `Persistent=true`. Skips if client connected. |

### 13.2 Collector Trigger Hierarchy
1. **Primary Trigger:** Farmer mobile app requests on-demand sync via `POST /api/v1/sync/trigger`.
2. **Boot Trigger:** `sih-collector-boot.service` runs once automatically after system boot and gateway startup.
3. **Background Periodic Timer:** `sih-collector.timer` is **DISABLED by default**. If enabled by operator, it enforces a minimum interval of 60 minutes and automatically skips execution if any mobile client is connected to `SIH-FIELD` AP (detected via `iw dev wlan0 station dump`).

### 13.3 Installation & Startup
```bash
# Run installer as root
sudo ./scripts/setup_nano_services.sh

# Enable and start essential production services
sudo systemctl enable --now sih-gateway.service
sudo systemctl enable --now sih-collector-boot.service
sudo systemctl enable sih-pipeline.service

# Optional: Enable 60-min background collector timer
sudo systemctl enable --now sih-collector.timer
```

---

## 14. Copernicus CDSE Sentinel-2 Satellite NDVI Fallback (M4.1–M4.8)

**Status:** `UNVERIFIED ON LIVE NETWORK` (Bench verified with synthetic mocked responses in CI; live network queries pending deployment credentials).

When on-pod NoIR hardware is absent or uncalibrated, the Nano can report Sentinel-2 L2A satellite NDVI for the configured field polygon.

### 14.1 CDSE Account & OAuth2 Client Setup
1. Register an account on the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu).
2. Open the [Copernicus Dashboard](https://shapps.dataspace.copernicus.eu/dashboard/#/).
3. Navigate to **OAuth Clients** -> **Create New OAuth Client**.
4. Set Grant Types to `Client Credentials`. Copy the generated `Client ID` and `Client Secret`.

### 14.2 Credentials Storage
Save the credentials to `/etc/sih/cdse.json` on the Jetson Nano with restrictive 600 permissions:
```bash
sudo mkdir -p /etc/sih
sudo cat << EOF > /etc/sih/cdse.json
{
  "client_id": "<YOUR_CLIENT_ID>",
  "client_secret": "<YOUR_CLIENT_SECRET>"
}
EOF
sudo chmod 600 /etc/sih/cdse.json
```

### 14.3 Field Geometry Configuration
By default, `configs/field.json` is unconfigured:
```json
{
  "status": "NOT_CONFIGURED",
  "field_id": null,
  "polygon_coordinates": null,
  "last_updated_utc": null
}
```
To configure a field, enter the field boundary polygon in `configs/field.json` as GeoJSON coordinates `[ [lon, lat], [lon, lat], ... ]` (with first and last points identical):
```json
{
  "status": "MEASURED",
  "field_id": "F01",
  "polygon_coordinates": [
    [
      [77.5850, 28.5200],
      [77.5870, 28.5200],
      [77.5870, 28.5220],
      [77.5850, 28.5220],
      [77.5850, 28.5200]
    ]
  ],
  "last_updated_utc": "2026-09-20T10:00:00Z"
}
```

### 14.4 Running Manual Satellite Fetch
Whenever the Jetson Nano is connected to the internet (via Ethernet or uplink):
```bash
python3 scripts/fetch_satellite_ndvi.py --credentials /etc/sih/cdse.json --field-config configs/field.json
```
The script performs a quick connectivity check, authenticates via CDSE token endpoint, executes the SCL-masked Statistical API call, and caches the result into `edge.db`.

---

## 15. Thermal Reference Surface Setup & Calibration (M2.1–M2.3)

Crop Water Stress Index (CWSI) is calculated via direct physical reference surfaces:
$$\text{CWSI} = \frac{T_c - T_{\text{wet}}}{T_{\text{dry}} - T_{\text{wet}}}$$

### 15.1 Step-by-Step Setup Procedure
1. Position physical wet (saturated water pad) and dry (sunlit non-transpiring pad) reference surfaces in the fixed view of the MLX90640.
2. Capture a calibration frame and generate the labelled 480x640 pixel grid:
   ```bash
   python3 scripts/thermal_ref_setup.py --capture
   ```
3. Open the resulting PNG image in `data/thermal_calibration/`. Identify the row ranges (0–23) and column ranges (0–31) covering the wet pad and dry pad.
4. Save the calibrated boxes into `configs/thermal_refs.json`:
   ```bash
   python3 scripts/thermal_ref_setup.py --wet-box 2,6,2,6 --dry-box 2,6,26,30
   ```
5. If `configs/thermal_refs.json` has `status: "NOT_CONFIGURED"`, CWSI is safely marked `"available": false` in emitted advisories.

