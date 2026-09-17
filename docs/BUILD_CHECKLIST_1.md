# BUILD CHECKLIST
### Every file that must exist, what it must do, and how to verify it

**Version:** 1.0 · 3 September 2026
**Companion to:** `ULTIMATE_IMPLEMENTATION_PLAN_1.md` (the *order*; this document is the *inventory*)
**Reference:** `AI_Handbook_4.md` · `SIH_Smart_Farming_AI_Report_4.md` · `sih_pipeline_v4/`

---

## How to use this document

The implementation plan tells you **when** to build things. This tells you **what each file must contain** and **when it counts as finished**.

- Tick a box only when its **Verify** command has been run and the output pasted into your build log
- Files marked **COPY** are already written and tested — copying them is the whole task
- Files marked **HUMAN** contain values only a person can measure or source

**Totals: 43 code files · 7 config/data files · 24 generated artifacts · 9 human deliverables**

## Legend

| Tag | Meaning |
|---|---|
| **COPY** | Already exists in `sih_pipeline_v4/`. Copy it. Do not rewrite, do not "improve". |
| **BUILD** | The agent writes this from scratch. |
| **HUMAN** | A person must measure, source, or click something. |
| **GEN** | Produced by running code. Not written by hand. |

---

## MASTER INVENTORY — quick scan

### Root
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `README.md` | BUILD | 1 |
| ☐ | `.gitignore` | BUILD | 1 |
| ☐ | `conftest.py` | BUILD | 1 |
| ☐ | `requirements-train.txt` | BUILD | 3 |
| ☐ | `requirements-edge.txt` | BUILD | 3 |

### `configs/`
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `classes.py` | **COPY** | 1 |
| ☐ | `paths.py` | BUILD | 2 |
| ☐ | `train_config.py` | BUILD | 2 |
| ☐ | `crop_baselines.py` | **HUMAN** | 24 |
| ☐ | `etl_thresholds.py` | **HUMAN** | 32 |
| ☐ | `trap_config.py` | **HUMAN** | 31 |
| ☐ | `field_demo.yaml` | BUILD | 34 |

### `core/`
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `aggregate.py` | **COPY** | 1 |
| ☐ | `rejection.py` | **COPY** | 1 |
| ☐ | `thermal.py` | **MODIFIED** | 1, 24 |
| ☐ | `trap_segmentation.py` | **MODIFIED** | 1, 31 |
| ☐ | `indices.py` | **IMPLEMENTED** | 26 |
| ☐ | `ndvi.py` | **IMPLEMENTED** | 26 |

### `train/` — modern Python, never runs on the Nano
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `download_data.py` | BUILD | 6 |
| ☐ | `build_manifest.py` | BUILD | 8 |
| ☐ | `dedup.py` | BUILD | 9 |
| ☐ | `split.py` | BUILD | 9 |
| ☐ | `bias_audit.py` | BUILD | 10 |
| ☐ | `transforms.py` | BUILD | 11 |
| ☐ | `dataset.py` | BUILD | 11 |
| ☐ | `model.py` | BUILD | 12 |
| ☐ | `train_model_a.py` | BUILD | 12–13 |
| ☐ | `evaluate.py` | BUILD | 14 |
| ☐ | `calibrate.py` | BUILD | 15 |
| ☐ | `export_onnx.py` | BUILD | 16–17 |
| ☐ | `train_model_b.py` | BUILD | 30 |

### `edge/` — **Python 3.6 compatible**, runs on the Nano
> **Scope Note (7 Sep 2026):** Autonomous irrigation actuation is dropped. No solenoid valve, MOSFET driver, or YF-S201 flow meter will be purchased. Actuation is advisory-only: the system produces irrigation prescriptions for manual farmer execution. `edge/actuation.py` and `edge/flow.py` are retained as validated reference implementations for future closed-loop deployment and are NOT wired into any runtime path.
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `build_engine.sh` | BUILD | 18 |
| ☐ | `trt_classifier.py` | BUILD | 19 |
| ☐ | `frame_gate.py` | BUILD | 20 |
| ☐ | `tiler.py` | BUILD | 20 |
| ☐ | `pipeline.py` | BUILD | 21 |
| ☐ | `sensors.py` | BUILD | 23 |
| ☐ | `agronomy.py` | BUILD | 25–26 |
| ☐ | `rules_engine.py` | BUILD | 27 |
| ☐ | `advisory.py` | BUILD | 28 |
| ☐ | `camera.py` | **IMPLEMENTED** | 23 |
| ☐ | `storage.py` | BUILD | 33 |
| ☐ | `actuation.py` | REFERENCE (ADVISORY-ONLY) | 27 |
| ☐ | `flow.py` | REFERENCE (ADVISORY-ONLY) | 27 |

### `scripts/`
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `calibrate_dual_bandpass.py` | **IMPLEMENTED** | 26 |

### `gateway/`, `dashboard/`, `firmware/`, `tests/`
| ✓ | File | Tag | Step |
|---|---|---|---|
| ☐ | `gateway/trap_receiver.py` | BUILD | 29 |
| ☐ | `gateway/trap_count.py` | BUILD | 31 |
| ☐ | `dashboard/app.py` | BUILD | 33 |
| ☐ | `firmware/esp32cam_trap/esp32cam_trap.ino` | BUILD | 29 |
| ☐ | `tests/test_pipeline.py` | **COPY** | 1 |
| ☐ | `tests/test_integration.py` | BUILD | 35 |

---

# DETAILED SPECIFICATIONS

---

## 1. ROOT FILES

### ☐ `README.md` — BUILD, Step 1
**Must contain:** one-paragraph project description; the two-environment rule (`train/` modern Python, `edge/` Python 3.6); quickstart commands; a link to the implementation plan.
**Verify:** a teammate who has not seen the project can set up the dev environment from it alone.

### ☐ `.gitignore` — BUILD, Step 1
**Must contain:** `data/`, `artifacts/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.engine`, `*.onnx`, `kaggle.json`, `.env`, `*.pt`
**Verify:** `git check-ignore -v kaggle.json data/ artifacts/` returns a match for each.

### ☐ `conftest.py` — BUILD, Step 1
**Must contain:** `import sys, os; sys.path.insert(0, os.path.dirname(__file__))` so `pytest` resolves `core/` and `configs/`.
**Verify:** `python -m pytest tests/ -v` → **15 passed**.

### ☐ `requirements-train.txt` — BUILD, Step 3
**Must contain:** torch≥2.0, torchvision, timm≥1.0.9, albumentations≥1.4, opencv-python, numpy, pandas, scikit-learn, scipy, imagehash, Pillow, matplotlib, seaborn, tqdm, onnx, onnxsim, onnxruntime, grad-cam, pytest, kaggle
**Verify:** `pip install -r requirements-train.txt` then `python -c "import timm; print('tf_efficientnet_lite0' in timm.list_models('*lite*', pretrained=True))"` → `True`

### ☐ `requirements-edge.txt` — BUILD, Step 3
**Must contain:** `numpy==1.19.5`, `pycuda==2020.1`, `Pillow==8.4.0`, `requests==2.27.1`
**MUST NOT contain:** `opencv-python` or `tensorrt` — both ship with JetPack 4.6.4 and pip versions will conflict.
**Verify:** on the Nano, `pip3 install -r requirements-edge.txt` completes and `python3 -c "import cv2, tensorrt, pycuda.driver; print('ok')"` succeeds.

---

## 2. `configs/` — every constant lives here

> Two files disagreeing about a constant is exactly how the batch-8-versus-9-tiles bug happened. Nothing outside `configs/` may hardcode a path or a threshold.

### ☐ `configs/classes.py` — **COPY**, Step 1
**Source:** `sih_pipeline_v4/configs/classes.py`
**Provides:** `CLASS_NAMES` (31), `NUM_CLASSES`, `IDX`, `HEALTHY_COLS`, `NOTCROP_COL`, `CROP_COLS`, `DISEASE_COLS`
**MUST NOT:** be edited. If you drop wheat, do it by not populating those classes — do not renumber the taxonomy mid-project.
**Verify:** `python -c "from configs.classes import NUM_CLASSES, CROP_COLS; assert NUM_CLASSES==31 and len(CROP_COLS)==30; print('ok')"`

### ☐ `configs/paths.py` — BUILD, Step 2
**Must contain:** `ROOT`, `DATA`, `RAW`, `INTERIM`, `PROCESSED`, `SPLITS`, `ARTIFACTS`, `CKPT`, `ONNX_DIR`, `ENGINE_DIR`, `REPORTS`, plus per-dataset dirs (`PADDY`, `SUGAR_THITE`, `SUGAR_DAPHAL`, `PLANTDOC`, `PLANTWILD`, `NOTCROP`, `OPENSET`). Must `mkdir(parents=True, exist_ok=True)` on import.
**Must use:** `pathlib.Path`, all paths relative to `ROOT` — never absolute.
**Verify:** `python -c "from configs.paths import ROOT, OPENSET; print(ROOT, OPENSET.exists())"`

### ☐ `configs/train_config.py` — BUILD, Step 2
**Must contain, in two clearly separated blocks:**

*Shared (changing these invalidates the engine):* `IMAGE_SIZE=224`, `TILE_SIZE=320`, `TILE_GRID=3`, `N_TILES=9`, `TILE_OVERLAP=0.20`, `ENGINE_BATCH=N_TILES`, `ONNX_OPSET=13`

*Training only:* `BACKBONE='tf_efficientnet_lite0'`, `BATCH_SIZE`, `EPOCHS`, `LR_HEAD`, `LR_BACKBONE`, `WEIGHT_DECAY`, `WARMUP_EPOCHS`, `LABEL_SMOOTH`, `DROP_RATE`, `DROP_PATH`, `EMA_DECAY`, `GRAD_CLIP`, `SEED`

*Thresholds (start as `None`, written back by Step 15):* `TAU_ENERGY`, `T_CAL` — plus `TAU_DISEASE=0.55`, `TAU_MARGIN=0.10`, `TAU_CONF=0.60`, `TAU_PRIOR=1.0`, `CELL_K=2`, `CELL_N=3`, `CELL_MIN_SCORE=0.55`

**MUST:** define `ENGINE_BATCH = N_TILES`, not a literal. **Never write `8` anywhere.**
**Verify:** `python -c "from configs.train_config import N_TILES, ENGINE_BATCH, ONNX_OPSET; assert N_TILES==ENGINE_BATCH==9 and ONNX_OPSET<=13; print('ok')"`

### ☐ `configs/crop_baselines.py` — **HUMAN**, Step 24
**Must contain:** published CWSI baselines per crop — `LL_SLOPE`, `LL_INTERCEPT` (the non-water-stressed baseline as a function of VPD) and `UL_OFFSET` (non-transpiring baseline), for rice and sugarcane. **Cite the source next to each number in a comment.**
**MUST NOT:** contain invented numbers. If you cannot source a baseline for a crop, omit that crop and say so in the PPT.
**Verify:** every constant has a citation comment; `python -c "from configs.crop_baselines import RICE; print(RICE)"` works.

### ☐ `configs/etl_thresholds.py` — **HUMAN**, Step 32
**Must contain:** Economic Threshold Levels per pest, from ICAR/NIPHM IPM packages, **with the source cited per number** (e.g. ICAR-IIRR sets ~10 hoppers per hill at tillering).
**Why it matters:** a judge may ask where a threshold came from. A citation is a much better answer than a chosen number.

### ☐ `configs/trap_config.py` — **HUMAN**, Step 31
**Must contain:** `MM_PER_PIXEL` measured by photographing a ruler at your exact camera-to-trap distance; `ABS_FLOOR_PX` derived from the smallest target pest radius (whitefly ≈ 1.0–1.5 mm); camera distance and resolution.
**MUST NOT:** set `ABS_FLOOR_PX` by guesswork. It is a physical constant of your fixed geometry. Guessing it can reintroduce the bug where one large moth erased every whitefly on the board.
**Verify:** `MM_PER_PIXEL` has a comment recording how it was measured.

### ☐ `configs/field_demo.yaml` — BUILD, Step 34
**Must contain:** which sensors are live vs simulated, the video/camera source, run duration, output DB path, log level.
**Verify:** `python3 edge/pipeline.py --config configs/field_demo.yaml --dry-run` parses it without error.

---

## 3. `core/` and Sensing Modules — Tested Implementations

> Note: `aggregate.py` and `rejection.py` remain tested and frozen. `thermal.py` and `trap_segmentation.py` have received required defect fixes and calibration gates. `indices.py`, `ndvi.py`, `edge/camera.py`, and `scripts/calibrate_dual_bandpass.py` implement the vegetation sensing pipeline.

### ☐ `core/aggregate.py` — **COPY**, Step 1
**Provides:** `aggregate_frame(tile_probs, healthy_cols, notcrop_col, ...)` → `(state, class_id, score)` with state in `DISEASE|HEALTHY|NOT_CROP|UNCERTAIN`; `aggregate_cell(frame_results, k, n, min_score)` → dict with state in `DISEASE|HEALTHY|UNCERTAIN|NO_DATA`.
**Guards:** frames with no healthy path (alarms on pristine fields); top-k averaging diluting a single-tile lesion; `None` conflating healthy with never-visited.
**Verify:** `pytest tests/test_pipeline.py -k aggregate -v`

### ☐ `core/rejection.py` — **COPY**, Step 1
**Provides:** `open_set_energy(logits, crop_cols, T)` (1D and 2D safe); `posthoc_logit_adjust`; `softmax`; `decide(...)`; `fit_energy_threshold(...)`
**Guards:** energy summed over all logits including `not_crop`; confidence gated on prior-adjusted probabilities; `AxisError` on 1D input.
**Verify:** `pytest tests/test_pipeline.py -k rejection -v`

### ☐ `core/thermal.py` — **MODIFIED**, Step 1, 24
**Status:** MODIFIED — pure-canopy bisection fixed via unimodality ratio guard (`PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO = 0.85`), fixed-mast rescope applied (veg_fraction=None in mast path), 14-observation baseline gate added.
**Provides:** `canopy_temperature(thermal, air_temp_c, veg_fraction=None, ...)`; `cwsi(...)`; `fit_non_water_stressed_baseline(...)`.
**Guards:** Pure-canopy bisection where sun/shade leaf variance falsely split pure canopy; `Ta+15` biophysical gate; NWSB baseline fit requiring $\ge 14$ solar-noon observations.
**Verify:** `pytest tests/test_pipeline.py -k thermal -v`

### ☐ `core/trap_segmentation.py` — **MODIFIED**, Step 1, 31
**Status:** MODIFIED — dual-threshold marker extraction applied, adaptive thresholding replacing Otsu, label off-by-one fixed.
**Provides:** `segment_trap_blobs(bgr, min_area, max_area, abs_floor_px, crop)`; `extract_markers`; `_fill_holes`.
**Guards:** A global `0.3 × dist.max()` threshold erasing micro-pests; `adaptiveThreshold` hollowing large insects; the label-loop off-by-one.
**Verify:** `pytest tests/test_pipeline.py -k trap -v`

### ☐ `core/indices.py` — **IMPLEMENTED**, Step 26
**Status:** IMPLEMENTED — six RGB vegetation indices (VARI, TGI, NGRDI, GMR, DGCI, ExG) with band-addressed mapping (`BandMap`) and byte-identical BGR/RGB delegation adapters.
**Provides:** `vari`, `tgi`, `ngrdi`, `gmr`, `dgci`, `exg`, `vegetation_mask` (absolute ExG threshold), `compute_canopy_indices`.
**Guards:** Pure-green canopy bisection guarded by absolute ExG threshold (`PROVISIONAL_EXG_VEG_THRESHOLD = 20`); canopy fraction floor (`PROVISIONAL_MIN_CANOPY_FRACTION = 0.15`).
**Verify:** `pytest tests/test_pipeline.py -k indices -v`

### ☐ `core/ndvi.py` — **IMPLEMENTED**, Step 26
**Status:** IMPLEMENTED — dual-bandpass (660nm Red, 850nm NIR) NDVI calculation, silicon cross-talk unmixing, and empirical line method (ELM) calibration.
**Provides:** `compute_ndvi`, `apply_channel_response_correction`, `apply_empirical_line_calibration`, `ndvi_from_dual_bandpass`.
**Guards:** Inverted Bayer channel mapping (Blue=NIR, Red=660nm); bench cross-talk calibration gate (raises `NotImplementedError` if `calib_matrix is None`); ELM panel reflectance gate (raises `RuntimeError` if unconfirmed and `--allow-provisional` is False).
**Verify:** `pytest tests/test_pipeline.py -k "ndvi or cross_talk or empirical_line" -v`

### ☐ `edge/camera.py` — **IMPLEMENTED**, Step 23
**Status:** IMPLEMENTED — dual CSI camera capture abstraction on Jetson Nano with hardware port isolation (CSI-0 = RGB inspection, CSI-1 = NIR survey).
**Provides:** `DualCameraPipeline`, `CameraConfig`, GStreamer pipeline generation with nvarguscamerasrc and locked AWB/AE.
**Guards:** Separation of survey and inspection cameras without inter-port pixel-level fusion (preventing rolling-shutter shear and temporal sync failure).
**Verify:** `pytest tests/test_pipeline.py -k camera -v`

### ☐ `scripts/calibrate_dual_bandpass.py` — **IMPLEMENTED**, Step 26
**Status:** IMPLEMENTED — optical bench cross-talk response calibration script for computing $K^{-1}$ from 660nm and 850nm reference illumination measurements.
**Provides:** `compute_unmixing_matrix`, verification of invertibility, export to JSON calibration file for `core/ndvi.py`.
**Verify:** `python scripts/calibrate_dual_bandpass.py --mock --output /tmp/test_calib.json`

---

## 4. `train/` — modern Python, never runs on the Nano

### ☐ `train/download_data.py` — BUILD, Step 6
**Must:** support `--all` and `--verify`; download Paddy Doctor (Kaggle CLI), PlantDoc (git clone); print clear **MANUAL DOWNLOAD** instructions with URLs for Mendeley and PlantWild; `--verify` prints a table of dataset → file count → class-folder count and writes `artifacts/reports/data_inventory.txt`.
**MUST NOT:** simulate a download, generate placeholder images, or report success on a failed fetch. If it fails, it says so and exits non-zero.
**Verify:** `python train/download_data.py --verify` → every row non-zero.

### ☐ `train/build_manifest.py` — BUILD, Step 8
**Must:** walk every raw dataset dir; map source folder names → canonical classes; emit `splits/all_images.csv` (`path,label,source_dataset,orig_folder`) and `splits/class_mapping.csv` recording every merge decision.
**Required merges:** `brown_rust`+`rust` → `sugarcane__rust`; `yellow`+`yellow_leaf_disease` → `sugarcane__yellow_leaf`; `red_rot` from the Daphal set only.
**MUST NOT:** invent class names. Every label must already exist in `configs/classes.py`. Unmappable folders are logged and skipped, never guessed.
**Verify:** `set(df.label) - set(CLASS_NAMES)` is empty; total rows ≈ 27,000–30,000.

### ☐ `train/dedup.py` — BUILD, Step 9
**Must:** `imagehash.phash(hash_size=8)`; group images with Hamming distance ≤ 5 into a shared `group_id`; also catch exact MD5 duplicates **across** the two sugarcane datasets (both from Maharashtra, may share images); write `group_id` back to the manifest.
**Verify:** `group_id` column exists and `df.group_id.nunique() < len(df)` (some grouping actually happened).

### ☐ `train/split.py` — BUILD, Step 9
**Must:** `StratifiedGroupKFold` grouped on `group_id`, stratified on `label`; ~80/10/10; emit `splits/train.csv`, `val.csv`, `test_indist.csv`, and `test_crossdomain.csv` built separately from PlantDoc + PlantWild.
**MUST NOT:** split by image or use a plain random split.
**Verify:**
```bash
python -c "
import pandas as pd
d=[pd.read_csv(f'splits/{s}.csv') for s in ('train','val','test_indist')]
g=[set(x.group_id) for x in d]
assert not(g[0]&g[1]) and not(g[0]&g[2]) and not(g[1]&g[2]), 'GROUP LEAK'
print('ok', [len(x) for x in d])"
```

### ☐ `train/bias_audit.py` — BUILD, Step 10
**Must:** extract 8 background pixels per image (4 corners + 4 edge midpoints → 24-dim); train a RandomForest on **only** those; evaluate on the test split; write `artifacts/reports/bias_audit.json` with `background_only_accuracy` and `chance_accuracy` (3.2%).
**Verify:** the JSON exists with both fields. **Interpretation:** 3–6% clean → slide it. 15–30% mild leakage → strengthen augmentation. >40% serious → investigate which classes separate.

### ☐ `train/transforms.py` — BUILD, Step 11
**Must contain, as `train_transform()`:** `RandomResizedCrop(scale=(0.35,1.0))`, `HorizontalFlip(0.5)`, `VerticalFlip(0.3)`, `ShiftScaleRotate`, `RandomBrightnessContrast(0.35)`, one-of `RandomShadow`/`RandomSunFlare`/`RandomToneCurve`, `HueSaturationValue(hue_shift_limit=12)`, one-of `MotionBlur`/`GaussianBlur`/`Defocus`, one-of `ISONoise`/`GaussNoise`, `ImageCompression(45,95)`, `CoarseDropout`, `Normalize`, `ToTensorV2`. Plus `eval_transform()`.
**MUST NOT:** exceed ±12 hue shift, or add channel shuffle / heavy elastic transforms. Colour is diagnostic information for plant disease.
**Verify:** save a 4×4 grid to `artifacts/reports/aug_samples.png` and **look at it**. Lesions must still be visible.

### ☐ `train/dataset.py` — BUILD, Step 11
**Must:** read a split CSV; load with `cv2.imread` then **`cv2.cvtColor(..., COLOR_BGR2RGB)`**; emit 2–4 extra random 320×320 crops per training image (slicing-aided fine-tuning); a one-off pre-resize of all images on disk to max side 512; `WeightedRandomSampler` with sqrt-inverse frequency for stage 2.
**MUST NOT:** forget the BGR→RGB conversion. It is the same class of bug that broke the export path.
**Verify:** `x.shape == (B,3,224,224)`, dtype float, normalized range.

### ☐ `train/model.py` — BUILD, Step 12
**Must:** `build_model(backbone, num_classes=None, pretrained=True, drop_rate, drop_path_rate)` returning a plain `timm` model. Flat classification head with width resolved dynamically from `configs/classes.py` at runtime (`len(CLASS_NAMES)`), not a literal constant.
**MUST NOT:** add attention modules, ensembles, or LSTM heads. Each addition risks ONNX export failure and none helps as much as the augmentation pipeline.
**Verify:** `build_model()(torch.randn(2,3,224,224)).shape == (2, len(CLASS_NAMES))`

### ☐ `train/train_model_a.py` — BUILD, Steps 12–13
**Must:** a `--stage {1,2,3}` flag. Stage 1 = plain baseline. Stage 2 = full augmentation + class balancing + slicing + mixup/cutmix. Stage 3 = **online** consistent-teaching distillation (teacher sees the *identical* augmented tensor). AdamW, discriminative LR, OneCycle, EMA, AMP, grad clip. **Select checkpoints on macro-F1, never accuracy.** Write `artifacts/reports/stage{N}_metrics.json` with `val_macro_f1`, `val_top1`, `per_class_recall`.
**MUST NOT:** cache teacher logits offline. Precomputed teacher targets measurably degrade distillation; teacher and student must see the same view.
**MUST NOT:** combine training-time logit-adjusted loss with distillation. Use post-hoc adjustment instead.
**Verify:** `stage2_macro_f1 > stage1_macro_f1`, and the rarest classes have non-zero recall.

### ☐ `train/evaluate.py` — BUILD, Step 14
**Must produce:** `eval_indist.json`, `eval_crossdomain.json`, `confusion_matrix.png` (row-normalised), `per_class_recall.csv`, `gradcam_panel.png` (healthy / correct / failure).
**MUST NOT:** be run against `test_crossdomain.csv` more than once, and never used to tune.
**Verify:** all five files exist with real values.

### ☐ `train/calibrate.py` — BUILD, Step 15
**Must fit, in this order:** (1) `T_CAL` by temperature scaling on validation NLL; (2) `TAU_ENERGY` via `core.rejection.fit_energy_threshold` using **`data/raw/openset/`**; (3) `TAU_PRIOR` by sweeping `[0,0.25,0.5,0.75,1.0]`; (4) `TAU_DISEASE` and `TAU_MARGIN` by grid sweep with a precision/recall plot. `--write-config` writes values back into `configs/train_config.py`.
**MUST NOT:** calibrate `TAU_ENERGY` on `data/raw/not_crop/`. Those images are in-distribution for the trained model and measure nothing.
**Verify:** `TAU_ENERGY` and `T_CAL` are no longer `None`; `ood_metrics.json` has FPR@95TPR and AUROC; `threshold_sweep.png` exists.

### ☐ `train/export_onnx.py` — BUILD, Steps 16–17
**Must contain `FusedModel` with this exact line:**
```python
x = x[:, [2, 1, 0], :, :].float() / 255.0     # BGR -> RGB. NEVER OMIT.
```
with ImageNet mean/std in **true RGB order**. Export at `opset=13`, `batch=ENGINE_BATCH` (9), `dynamic_axes=None`. Provide `--verify` running all three checks.
**MUST NOT:** verify only with `torch.randint` noise — uniform noise has identical statistics in all three channels and cannot detect a channel swap.
**Verify:**
1. **Red-flag image** (strongly red patch) matches an independent RGB reference path to < 1e-3
2. **End-to-end** macro-F1 on real `cv2.imread` images within 0.5% of the reference pipeline
3. ONNX vs PyTorch tensor diff < 1e-3

### ☐ `train/train_model_b.py` — BUILD, Step 30
**Must:** `TrapPestCNN` (4 conv blocks, ~150k params, **from scratch**); classes = target pests + `not_pest` (dust, debris, **and beneficials**); data from RP11 + sticky-trap datasets + synthetic composites (segmented insects pasted onto real yellow trap backgrounds at random position/rotation); rotation at **any** angle; ~40 epochs, Adam 1e-3, cosine.
**Verify:** validation macro-F1 plus per-class recall. **`not_pest` recall must be high** — false pest counts trigger unnecessary spraying.

---

## 5. `edge/` — Python 3.6 compatible, runs on the Nano

> **Every file here must parse under Python 3.6.** No f-string `=`, no walrus, no `dataclasses`, no `multiprocessing.shared_memory`.

### ☐ `edge/build_engine.sh` — BUILD, Step 18
**Must:** set `nvpmodel -m 0` and `jetson_clocks`; run `trtexec --onnx=... --saveEngine=... --fp16 --workspace=1024 --verbose | tee build.log`; then benchmark with `--loadEngine --iterations=200 --avgRuns=100`.
**MUST NOT:** pass `--int8`. Maxwell has no INT8 hardware — it will be no faster and less accurate.
**Verify:** `.engine` exists; benchmark prints a mean latency (**write it down**, expect 60–120 ms for 9 tiles).

### ☐ `edge/trt_classifier.py` — BUILD, Step 19
**Must provide:** `TRTClassifier` with `infer()` returning **raw logits**, `infer_all()` chunk-looping so an unexpected tile count can never truncate, and `close()` freeing `d_in`, `d_out`, `stream`, `h_in`, `h_out`, `context`, `engine` then `gc.collect()`. Pagelocked buffers are `uint8 NHWC` (the fused graph does the rest).
**MUST NOT:** contain `import pycuda.autoinit` anywhere. **MUST NOT** return probabilities from `infer()` — energy must be computed on raw logits.
**Verify:** `python3 edge/trt_classifier.py --selftest --engine ...` prints latency, confirms output shape `(9,31)`, exits with no `LogicError`.

### ☐ `edge/frame_gate.py` — BUILD, Step 20
**Must reject unless all pass:** altitude in the 1.5–2.5 m band; |roll|,|pitch| under threshold; variance-of-Laplacian above `TAU_BLUR`; histogram not clipped; scene displacement vs last kept frame > 60%.
**Verify:** `--selftest` on a sample video reports a pass rate of **3–8%**. This is your single biggest optimisation, worth 10–30×.

### ☐ `edge/tiler.py` — BUILD, Step 20
**Must:** deterministic **3×3 grid**, 320×320 tiles, 20% overlap; keep tiles with >40% vegetation using `core.thermal.vegetation_mask`; **pad to exactly `N_TILES`** so the engine batch always matches.
**MUST NOT:** use Otsu for the vegetation mask. On a closed canopy the ExG histogram is unimodal and Otsu bisects it, reporting ~50% vegetation on a 100% green field.
**Verify:** `--selftest` **always** returns exactly 9 tiles, for any input.

### ☐ `edge/pipeline.py` — BUILD, Step 21
**Must:** four threads — capture+tagging / gate+tile / GPU inference / decide+aggregate+rules+store. Bounded queues with drop-oldest. **CUDA context created inside the inference thread**, classifier constructed there too, `clf.close()` before `ctx.pop(); ctx.detach()`.
**Must import, not reimplement:** `core.rejection.decide`, `core.aggregate.aggregate_frame`, `core.aggregate.aggregate_cell`.
**MUST NOT:** switch to multiprocessing. NumPy and OpenCV release the GIL, and `shared_memory` is Python 3.8+.
**Verify:** `--source test_video.mp4 --dry-run --report` prints frames seen, gate pass rate, tiles classified, per-cell verdicts, and **2–4 scenes/sec**; exits cleanly.

### ☐ `edge/sensors.py` — BUILD, Step 23
**Must provide:** fixed ground mast thermal station (MLX90640 spatial array for canopy extraction + MLX90614 for continuous cross-check/drift), DHT22/SHT31 → temp + RH + derived VPD, capacitive soil probes, dual CSI cameras (CSI-0 RGB inspection, CSI-1 dual-bandpass NIR survey) with **AWB and AE locked to fixed gains**, GPS + IMU + rangefinder. **Every class needs a `--simulate` mode** returning plausible synthetic values.
**MUST NOT:** leave auto-white-balance enabled. A camera that re-balances every frame makes colourimetry meaningless.
**Verify:** `python3 edge/sensors.py --selftest --simulate` prints one reading per sensor with units and timestamp.

### ☐ `edge/agronomy.py` — BUILD, Steps 25–26
**Must provide:** `relative_nutrient_map()` for the **drone** (compares cell b\* to the field median from the same pass — no card); `absolute_lcc_panel()` for the **rover/phone** (grey card, von Kries gains on the **leaf ROI only**); RGB vegetation indices (ExG, VARI, TGI, NGRDI) per cell; canopy cover fraction for growth staging.
**MUST NOT:** white-balance the classifier's input (train/test mismatch), or apply gains to the full 1080p frame (~25 MB per float32 copy for no benefit).
**Note in code:** OpenCV `COLOR_BGR2LAB` on uint8 returns b with a **+128 offset** — this is `b*+128`, not published CIELAB `b*`.
**Verify:** on the Mendeley nitrogen dataset, the b\* index is **monotonic** across the four LCC panels. Save the plot.

### ☐ `edge/rules_engine.py` — BUILD, Step 27
**Must implement:** heat stress (consecutive hours above crop-critical temperature); flood (24/72 h rainfall + saturation persistence); drought (water-balance deficit + CWSI trend); disease-favourable (leaf wetness × temperature windows); pest pressure (trap counts vs ETL + slope).
**Must emit the structured finding JSON:** `{crop, condition, confidence, area_pct, cell, cwsi, soil_moisture_pct, risk_flags, action, severity, timestamp}` — this is the contract with `advisory.py`.
**Irrigation Actuation Scope Note (7 Sep 2026):** Autonomous irrigation actuation is dropped. No solenoid valve, MOSFET driver, or YF-S201 flow meter will be purchased. The system produces irrigation prescriptions displayed to the farmer, who irrigates manually. Actuation is advisory-only. `edge/actuation.py` and `edge/flow.py` are retained as validated reference implementations only and are NOT wired into any runtime path.
**Must:** attach the triggering rule name to every alert. Explainability is the point of doing this in code.
**Verify:** a synthetic 7-day time series fires each rule at the right point and stays silent otherwise.

### ☐ `edge/advisory.py` — BUILD, Step 28
**Must:** take a structured finding, call the LLM API with a system prompt stating *"You are a translator. Render the finding below into simple [language] for a smallholder farmer. Do not add diagnoses, do not add recommendations not present in the JSON, do not change any number."* Support Hindi + one regional language + English, plus a 160-char SMS variant. **Offline fallback: a pre-written template per finding type.** Irrigation recommendations are rendered as actionable manual prescriptions for the farmer.
**MUST NOT:** pass raw sensor readings or model logits to the LLM for interpretation. The LLM renders a decided finding; it never diagnoses.
**Verify:** **with the network disabled**, every finding type still produces sensible template output. Run it with WiFi off — that is the test that matters.

### ☐ `edge/storage.py` — BUILD, Step 33
**Must:** SQLite with tables `frames`, `cell_verdicts`, `sensor_readings`, `trap_counts`, `alerts`; survive connectivity loss; opportunistic sync when a network appears.
**Verify:** a dry run populates every table; `sqlite3 db.sqlite ".tables"` lists all five.

---

## 6. `gateway/`, `dashboard/`, `firmware/`, `tests/`

### ☐ `gateway/trap_receiver.py` — BUILD, Step 29
**Must:** accept POSTed JPEGs from the ESP32-CAM; store with device ID and timestamp; trigger `trap_count.py`.
**Verify:** a curl-posted JPEG lands on disk with correct metadata.

### ☐ `gateway/trap_count.py` — BUILD, Step 31
**Must:** downscale to ~1280 px long side; call `core.trap_segmentation.segment_trap_blobs` with `ABS_FLOOR_PX` from `configs/trap_config.py`; classify each 64×64 crop with Model B; write per-species daily counts.
**MUST NOT:** reimplement blob segmentation.
**Verify:** on a photo of a real trap, counts are within **±20%** of a manual count. Save the side-by-side image — it is excellent video material.

### ☐ `dashboard/app.py` — BUILD, Step 33
**Must:** field map with per-cell colour where **all four states are visually distinct** (`DISEASE` / `HEALTHY` / `UNCERTAIN` / `NO_DATA`); trend charts for CWSI, soil moisture, trap counts; alert feed showing the triggering rule; advisory text in the selected language.
**MUST NOT:** render `HEALTHY` and `NO_DATA` the same colour. Green means "don't spray"; grey means "re-fly". That distinction is the reason `aggregate_cell` returns explicit states.
**Verify:** renders from a SQLite file with no hardware attached.

### ☐ `firmware/esp32cam_trap/esp32cam_trap.ino` — BUILD, Step 29
**Must:** wake on timer (twice daily is sufficient); capture JPEG at max PSRAM-supported resolution; POST to gateway; deep sleep; include device ID and timestamp.
**MUST NOT:** attempt on-device inference. A plain ESP32 (Xtensa LX6) gets only generic C kernels from ESP-NN — the SIMD speedups people quote belong to the ESP32-S3.
**Verify:** gateway receives a JPEG on schedule; board draws sleep-level current between captures.

### ☐ `tests/test_pipeline.py` — **COPY**, Step 1
**Source:** `sih_pipeline_v4/tests/test_pipeline.py` — 15 tests, each named for the red-team finding it guards.
**Only permitted edit:** import paths.
**Verify:** `python -m pytest tests/ -v` → **15 passed**

### ☐ `tests/test_integration.py` — BUILD, Step 35
**Must cover:** tiler always returns 9; `infer_all` never truncates; rules engine fires and stays silent correctly; advisory offline fallback; storage round-trip.
**Verify:** all pass alongside the original 15.

---

## 7. GENERATED ARTIFACTS

Not written by hand. Each is produced by running a step, and each is evidence.

### `splits/`
| ✓ | File | Step |
|---|---|---|
| ☐ | `all_images.csv` (with `group_id`) | 8, 9 |
| ☐ | `class_mapping.csv` | 8 |
| ☐ | `train.csv` / `val.csv` / `test_indist.csv` | 9 |
| ☐ | `test_crossdomain.csv` — **open once** | 9 |
| ☐ | `openset_categories.csv` | 7 |

### `artifacts/checkpoints/`, `onnx/`, `engines/`
| ✓ | File | Step |
|---|---|---|
| ☐ | `stage1.pt` | 12 |
| ☐ | `stage2.pt` | 13 |
| ☐ | `model_b.pt` | 30 |
| ☐ | `model_a_fused.onnx` | 17 |
| ☐ | `model_a_sim.onnx` | 17 |
| ☐ | `model_a_fp16.engine` — **built on the Nano** | 18 |

### `artifacts/reports/` — this is your PPT evidence pack
| ✓ | File | Step | Goes on a slide? |
|---|---|---|---|
| ☐ | `data_inventory.txt` | 6 | no |
| ☐ | `class_distribution.txt` | 8 | maybe |
| ☐ | `bias_audit.json` | 10 | **yes — differentiator** |
| ☐ | `aug_samples.png` | 11 | yes |
| ☐ | `stage1_metrics.json` | 12 | **yes — ablation** |
| ☐ | `stage2_metrics.json` | 13 | **yes — ablation** |
| ☐ | `eval_indist.json` | 14 | **yes** |
| ☐ | `eval_crossdomain.json` | 14 | **yes — the honest number** |
| ☐ | `confusion_matrix.png` | 14 | **yes** |
| ☐ | `per_class_recall.csv` | 14 | yes |
| ☐ | `gradcam_panel.png` | 14 | **yes** |
| ☐ | `ood_metrics.json` | 15 | yes |
| ☐ | `threshold_sweep.png` | 15 | yes |
| ☐ | `trt_benchmark.txt` | 18 | **yes — measured latency** |
| ☐ | `pipeline_dryrun.txt` | 34 | yes |
| ☐ | `trap_vs_manual.png` | 31 | yes |
| ☐ | `FINAL_RESULTS.md` | 40 | **yes — the pack** |

---

## 8. HUMAN DELIVERABLES

| ✓ | Item | Step | Notes |
|---|---|---|---|
| ☐ | Kaggle API token + competition rules **accepted** | 5 | CLI 403s until you click accept |
| ☐ | Mendeley datasets downloaded via browser | 6 | no CLI available |
| ☐ | `not_crop` collection (~1,800 images) | 7 | soil, sky, hands, pavement, blur, ImageNet |
| ☐ | **Open-set collection (~500), category-disjoint** | 7 | weeds, pipe, plastic, machinery, textures |
| ☐ | Jetson Nano flashed, headless, swap, MAXN | 4 | verify CC = (5,3) |
| ☐ | Sensors purchased and wired | 23 | see Report 4 §7 |
| ☐ | Grey card + ruler photo for trap calibration | 25, 31 | `MM_PER_PIXEL` |
| ☐ | Demo field / potted plants / printed leaves | 38 | say which on the slide |
| ☐ | Published CWSI baselines + ETLs sourced | 24, 32 | cite every number |

---

## 9. PRE-SUBMISSION FINAL CHECK

Run through this the day before you submit.

**Code**
- ☐ `python -m pytest tests/ -v` → all green
- ☐ `git diff --stat core/` → no changes since the initial copy
- ☐ No `import pycuda.autoinit` anywhere: `grep -rn "pycuda.autoinit" edge/ | wc -l` → 0
- ☐ No literal `8` used as a batch size: `grep -rn "batch.*=.*8\b" edge/ train/`
- ☐ Fused model contains the channel swap: `grep -n "\[2, 1, 0\]" train/export_onnx.py`
- ☐ `edge/` parses under Python 3.6

**Numbers**
- ☐ Every number in the PPT traces to a file in `artifacts/reports/`
- ☐ No PlantVillage accuracy quoted anywhere
- ☐ Both accuracy numbers shown (in-distribution and cross-domain)
- ☐ Per-class recall included, not just macro-F1
- ☐ Nano latency measured on your own board, not estimated

**Honesty**
- ☐ Printed-leaf footage disclosed on the slide, if used
- ☐ CWSI baselines cited as published, with local calibration named as future work
- ☐ Aerial nutrient stated as relative, not absolute
- ☐ Wheat marked roadmap if its numbers are weak
- ☐ Limitations slide written by you, before a judge finds them

**Rehearsed answers** — see Report 4 §10
- ☐ "What's your accuracy?" — both numbers, and why the second is honest
- ☐ "Why isn't it 99%?" — the bias audit
- ☐ "Why not YOLO?" — diffuse symptoms, class-labelled data
- ☐ "How does a drone detect insects?" — it detects damage; the trap counts insects
- ☐ "Why edge not cloud?" — connectivity, latency, cost, data ownership
- ☐ "Isn't the Nano obsolete?" — that's the point; here is the measured latency

---

## 10. Priority if you run short

**Never cut these four.** They are what separate this from a project that claims 99% and collapses under questioning:
`bias_audit.py` (10) · `evaluate.py` cross-domain (14) · `build_engine.sh` (18) · failure rehearsal (36)

**Cut in this order:** wheat classes → Stage 3 distillation → segmentation (never start it) → trap node (the classifier already covers pest *damage*) → drone flight (present the rover plus the architecture).

**And the one line that matters most:** you have four document versions, three audit rounds, five tested modules, and zero trained models. Files 1–20 of this checklist exist to change that. Everything after STEP 15 is optional in a way that Steps 1–15 are not.
