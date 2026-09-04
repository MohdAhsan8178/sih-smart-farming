# ULTIMATE IMPLEMENTATION & EXECUTION PLAN
### Smart Farming Assistant — SIH 2026 · Build order for an agentic IDE (Google Antigravity)

**Version:** 1.0 · 3 September 2026
**Audience:** (a) the AI agent executing the build, (b) Ahsan
**Reference documents (must be in the workspace):**
`SIH_Smart_Farming_AI_Report_4.md` · `AI_Handbook_4.md` · `sih_pipeline_v4/` (five tested Python modules + tests)
**Hard deadline:** SIH portal submission **20 September 2026**. Today is 3 September. **17 days.**

---

# PART 0 — HOW TO USE THIS DOCUMENT

## 0.1 Bootstrap prompt — paste this into Antigravity first

Copy this verbatim as your first message to the agent. It orients the agent before any work starts.

```
You are building a Smart India Hackathon 2026 project: an edge-deployed smart
farming assistant. Everything you need is in this workspace.

READ THESE FIRST, IN THIS ORDER, BEFORE WRITING ANY CODE:
  1. ULTIMATE_IMPLEMENTATION_PLAN_1.md   <- your build order. Follow it step by step.
  2. BUILD_CHECKLIST_1.md                <- what each file must contain, verify against this
  3. AI_Handbook_4.md                    <- models, datasets, training, export
  4. SIH_Smart_Farming_AI_Report_4.md    <- system architecture, sensors, scripts
  5. sih_pipeline_v4/                    <- five ALREADY-TESTED Python modules, import only

IGNORE THESE — present in the folder as history only, DO NOT read or reference:
  - Any file ending _1.md, _2.md, or _3.md EXCEPT ULTIMATE_IMPLEMENTATION_PLAN_1.md
    and BUILD_CHECKLIST_1.md themselves (only _4 versions of the Handbook/Report are current)
  - RED_TEAM_ADJUDICATION_*.md (background only — the fixes are already in the _4 docs
    and sih_pipeline_v4/, do not re-derive them)
    
RULES YOU MUST FOLLOW:
- Work through ULTIMATE_IMPLEMENTATION_PLAN_1.md one STEP at a time, in order.
- After each step, run its stated Acceptance Criteria and REPORT the actual
  output. Do not proceed to the next step until they pass.
- The five modules in sih_pipeline_v4/ are already written and tested. IMPORT
  them. Never rewrite, never copy their code into a new file, never "improve"
  them without being asked.
- Never invent numbers. If you have not measured an accuracy, latency, or file
  count, say "not measured yet". Do not write plausible placeholders.
- Never silently swallow an exception. If something fails, stop and report it.
- Target hardware is a 2019 Jetson Nano: Python 3.6, CUDA 10.2, TensorRT 8.2,
  NO INT8 support. Code that must run on it cannot use PyTorch, f-string '=',
  dataclasses from 3.7+, or multiprocessing.shared_memory (3.8+).
- Training code runs on a dev machine / Colab with modern Python. Runtime code
  runs on the Nano with Python 3.6. Keep them in separate directories.

Start with STEP 1. Tell me what you are about to do, then do it.
```

## 0.2 Rules the agent must not break

These are the failure modes that have already cost this project three rounds of rework. They are non-negotiable.

| # | Rule | Why |
|---|---|---|
| R1 | **Import `sih_pipeline_v4/` modules. Never retype their logic.** | A bug was fixed in one document section and left broken in another. One definition cannot drift. |
| R2 | **Never put PlantVillage images in validation or test.** | A model trained on 8 background pixels of it scores 49% across 38 classes. It measures capture conditions, not pathology. |
| R3 | **Never use INT8 on the Jetson Nano.** FP16 only. | Maxwell GM20B (compute capability 5.3) has no INT8 hardware. |
| R4 | **ONNX opset ≤ 13, static shapes, batch = 9.** | TensorRT 8.2 rejects newer opsets. The tile grid is 3×3 = 9. |
| R5 | **The fused export model must contain `x[:, [2,1,0], :, :]`.** | OpenCV gives BGR; the backbone expects RGB. Omitting this silently destroys accuracy. |
| R6 | **Report macro-F1, not accuracy, and always report per-class recall.** | Class imbalance reaches 40:1. Accuracy rewards ignoring rare diseases. |
| R7 | **Never invent a metric.** Unmeasured = "not measured yet". | Fabricated numbers in a PPT are a viva death sentence. |
| R8 | **Run `pytest sih_pipeline_v4/tests/` after touching anything it covers.** | 15 tests, each guarding a bug that already shipped once. |
| R9 | **No `import pycuda.autoinit` anywhere.** Create the CUDA context inside the thread that uses it. | It binds the context to the importing thread and crashes elsewhere. |
| R10 | **Two Python environments, never mixed.** `train/` = modern Python. `edge/` = Python 3.6-compatible. | The Nano cannot run modern PyTorch. Ever. |

## 0.3 Step format

Every step below uses the same structure. The agent must satisfy **Acceptance** before moving on.

```
STEP n — Name                           [AGENT | HUMAN | BOTH]   ~duration
Depends on : which steps must be done first
Goal       : one sentence
Build      : exact files to create
Commands   : exact commands to run
Acceptance : how to VERIFY it worked (must be checkable, not "looks fine")
DO NOT     : specific mistakes to avoid at this step
If it fails: the fallback
```

## 0.4 Master timeline

| Phase | Days | Steps | Outcome |
|---|---|---|---|
| **A — Foundation** | 3–4 Sep | 1–5 | Repo, env, data downloaded |
| **B — Data** | 4–5 Sep | 6–10 | Clean splits, bias audit, taxonomy locked |
| **C — Model A** | 5–6 Sep | 11–15 | Trained classifier, measured numbers |
| **D — Export** | 6–7 Sep | 16–19 | TensorRT engine on the Nano, benchmarked |
| **E — Internal demo** | 7–8 Sep | 20–22 | Rover demo working |
| **F — Agronomy scripts** | 9–11 Sep | 23–28 | CWSI, nutrient, risk rules, advisory |
| **G — Trap node** | 11–12 Sep | 29–32 | Model B + gateway counting |
| **H — Full integration** | 13–14 Sep | 33–36 | End-to-end pipeline on Nano |
| **I — Drone** | 15–16 Sep | 37–39 | Flight capture, footage |
| **J — Submission** | 17–19 Sep | 40–43 | Video, PPT, submit |
| **Buffer** | 20 Sep | — | Submit with a day spare |

**The single highest-risk step is STEP 18 (TensorRT engine build). It is scheduled on Day 4, not Day 15, deliberately.** If it fails you have thirteen days to recover instead of one.

---

# PART 1 — FOUNDATION (Days 1–2)

---

## STEP 1 — Create the repository scaffold                    [AGENT] ~15 min

**Depends on:** nothing
**Goal:** One repo with a strict separation between modern-Python training code and Python 3.6-compatible edge code.

**Build:**

```
sih-smart-farming/
├── README.md
├── requirements-train.txt          # modern Python (dev machine / Colab)
├── requirements-edge.txt           # Python 3.6 compatible (Jetson Nano)
├── configs/
│   ├── classes.py                  # <- COPY from sih_pipeline_v4/, do not rewrite
│   ├── paths.py                    # all filesystem paths in ONE place
│   └── train_config.py             # all hyperparameters in ONE place
├── core/                           # <- the five TESTED modules. Import these.
│   ├── aggregate.py
│   ├── rejection.py
│   ├── thermal.py
│   └── trap_segmentation.py
├── train/                          # modern Python only. Never runs on the Nano.
│   ├── download_data.py
│   ├── build_manifest.py
│   ├── dedup.py
│   ├── split.py
│   ├── bias_audit.py
│   ├── transforms.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_model_a.py
│   ├── train_model_b.py
│   ├── evaluate.py
│   ├── calibrate.py
│   └── export_onnx.py
├── edge/                           # Python 3.6 compatible. Runs on the Nano.
│   ├── build_engine.sh
│   ├── trt_classifier.py
│   ├── frame_gate.py
│   ├── tiler.py
│   ├── sensors.py
│   ├── agronomy.py
│   ├── rules_engine.py
│   ├── advisory.py
│   ├── storage.py
│   └── pipeline.py                 # the main runtime loop
├── gateway/                        # trap node processing (laptop or Nano)
│   ├── trap_receiver.py
│   └── trap_count.py
├── dashboard/
│   └── app.py
├── firmware/
│   └── esp32cam_trap/esp32cam_trap.ino
├── tests/
│   └── test_pipeline.py            # <- COPY from sih_pipeline_v4/tests/
├── data/                           # gitignored
│   ├── raw/  interim/  processed/
├── splits/
├── artifacts/
│   ├── checkpoints/  onnx/  engines/  reports/
└── docs/                           # the four reference .md files live here
```

**Commands:**
```bash
mkdir -p sih-smart-farming/{configs,core,train,edge,gateway,dashboard,tests,splits,docs}
mkdir -p sih-smart-farming/data/{raw,interim,processed}
mkdir -p sih-smart-farming/artifacts/{checkpoints,onnx,engines,reports}
mkdir -p sih-smart-farming/firmware/esp32cam_trap
cd sih-smart-farming && git init
printf 'data/\nartifacts/\n__pycache__/\n*.pyc\n.pytest_cache/\n*.engine\n*.onnx\nkaggle.json\n.env\n' > .gitignore
```

Then copy the pre-tested modules in:
```bash
cp <path>/sih_pipeline_v4/configs/classes.py       configs/
cp <path>/sih_pipeline_v4/aggregate.py             core/
cp <path>/sih_pipeline_v4/rejection.py             core/
cp <path>/sih_pipeline_v4/thermal.py               core/
cp <path>/sih_pipeline_v4/trap_segmentation.py     core/
cp <path>/sih_pipeline_v4/tests/test_pipeline.py   tests/
```

Fix the imports in `tests/test_pipeline.py` so they resolve against `core/` and `configs/`.

**Acceptance:**
```bash
python -m pytest tests/ -v          # must print: 15 passed
git log --oneline                   # must show an initial commit
```

**DO NOT:** modify the logic inside the five copied modules. Adjusting import paths is fine; changing a threshold, a comparison, or a branch is not.

**If it fails:** if the tests error on imports, add a `conftest.py` at repo root containing `import sys, os; sys.path.insert(0, os.path.dirname(__file__))`.

---

## STEP 2 — Centralise every path and constant             [AGENT] ~20 min

**Depends on:** 1
**Goal:** No magic strings or numbers anywhere else in the codebase. Two files disagreeing about a constant is exactly how the batch-8-versus-9-tiles bug happened.

**Build `configs/paths.py`:**
```python
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW, INTERIM, PROCESSED = DATA / 'raw', DATA / 'interim', DATA / 'processed'
SPLITS = ROOT / 'splits'
ARTIFACTS = ROOT / 'artifacts'
CKPT, ONNX_DIR, ENGINE_DIR, REPORTS = (ARTIFACTS / 'checkpoints',
                                       ARTIFACTS / 'onnx',
                                       ARTIFACTS / 'engines',
                                       ARTIFACTS / 'reports')
# dataset subdirectories under RAW
PADDY, SUGAR_THITE, SUGAR_DAPHAL = RAW/'paddy_doctor', RAW/'sugarcane_thite', RAW/'sugarcane_daphal'
PLANTDOC, PLANTWILD = RAW/'plantdoc', RAW/'plantwild'
NOTCROP, OPENSET = RAW/'not_crop', RAW/'openset'
for p in (RAW, INTERIM, PROCESSED, SPLITS, CKPT, ONNX_DIR, ENGINE_DIR, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
```

**Build `configs/train_config.py`:**
```python
# ---- SHARED between training and edge. Changing these breaks the engine. ----
IMAGE_SIZE   = 224      # model input
TILE_SIZE    = 320      # tile cut from the frame before resize to IMAGE_SIZE
TILE_GRID    = 3        # 3x3
N_TILES      = TILE_GRID * TILE_GRID          # 9  <- MUST equal engine batch
TILE_OVERLAP = 0.20
ENGINE_BATCH = N_TILES                        # 9. Never hardcode 8 anywhere.
ONNX_OPSET   = 13                             # TensorRT 8.2 ceiling

# ---- training only ----
BACKBONE      = 'tf_efficientnet_lite0'
BATCH_SIZE    = 64
EPOCHS        = 30
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 0.05
WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.1
DROP_RATE     = 0.2
DROP_PATH     = 0.1
EMA_DECAY     = 0.9998
GRAD_CLIP     = 1.0
SEED          = 42

# ---- decision thresholds. Fitted in STEP 15, written back here. ----
TAU_DISEASE = 0.55      # per-tile disease probability floor
TAU_MARGIN  = 0.10      # disease must beat healthy on the SAME tile by this
TAU_CONF    = 0.60      # abstain below this (on UNADJUSTED probabilities)
TAU_ENERGY  = None      # <- fit in STEP 15 on the OPEN-SET data, not not_crop
T_CAL       = None      # <- fit in STEP 15 (temperature scaling)
TAU_PRIOR   = 1.0       # post-hoc logit adjustment strength
CELL_K, CELL_N, CELL_MIN_SCORE = 2, 3, 0.55
```

**Acceptance:** `python -c "from configs.paths import ROOT; from configs.train_config import N_TILES, ENGINE_BATCH; assert N_TILES == ENGINE_BATCH == 9; print('ok')"`

**DO NOT:** write `224`, `9`, `0.55` or any path literal anywhere else in the codebase. Import them.

---

## STEP 3 — Set up the two Python environments             [BOTH] ~30 min

**Depends on:** 1

**`requirements-train.txt`** (dev machine / Colab, Python 3.10+):
```
torch>=2.0
torchvision
timm>=1.0.9
albumentations>=1.4
opencv-python
numpy
pandas
scikit-learn
scipy
imagehash
Pillow
matplotlib
seaborn
tqdm
onnx
onnxsim
onnxruntime
grad-cam
pytest
kaggle
```

**`requirements-edge.txt`** (Jetson Nano, Python 3.6 — install only what is missing):
```
numpy==1.19.5
pycuda==2020.1
Pillow==8.4.0
requests==2.27.1
# opencv and tensorrt come WITH JetPack 4.6.4. Do NOT pip install them.
```

**Commands (dev machine):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
python -c "import timm, torch, cv2, albumentations; print(timm.__version__, torch.__version__)"
python -c "import timm; print([m for m in timm.list_models('*efficientnet_lite*', pretrained=True)])"
```

**Acceptance:** the last command prints a list containing `tf_efficientnet_lite0`. If it does not, the timm version is too old — upgrade before continuing.

**DO NOT:** `pip install opencv-python` or `tensorrt` on the Jetson. Both ship with JetPack and pip versions will conflict.

---

## STEP 4 — Verify the Jetson Nano is usable                [HUMAN] ~45 min

**Depends on:** nothing (run in parallel with STEP 3)
**Goal:** Confirm the target board actually works *before* building anything for it.

**Commands (on the Nano):**
```bash
cat /etc/nv_tegra_release              # expect R32 (JetPack 4.6.x)
python3 --version                      # expect 3.6.9
python3 -c "import tensorrt; print(tensorrt.__version__)"    # expect 8.2.x
python3 -c "import cv2; print(cv2.__version__)"
nvcc --version                         # expect CUDA 10.2
free -h                                # confirm 4GB
ls /usr/src/tensorrt/bin/trtexec       # must exist

sudo systemctl set-default multi-user.target    # boot headless, frees ~0.5-1GB
sudo fallocate -l 6G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo reboot
# after reboot:
sudo nvpmodel -m 0 && sudo jetson_clocks
pip3 install pycuda==2020.1
python3 -c "import pycuda.driver as d; d.init(); print(d.Device(0).name(), d.Device(0).compute_capability())"
```

**Acceptance:** the last command prints a device name and `(5, 3)`. **Compute capability 5.3 confirms no INT8 hardware** — this is expected, not a problem.

**DO NOT:** try to install JetPack 5 or 6. They do not support the original Nano and never will.

**If it fails:** if the Nano does not boot or TensorRT is missing, reflash JetPack 4.6.4 today. If the board is dead, proceed with the plan anyway — STEP 19 has an ONNX-Runtime-on-CPU fallback and STEP 34 has a laptop-gateway fallback. Do not let a hardware problem block Days 1–3.

---

## STEP 5 — Set up Kaggle and Mendeley access              [HUMAN] ~15 min

**Depends on:** 3
**Goal:** Credentials in place so STEP 6 can run unattended.

**Actions (human — an agent cannot click through consent forms):**
1. kaggle.com → Account → Create New API Token → save `kaggle.json`
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. **Open https://www.kaggle.com/competitions/paddy-disease-classification and click "I Understand and Accept".** The CLI download fails with a 403 until you do this.
4. Open https://data.mendeley.com/datasets/355y629ynj and https://data.mendeley.com/datasets/9424skmnrk/1 and note the direct download links.

**Acceptance:** `kaggle competitions list -s paddy` returns rows without an auth error.

**DO NOT:** commit `kaggle.json`. It is in `.gitignore` — verify with `git check-ignore -v kaggle.json`.

---

# PART 2 — DATA (Days 2–3)

> This part decides your accuracy. The previous version of this project failed because of data, not architecture. Do not rush Part 2 to reach Part 3 sooner.

---

## STEP 6 — Download all datasets                          [BOTH] ~90 min

**Depends on:** 2, 3, 5
**Goal:** Every image on disk under `data/raw/`, with a manifest of what arrived.

**Build `train/download_data.py`** — a script that downloads what it can and prints clear manual instructions for what it cannot.

| Dataset | Method | Target dir | Expect |
|---|---|---|---|
| Paddy Doctor | `kaggle competitions download -c paddy-disease-classification` | `raw/paddy_doctor/` | ~10,407 train images, 10 class folders, `train.csv` |
| Sugarcane (Thite) | Mendeley `355y629ynj` — browser download, then unzip | `raw/sugarcane_thite/` | ~6,748 images, 11 folders |
| Sugarcane (Daphal) | Mendeley `9424skmnrk` — browser download | `raw/sugarcane_daphal/` | ~2,521 images, 5 folders |
| PlantDoc | `git clone https://github.com/pratikkayal/PlantDoc-Dataset` | `raw/plantdoc/` | ~2,598 images |
| PlantWild | github.com/tqwei05/MVPDR, UQRDM password `plantwildv1` | `raw/plantwild/` | ~18,542 images |

**Commands:**
```bash
python train/download_data.py --all
# then follow any printed MANUAL DOWNLOAD instructions, and re-run:
python train/download_data.py --verify
```

**Acceptance:** `--verify` prints a table of dataset → file count → class-folder count, and every row is non-zero. Save it to `artifacts/reports/data_inventory.txt`.

**DO NOT:**
- Do not download PlantVillage at this step. You do not need it (see rule R2). If you later want it as auxiliary pretraining, it goes in `raw/plantvillage/` and is used only for that.
- Do not let the agent "simulate" a download or generate placeholder images. If a download fails, it must say so.

**If it fails:** the Kaggle competition variant gives 10 classes; the full 13-class version is on IEEE DataPort. Ten classes is enough — proceed rather than stalling. A mirror without competition signup: `kaggle datasets download -d imbikramsaha/paddy-doctor`.

---

## STEP 7 — Assemble the `not_crop` and open-set collections   [HUMAN] ~60 min

**Depends on:** 6
**Goal:** Two collections that are **disjoint by category**, not merely by file.

This is the step most likely to be done wrong, so read the distinction carefully.

| Collection | Target | Contents | Used for |
|---|---|---|---|
| `raw/not_crop/` | ~1,800 images | soil, sky, hands, feet/shoes, pavement, walls, blurred frames, ~500 random ImageNet | **Training** the `not_crop` class (outlier exposure) |
| `raw/openset/` | ~500 images | **categories never in `not_crop`**: broadleaf weeds, plastic mulch, irrigation pipe, farm machinery, brick, straw/mulch, DTD textures | **Calibrating** `TAU_ENERGY` (STEP 15) |

**Why they must be different categories:** once you train a `not_crop` class on soil, soil is *in-distribution* for that model — its energy score looks perfectly normal. Calibrating a threshold on it measures nothing. The OOD literature requires auxiliary training outliers to be entirely disjoint from evaluation outliers.

**Actions:** phone photos around campus plus image search. Twenty minutes of shooting covers most of `not_crop`. Record the category of every folder in `splits/openset_categories.csv` so you can show a judge the two sets do not overlap.

**Acceptance:**
```bash
python -c "
from pathlib import Path
from configs.paths import NOTCROP, OPENSET
a={p.name for p in NOTCROP.iterdir() if p.is_dir()}
b={p.name for p in OPENSET.iterdir() if p.is_dir()}
assert not (a & b), f'OVERLAPPING CATEGORIES: {a & b}'
print('not_crop cats:', sorted(a)); print('openset cats:', sorted(b))
"
```

**DO NOT:** create `openset/` by holding out 20% of `not_crop/`. That is the exact mistake this step exists to prevent.

---

## STEP 8 — Build the unified manifest and class taxonomy   [AGENT] ~45 min

**Depends on:** 6, 7
**Goal:** One CSV describing every image, with the class mapping decisions recorded.

**Build `train/build_manifest.py`:**
- Walk every raw dataset directory
- Map each source folder name → a canonical class from `configs/classes.py`
- Emit `splits/all_images.csv` with columns: `path, label, source_dataset, orig_folder`
- Emit `splits/class_mapping.csv` recording every merge decision — a judge may ask why "brown rust" and "rust" became one class

**Required merges (sugarcane):** `brown_rust` + `rust` → `sugarcane__rust`; `yellow` + `yellow_leaf_disease` → `sugarcane__yellow_leaf`. Take `red_rot` from the Daphal set only.

**Acceptance:**
```bash
python train/build_manifest.py
python -c "
import pandas as pd
from configs.classes import CLASS_NAMES
df = pd.read_csv('splits/all_images.csv')
unknown = set(df.label) - set(CLASS_NAMES)
assert not unknown, f'labels not in taxonomy: {unknown}'
print(df.label.value_counts())
print('TOTAL:', len(df))
"
```
Expect roughly 27,000–30,000 rows. Save the class distribution to `artifacts/reports/class_distribution.txt`.

**DO NOT:** invent class names. Every label must already exist in `configs/classes.py`. If a source folder does not map cleanly, log it and skip — do not guess.

---

## STEP 9 — Deduplicate and split                          [AGENT] ~60 min

**Depends on:** 8
**Goal:** Splits where near-duplicate images never straddle train and test.

**Build `train/dedup.py`:** perceptual hash (`imagehash.phash`, hash_size=8), group images with Hamming distance ≤ 5 into a shared `group_id`, and also catch exact MD5 duplicates across the two sugarcane datasets (both are from Maharashtra and may share images). Adds a `group_id` column.

**Build `train/split.py`:** `StratifiedGroupKFold` grouped on `group_id`, stratified on `label`. Roughly 80/10/10.

Outputs:
- `splits/train.csv`, `splits/val.csv`, `splits/test_indist.csv`
- `splits/test_crossdomain.csv` — built separately from PlantDoc + PlantWild rice/sugarcane classes. **Never trained on. Opened exactly once, in STEP 14.**

**Commands:**
```bash
python train/dedup.py        # ~20-30 min on 30k images
python train/split.py
```

**Acceptance:**
```bash
python -c "
import pandas as pd
tr,va,te = [pd.read_csv(f'splits/{s}.csv') for s in ('train','val','test_indist')]
g = [set(d.group_id) for d in (tr,va,te)]
assert not (g[0]&g[1]) and not (g[0]&g[2]) and not (g[1]&g[2]), 'GROUP LEAK ACROSS SPLITS'
p = [set(d.path) for d in (tr,va,te)]
assert not (p[0]&p[1]) and not (p[0]&p[2]), 'PATH LEAK'
print({'train':len(tr),'val':len(va),'test':len(te)})
print('classes present in all three:', len(set(tr.label)&set(va.label)&set(te.label)))
"
```

**DO NOT:** split by image, or use a plain random split. Five photos of the same leaf in both train and test inflate your accuracy and hide the problem you are trying to solve.

**If it fails:** if dedup takes too long, bucket by the first 16 hash bits and compare only within buckets.

---

## STEP 10 — Run the dataset bias audit                    [AGENT] ~30 min

**Depends on:** 9
**Goal:** Measure whether your labels leak through capture conditions. This produces one of your best slides.

**Build `train/bias_audit.py`:** extract 8 background pixels per image (four corners + four edge midpoints → 24-dim vector), train a `RandomForestClassifier` on **only** those, evaluate on the test split. Chance is 1/31 = 3.2%.

**Commands:**
```bash
python train/bias_audit.py --out artifacts/reports/bias_audit.json
```

**Acceptance:** the JSON contains `background_only_accuracy` and `chance_accuracy`. Interpret:

| Result | Meaning | Action |
|---|---|---|
| 3–6% | Clean. | Put it on a slide. |
| 15–30% | Mild leakage. | Increase `RandomResizedCrop` aggressiveness and `CoarseDropout` in STEP 11. |
| >40% | Serious, PlantVillage-like. | Identify which classes separate and consider dropping or re-sourcing them. |

**DO NOT:** skip this because it feels like a detour. It takes 30 minutes and it is the direct evidence that you diagnosed the previous project's failure correctly.

---

# PART 3 — MODEL A (Days 3–4)

---

## STEP 11 — Build the data pipeline                        [AGENT] ~45 min

**Depends on:** 9
**Goal:** Dataset and augmentation code matching the deployment domain.

**Build `train/transforms.py`** — exactly the recipe in `AI_Handbook_4.md` §6.6. The critical elements, and why each is there:

| Transform | Why |
|---|---|
| `RandomResizedCrop(scale=(0.35, 1.0))` | **Most important line.** Repeatedly removes background context the model would use as a shortcut, and matches tile-based inference. |
| `RandomShadow`, `RandomSunFlare`, `RandomToneCurve` | Indian midday field light is nothing like a photo studio |
| `MotionBlur`, `Defocus` | You are on a moving drone |
| `ISONoise`, `ImageCompression(45–95)` | Deployment camera is an IMX219, not a CAT S62 Pro |
| `CoarseDropout` | Forces use of multiple leaf regions, not one memorised patch |
| `VerticalFlip(p=0.3)` | A nadir drone view has no natural "up" |

**Build `train/dataset.py`:**
- Reads a split CSV, loads with `cv2.imread` then **`cv2.cvtColor(..., COLOR_BGR2RGB)`**
- **Slicing-aided fine-tuning:** for each training image, additionally emit 2–4 random 320×320 crops with the same label, so the training distribution matches tile-based inference
- Pre-resize all images on disk to max side 512 first (one-off) — Paddy Doctor is 1080×1440 and full-size JPEG decode will bottleneck the dataloader

**Acceptance:**
```bash
python -c "
from train.dataset import build_loaders
tl, vl = build_loaders()
x, y = next(iter(tl))
print(x.shape, x.dtype, float(x.min()), float(x.max()), y.shape)
assert x.shape[1] == 3 and x.shape[2] == 224
"
```
Then **visually inspect**: save a 4×4 grid of augmented samples to `artifacts/reports/aug_samples.png` and look at it. Leaves should still be recognisable — if augmentation destroys the lesion, it is too strong.

**DO NOT:** add hue shifts beyond ±12. Colour is diagnostic information for plant disease; destroying it destroys your signal.

---

## STEP 12 — Train Stage 1 (baseline)                       [AGENT] ~1.5 h

**Depends on:** 11
**Goal:** A deliberately plain baseline so the augmentation delta can be measured. **This exists to be beaten.**

**Build `train/model.py`:** flat 31-class head via `timm.create_model(BACKBONE, pretrained=True, num_classes=31, drop_rate, drop_path_rate)`. Nothing else. No attention modules, no ensembles.

**Build `train/train_model_a.py`:** with a `--stage` flag.
- Stage 1: standard augmentation only (resize, flip, normalize), no class balancing, no distillation
- AdamW, discriminative LR (backbone 1e-4, head 1e-3), OneCycle, EMA, AMP
- **Select checkpoints on macro-F1, never accuracy**
- Log to `artifacts/reports/stage1_metrics.json`

**Commands:**
```bash
python train/train_model_a.py --stage 1 --epochs 30 --out artifacts/checkpoints/stage1.pt
```

**Acceptance:** `stage1_metrics.json` contains `val_macro_f1`, `val_top1`, and `per_class_recall` for all 31 classes. Record the macro-F1 — it is the "before" number in your headline slide.

**DO NOT:** tune anything at this stage. A tuned baseline understates your improvement.

---

## STEP 13 — Train Stage 2 (the real model)                 [AGENT] ~2 h

**Depends on:** 12
**Goal:** The model you actually ship. The Stage 1 → Stage 2 delta is your best slide.

Stage 2 adds, one at a time so you can attribute the gain:
1. Full field-conditioned augmentation (STEP 11)
2. Class balancing — `WeightedRandomSampler` with sqrt-inverse frequency + class-weighted CE with `label_smoothing=0.1`
3. Slicing-aided fine-tuning (tiles in the training set)
4. Mixup/CutMix

**Commands:**
```bash
python train/train_model_a.py --stage 2 --epochs 30 --out artifacts/checkpoints/stage2.pt
```

**Acceptance:**
- `stage2_macro_f1 > stage1_macro_f1`. If not, something is wrong — check the augmentation grid image from STEP 11 before continuing.
- Per-class recall table shows the rarest classes are **not** at zero. A class with 43 images at 0.00 recall means balancing is not working.
- Training accuracy looks bad while validation improves → **normal**, that is mixup. Watch validation.

**DO NOT:** run Stage 3 (distillation) if you are behind schedule. It is genuinely optional. A well-trained Stage 2 beats a rushed Stage 3.

**If it fails:** if the model collapses to predicting one class, lower the backbone LR to 5e-5 and disable mixup for the first 5 epochs.

---

## STEP 14 — Evaluate honestly                              [AGENT] ~45 min

**Depends on:** 13
**Goal:** The four numbers that go in your PPT, including the uncomfortable one.

**Build `train/evaluate.py`** producing:

| Output | File |
|---|---|
| In-distribution macro-F1 + top-1 | `artifacts/reports/eval_indist.json` |
| **Cross-domain macro-F1** (PlantDoc/PlantWild) | `artifacts/reports/eval_crossdomain.json` |
| Row-normalised confusion matrix | `artifacts/reports/confusion_matrix.png` |
| Per-class recall table | `artifacts/reports/per_class_recall.csv` |
| Grad-CAM panel: healthy / correct / failure | `artifacts/reports/gradcam_panel.png` |

**Commands:**
```bash
python train/evaluate.py --ckpt artifacts/checkpoints/stage2.pt --split test_indist
python train/evaluate.py --ckpt artifacts/checkpoints/stage2.pt --split test_crossdomain
python train/evaluate.py --ckpt artifacts/checkpoints/stage2.pt --gradcam
```

**Acceptance:** all five files exist and contain real measured values.

**Expect the cross-domain number to be much lower.** That gap is documented in the literature — lab-trained models tested on field data collapse to around 33%. Your job is to shrink and *measure* it, not to hide it. This is the "why isn't your accuracy 99%" answer, and delivered well it earns more credibility than the number itself.

**DO NOT:** open `test_crossdomain.csv` more than once, and never tune against it. The moment you tune on it, it stops being a held-out measurement.

---

## STEP 15 — Calibrate all thresholds                       [AGENT] ~45 min

**Depends on:** 14
**Goal:** Fill in every `None` in `configs/train_config.py` with a fitted value.

**Build `train/calibrate.py`** doing four things in this order:

1. **Temperature scaling** → `T_CAL`. LBFGS on validation NLL.
2. **Energy threshold** → `TAU_ENERGY`. Use `core.rejection.fit_energy_threshold(id_logits, openset_logits, CROP_COLS, tpr=0.95)`. **`openset_logits` comes from `raw/openset/` — the disjoint categories from STEP 7 — never from `not_crop`.** Report FPR@95TPR and AUROC.
3. **Post-hoc prior strength** → `TAU_PRIOR`. Sweep `[0, 0.25, 0.5, 0.75, 1.0]`, pick the best validation macro-F1.
4. **Aggregation thresholds** → `TAU_DISEASE`, `TAU_MARGIN`. Grid-sweep both, plot precision/recall for the DISEASE verdict, choose your operating point. Save the plot — it shows you tuned a decision rule rather than guessing a constant.

**Commands:**
```bash
python train/calibrate.py --ckpt artifacts/checkpoints/stage2.pt \
    --openset data/raw/openset --write-config
```

**Acceptance:**
```bash
python -c "
from configs.train_config import TAU_ENERGY, T_CAL
assert TAU_ENERGY is not None and T_CAL is not None, 'thresholds not fitted'
print('TAU_ENERGY', TAU_ENERGY, 'T_CAL', T_CAL)
"
```
Also save `artifacts/reports/ood_metrics.json` with FPR@95TPR and AUROC, and `artifacts/reports/threshold_sweep.png`.

**DO NOT:** hand-pick a threshold because it "looks right". Every one of these is fitted on data and recorded.

**If it fails:** if you cannot assemble the open-set collection in time, **log the energy score but do not apply a hard threshold** — rely on the `not_crop` class and the confidence gate. An uncalibrated threshold fails in an unpredictable direction and is worse than no threshold.

---

# PART 4 — EXPORT & EDGE DEPLOYMENT (Days 4–5)

> **This is the highest-risk part of the project and it is scheduled early on purpose.** If TensorRT export fails, you need days to recover, not hours.

---

## STEP 16 — Build the fused export model                   [AGENT] ~45 min

**Depends on:** 13
**Goal:** A model that takes raw uint8 BGR straight from OpenCV and does all preprocessing on the GPU.

**Build `train/export_onnx.py` containing:**

```python
class FusedModel(nn.Module):
    """Accepts (B, H, W, 3) uint8 BGR. Permute + BGR->RGB + scale + normalize on GPU."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Standard ImageNet stats in TRUE RGB order.
        # The channel swap happens on the TENSOR, not on these vectors.
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x_uint8):
        x = x_uint8.permute(0, 3, 1, 2)               # (B,3,H,W) — still BGR
        x = x[:, [2, 1, 0], :, :].float() / 255.0     # <-- BGR->RGB. NEVER OMIT.
        return self.backbone((x - self.mean) / self.std)
```

> **Rule R5 lives here.** An earlier version of this project reordered the mean/std vectors and forgot to reorder the channels, feeding BGR data to an RGB-trained backbone. Since hue separates chlorosis from rust from mosaic, that corrupts the most informative signal the model has — and it fails *silently*.

**Acceptance — three checks, and the first two are the ones that matter:**

```python
# CHECK 1 — RED FLAG IMAGE. Random noise CANNOT catch a channel swap because
# uniform noise has identical statistics in all three channels.
bgr = np.zeros((1,224,224,3), np.uint8)
bgr[...,2], bgr[...,1], bgr[...,0] = 200, 40, 20     # strongly RED
out_fused = fused(torch.from_numpy(bgr)).numpy()
rgb = cv2.cvtColor(bgr[0], cv2.COLOR_BGR2RGB).astype(np.float32)/255.
ref = ((rgb - MEAN)/STD).transpose(2,0,1)[None]
out_ref = backbone(torch.from_numpy(ref)).numpy()
assert np.abs(out_fused - out_ref).max() < 1e-3, 'CHANNEL ORDER MISMATCH'

# CHECK 2 — END TO END on real images loaded by cv2.imread
assert abs(f1_fused - f1_reference) < 0.005

# CHECK 3 — ONNX export fidelity (least important of the three)
```

**DO NOT:** verify only with `torch.randint` noise. That was exactly the check that let this bug through before.

---

## STEP 17 — Export to ONNX                                 [AGENT] ~20 min

**Depends on:** 16

**Commands:**
```bash
python train/export_onnx.py --ckpt artifacts/checkpoints/stage2.pt \
    --out artifacts/onnx/model_a_fused.onnx --batch 9 --opset 13
python -m onnxsim artifacts/onnx/model_a_fused.onnx artifacts/onnx/model_a_sim.onnx \
    --overwrite-input-shape input_bgr_uint8:9,224,224,3
python train/export_onnx.py --verify artifacts/onnx/model_a_sim.onnx
```

**Acceptance:** all three checks from STEP 16 pass against the **simplified ONNX file**, not just the PyTorch model.

**DO NOT:**
- Use `dynamic_axes`. Static shapes give TensorRT more to optimise and are more stable on TRT 8.
- Use batch 8. `ENGINE_BATCH` is 9 and comes from `configs/train_config.py` — the tile grid is 3×3.
- Use opset > 13. TensorRT 8.2 will reject it.

**If it fails:** if the fused graph will not export (permute or uint8 cast unsupported), fall back to the plain model with float RGB input plus explicit `cv2.cvtColor` on the host. The fusion is an optimisation, not a requirement — and the plain path makes the channel conversion visible in code.

---

## STEP 18 — Build the TensorRT engine on the Nano   ⚠ HIGHEST RISK   [BOTH] ~1 h

**Depends on:** 4, 17
**Goal:** A `.engine` file and a measured latency number, on the actual board.

**Commands (on the Nano):**
```bash
scp artifacts/onnx/model_a_sim.onnx  nano@<ip>:~/
sudo nvpmodel -m 0 && sudo jetson_clocks

/usr/src/tensorrt/bin/trtexec \
  --onnx=model_a_sim.onnx \
  --saveEngine=model_a_fp16.engine \
  --fp16 --workspace=1024 --verbose 2>&1 | tee build.log

/usr/src/tensorrt/bin/trtexec --loadEngine=model_a_fp16.engine \
  --iterations=200 --avgRuns=100
```

**Acceptance:**
- `model_a_fp16.engine` exists
- The benchmark prints a mean latency. **Write it down — it goes on your PPT.** Expect roughly 60–120 ms for a batch of 9 tiles.
- Engine outputs match PyTorch: run the test set through the engine and confirm macro-F1 is within 0.5% of the PyTorch number. A larger gap means an op fell back or overflowed, not a precision tradeoff.

**DO NOT:**
- `--int8`. Maxwell has no INT8 hardware. It will be no faster and less accurate.
- Build the engine on your laptop and copy it. Engines are not portable across TensorRT versions, GPU architectures, or drivers.
- Rely on swap during inference. Swap is for the build only; using it at runtime destroys latency.

**If it fails:**
1. Run `onnxsim` again with different settings; check `build.log` for the exact unsupported node
2. Drop to opset 11
3. Move down the backbone risk ladder (`AI_Handbook_4.md` §3.1) — rung 1 `tf_efficientnet_lite0` is the safest
4. **Fallback:** ONNX Runtime on the Nano's CPU. ~200–400 ms/image — slow, but your requirement is only 1–3 inferences/second, so it *works*
5. **Last resort:** run the model on a laptop and stream frames from the drone; present it as a ground-station architecture and say so

---

## STEP 19 — Build the TensorRT runtime wrapper             [AGENT] ~1 h

**Depends on:** 18
**Goal:** Python 3.6-compatible inference that never truncates tiles and never crashes on teardown.

**Build `edge/trt_classifier.py`:**

```python
# NO `import pycuda.autoinit` ANYWHERE IN THIS FILE. See rule R9.
import numpy as np, cv2, tensorrt as trt, pycuda.driver as cuda

class TRTClassifier:
    def __init__(self, engine_path, batch=9, size=224, num_classes=31):
        ...
        # fused graph takes uint8 NHWC BGR — no float conversion on the host
        self.h_in  = cuda.pagelocked_empty((batch, size, size, 3), np.uint8)
        self.h_out = cuda.pagelocked_empty((batch, num_classes), np.float32)

    def infer(self, bgr_tiles):
        """Returns RAW LOGITS. Not probabilities."""

    def infer_all(self, bgr_tiles):
        """Chunk-loop so an unexpected tile count can NEVER silently truncate."""
        if not bgr_tiles:
            return np.empty((0, self.nc), np.float32)
        return np.concatenate([self.infer(bgr_tiles[i:i+self.batch])
                               for i in range(0, len(bgr_tiles), self.batch)], axis=0)

    def close(self):
        """Free d_in, d_out, stream, h_in, h_out, context, engine, then gc.collect().
        ctx.detach() before these are freed raises LogicError and blocks restart."""
```

**Requirements:**
- `infer()` returns **raw logits**. The energy score must be computed on unadjusted logits, and returning probabilities throws that away.
- Always call `infer_all()` from the pipeline, never `infer()` directly.
- Python 3.6 compatible — no f-string `=`, no walrus, no dataclasses.

**Acceptance (on the Nano):**
```bash
python3 edge/trt_classifier.py --selftest --engine artifacts/engines/model_a_fp16.engine
```
Must print per-batch latency, confirm output shape `(9, 31)`, and exit cleanly with no `LogicError`.

**DO NOT:** call `min(len(tiles), batch)` and move on. That silently drops the 9th tile — and if that is where the lesion is, it is never classified.

---

## STEP 20 — Build the frame gate and tiler                 [AGENT] ~1 h

**Depends on:** 2
**Goal:** Reject 92–97% of frames before the model runs. **This is your single biggest optimisation** — worth 10–30×, far more than anything on the GPU.

**Build `edge/frame_gate.py`** — reject a frame unless **all** pass:

| Check | Method | Reject if |
|---|---|---|
| Altitude | rangefinder | outside 1.5–2.5 m inspection band |
| Attitude | IMU | \|roll\| or \|pitch\| above threshold (motion smear) |
| Sharpness | variance of Laplacian | below `TAU_BLUR` |
| Exposure | histogram | clipped at either end |
| Novelty | scene displacement vs last kept frame | below 60% |

**Build `edge/tiler.py`** — deterministic **3×3 grid, 320×320 tiles, 20% overlap**, keeping only tiles with >40% vegetation (use `core.thermal.vegetation_mask`, which uses an **absolute** ExG threshold, not Otsu). Pad to exactly `N_TILES` so the engine batch always matches.

**Acceptance:**
```bash
python3 edge/frame_gate.py --selftest    # pass rate on a sample video: expect 3-8%
python3 edge/tiler.py --selftest         # must ALWAYS return exactly 9 tiles
```

**DO NOT:** use Otsu for the vegetation mask. On a fully closed canopy the ExG histogram is unimodal and Otsu bisects it, reporting ~50% vegetation on a 100% green field.

---

## STEP 21 — Wire the runtime pipeline                      [AGENT] ~2 h

**Depends on:** 19, 20
**Goal:** The complete on-device loop, four threads, using the tested core modules.

**Build `edge/pipeline.py`:**

```
Thread 1  capture + GPS/IMU/altitude tagging  -> ring buffer
Thread 2  frame_gate -> tiler                 -> tile queue      (CPU, numpy/cv2)
Thread 3  TRTClassifier.infer_all             -> logits queue    (GPU)
Thread 4  rejection.decide -> aggregate_frame -> aggregate_cell
          -> rules_engine -> storage
```

**The CUDA context must be created inside Thread 3**, and the classifier must be constructed there too:

```python
def inference_thread(engine_path, tile_q, result_q, stop_evt):
    cuda.init()
    ctx = cuda.Device(0).make_context()      # belongs to THIS thread
    clf = None
    try:
        clf = TRTClassifier(engine_path)
        while not stop_evt.is_set():
            ...
    finally:
        if clf is not None: clf.close()      # ORDER MATTERS
        ctx.pop(); ctx.detach()
```

**Threading is correct here — do not switch to multiprocessing.** NumPy and OpenCV release the GIL for their heavy operations, and `multiprocessing.shared_memory` is Python 3.8+ while the Nano is 3.6.

**Use the tested modules for all decisions:**
```python
from core.rejection import decide
from core.aggregate import aggregate_frame, aggregate_cell
```

**Acceptance:**
```bash
python3 edge/pipeline.py --source test_video.mp4 --dry-run --report
```
Must print: frames seen, frames passing the gate, tiles classified, per-cell verdicts, and end-to-end scenes/second. Target **2–4 scenes/sec**. Must exit cleanly.

**DO NOT:** reimplement `aggregate_frame` or `decide`. Import them. That is rule R1 and it is the rule this project has broken most often.

---

## STEP 22 — Internal hackathon demo (rover)                [BOTH] ~4 h

**Depends on:** 21
**Goal:** A working demo for the internal round. No drone.

**Deliverables:** Nano + camera on the rover, live inference with a tile-probability heatmap overlay, a laptop screen showing verdicts, and the Stage 1 → Stage 2 ablation slide.

**Rehearse these three answers:**
1. *"What's your accuracy?"* → give both numbers, in-distribution and cross-domain, and explain why the second is the honest one
2. *"Why isn't it 99% like the papers?"* → because those evaluate on PlantVillage, where a model trained on 8 background pixels scores 49% against 2.6% chance. Show your bias audit.
3. *"Why a drone and not a rover?"* → two-tier survey-then-inspect; the drone's advantage is targeted revisit over terrain a rover can't cross, not speed

**Acceptance:** the demo runs for 5 minutes without a crash, and every number quoted was measured on your own hardware.

---

# PART 5 — DETERMINISTIC AGRONOMY (Days 7–9)

> None of Part 5 uses AI. Saying that confidently is a strength — it is what lets the whole system run on a 2019 board.

---

## STEP 23 — Sensor abstraction layer                       [AGENT] ~1 h

**Depends on:** 2
**Goal:** One interface for all sensors, with a simulation mode so nothing blocks on hardware.

**Build `edge/sensors.py`:**

| Sensor | Reads | Notes |
|---|---|---|
| MLX90640 (or MLX90614) | 32×24 thermal array (or single point) | I2C; **lock the refresh rate** |
| DHT22 / SHT31 | air temperature, RH → VPD | needed for CWSI |
| Capacitive soil probe ×2–3 | volumetric moisture | ADC |
| CSI camera | frames | **AWB and AE LOCKED to fixed gains** |
| GPS + IMU + rangefinder | position, attitude, altitude | frame tagging |

Every sensor class needs a `--simulate` mode returning plausible synthetic values, so Parts 5–8 can be built and tested before hardware arrives.

**Acceptance:** `python3 edge/sensors.py --selftest --simulate` prints one reading per sensor with units and a timestamp.

**DO NOT:** leave auto-white-balance on. A camera that re-balances every frame makes colourimetry meaningless, and STEP 25 depends on it.

---

## STEP 24 — CWSI and water stress                          [AGENT] ~1 h

**Depends on:** 23
**Goal:** Irrigation advice from a closed-form equation. No model.

**Use `core/thermal.py` — already written and tested.** Do not reimplement.

```python
from core.thermal import canopy_temperature, cwsi
tc, info = canopy_temperature(thermal_frame, air_temp_c=ta, veg_fraction=veg_frac)
if tc is None:
    log_rejection(info)     # 'no_vegetation_bare_soil' | 'implausibly_cold' | ...
else:
    index = cwsi(tc, ta, vpd, LL_SLOPE, LL_INTERCEPT, UL_OFFSET)
```

**Three rules that are easy to get wrong:**
1. **Compute CWSI on the inspection pass only (1.5–2.5 m), never the survey pass.** At 10 m an MLX90640 pixel spans ~32 cm, mixing 55–65 °C soil with 30 °C canopy — the average reads as severe stress on a healthy plant.
2. **The gate separates plant tissue from bare soil, not healthy from stressed.** A non-transpiring canopy *is* CWSI = 1.0 — the drought alarm. It must never be rejected.
3. **CWSI is valid only in the early-afternoon window under clear skies** (~11:00–16:00). Gate on time of day.

**Human task:** obtain published `(Tc−Ta)_LL` baselines for rice and sugarcane and put them in `configs/crop_baselines.py`. State in the PPT that you use published baselines and would calibrate locally in deployment.

**Acceptance:** `python3 -m pytest tests/ -k thermal -v` passes, plus a manual check that a synthetic 45.5 °C canopy at Ta = 38 is **accepted** (not rejected as soil) and 60 °C soil at the same Ta is **rejected**.

---

## STEP 25 — Nutrient index                                 [AGENT] ~1.5 h

**Depends on:** 23
**Goal:** Nitrogen status from leaf colour, scoped correctly by platform.

**Build `edge/agronomy.py`:**

| Platform | Output | Method |
|---|---|---|
| **Drone** | **Relative only** — "this cell is greener/yellower than the field median from this pass" | No card needed. Illumination cancels within a pass. |
| **Rover / phone** | **Absolute LCC panel 1–4** | Grey card in frame, von Kries gains on the **leaf ROI only** |

> You cannot hold a reference card in frame while flying over a flooded paddy. Relative comparison is the honest aerial answer — and it is still actionable, because it tells the farmer *where* to apply nitrogen.

**Implementation notes:**
- Apply white balance to the **leaf ROI (~200×200 px)**, never the full 1080p frame — that is ~25 MB per float32 copy for no benefit
- Do **not** white-balance the classifier's input. It was trained on un-corrected images with heavy colour augmentation; correcting it is a train/test mismatch
- OpenCV's `COLOR_BGR2LAB` on uint8 returns b with a **+128 offset** — this is `b* + 128`, not published CIELAB `b*`. Fine as a relative index; note it if comparing to literature

**Acceptance:** on the Mendeley *Nitrogen Deficiency of Rice Crop* dataset (4 LCC categories), the b\* index must be **monotonic** across the four panels. Save the plot.

---

## STEP 26 — Vegetation indices and growth staging          [AGENT] ~45 min

**Depends on:** 23
**Goal:** Survey-pass stress mapping without a multispectral camera.

Compute ExG, VARI, TGI, NGRDI per grid cell; track canopy cover fraction over time for growth staging.

**Be honest about the limits in the PPT:** RGB indices saturate at high biomass, and at least one comparative study found neither TGI nor VARI reliable as a general-purpose crop health indicator. **Use them for relative within-field comparison and change detection, not absolute health scoring** — which is exactly what flagging cells for the inspection pass needs.

**Acceptance:** given a test image, outputs a per-cell index map plus a flagged-cell list. Cells below the 20th percentile are flagged for inspection.

---

## STEP 27 — Environmental risk rules engine                [AGENT] ~1.5 h

**Depends on:** 23
**Goal:** Drought, flood, heat and disease-favourable alerts from thresholds over time series.

**Build `edge/rules_engine.py`:**

| Risk | Rule |
|---|---|
| Heat stress | consecutive hours above a crop-specific critical temperature (rice spikelet sterility rises sharply above ~35 °C at anthesis) |
| Flood | rainfall accumulation over 24/72 h + soil moisture saturation persistence |
| Drought | cumulative water-balance deficit + CWSI trend |
| Disease-favourable | leaf wetness duration × temperature windows |
| Pest pressure | trap counts vs Economic Threshold Levels + trend slope |

**Output is a structured JSON finding — this is the contract with STEP 28:**
```json
{"crop":"rice","condition":"brown_spot","confidence":0.83,"area_pct":12,
 "cell":[lat,lon],"cwsi":0.41,"soil_moisture_pct":22,
 "risk_flags":["heat_stress"],"action":"fungicide_targeted",
 "severity":"moderate","timestamp":"..."}
```

**Acceptance:** feed a synthetic 7-day sensor time series and confirm each rule fires at the right point and stays silent otherwise. Every alert must carry the rule that triggered it — explainability is the point of doing this in code.

---

## STEP 28 — Advisory generation (LLM as renderer only)     [AGENT] ~1.5 h

**Depends on:** 27
**Goal:** Farmer-readable Hindi/regional text, without letting the LLM make decisions.

**Build `edge/advisory.py`:**

```
rules_engine → structured finding (JSON) → LLM API → readable text
```

**The critical design rule:** the LLM **never diagnoses**. It receives a decided finding and verbalises it. An LLM hallucination cannot invent a disease or recommend a wrong chemical — the worst case is awkward phrasing.

**Requirements:**
- System prompt states explicitly: *"You are a translator. Render the finding below into simple [language] for a smallholder farmer. Do not add diagnoses, do not add recommendations not present in the JSON, do not change any number."*
- **Offline fallback:** a pre-written template per finding type. The whole edge claim collapses if the advisory needs internet.
- Support Hindi + one regional language + English
- SMS-length variant (160 chars)

**Acceptance:** with the network disabled, every finding type still produces sensible template output. This is the test that matters — run it with WiFi off.

**DO NOT:** pass raw sensor readings or model logits to the LLM and ask it to interpret them. That inverts the entire architecture and is the thing a judge will probe.

---

# PART 6 — TRAP NODE / MODEL B (Days 9–10)

> This is not an improvised contraption. Yellow sticky traps at 4–5 per acre are the official ICAR/NIPHM IPM recommendation for sugarcane whitefly, and pheromone traps at 5/ha for rice yellow stem borer. **The farmer already has the trap. You are automating the counting.** Frame it that way to judges.

---

## STEP 29 — ESP32-CAM firmware                             [AGENT] ~1.5 h

**Depends on:** nothing
**Goal:** Capture-and-send only. **All inference happens on the gateway.**

**Build `firmware/esp32cam_trap/esp32cam_trap.ino`:**
- Wake on timer (twice daily is enough — trap monitoring is not a latency-sensitive task)
- Capture JPEG at the highest resolution PSRAM allows
- POST to the gateway over WiFi (or ESP-NOW / LoRa if range demands)
- Deep sleep between captures
- Include a device ID and timestamp

**Why inference is not on the device:** the common ESP32-CAM uses an Xtensa LX6 with no vector instructions. Espressif's ESP-NN provides assembly/SIMD kernels only for ESP32-S3/P4/S31 — a plain ESP32 gets generic C. The impressive INT8 speedups people quote belong to the S3. If you specifically want on-device inference as a differentiator, buy an **ESP32-S3-CAM** (~₹800–1,200) rather than spending days optimising an LX6.

**Acceptance:** the gateway receives a JPEG with correct EXIF/timestamp on schedule, and the board draws sleep-level current between captures.

---

## STEP 30 — Train Model B                                  [AGENT] ~2 h

**Depends on:** 6
**Goal:** A tiny CNN classifying 64×64 insect crops.

**Build `train/train_model_b.py`:**
- Architecture: `TrapPestCNN` from `AI_Handbook_4.md` §7.5.2 — 4 conv blocks, ~150k params, **from scratch** (the task is genuinely simple: uniform background, fixed scale, controlled lighting)
- Classes: your target pests + **`not_pest`** (dust, debris, and **beneficials** — distinguishing pests from natural enemies is agronomically meaningful and a good detail to mention)
- Data: **RP11** (rice-specific IP102 refinement, re-annotated — prefer it over raw IP102), plus sticky-trap datasets, plus **synthetic composites**: paste segmented insects onto real yellow trap backgrounds at random positions and rotations. Cheap and very effective here.
- Augmentation: rotation at **any** angle (insects land arbitrarily), flips, brightness, blur, scale ±20%. No colour-destroying transforms.
- ~40 epochs, Adam, lr 1e-3, cosine. Minutes on a T4.

**Acceptance:** validation macro-F1 and a per-class recall table. `not_pest` recall must be high — false pest counts are worse than missed ones, because they trigger unnecessary spraying.

---

## STEP 31 — Gateway trap counting                          [AGENT] ~1.5 h

**Depends on:** 29, 30
**Goal:** JPEG in, per-species counts out.

**Build `gateway/trap_count.py`. Use `core/trap_segmentation.py` — already written and tested.**

```python
from core.trap_segmentation import segment_trap_blobs
blobs = segment_trap_blobs(bgr, min_area=8, max_area=1200, abs_floor_px=ABS_FLOOR)
```

**One human calibration task, and it matters:** `abs_floor_px` is **not a tuning knob**. The trap camera has fixed geometry, so mm-per-pixel is a known constant. Photograph a ruler at your exact camera-to-trap distance, compute mm/px, and set the floor from the smallest target pest's radius (whitefly ≈ 1.0–1.5 mm). Record the measurement in `configs/trap_config.py`.

> **Why this matters:** the module already fixes a bug where a single global `0.3 × dist.max()` threshold — set by one large moth — erased every whitefly and thrip on the board. Verified: 1 marker of 6 before the fix, 6 of 6 after. If you set `abs_floor_px` by guesswork you can reintroduce it.

Then: downscale to ~1280 px long side → segment → classify each 64×64 crop → counts per species per day → compare against published Economic Threshold Levels → trend slope for "infestation increasing".

**Acceptance:** on a photo of a real trap with insects stuck to it, counts are within ±20% of a manual count. Save a side-by-side of your count and the manual count — that image is excellent for the video.

---

## STEP 32 — Trap alerting                                  [AGENT] ~45 min

**Depends on:** 31, 27
**Goal:** Counts become ETL-referenced alerts.

Store daily counts in SQLite; alert when a count crosses the published ETL, or when the 3-day slope projects a crossing within 48 hours. Emit the same structured-finding JSON as STEP 27 so the advisory layer handles it identically.

**Human task:** collect published ETLs for your target pests from ICAR/NIPHM IPM packages and put them in `configs/etl_thresholds.py` **with the source cited next to each number**. A judge may ask where the threshold came from, and "ICAR-IIRR sets ~10 hoppers per hill at tillering" is a much better answer than a number you chose.

---

# PART 7 — INTEGRATION (Days 11–12)

---

## STEP 33 — Storage and dashboard                          [AGENT] ~2 h

**Depends on:** 21, 27

**Build `edge/storage.py`:** SQLite on device, surviving connectivity loss. Tables: `frames`, `cell_verdicts`, `sensor_readings`, `trap_counts`, `alerts`. Opportunistic sync when a network appears.

**Build `dashboard/app.py`:** Streamlit or Flask.
- Field map with per-cell colour by verdict — and **all four states must be visually distinct**: `DISEASE` / `HEALTHY` / `UNCERTAIN` / `NO_DATA`. Grey for NO_DATA means "re-fly", green for HEALTHY means "don't spray". Conflating them is a real bug that the tested `aggregate_cell` now prevents.
- Historical trends: CWSI, soil moisture, trap counts
- Alert feed with the triggering rule shown
- Advisory text in the selected language

**Acceptance:** the dashboard renders from a SQLite file populated by a dry run, with no live hardware attached.

---

## STEP 34 — Full end-to-end bench integration              [BOTH] ~3 h

**Depends on:** 21, 24, 25, 27, 28, 31, 33
**Goal:** Everything running together on the Nano, on the bench, before the drone is involved.

**Commands:**
```bash
python3 edge/pipeline.py --config configs/field_demo.yaml --duration 600 --report
```

**Acceptance — a 10-minute continuous run producing:**
- ≥ 2 scenes/second sustained
- No crash, no memory growth (watch `tegrastats` — RSS must be flat)
- Clean shutdown, no `LogicError`
- Every subsystem populating its SQLite table
- At least one alert of each type in the dry-run log

**DO NOT** move to the drone until this passes. Debugging a pipeline in the air is not debugging.

---

## STEP 35 — Run the full regression suite                  [AGENT] ~15 min

**Depends on:** 34
```bash
python -m pytest tests/ -v          # must remain 15 passed
```
Add integration tests for anything you built in Parts 5–7 that makes a decision.

---

## STEP 36 — Failure-mode rehearsal                         [BOTH] ~1 h

**Depends on:** 34
**Goal:** Know how the system fails before a judge finds out for you.

| Test | Expected behaviour |
|---|---|
| Point the camera at a hand | `NOT_CROP`, no diagnosis |
| Point at a weed / plastic pipe | `UNKNOWN` via the energy gate, no diagnosis |
| Cover the lens | frame gate rejects; cell goes `NO_DATA`, not `HEALTHY` |
| Disconnect WiFi | advisory falls back to templates; SQLite keeps writing |
| Unplug the thermal sensor | CWSI unavailable; disease path keeps working |
| Feed a pristine healthy leaf | `HEALTHY`, no alert |
| Feed one lesion in one tile | `DISEASE` with that tile highlighted |

**Acceptance:** all seven behave as described. **Film this** — a graceful-degradation montage is one of the strongest 30 seconds you can put in the submission video.

---

# PART 8 — DRONE & SUBMISSION (Days 13–17)

---

## STEP 37 — Drone integration                              [HUMAN] ~1 day

**Depends on:** 34

**Physical build:**
- Jetson Nano + carrier + heatsink/fan: ~250–300 g at 5–10 W. Budget it into the airframe. MAXN needs a proper barrel-jack supply, not micro-USB.
- **Mount the camera on a ~0.6 m boom.** CFD work at 1 m above canopy measured the downwash footprint as an ellipse of roughly 0.45 × 0.4 m semi-axes; the boom shoots outside your own rotor wash.
- Rangefinder for altitude hold in the 1.5–2.5 m inspection band

**Two flight modes:**
```
SURVEY    8-15 m AGL, fast   : RGB vegetation indices -> flag cells.  NO CWSI here.
INSPECT   1.5-2.5 m, slow    : full AI pipeline + CWSI on flagged cells only
```

**Acceptance:** a 3-minute flight producing gated frames with valid GPS/altitude tags and at least one correct diagnosis.

---

## STEP 38 — Demo field setup                               [HUMAN] ~3 h

**Depends on:** 37
**Goal:** Something to fly over, given that field access is unlikely.

In order of preference:
1. Any accessible green space — campus lawn, a relative's kitchen garden, a roadside plot. You need 10 minutes of flight, not a research station.
2. A dozen potted plants for real 3D geometry, shadows and wind
3. **Printed leaf images on stakes** — print 20–30 high-resolution diseased leaves from **held-out test data** at life size

> Option 3 is completely legitimate for a demonstration video **as long as you say so on the slide.** Judges penalise concealment, not honest scoping. "Field access was not available in our timeframe, so flight footage uses printed held-out test images at life size; all reported accuracy is measured on real field datasets" is a strong sentence.

---

## STEP 39 — Capture all footage                            [HUMAN] ~1 day

**Depends on:** 37, 38

Shoot these separately — it is a demonstration video, not a single continuous take:
1. Drone taking off and flying the survey pass
2. Drone at inspection altitude over the target
3. **Screen recording: live on-device inference with the tile heatmap overlay**
4. Trap node: the trap, the capture, the counted output beside a manual count
5. Dashboard: map, trends, alerts
6. Advisory in Hindi on a phone
7. The graceful-degradation montage from STEP 36
8. `pytest` running green, and `trtexec` printing your measured latency

Items 3, 4 and 8 are the ones that make it look real rather than staged. Prioritise them if time runs short.

---

## STEP 40 — Assemble the results pack                      [AGENT] ~1 h

**Depends on:** 14, 15, 18

Collect into `artifacts/reports/FINAL_RESULTS.md`, every number measured, none invented:

| Number | Source |
|---|---|
| In-distribution macro-F1 + top-1 | STEP 14 |
| **Cross-domain macro-F1** | STEP 14 |
| Per-class recall table | STEP 14 |
| Background-only bias audit accuracy vs 3.2% chance | STEP 10 |
| Stage 1 → Stage 2 ablation table | STEPS 12, 13 |
| OOD: FPR@95TPR, AUROC | STEP 15 |
| **Measured Nano latency, ms per 9-tile batch** | STEP 18 |
| End-to-end scenes/second | STEP 34 |
| Trap count vs manual count | STEP 31 |

**Acceptance:** every cell is filled with a measured value or explicitly marked "not measured". **No estimates. No placeholders.** Rule R7.

---

## STEP 41 — Video                                          [HUMAN] ~1 day

**Depends on:** 39, 40

Suggested 4–5 minute structure:
- 0:00 problem + the scope decision (2–3 crops done well, not 5 badly)
- 0:30 architecture: two-tier drone + static trap + edge inference
- 1:15 the dataset-bias finding and why your numbers are honest ← **your differentiator**
- 2:00 live inference footage
- 2:45 trap node
- 3:15 irrigation, nutrient, risk, advisory in Hindi
- 4:00 graceful degradation montage
- 4:30 measured performance on 2019 hardware + limitations stated plainly

---

## STEP 42 — PPT                                            [HUMAN] ~1 day

**Depends on:** 40, 41

Slides that carry the most weight:
1. **The bias audit** — background-only accuracy vs chance. Almost no student team will have done this.
2. **Stage 1 → Stage 2 ablation** — proof you measured each decision rather than guessing.
3. **Two accuracy numbers** with the honest explanation of why the second is lower.
4. **Measured latency on a 2019 board with no INT8** — constraint-driven engineering.
5. **Confusion matrix + Grad-CAM** — the model attends to lesions, not backgrounds.
6. **Limitations, stated by you first.** CWSI needs local baseline calibration; aerial nutrient is relative not absolute; wheat is roadmap; field validation pending.

---

## STEP 43 — Submit                                         [HUMAN] ~2 h

**Depends on:** 41, 42
Upload video (unlisted YouTube), attach the PPT, submit via SPOC. **Target 19 September, one day before the deadline.**

**Acceptance:** submission confirmation received, with a day in hand.

---

# PART 9 — REFERENCE

## 9.1 Constraint card — pin this where the agent can see it

```
Jetson Nano 4GB (Maxwell GM20B, compute capability 5.3)
  JetPack   4.6.4 MAX — no JetPack 5 or 6, ever
  Python    3.6      — no f-string '=', no walrus, no multiprocessing.shared_memory
  CUDA      10.2     — no PyTorch 2.x, ever
  TensorRT  8.2      — ONNX opset <= 13, static shapes
  INT8      NOT SUPPORTED IN HARDWARE — FP16 is the floor
  Zero-copy REJECTED — pinned memory is uncached below CC 7.2; ours is 5.3
  Memory    4GB shared CPU/GPU — boot headless, no PyTorch at runtime
  Batch     9 (= 3x3 tile grid). Never 8.
  Threading OK — numpy/cv2 release the GIL. CUDA context INSIDE the worker thread.
```

## 9.2 Definition of done

A step is done when its Acceptance commands run, produce the stated output, and that output is **pasted into the build log**. "It should work" is not done. "I ran it and here is the output" is done.

## 9.3 Escalate to the human when

- A download requires clicking a consent form
- Hardware is unavailable or misbehaving
- An acceptance criterion fails twice after following "If it fails"
- A decision would contradict a rule in §0.2
- A measured number is much worse than expected — **report it, do not tune until it looks better**

## 9.4 If you fall behind — cut in this order

1. **Wheat** (STEP 8) — rice + sugarcane is 26 classes, past the 15–20 target
2. **Stage 3 distillation** (STEP 13) — genuinely optional
3. **Segmentation / severity** — never started; keep it that way
4. **Trap node** (Part 6) — the disease classifier already covers pest *damage* via Hispa, leaf roller and stem borer classes
5. **Drone flight** (STEP 37) — present the rover plus the two-tier architecture as designed

**Never cut:** STEP 10 (bias audit), STEP 14 (cross-domain eval), STEP 18 (engine build), STEP 36 (failure rehearsal). Those four are what separate this from a project that claims 99% and falls apart under questioning.

## 9.5 The thing to remember

You have four versions of two design documents, three rounds of red-team findings, five tested modules — and, as of today, **zero trained models.** Everything in Parts 1–3 exists to change that. If you get to STEP 15 with measured numbers and nothing else works, you still have a defensible submission. If you polish Parts 5–8 and never train Model A, you have nothing.

**Start at STEP 1. Do not skip ahead.**
