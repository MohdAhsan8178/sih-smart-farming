# The AI Handbook — Version 4
### Smart Farming Assistant — models, data, training, deployment

**Version:** 4.0 · 3 September 2026
**Supersedes:** `AI_Handbook_3.md`, `_2.md`, `.md` — all kept for reference
**Companions:** `SIH_Smart_Farming_AI_Report_4.md` · `RED_TEAM_ADJUDICATION_3.md` · **`sih_pipeline_v4/` — runnable, tested code**

> ### ⚠ Read this first: the code now lives in files, not in this document
>
> Three rounds of red-team review found bugs in the *fixes* for the previous round's bugs. Round 3 found a bug that Round 2's fix introduced, and a Round 2 bug **reintroduced two sections later in this same document** because I patched one section and left a stale copy in another.
>
> Prose in two places drifts. An imported function cannot.
>
> **All decision logic is now in `sih_pipeline_v4/`, with a passing test suite.** The code blocks below are documentation of what those modules do. **Import the modules; do not copy code out of this document.** That single change is what stops the drift.
>
> ```
> $ cd sih_pipeline_v4 && python -m pytest tests/ -v
> 15 passed
> ```

---

## Changelog: v3 → v4

Round 3 upheld 7 of 8 findings. Three were bugs in code v3 introduced *as fixes* for Round 2. A ninth bug was found by **executing** the code — it appears in no audit.

| § | Change | Severity | Verified |
|---|---|---|---|
| **8.5** | **Confidence gate moved to UNADJUSTED probabilities.** Evidence (energy + confidence) separated from label choice (prior-adjusted argmax). | High | 0/50 OOD inputs become confident rare-class calls; 50/50 real tiles still pass |
| **8.6** | **`aggregate_cell` returns explicit states** — `DISEASE`/`HEALTHY`/`UNCERTAIN`/`NO_DATA`. Never a bare `None`. | Medium | v3 returned `(None,0.0,0)` for healthy *and* never-visited cells |
| **9.2** | **Tile grid fixed at 3×3; engine batch = 9.** Plus `infer_all` chunk-looping. | High | v3 exported batch 8 while tiling emitted up to 9 — the 9th tile was silently dropped |
| **9.5** | **Stale all-logit `energy_np` removed.** Runtime now imports `rejection.open_set_energy`. | High | v3's helper gave E = −14.0 on a confident `not_crop` vector (looks in-distribution) and raised `AxisError` on 1D input |
| **15** | Test suite expanded to 15 tests and **shipped as runnable code**. | | All 15 pass; 5 fail against v3 implementations |

**Also changed in the companion report:** the thermal canopy gate (`Ta+7` → `Ta+15`, which was suppressing every drought alarm), the vegetation mask (Otsu → absolute threshold, which was bisecting pure canopies), the trap marker extraction (global → dual threshold, which was erasing every whitefly), and the nutrient scope (drone = relative only; the grey-card-in-every-frame requirement was physically impossible).

### Earlier changelogs (retained)

**v2 → v3:** BGR/RGB channel swap in `FusedModel`; `aggregate_frame` had no HEALTHY path; energy calibrated on trained `not_crop`; inverted thermal spread gate; PyCUDA teardown.
**v1 → v2:** offline teacher-logit caching → online consistent teaching; energy OOD added; spatial/temporal aggregation separated; fused ONNX preprocessing; CUDA context in worker thread; zero-copy rejected.

---

## How to use this document

| If you want to know... | Go to |
|---|---|
| What changed since v3? | Changelog above |
| Which models am I building in total? | §1 |
| Pretrained or from scratch? | §2 |
| Exactly which model, and where do I download it? | §3 |
| What does the architecture look like in code? | §4 |
| Which datasets, with links? | §5 |
| What do I do to the data before training? | §6 |
| Augmentation recipe | §6.6 |
| Class imbalance handling | §6.7 |
| Training hyperparameters and schedule | §7 |
| Knowledge distillation recipe **(rewritten in v2, note added v3)** | §7.4 |
| How do I know if it's actually good? | §8 |
| Calibration, abstain, and OOD rejection **(corrected in v3)** | §8.5 |
| How do tile predictions become one diagnosis? **(rewritten in v3)** | §8.6 |
| PyTorch → ONNX → TensorRT export **(channel bug fixed in v3)** | §9 |
| Inference code for the Nano **(teardown fixed in v3)** | §9.5 |
| Things will break — what do I do? | §10 |
| What changes if we buy the Orin Nano? | §11 |
| Day-by-day build order | §12 |
| Quick answers to likely questions | §13 |
| **Regression tests to stop bugs recurring** | **§15 + `sih_pipeline_v4/tests/`** |

---

## 1. Model inventory — the complete list

You are training **two models**, with a third as an optional stretch. That is the entire AI surface of this project. Everything else in the problem statement is deterministic Python (see the companion report §1).

| ID | Model | Task | Runs on | Pretrained? | Priority |
|---|---|---|---|---|---|
| **A** | Crop disease + pest-damage classifier | Classify a leaf/canopy tile into a crop-disease class | Jetson Nano (drone) | **Yes — fine-tune** | **Critical** |
| **B** | Trap insect classifier | Classify a cropped blob from a sticky trap into pest species / not-pest | **Gateway (Nano/laptop)** — see §7.5 | **From scratch (tiny CNN)** | High |
| **C** | Disease severity segmentation | Pixel mask of diseased area → % leaf affected | Jetson Nano | Yes — fine-tune | Optional |

Note that A and B answer opposite questions on the pretrained-vs-scratch axis, and the reason why is instructive — see §2.

**Model A is the project.** If Model A works and nothing else does, you have a demo. If everything else works and Model A doesn't, you have nothing. Budget your time accordingly: Model A should be working end-to-end (trained, exported, running on the Nano) before you write a single line of Model B.

---

## 2. Pretrained or from scratch? — the decision, with reasoning

**Model A: fine-tune a pretrained backbone. This is not a close call.**

1. **Data volume.** You will have roughly 20,000–25,000 training images. Training a modern CNN from random initialization needs something like ImageNet scale (1.28M images) to learn generic visual primitives — edges, textures, colour opponency, blob detectors. With 20k images you would spend most of your capacity relearning what a texture is, and you would overfit badly.

2. **Time.** You have two weeks. From-scratch ImageNet-scale pretraining is days of GPU time you do not have on Colab.

3. **Robustness is precisely what pretraining buys you.** Your entire failure mode last time was poor generalization. ImageNet-pretrained features are broad and transfer; features learned from 20k in-domain images are narrow and brittle.

4. **There is no novelty penalty.** Transfer learning is what Plantix, every published field-disease paper, and every production system does. Your novelty is edge deployment, the two-tier drone architecture, and multi-modal fusion — not a hand-rolled backbone.

**When would from-scratch be right for Model A?** Essentially never at your scale. If a judge asks "did you build your own model?", the correct answer is: "we fine-tuned an ImageNet-pretrained backbone with a custom multi-crop head, a field-conditioned augmentation pipeline we designed for domain shift, consistent-teaching distillation from a self-supervised foundation model, and an energy-based rejection layer — training from scratch on 20k images would have produced a strictly worse model, and here is the ablation showing that."

**Model B: build a small CNN from scratch — and here's why the answer flips.**
The trap-node task is genuinely easier: 64×64 crops, uniform yellow background, controlled lighting, fixed scale, 6–10 classes. A 4-block CNN with ~150k parameters trains in minutes. Pretrained ImageNet models are overkill.

**Model C: fine-tune pretrained.** Same reasoning as A, and segmentation is even more data-hungry.

---

## 3. Exact model selection for Model A

### 3.1 The risk ladder

Your binding constraint is not accuracy — it is **whether the model survives ONNX → TensorRT 8.2 conversion on a Maxwell GPU**. Ordered by conversion risk, lowest first. **Start at rung 1. Only climb if you have time left.**

| Rung | timm name | Params | GMACs | Export risk | Notes |
|---|---|---|---|---|---|
| **1** | `tf_efficientnet_lite0.in1k` | 4.7M | 0.4 | **Lowest** | Purpose-built for edge: SE blocks removed, swish replaced with ReLU6. Converts cleanly. **Start here.** |
| **2** | `mobilenetv3_large_100.ra_in1k` | 5.5M | 0.22 | Low-medium | Has squeeze-excite + hard-swish. Usually fine on TRT 8.2 but test early. |
| **3** | `edgenext_x_small.in1k` | 2.3M | 0.5 | Medium | Best accuracy-per-latency *with published Jetson Nano FP16 TensorRT benchmarks*. |
| **4** | `edgenext_small.usi_in1k` | 5.6M | 1.3 | Medium | Distilled (USI), 79.4% ImageNet top-1. Highest accuracy of the realistic options. |

**v2 note on rungs 3–4:** the red team independently flagged EdgeNeXt's 4D LayerNorm and split-transposed-attention (SDTA) blocks as likely to decompose into unfused elementwise ops on TensorRT 8.2 without tensor cores. That is a second opinion agreeing with this ladder. Treat rung 1 as the default, not as excessive caution.

**Plan:** train rung 1 and rung 3 in parallel on Colab (they're cheap). Export rung 1 first and get it running on the Nano. If rung 3 also converts and is faster/better, swap it in. Never let rung 3 block your demo.

```python
import timm
model = timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=0)
```

Model cards:
- https://huggingface.co/timm/edgenext_xx_small.in1k — 1.3M params, 0.3 GMACs
- https://huggingface.co/timm/edgenext_x_small.in1k — 2.3M params, 0.5 GMACs
- https://huggingface.co/timm/edgenext_small.usi_in1k — 5.6M params, 1.3 GMACs
- https://huggingface.co/timm/edgenext_base.usi_in1k — 18.5M params, 3.8 GMACs (too big for Maxwell; fine on Orin)

Confirm strings exist in your timm version before relying on them:
```python
print(timm.list_models('*efficientnet_lite*', pretrained=True))
print(timm.list_models('edgenext*', pretrained=True))
```

### 3.2 Models to avoid on the Maxwell Nano

| Model | Why not |
|---|---|
| MobileViT (any size) | EfficientFormer's authors measured it as *significantly slower than MobileNetV2 and EfficientNet-B0* while also 2.3% less accurate than EfficientNet-B0. |
| ViT / DeiT / Swin | No tensor cores, no INT8, poor TRT 8.2 attention kernels on Maxwell. |
| ConvNeXt (any size) | LayerNorm-heavy; TRT quirks. Fine on Orin, not here. |
| ResNet50+ | ~42 ms/image FP16 on Nano before tiling. Too slow for batched tiles. |
| Anything needing PyTorch 2.x export | Cannot run on CUDA 10.2 at all. |

### 3.3 The teacher model (training only — never deployed)

| Purpose | timm name | Params | Notes |
|---|---|---|---|
| Teacher (recommended) | `vit_small_patch14_dinov2.lvd142m` | 22.1M | DINOv2 self-supervised on LVD-142M. Native 518² but run at 224. |
| Teacher (alternative) | `vit_small_patch14_reg4_dinov2.lvd142m` | 22.1M | With registers; often cleaner attention. |
| Teacher (newest) | `vit_small_patch16_dinov3.lvd1689m` | ~21M | DINOv3, distilled from a 7B model. Patch-16 makes resolution maths easier. |
| Teacher (simplest) | `convnext_tiny.in12k_ft_in1k` | 28.6M | If DINOv2 gives trouble, a strong supervised CNN teacher works nearly as well and is much cheaper to run online. |

DINOv2 gave the highest linear-probing performance among tested backbones on agricultural disease and species datasets, which is exactly the property you want in a teacher.

**Input-size arithmetic:** patch-14 models need dimensions divisible by 14 — **224 works** (16 × 14). Never feed 256 to a patch-14 model.

**v2 note:** since distillation is now online (§7.4), teacher *inference cost* matters where it didn't before. `convnext_tiny` is roughly 3× cheaper per forward pass than DINOv2 ViT-S at 224 and is a legitimate choice if your Colab session is slow.

---

## 4. Model A architecture — exact specification

### 4.1 Head design: start flat, upgrade later

| | Flat unified head | Crop-gated multi-head |
|---|---|---|
| Output | One softmax over ~31 classes, names encode crop (`rice__blast`) | Crop head + per-crop disease heads |
| ONNX export | Trivial — single tensor out | Needs a wrapper to concatenate dict outputs |
| Training code | Standard cross-entropy | Masked multi-task loss, more bug surface |
| Time to working | ~1 hour | ~4 hours + debugging |

Rice, sugarcane and wheat leaves look nothing alike, so the flat model learns the crop distinction almost for free. **Build flat for v1. Upgrade only if you finish early.**

**v2 note:** the red team argued a flat softmax causes "inter-crop suppression" — rice and sugarcane logits competing and depressing max confidence below the abstain threshold. This is theoretically possible but unlikely given how visually distinct your three crops are. **Check your confusion matrix for it rather than pre-emptively restructuring.** If you see systematic cross-crop confusion, that is your signal to move to the multi-head design in §4.5.

### 4.2 Class taxonomy

Define once, in one file, and never change mid-project.

```
# rice (Paddy Doctor, 13 classes)
rice__normal
rice__bacterial_leaf_blight
rice__bacterial_leaf_streak
rice__bacterial_panicle_blight
rice__blast
rice__brown_spot
rice__downy_mildew
rice__tungro
rice__hispa                 # ← pest damage
rice__leaf_roller           # ← pest damage
rice__black_stem_borer      # ← pest damage
rice__white_stem_borer      # ← pest damage
rice__yellow_stem_borer     # ← pest damage

# sugarcane (union of Thite + Daphal/Koli, deduplicated)
sugarcane__healthy
sugarcane__dried_leaf
sugarcane__mosaic
sugarcane__red_rot
sugarcane__rust             # merge "brown rust" + "rust"
sugarcane__yellow_leaf      # merge "yellow" + "yellow leaf disease"
sugarcane__smut
sugarcane__pokkah_boeng
sugarcane__grassy_shoot
sugarcane__brown_spot
sugarcane__banded_chlorosis
sugarcane__sett_rot

# wheat (optional third crop)
wheat__healthy
wheat__yellow_rust
wheat__brown_rust
wheat__septoria
wheat__powdery_mildew

# reject class — see §4.4
not_crop
```

**31 classes**, of which **five are pest damage** — that is how your drone satisfies the pest requirement without seeing an insect.

**Class merging:** Thite et al. has 11 categories; Daphal & Koli has 5. Merge "brown rust"/"rust", merge "yellow"/"yellow leaf disease", take red rot from the second set. Document every merge in a CSV — a judge may ask.

### 4.3 Flat model (build this first)

```python
# model.py
import timm
import torch
import torch.nn as nn

NUM_CLASSES = 31

def build_model(backbone_name='tf_efficientnet_lite0',
                num_classes=NUM_CLASSES,
                pretrained=True,
                drop_rate=0.2):
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,          # dropout before classifier
        drop_path_rate=0.1,           # stochastic depth — helps generalization
    )

if __name__ == '__main__':
    m = build_model()
    x = torch.randn(2, 3, 224, 224)
    print(m(x).shape)                 # torch.Size([2, 31])
    print(sum(p.numel() for p in m.parameters())/1e6, 'M params')
```

That's the whole model. Resist bolting on attention modules, LSTM heads, or ensembles — every addition is another ONNX export failure mode, and none will help as much as fixing your data.

### 4.4 The `not_crop` class — keep it, but understand what it is ⚠ **CORRECTED IN v3**

Train on soil, sky, hands, shoes, pavement, buildings, blurred frames, and random ImageNet images. Roughly 1,500–2,000 images.

**Why it matters:**
- **Demo safety.** Without it, pointing the camera at a judge's hand gives a confident "rice blast."
- **Drone reality.** Many aerial frames contain soil, sky or field margins.
- **It is your outlier-exposure set** — auxiliary negatives that teach the model where the crop manifold ends.

> **⚠ v3 correction.** Version 2 of this handbook stated: *"Your `not_crop` set is the calibration data for τ_energy. Delete it and you have a score with no defensible threshold."* **That was wrong.**
>
> Once you train a class on soil and sky, those images are **in-distribution for that model**. A soil image produces a large logit on the `not_crop` index, so `logsumexp` over all logits is large, so the energy is strongly negative — indistinguishable from a confidently-classified rice leaf. Calibrating a threshold on that data measures nothing.
>
> The OOD literature is explicit that Outlier Exposure "uses auxiliary datasets **completely disjoint from the test time data**," and that drawing calibration outliers from the training auxiliary set violates the separation-of-information principle and inflates metrics.
>
> `not_crop` remains valuable — as outlier exposure and as a cheap first rejection layer. It is simply **not** the data you calibrate the energy threshold on. See §8.5 for the three-way split.

### 4.5 Multi-head model (v2 of the model, optional)

```python
# model_multihead.py
import timm, torch, torch.nn as nn

CROPS = ['rice', 'sugarcane', 'wheat', 'not_crop']
DISEASE_COUNTS = {'rice': 13, 'sugarcane': 12, 'wheat': 5}

class MultiHeadCropModel(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_lite0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0)
        f = self.backbone.num_features
        self.crop_head = nn.Linear(f, len(CROPS))
        self.disease_heads = nn.ModuleDict(
            {c: nn.Linear(f, n) for c, n in DISEASE_COUNTS.items()})

    def forward(self, x):
        feat = self.backbone(x)
        # ONE concatenated tensor keeps ONNX export simple.
        # Layout: [crop_logits | rice | sugarcane | wheat]
        parts = [self.crop_head(feat)]
        for c in DISEASE_COUNTS:
            parts.append(self.disease_heads[c](feat))
        return torch.cat(parts, dim=1)
```

Slice at known offsets on the host. Dict outputs are a common ONNX export failure — always return one tensor.

**Loss:** crop cross-entropy always applies; disease loss for crop *c* applies only to samples whose true crop is *c* (mask the rest). Weight roughly `1.0 × crop + 1.0 × disease`.

---

## 5. Datasets — exact sources and links

*(Unchanged from v1. Reproduced in full so this document stands alone.)*

### 5.1 Core training data

**① Paddy Doctor — your anchor dataset**

| | |
|---|---|
| **What** | 16,225 expert-annotated rice leaf images, 13 classes (12 diseases/pests + normal) |
| **Where captured** | Real paddy fields near Tirunelveli, **Tamil Nadu, India**; Feb–Apr 2021; crop age 40–80 days |
| **Camera** | CAT S62 Pro smartphone, 1,080 × 1,440 |
| **Extras** | Per-image metadata: paddy variety and crop age |
| **Effort behind it** | ~500 man-hours, annotated with an agricultural officer |
| **Easiest download** | **https://www.kaggle.com/competitions/paddy-disease-classification** (10,407 labelled train images, 10 classes + metadata CSV) |
| **Full 13-class version** | https://ieee-dataport.org/documents/paddy-doctor-visual-image-dataset-automated-paddy-disease-classification-and-benchmarking |
| **Project site / code** | https://paddydoc.github.io · https://github.com/paddydoc/paddy-doctor-dataset |
| **Mirror (480×640)** | https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor |
| **Paper** | arXiv:2205.11108 |
| **Published baseline** | ResNet34, F1 = 97.50% (in-distribution only) |

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ and ACCEPT THE COMPETITION RULES on the website first
kaggle competitions download -c paddy-disease-classification
unzip paddy-disease-classification.zip -d data/raw/paddy_doctor/
```

**② Sugarcane — Thite et al.** — 6,748 images, 11 categories (yellow leaf, smut, pokkah boeng, mosaic, grassy shoot, brown spot, brown rust, banded chlorosis, sett rot, healthy, dried). Maharashtra, India (≈18.785 N, 74.022 E).
→ https://data.mendeley.com/datasets/355y629ynj — DOI `10.17632/355y629ynj.1`
→ Paper: https://www.sciencedirect.com/science/article/pii/S2352340924002373

**③ Sugarcane — Daphal & Koli** — ~2,521 images, 5 classes (healthy, mosaic, red rot, rust, yellow). Deliberately captured with **smartphones of varying configuration to maintain diversity** — one of very few plant datasets that varies capture device on purpose, which directly counteracts capture bias. Maharashtra, India.
→ https://data.mendeley.com/datasets/9424skmnrk/1

**④ Wheat (optional third crop)**
- CGIAR Wheat Rust (Ethiopia/Tanzania field sites): https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/data
- WFD2020 — 2,414 expert-labelled images, 5 fungal diseases incl. multi-disease: https://www.mdpi.com/2223-7747/10/8/1500
- Wheat Leaf Dataset — small (208/102/97): https://data.mendeley.com/datasets/wgd66f8n6h/1
- CerealConv — field + glasshouse; model beat the best human pathologist by 2 points: https://doi.org/10.1111/ppa.13684

**⑤ Multi-Crop Disease Dataset (India)** — banana, chilli, radish, groundnut, cauliflower; Tamil Nadu: https://data.mendeley.com/datasets/6243z8r6t6

### 5.2 Robustness / cross-domain evaluation data

**⑥ PlantDoc** — 2,598 in-the-wild images, 13 crops, 17 diseases → https://github.com/pratikkayal/PlantDoc-Dataset

**⑦ PlantWild** — 18,542 in-the-wild images, 89 classes (56 diseased + 33 healthy). Every image cross-validated by ≥2 experts with a third adjudicating disagreements. → https://github.com/tqwei05/MVPDR (UQRDM, password `plantwildv1`) · arXiv:2408.03120

**⑧ PlantSeg** (for Model C) — 11,400 images with disease segmentation masks across 115 diseases, plus ~8,000 healthy images. In-the-wild. 70/10/20 split provided. CC BY-NC 4.0.
→ https://github.com/tqwei05/PlantSeg · Zenodo https://doi.org/10.5281/zenodo.13762907 · arXiv:2409.04038

### 5.3 Pest datasets (Model B)

**⑨ IP102** — 75,222 images, 102 classes, ~19,000 with boxes → https://github.com/xpwu95/IP102 · https://mmcheng.net/ip102/
Caveats: web-crawled from Google/Flickr/Bing, "consistently suffers from poor resolution", long-tailed, documented inter/intra-class variance problems. Pretraining only, not a benchmark.

**⑩ RP11** — rigorous rice-specific refinement of IP102: adults/larvae separated, extra images crawled, **all adult samples re-annotated**, Latin family names. 11 adult categories / 4,559 images; 7 larval / 2,467. **Prefer this over raw IP102 for rice.** → PMC12194132

**⑪ Sticky-trap-specific data** — closest to your trap node's real conditions:
- Grapevine yellow sticky trap: 600+ entomologist-annotated images, ~1,500 IDs per class (PMC11669504)
- Multi-device/multi-colour trap dataset: DSLR, webcam, smartphone on blue/yellow/white/transparent (PMC11327826)

### 5.4 Auxiliary — nutrient validation

**⑫ Nitrogen Deficiency of Rice Crop** — four LCC-matched categories → https://data.mendeley.com/datasets/gzm5pxntyv/1
Use to *validate* the CIELAB b\* script (companion report §1.1).

### 5.5 PlantVillage — handling rules

→ https://www.kaggle.com/datasets/emmarex/plantdisease (54,305 images, 38 classes)

1. **Never** in validation or test.
2. **Never** quote a PlantVillage accuracy as a headline number.
3. May be used as auxiliary pre-training, but measure whether it helps — it often doesn't.
4. If used at all, say so on your slides.

Noyan (2022, arXiv:2206.04374): a model trained on **8 background pixels only** hit **49.0% accuracy across 38 classes** where chance is 2.6%. Cross-domain work reports lab→field transfer at **33.27%**, and PlantVillage-trained models at **45.95%** and **33.97%** on two independent field sources.

### 5.6 Scope recommendation

**Do rice and sugarcane properly. Wheat is optional.** Rice (13) + sugarcane (12) + `not_crop` is already 26 classes, past your 15–20 target. If wheat costs more than half a day, drop it and put it on the roadmap slide.

| Source | Images |
|---|---|
| Paddy Doctor | ~16,200 |
| Sugarcane (Thite) | ~6,750 |
| Sugarcane (Daphal/Koli) | ~2,520 |
| Wheat (if included) | ~2,500–8,000 |
| `not_crop` (you assemble) | ~1,800 |
| **Total** | **~27,000–35,000** |

---

## 6. Data preparation — everything before `.fit()`

Most teams spend 80% of their time on architecture and 20% on data. **Invert that.**

### 6.1 Directory layout

```
project/
├── data/
│   ├── raw/                       # untouched downloads
│   ├── interim/                   # after dedup + class mapping
│   └── processed/
├── splits/
│   ├── train.csv                  # path, label, source_dataset, group_id
│   ├── val.csv
│   ├── test_indist.csv
│   └── test_crossdomain.csv       # PlantDoc / PlantWild — never trained on
├── configs/
│   └── classes.py                 # the ONE source of truth
├── src/
└── artifacts/
    ├── checkpoints/
    ├── onnx/
    └── engines/
```

Use CSV manifests, not folder structure, as the source of truth.

### 6.2 Step 1 — deduplication (before splitting)

```python
# dedup.py
import imagehash
from PIL import Image
from pathlib import Path
import pandas as pd

def phash_all(root):
    rows = []
    for p in Path(root).rglob('*'):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'), hash_size=8)
        except Exception as e:
            print('skip', p, e); continue
        rows.append({'path': str(p), 'label': p.parent.name, 'phash': str(h)})
    return pd.DataFrame(rows)

def group_near_duplicates(df, max_distance=5):
    """Assign a group_id so near-duplicates never straddle a split."""
    hashes = [imagehash.hex_to_hash(h) for h in df.phash]
    group = [-1] * len(df); gid = 0
    for i in range(len(df)):
        if group[i] != -1: continue
        group[i] = gid
        for j in range(i + 1, len(df)):
            if group[j] == -1 and (hashes[i] - hashes[j]) <= max_distance:
                group[j] = gid
        gid += 1
    df = df.copy(); df['group_id'] = group
    return df
```

O(n²) is ~20–30 min on 30k images. Acceptable once. Bucket by the first 16 hash bits if too slow. Also run exact MD5 comparison — the two sugarcane datasets are both from Maharashtra and may share images.

### 6.3 Step 2 — grouped, stratified splitting

**Split by `group_id`, not by image.** Five photos of one leaf all go to the same split, or you are testing on training data with extra steps.

```python
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

df = pd.read_csv('splits/all_images.csv')      # path,label,source,group_id

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, hold_idx = next(sgkf.split(df, y=df.label, groups=df.group_id))
train_df, hold_df = df.iloc[train_idx], df.iloc[hold_idx]

sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
v_idx, t_idx = next(sgkf2.split(hold_df, y=hold_df.label, groups=hold_df.group_id))

hold_df.iloc[v_idx].to_csv('splits/val.csv', index=False)
hold_df.iloc[t_idx].to_csv('splits/test_indist.csv', index=False)
train_df.to_csv('splits/train.csv', index=False)
```

Then build `test_crossdomain.csv` from PlantDoc and PlantWild — **never trained on, opened exactly once, at the end.**

### 6.4 Step 3 — the bias audit (your most impressive slide)

```python
# bias_audit.py — train a classifier on BACKGROUND ONLY.
import numpy as np, cv2

def eight_pixel_features(path):
    """Four corners + four edge midpoints. No leaf content whatsoever."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    pts = [(0,0), (0,w-1), (h-1,0), (h-1,w-1),
           (0,w//2), (h-1,w//2), (h//2,0), (h//2,w-1)]
    return np.concatenate([img[y, x] for y, x in pts])  # 24-dim
# then: sklearn RandomForestClassifier on these vectors. Chance = 1/31 = 3.2%.
```

- Near chance (3–6%) → dataset is clean. Put it on a slide.
- 15–30% → mild leakage. Strengthen background augmentation.
- >40% → serious leakage, like PlantVillage's 49%. Investigate which classes separate.

### 6.5 Step 4 — preprocessing decisions

| Decision | Recommendation | Reason |
|---|---|---|
| Input resolution | **224 × 224** | Sweet spot. Test 192/160 for speed later. |
| Resize | Shorter side → 256, random-crop 224 (train) / center-crop (val) | Preserves aspect ratio |
| Normalization | ImageNet mean/std from timm | Your backbone is ImageNet-pretrained |
| Colour space | RGB | OpenCV loads BGR — this bug is extremely common |
| Pre-resize on disk | **Yes** — max side 512, re-saved | Paddy Doctor is 1,080×1,440; full-size JPEG decode bottlenecks the dataloader. Can double training speed. |

```python
import timm
m = timm.create_model('tf_efficientnet_lite0', pretrained=True)
cfg = timm.data.resolve_model_data_config(m)
print(cfg)   # never hardcode mean/std
```

### 6.6 Step 5 — the augmentation recipe

**Highest-leverage change from your previous attempt.** The published TDR-Model work used exactly this strategy — MobileNetV3 + field-conditioned Albumentations — to reach 82.94% on a hard real-world field test set.

Principle: **augment toward the domain you deploy into** — a moving drone, variable Indian daylight, a cheap CSI camera.

```python
# transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def train_transform(size=224):
    return A.Compose([
        # --- geometry ---
        A.RandomResizedCrop(height=size, width=size,
                            scale=(0.35, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),            # nadir drone view has no natural "up"
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=45, border_mode=0, p=0.7),

        # --- illumination ---
        A.RandomBrightnessContrast(brightness_limit=0.35,
                                   contrast_limit=0.35, p=0.8),
        A.OneOf([
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=120, p=1.0),
            A.RandomToneCurve(scale=0.3, p=1.0),
        ], p=0.5),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25,
                             val_shift_limit=15, p=0.5),

        # --- sensor & motion ---
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.GaussNoise(var_limit=(10, 60), p=1.0),
        ], p=0.4),
        A.ImageCompression(quality_lower=45, quality_upper=95, p=0.4),

        # --- anti-shortcut ---
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.3),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def eval_transform(size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=int(size * 256 / 224)),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

**Why each group is there:**
- **Aggressive `RandomResizedCrop` (down to 0.35)** — the most important line. Removes background shortcuts and matches your tile-based inference.
- **Shadow / sun flare / tone curve** — Indian midday field light is nothing like a shaded studio.
- **Motion blur** — you are on a drone.
- **ISO noise + JPEG compression** — your deployment camera is an IMX219, not a CAT S62 Pro.
- **CoarseDropout** — forces use of multiple leaf regions.

**Deliberately excluded:** heavy elastic transforms and channel shuffle. Colour is *diagnostic information* for plant disease — destroying it destroys your signal. That's why `hue_shift_limit` is only 12.

**⚠ v2 warning — augmentation and distillation interact.** If you use §7.4's online distillation, the teacher must receive the **identical augmented tensor** as the student. Do not build a separate mild transform for the teacher. This is the single biggest change from v1.

**Slicing-aided fine-tuning.** Since inference runs on tiles, add tiles of your training images to the training set: for each image, emit 2–4 random 320×320 crops as extra samples with the same label. SAHI's authors report inference-only slicing gives +5–7 AP but slicing-aided fine-tuning raises the cumulative gain to +12.7–14.5 AP.

### 6.7 Step 6 — class imbalance

Your combined sugarcane data spans ~1,722 images (brown spot) down to 43 (red leaf spot) — 40× imbalance.

**Option A — weighted sampler + class-weighted loss (v1 approach, still fine):**
```python
import numpy as np, torch
from torch.utils.data import WeightedRandomSampler

counts = train_df.label.value_counts().to_dict()
weights = train_df.label.map(lambda c: 1.0 / np.sqrt(counts[c])).values
sampler = WeightedRandomSampler(torch.DoubleTensor(weights),
                                num_samples=len(weights), replacement=True)

cls_w = torch.tensor([1.0/np.sqrt(counts[c]) for c in CLASS_NAMES])
cls_w = (cls_w / cls_w.mean()).float().cuda()
criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
```

**Option B — logit-adjusted loss (new in v2, cleaner theory):**
Menon et al. (ICLR 2021). Adds the log class prior to the logits during training so the model learns to compensate, with no resampling instability.

```python
import torch, numpy as np

class LogitAdjustedLoss(torch.nn.Module):
    def __init__(self, class_counts, class_names, tau=1.0, label_smoothing=0.1):
        super().__init__()
        priors = np.array([class_counts[c] for c in class_names], dtype=np.float64)
        priors = priors / priors.sum()
        self.register_buffer('log_priors',
                             torch.tensor(np.log(priors + 1e-12)).float())
        self.tau = tau
        self.ls = label_smoothing

    def forward(self, logits, targets):
        adjusted = logits + self.tau * self.log_priors.to(logits.device)
        return torch.nn.functional.cross_entropy(
            adjusted, targets, label_smoothing=self.ls)
```

**Option C — post-hoc logit adjustment (recommended in v3 if you are distilling):**
Apply the adjustment at *inference only*, subtracting `tau * log_prior` from the logits of an already-trained model. Menon et al. propose exactly this as a standard, Fisher-consistent post-processing step based on training label frequencies.

```python
def posthoc_logit_adjust(logits, log_priors, tau=1.0):
    """Balanced scores from a model trained with plain CE. Inference-time only."""
    return logits - tau * log_priors      # note the MINUS: removes the prior
```

Sweep `tau` over `[0, 0.25, 0.5, 0.75, 1.0]` on validation and pick the best macro-F1. Takes seconds, needs no retraining, and if it hurts you set `tau = 0`.

**Which to use?**

| Situation | Use |
|---|---|
| Not distilling | **Option A or B.** B is cleaner; A is easier to debug. Try A first. |
| **Distilling (§7.4)** | **Option C.** Training-time LA loss conflicts with KD — see the warning below. |
| Already have a trained checkpoint | **Option C.** Free improvement, no retraining. |

Never use two at once — you would be correcting the imbalance twice.

> **⚠ v3 warning — training-time logit adjustment conflicts with distillation.**
> Your teacher is trained on the same long-tailed data, so its soft targets place most mass on head classes. The KD term pulls the student toward that biased distribution while the LA loss pushes tail margins outward — the two gradients fight. This is documented: naive KD on long-tailed data "transfers this teacher bias and can further degrade tail-class performance of the student," which is why dedicated methods exist (Balanced KD, LTKD, DiVE) that reweight the distillation term by class.
>
> Do **not** try to fix this by adding the prior to both student and teacher logits before the KL. Adding the same vector to both leaves the minimiser unchanged (`student == teacher` still minimises it), so it rebalances nothing.
>
> **For a two-week project: train with plain CE + KD, then apply post-hoc adjustment at inference.** One objective during training, rebalancing afterwards, no conflict.

**Always report per-class recall.** 92% overall accuracy that never detects Red Leaf Spot is a failed model.


---

## 7. Training protocol

### 7.1 The four stages

| Stage | What | Time (Colab T4) | Skippable? |
|---|---|---|---|
| **1** | Baseline fine-tune, standard augmentation | ~1.5 h | No |
| **2** | Full recipe: field augmentation + balancing + slicing | ~2 h | No |
| **3** | **Online consistent-teaching distillation** | ~4–6 h | **Yes — skip if behind** |
| **4** | Temperature calibration + energy threshold + abstain | ~15 min | No |

**v2 note on Stage 3 timing.** v1 estimated ~3 h using cached teacher logits. Online distillation runs the teacher every batch, so budget **2–3× the student-only training time**. It is still affordable, but it is now clearly the stage to cut if you are behind schedule.

The delta between Stage 1 and Stage 2 is your best slide — it quantifies what fixing the data pipeline bought you, and answers "what did you learn from your previous attempt?"

### 7.2 Hyperparameters

```python
CONFIG = {
    'backbone':        'tf_efficientnet_lite0',
    'image_size':      224,
    'batch_size':      64,            # 32 if Colab OOM; 32 also safer with online teacher
    'epochs':          30,
    'optimizer':       'AdamW',
    'lr_head':         1e-3,
    'lr_backbone':     1e-4,          # 10x lower than head
    'weight_decay':    0.05,
    'scheduler':       'cosine',
    'warmup_epochs':   3,
    'label_smoothing': 0.1,
    'drop_rate':       0.2,
    'drop_path_rate':  0.1,
    'mixup_alpha':     0.2,           # see note in §7.4
    'cutmix_alpha':    1.0,
    'ema_decay':       0.9998,
    'amp':             True,
    'grad_clip':       1.0,
    'early_stop_patience': 7,
}
```

**Discriminative learning rates:**
```python
import torch
head_params, backbone_params = [], []
for name, p in model.named_parameters():
    (head_params if ('classifier' in name or 'head' in name or 'fc' in name)
     else backbone_params).append(p)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': head_params,     'lr': 1e-3},
], weight_decay=0.05)
```

**Warm-up trick:** freeze the backbone for the first 2 epochs (train head only), then unfreeze. Stops large random-head gradients from wrecking pretrained features. Worth ~0.5–1%, costs nothing.

**EMA** is nearly free accuracy — `timm.utils.ModelEmaV2`. Evaluate with EMA weights.

### 7.3 Training loop skeleton

```python
# train.py (essential parts)
import torch, timm
from timm.utils import ModelEmaV2

model = build_model(CONFIG['backbone']).cuda()
ema   = ModelEmaV2(model, decay=CONFIG['ema_decay'])
scaler = torch.cuda.amp.GradScaler()

sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[1e-4, 1e-3],
    total_steps=CONFIG['epochs'] * len(train_loader),
    pct_start=CONFIG['warmup_epochs'] / CONFIG['epochs'])

best_macro_f1 = 0.0
for epoch in range(CONFIG['epochs']):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer); scaler.update(); sched.step()
        ema.update(model)

    macro_f1 = evaluate(ema.module, val_loader)
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save({'model': ema.module.state_dict(),
                    'config': CONFIG, 'classes': CLASS_NAMES},
                   'artifacts/checkpoints/best.pt')
```

**Select on macro-F1, never accuracy.** With 40× imbalance, accuracy rewards ignoring rare diseases.

---

### 7.4 Stage 3 — knowledge distillation ⚠ **REWRITTEN IN v2**

> **What changed and why.** Version 1 of this handbook told you to cache the teacher's logits once over lightly-augmented images, then train the student under heavy augmentation. **That was wrong.** Beyer et al., *"Knowledge distillation: A good teacher is patient and consistent"* (CVPR 2022), tested exactly that configuration and found that distilling with precomputed teacher targets **"works much worse."** Their central finding is that the consistency criterion — student and teacher seeing the *same* views — "is the only way of performing distillation which reaches peak student performance across all datasets consistently."
>
> **The failure mechanism:** if the teacher saw a full leaf with a lesion at the tip and the student's `RandomResizedCrop(scale=0.35)` returns only clean green tissue, the KL term actively teaches the student that healthy texture means "blast." You get feature drift, not knowledge transfer.

#### 7.4.1 The rule

**The teacher and the student must receive the identical augmented tensor.** Not a mild version. Not a centre crop. The same tensor object.

This means the teacher forward pass happens **online, inside the training loop**. There is no logit cache.

#### 7.4.2 Prepare the teacher

```python
import timm, torch

teacher = timm.create_model('vit_small_patch14_dinov2.lvd142m',
                            pretrained=True, num_classes=31).cuda()
# Cheapest: freeze backbone, train only the head (linear probe) for a few epochs
for n, p in teacher.named_parameters():
    if 'head' not in n:
        p.requires_grad = False
# Better if you have time: LoRA-adapt, or unfreeze the last 2 blocks.
# patch-14 → input dims must be divisible by 14. Use 224 (= 16 x 14).

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
```

Train (or linear-probe) the teacher first, save it, then freeze it for the student run.

**Cheaper teacher option:** `convnext_tiny.in12k_ft_in1k` costs roughly a third of DINOv2 ViT-S per forward pass at 224 and is a legitimate substitute if your Colab session is slow. The point of the teacher is soft inter-class information, and a strong supervised CNN provides that.

#### 7.4.3 The distillation loss

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets,
                      T=3.0, alpha=0.5, criterion=None):
    """
    Consistent-teaching KD. Both logit sets MUST come from the same augmented input.
    alpha = weight on the soft teacher term; (1-alpha) on the hard-label term.
    T     = temperature. Higher = softer, more inter-class information.
    """
    hard = criterion(student_logits, targets)      # your CE / logit-adjusted loss
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)                                     # T^2 restores gradient magnitude
    return (1.0 - alpha) * hard + alpha * soft
```

#### 7.4.4 The training loop

```python
for imgs, labels in train_loader:
    imgs   = imgs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    # ---- Teacher sees the EXACT SAME tensor. This is the whole point. ----
    with torch.no_grad(), torch.cuda.amp.autocast():
        teacher_logits = teacher(imgs)

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        student_logits = model(imgs)
        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 T=3.0, alpha=0.5, criterion=criterion)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update(); sched.step()
    ema.update(model)
```

#### 7.4.5 Mixup and CutMix — keep them, but apply consistently

A subtlety the red team's write-up glossed over: **mixup is not the problem; inconsistent mixup is.** Beyer et al.'s best-performing configuration ("function matching") is consistent teaching *plus* mixup — mixup expands the input manifold, which helps. Independent replications report CutMix and MixUp substantially enhance KD performance, while flips and shears *degrade* it when applied inconsistently.

So: mix the batch **once**, then feed the mixed tensor to both networks.

```python
mixed_imgs, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=0.2)

with torch.no_grad(), torch.cuda.amp.autocast():
    teacher_logits = teacher(mixed_imgs)        # SAME mixed tensor

with torch.cuda.amp.autocast():
    student_logits = model(mixed_imgs)
    hard = lam * criterion(student_logits, y_a) + (1-lam) * criterion(student_logits, y_b)
    soft = F.kl_div(F.log_softmax(student_logits / T, 1),
                    F.softmax(teacher_logits / T, 1), reduction='batchmean') * T*T
    loss = (1 - alpha) * hard + alpha * soft
```

#### 7.4.6 Cost, and when to skip

DINOv2 ViT-S/14 is documented at 46.8 GMACs at 518², which scales to roughly 8.8 GMACs per image at 224². A batch of 64 is about 1.1 TFLOP; on a Colab T4 at realistic FP16 utilization that is roughly **50–150 ms per batch**, not the ~11 ms figure that circulates. Expect total training time to rise **2–3×**.

Beyer et al.'s other headline finding is that distillation needs *patient* (long) schedules. At 30 epochs you capture some but not all of the benefit.

**Therefore: Stage 3 remains optional.** If you are behind schedule, skip it entirely rather than doing it badly — a well-trained Stage 2 model beats a rushed Stage 3 one. Expected gain when it works: 1–3 points in-distribution, often more cross-domain, at **zero inference cost**.

**Sanity check to run:** if the distilled student underperforms its own non-distilled Stage 2 baseline, either your teacher is weak or something is inconsistent between the two input paths. Raise `alpha` toward 0.3 (trust hard labels more) and verify both networks receive the same tensor by asserting `teacher_input is student_input`.

---

### 7.5 Model B training (trap insect classifier) — **UPDATED IN v2**

> **What changed.** v1 offered "ESP32-CAM node or Nano" as equal options. v2 makes **gateway inference the default** and on-device the exception, for hardware reasons below.

#### 7.5.1 Where inference runs

**Default — distributed (recommended):**
```
[ESP32-CAM] --capture JPEG on timer--> Wi-Fi / ESP-NOW --> [Jetson or laptop gateway]
 deep-sleep between captures                                blob detection + CNN here
```

**Why this is right for you specifically:**
- It is what commercial trap stations do.
- Trap monitoring is a **once- or twice-daily** task. Latency is irrelevant.
- It eliminates an entire problem class instead of optimizing around it.
- It saves you the TFLite-Micro toolchain, INT8 calibration, and flashing work — days you do not have.

**Why on-device on a plain ESP32-CAM is a trap:** the ESP32-CAM's Xtensa LX6 has no vector instructions. Espressif's own ESP-NN documentation states that assembly/SIMD-optimized kernels are provided for **ESP32-S3, ESP32-P4 and ESP32-S31**, while **ESP32 and ESP32-C3 receive only "generic optimisations."** The impressive INT8 speedup figures that circulate for ESP-NN belong to the S3's 128-bit SIMD, not to an LX6. INT8 on a plain ESP32 still gives roughly 4× memory reduction and 2–4× time, but not the order-of-magnitude gains people quote.

**If on-device inference is a differentiator you want:** buy an **ESP32-S3-CAM** (~₹800–1,200), not more optimization effort on an LX6. Then TFLite Micro + ESP-NN INT8 becomes genuinely viable.

#### 7.5.2 The model

```python
# model_b.py — from scratch, ~150k params
import torch.nn as nn

class TrapPestCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):                 # x: (B,3,64,64)
        return self.head(self.pool(self.features(x)).flatten(1))
```

- **Input:** 64×64 crops from the blob detector
- **Classes:** target pests + `not_pest` (dust, debris, beneficials). Distinguishing pests from natural enemies is agronomically meaningful and a good detail to mention.
- **Data:** RP11 rice pest images + sticky trap datasets + synthetic composites (paste segmented insects onto real yellow trap backgrounds at random positions/rotations — cheap and very effective here)
- **Augmentation:** rotation (any angle — insects land arbitrarily), flips, brightness, blur, scale ±20%. No colour-destroying transforms.
- **Training:** ~40 epochs, Adam, lr 1e-3, cosine. Minutes on a T4.
- **Resolution requirement:** the thrips/whitefly study identified **80 µm/pixel** as the minimum for species-level ID, achievable with modern smartphones, action cameras or low-cost camera modules. A 5MP sensor ~15 cm from an A5 trap clears this.

#### 7.5.3 Blob detection (runs on the gateway)

HSV threshold to isolate non-yellow objects → morphological open/close → connected components → filter by area and aspect ratio → 64×64 crops. Pure OpenCV, milliseconds on the gateway. **Downscale to ~1280 px on the long side first** — you do not need 5MP for blob localization, and it makes everything faster.

---

## 8. Evaluation — how to know it actually works

### 8.1 The four numbers to report

| Metric | On what | What it tells you |
|---|---|---|
| **Macro-F1** | in-distribution test | Real performance, imbalance-aware. Primary number. |
| **Top-1 accuracy** | in-distribution test | What everyone expects. Report alongside. |
| **Cross-domain macro-F1** | PlantDoc / PlantWild held-out | **Your honest number.** |
| **Background-only accuracy** | bias audit (§6.4) | Whether your dataset leaks. Lower is better. |

Expect cross-domain to be substantially lower. That gap is documented and normal — it is the same phenomenon that produces 33% lab→field transfer in the literature. Your job is to *shrink and measure* it, not pretend it doesn't exist.

### 8.2 Also produce

- **Confusion matrix** (row-normalized). Rice blast vs. brown spot confusion is a *real* diagnostic difficulty, not just a model failure — worth saying so.
- **Per-class recall table.**
- **Grad-CAM overlays** — show the model attending to lesions, not background. The CerealConv authors used image masks precisely to verify their model used correct information. A 3-image panel (healthy / correct / failure) is one of your highest-value slides.

```python
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
cam = GradCAM(model=model, target_layers=[model.blocks[-1]])
heatmap = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_class)])
```

### 8.3 The comparison table for your PPT

| Configuration | In-dist. macro-F1 | Cross-domain macro-F1 |
|---|---|---|
| Stage 1 — baseline, standard aug | *fill in* | *fill in* |
| Stage 2 — + field augmentation | *fill in* | *fill in* |
| Stage 2 — + class balancing | *fill in* | *fill in* |
| Stage 3 — + consistent-teaching distillation | *fill in* | *fill in* |
| After FP16 TensorRT export | *fill in* | *fill in* |

### 8.4 Verify FP16 export didn't hurt

Re-run the full test set through the engine **on the Nano** and compare to PyTorch. FP16 should cost well under 0.5% macro-F1. More than that means a numerical issue (an op falling back or overflowing), not a precision tradeoff.

### 8.5 Rejection: calibration, abstain, and energy ⚠ **CORRECTED IN v3**

Three layers of rejection doing three different jobs. Do not collapse them.

```
                          [ raw logits ]
                               |
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  argmax == not_crop    E_crop(x) > τ_energy    max softmax < τ_conf
  "known negative"      "unknown object"        "uncertain diagnosis"
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               ▼
                     ABSTAIN → "flagged for review"
```

#### Layer 1 — the `not_crop` class
Anticipated negatives: soil, sky, hands, pavement. Cheap, and it makes your live demo safe.

#### Layer 2 — open-set energy score ⚠ **the v3 fix is here**

> **What was wrong in v2.** The v2 energy score summed over **all 31 logits**, and the calibration snippet chose τ by comparing in-distribution energies against energies on the **`not_crop` images used to train class 31**. Both halves are broken:
>
> 1. **Summing over all logits defeats the purpose.** A soil image produces a large `not_crop` logit, so `logsumexp` is large and energy is strongly negative — it looks *in*-distribution, because for that model it *is*.
> 2. **Calibrating on training auxiliaries is invalid.** Outlier Exposure requires auxiliary data "completely disjoint from the test time data"; reusing it for calibration violates the separation-of-information principle and produces artificially inflated numbers.

**Fix 1 — compute energy over the crop logits only.** This turns the score from *"how much does the model know about this?"* into *"how much evidence is there that this is one of my known crop conditions?"* — which is the question you want answered. A weed suppresses all 30 crop logits, so its energy is high and it is rejected, even though the model has no `weed` class.

```python
import numpy as np

def open_set_energy(logits, crop_cols, T=1.0):
    """
    Free energy over CROP diagnostic logits only, EXCLUDING the not_crop column.
    LOW energy  = confident it is a known crop condition (in-distribution)
    HIGH energy = no crop evidence (out-of-distribution / unknown object)
    logits: (N, n_classes) RAW logits — not temperature-scaled, not prior-adjusted.
    """
    z = logits[:, crop_cols] / T
    m = z.max(axis=1, keepdims=True)
    return -T * (m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))
```

**Fix 2 — three disjoint data roles.** The single `not_crop` folder in v2 was serving two incompatible purposes. Split it by **category**, not by file listing — a held-out 20% of your soil photos is still soil, and the model has seen soil.

| Role | Contents | Used for | Size |
|---|---|---|---|
| **Outlier exposure** | soil, sky, hands, pavement, blur | Training the `not_crop` class | ~1,500–2,000 |
| **Open-set calibration** | *categories never seen in training* — weeds, plastic mulch, irrigation pipe, farm machinery, brick, textures | Choosing τ_energy | ~300–500 |
| **Open-set test** | further unseen categories | Reporting OOD performance | ~200–300 |

Assembling the calibration set is about an hour of downloading and photographing. Record the split in a CSV so you can show a judge the categories are disjoint.

```python
# calibrate_energy.py
id_e  = open_set_energy(val_logits,      CROP_COLS)   # real crop images
ood_e = open_set_energy(openset_logits,  CROP_COLS)   # UNSEEN categories only

tau_energy = np.percentile(id_e, 95)          # accept 95% of real crops
fpr95 = (ood_e < tau_energy).mean()           # unknowns wrongly accepted
auroc = roc_auc_score(np.r_[np.zeros(len(id_e)), np.ones(len(ood_e))],
                      np.r_[id_e, ood_e])
print(f'tau={tau_energy:.3f}  FPR@95TPR={fpr95:.3f}  AUROC={auroc:.3f}')
```

Report **FPR@95TPR and AUROC** on the open-set *test* split. Those are the standard OOD metrics and they are honest numbers.

**If you run out of time to build the calibration set:** compute and log the energy score but do **not** apply a hard threshold. Rely on Layers 1 and 3. An uncalibrated threshold is worse than no threshold, because it fails in an unpredictable direction.

#### Layer 3 — temperature-scaled confidence
Handles *in-distribution* uncertainty: it is definitely a rice leaf, but the model cannot separate blast from brown spot.

```python
import torch, torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        return logits / self.log_T.exp()

def fit_temperature(val_logits, val_labels):
    ts = TemperatureScaler().cuda()
    opt = torch.optim.LBFGS([ts.log_T], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(ts(val_logits), val_labels)
        loss.backward(); return loss
    opt.step(closure)
    return ts.log_T.exp().item()
```

Pick τ_conf for ≥90% precision on accepted predictions, and report coverage.

#### Order of operations at inference ⚠ **CORRECTED IN v4**

> **What was wrong in v3.** The abstain gate read `max softmax(adjusted / T_cal) < τ_conf`. That gates *confidence* on **prior-adjusted** probabilities, which mixes two questions that must stay separate. A 43-image class carries a prior shift of about +6.4, so on an input with no evidence at all — every logit near zero — the rarest disease can manufacture confidence out of the shift alone.

Separate **how much evidence there is** from **which label it gets**:

```
raw logits
  1. energy on RAW logits              -> is this a known crop condition at all?
  2. confidence on UNADJUSTED probs    -> is the model sure about anything?
  3. prior-adjusted argmax             -> WHICH class, given 1 and 2 passed
```

Steps 1 and 2 measure evidence. Step 3 chooses a label. Only step 3 sees the prior.

```python
from rejection import decide      # sih_pipeline_v4/rejection.py

results = decide(logits, CROP_COLS, NOTCROP_COL, log_priors,
                 tau_energy=TAU_E, T_cal=T_CAL, tau_conf=0.60, tau_prior=1.0)
# -> [{'state': 'NOT_CROP'|'UNKNOWN'|'ABSTAIN'|'OK',
#      'class_id': int|None, 'conf': float, 'energy': float}, ...]
```

**Verified both directions** (`tests/test_pipeline.py`): 0 of 50 random OOD vectors become confident rare-class calls, and 50 of 50 genuine in-distribution tiles still pass. A defence that rejected real crops would be worse than the bug it fixes.

**On the magnitude.** Red-team Round 3 claimed OOD inputs reach 0.86–0.92 confidence on the rarest class. Computed with your actual class counts (~27,000 images, 31 classes, rarest 43) it is **0.33–0.44** — below a 0.60 gate even before the fix. The mechanism is real; the number is not. Do not put 0.86 in your PPT.

### 8.6 Inference-time aggregation ⚠ **REWRITTEN IN v3**

> **What was wrong in v2.** The v2 `aggregate_frame` masked healthy columns to `-inf` and took the argmax over the remaining disease columns. **It had no path that returns "healthy."** On a pristine field it returns whichever disease class holds the largest softmax noise floor — and because `aggregate_cell` counted class agreement with **no score threshold**, three consecutive frames of noise-level agreement fired a disease alert on a healthy crop.
>
> The v2 default of `topk=2` was also wrong. One lesion tile at 0.92 among seven healthy tiles at 0.02 gives a top-2 mean of **0.47**, which fails a 0.60 confidence gate — a false negative on exactly the early-stage infection the tiling architecture exists to catch.

#### The two axes (unchanged, still correct)

| Axis | Operator | Reasoning |
|---|---|---|
| **Spatial** — tiles within one frame | **Max over per-tile decisions** | A lesion anywhere means the frame is diseased. Logical OR, not a vote. |
| **Temporal** — frames of the same GPS cell | **k-of-n agreement + score floor** | A bird dropping appears once; a lesion appears every pass. This is where voting belongs. |

#### The implementation — now a module, not a code block

```python
from aggregate import aggregate_frame, aggregate_cell   # sih_pipeline_v4/

state, class_id, score = aggregate_frame(tile_probs, HEALTHY_COLS, NOTCROP_COL,
                                         tau_disease=0.55, tau_margin=0.10)
# state in ('DISEASE', 'HEALTHY', 'NOT_CROP', 'UNCERTAIN')

cell = aggregate_cell(frame_results, k=2, n=3, min_score=0.55)
# {'state': 'DISEASE'|'HEALTHY'|'UNCERTAIN'|'NO_DATA',
#  'class_id': int|None, 'score': float, 'n_frames': int, 'n_agree': int}
```

The frame logic decides **within each tile** first — where disease and healthy probabilities are directly comparable — then max-pools those decisions. That removes the max-versus-mean asymmetry and the dependence on tile count.

> **⚠ v4 fix — `aggregate_cell` no longer returns a bare `None`.**
> v3 returned `(None, 0.0, 0)` for a healthy cell, an all-uncertain cell, **and** a never-visited cell. Downstream prescription mapping could not tell *do not spray* from *re-fly this cell* — on a demo map, the difference between a green cell and a grey one. The four states are now distinct and tested.

| Cell state | Meaning | Action |
|---|---|---|
| `DISEASE` | k-of-n agreement **and** mean score ≥ threshold | Treat |
| `HEALTHY` | k-of-n healthy frames | Do not spray |
| `UNCERTAIN` | conflicting, or agreement **without** evidence | Re-inspect |
| `NO_DATA` | fewer than `min_frames` usable frames | Re-fly |

That third row matters: three frames agreeing on `rice__blast` at 0.09 confidence is agreement without evidence, and it returns `UNCERTAIN`, not `DISEASE`.

**Two properties worth checking against the v2 failures:**

- *Single-tile sensitivity preserved.* One tile at 0.92 against healthy 0.05 gives margin 0.87 — it votes, and `d_best[i] = 0.92` carries the frame undiluted. No top-k averaging.
- *Healthy fields return healthy.* Eight tiles at `rice__normal = 0.98`, disease noise at 0.008: no tile clears `tau_disease`, `h_best.mean() = 0.98`, verdict is `HEALTHY`. There is now a path that says so.

**Calibrate `tau_disease` and `tau_margin` on validation data.** Sweep both over a small grid, plot precision/recall for the disease verdict, and choose your operating point. That plot is a strong slide — it shows you tuned a decision rule rather than guessing a constant.

**Free localization.** The per-tile probability map still gives you a lesion heatmap for the demo video with no attention mechanism — overlay `d_best` per tile on the frame.

**Stretch — attention MIL.** Gated attention pooling (Ilse et al., ICML 2018) learns trainable tile weights and "performs much better than other methods in the small sample size regime." It needs **bag-level training data**, which your single-leaf datasets do not provide — you would have to synthesise bags. Do not start before Stage 2 is trained and exported.

## 9. Optimization and export — PyTorch → ONNX → TensorRT

### 9.1 The hard constraints, restated

| Constraint | Value | Consequence |
|---|---|---|
| Max JetPack on original Nano | **4.6.4** | Ubuntu 18.04, forever |
| Python | **3.6** | PyTorch 1.11+ needs 3.7. Also: no `multiprocessing.shared_memory` (3.8+), no f-string `=`, no dataclasses niceties from 3.7+ |
| CUDA | **10.2** | PyTorch 2.x needs CUDA 11 — impossible on Maxwell |
| TensorRT | **8.2** | ONNX opset ≤13; TRT 8 API, not TRT 10 |
| INT8 | **Not supported in hardware** | FP16 is your floor |
| **Compute capability** | **5.3** | **Pinned memory is uncached below CC 7.2 — see §9.6** |

**Train on Colab, deploy TensorRT engines. Never run PyTorch on the Nano.**

### 9.2 Export to ONNX — with fused preprocessing ⚠ **CRITICAL BUG FIXED IN v3**

> **What was wrong in v2 — read this before touching the code.**
> The v2 `FusedModel` permuted NHWC→NCHW, giving channel order **B, G, R**, then normalized with BGR-ordered mean/std, then passed the tensor straight to a backbone whose first convolution expects channel 0 to be **Red**.
>
> **Reordering the statistics does not reorder the channels.** Each channel got divided by its own correct number, and then blue pixel data was convolved with filters trained on red. v2 even carried a note saying "read that BGR reordering carefully" — pointing directly at the wrong line.
>
> This matters more here than in a generic vision task: colour is *diagnostic information* for plant disease. Chlorosis, necrosis, rust and mosaic are separated largely by hue. A red/blue swap does not cost a couple of points — it corrupts your most informative signal.

#### 9.2.1 The corrected fused wrapper

```python
# export_onnx.py
import torch
import torch.nn as nn
from model import build_model

class FusedModel(nn.Module):
    """
    Accepts raw uint8 NHWC BGR straight from OpenCV.
    Permutes, CONVERTS BGR→RGB, scales and normalizes on the GPU,
    inside the TensorRT graph.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Standard ImageNet statistics in TRUE RGB order.
        # The channel swap happens on the tensor, not on these vectors.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x_uint8):                     # (B, H, W, 3) uint8, BGR
        x = x_uint8.permute(0, 3, 1, 2)             # (B, 3, H, W) — still BGR
        x = x[:, [2, 1, 0], :, :].float() / 255.0   # ← BGR→RGB. THE FIX.
        x = (x - self.mean) / self.std              # standard RGB normalization
        return self.backbone(x)
```

The single line `x[:, [2, 1, 0], :, :]` is the whole correction. It exports to ONNX as a `Gather` on the channel axis and costs essentially nothing at runtime.

#### 9.2.2 Export

```python
CKPT = 'artifacts/checkpoints/best.pt'
ck = torch.load(CKPT, map_location='cpu')

backbone = build_model(ck['config']['backbone'], pretrained=False)
backbone.load_state_dict(ck['model'])
model = FusedModel(backbone).eval()

# v4: BATCH must equal the MAXIMUM number of tiles a frame can produce.
# v3 exported BATCH=8 while the tiler emitted up to 9 tiles, so the 9th tile
# was silently truncated by `min(len(tiles), batch)` and never classified.
# Fix the tile grid at 3x3 (= 9 tiles, 20% overlap) and match the engine to it.
BATCH = 9            # STATIC. Do not use dynamic axes.
dummy = torch.randint(0, 255, (BATCH, 224, 224, 3), dtype=torch.uint8)

torch.onnx.export(
    model, dummy, 'artifacts/onnx/model_a_fused.onnx',
    input_names=['input_bgr_uint8'],
    output_names=['logits'],
    opset_version=13,               # TRT 8.2 ceiling. Try 11 if 13 fails.
    do_constant_folding=True,
    dynamic_axes=None,
    export_params=True,
)
```

Then simplify:
```bash
pip install onnx onnxsim onnxruntime
python -m onnxsim artifacts/onnx/model_a_fused.onnx artifacts/onnx/model_a_sim.onnx \
       --overwrite-input-shape input_bgr_uint8:9,224,224,3
```

**Keep the tile count and the engine batch in one place.** Put `N_TILES = 9` in `configs/` and import it in both the tiler and the export script. The v3 bug was two documents disagreeing about a constant; two files disagreeing would be the same bug.

#### 9.2.3 Verification ⚠ **REWRITTEN IN v3 — the v2 check could not catch this bug**

> **Why v2's verification failed.** It compared PyTorch `FusedModel` against ONNX `FusedModel` on `torch.randint` noise. Two problems: both sides carried the same bug, and **uniform random noise has identical marginal statistics in all three channels**, so a channel swap is invisible to it. A noise-based tensor diff is structurally incapable of detecting channel errors.

Run all three checks. The first two are new in v3 and specifically target channel order.

**Check 1 — red-flag image.** Deliberately unequal channel means. If channels are swapped, the outputs diverge sharply.

```python
import numpy as np, torch, cv2

def red_flag_check(fused_model, backbone, size=224):
    """A strongly red patch. Swapped channels give a very different answer."""
    bgr = np.zeros((1, size, size, 3), np.uint8)
    bgr[..., 2] = 200          # OpenCV channel 2 = RED
    bgr[..., 1] = 40
    bgr[..., 0] = 20

    with torch.no_grad():
        out_fused = fused_model(torch.from_numpy(bgr)).numpy()

        # Independent reference path: convert to RGB and normalize by hand
        rgb = cv2.cvtColor(bgr[0], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        ref_in = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None])
        out_ref = backbone(ref_in).numpy()

    diff = np.abs(out_fused - out_ref).max()
    assert diff < 1e-3, f'CHANNEL ORDER MISMATCH: max diff {diff:.4f}'
    print(f'red-flag check passed (max diff {diff:.2e})')
```

**Check 2 — real images, end to end.** Run your whole test set through the fused path with images loaded by `cv2.imread`, and confirm macro-F1 matches the RGB reference pipeline to within ~0.5%. This is the check that would have caught the v2 bug in production.

```python
f1_fused = evaluate_fused(model, test_csv)      # cv2.imread → uint8 BGR → FusedModel
f1_ref   = evaluate_reference(backbone, test_csv)  # Albumentations RGB path
assert abs(f1_fused - f1_ref) < 0.005, f'{f1_fused:.4f} vs {f1_ref:.4f}'
```

**Check 3 — ONNX export fidelity** (this is the v2 check; keep it, but it is now the *least* important of the three).

```python
import onnxruntime as ort
sess = ort.InferenceSession('artifacts/onnx/model_a_sim.onnx',
                            providers=['CPUExecutionProvider'])
x = torch.randint(0, 255, (8, 224, 224, 3), dtype=torch.uint8)
with torch.no_grad():
    torch_out = model(x).numpy()
onnx_out = sess.run(None, {'input_bgr_uint8': x.numpy()})[0]
print('max abs diff:', np.abs(torch_out - onnx_out).max())   # want < 1e-3
```

If any check fails, stop. Do not proceed to TensorRT — you will be debugging two problems at once.

**Fallback:** if the fused graph refuses to export or build, fall back to the plain model (RGB float input) plus explicit `cv2.cvtColor(..., COLOR_BGR2RGB)` on the host. The fusion is an optimization, not a requirement — and the plain path makes the channel conversion visible in code where you can see it.

### 9.3 Build the TensorRT engine (on the Nano itself)

Engines are **not portable** across TensorRT versions, GPU architectures, or driver versions.

```bash
sudo nvpmodel -m 0        # MAXN (needs 5V/4A barrel jack + fan)
sudo jetson_clocks        # lock clocks to maximum

/usr/src/tensorrt/bin/trtexec \
  --onnx=model_a_sim.onnx \
  --saveEngine=model_a_fp16.engine \
  --fp16 \
  --workspace=1024 \
  --verbose 2>&1 | tee build.log
```

5–20 minutes. If OOM-killed, add swap:
```bash
sudo fallocate -l 6G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```
Swap is for the **build only** — never rely on it at inference.

Benchmark immediately, and record the number for your PPT:
```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=model_a_fp16.engine \
                              --iterations=200 --avgRuns=100
```

### 9.4 Free memory before running

```bash
sudo systemctl set-default multi-user.target   # headless — frees ~0.5-1 GB
sudo reboot
# revert with: sudo systemctl set-default graphical.target
```

No PyTorch at runtime (saves ~1 GB and seconds of startup). No Jupyter.

### 9.5 Inference code for the Nano ⚠ **TEARDOWN FIXED IN v3**

> **Two changes.** (1) The per-image Python preprocessing loop is gone — with the fused graph, you copy raw uint8 tiles and nothing else. (2) The CUDA context is now created **inside the worker thread**, because `pycuda.autoinit` binds the context to the importing thread and calling `execute_async_v2()` from a different thread raises an invalid-context error.

```python
# trt_infer.py — pure TensorRT + PyCUDA. No PyTorch. No pycuda.autoinit.
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTClassifier:
    """
    IMPORTANT: construct and use this object entirely within ONE thread,
    and create the CUDA context in that same thread (see run_inference_thread).
    """
    def __init__(self, engine_path, batch=8, size=224, num_classes=31,
                 temperature=1.0):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.batch, self.size, self.nc = batch, size, num_classes
        self.T = temperature                       # from §8.5 calibration

        # Fused graph takes uint8 NHWC BGR — no float conversion on the host.
        self.h_in  = cuda.pagelocked_empty((batch, size, size, 3), np.uint8)
        self.h_out = cuda.pagelocked_empty((batch, num_classes), np.float32)
        self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

    def _fill(self, bgr_tiles):
        """Vectorized. No per-image Python arithmetic."""
        n = min(len(bgr_tiles), self.batch)
        for i in range(n):
            t = bgr_tiles[i]
            if t.shape[0] != self.size or t.shape[1] != self.size:
                t = cv2.resize(t, (self.size, self.size))
            self.h_in[i] = t                       # raw uint8 memcpy
        if n < self.batch:                         # pad the tail
            self.h_in[n:] = self.h_in[max(n - 1, 0)]
        return n

    def infer(self, bgr_tiles):
        n = self._fill(bgr_tiles)
        if n == 0:
            return np.empty((0, self.nc), np.float32)
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_in), int(self.d_out)],
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()

        logits = self.h_out[:n].astype(np.float32)
        return logits                              # raw logits — see below

    def close(self):
        """
        v3: release EVERY PyCUDA object before the CUDA context is destroyed.
        If a DeviceAllocation, Stream or pagelocked host buffer is still alive
        when ctx.detach() runs, garbage collection calls cuMemFree on a dead
        context and PyCUDA raises LogicError, corrupting process exit.
        """
        for attr in ('d_in', 'd_out'):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, attr, None)
        # execution context and engine hold their own device memory
        for attr in ('context', 'engine', 'stream', 'h_in', 'h_out'):
            if hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()


def softmax_np(logits, T=1.0):
    z = logits / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ⚠ v4: the stale all-logit `energy_np` that lived here in v3 is DELETED.
# It summed over all 31 logits — including not_crop — which made a confidently
# detected soil image score E = -14.0 and look perfectly in-distribution,
# reintroducing the Round 2 bug that §8.5 had already fixed. It also hardcoded
# axis=1 and raised AxisError on a 1D vector.
#
# Import the single implementation instead. One definition cannot drift.
from rejection import open_set_energy          # sih_pipeline_v4/rejection.py
```

#### Processing every tile — `infer_all`

```python
    def infer_all(self, bgr_tiles):
        """
        v4: chunk-loop so an unexpected tile count can NEVER silently truncate.
        With a deterministic 3x3 grid this runs once; the loop is the safety net.
        """
        if not bgr_tiles:
            return np.empty((0, self.nc), np.float32)
        chunks = [self.infer(bgr_tiles[i:i + self.batch])
                  for i in range(0, len(bgr_tiles), self.batch)]
        return np.concatenate(chunks, axis=0)
```

**Return raw logits, not probabilities.** The energy score (§8.5) must be computed on unscaled logits, while the confidence gate uses temperature-scaled ones. Returning probabilities from `infer()` throws away the information you need for Layer 2.

#### Running it in a worker thread — the CUDA context fix

```python
# pipeline.py
import threading, queue
import pycuda.driver as cuda

def run_inference_thread(engine_path, tile_q, result_q, stop_evt, temperature):
    # CUDA context MUST be created in the thread that will use it.
    cuda.init()
    ctx = cuda.Device(0).make_context()
    clf = None
    try:
        clf = TRTClassifier(engine_path, temperature=temperature)
        while not stop_evt.is_set():
            try:
                frame_id, tiles = tile_q.get(timeout=0.5)
            except queue.Empty:
                continue
            logits = clf.infer(tiles)
            result_q.put((frame_id, logits))
    finally:
        # v3: ORDER MATTERS. Free all GPU/pinned allocations, THEN destroy
        # the context. Reversing these two lines is the teardown crash.
        if clf is not None:
            clf.close()
        ctx.pop()
        ctx.detach()

# start it:
t = threading.Thread(target=run_inference_thread,
                     args=(ENGINE, tile_q, result_q, stop_evt, TEMP),
                     daemon=True)
t.start()
```

Note there is **no `import pycuda.autoinit`** anywhere. That import is what creates a context on the main thread and causes the invalid-context crash.

**v3 addition — the teardown order.** `ctx.detach()` destroys the CUDA context. Any `DeviceAllocation`, `Stream` or `pagelocked_empty` buffer still alive afterwards will be collected by the garbage collector, which calls `cuMemFree`/`cuMemFreeHost` against a dead context and raises `pycuda._driver.LogicError`. That corrupts process exit and can block a clean restart of the capture thread — which is exactly when you would want to restart, mid-demo. `clf.close()` before `ctx.pop()` fixes it.

### 9.6 Optimization levers, ranked by actual payoff ⚠ **UPDATED IN v2**

| Rank | Lever | Gain | Where |
|---|---|---|---|
| 1 | **Frame gating** (only 3–8% of frames reach the model) | 10–30× effective | Companion report §5.2 Stage 1 |
| 2 | **FP16 TensorRT vs. PyTorch FP32** | ~4–6× | §9.3 |
| 3 | **Input resolution** 224 → 192 → 160 | quadratic | Retrain and measure accuracy cost |
| 4 | **Backbone choice** (lite0 vs. ResNet50) | 2–5× | §3.1 |
| 5 | **In-graph preprocessing fusion** (new in v2) | removes a CPU stall | §9.2 |
| 6 | **Batching tiles** (one call for 8 tiles) | 1.3–2× | §9.5 |
| 7 | **Headless boot** | enables the rest | §9.4 |
| 8 | **Async capture/infer threads** | hides I/O latency | Companion report §5.3 |
| — | ~~INT8 quantization~~ | **zero — unavailable** | No Maxwell INT8 hardware |
| — | ~~Zero-copy mapped memory~~ | **likely negative — see below** | |

#### Why zero-copy is rejected on this hardware (new in v2)

You will encounter advice — including in the red team review — to use `cudaHostAllocMapped` zero-copy memory on the Jetson, on the grounds that CPU and GPU share physical RAM so the `memcpy` is wasteful. On Xavier and later this is correct and can be a large win.

**On your board it is likely a regression.** NVIDIA's CUDA-for-Tegra documentation states that pinned memory **is not cached on Tegra devices with compute capability below 7.2**. The Jetson Nano's Maxwell GM20B is **compute capability 5.3**. Uncached mapped memory means every GPU access is a cache miss served from main memory. Analyses of this exact hardware generation note that zero-copy on TX1/TX2-class devices "would produce higher latencies and more bandwidth usage, since every shared memory access by the device will be a cache miss." The widely-circulated ~6× zero-copy benchmark was measured on a **Xavier** (CC 7.2, which has I/O coherency).

**Decision: do not restructure your inference path around zero-copy.** If you are curious, benchmark it as a side experiment — but the null hypothesis on CC 5.3 is "no gain or a regression," and you do not have time to chase it. This is also a *good* thing to be able to say to a judge (see §13).

### 9.7 Realistic performance targets

Your drone moves slowly and consecutive frames are ~95% redundant, so the real requirement is **1–3 inferences per second on selected frames**, not 30 FPS on a stream.

| Metric | Target on Maxwell Nano |
|---|---|
| Single 224² image, FP16 TRT, EfficientNet-Lite0 | 12–25 ms |
| Batch of 8 tiles (fused graph) | 60–120 ms |
| Full pipeline (capture → gate → tile → infer → aggregate) | 2–4 scenes/sec |

Reference points: on Nano, ResNet18 runs ~26 ms FP32 → ~18 ms FP16; ResNet50 ~79 ms → ~42 ms; YOLO-class detectors manage roughly 5–8 FPS.

**Measure your own numbers and report those.** A measured 2.8 scenes/sec beats a claimed 30 FPS in front of anyone who knows the hardware.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.onnx.export` fails on unsupported op | Backbone uses an op absent at opset 13 | Drop to rung 1 (`tf_efficientnet_lite0`). Or try opset 11. |
| Fused ONNX export fails on `permute`/uint8 cast | Older exporter quirk | Fall back to plain model + vectorized CPU preprocess (§9.5) |
| `trtexec` fails: "no implementation for node" | TRT 8.2 lacks a kernel (common with attention/SDTA) | Run `onnxsim` first; else fall back a rung in §3.1 |
| Engine builds but outputs garbage | BGR/RGB swap in the fused wrapper | Run the §9.2.3 **red-flag check** — a noise-based tensor diff cannot detect channel order |
| Accuracy fine on val, ~random on real camera frames | Channel order: missing `x[:, [2,1,0]]` | §9.2.1. This is the single most likely silent bug in the whole pipeline. |
| **Healthy field triggers disease alerts** | `aggregate_frame` has no HEALTHY path, or `aggregate_cell` has no score floor | §8.6 — both fixed in v3 |
| **Obvious lesion reported as uncertain** | top-k averaging diluting a single-tile lesion | §8.6 — use per-tile max, not top-2 mean |
| Energy score rejects nothing / rejects everything | Energy summed over all logits including `not_crop`, or τ calibrated on training auxiliaries | §8.5 — energy over crop columns only, calibrate on disjoint unseen categories |
| `LogicError: cuMemFree failed: context is destroyed` on exit | GPU buffers outlived `ctx.detach()` | §9.5 — call `clf.close()` before `ctx.pop()` |
| **Rare disease fires on weeds / bare ground** | Confidence gate reading prior-adjusted probabilities | §8.5 — gate on unadjusted probs; prior affects the label choice only |
| **Lesion in the last tile never detected** | Engine batch < tile count; the tail is truncated | §9.2 — `BATCH = 9`, deterministic grid, `infer_all` chunk loop |
| **Healthy and unvisited cells look identical on the map** | `aggregate_cell` returning bare `None` | §8.6 — explicit `HEALTHY` / `NO_DATA` states |
| **Every drought alarm suppressed** | Canopy gate at `Ta+7` rejecting stressed canopies as soil | Report 4 §1.2 — gate separates tissue from soil, not healthy from stressed |
| **Pure green field reports ~50% vegetation** | Otsu on a unimodal ExG histogram | Report 4 §1.2 — absolute ExG threshold |
| **Whiteflies vanish whenever a moth is on the trap** | Global `0.3*dist.max()` marker threshold | Report 4 §3.3 — dual-threshold markers |
| Tail-class recall collapses when distilling | Training-time logit adjustment fighting biased teacher soft targets | §6.7 — use post-hoc adjustment instead |
| **`CUDA_ERROR_INVALID_CONTEXT` / context errors** | `pycuda.autoinit` bound the context to the importing thread | **Create the context inside the worker thread (§9.5). Remove all `import pycuda.autoinit`.** |
| **`ImportError: cannot import name 'shared_memory'`** | `multiprocessing.shared_memory` is Python 3.8+; Nano has 3.6 | Use `threading` (NumPy/OpenCV release the GIL) or `multiprocessing.RawArray` |
| Nano OOM during engine build | 4GB shared memory | Add swap, boot headless, reduce `--workspace` |
| Nano OOM at inference | PyTorch imported, or GUI running | Remove torch from the runtime path; boot headless |
| Inference slower than benchmark | Clocks not locked, or thermal throttling | `nvpmodel -m 0 && jetson_clocks`; check `tegrastats`; add a fan |
| Great val accuracy, terrible on real footage | Domain gap / dataset leakage | Bias audit (§6.4); strengthen augmentation (§6.6); re-check split grouping (§6.3) |
| Model predicts one class constantly | Class imbalance, or LR too high | §6.7; lower backbone LR |
| Training accuracy near zero early on | Mixup/CutMix active | Normal. Watch validation, not training accuracy. |
| **Distilled student worse than its own baseline** | Teacher and student seeing different views | Assert both receive the identical tensor (§7.4). Raise `alpha` toward 0.3. |
| Frames report healthy despite visible lesions | Tile aggregation using voting instead of max | §8.6 — spatial max/top-k, temporal k-of-n |
| Colab disconnects mid-training | Runtime limits | Checkpoint every epoch to Drive; resume |

**Contingency ladder.** If TensorRT export defeats you entirely:
1. ONNX Runtime on the Nano's CPU — slow (~200–400 ms/image) but *works*, and your throughput requirement is low
2. Run the model on a laptop, stream frames from the drone, present it as a ground-station architecture
3. Run on recorded footage for the video and state clearly that live on-device inference is in progress

Any of these is recoverable. Having no working model is not — which is why §12 puts export on Day 3.

---

## 11. If you buy the Jetson Orin Nano Super (8GB)

Specs: Ampere GPU (**compute capability 8.7**), 1024 CUDA cores, **32 tensor cores**, 6-core ARM CPU, **67 INT8 TOPS** (40 before the JetPack 6.2 update), memory bandwidth **102 GB/s** (up from 68), JetPack 6.x, CUDA 12.6, TensorRT 10.3, Python 3.10, PyTorch 2.x. ~$249; available in India through authorized distributors.

| Area | Change |
|---|---|
| **Quantization** | INT8 becomes available and fast. PTQ with a ~500-image calibration set; expect ~2× over FP16. |
| **Backbone** | Skip the risk ladder. Run `edgenext_small`, `efficientformer_l1`, `fastvit`, `repvit`, or `convnext_tiny` directly. |
| **Distillation** | Optional — you can often deploy a model near the teacher's size instead. |
| **Resolution** | Move to 320 or 384. Fine lesion texture usually benefits materially. |
| **Model C** | Segmentation becomes practical to run concurrently. |
| **Model B** | Run a real detector (YOLO11s / RT-DETR) with SAHI in real time; drop the classical blob stage. |
| **Concurrency** | Python 3.10 → `multiprocessing.shared_memory` becomes available if you want process isolation. |
| **Zero-copy** | **Now worth testing.** CC 8.7 ≥ 7.2, so I/O coherency and cached pinned memory apply — the technique rejected in §9.6 becomes legitimate here. Benchmark it. |
| **Throughput** | 30–60+ FPS on YOLO-class models vs. 5–8 FPS on the original Nano. |

**Do not let this possibility delay you.** Build for Maxwell. The Orin is a performance upgrade, not an architecture change.

---

## 12. Build order

Ordered so the highest-risk item (export) is attempted early, while you can still recover.

**Day 1 — data, no training**
- Download Paddy Doctor (Kaggle CLI), both sugarcane sets, PlantDoc
- Assemble `not_crop` (~1,800 images) — outlier exposure for training. Keep the **open-set calibration set separate and category-disjoint** (§8.5)
- Write `configs/classes.py`
- Dedup (§6.2) → grouped stratified split (§6.3) → CSV manifests
- Pre-resize everything to max side 512

**Day 2 — baseline + bias audit**
- Bias audit (§6.4). Record the number. This is a slide.
- Stage 1 (baseline). Record macro-F1.
- Stage 2 (field augmentation + balancing). Record macro-F1.
- **The delta is your headline result.**

**Day 3 — export, before anything else**
- Build the fused wrapper (§9.2.1), export ONNX, simplify, verify numerically **and end-to-end on real BGR images**
- Copy to Nano, build TensorRT engine, benchmark
- **If this fails, you now have 11 days to fix it instead of 1.**

**Day 4 — evaluation, calibration, rejection**
- **Write `tests/test_pipeline.py` first (§15). Four assertions, ten minutes.** Every critical bug found in red-team Round 2 is caught by these.
- Cross-domain test on PlantDoc/PlantWild (open this once)
- Confusion matrix, per-class recall, Grad-CAM panel
- Temperature calibration; sweep `tau` for post-hoc logit adjustment (§6.7)
- Assemble the **open-set calibration set** (~300–500 images of *unseen* categories: weeds, pipe, plastic, machinery) and fit τ_energy on it (§8.5) — **not** on your `not_crop` training images
- Implement spatial/temporal aggregation (§8.6) and calibrate `tau_disease` / `tau_margin` on validation
- Verify FP16 engine accuracy matches PyTorch

**Day 5 — distillation (optional) + Model B**
- Online consistent-teaching distillation if time permits; re-export if it wins
- Train Model B; build the gateway blob detector

**Day 6–7 — integration and internal hackathon**
- Wire into the pipeline (CUDA context inside the inference thread, §9.5)
- Rehearse; prepare the "why not 99%" answer

**Days 8–14 — trap node, drone integration, filming, PPT**
(See companion report §8.)

---

## 13. Quick answers

**"Are we using pretrained models or building from scratch?"**
Model A and C: fine-tuning ImageNet-pretrained backbones — mandatory at 20–30k images. Model B: a small CNN from scratch, because the task is simple and constrained. The custom parts of Model A are the head, the class taxonomy, the augmentation pipeline, the consistent-teaching distillation setup, the calibration and energy-rejection layers, and the MIL-style aggregation.

**"What exactly do I type to get the model?"**
```python
import timm
model = timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=31)
```

**"How many models total?"** Two required, one optional.

**"Which dataset is most important?"** Paddy Doctor. Indian, field-captured, expert-annotated, large, includes pest-damage classes.

**"Why is my accuracy lower than papers claiming 99%?"**
Because those papers evaluate on PlantVillage, where a model trained on 8 background pixels scores 49% against 2.6% chance. You measure on field data with a grouped, deduplicated split.

**"Do I need INT8 quantization?"** No — you cannot have it. Maxwell has no INT8 hardware.

**"Should I use zero-copy memory since the Jetson shares RAM?"**
No, not on this board. Pinned memory is uncached below compute capability 7.2 and yours is 5.3, so every GPU access becomes a cache miss. The benchmarks people cite for this were measured on Xavier. It becomes worth testing if you upgrade to Orin.

**"Why did my distilled model get worse?"**
Almost certainly the teacher and student saw different augmented views. They must receive the identical tensor — see §7.4.

**"My frame has an obvious lesion but the system says healthy."**
Either you are voting across tiles, or you are averaging a top-k that dilutes the single lesion tile. Use the per-tile decision + max-pool in §8.6. Keep voting for the *temporal* axis only.

**"My healthy field is throwing disease alerts."**
Your aggregation has no HEALTHY path. The v2 version of `aggregate_frame` masked healthy columns and took an argmax over diseases, so it could only ever return a disease. §8.6 fixes it.

**"Do I calibrate the energy threshold on my `not_crop` images?"**
No — and v2 of this handbook wrongly said yes. Once you train a `not_crop` class, those images are in-distribution for the model and their energy looks perfectly normal. Calibrate on a *third*, category-disjoint set of things the model has never seen. §8.5.

**"My model works in validation but is near-random on the live camera."**
Check channel order first. If your fused ONNX graph does not contain `x[:, [2,1,0]]`, you are feeding BGR to an RGB backbone. §9.2.1.

**"Can I add attention modules / an ensemble / an LSTM head?"**
Almost certainly not worth it. Each addition risks ONNX export failure and costs Nano latency, and none will help as much as the augmentation pipeline in §6.6. The one attention mechanism worth considering is MIL pooling (§8.6), and only after everything else works.

**"How do I answer 'what did you actually build?'"**
"A crop-conditioned disease classifier fine-tuned from an ImageNet backbone with a field-conditioned augmentation pipeline designed against documented agricultural domain shift; consistent-teaching distillation from a DINOv2 foundation model; a three-layer rejection stack combining a trained negative class, energy-based OOD scoring and temperature-calibrated abstention; and MIL-style spatial aggregation so single-tile lesions aren't outvoted. Exported to FP16 TensorRT, running at *N* ms on a 2019 Jetson Nano with no INT8 support. Here's our ablation table and our dataset bias audit."

---

## 14. Reference index

**Models**
- EdgeNeXt — arXiv:2206.10589 · https://github.com/mmaaz60/EdgeNeXt
- EfficientFormer (MobileViT latency critique) — arXiv:2206.01191
- DINOv2 — arXiv:2304.07193 · https://github.com/facebookresearch/dinov2
- DINOv3 — arXiv:2508.10104
- timm — https://github.com/huggingface/pytorch-image-models

**Datasets** — all links in §5

**Methods**
- **Consistent-teaching distillation — Beyer et al., CVPR 2022, arXiv:2106.05237** *(basis for the v2 §7.4 rewrite)*
- **Energy-based OOD detection — Liu et al., NeurIPS 2020, arXiv:2010.03759** *(basis for §8.5)*
- **Outlier Exposure — Hendrycks et al., ICLR 2019, arXiv:1812.04606** *(auxiliary data must be disjoint from test-time data — basis for the v3 §8.5 correction)*
- **Balanced KD for long-tailed learning — arXiv:2104.10510**; LTKD — arXiv:2506.18496 *(basis for the v3 §6.7 warning)*
- **Attention-based deep MIL — Ilse et al., ICML 2018, arXiv:1802.04712** · https://github.com/AMLab-Amsterdam/AttentionDeepMIL *(basis for v2 §8.6)*
- Logit adjustment for long-tail — Menon et al., ICLR 2021
- SAHI (slicing) — arXiv:2202.06934 · https://github.com/obss/sahi
- PlantVillage bias — arXiv:2206.04374
- DINOv2 + LoRA + KD in agriculture — *Comput. Electron. Agric.*, S0168169925010063
- Albumentations — https://albumentations.ai/docs/

**Hardware**
- Jetson Nano INT8 unsupported — NVIDIA Developer Forums threads 84060, 83036
- **CUDA for Tegra — pinned memory uncached below CC 7.2** — https://docs.nvidia.com/cuda/archive/12.2.2/cuda-for-tegra-appnote/
- PyTorch on Jetson Nano (Python 3.6 / CUDA 10.2 ceiling) — https://qengineering.eu/install-pytorch-on-jetson-nano.html
- ESP-NN supported chips — https://github.com/espressif/esp-nn
- Jetson Orin Nano Super — https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/


---

## 15. Regression tests — **now shipped as runnable code**

Across three red-team rounds, roughly two-thirds of findings were real, and **most of Round 3's findings were bugs in Round 2's fixes.** One Round 2 bug was reintroduced two sections later in the same document. That loop does not converge while the code lives in prose.

`sih_pipeline_v4/` is the fix. Every function the audits have broken is now an importable module with a test that names the finding it guards.

```
sih_pipeline_v4/
├── configs/classes.py        31-class taxonomy, column groups (ONE source of truth)
├── rejection.py              open_set_energy (1D/2D safe), posthoc_logit_adjust, decide()
├── aggregate.py              aggregate_frame (MIL max-pool), aggregate_cell (explicit states)
├── thermal.py                canopy_temperature (corrected gates), vegetation_mask (absolute)
├── trap_segmentation.py      dual-threshold markers, hole filling, watershed
└── tests/test_pipeline.py    15 tests
```

```
$ cd sih_pipeline_v4 && python -m pytest tests/ -v
15 passed in 0.22s
```

### What each test guards

| Test | Guards against |
|---|---|
| `r2_healthy_frame_is_not_disease` | R2: no HEALTHY path; noise won the argmax |
| `r2_single_lesion_tile_is_detected` | R2: top-2 mean diluted 0.92 → 0.47 |
| `r2_empty_and_tiny_input_never_nan` | R2: `[-0:]` returned the full array → NaN |
| `r2_bare_soil_rejected_for_cwsi` | R2: inverted spread gate accepted hot soil |
| `r2_mixed_canopy_soil_accepted` | R2: same gate rejected every valid field frame |
| `r2_energy_excludes_notcrop_column` | R2: all-logit energy made `not_crop` look in-distribution |
| `r3_stressed_canopy_is_not_rejected` | R3: `Ta+7` discarded the drought it exists to find |
| `r3_stressed_canopy_still_separates_from_soil` | the widened gate must still reject soil |
| `r3_pure_canopy_vegetation_mask_not_bisected` | R3: Otsu bisected a 100% green field |
| `r3_soil_frame_vegetation_mask_near_zero` | the absolute threshold must still reject soil |
| `r3_ood_input_does_not_explode_into_rare_class` | R3: prior shift manufacturing confidence |
| `r3_real_crop_still_passes_the_gates` | the OOD defence must not reject real crops |
| `r3_cell_states_are_distinguishable` | R3: `None` conflating healthy with no-data |
| `r3_energy_handles_1d_and_2d` | R3: stale helper crashed on 1D input |
| `r3_micro_pests_survive_next_to_a_large_insect` | R3: global threshold erased every whitefly |

### These are tripwires, not tautologies

Verified by running the v4 assertions against the **v3** implementations:

```
FAIL  stressed canopy 45.5C at Ta=38   -> v3 returns (None, 'no_transpiring_vegetation')
FAIL  pure canopy vegetation fraction  -> v3 Otsu gives 47.60%, need >95%
FAIL  cell state collapse              -> healthy (None,0.0,0) == no-data (None,0.0,0)
FAIL  all-logit energy on not_crop     -> E = -14.00, looks in-distribution
FAIL  1D energy vector                 -> AxisError: axis 1 out of bounds
FAIL  micro-pests beside a large insect-> v3 finds 1 marker of 6; v4 finds 6 of 6
```

### One thing worth internalising

Building the micro-pest test surfaced a bug **no audit found**: `adaptiveThreshold` with `blockSize=25` responds only near the *edges* of regions larger than the block, so a large insect segments as a hollow ring and `dist.max()` never inflates. Two bugs were partially cancelling each other. Twenty minutes of execution found what three rounds of reading did not.

**Run `pytest` before every training job, every export, and once on the Nano.** If a Round 4 audit arrives, run its claims against this suite before changing a line — most will resolve in minutes.
