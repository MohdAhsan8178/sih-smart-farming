# Dataset Provenance Specification

orig_folder is the provenance key. One folder = one capture source. Do not mix sources into a single subfolder.

---

## 1. Provenance Mapping

| orig_folder | Provenance & Device Characteristics |
| :--- | :--- |
| `hands` | WhatsApp-recompressed, 3120×4160, EXIF stripped |
| `feet_shoes`, `pavement`, `walls`, `soil_field`, `green_noncrop` | Field capture, Galaxy A52, 4624×3468 |
| `soil` | Public soil-type dataset (5 soil type subdirectories) |
| `sky` | Public dataset |

Rule: orig_folder is the provenance key. One folder = one capture source. Do not mix sources into a single subfolder.
Future web-sourced weed images will go into a separate `green_noncrop_web/` directory, never into `green_noncrop/`.

---

## 2. EXIF Orientation Findings

- **Smartphone Field Captures**: 1,063 of 1,154 newly shot images carry non-standard EXIF orientation tags (`Tag 6` rotate 90 CW, `Tag 8` rotate 270 CW, `Tag 3` rotate 180).
  - `feet_shoes`: 224 / 224 have EXIF (99 tag 6, 67 tag 8, 58 tag 3).
  - `pavement`: 253 / 253 have EXIF (250 tag 6, 3 tag 1).
  - `walls`: 114 / 114 have EXIF (114 tag 6).
  - `soil_field`: 284 / 284 have EXIF (279 tag 6, 5 tag 1).
  - `green_noncrop`: 214 / 214 have EXIF (194 tag 6, 20 tag 1).
- **WhatsApp Transfer**: `hands` contains 0 / 65 images with EXIF tags. WhatsApp strips metadata during compression and pre-rotates the pixel raster into vertical 3120×4160.

---

## 3. F2 Architectural Note: Dedup vs. Dataset Orientation Discrepancy

- `train/dedup.py:28` loads images using `PIL.Image.open()` without calling `PIL.ImageOps.exif_transpose()`. Consequently, perceptual hashing is calculated on the raw un-rotated camera buffer.
- `train/dataset.py:110` loads images using OpenCV (`cv2.imread()`), which parses EXIF metadata and applies orientation correction automatically.
- Images with orientation tags 6, 8, or 3 are perceptual-hashed in landscape but trained in portrait.
- dedup cannot detect rotation-differing duplicates (measured self-distance ~32 vs threshold 5). None were present in the September 2026 ingest. Not verified safe — verified absent.

---

## 4. D1 Class Imbalance Remediation & Prior Adjustment Interaction

- **Decision**: Sqrt-inverse-frequency class weighting is used in training loss:
  $$w_c = \sqrt{\frac{N_{\text{total}}}{N_{\text{classes}} \cdot N_c}}$$
  normalized such that $\text{mean}(w) = 1.0$.
- **Arithmetic & Dynamic Range (Manifest vs Training Split Provenance)**:
  - **Full Manifest (`splits/all_images.csv`, 30,039 images)**: Counts range from $N_{\text{max}} = 3,178$ (`rice__blast`) down to $N_{\text{min}} = 224$ (`wheat__healthy`), yielding a raw imbalance ratio of $\frac{3178}{224} = 14.1875 \approx 14.19:1$, with $\sqrt{14.1875} \approx 3.77:1$. This reflects total harvested inventory across all partitions.
  - **Training Partition (`splits/train.csv`, 16,526 images)**: Counts range from $N_{\text{max}} = 1,535$ (`rice__normal`) down to $N_{\text{min}} = 102$ (`wheat__brown_rust`), yielding a raw imbalance ratio of $\frac{1535}{102} = 15.0490 \approx 15.05:1$, with $\sqrt{15.0490} \approx 3.88:1$.
  - **Canonical Status**: Training loss weights $w_c = \sqrt{\frac{N_{\text{total}}}{N_{\text{classes}} \cdot N_c}}$ are canonically derived from `splits/train.csv` ($15.05:1$ raw, $3.88:1$ sqrt dynamic range). Loss weights must reflect the empirical batch sampling distribution encountered by the optimizer during gradient descent; computing loss weights against the full manifest would incorrectly contaminate training penalties with holdout proportions from validation, in-distribution test, and cross-domain test splits.
  - As noted in `core/rejection.py`, post-hoc logit prior adjustment parameters (`log_priors`, `tau_prior`) must be re-calibrated in Step 15 against the weighted model.

---

## 5. Open-Set vs. Outlier-Exposure Roles & Calibration Governance

Per `core/rejection.py:116-118` and `docs/ULTIMATE_IMPLEMENTATION_PLAN_1.md:442`, `openset_holdout/` (174 images carved from `not_crop`) cannot calibrate `TAU_ENERGY` because it represents outlier-exposure data, which is in-distribution for the trained model's rejection head. Both sets are preserved with strictly separated roles:

1. **`data/raw/openset/` (and `data/packaged_min/openset/`) [5,255 images]**:
   - **Role**: Fits `TAU_ENERGY` in Step 15 via `core.rejection.fit_energy_threshold(id_logits, openset_logits, CROP_COLS, tpr=0.95)`.
   - **Status**: True Out-of-Distribution (OOD). Categories are disjoint from the 29-class training taxonomy and disjoint from `not_crop`.
   - **Composition**:
     - 3,955 images are plant-disease-like (near-OOD): `hasan_openset` (584 files: sheath blight, leaf scald, narrow brown spot), `plantwild_unmatched_wheat` (799 files: head scab, loose smut, stem rust, black chaff), and `other_crop_diseases` (2,572 files: apple, tomato, bell pepper, blueberry, etc.).
     - 1,300 images are weeds (`broadleaf_weeds`).
     - 6 of the 7 categories originally declared in `splits/openset_categories.csv` (`brick`, `dtd_textures`, `farm_machinery`, `irrigation_pipe`, `plastic_mulch`, `straw_mulch`) remain empty placeholder directories.
   - **Known Limitation**: A threshold fitted predominantly on near-OOD (unseen crop diseases and weeds) may exhibit different rejection characteristics on far-OOD inputs (e.g. agricultural machinery, masonry, synthetic textures). Documented as an intentional calibration constraint; no additional image collection is to be conducted at this stage.

2. **`data/raw/openset_holdout/` (and `data/packaged_min/openset_holdout/`) [174 images]**:
   - **Role**: Evaluates `not_crop` outlier-exposure generalisation only.
   - **Constraint**: **Never used to fit a rejection threshold (`TAU_ENERGY`).**
   - **Categories**: Shares categories directly with training `not_crop` (`hands`: 10, `feet_shoes`: 34, `pavement`: 38, `walls`: 17, `soil`: 43, `green_noncrop`: 32).

---

## 6. V3 Cross-Dataset Audit, Duplicate Scrapes, and Source Provenance

During the V3 expansion audit across 72,395 image hashes, cross-dataset perceptual hash deduplication was conducted using **dhash** (difference hash, 64-bit) with a Hamming distance threshold $\le 4$ and a 2.0% class-level duplication rejection threshold. This revealed significant scraping, uncredited re-uploading, and cross-dataset contamination in several candidate datasets:

- **Philippines Rice Diseases (Source 3)**:
  - `bacterial_leaf_streak`: 98/99 (99.0%) dhash duplicates of `paddy_doctor` -> entire dataset rejected for all 29 taxonomy classes.
  - `sheath_rot`: 45/91 (49.5%) duplicates of `Bacterialblight` from existing datasets -> dropped from openset.
  - `narrow_brown_spot`: 46/98 (46.9%) duplicates, mostly `original_leaffolder` -> dropped from openset.
  - `sheath_blight`: 7/98 (7.1%) duplicates across multiple training classes -> dropped from openset.
  - *Clean near-OOD classes retained*: `bakanae` (100 images, 0 duplicates), `rice_false_smut` (99 images, 0 duplicates), `ragged_stunt_virus` (100 images, 0 duplicates) passed with 0% overlap and are routed strictly to `data/raw/openset/`.
- **Rice Mendeley (Source 1, Mendeley `hx6f852hw4/2`)**:
  - `Rice Hispa`: 186/215 (86.5%) duplicates of `rice_hasan` -> excluded from training pool (`rice__hispa` stays at 2 verified sources).
  - `Healthy Rice Leaf`: 125/157 (79.6%) duplicates of `rice_hasan` -> excluded from training pool.
  - *Genuine classes accepted*: `Bacterial Leaf Blight` (180 images, 100% clean), `Brown Spot` (266 images, 1 duplicate removed), `Leaf Blast` (300 images, 5 duplicates removed).
- **Kushagra Wheat (Source 7)**:
  - `Mildew`: 454 duplicates vs `plantwild` (139) and `wheat_small` (322) -> 690 clean images remain; capped at exactly 600 unique images (seed 42) into `wheat__powdery_mildew`.
- **BanglaRiceLeaf (Source 4)**:
  - Authentic, verified source collected at Bangladesh Rice Research Institute (BRRI), Gazipur.
  - Correct DOI: Harvard Dataverse `doi:10.7910/DVN/XAOBYW` (resolving Elsevier PII mismatch).
  - 0% duplication against all existing project images. Capped at 600 images/class (seed 42) across BLS, BLB, Leaf Blast, and Normal (500). Sheath Blight (416 clean) routed to openset.
- **SugarcaneLD-BD (Source 8)**:
  - RedRot: 222 images ingested (2 duplicates removed). Breaks single-source lock on `sugarcane__red_rot` (1 -> 2 sources).
  - Healthy: 194 images at 224×224 ingested. Weak third source; documented as low resolution, but adds genuine field capture diversity.
  - EyeSpot (71), RingSpot (83), RedLeafSpot (43) routed to openset.

- **Known Held-Out Deviation (`rice__blast`)**:
  - `rice__blast` holds out `dhan_shomadhan` (255 images) as the smallest available source. Although `dhan_shomadhan` contains field-background images, blast retains 4 diverse sources in training (`paddy_doctor`, `rice_sethy`, `banglariceleaf`, `rice_mendeley` totaling 3,238 train images). Documented as an intentional governance deviation/limitation of the held-out set for `rice__blast`.

---

## 7. Model B (Sticky-Trap Pest Patch Classifier) Provenance & Governance

### 7.1 Dataset Catalog & Licences

| Dataset Name | Primary Identifier | Licence | Modality & Scope | Role in Model B |
|---|---|---|---|---|
| **Wageningen 4TU** | [doi:10.4121/uuid:8b8ba63a-1010-4de7-a7fb-6f9e3baf128e](https://doi.org/10.4121/uuid:8b8ba63a-1010-4de7-a7fb-6f9e3baf128e) | CC BY 4.0 | 284 high-res yellow cards (5184×3456); 5,591 whiteflies, 1,312 *Macrolophus*, 510 *Nesidiocoris*, 7 thrips (Pascal VOC XML). | Target `small_pale_winged` (WF) and `larger_insect` (MR/NC) crops, plus clean card `debris`. |
| **PST (Zenodo)** | [zenodo.org/records/7801239](https://zenodo.org/records/7801239) | CC BY 4.0 | 28 ultra-high-res yellow chromotropic cards (4288×2848); 17,005 point-annotated whitefly centroids (*B. tabaci*, *T. vaporariorum*). | Target `small_pale_winged` crops and clean glue `debris`. |
| **GinJinn2 (BGBM)** | [doi:10.34656/41pk-rn18.1](https://doi.org/10.34656/41pk-rn18.1) | CC0 1.0 | 120 yellow sticky cards (COCO format); 4,913 bounding boxes (Whitefly, *Macrolophus*, *Nesidiocoris*). | Verified 100% duplicate subset of Wageningen 4TU. Absorbed via 4TU to avoid double-counting. |
| **Ong & Høye (Figshare)** | [doi:10.6084/m9.figshare.23617383.v2](https://doi.org/10.6084/m9.figshare.23617383.v2) | CC BY 4.0 | 8,680 beetle images (*S. oryzae*, *T. castaneum*) across DSLR, Webcam, and Smartphone + 348 `Other_objects-samples`. | Non-target by-catch in `larger_insect` providing multi-device degradation exposure. `Other_objects` in `debris`. |

### 7.2 Perceptual Hash (dhash) Deduplication & Cross-Dataset Leakage Audit

1. **Cross-Dataset Contamination**:
   - Automated filename and metadata audit confirmed that **all 120 images in GinJinn2 are direct duplicates of Wageningen 4TU** (images 1000.jpg through 1119.jpg).
   - Ingestion was routed directly from the master Wageningen 4TU XML files to eliminate inter-dataset cross-leakage.
2. **Crop-Level Deduplication**:
   - 64-bit difference hashing (`imagehash.dhash(pil_patch, hash_size=8)`) was executed across all extracted crops with a strict Hamming distance threshold ($\le 4$).
   - Total raw crops extracted: 20,627 whitefly crops, 3,622 larger insect crops, 1,761 debris crops.
   - Total unique crops retained after deduplication: **7,590 crops** (`small_pale_winged`: 3,000; `larger_insect`: 3,000; `debris`: 1,590).
   - Split partitions:
     * `train`: 5,314 crops (70%)
     * `val`: 1,138 crops (15%)
     * `test_clean`: 948 crops (DSLR & high-res cards)
     * `test_degraded`: 190 crops (Webcam & Smartphone sensors)

### 7.3 Modeling Assumptions & ESP32-CAM Augmentation Pipeline

To bridge the substantial domain shift between high-resolution European greenhouse DSLRs and the in-field OV2640 sensor on the ESP32-CAM handheld/station node, training employs an aggressive physics-based degradation transform:
- **Footprint Downscaling**: Downscale-then-upscale (`scale_range=(0.25, 0.60)`, $p=0.6$) to simulate the true 6-pixel insect footprint at nominal trap distance.
- **Hardware JPEG Artefacts**: `ImageCompression(quality_range=(15, 60), p=0.7)` matching the low-bitrate DCT quantization of the OV2640 hardware JPEG engine.
- **Motion & Wind Smear**: `OneOf([MotionBlur(blur_limit=(3, 7)), Defocus(radius=(1, 3)), GaussianBlur(blur_limit=(3, 5))], p=0.5)`.
- **CMOS Sensor Noise**: `ISONoise(color_shift=(0.02, 0.08), intensity=(0.1, 0.4))` and `GaussNoise(std_range=(0.05, 0.20))` with white-balance hue jitter ($\le \pm 0.08$).
- **Glue Glare & Occlusion**: `CoarseDropout(num_holes=(1, 4), size=(4, 12), fill=255)` to simulate specular glue reflection and dust speckling.

### 7.4 Three Known Unfixable Gaps

1. **Sugarcane Woolly Aphid (*Ceratovacuna lanigera*) Wax Morphology**:
   - Ground reality: *C. lanigera* presents as white cottony flocculent wax secretions rather than discrete insect bodies.
   - Dataset status: Zero public sticky-trap datasets contain this morphology.
   - Expected behavior: Segmented as irregular dark/pale blobs by watershed; classified as `debris` or `small_pale_winged`.
   - Resolution: Requires physical trap image harvesting from infested Indian sugarcane fields.
2. **Soft-Bodied Aphids & Thrips (*Rhopalosiphum padi*, *Sitobion avenae*, *Anaphothrips obscurus*)**:
   - Ground reality: Soft-bodied, elongate, matte bodies with cornicles or fringed wings.
   - Dataset status: Unlabelled in public European trap datasets (only 7 thrips in 4TU; zero labelled aphids in WUR, PST, GinJinn2).
   - Expected behavior: Detected and counted in **total blob density** and daily interval rate by deterministic watershed (`core/trap_segmentation.py`), but unclassified into a dedicated pest taxon by Model B.
   - Operational impact: Wheat aphid and thrips rely on relative rate-of-change trend alerting (`TrapCountHistory`) on the total blob count, not on a quantitative per-species classifier threshold.
3. **Weathered Glue & High-Variance Field Debris Gap**:
   - Ground reality: Real Indian yellow sticky cards exposed to open agricultural fields accumulate dust films, pollen crust, water spots, dried foliage fibers, and fungal spores.
   - Dataset status: The v4 sanitization enforced strict flatness ($S \ge 95, B < 75, \sigma < 6.0$, zero blobs $\ge 6$ px) to prevent label contamination from unannotated insects. While clean, this debris class represents pristine uniform yellow adhesive, not weathered field cards.
   - Expected behavior: The trivial variance baseline ($\sigma$-only decision tree) separates debris from insects with a threshold of $\sigma \approx 2.87$, scoring 0.5594 / 0.5869 Macro-F1. Because the model relies heavily on low variance to identify debris, weathered non-insect textures with $\sigma > 4.0$ will likely be misclassified as insect classes, inflating pest counts.
   - Operational impact: In-field deployments must treat CNN pest proportions as secondary to deterministic watershed blob density and rely on trend alerting rather than absolute counts.

### 7.5 Model Performance & Honest Baseline Audit

> [!WARNING]
> **Mandatory Performance Caveat**:
> Reported accuracy measures benchmark performance on group-isolated trap cards and cross-source European benchmarks; it predicts nothing about Indian field recall.

> [!CAUTION]
> **INVALIDATION NOTICE (CARD-LEVEL LEAKAGE & CLASS-IMBALANCE AUDIT)**:
> The previously reported **76.79%** clean accuracy is **STRUCK AND INVALIDATED** due to card-level data leakage.
> Furthermore, the intermediate **97.11%** raw accuracy is **STRUCK AND INVALIDATED** because it was an artifact of severe class imbalance (85.6% whitefly in test). Raw accuracy is permanently disqualified as a headline metric for Model B.
> Primary evaluation metrics are **Macro-F1**, **Balanced Accuracy** (unweighted mean of per-class recalls), and per-class recall tables.

- **Trivial Standard Deviation ($\sigma$-Only) Baseline**:
  - Model: Depth-2 decision tree trained strictly on crop grayscale standard deviation ($\sigma$).
  - **In-Distribution Macro-F1**: **0.5594** (Balanced Acc: **58.51%**)
  - **Cross-Card Macro-F1**: **0.5869** (Balanced Acc: **66.06%**)
  - Debris Recall: **98.17%** (In-Dist) / **98.18%** (Cross-Card) using threshold $\sigma \approx 2.87$.
  - Insect Discrimination Recall: **0.00%** on `larger_insect` (both whiteflies and larger insects share the variance range $\sigma \in [5, 30]$).
  - **Empirical Interpretation**: Debris separation is trivialized by the mathematical flatness of the sanitization filter ($\sigma < 6.0$). A benchmark score of 0.9911 or 0.9735 reflects the combination of trivial debris separation and genuine CNN morphological discrimination between whiteflies and larger insects.

- **Disciplined In-Distribution Evaluation (Group-Isolated Cards & Specimens — 2,991 crops)**:
  - Dataset: PST + 200 Wageningen 4TU cards + Ong & Høye (`RPYellow` only). Zero card/specimen overlap.
  - **Macro-F1**: **0.9921**
  - **Balanced Accuracy**: **99.52%**
  - Per-Class Recall: `small_pale_winged`: **99.40%** (1,992 / 2,004), `larger_insect`: **99.52%** (416 / 418), `debris`: **99.65%** (567 / 569).
  - In-Distribution `small_pale_winged` $\to$ `larger_insect` error rate: **0.10%** (2 / 2,004) — **PASS ($\le 5.0\%$)**

- **Final Cross-Card Generalization Benchmark (84 Fully Held-Out Wageningen 4TU Cards — 2,107 crops) [HEADLINE]**:
  - Dataset: 84 complete Wageningen 4TU cards held out entirely from training and validation pools.
  - **Macro-F1 (HEADLINE)**: **0.9861** (vs. 0.5869 for $\sigma$-baseline)
  - **Balanced Accuracy**: **98.61%**
  - Confusion Matrix (rows = true, cols = predicted):
    * `small_pale_winged`: 934 true, 7 misclassified as `larger_insect`, 7 misclassified as `debris`
    * `larger_insect`: 13 misclassified as `small_pale_winged`, 541 true, 2 misclassified as `debris`
    * `debris`: 0 misclassified as `small_pale_winged`, 0 misclassified as `larger_insect`, 603 true
  - Per-Class Recall:
    * `small_pale_winged`: **98.52%** (934 / 948)
    * `larger_insect`: **97.30%** (541 / 556) — *the morphology-bearing number*
    * `debris`: **100.00%** (603 / 603)
  - Operational ETL Leakage (`small_pale_winged` $\to$ `larger_insect`): **0.74%** (7 / 948) — **PASS [ceiling 5.0%]**

- **Post-Hoc Calibration & Temperature Honesty Note**:
  - Fitted Temperature: $T_{cal} = \mathbf{0.9997}$ (fitted via LBFGS on validation negative log-likelihood).
  - **Honest Note on "Calibration"**: $T_{cal} = 0.9997$ indicates that temperature scaling found no adjustment worth making ($T \approx 1.0$). The raw logits from the MobileNetV3-Small backbone were already well-scaled on the group-isolated validation split. Do NOT describe the model as "calibrated" on that basis — the scalar simply indicates that empirical post-hoc temperature scaling left the network probabilities essentially unadjusted.

- **Numerically Stable Energy-Based Out-of-Distribution (OOD) Rejection (579 Pure Non-Target Samples)**:
  - Zero plain yellow glue tiles. Contains exclusively verified non-targets: thrips on yellow glue (`TH`), incidental non-whitefly insects, card printed fiducials/grid markers, and agricultural pest wax.
  - Numerically stable energy formulation: $E(x) = -T_{cal} \cdot \text{logsumexp}(z / T_{cal})$.
  - Fitted Energy Threshold: $\tau_{energy} = \mathbf{-3.8061}$ (at 95% in-distribution validation TPR).
  - OOD Rejection on hard non-targets: **80.31%** (465 / 579 rejected with energy $> \tau_{energy}$).
  - Confidence Floor: $\tau_{conf} = 0.60$. Blobs with $\max P(y|x) < \tau_{conf}$ or $E(x) > \tau_{energy}$ abstain and emit `UNCERTAIN_NON_TARGET`.

### 7.6 Advisory Wire Contract & Claim Discipline

All Model B telemetry surfaced to farmers or downstream edge storage enforces strict provenance tags:
```json
{
  "pest": {
    "total_blobs_counted": 142,
    "primary_count_source": "deterministic_watershed",
    "card_saturated": false,
    "monitoring_window_days": 3.0,
    "daily_rate": 47.33,
    "morphological_distribution": {
      "small_pale_winged": {"count": 104, "fraction": 0.7324},
      "larger_insect": {"count": 26, "fraction": 0.1831},
      "debris": {"count": 12, "fraction": 0.0845}
    },
    "verification_status": "RECALLED_UNVERIFIED",
    "classification_source": "CROSS_DOMAIN_PRETRAINED",
    "provenance_disclaimer": "Morphological proportions derived from European sticky-trap CNN. Total blob count is primary. Accuracy reflects in-distribution benchmark only and does not predict Indian field recall.",
    "known_gap_warning": "Sugarcane woolly aphid (C. lanigera) wax morphology and soft-bodied aphids/thrips are not represented in training classes. They are counted in total blobs by watershed, unclassified by the CNN."
  }
}
```

---

## 8. Training Crop vs. Edge 9-Tile Grid Domain Shift (Model A)

### 8.1 Architectural Discrepancy
- **Training Input Pipeline (`train/transforms.py`)**:
  - Training images are sourced from full-resolution smartphone captures (typically 3000×4000 to 3456×4624).
  - The pipeline applies `RandomCrop(height=320, width=320)` (or random resized scale `[0.7, 1.0]`) centered arbitrarily on leaf surfaces, followed by bilinear resize to $224 \times 224 \times 3$.
- **Edge Inference Pipeline (`edge/tiler.py`, `configs/train_config.py`)**:
  - The handheld pod camera captures frames at $1920 \times 1080$ (RGB).
  - The frame is partitioned into a **fixed $3 \times 3$ grid of 9 tiles** of size $320 \times 320$ with $20\%$ overlap (`TILE_GRID = 3`, `N_TILES = 9`, `TILE_OVERLAP = 0.20`).
  - Each of the 9 tiles is resized to $224 \times 224 \times 3$ and packed into an engine batch of 9 (`ENGINE_BATCH = 9`) for TensorRT inference.

### 8.2 Operational Domain Shift & Failure Modes
1. **Lesion Centering vs. Boundary Clipping**:
   - Training crops predominantly frame lesions centrally or near-centrally with surrounding leaf context.
   - Fixed spatial tiles regularly slice lesions along tile boundaries, presenting incomplete morphology (e.g. half a blast lesion or partial streak).
2. **Peripheral Non-Vegetation Ingestion**:
   - In fixed $3 \times 3$ tiling, edge and corner tiles frequently capture background clutter (soil, feet/shoes, horizon, sky, operator fingers) rather than pure foliage.
3. **Mitigations Implemented in Edge Pipeline**:
   - **Vegetation Fraction Gating**: `edge/tiler.py` filters each tile through an ExG mask (`TILE_MIN_VEG_FRACTION = 0.40`), discarding any tile containing $< 40\%$ green foliage before classification.
   - **Outlier Exposure Rejection**: The 29th class `not_crop` (trained on soil, hands, shoes, sky, pavement) explicitly catches peripheral background tiles.
   - **Spatial Aggregation**: `edge/storage.py` and `core/aggregate.py` enforce multi-tile consensus (`CELL_K = 2` out of `CELL_N = 3` agreeing tiles) before asserting a disease verdict.

### 8.3 Audited Performance Governance: Headline vs. Generalization Collapse
Every citation of Model A's in-distribution performance must accompany the cross-source reality:

| Evaluation Split | Total Images | Leaked Train Duplicates | Top-1 Accuracy | Macro-F1 | Empirical Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Distribution (`test_indist` Original)** | 3,281 | 1,248 (38.04%) | **96.56%** (96.04% ONNX) | **94.85%** (94.54% ONNX) | Optimistic; inflated by 38% near-duplicate leakage. |
| **In-Distribution (`test_indist` Cleaned)** | 2,033 | 0 (0.00%) | **95.08%** (-0.96 pts) | **94.08%** (-0.46 pts) | Genuine in-distribution morphology benchmark. |
| **Validation Set (`val` Original)** | 3,320 | 1,349 (40.63%) | **94.28%** | **91.32%** | Calibration split contaminated by 40% leakage. |
| **Validation Set (`val` Cleaned)** | 1,971 | 0 (0.00%) | **93.10%** | **90.99%** | Clean calibration: $T_{cal} = 0.6100$, $\tau_{energy} = -2.7424$. |
| **Cross-Source Held-Out (`test_sourceheldout`)** | 1,571 | 0 (Unseen cameras) | **32.08%** (33.93% ONNX) | **37.40%** (39.71% ONNX) | **CRITICAL COLLAPSE (-57.45 pts macro-F1)**. |

> [!WARNING]
> **57.45 Percentage-Point Generalization Collapse (Model A)**:
> Testing Model A on completely unseen cameras/geographic sources for the exact same 9 disease classes results in a catastrophic drop from **94.85%** to **37.40%** Macro-F1 (32.08% top-1).
> Three classes suffer near-total collapse due to background shortcut learning:
> - `rice__brown_spot`: **1.50%** recall (131/133 missed)
> - `rice__blast`: **7.45%** recall (236/255 missed)
> - `rice__bacterial_leaf_blight`: **11.67%** recall (159/180 missed)
> The headline figure of 94.8% / 96.5% measures memorization of source capture conditions, not robust crop leaf pathology.




