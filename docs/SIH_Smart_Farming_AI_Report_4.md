# Smart Farming Assistant — AI Architecture, Datasets, Pipeline & Edge Optimization
### Version 4 · Technical research report for SIH 2026

**Version:** 4.0 · 3 September 2026
**Supersedes:** `..._Report_3.md`, `_2.md`, `.md` — all kept for reference
**Companions:** `AI_Handbook_4.md` · `RED_TEAM_ADJUDICATION_3.md` · **`sih_pipeline_v4/` — runnable, tested code**
**Target hardware:** Jetson Nano 4GB (Maxwell GM20B, compute capability 5.3); contingency Orin Nano Super 8GB
**Deadline:** SIH portal submission 20–30 September 2026

> ### ⚠ The decision logic now lives in code, not in this document
> Three audit rounds found bugs in the *fixes* for the previous round. All decision logic is now in `sih_pipeline_v4/` with a passing test suite. **Import the modules; do not copy code from these pages.**

---

## Changelog: v3 → v4

Round 3 upheld 7 of 8 findings. Three were bugs in fixes v3 introduced for Round 2. A ninth was found by **executing** the code and appears in no audit.

| § | Change | Severity | Verified by running it |
|---|---|---|---|
| **1.2** | **Canopy gate widened `Ta+7` → `Ta+15`**, and it now accepts an optional RGB `veg_fraction` as the primary discriminator. | **High, silent** | v3 returned `(None,'no_transpiring_vegetation')` for a canopy at 45.5 °C with Ta = 38 — **suppressing exactly the drought it was deployed to detect** |
| **1.2** | **Vegetation mask: Otsu → absolute ExG threshold.** | **High, silent** | v3 Otsu reported **47.6% vegetation on a 100% green field**, failing every purity gate |
| **1.1** | **Nutrient scope split by platform.** Drone = relative only; rover/handheld = absolute LCC with a card. | Medium | v3 required a grey card in every aerial frame — physically impossible over a flooded paddy |
| **3.3** | **Trap markers: global `0.3·dist.max()` → dual threshold.** Plus hole filling and a label-loop fix. | **High** | v3 produced **1 marker of 6** with one moth on the board; v4 produces 6 of 6 |
| **3.3** | **Hole filling added** — `adaptiveThreshold` hollows insects larger than its block. | Medium | Found by execution, in no audit. Two bugs were mutually masking. |
| **5.2** | Stage 4/5 updated: confidence on unadjusted probabilities; explicit cell states. | High | v3 conflated healthy and never-visited cells |
| **9** | Risk register updated. | | |
| **12** | Test suite shipped as runnable code. | | 15 tests pass; 6 fail against v3 |

### Earlier changelogs (retained)
**v2 → v3:** BGR/RGB channel swap; `aggregate_frame` had no HEALTHY path; energy calibrated on trained `not_crop`; inverted thermal spread gate; PyCUDA teardown.
**v1 → v2:** AWB/AE lock; soil-heat-bleed identified; gateway pest inference; spatial/temporal aggregation separated; zero-copy rejected.

---

## 0. Executive summary — the findings that drive the design

### 0.1 — Your Jetson Nano cannot run INT8. At all.
The Maxwell GM20B is derived from the Jetson TX1 SoC and has no INT8 hardware path (no DP4A, no tensor cores); NVIDIA staff confirm this directly. Every "quantize to INT8 for 4× speedup" tutorial is inapplicable to you. **FP16 is your floor.** Independent Nano benchmarks show INT8 either matching FP16 in speed or being worse, with degraded mAP. Model *architecture* and *input resolution* must do all the optimization work instead.

### 0.2 — Your software stack is frozen in 2021.
Maximum JetPack 4.6.4 → Ubuntu 18.04, **Python 3.6**, CUDA 10.2, TensorRT 8.2. PyTorch 1.11+ needs Python 3.7; PyTorch 2.x needs CUDA 11, impossible on Maxwell. There is no JetPack 5 or 6 for this board and never will be. Python 3.6 also means several standard-library modules people assume you have — `multiprocessing.shared_memory` among them — simply do not exist. **Train on Colab, deploy TensorRT engines, never run PyTorch on the Nano.**

### 0.3 — Your previous YOLO failure was a data problem, and the mechanism is documented.
PlantVillage — which underpins most "99% accuracy" papers — is severely biased. Noyan (2022) trained a model on **8 background pixels alone** and achieved **49.0% accuracy across 38 classes** where chance is 2.6%. Cross-domain studies report lab→field collapse to **33.27%**, and PlantVillage-trained models scoring **45.95%** and **33.97%** on two independent field sources. You trained a competent model on a dataset that taught it to recognise photo studios. **Fixing this matters far more than which architecture you pick.**

### 0.4 — Thermal cameras cannot find insects. They find thirsty plants.
Insects are ectothermic and millimetres across — at any useful standoff they sit at ambient leaf temperature. Thermal's real job is **canopy temperature as a water-stress proxy** via the Crop Water Stress Index (Idso et al., 1981). **But see §1.2 — the naive version of this does not work either, and v1 got it wrong.**

### 0.5 — Nobody in production finds insects by flying a drone at leaves.
Commercial pest monitoring (Trapview, iSCOUT, Z-Trap) uses **static camera traps** at fixed distance. Recent work achieves species-level thrips/whitefly detection at 79–89% mAP@50, with YOLO11n reaching 80%, and identifies 80 µm/pixel as the minimum viable resolution. That works because background is uniform, scale is fixed, lighting is controlled, and there is no domain shift. None of that holds for a drone hunting aphids.

### 0.6 — The drone is defensible as a two-tier sensing platform, not a fast rover.
A granted UAV agronomy patent describes the architecture physics forces on you: a **survey pass** capturing vegetation indices, then **return and fly low (≤20 ft) or hover** over flagged areas for high-resolution capture. The drone's advantage is *targeted revisit over terrain a rover can't cross*.

Two physical constraints: at 1 m above canopy, rotor downwash creates a disturbed ellipse of roughly 0.9 × 0.8 m directly beneath, and CFD work recommends a **~0.6 m sensor boom** to shoot outside your own wash. And "slowly, one plant at a time" means your inference requirement is **1–3 FPS, not 30**.

### 0.7 — The largest deployed system in India is a cloud image classifier.
Plantix — ~10 million farmers annually, 135 million downloads, built with ICRISAT — is classification (not detection), close-up (not aerial), cloud (not edge), trained on 120M+ real field images, claiming >90% accuracy *on field images specifically*. Copy three of those four. The fourth is what your problem statement asks you to invert, so be ready to articulate *why* edge matters: connectivity, latency, per-query cost, data ownership.

---

## 1. Task allocation — what needs AI, and what emphatically does not

Roughly half the problem statement is solved better by deterministic code. Judges reward this when framed as engineering judgement.

| # | Requirement | Approach | Why |
|---|---|---|---|
| 1a | Detect crop diseases from images | **AI — image classifier** | Genuine perceptual task. Flagship model. |
| 1b | Nutrient deficiencies via colour/texture | **Script (hardened in v2)** | Nitrogen maps to leaf colour through a validated linear relationship — §1.1 |
| 1c | Growth stages / field health | **Script** | Vegetation-index time series + canopy cover. Curve-fitting. |
| 2a | Detect insect pests | **AI — small classifier on trap node** | Needed, but on a static trap, not the drone. §3.3 |
| 2b | Early alerts before spread | **Script** | Threshold on trap counts vs. economic threshold levels + trend slope |
| 2c | Targeted vs. blanket spraying | **Script** | Spatial aggregation → prescription map |
| 3a | Monitor soil moisture, temp, humidity | **Script** | Sensor reads |
| 3b | Detect water stress / over-irrigation | **Script (rewritten in v2)** | CWSI — but only with canopy purity handling. §1.2 |
| 3c | Irrigation schedules | **Script — FAO-56 water balance** | Deterministic soil-water model |
| 4 | Environmental risk | **Script — agro-met rules** | Degree-days, humidity-hours, rainfall accumulation. §1.3 |
| 5 | Edge AI processing | **Infrastructure** | §5, §6 |
| 6 | Farmer advisory | **Rules → LLM API for phrasing only** | Your plan is correct. §5.4 |
| 7 | Analytics dashboard | **Script + classical forecasting** | Pandas + linear/ARIMA trend. Do not deep-learn this. |
| 8 | Scalable deployment | **Architecture** | §5.5 |

**Net: two trained models, plus one optional third.** Everything else is Python.

### 1.1 Nutrient deficiency — script, but harder than v1 implied ⚠ **UPDATED**

The Leaf Colour Chart (LCC), from IRRI and PhilRice, is already used by Indian extension workers to time nitrogen application; ICAR/state-university trials in West Bengal showed LCC-guided dosing saved 20–42.5 kg N/ha with no yield penalty. The underlying signal is well-characterised: across six rice cultivars under natural light, the **CIELAB b\*** index showed the closest *linear* relationship with both SPAD readings and leaf nitrogen concentration, out of 13 colour indices tested across RGB, HSV and Lab.

**So the method is sound. The problem v1 understated is that colour measurement in a field is unstable.**

An IMX219's auto-white-balance and auto-exposure continuously re-gain the image based on whatever is in frame. Direct sun, cloud shadow and low-angle light differ substantially in colour temperature. Uncorrected, illumination-driven drift in the b\* channel can be comparable to or larger than the entire difference between adjacent LCC panels — which makes an absolute panel assignment meaningless.

**Mandatory mitigations, cheapest first:**

1. **Lock AWB and AE to fixed gains** in the camera driver. Free, and non-negotiable. A camera that re-balances every frame cannot do colourimetry.
2. **Reference card — but only where a card can physically exist** ⚠ **SCOPED IN v4**

   > **What was wrong in v3.** It required "a grey/white reference card in every frame," including aerial frames. **You cannot hold a card in frame while a drone flies over a flooded paddy**, and no card-detection step was ever specified. That was a requirement that cannot be satisfied in the deployment mode the same document designed.

   Scope it by platform instead:

   | Platform | Nutrient output | Why it works |
   |---|---|---|
   | **Drone (aerial)** | **Relative only** — "this cell is greener/yellower than the field median from the same pass" | Illumination cancels within a pass. Still actionable: it tells the farmer *where* to apply nitrogen, which is the decision that matters. |
   | **Rover / handheld phone** | **Absolute LCC panel (1–4)** with a card in frame | Close range, card trivially placed, standard LCC protocol applies |

   Optional middle path: stake a grey card at canopy height at a known GPS point, fly over it at the start and end of each pass, and interpolate gains across the pass. Cheap, and it gives you a real answer if a judge presses.

   ```python
   def calibrated_leaf_b_star(leaf_roi_bgr, card_patch_bgr):
       """
       Von Kries gains from a neutral card, applied ONLY to the leaf ROI
       (~200x200 px). ROVER / HANDHELD PATH ONLY — see the table above.

       v3 applied gains to the whole 1080p frame: ~25 MB per float32 copy for
       no benefit. White balance is needed for the nutrient index, not for the
       classifier — the classifier is trained on UN-white-balanced images with
       heavy colour augmentation, so corrected input is a train/test mismatch.
       """
       card_mean = card_patch_bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)
       gains = card_mean.mean() / np.maximum(card_mean, 1e-6)
       leaf = np.clip(leaf_roi_bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
       lab = cv2.cvtColor(leaf, cv2.COLOR_BGR2LAB)
       return float(np.median(lab[:, :, 2]))
   ```
   **⚠ On the returned value.** OpenCV's `COLOR_BGR2LAB` on `uint8` returns `a`/`b` in [0,255] with a **+128 offset**, so this is `b* + 128`, uniformly scaled — not CIELAB `b*` as published. Fine as a relative index; do not compare to literature values without converting.

   Locking AWB/AE is still correct — it makes the sensor consistent — but it does **not** fix changing illumination, and v3 implied it did. Say so on the slide: absolute colourimetry from a moving aerial platform under variable cloud is not something you claim.

3. **Restrict readings to a fixed solar window.** The LCC field protocol already requires consistent time of day; honour it.
4. **Prefer relative to absolute.** Report *"this cell is greener than the field median"* rather than *"this is LCC panel 2."* Relative comparison within a single frame cancels most illumination error and is still agronomically actionable — it tells the farmer *where* to apply nitrogen, which is the decision that matters.

**Honest caveats for the PPT:** the b\*–nitrogen relationship is phenology-dependent (it shifts with growth stage), and LCC readings are sensitive to time of day and direct sun. Naming these before a judge finds them is worth more than hiding them.

**Validation:** the Mendeley *Nitrogen Deficiency of Rice Crop* dataset (four LCC-matched classes) is the right set to check your script against.

### 1.1.2 Field Vegetation Indices, Canopy Masking, and Dual-Bandpass NDVI ⚠ **IMPLEMENTED IN v4**

To support aerial crop vigour mapping, canopy growth staging, and candidate stress flagging without an expensive commercial multispectral camera, six RGB-only vegetation indices and a dormant dual-bandpass NDVI pipeline are implemented in `core/indices.py` and `core/ndvi.py`.

#### Implemented RGB Vegetation Indices

| Index | Formula | Spectral Bands | Target Biophysical Property | Known Failure Mode |
|---|---|---|---|---|
| **VARI** (Visible Atmospherically Resistant Index) | $\frac{G - R}{G + R - B}$ | Green (520–560 nm), Red (620–660 nm), Blue (450–490 nm) [Gitelson et al. 2002] | Green vegetation fraction, crop vigour; designed to reduce atmospheric scattering effects in low-altitude aerial surveys. | **Atmospheric haze & blue noise**: Highly sensitive to blue-channel sensor noise in low light, and the denominator $(G + R - B)$ approaches zero on dark moist soil or shadows, producing extreme noise and division-by-zero instability. |
| **TGI** (Triangular Greenness Index) | $G - 0.39 R - 0.61 B$ | Green, Red, Blue [Hunt et al. 2011] | Leaf chlorophyll concentration at canopy scale; measures the area of the triangular spectral curve between blue, green, and red reflectance peaks. | **High-nitrogen saturation**: Green reflectance reaches an asymptotic saturation plateau at moderate-to-high chlorophyll concentrations, so TGI cannot resolve luxury nitrogen uptake; also highly sensitive to bright soil background reflectance in sparse canopies. |
| **NGRDI** (Normalized Green-Red Difference Index) | $\frac{G - R}{G + R}$ | Green, Red [Tucker 1979] | Relative green biomass, phenology tracking (greening vs senescence). | **Low dynamic range & soil confusion**: Narrow dynamic range compared to NIR indices; cannot distinguish yellowing senescent leaves from dry sandy/loam soil, and saturates at canopy closure (LAI > 2–3). |
| **GMR** (Green-Minus-Red Difference) | $G - R$ | Green, Red | Raw green contrast over red reflectance. | **Unnormalized illumination sensitivity**: Completely unnormalized by total intensity; readings shift dramatically with cloud passage, diurnal solar angle, and camera auto-exposure. Requires rigidly locked AWB/AE. |
| **DGCI** (Dark Green Colour Index) | $\frac{\frac{H - 60}{60} + (1 - S) + (1 - V)}{3}$ | Full RGB converted to float32 HSV [Karcher & Richardson 2003] | Nitrogen status and dark green leaf colour; correlates linearly with SPAD chlorophyll meter readings. | **Leaf glint & specular reflection**: Sun glint on waxy leaves (common in rice and sugarcane) washes out saturation $S$ and inflates value $V$, corrupting DGCI; sensitive to diurnal sun angle changes. |
| **ExG** (Excess Green) | $2G - R - B$ (clipped to $[0, 255]$ uint8) | Green, Red, Blue [Woebbecke et al. 1995] | Binary segmentation of green plant tissue from bare soil, shadows, and residue. | **Non-green crop failure**: Fails completely on senescent/ripening crops, non-green cultivars (e.g. purple rice), or under heavy shadow where $2G \le R + B$. |

#### Canopy Masking Strategy Prior to Index Computation

Every vegetation index in `core/indices.py` is calculated **strictly over segmented canopy pixels**, never across raw scene averages containing soil.
1. **Absolute ExG Thresholding (`PROVISIONAL_EXG_VEG_THRESHOLD = 20`):**
   - We explicitly reject Otsu thresholding for vegetation masking. On a mature, closed canopy, the ExG distribution is unimodal; Otsu's algorithm maximises inter-class variance by forcing a split near the distribution mean. On a 100% green canopy, Otsu falsely discards ~50% of the greenest leaves as "background".
   - Instead, an absolute threshold `ExG > PROVISIONAL_EXG_VEG_THRESHOLD (20)` is used to isolate plant tissue.
   - *Provisional status:* The default value of 20 is an uncalibrated heuristic tuned against synthetic and field tiles; it is marked `PROVISIONAL` and requires empirical calibration against local bare soil and shadow spectra in the deployment region.
2. **Vegetation Fraction Gate (`PROVISIONAL_MIN_CANOPY_FRACTION = 0.15`):**
   - An index is only computed if the canopy fraction across the ROI or cell is at least 15% ($\ge 0.15$). If $f_{\text{canopy}} < 0.15$ or if zero canopy pixels are detected, the function returns `(None, 'insufficient_canopy')`. This prevents background soil noise from generating pseudo-vigour alerts.

#### Dormant Dual-Bandpass NDVI Path

In addition to RGB proxies, the project implements a committed hardware path for true near-infrared NDVI using a single sensor:
- **Sensor & Optics:** Waveshare IMX219-77IR (NoIR, 79.3° FOV) fitted with a MidOpt DB660/850 dual-bandpass interference filter (passbands at 660 nm visible red and 850 nm near-infrared).
- **Counterintuitive Channel Assignment:** Silicon Bayer micro-filters leak NIR across all channels. Under the DB660/850 filter:
  - The Blue Bayer filter blocks 660 nm red but transmits 850 nm NIR.
  - The Red Bayer filter transmits 660 nm red and leaks 850 nm NIR.
  - Consequently, **Blue channel = NIR (850 nm)**, and **Red channel = Visible Red (660 nm)**. NDVI is computed as $(B - R) / (B + R)$.
- **Silicon Cross-Talk Gate (`apply_channel_response_correction`):**
  - Because organic Bayer dyes leak NIR into the Red channel, raw digital numbers represent mixed radiances: $DN_{\text{Red}} = k_{RR} L_{660} + k_{RNIR} L_{850}$.
  - True NDVI requires unmixing via an empirical $2 \times 2$ inverse response matrix $K^{-1}$ measured on an optical bench using reference 660 nm and 850 nm narrow-band sources (`scripts/calibrate_dual_bandpass.py`).
  - **The code is fully implemented and tested, but `apply_channel_response_correction()` raises `NotImplementedError` by default if `calib_matrix is None`.**
  - *Critical framing:* This is a **hardware bench-calibration gate**, not a software limitation. Computing NDVI from raw Bayer channels without measuring the physical sensor's cross-talk produces scientifically invalid, ungrounded numbers. The software refuses to guess.
- **Empirical Line Method (ELM) Across Flights:**
  - To normalise changing solar irradiance across flight passes, reference reflectance panels (nominal 5%, 50%, 84%) must be imaged pre- and post-flight (`apply_empirical_line_calibration`).
  - In `core/ndvi.py`, `PROVISIONAL_PANEL_REFLECTANCES = (0.05, 0.50, 0.84)` are placeholder values and `PANEL_REFLECTANCES_CONFIRMED = False`. Calling ELM without confirmed panels raises a `RuntimeError` unless `--allow-provisional` is explicitly passed.

### 1.2 Irrigation and water stress ⚠ **REWRITTEN IN v2**

> **What changed and why.** Version 1 specified an MLX90640 on the drone feeding a CWSI calculation. The physics does not support that at survey altitude, and the resulting index would have been systematically wrong in a way that looks plausible — the worst kind of error.

#### 1.2.1 The equation (unchanged, still correct)

The empirical CWSI (Idso et al., 1981):

```
CWSI = ((Tc − Ta) − (Tc − Ta)_LL) / ((Tc − Ta)_UL − (Tc − Ta)_LL)
```

where `Tc` = canopy temperature, `Ta` = air temperature, and the lower (well-watered) baseline is a crop-specific linear function of vapour pressure deficit, `(Tc − Ta)_LL = a·VPD + b`. CWSI = 0 is fully transpiring, 1 is fully stressed. Published sunflower and mung bean studies use exactly this form with a handheld IR thermometer; the mung bean work derived `(Tc − Ta)_LL = −3.9283·VPD + 2.1424` (R² = 0.87) and used pre-irrigation CWSI ≈ 0.30 as the irrigation trigger.

Everything needed is arithmetic: canopy temperature from the thermal sensor, air temperature and RH from a DHT22/SHT31 (VPD derives from those two), and a published crop baseline. **No AI.**

#### 1.2.2 The problem: `Tc` is not what the sensor measures

An MLX90640-D55 is a 32 × 24 array with a 55° horizontal field of view. At altitude *H* the horizontal ground footprint is `2·H·tan(27.5°)`, spread over 32 pixels:

| Altitude | Ground footprint | **Ground sampling distance** |
|---|---|---|
| 10 m (survey) | 10.4 m | **32.5 cm / pixel** |
| 5 m | 5.2 m | 16.3 cm / pixel |
| 2 m (inspection) | 2.1 m | **6.5 cm / pixel** |
| 1 m | 1.0 m | 3.3 cm / pixel |

A single 32.5 cm pixel spans both foliage and bare soil. In Indian midday conditions dry soil reaches roughly 55–65 °C while a transpiring crop sits at 28–32 °C. The sensor returns a radiometric average, so a pixel that is 70% canopy and 30% soil reads far hotter than the canopy actually is — and CWSI computed from that reports **severe drought stress on a fully hydrated plant.**

This is not noise. It is a systematic bias that always points the same direction, which means it will not average out and it will look like a real signal.

#### 1.2.3 Three fixes, cheapest first ⚠ **GATES CORRECTED IN v4**

**Fix 1 — measure CWSI at inspection altitude, not survey altitude.** Costs nothing but a scheduling decision. Drops GSD from 32.5 cm to ~6.5 cm. Do this regardless.

**Fix 2 — separate the canopy population, and gate on the right physics.**

> **⚠ Two generations of bug in this one function. Both now fixed and tested.**
>
> **v2** rejected frames with `P95 − P5 > 25 °C`, reasoning that a large spread meant low canopy fraction. Backwards: a large spread means *both* populations are present, which is normal. Valid mixed frames were rejected; uniform hot soil passed and returned ~57 °C as "canopy temperature."
>
> **v3** replaced that with a `Ta + 7 °C` ceiling — and introduced a worse failure. Running it: a canopy at 45.5 °C with Ta = 38 returns `(None, 'no_transpiring_vegetation')`. **A crop in severe water stress was discarded as bare soil.** The one condition the sensor exists to detect was the one it silently dropped.
>
> The v3 error was conceptual. I conflated the **CWSI upper baseline** `(Tc−Ta)_UL` — where CWSI = 1.0, the maximum stress the *index* measures, derived per crop from energy balance and dependent on radiation and wind — with the **physical limit of plant tissue**. A gate separating vegetation from soil must sit well above the upper baseline, not at it.

The separation you actually need is coarse, so the fix does not depend on any contested number: with Ta = 38, dry Indian midday soil at 55–65 °C is **Ta+17 to Ta+27**, while even a fully non-transpiring canopy sits well below that. Anywhere in Ta+12 to Ta+15 separates them with margin on both sides.

```python
from thermal import canopy_temperature       # sih_pipeline_v4/thermal.py

tc, info = canopy_temperature(thermal_frame, air_temp_c=ta,
                              veg_fraction=veg_frac)   # veg_frac optional
# -> (canopy_temp_c, canopy_fraction)  on success
# -> (None, 'no_vegetation_bare_soil' | 'implausibly_cold'
#          | 'canopy_fraction_too_low' | 'too_few_pixels')
```

Three properties that matter:

- **The gate separates plant tissue from bare soil, never healthy from stressed.** A non-transpiring canopy is a valid and important reading — it *is* CWSI = 1.0, the drought alarm. It must never be discarded here.
- **When a co-registered RGB frame is available, pass `veg_fraction`.** *Is there vegetation here* is a question the RGB channel answers directly; temperature then only has to catch sensor faults.
- **Named rejection reasons.** "We rejected 40% of frames because canopy fraction was too low" is diagnosable and belongs in a log and on a slide. `None` is not.

Verified in `tests/test_pipeline.py`:
```
stressed canopy 44.0 / 45.5 / 47.0 / 48.0 degC at Ta=38  -> ACCEPTED
bare soil       56.0 / 60.0 / 65.0        degC at Ta=38  -> REJECTED
```

**Fix 3 — RGB–thermal co-registration with vegetation masking** (optional; requires real calibration work between two sensors with different fields of view).

> **⚠ v4 fix — the vegetation mask no longer uses Otsu.**
> Otsu maximises between-class variance and **assumes bimodality**. On a mature closed canopy the ExG histogram is unimodal, so Otsu splits it near the mean: running v3 on a 100% green synthetic field reported **47.6% vegetation**, failing the `>0.85` purity gate on every healthy field you have.

```python
from thermal import vegetation_mask          # absolute ExG threshold

mask, veg_fraction = vegetation_mask(rgb_bgr)     # ExG > 20 on the 0..255 scale
```

Verified: 99.9% on a pure canopy, 0.0% on a soil frame. Note the thermal function still uses Otsu on the **thermal** histogram, which is legitimate — a mixed canopy/soil thermal frame genuinely is bimodal, and `bimodal_gap` falls back to single-population handling when it is not.

Note also that ExG can be **negative**; clip to `[0, 255]` before any `uint8` cast, or negatives wrap into large positives and silently invert the mask.

**Sequencing:** Fix 1 and Fix 2 today. Fix 3 only with half a day spare — and keep Fix 2's gates in place either way, since they catch sensor faults that co-registration cannot.

#### 1.2.4 Honest caveats for the PPT

- CWSI is only valid in the early-afternoon window under clear skies (most studies use ~11:00–16:00).
- Baselines are crop- and region-specific. State that you use published baselines and would calibrate locally in deployment.
- **State the GSD limitation explicitly.** "We compute canopy temperature from the coolest quartile of the thermal field, because at our sensor's ground sampling distance a naive frame average would be contaminated by soil at 55 °C" is one of the strongest sentences you can say to an agricultural judge. It shows you understand your instrument.

### 1.3 Environmental risk — script

Drought, flood, heat stress and disease-favourable conditions are threshold-and-accumulation problems over time series:

- **Heat stress:** consecutive hours above a crop-specific critical temperature (rice spikelet sterility risk rises sharply above ~35 °C at anthesis).
- **Flood risk:** rainfall accumulation over 24/72 h vs. local drainage, plus soil-moisture saturation persistence.
- **Drought:** cumulative water-balance deficit + CWSI trend.
- **Disease-favourable conditions:** classical agro-meteorological rules — leaf wetness duration × temperature windows. This is how commercial disease-warning services work.

A rules engine over a time-series buffer. Doing it in code rather than with a model also makes your alerts *explainable*, which matters for farmer trust and for the "why should we believe your system" question.

---

## 2. The YOLO question — a direct answer

**Your diagnosis of the symptom was right, your diagnosis of the cause was probably wrong, and you should still not use YOLO for the disease task — but for a different reason than you think.**

**Why YOLO was probably not your problem.** The failure mode you describe — works in testing, falls apart in the field — is the exact signature of the PlantVillage domain gap in §0.3. A model that learned background statistics fails identically whether it is YOLOv8, a ResNet, or a ViT. Note also the review finding that a Swin Transformer reached 88% on real-world data where traditional CNNs got 53% — that gap is real, but it is a gap in *robustness to domain shift*, which is substantially a data and training-recipe property.

**Why you should still not use YOLO for disease.** Three reasons, none of which is "YOLO is bad":

1. **Your task is classification, not detection.** Chlorosis, yellowing, mosaic patterning and general blight have no well-defined bounding box. Two annotators will disagree by 40%.
2. **Your data is classification-labelled.** Paddy Doctor and the Maharashtra sugarcane sets ship **class labels, not boxes.** Detection means annotating ~16,000 images in two weeks. You cannot.
3. **Detection heads cost Nano budget you don't have.** A detection head plus NMS is meaningful overhead on a device with no INT8 and 4GB shared RAM.

**Where a detector *is* correct: the sticky-trap node.** Countable discrete objects, uniform background, fixed scale, no domain shift. Published results show YOLO11n at 80% mAP@50 / 77% F1 for species-level thrips and whiteflies. If you want to avoid YOLO entirely even here, the automated-trap literature documents a second approach — **segmentation and classification separated**, i.e. classical blob detection followed by a tiny CNN per blob. **That is what §3.3 recommends:** cheaper, easier to debug, and it sidesteps your YOLO history.

**How to replace detection for drone imagery: tile-and-classify.** Slice each frame into overlapping tiles, classify each, aggregate. This is the SAHI insight — a small object that becomes a few pixels when a 4K frame is resized to 640² becomes large and detectable in a native-resolution crop. SAHI raises AP by 6.8/5.1/5.3 points on VisDrone and xView, rising to 12.7–14.5 with slicing-aided *fine-tuning*. Apply it with a classifier and you get localization for free without box annotation.

**⚠ v2 addition — how you aggregate those tiles matters enormously.** See §5.2 Stage 5. Version 1 of this report specified an aggregation rule that would have masked exactly the early-stage lesions you most want to catch.

---

## 3. Model recommendations

*(Summary. Full specification, code and training recipes are in `AI_Handbook_2.md`.)*

### 3.1 Primary — crop disease classifier

**Start with `tf_efficientnet_lite0`** (4.7M params, 0.4 GMACs). Purpose-built for edge: squeeze-and-excite removed, swish replaced with ReLU6, so it converts to TensorRT 8.2 cleanly. The binding constraint is not accuracy but conversion survival.

**Stretch: EdgeNeXt-XS / EdgeNeXt-S** — the one modern CNN-transformer hybrid with published latency measured on an actual Jetson Nano in FP16 TensorRT. EdgeNeXt-XXS reaches 71.2% ImageNet top-1 at 1.3M params (+2.2% over comparable MobileViT); EdgeNeXt-S reaches 79.4% at 5.6M. Flagged as medium export risk — its 4D LayerNorm and split-transposed-attention blocks are the most likely thing to trip TRT 8.2.

**Expect realistic field numbers.** A MobileNetV3 + CBAM + field-conditioned augmentation model reported **82.94%** on a hard 10-class real-world field set. That is what honest field performance looks like. Quoting realistic figures earns more credibility than claiming 99.6%.

**Three-layer rejection stack (expanded in v2):** trained `not_crop` class + energy-based OOD score + temperature-calibrated abstention. Details in Handbook 2 §8.5.

**Training strategy:** consistent-teaching knowledge distillation from a DINOv2 teacher, run **online** so teacher and student see identical augmented views. Handbook 2 §7.4 explains why the offline-caching version in v1 was wrong.

### 3.2 Optional — disease severity segmentation

Lightweight DeepLabV3-MobileNet or U-Net over PlantSeg turns "disease present" into "12% of leaf area affected", which is what drives spray decisions. Note the PlantSeg authors found "traditional CNNs like MobileNetV3 perform poorly in semantic segmentation due to limited context modelling" — budget it as a stretch goal, not a core claim.

### 3.3 Second model — pest detection on the trap node ⚠ **UPDATED IN v3**

**Architecture: classical blob segmentation + tiny CNN classifier, running on a gateway.**

> **What changed.** v1 left the inference location open ("ESP32-CAM node or Nano"). v2 makes the gateway the default.

```
[ESP32-CAM]  --JPEG on a timer-->  Wi-Fi / ESP-NOW  -->  [Jetson or laptop]
 deep-sleep between captures                              blob detect + classify here
```

**Why:** trap monitoring is a once- or twice-daily task, so latency is irrelevant; this is what commercial trap stations do; and it eliminates an entire problem class rather than optimizing around it. It also saves you the TFLite-Micro toolchain and INT8 calibration work — days you do not have.

**Why on-device on a plain ESP32-CAM is a trap.** The ESP32-CAM's Xtensa LX6 has no vector instructions. Espressif's ESP-NN documentation states that assembly/SIMD-optimized kernels are provided for **ESP32-S3, ESP32-P4 and ESP32-S31**, while **ESP32 and ESP32-C3 get only "generic optimisations."** The impressive INT8 speedups people quote belong to the S3's 128-bit SIMD. INT8 on a plain ESP32 still gives ~4× memory reduction and 2–4× time, but not an order of magnitude.

**If on-device inference is a differentiator you want:** buy an **ESP32-S3-CAM** (~₹800–1,200) rather than spending days optimizing an LX6.

**Pipeline on the gateway ⚠ UPDATED IN v3 — watershed splitting added:**

> **What was wrong in v2.** Plain connected components merges touching insects into one blob. Trap catches are spatially clumped, so under a real infestation a dozen adjacent whiteflies become one large component — which the area filter then either discards as debris or counts as a single insect. The failure is **anti-correlated with when accuracy matters**: counts degrade worst exactly when the Economic Threshold Level is being crossed.

Distance-transform watershed is the established remedy in the insect-trap literature specifically. A sticky-trap detection study applies threshold segmentation "combined with watershed theory" precisely to separate multiple insects within one region, and a moth-counting study reports **82% of touching insects correctly detected as touching** and separated after contour dilation, region merging and watershed.

- Downscale to ~1280 px long side (you do not need 5MP to localize blobs)
- **Adaptive threshold on the LAB `b` channel**, not a hardcoded HSV range — yellow traps fade under UV within days and specular glints off polybutene glue fragment a fixed mask
- Morphological open to remove glue glints
- **Distance transform → watershed → connected components** to split touching bodies
- Pad-and-crop 64×64 around each centroid (**pad at borders** — do not silently drop edge insects, which under-counts trap margins)
- Tiny CNN → classes = target pests + `not_pest` (dust, debris, beneficials)
- Counts per class per day → Economic Threshold Levels → trend slope for "infestation increasing"

```python
from trap_segmentation import segment_trap_blobs    # sih_pipeline_v4/

blobs = segment_trap_blobs(bgr, min_area=8, max_area=1200,
                           abs_floor_px=1.8)   # set from YOUR camera geometry
# -> [(crop_64x64_bgr, (cx, cy), area_px), ...]
```

> **⚠ v4 fix 1 — the v3 marker threshold erased every micro-pest.**
> v3 used `cv2.threshold(dist, 0.3 * dist.max(), ...)`. `dist.max()` is a **global** statistic: one 40 px-radius moth sets the seed threshold to ~12 px, while a 1–1.5 mm whitefly or thrip peaks at 2–4 px in the distance transform. Every micro-pest lost its marker and `cv2.watershed` flooded it as background.
>
> Running it on a synthetic trap with one moth and five whiteflies: **v3 produced 1 marker of 6. v4 produces 6 of 6.**
>
> This is the standard OpenCV watershed recipe, and it is only valid when objects are **similar in size** — the canonical tutorial uses identical coins. A sticky trap is the opposite case, and it fails hardest for exactly the pests yellow traps exist to catch.
>
> The absolute floor is not a tuning knob. **The trap node has fixed camera-to-trap geometry**, so mm/pixel is a known constant — derive `abs_floor_px` from the smallest target pest's radius at your setup, once.

> **⚠ v4 fix 2 — found by executing the code, present in no audit.**
> `adaptiveThreshold` with `blockSize=25` responds only near the **edges** of regions larger than the block, so a large insect segments as a **hollow ring** and its interior is never foreground. Large bodies fragment — and, perversely, the hollowing kept `dist.max()` at 3.0 in the first test, *masking* fix 1 until the blob was solidified. Two bugs partially cancelling, which document review cannot find and a five-line test can.
>
> The mask is now `adaptive OR global-Otsu`, morphologically closed, with interior holes flood-filled. Verified: 6 blobs, 5/5 micro-pests, large body 4,976 px solid.

**Priority: medium.** This runs on the gateway where compute is free, and the trap node is a secondary feature. Do it after the Tier A fixes.

Resolution requirement: 80 µm/pixel minimum for species-level ID, per the thrips/whitefly study — a 5MP sensor ~15 cm from an A5 trap clears this comfortably.

### 3.4 What you are explicitly *not* building

State these as scope decisions with reasons. A defended scope beats an over-claimed one.

- **No aerial insect detection.** Physically implausible at drone standoff; the industry uses static traps.
- **No hyperspectral / multispectral.** RGB systems run $500–2,000 versus $20,000–50,000 for hyperspectral. Out of budget, unnecessary for your claims.
- **No on-device LLM.** The Nano cannot host one. Cloud LLM API with an offline rule-based fallback is correct.
- **No yield prediction model.** Needs multi-season ground-truth yield data you do not have. Offer trend-based *risk* indicators and be clear that is what they are.

---

## 4. Datasets

*(Summary. Full links, sizes, licences and handling rules in `AI_Handbook_2.md` §5.)*

**Core, Indian, field-captured:**
- **Paddy Doctor** — 16,225 images, 13 classes, real Tamil Nadu paddy fields, ~500 man-hours of expert annotation. **Four classes are pest damage** (Hispa, leaf roller, two stem borers) — this is how the drone satisfies the pest requirement without seeing an insect.
- **Sugarcane (Thite et al.)** — 6,748 images, 11 classes, Maharashtra.
- **Sugarcane (Daphal & Koli)** — ~2,521 images, 5 classes, Maharashtra, deliberately multi-device.

**Cross-domain validation (never trained on):** PlantDoc (2,598), PlantWild (18,542, expert cross-validated).

**Pest:** RP11 (rice-specific IP102 refinement, re-annotated) preferred over raw IP102.

**PlantVillage:** auxiliary pre-training only. Never in val/test. Never a headline number. See §0.3.

**Validation protocol — non-negotiable:** deduplicate by perceptual hash, split by group so near-duplicates never straddle splits, hold out a cross-domain test set, and run the background-only bias ablation. Details in Handbook 2 §6.

---

## 5. The AI pipeline

Design principle: **cheap deterministic filters run first and reject most data, so the expensive model runs rarely.** On a device with no INT8 and 4GB shared RAM, what you *don't* run matters more than what you do.

### 5.0 The throughput insight that makes this feasible

At ~0.5 m/s with a ~1 m ground footprint, you generate roughly **1–2 genuinely new scenes per second**. Consecutive video frames are ~95% redundant. Your real requirement is **1–3 inferences/second on selected frames**, not 30 FPS on a stream.

Design as an **event-driven still-image pipeline with a frame selector**, not a video pipeline. "We don't process 30 FPS because 29 of those frames are the same plant" is a strong engineering answer. PatchNet-style keyframe strategies on Jetson Nano delivered 4.1–4.3× speedups on detection models by skipping redundant inter-frames.

### 5.1 Two flight modes

```
MODE A — SURVEY PASS (8–15 m AGL, fast, whole field)
   RGB + thermal, GPS-tagged, nadir
   → per-cell RGB vegetation index
   → coarse stress map, no AI
   → flag anomalous grid cells
   ⚠ Do NOT compute CWSI here — GSD is 32.5 cm/px at 10 m (see §1.2)

MODE B — INSPECTION PASS (1.5–2.5 m AGL, slow, flagged cells only)
   RGB close-up, sensor on ~0.6 m boom (outside rotor wash)
   → full AI pipeline
   → per-plant diagnosis
   → CWSI computed HERE, where GSD is ~6.5 cm/px
```

This mirrors the survey-then-revisit architecture in granted UAV agronomy patents and is your defence of the drone platform. Mount the sensor on a boom — CFD studies at 1 m above canopy measured the downwash footprint as an ellipse of ~0.45 and ~0.4 m semi-axes and recommended a **0.6 m sensor support**.

**v2 note:** moving CWSI from Mode A to Mode B is a direct consequence of §1.2. It costs nothing and removes the dominant error source.

### 5.2 Full pipeline, stage by stage

```
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 0 — CAPTURE & SYNC                              [script, CPU]  │
│  CSI camera @1080p → ring buffer                                     │
│  Tag every frame: GPS, altitude (rangefinder), IMU attitude, time    │
│  Thermal (MLX90640, ~2 Hz) + DHT22/SHT31 → separate slow buffer      │
│  AWB + AE LOCKED to fixed gains (required for §1.1 colourimetry)     │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — FRAME GATE                                  [script, CPU]  │
│  Reject frame unless ALL pass:                                       │
│   • Altitude within inspection band (1.5–2.5 m)                      │
│   • |roll|,|pitch| < threshold  (no attitude smear)                  │
│   • Sharpness: variance of Laplacian > τ_blur                        │
│   • Exposure: histogram not clipped at either end                    │
│   • Novelty: scene displacement vs. last kept frame > 60%            │
│  Typical pass rate: 3–8%. This is the main compute saver.            │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — VEGETATION MASK & TILING                    [script, CPU]  │
│  ExG = 2G − R − B → CLIP to [0,255] before uint8 → Otsu threshold    │
│  Reject frame if vegetation fraction < 25% (soil / sky / operator)   │
│  Slice into 320×320 tiles, 20% overlap  (SAHI-style)                 │
│  Keep tiles with >40% vegetation → typically 4–9 tiles/frame         │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — DISEASE CLASSIFIER              [AI — TensorRT FP16, GPU]  │
│  Deterministic 3x3 grid = 9 tiles; engine BATCH = 9 (v4 fix)         │
│    v3 exported batch 8 and silently dropped the 9th tile             │
│  Engine has FUSED preprocessing incl. BGR→RGB channel swap (v3 fix)  │
│  Backbone: tf_efficientnet_lite0 @224  (stretch: EdgeNeXt-XS)        │
│  Returns RAW LOGITS (needed for energy scoring in Stage 4)           │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — REJECTION STACK              ⚠ CORRECTED IN v3            │
│  Layer 1: argmax == not_crop            → reject (known negative)    │
│  Layer 2: E_crop(x) > τ_E               → reject (unknown object)    │
│         ↳ energy over CROP logits ONLY, excluding not_crop           │
│         ↳ τ_E calibrated on UNSEEN categories, not on not_crop data  │
│  Layer 3: max softmax(UNADJUSTED/T_cal) < τ → abstain  (v4 fix)      │
│  Order: 1 energy on RAW logits  -> is it a known crop condition?     │
│         2 confidence on UNADJUSTED -> is the model sure of anything? │
│         3 prior-adjusted argmax -> WHICH class (label choice only)   │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — AGGREGATION            ⚠ REWRITTEN IN v3  [script, CPU]    │
│                                                                      │
│  SPATIAL (tiles within one frame) — PER-TILE DECISION, THEN MAX:     │
│     tile votes disease if  p_disease >= τ_d  AND                     │
│                            p_disease - p_healthy >= τ_margin         │
│     frame = DISEASE if ANY tile votes (MIL max-pool)                 │
│     frame = HEALTHY if mean healthy prob >= 0.50                     │
│     frame = NOT_CROP / UNCERTAIN otherwise                           │
│     → v2 had NO healthy path and alarmed on pristine fields          │
│     → v2's top-2 mean diluted a 0.92 lesion tile to 0.47             │
│                                                                      │
│  TEMPORAL (frames of the same GPS cell, ~2 m grid):                  │
│     SAME class in k-of-n frames (k=2,n=3) AND mean score >= 0.55     │
│     → v2 checked agreement only; noise-level agreement fired alerts  │
│                                                                      │
│  Cell state is EXPLICIT (v4): DISEASE | HEALTHY | UNCERTAIN | NO_DATA│
│    v3 returned None for healthy AND never-visited cells alike        │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 6 — NUTRIENT & INDEX ANALYTICS                  [script, CPU]  │
│  On vegetation-masked pixels of healthy-classified tiles:            │
│   • Grey-card white balance → CIELAB → mean b* → RELATIVE N map      │
│   • ExG / VARI / TGI → canopy vigour, growth-stage tracking          │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 7 — SENSOR FUSION & RISK ENGINE                 [script, CPU]  │
│  Canopy temp via OTSU COOL-MODE + biophysical gates (§1.2 v3 fix)    │
│    reject if T_cool > T_air + 15°C → bare soil, not vegetation       │
│    (v3 used +7°C and suppressed EVERY drought alarm — see §1.2)      │
│  CWSI = f(Tc, Ta, VPD, crop baseline)  — inspection altitude only    │
│  Soil water balance (FAO-56) + capacitive probe → irrigation         │
│  Agro-met rules → heat / flood / drought / disease-favourable alerts │
│  Trap counts + ETL thresholds + slope → pest pressure alert          │
│  Cross-checks: disease flag + high humidity-hours ⇒ raise confidence  │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 8 — ADVISORY GENERATION                    [rules + LLM API]   │
│  Rules engine emits a STRUCTURED FINDING (JSON):                     │
│    {crop: rice, condition: brown_spot, conf: 0.83, area_pct: 12,     │
│     cell: [lat,lon], cwsi: 0.41, action: fungicide_targeted, ...}    │
│  LLM API converts finding → farmer-readable Hindi/regional text.     │
│  ⚠ The LLM NEVER diagnoses. It only verbalises a decided finding.    │
│  Offline fallback: pre-written templates per finding type + SMS.     │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 9 — STORE & SYNC                                [script, CPU]  │
│  SQLite on device (survives connectivity loss)                       │
│  Opportunistic sync when network available → dashboard, trends       │
└──────────────────────────────────────────────────────────────────────┘
```

#### Why Stage 5 changed again in v3

**v3:** the v2 implementation fixed the *concept* (spatial OR, temporal vote) but shipped an implementation that could not express "healthy." It masked healthy columns to `-inf` and took an argmax over the remaining disease columns, so on a pristine field it returned whichever disease held the largest softmax noise floor — and `aggregate_cell` counted class agreement with no score floor, so three frames of noise-level agreement fired an alert. The corrected version decides *within each tile* first, where disease and healthy probabilities are directly comparable, then max-pools those decisions. Full code in `AI_Handbook_3.md` §8.6.

**v2 (the original problem, retained for context):** version 1 said "require k-of-n agreement across overlapping tiles & frames." Read literally — and anyone implementing the document would read it literally — that means voting across the tiles of a frame. An early-stage lesion (5–15 mm focal spots for paddy blast or sugarcane brown spot) occupies **1 tile out of 9–12**; the healthy tiles outvote it every single time, and you get a false negative on precisely the early infections that early warning exists to catch.

Lowering k to 1 swaps the problem: a single glare artefact, bird dropping or blurred leaf edge triggers a field-wide alarm.

The resolution is that these are two different problems on two different axes. **Spatial aggregation is an OR** (max, or top-k mean — the established max-pooling family of multiple-instance-learning operators). **Temporal aggregation is a vote** (k-of-n across independent passes over the same cell). Separating them fixes both failure modes, and it costs nothing — no retraining, no architecture change, about thirty minutes of code.

**Free bonus:** the per-tile probability map gives you a lesion heatmap for the demo video without any attention mechanism. Overlay tile scores on the frame — one of the best visuals in your submission.

**Stretch:** gated attention MIL (Ilse et al., ICML 2018) learns trainable tile weights and "performs much better than other methods in the small sample size regime." But it needs **bag-level training data**, which your single-leaf datasets do not provide — you would have to synthesise bags. Do not start this before Stage 2 training is done and exported.

### 5.3 Concurrency on the Nano ⚠ **TEARDOWN FIXED IN v3**

Four CPU cores (Cortex-A57). Do not run single-threaded.

- **Thread 1** — capture + tagging (Stage 0), writes to a ring buffer
- **Thread 2** — frame gate + tiling (Stages 1–2), CPU-bound NumPy/OpenCV
- **Thread 3** — GPU inference (Stage 3) via a TensorRT execution context
- **Thread 4** — rejection, aggregation, fusion, persistence (Stages 4–9)

Bounded queues with drop-oldest semantics so a slow stage degrades gracefully instead of exhausting RAM.

#### The one thing that will crash it, and the fix

`import pycuda.autoinit` creates a CUDA context bound to the **importing** thread. If Thread 3 then calls `execute_async_v2()`, the driver raises an invalid-context error because that OS thread has no active context on its stack. TensorRT execution contexts are also not thread-safe.

**Fix — create the context inside the thread that uses it:**

```python
import threading, queue
import pycuda.driver as cuda
# NOTE: there is deliberately NO `import pycuda.autoinit` anywhere.

def inference_thread(engine_path, tile_q, result_q, stop_evt):
    cuda.init()
    ctx = cuda.Device(0).make_context()      # context belongs to THIS thread
    clf = None
    try:
        clf = TRTClassifier(engine_path)     # build the engine wrapper in-thread
        while not stop_evt.is_set():
            try:
                frame_id, tiles = tile_q.get(timeout=0.5)
            except queue.Empty:
                continue
            result_q.put((frame_id, clf.infer(tiles)))
    finally:
        # v3: ORDER MATTERS. Free every PyCUDA object BEFORE destroying the
        # context. ctx.detach() invalidates the context; if a DeviceAllocation,
        # Stream or pagelocked buffer is still alive, garbage collection calls
        # cuMemFree against a dead context and raises LogicError, corrupting
        # process exit and blocking a clean restart of the capture thread.
        if clf is not None:
            clf.close()          # frees d_in/d_out, drops engine + stream
        ctx.pop()
        ctx.detach()
```

#### On the GIL — why threading is still the right choice

You may be advised to abandon threading for multiprocessing on GIL grounds. Two reasons not to:

1. **NumPy and OpenCV release the GIL.** NumPy's documentation states plainly that "many NumPy operations release the GIL, so unlike many situations in Python, it is possible to improve parallel performance by exploiting multithreaded parallelism." SciPy documents the same, and OpenCV's C++ core behaves likewise. Stages 1–2 are almost entirely `cv2` and NumPy calls — the best case for Python threading.

2. **The usual multiprocessing recipe does not exist on your device.** `multiprocessing.shared_memory` is **new in Python 3.8**; the Nano runs **Python 3.6**. Code using it will raise `ImportError` on the target hardware. If you genuinely needed process isolation on 3.6 you would fall back to `multiprocessing.RawArray` or third-party `posix_ipc` — but per point 1, you don't.

**Keep the four threads. Fix only the CUDA context.**

### 5.4 Why Stage 8's separation matters

A judge will ask whether the LLM is making decisions about a farmer's crop. The answer is no: your rules engine and classifier decide; the LLM is a **renderer** turning a structured finding into readable multilingual text. An LLM hallucination cannot invent a disease or recommend a wrong chemical — the worst case is awkward phrasing. A review of deployed platforms identifies **offline functionality and multilingual support** as decisive adoption factors; rules-first with LLM-as-renderer gives you both.

### 5.5 Onboard vs. ground-station streaming

**Compute onboard, and say so.**

- *For the demo:* onboard removes the failure mode most likely to embarrass you on video — a dropped link mid-flight. It also satisfies the problem statement's "operate locally... in areas with poor connectivity."
- *Against streaming:* 1080p over a hobby link at range is exactly where latency and packet loss appear, and it makes your headline claim false.
- *The catch:* Nano + carrier + heatsink/fan is roughly 250–300 g at ~5–10 W. Budget it into the airframe. MAXN needs a proper barrel-jack supply.
- *Sensible hybrid:* process onboard, stream only a low-bitrate annotated preview + JSON findings to the ground. Looks excellent on video.

---

## 6. Optimization — fitting this into 4GB of Maxwell

*(Full export procedure and code in `AI_Handbook_2.md` §9.)*

### 6.1 Deployment mechanics (where projects die)

1. **Train on Colab/laptop.** Never on the Nano.
2. **Export ONNX at opset 11–13** with static input shapes. TRT 8.2 rejects newer opsets and many modern ops.
3. **Fuse preprocessing into the graph** — channel order, `/255`, mean/std normalization all happen on the GPU. New in v2; removes a CPU stall.
4. **Build the TensorRT engine *on the Nano itself*.** Engines are not portable across TRT versions, GPU architectures or drivers.
5. **Run inference through TensorRT + PyCUDA only.** No PyTorch at runtime — saves ~1 GB of RAM and seconds of startup.
6. **Boot headless** (`systemctl set-default multi-user.target`) — frees ~0.5–1 GB of the shared 4 GB. On unified memory this is one of your highest-value changes.
7. **Add swap** (4–6 GB) for the engine *build* only. Never rely on it at inference.
8. **`sudo nvpmodel -m 0 && sudo jetson_clocks`** for MAXN and locked clocks. Needs 5 V/4 A barrel jack and active cooling; without it you leave ~30% of performance unused.

### 6.2 Training-side optimization (where your accuracy comes from)

- **Field-conditioned augmentation** — random shadow, sun flare, brightness jitter, motion blur, ISO noise, JPEG artefacts, aggressive random-resized-crop. The TDR-Model work used exactly this to reach 82.94% on real field data.
- **Class imbalance** — weighted sampling or logit-adjusted loss. Report per-class recall, not just accuracy.
- **Slicing-aided fine-tuning** — train on tiles as well as full images, matching your tile-based inference.
- **Consistent-teaching distillation** — DINOv2 teacher, run online on the identical augmented tensor. Free accuracy at zero inference cost. **See Handbook 2 §7.4 for why the offline version in v1 was wrong.**

### 6.3 Inference-side optimization, ranked by payoff

| Rank | Technique | Expected gain | Notes |
|---|---|---|---|
| 1 | **Frame gating** | 10–30× effective | You process 3–8% of frames. Nothing else comes close. |
| 2 | **FP16 TensorRT vs PyTorch FP32** | ~4–6× | ResNet50 on Nano: ~79 ms → ~42 ms |
| 3 | **Input resolution** 224→192→160 | quadratic | Test the accuracy cost; often small |
| 4 | **Architecture choice** | 2–5× | lite0 vs ResNet50-class |
| 5 | **In-graph preprocessing fusion** | removes a CPU stall | New in v2 |
| 6 | **Batching tiles** | 1.3–2× | One call for 8 tiles beats 8 calls |
| 7 | **Headless boot / RAM discipline** | enables the above | Prevents OOM |
| — | ~~INT8 quantization~~ | **zero — unavailable** | No Maxwell INT8 hardware |
| — | ~~Zero-copy mapped memory~~ | **likely negative** | See below |

#### Zero-copy memory: explicitly rejected (new in v2)

You will meet advice to use `cudaHostAllocMapped` zero-copy memory on the Jetson, reasoning that CPU and GPU share physical RAM so the `memcpy` is wasteful. On Xavier and later that is correct and can be a large win.

**On your board it is likely a regression.** NVIDIA's CUDA-for-Tegra documentation states that pinned memory **is not cached on Tegra devices with compute capability below 7.2**. The Nano's Maxwell GM20B is **compute capability 5.3** — so mapped memory is uncached from the GPU side, and every access becomes a cache miss served from main memory. Analyses of this exact hardware generation note that zero-copy on TX1/TX2-class devices "would produce higher latencies and more bandwidth usage, since every shared memory access by the device will be a cache miss." The widely-cited ~6× zero-copy benchmark was measured on a **Xavier** (CC 7.2, which has I/O coherency).

**Decision: do not restructure the inference path around it.** Benchmark it as a side experiment if curious, but the null hypothesis on CC 5.3 is "no gain or a regression." This becomes worth testing if you upgrade to Orin (CC 8.7).

### 6.4 Realistic performance expectations

| Metric | Target |
|---|---|
| Single 224² image, FP16 TRT, EfficientNet-Lite0 | 12–25 ms |
| Batch of 8 tiles (fused graph) | 60–120 ms |
| Full pipeline end-to-end | 2–4 scene analyses/sec |
| YOLO-class on original Nano (reference) | 5–8 FPS |
| YOLO-class on Orin Nano Super (reference) | 30–60+ FPS |

**Measure your own numbers and report those.**

> **If you buy the Orin Nano Super:** re-enable INT8 (67 INT8 TOPS, 32 tensor cores), move to TensorRT 10.x, raise input resolution to 320–384, run segmentation concurrently, and run a real detector with SAHI on trap imagery. Memory bandwidth rises 68 → 102 GB/s with JetPack 6.2, which matters more than TOPS at these model sizes. Zero-copy also becomes legitimate (CC 8.7 ≥ 7.2).

---

## 7. Hardware shopping list (₹5,000–10,000 budget) ⚠ **UPDATED IN v2**

| Item | Approx. ₹ | Purpose | Priority |
|---|---|---|---|
| **Grey/white balance reference card** | 100–300 | **Colour calibration for the nutrient index — §1.1** | **Highest value per rupee** |
| MLX90640 thermal array (32×24, 55° FOV) | 4,000–6,000 | Canopy temperature → CWSI | High |
| — *or* **MLX90614 single-point IR thermometer** | 400–700 | Same job, narrow FOV (5–10°) so **less soil contamination by construction**. ±0.5 °C, RMSE <1.0 °C vs. reference in vineyard trials. | **Strong budget option — see note** |
| Capacitive soil moisture sensor ×2–3 | 150 each | Soil water balance | High |
| DHT22 or SHT31 (temp + RH) | 200–600 | VPD for CWSI, agro-met rules | High |
| IMX219 / Pi Camera v2 CSI (8MP) | 700–1,200 | Primary RGB (you may have one) | High |
| Yellow sticky traps (pack) | 200–400 | Trap node substrate | High |
| ESP32-CAM | 500–800 | Trap capture node (inference on gateway) | Medium |
| — *or* **ESP32-S3-CAM** | 800–1,200 | Only if you want genuine on-device inference — §3.3 | Optional |
| Lidar/ultrasonic rangefinder | 800–2,000 | Altitude hold for the inspection pass | Medium |
| Active cooling fan for Nano | 400–700 | Required for MAXN | High |

**Note on Ground-Mast Thermal Rescope (v4):** Thermal sensing (MLX90640 array and MLX90614 single-point IR) is rescoped from the drone payload to a fixed ground mast facing the canopy. Flying at 1.5–2.5 m subjects the crop to rotor downwash (forced convection pulling sunlit leaf temp toward ambient air temp, destroying the CWSI signal). The fixed mast provides stable geometry and continuous solar-noon monitoring without downwash distortion. MLX90640 spatial resolution enables Otsu soil rejection; MLX90614 provides continuous drift trace / cross-check.

Total for the high-priority baseline set: roughly **₹6,000–10,000** (excluding dual-bandpass filter).

### 7.1 Sensing Scope Decisions and Trade-offs (Amended per R3)

#### Hardware & Optical Specification Provenance Register

| Item / Parameter | Value / Range | Unit | Provenance Classification | Basis / Source |
|---|---|---|---|---|
| **Waveshare IMX219-77IR Camera** | ~2,500 | INR (₹) | **RECALLED — UNVERIFIED** | Indian hobbyist retail websites (Robu / Silverline catalog memory). Requires live distributor invoice. |
| **MidOpt DB660/850 Filter (Mounted)** | 12,000–18,000 (~$140–$210) | INR (₹) | **RECALLED — UNVERIFIED** | Midwest Optical Systems distributor pricing memory. Subject to import duty and freight. |
| **Total 2-Band Dual-Bandpass Build** | ~14,500–20,500 | INR (₹) | **DERIVED — UNVERIFIED** | Sum of recalled sensor and filter costs. |
| **Entry-level 5-Band Multispectral Camera** | 40,000–80,000+ | INR (₹) | **RECALLED — UNVERIFIED** | Drone agronomy retail recall (e.g. MAPIR Survey3, Parrot Sequoia discontinued baseline). |
| **Research-grade Multispectral Payload** | 2,00,000–5,00,000+ ($2.5k–$6k+) | INR (₹) | **RECALLED — UNVERIFIED** | Enterprise UAV distributor quote recall (e.g. MicaSense RedEdge-P / Altum-PT). |
| **UAV Hyperspectral Payload System** | 8,00,000–25,00,000+ ($10k–$30k+) | INR (₹) | **RECALLED — UNVERIFIED** | Academic tender memory (e.g. Corning MicroHSI 410, Headwall Nano-Hyperspec). |
| **Hyperspectral Payload Mass** | 500–1,500 | grams (g) | **RECALLED — UNVERIFIED** | Manufacturer product brief memory (sensor head + IMU/GPS + data logger). |
| **Dual-Bandpass Camera Mass** | <15 | grams (g) | **RECALLED — UNVERIFIED** | Waveshare IMX219 camera board datasheet memory (<15 g including lens). Unverified against physical scale. |
| **RedEdge Band Range** | 705–740 | nm | **PRIMARY-LITERATURE** | Standard red edge transition region (Horler et al. 1983, Gitelson & Merzlyak 1994). |
| **Visible Red Passband** | 660 (FWHM ~25) | nm | **RECALLED — UNVERIFIED** | Midwest Optical Systems DB660/850 transmission curve memory. Unverified against manufacturer optical test sheet. |
| **Near-Infrared (NIR) Passband** | 850 (FWHM ~35) | nm | **RECALLED — UNVERIFIED** | Midwest Optical Systems DB660/850 transmission curve memory. Unverified against manufacturer optical test sheet. |

#### 1. Two-Band Dual-Bandpass on IMX219 vs Commercial Multispectral Camera
- **The Decision:** Rather than procuring a dedicated commercial multispectral camera, the system implements a dual-bandpass optical path on a secondary CSI port (SENSOR_ID = 1) using a Waveshare IMX219-77IR (NoIR) sensor paired with a MidOpt DB660/850 dual-bandpass interference filter (transmitting at 660 nm Red and 850 nm NIR).
- **Cost Delta [RECALLED — UNVERIFIED]:**
  - Custom 2-band path: Waveshare IMX219-77IR (~₹2,500) + MidOpt DB660/850 mounted filter (~₹12,000–18,000) = **₹14,500–20,500 total**.
  - Commercial 5-band multispectral camera: **₹40,000–80,000+** for entry-level hobbyist/agronomy sensors (e.g. MAPIR Survey3, Parrot Sequoia), and **₹2,00,000–5,00,000+** for research-grade agronomy payloads (e.g. MicaSense RedEdge-P, MicaSense Altum-PT).
  - *Budget Rationale:* For a Tier 1 smart farming system with a total edge BOM target of ₹10,000–25,000, a commercial multispectral camera exceeds the entire system hardware budget by 2× to 10×. The dual-bandpass filter path delivers genuine 660/850 nm physical band separation within the edge BOM constraints.

#### 2. Why Not Hyperspectral?
- **Cost [RECALLED — UNVERIFIED]:** Commercial UAV hyperspectral imaging systems (e.g. Corning MicroHSI 410, Headwall Nano-Hyperspec) cost **₹8,00,000 to ₹25,00,000+** ($10,000–$30,000+ USD), completely inaccessible for smallholder deployment.
- **Payload Weight [RECALLED — UNVERIFIED]:** Hyperspectral imagers weigh 500 g to 1.5 kg, requiring heavy-lift enterprise UAV platforms (e.g. DJI Matrice 300/350 at ₹10,00,000+ airframe cost), whereas the IMX219 weighs under 15 g [RECALLED — UNVERIFIED] and flies on sub-2 kg lightweight drones.
- **Compute and Bandwidth [EMPIRICAL-SYSTEM]:** Hyperspectral sensors capture 100–300 contiguous narrow bands (400–1000 nm), producing 10–50 GB of raw datacubes per 15-minute flight. Processing these cubes requires high-performance desktop GPU workstations running complex spectral unmixing and radiative transfer models, directly contradicting our real-time on-device edge architecture (Jetson Nano).

#### 3. What the Two-Band Approach CANNOT Do (Compared to 5-Band Commercial Payloads)
- **No RedEdge Band (705–740 nm) [PRIMARY-LITERATURE]:** The two-band sensor cannot compute the Normalized Difference Red Edge index ($\text{NDRE} = \frac{\text{NIR} - \text{RE}}{\text{NIR} + \text{RE}}$). In high-biomass, dense-canopy crops (e.g. sugarcane and mature rice at $\text{LAI} > 3$), standard NDVI saturates asymptotically, whereas NDRE penetrates deeper into the canopy to detect late-season nitrogen stress and chlorophyll dynamics.
- **Spectral Leakage and Silicon Cross-Talk [MANUFACTURER-SPEC]:** Unlike commercial sensors with discrete, isolated photodiode arrays and narrow bandpass interference filters for each band, an RGB Bayer array without an IR-cut filter suffers from broad organic dye transmission in the NIR region. 850 nm photons leak heavily into the Red Bayer channel. This cross-talk degrades NDVI dynamic range unless corrected via a measured bench unmixing matrix ($K^{-1}$).
- **Illumination Normalisation (Empirical Line Method vs DLS) [PRIMARY-LITERATURE]:** Commercial 5-band payloads incorporate an upward-facing Downwelling Light Sensor (DLS / sunshine sensor) that measures instantaneous ambient solar irradiance in real time on every exposure. The two-band IMX219 lacks a DLS; to achieve radiometric comparability across different times of day or cloud cover, it requires manual **Empirical Line Method (ELM)** calibration using ground reference reflectance panels (5%, 50%, 84%) imaged pre- and post-flight.

---

## 8. Execution plan

SIH 2026 launched 21 August 2026; portal submission is reported as **20–30 September 2026** — work to **20 September** and confirm via your SPOC. Grand finale December 2026.

### Week 1 — internal hackathon (rover demo, no drone)

- **Day 1:** Download Paddy Doctor, both sugarcane sets, PlantDoc. Assemble `not_crop` (~1,800 images — also your OOD calibration set). Dedup → grouped split → CSV manifests. Pre-resize to max side 512.
- **Day 2:** Bias audit (record the number — it's a slide). Stage 1 baseline. Stage 2 with field augmentation + balancing. **The delta is your headline result.**
- **Day 3:** **Export day.** Fused ONNX wrapper → simplify → numerical *and* end-to-end BGR verification → TensorRT engine on the Nano → benchmark. Do this now, not on Day 13.
- **Day 4:** Cross-domain test (open once). Confusion matrix, per-class recall, Grad-CAM. Temperature calibration + energy threshold. **Implement Stage 5 aggregation (30 min).** Verify FP16 accuracy.
- **Day 5:** Scripts — CWSI with cold-percentile extraction, grey-card white balance + b\* index, ExG masking (with the clipping fix), tiling, frame gate.
- **Day 6:** Rules engine + LLM advisory + minimal dashboard. Wire onto the rover with the CUDA-context-in-thread pattern.
- **Day 7:** Rehearse. Prepare the "why our accuracy isn't 99%" answer.

### Week 2 — PPT + YouTube video

- **Days 8–9:** Trap node (ESP32-CAM capture + gateway blob detection + CNN).
- **Days 10–11:** Drone integration — Nano + camera on boom, altitude hold, capture in flight. Budget more time than you think.
- **Days 12–13:** Film. Survey pass and inspection pass can be shot separately — this is a demonstration video, not a single take. Get clean footage of: flight, live on-device inference with tile-heatmap overlay, trap node counting, dashboard, advisory in Hindi on a phone.
- **Day 14:** Edit, PPT, submit with buffer.

### On not having field data or field access

1. **Paddy Doctor is your field data** — captured in real Tamil Nadu paddy fields over ~500 man-hours. You are not training on lab data.
2. **Field-conditioned augmentation** substitutes for capture diversity better than most people expect.
3. **Printed leaf images** for the flight demo. Print 20–30 high-resolution diseased leaves from held-out test data at life size, mount on stakes in any grass. Completely legitimate for a demonstration video **as long as you say so on the slide.** Judges penalise concealment, not honest scoping.
4. **A dozen potted plants** give you real 3D geometry, shadows and wind for flight footage, with printed lesion cards for diseased classes.
5. **Any accessible green space.** You need 10 minutes of flight, not a research station.

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ONNX→TensorRT conversion fails | **High** | Severe | Rung 1 backbone first. Test on Day 3. Keep an ONNX-Runtime CPU fallback. |
| Fused-preprocessing graph won't export/build | Medium | Moderate | Fall back to plain model + vectorized CPU preprocessing. The fusion is an optimization, not a requirement. |
| BGR/RGB error in the fused wrapper | Medium | Severe (silent) | End-to-end accuracy check on real `cv2.imread` images, not just tensor diff |
| **`CUDA_ERROR_INVALID_CONTEXT` at runtime** | **High if unaddressed** | Severe | Create the CUDA context inside the inference thread; no `pycuda.autoinit` (§5.3) |
| Nano OOM | Medium | Severe | Headless boot, no PyTorch at runtime, bounded queues, batch ≤8 |
| Cross-domain accuracy disappointing | **High** | Moderate | Expected. Frame as honest measurement; show the augmentation ablation. |
| **Drought alarms silently suppressed** | **High if unaddressed** | **Severe (silent)** | Canopy gate at `Ta+15`, not `Ta+7` (§1.2). *v3 rejected a 45.5 °C canopy at Ta=38 as bare soil.* |
| **Pure canopy reports ~50% vegetation** | High if unaddressed | Moderate (silent) | Absolute ExG threshold, not Otsu (§1.2) |
| **Whiteflies vanish when a moth is on the trap** | High if unaddressed | Moderate | Dual-threshold markers (§3.3). *v3: 1 marker of 6.* |
| **Lesion in the 9th tile never classified** | Medium | Severe (silent) | Deterministic 3×3 grid, engine batch 9, chunk loop (Handbook 4 §9.2) |
| **CWSI reports false drought on healthy crop** | **High if unaddressed** | Severe (silent) | Otsu cool-mode + biophysical gates + inspection altitude (§1.2). *The v2 spread gate was inverted and would have caused exactly this.* |
| **BGR/RGB channel swap in the fused engine** | **High if unaddressed** | Severe (silent) | `x[:, [2,1,0]]` in the fused wrapper; red-flag verification test (Handbook 3 §9.2) |
| **Healthy fields trigger disease alerts** | **High if unaddressed** | Severe | Per-tile decision + explicit HEALTHY verdict + score floor on temporal agreement (Handbook 3 §8.6) |
| **Energy OOD threshold is meaningless** | Medium | Moderate | Energy over crop logits only; calibrate on category-disjoint unseen data (Handbook 3 §8.5) |
| Trap counts collapse during an outbreak | Medium | Moderate | Watershed splitting of touching insects (§3.3) |
| `LogicError` on pipeline shutdown blocks restart | Medium | Low-med | Free PyCUDA objects before `ctx.detach()` (§5.3) |
| **Nutrient index drifts with lighting** | **High if unaddressed** | Moderate (silent) | Lock AWB/AE, grey card, relative reporting (§1.1) |
| **Obvious lesions reported as healthy** | **High if unaddressed** | Severe | Per-tile max, not top-k averaging (§5.2 Stage 5). *v2's top-2 mean diluted a 0.92 lesion tile to 0.47.* |
| Drone downwash degrades image quality | Medium | Moderate | 0.6 m sensor boom; 2.5 m altitude; shoot on hover, not in translation |
| Battery limits flight demo | **High** | Low | Film in segments |
| Judges ask "why not just a phone app like Plantix?" | **High** | Moderate | Connectivity independence, per-plant coverage without walking the field, terrain a rover can't cross, multi-modal fusion a phone cannot do |
| Judges challenge drone vs rover | **High** | Moderate | Two-tier survey/inspect architecture + patent precedent + tall-crop reach |
| Teammate's reluctance about the drone | Certain | — | The two-tier design is the compromise: the drone earns its place for survey and targeted revisit; leaf-level diagnosis works from any close-range camera including the rover. Build platform-agnostic and keep both options. |

---

## 10. Answers to prepare for judges

**"What's your accuracy?"**
Two numbers: in-distribution validation and cross-dataset test. Explain why the second is the honest one. Most teams give one inflated number.

**"Why not YOLO?"**
Disease symptoms are diffuse with no natural bounding box, and the high-quality Indian field datasets are class-labelled. We use a detector where detection fits — countable insects on a uniform trap.

**"How does a drone detect insects?"**
It doesn't, and no production system does. It detects *pest damage* — rice Hispa, leaf roller and stem borer damage are explicit classes in our training data — while a static trap node counts the insects, which is what Trapview, iSCOUT and Z-Trap do commercially.

**"How do you prevent misclassifying soil or random objects as disease?"**
Three layers. A trained negative class handles anticipated cases like soil and sky. For genuinely unseen objects — weeds, irrigation pipe, plastic mulch — we compute free energy over the **crop diagnostic logits only**, excluding the negative class. That matters: once you train a `not_crop` class, soil is in-distribution for the model and its energy looks normal, so summing over all logits would defeat the detector. Restricting to crop logits turns the score into "how much crop evidence is there", which a weed fails even though we have no weed class. We calibrate the threshold on a separate set of categories the model has never seen, because outlier exposure data used in training can't honestly calibrate a detector. Third, temperature-scaled confidence handles in-distribution uncertainty, and below threshold we abstain.

**"Your thermal sensor sees soil as well as crop. Doesn't that break your water stress index?"**
It would, and that's why we don't compute it naively. At 10 m our sensor's ground sampling distance is about 32 cm per pixel, and Indian midday soil runs 55–65 °C against a canopy at 30 °C, so a frame average would report severe stress on a healthy plant. We take the measurement on the low-altitude inspection pass where ground sampling distance is about 6 cm, we Otsu-split the thermal histogram and use the cool mode as canopy, and we gate on physics rather than on statistics: if the cool population is still more than 7 °C above air temperature, nothing in that frame is transpiring, so we reject it as bare soil rather than reporting drought.

**"How do you know your alert isn't just noise on a healthy field?"**
Two independent conditions have to hold. Within a frame, a tile only votes for disease if its disease probability clears a calibrated threshold *and* beats that same tile's healthy probability by a margin — comparing within a tile rather than against a frame average. Then across repeated passes over the same GPS cell we require the same condition in at least two of three frames *and* a mean score above threshold. Class agreement alone isn't enough; agreement without evidence is exactly how a healthy field produces a false alarm.

**"How did you fuse preprocessing into the engine without corrupting colour?"**
OpenCV gives BGR and the backbone was trained on RGB, so the fused graph does an explicit channel permutation on the GPU before normalization. It's a one-line gather that costs nothing at runtime. The subtle part is testing it — a random-noise tensor comparison can't detect a channel swap, because uniform noise has identical statistics in all three channels. We verify with a deliberately red test image plus an end-to-end macro-F1 comparison against the reference RGB path.

**"How do you run live inference on a 2019 Jetson Nano with no INT8?"**
FP16 is our floor because Maxwell has no INT8 hardware. Our main lever is not running the model at all — a CPU frame gate rejects over 90% of frames before inference, since consecutive frames of a slow-moving drone are ~95% redundant. We fused normalization and channel-order conversion into the TensorRT graph, and we measured **[X] ms** per 8-tile batch on our own board. We also evaluated zero-copy mapped memory and **rejected it** — pinned memory is uncached below compute capability 7.2 and ours is 5.3, so it would have cost us bandwidth rather than saving it.

> That last sentence is worth rehearsing. Demonstrating that you understood a hardware subtlety well enough to *decline* an optimization is more convincing than any optimization you did adopt.

**"Isn't the Jetson Nano obsolete?"**
Yes, and that's the point: if it runs on 2019 hardware with no INT8 support, it runs on anything a cooperative can afford. Then show your measured latency.

**"How do you handle a disease you've never seen?"**
Energy-based rejection plus a calibrated abstain threshold: the system returns "unrecognised condition — flagged for expert review" rather than guessing. Safer than a confident wrong pesticide recommendation.

---

## 11. Key sources

**Dataset bias / domain gap**
- Noyan (2022), *Uncovering bias in the PlantVillage dataset*, arXiv:2206.04374 — the 8-pixel experiment
- Cross-dataset generalization survey, S277237552200048X — lab→field 33.27%
- *Critical analysis of ML/DL models for crop disease detection*, S2773186326001003

**Datasets** — full links in `AI_Handbook_2.md` §5

**Methods**
- EdgeNeXt (Jetson Nano FP16 TRT latencies) — arXiv:2206.10589
- EfficientFormer (MobileViT latency critique) — arXiv:2206.01191
- SAHI — arXiv:2202.06934
- **Consistent-teaching distillation — Beyer et al., CVPR 2022, arXiv:2106.05237**
- **Energy-based OOD — Liu et al., NeurIPS 2020, arXiv:2010.03759**
- **Attention-based deep MIL — Ilse et al., ICML 2018, arXiv:1802.04712**
- DINOv2 + LoRA + KD in agriculture — *Comput. Electron. Agric.*, S0168169925010063

**Hardware constraints**
- Jetson Nano INT8 unsupported — NVIDIA Developer Forums threads 84060, 83036
- **CUDA for Tegra (pinned memory uncached below CC 7.2)** — https://docs.nvidia.com/cuda/archive/12.2.2/cuda-for-tegra-appnote/
- PyTorch on Jetson Nano — https://qengineering.eu/install-pytorch-on-jetson-nano.html
- **ESP-NN supported chips** — https://github.com/espressif/esp-nn
- NumPy thread safety / GIL release — https://numpy.org/doc/stable/reference/thread_safety.html
- Jetson Orin Nano Super — 67 INT8 TOPS, 102 GB/s, JetPack 6.2, $249

**Agronomy / sensing**
- CWSI — Idso et al. (1981); mung bean and sunflower baseline studies
- Leaf Colour Chart — IRRI Rice Knowledge Bank; Maiti et al., 10.1100/tsw.2004.137
- Rice N from RGB (CIELAB b\*) — *Plant Methods* 10:36
- MLX90640 / MLX90614 low-cost thermal for CWSI — S2772375525002813; PMC10780677
- UAV downwash and sensor boom — PMC6412810
- Sticky trap species-level detection — PMC12669111

**Deployed systems**
- Plantix — GSMA AgriTech; CGIAR Big Data Platform case study; ICRISAT collaboration

---

## 12. Regression tests — **shipped as runnable code in v4**

Three audit rounds. Roughly two-thirds of findings real. **Most of Round 3's findings were bugs in Round 2's fixes**, and one Round 2 bug was reintroduced two sections later in the same document. That loop does not converge while the code lives in prose.

All decision logic now sits in `sih_pipeline_v4/` with a test naming each finding it guards:

```
$ cd sih_pipeline_v4 && python -m pytest tests/ -v
15 passed in 0.22s
```

Verified as tripwires by running the same assertions against the **v3** code — six fail, including `1 marker of 6` versus `6 of 6` on the trap test.

Within twenty minutes of actually executing this logic: five audit findings confirmed with numbers rather than argument, one shown to be **overstated by ~2.5×**, and **one new bug found that three rounds of review missed** — `adaptiveThreshold` hollowing large insects, which was mutually masking the micro-pest bug.

**Import these modules. Do not copy code out of the documents.** That is what stops a fixed bug reappearing elsewhere.

---

## 13. Bottom line

Build **two models**: a crop disease classifier (`tf_efficientnet_lite0` → EdgeNeXt-XS as a stretch) and a tiny pest classifier on blob crops from a static sticky trap, **with inference on a gateway rather than on the ESP32**. Everything else — irrigation, nutrient status, environmental risk, growth staging, analytics — is deterministic Python.

Train on **Paddy Doctor plus the two Maharashtra sugarcane sets**, validate cross-dataset against PlantDoc and PlantWild, keep PlantVillage out of your test set entirely. Your previous failure was the dataset, not YOLO — but classification over tiles is still right here, because your symptoms are diffuse and your labels are class labels.

**Aggregate tiles with max, not votes.** Vote only across repeated passes over the same ground. This is the change from v1 most likely to affect whether your demo detects anything.

**Compute canopy temperature from the coolest quartile, on the low pass.** A frame average at survey altitude reports drought on healthy plants, and it does so convincingly.

Optimize by **not running the model**: gate frames aggressively, then FP16 TensorRT, then resolution. Forget INT8 — your GPU cannot do it. And forget zero-copy — your GPU cannot cache it.

Defend the drone as a **two-tier survey-then-inspect platform**, not a faster rover.

**And run `pytest` before every training job.** Three review rounds found bugs that hand-made test inputs catch in seconds — and executing the code found a ninth that no review did. Correctness you can re-verify in ten minutes beats correctness you have to re-audit.

One last thing, said plainly: you now have four versions of two documents, twenty-three audit findings, and **zero trained models**, with roughly seventeen days left. That has been the dominant risk since the first report and it has not moved. The pipeline logic is finished and tested. Go train Model A.
