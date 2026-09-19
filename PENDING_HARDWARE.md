# Consolidated Pending Hardware & Calibration Register

> [!IMPORTANT]
> **LIVING DOCUMENT DIRECTIVE**: This is a living tracking document. Every future session that touches hardware-gated code, sensor drivers, calibration parameters, or edge hardware must update this document, not just this one pass. When hardware arrives, is wired, or is calibrated, update its status, remove obsolete blockers, and record the empirical values here and in the corresponding source files.

**SIH 2026 — Smart Farming Assistant**  
**Repository**: `sih-smart-farming`  
**Last Updated**: 19 September 2026  
**Architecture Baseline**: Confirmed 12 September 2026 (`ans for vitthal.md §0`) — Two physical devices (Handheld Nano Pod + Fixed ESP32 Mast Node); Aerial Drone Dropped; LoRa Dropped in favor of WiFi.  
**Scope**: Consolidated master tracking register of all physical hardware, uncalibrated provisional defaults, software exception guards (`RuntimeError`, `NotImplementedError`), and physical test data dependencies across the entire project.

---

## Executive Status Summary

The edge software architecture is rigorously defended with explicit software fail-safe guards. When physical hardware is missing or uncalibrated, modules deliberately fail loud (`RuntimeError`, `NotImplementedError`, or `None` with explicit status codes) rather than producing silent, plausible-looking false predictions.

| Subsystem | Hardware Status | Software Guard Status | Primary Bottleneck / Risk |
|---|---|---|---|
| **1. Thermal / CWSI** | MLX90640 wired & verified on Pod (I2C-1, 0x33); wet/dry pad calibration & absolute scale validation pending; MLX90614 (Mast) pending purchase | `canopy_temperature()` live `tc_c` populated (29.81 °C); CWSI guarded by `THERMAL_REFS_NOT_CONFIGURED` (`pod_thermal: PENDING_CALIBRATION`) | Absolute scale unvalidated against reference thermometer (~32 °C ceiling, −8.6 °C ice); wet/dry pad bounding box calibration in `configs/thermal_refs.json` needed for CWSI |
| **2. Weather** | SHT31 & radiation shield pending purchase | `canopy_temperature(air_temp_c)` requires air data | Blocking gap: no air temp / VPD exists to drive CWSI or FAO-56 |
| **3. Sticky Trap Node** | ESP32-CAM & sticky cards pending arrival | `MM_PER_PIXEL` & fiducials guarded by `RuntimeError` | Cannot classify pest density without metric scale calibration |
| **4. Cameras & NDVI** | Pod CSI-0 RGB (IMX219) wired & verified; CSI-1 NoIR & DB660/850 filter pending import | `calib_matrix` guarded by `NotImplementedError`; panel reflectances guarded by `RuntimeError` | MidOpt DB660/850 filter import lead time (2–4 weeks); bench unmixing matrix $K^{-1}$ unmeasured |
| **5. GPS / Telemetry / IMU** | NEO-6M GPS UART wired to `/dev/ttyTHS1` & verified on background thread; Drone attitude/altitude streams descoped | Pure Python NMEA-0183 parsed on background thread (missing fix costs 0.001 s vs 2.5 s); Gate 1 attitude/altitude is legacy drone spec | Outdoor open-sky walking fix for geotagging manual pod walk scan cells (`aggregate_cell`) |
| **6. Irrigation Sensing** | JSN-SR04T & soil probes pending purchase | `WATER_LEVEL_SENSOR_PRESENT = False` guarded by `RuntimeError` | Autonomous actuation dropped (7 Sep 2026); closed-loop paddy advisory blocked until ultrasonic gauge installed |
| **7. Networking / WiFi** | Atheros AR9271 WiFi dongle pending receipt; LoRa SX1278 descoped | Sequential AP/STA mode-switching architecture decided | `SIH-FIELD` hotspot hosting, gateway over WiFi, and `setup_nano_services.sh` pending AR9271 dongle arrival |
| **8. Mast Infrastructure** | 3 m pole, 1.5 m boom arm, IP65 box pending build | Field geometry constraints documented | Need 1.2–1.5 m horizontal offset to keep sticky card out of MLX90614 FOV |
| **9. Compute & Storage** | Jetson Nano verified (TRT FP16 pipeline: 30 frames in 10.6 s, 12 scenes, 108 tiles, ~126 ms latency, soak 60 runs @ ~16 s, 14 MB drift, swap 0, 24–28 °C); eMMC at ~85% capacity | `model_a_fp16.engine` fully verified on hardware (10/10 smoke test) | ~2 GB free space limits local dataset storage |
| **10. Video Test Data** | Validation clip verified (12 passed, 18 rejected balancing exactly); field walk video pending | `edge/frame_gate.py` qualified on synthetic and validation video | Gate rejection rate (92–97% target) to be re-verified on extended live field walk footage |

---

## Subsystem 1 — Thermal / CWSI Subsystem

> **Architecture Note (Confirmed 12 Sep 2026)**: Thermal sensing was rescoped from drone flight and ground mast to the **Handheld Nano Pod** (`ans for vitthal.md §0, B10`). The MLX90640 thermal array is mounted on a rod carried by the farmer, pointing nadir (−90°) at the crop canopy while walking down furrows.  
> **Method**: **Jones (1999) direct method**, using physical wet and dry reference surfaces (wetted and dry cotton) mounted on two forward arms in the MLX90640 FOV:
> $$\text{CWSI} = \frac{T_c - T_{wet}}{T_{dry} - T_{wet}}$$
> Every term ($T_c, T_{wet}, T_{dry}$) is measured in the exact same thermal frame at the same instant. This eliminates the multi-day non-water-stressed baseline campaign requirement for field demonstrations.  
> The fixed ground mast retains the **MLX90614ESF-BAA single-point IR sensor** for unattended continuous canopy temperature logging. **It cannot produce CWSI and must never be presented as doing so** (asserted by unit test `test_section_a_mlx90614_cannot_report_cwsi_without_array`).

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **MLX90640 Thermal Array (32×24, 55°×35° FOV)** | **WIRED & VERIFIED ON JETSON NANO** (7Semi breakout, I²C bus 1, 0x33). Driver operational after register and EEPROM decoding fixes. | Emits live `tc_c` (29.81 °C) with `reason: THERMAL_REFS_NOT_CONFIGURED` and `inputs` status `pod_thermal: PENDING_CALIBRATION`. | Handheld Jones CWSI computation (canopy temperature extraction and spatial response verified). | Absolute scale validation against reference thermometer (currently reads ceiling ~32 °C, ice −8.6 °C; appears stretched at both ends, while relative spatial response is correct); wet/dry reference surface mounting on pod arms and bounding box calibration in `configs/thermal_refs.json`. | Completed (Wired & Verified on Nano) | **VERIFIED (Calibration Pending)** |
| **MLX90614ESF-BAA Single-Point IR Sensor** | Not purchased / arrived | Tested with mock values in `tests/test_edge.py`. `test_section_a_mlx90614_cannot_report_cwsi_without_array` asserts single-point sensor cannot compute CWSI. | Continuous mast canopy temperature tracking and MLX90640 cross-sensor drift trace. | Procurement and mounting on fixed mast station pointing at canopy. | ₹1,200–1,800 (Robu/Robocraze, **ESTIMATED**) | **HIGH** |
| **`MIN_BASELINE_OBSERVATIONS = 14`** (`core/thermal.py:44, 203`) | Value: `14` solar-noon observations. 0 observations recorded. | `fit_non_water_stressed_baseline()` returns `(None, "baseline_insufficient")`. Idso empirical CWSI returns `None`. *(Note: Fallback path; primary path is Jones direct method on pod)*. | Empirical Idso CWSI baseline fitting. Live demos unblocked by Jones direct wet/dry method on pod. | Conducting at least 14 clear-sky solar noon (11:30–14:00) observation sessions with mast station over fully watered healthy crop. | Human protocol (agronomic trial) | **MEDIUM** |
| **`PROVISIONAL_BIMODAL_GAP_C = 4.0`** (`core/thermal.py:48`) | Value: `4.0` °C (uncalibrated default). | When thermal span `hi - lo < 4.0` °C, `canopy_temperature()` short-circuits directly to `cool = t, frac = 1.0` without evaluating Otsu thresholding or the $\eta$ test. Hot soil pixels are included in `cool`, pulling $T_c = \text{np.median}(cool)$ upward. | Accurate soil rejection when canopy-soil temperature delta is narrow (<4 °C). | Field thermal sweep over paired canopy and soil at solar noon across 3 clear sunny days to find minimum observed soil-canopy delta. | Empirical calibration | **MEDIUM** |
| **`PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO = 0.85`** (`core/thermal.py:67`) | Value: `0.85` (Otsu $\eta = \sigma_B^2 / \sigma_T^2$). | Guard prevents bisecting pure canopy sun/shade distributions ($\eta \le 0.77$). If genuine bimodal mixture has $\eta < 0.85$ (soil fraction $\ge 0.25$, $\Delta T \le 10$ °C, canopy spread $\ge 6$ °C), split is rejected and array kept whole, biasing $T_c$ upward and raising false positive drought alarms. | Precision bimodal separation at high soil fraction. | Field sweep with handheld pod over 100% closed canopy and partially covered canopy; verify against ground-truth IR spot readings. | Empirical calibration | **MEDIUM** |

---

## Subsystem 2 — Weather Subsystem (Air Temperature, RH, VPD)

The system requires ambient air temperature and relative humidity at canopy height. `canopy_temperature()` requires `air_temp_c` for its $T_a + 15^\circ\text{C}$ biophysical bare-soil gate; empirical CWSI baseline fitting requires Vapor Pressure Deficit ($\text{VPD} = f(T_a, \text{RH})$); FAO-56 reference evapotranspiration ($ET_0$) requires air temperature and RH.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **SHT31-D I²C Sensor Breakout (GY-SHT31-D)** | Not purchased / arrived | No sensor currently produces air temperature or RH. Without it, CWSI baseline fitting fails, and `canopy_temperature()` cannot execute. | Real-time VPD calculation, CWSI baseline fitting, FAO-56 Hargreaves-Samani $ET_0$. | Procurement and I²C wiring to mast controller. | ₹700–1,100 (GY-SHT31-D, **ESTIMATED**) / ₹1,800–2,600 (DFRobot SEN0385 outdoor, **ESTIMATED**) | **CRITICAL** |
| **Radiation Shield (Stevenson screen / stacked saucers)** | Not built / purchased | Temperature sensor in direct sunlight reads 5–15 °C high due to radiative heating, corrupting VPD and CWSI regressions. | Radiometric and hygrometric measurement accuracy in open field. | Assembling DIY stacked white plastic saucer radiation shield (~₹200) or procuring commercial shield. | ~₹200 DIY / ₹1,500–3,000 commercial (**ESTIMATED**) | **HIGH** |

---

## Subsystem 3 — Sticky Trap Node Subsystem

Consolidated smart sticky trap node at canopy top (10–30 cm above crop canopy, opposite side of mast from thermal sensor). Mounts on a sliding collar / hinged arm for manual card servicing.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **ESP32-CAM AI-Thinker (OV2640) + Programmer** | Not purchased / arrived | Software evaluated in unit test mode. | Automated daily trap image capture, Model B real trap dataset collection, gateway POST receiver end-to-end testing. | Procuring ESP32-CAM (external antenna variant), USB programmer, acrylic lens enclosure, and flashing firmware. | ₹500–750 (ESP32-CAM) + ₹150–300 (programmer) (**ESTIMATED**) | **HIGH** |
| **Yellow Sticky Cards (15×25 cm) + Adhesive Paper** | Not purchased / printed | Evaluated via synthetic images. | Field trap deployment, pest capture, physical scale verification. | Purchasing yellow sticky card pack and printing ArUco fiducials on matte adhesive sheets. | ₹150–400/pack cards + ~₹50 printing (**ESTIMATED**) | **HIGH** |
| **`MM_PER_PIXEL`** (`core/trap_segmentation.py:26`) | Value: `None`. Unit test target: `0.125` mm/px. | `segment_trap_blobs()` raises `RuntimeError("Scale uncalibrated: MM_PER_PIXEL must be set...")`. Micro-pests cannot be sized or classified. | Physical blob sizing, micro-pest filtering (whiteflies, thrips, aphids vs debris), ETL threshold comparisons. | Mounting camera in final trap enclosure, photographing a metric ruler at exact focal plane, and setting calibrated `MM_PER_PIXEL`. | Physical calibration step | **CRITICAL** |
| **`PROVISIONAL_MARKER_SPACING_W_MM = 80.0`, `_H_MM = 55.0`, `MARKER_SPACING_CONFIRMED = False`** (`core/trap_segmentation.py:25, 38-39`) | Values: `80.0` mm, `55.0` mm; flag: `False`. | `calibrate_scale_from_fiducials()` raises `RuntimeError("Fiducial marker physical spacing unconfirmed...")` on production cards unless `allow_provisional=True`. | Automated run-time optical scale derivation from printed card fiducials. | Measuring printed fiducial dots on physical yellow sticky cards with vernier calipers and flipping `MARKER_SPACING_CONFIRMED = True`. | Physical measurement step | **HIGH** |
| **`PROVISIONAL_FOCUS_FLOOR_LAPLACIAN = 100.0`, `PROVISIONAL_MAX_ROTATION_DEG = 10.0`, `PROVISIONAL_MAX_CARD_SATURATION = 0.30`** (`core/trap_segmentation.py:32-34`) | Values: `100.0`, `10.0` deg, `0.30` (30% area). | Heuristic quality gates for trap photos. Saturated card (>30% blob area) triggers card replacement warning. | Trap photo validation under adverse field conditions. | Field validation across dirty, shadowed, or heavily pest-laden cards. | Field tuning | **MEDIUM** |
| **`PROVISIONAL_TREND_*` (Min obs = 3, Window = 7, Rising = 1.50, Falling = 0.67, Min rate diff = 1.0)** (`core/trap_segmentation.py:42-46`) | Heuristic defaults based on engineering judgement. | Trend analyzer produces alerts based on provisional ratios. | Pest population trend monitoring and early surge alerting. | Multi-week daily pest count field trials comparing automated alerts against expert entomologist observations. | Longitudinal agronomic trial | **MEDIUM** |

---

## Subsystem 4 — Cameras & Multispectral NDVI Subsystem

> **Architecture Note (Confirmed 12 Sep 2026)**: Rescoped from aerial drone payload to **Handheld Inspection Pod** (`ans for vitthal.md §0`). The farmer walks the field holding the Jetson Nano pod on a rod. The pod carries co-mounted dual CSI cameras: CSI-0 RGB (IMX219-77) for high-resolution disease leaf inspection; CSI-1 NoIR (IMX219-77IR) fitted with a MidOpt DB660/850 dual-bandpass filter for NDVI canopy vigor scanning.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **Raspberry Pi Camera Module V2 (IMX219 RGB)** | **WIRED & VERIFIED ON JETSON NANO** (CSI-0). Stills and 1080p30 H.264 video capture both working. | Operational on `/dev/video0`. | High-resolution disease leaf inspection and live Model A pipeline video stream. | Completed (Wired & Verified on Nano). | Completed | **VERIFIED** |
| **MidOpt DB660/850 Dual-Bandpass Filter** | Not ordered / arrived | Inverted Bayer mapping (Blue=NIR, Red=660nm) cannot separate spectra without filter. | Multispectral NDVI field scanning pass. **On-pod NDVI pipeline completely blocked without this optical filter.** | Urgent ordering and importing (2–4 week lead time). Mounting filter over IMX219-77 NoIR lens on pod. | ₹8,000–18,000 (**ESTIMATED**, import) | **CRITICAL (Highest Lead-Time Risk)** |
| **IMX219-77IR NoIR Camera Module** | Not arrived / connected | `edge/camera.py` DualCameraPipeline implemented; NoIR port CSI-1 unverified against physical sensor. | Dual-camera synchronized capture on pod. | Procurement, connecting to Jetson Nano CSI port 1, running `edge/camera.py` self-test. | ₹1,500–2,500 (NoIR) (**ESTIMATED**) | **HIGH** |
| **`calib_matrix` (Bench cross-talk unmixing matrix $K^{-1}$)** (`core/ndvi.py:75, 101`) | Value: `None`. | `apply_channel_response_correction()` raises `NotImplementedError("Bench response calibration on assembled IMX219-77IR + MidOpt DB660/850...")`. If bypassed with identity matrix, NIR leakage into Red channel collapses NDVI dynamic range. | Radiometrically valid dual-bandpass NDVI computation. | Placing assembled camera before integrating sphere / monochromatic light sources at 660 nm and 850 nm; running `scripts/calibrate_dual_bandpass.py` to generate `configs/db660_850_calibration.json`. | Optical bench calibration step | **CRITICAL** |
| **`PROVISIONAL_PANEL_REFLECTANCES = (0.05, 0.50, 0.84)`, `PANEL_REFLECTANCES_CONFIRMED = False`** (`core/ndvi.py:67-68`) | Values: `(0.05, 0.50, 0.84)`; flag: `False`. | `apply_empirical_line_calibration()` raises `RuntimeError("Panel reflectances have not been confirmed...")` on production data unless `allow_provisional=True`. | Empirical Line Method (ELM) radiometric calibration across walk sessions. | Procuring calibrated diffuse reflectance panels with manufacturer certificates; entering certified values in config and setting flag `True`. | ₹3,000–12,000 (**ESTIMATED**) | **HIGH** |
| **`PROVISIONAL_EXPOSURE_NS = 10000000` (10 ms), `PROVISIONAL_GAIN = 1.0`** (`edge/camera.py:47-49`) | Values: 10 ms, 1.0x analog gain (untested defaults). | Under bright Indian midday sun (~80,000–100,000 lux), 10 ms at gain 1.0 severely saturates the Red channel. Saturation invalidates division-based indices (NDVI, VARI). | Radiometrically non-saturating handheld pod capture. | Field exposure calibration over 18% gray card at solar noon; setting non-saturating exposure time in config. | Field optical calibration | **HIGH** |
| **`PROVISIONAL_EXG_VEG_THRESHOLD = 20`** (`core/indices.py:37`, `configs/train_config.py:46`) | Value: `20` (uint8 ExG scale). | Soil background spectra vary by plot, moisture, and sun angle. If set too high: shadowed canopy is lost. If set too low: wet dark soil is misclassified as crop canopy. | Absolute-threshold canopy masking in `core/indices.py` and `edge/tiler.py`. | Capturing field imagery over local wet and dry bare soil; computing ExG histogram; setting threshold 3 standard deviations above soil peak. | Field radiometric calibration | **HIGH** |
| **`PROVISIONAL_MIN_CANOPY_FRACTION = 0.15`** (`core/indices.py:34`) | Value: `0.15` (15% canopy coverage). | Low-canopy frames return `None` with `"insufficient_canopy"`. If set too high: early seedling emergence is rejected. | Canopy vigor tracking during early emergence stages. | Agronomist review of crop emergence canopy coverage thresholds for rice, sugarcane, wheat. | Agronomic review | **LOW** |

---

## Subsystem 5 — GPS & Telemetry / IMU Subsystem

> **Architecture Note (Confirmed 12 Sep 2026)**: Rescoped from aerial drone flight telemetry to **Handheld Inspection Pod Geotagging** (`ans for vitthal.md §0`).  
> **Handheld Pod Telemetry Role**: GPS receiver on the pod (`NEO-6M / NEO-M8N UART`) tags captured frames with geographic coordinates for field cell aggregation (`aggregate_cell`, SQLite `cell_verdicts`), enabling spatial disease mapping.  
> **Legacy Drone Telemetry Framing**: Drone flight cruise altitude ($1.5\text{–}2.5\text{ m}$) and attitude bounds ($|\text{roll}|, |\text{pitch}| \le 15^\circ$) in `BUILD_CHECKLIST_1.md:340` were drone flight controller specifications. A handheld pod carried by a walking person does not produce a downward rangefinder stream (GPS vertical error is $\pm 10\text{–}20\text{ m}$), and walking motion induces natural periodic sway. In `edge/frame_gate.py`, Gate 1 safely passes when `telemetry is None` (optional). The active, mission-critical quality gates for the handheld pod are **Gate 2 (exposure clipping)**, **Gate 3 (variance of Laplacian blur rejection $\ge 100$)**, and **Gate 4 (scene novelty phase correlation displacement $> 60\%$)**.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **NEO-6M GPS UART Module** | **WIRED & VERIFIED ON JETSON NANO** (Connected to `/dev/ttyTHS1` at 9600 baud). Emits valid NMEA sentences. Read on a dedicated background thread (`get_latest_fix()`); missing fix costs 0.001 s per frame instead of 2.5 s blocking delay. Records `gps: null`, `cell_id: "cell_walk_pod"` when no fix is available indoors. | Pure Python NMEA-0183 parser verified; non-blocking background reader active. | Spatial GPS cell indexing, repetitive walk pass aggregation (`aggregate_cell`), map coordinate tagging outdoors. | Outdoor walking test under open sky with real satellite lock. | Completed (Wired & Verified on Nano) | **VERIFIED (Outdoor Fix Pending)** |
| **Flight IMU / Rangefinder Feed** | **LEGACY DRONE SPEC (DESCOPED)** | Not present on Handheld Pod BOM (`ans for vitthal.md §0`). Bypassed in `edge/frame_gate.py` when `telemetry=None`. | None on handheld pod. | Descoped with drone. Retained only if an IMU is optionally added to monitor severe rod tilt. | Out of scope | **DESCOPED** |
| **`PROVISIONAL_GATE_MAX_ROLL_DEG = 15.0`, `PROVISIONAL_GATE_MAX_PITCH_DEG = 15.0`** (`configs/train_config.py:37-38`) | Values: `15.0` deg, `15.0` deg (uncalibrated defaults). | Legacy drone attitude limits. If an IMU is mounted on the pod rod, excessive tilt indicates non-nadir scanning. | Gating pod tilt (if IMU present); bypassed if `telemetry=None`. | Field walking trials measuring natural hand sway; calibrate or bypass for handheld mode. | Field calibration | **LOW (Legacy)** |
| **`GATE_ALTITUDE_MIN_M = 1.5`, `GATE_ALTITUDE_MAX_M = 2.5`** (`configs/train_config.py:35-36`) | Values: `1.5` m, `2.5` m. | Legacy drone cruise altitude band. Handheld pod rod height varies by crop height ($1.0\text{–}1.8\text{ m}$). | Bypassed when `telemetry=None` or when `altitude_m` is omitted from pod telemetry. | Maintain omission of `altitude_m` on handheld scans to avoid false rejection. | Software convention | **LOW (Legacy)** |
| **`PROVISIONAL_TAU_BLUR = 100.0`** (`configs/train_config.py:39`) | Value: `100.0` (variance of Laplacian on grayscale). | **ACTIVE & CRITICAL FOR POD**: Farmer walking motion causes motion blur. If set too low: blurred frames reach classifier, raising false negative rate. | Walking motion blur rejection in `edge/frame_gate.py`. | Walking field rows at normal scan pace (1–1.5 m/s) with pod; recording video; calibrating threshold between sharp and footstep-blurred frames. | Test walk video | **HIGH** |
| **`PROVISIONAL_GATE_CLIPPING_DARK_DN = 5`, `PROVISIONAL_GATE_CLIPPING_BRIGHT_DN = 250`, `PROVISIONAL_GATE_CLIPPING_MAX_FRACTION = 0.02`** (`configs/train_config.py:40-42`) | Values: DN `5`, DN `250`, fraction `0.02` (2%). | **ACTIVE & CRITICAL FOR POD**: Shadows from the operator/rod or midday specular glints trigger exposure clipping. | Exposure rejection in `edge/frame_gate.py`. | Midday field histogram sweep with handheld pod over crop canopy and shadows. | Field calibration | **MEDIUM** |

---

## Subsystem 6 — Irrigation Sensing Subsystem (Advisory Only)

> **Scope Note (7 Sep 2026)**: Autonomous irrigation actuation hardware (solenoid valves, MOSFET drivers, YF-S201 flow meters) was dropped from the project scope. The system computes agronomic water requirements and presents manual prescriptions to the farmer. `edge/actuation.py` and `edge/flow.py` are retained as validated reference implementations only and are NOT wired into runtime paths. **Sensing is unchanged and mandatory.**

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **JSN-SR04T / AJ-SR04M Waterproof Ultrasonic Sensor** | Not purchased / arrived | Evaluated via simulated depths in `tests/test_edge.py`. | Paddy AWD ponded water depth measurement. **Paddy AWD closed loop is completely inoperable without this sensor.** | Purchasing JSN-SR04T, mounting transducer 40 cm above paddy basin looking down, temperature-compensating via SHT31. | ₹232–500 (Robokits RKI-1055, Robocraze, **CHECKED 07-SEP-2026**) | **CRITICAL** |
| **`WATER_LEVEL_SENSOR_PRESENT = False`** (`edge/irrigation_model.py:111`) | Flag: `False`. Mode forced to `"open_loop"`. | Passing `water_level_mm` while flag is `False` raises `RuntimeError("water_level_mm passed but WATER_LEVEL_SENSOR_PRESENT is False...")`. | Closed-loop paddy AWD target ponding depth control ($\Delta WL$). | Procuring JSN-SR04T, connecting to node, flipping flag to `True`. | Hardware purchase & wiring | **CRITICAL** |
| **ADS1115 16-Bit I²C ADC + Capacitive Soil Probes v2.0 (×2–3)** | Not purchased / arrived | Evaluated via mock ADC in `tests/test_edge.py`. | Upland soil moisture depletion tracking for FAO-56 irrigation scheduling. | Purchasing ADS1115 and capacitive soil probes; burying probes 15–20 cm in soil outside pole shadow. | ₹250–450 (ADC) + ₹60–300 ea (probes) (**CHECKED 07-SEP-2026**) | **HIGH** |
| **`PROVISIONAL_V_DRY = 3.00` V, `PROVISIONAL_V_WET = 1.20` V** (`edge/adc.py:80-81`) | Values: `3.00` V (dry air), `1.20` V (saturated). | Capacitive probes are qualitative unless calibrated. Using uncalibrated voltages leads to gross volumetric water content errors. | Quantitative volumetric soil water content estimation. | Measuring dry air voltage and water-saturated soil voltage on the specific purchased probe batch; calibrating against oven-dried soil cores. | Physical calibration step | **HIGH** |
| **`PROVISIONAL_PADDY_PERCOLATION_MM_DAY = 5.0`, `PROVISIONAL_PADDY_PERCOLATION_BY_SOIL`** (`edge/irrigation_model.py:119-124`) | Values: Default `5.0` mm/day (`clay: 4.0`, `loam: 6.0`, `sand: 8.0`). | Subsoil percolation and seepage vary by an order of magnitude based on puddled hardpan quality. Model miscalculates AWD dry-down intervals by multiple days. | Precision daily paddy water balance modeling. | Conducting double-ring infiltrometer percolation tests on the farm plot after puddling; updating table in config. | Field soil measurement | **MEDIUM** |
| **`PROVISIONAL_PADDY_SATURATION_MM = 200.0`, `PROVISIONAL_PADDY_LAND_PREP_PONDING_MM = 25.0`** (`edge/irrigation_model.py:117-118`) | Values: `200.0` mm, `25.0` mm (net net depth = 225.0 mm). | Land prep volume calculation is based on unmeasured pre-season moisture deficit. | Puddling water delivery volume accuracy. | Measuring soil moisture profile prior to land prep; computing exact water depth to reach liquid limit plus target ponding. | Field soil test | **LOW** |
| **`is_flowering` Phenological Stage Input** (`edge/irrigation_model.py:690, 712-713`) | Input default: `False`. | If caller omits stage information, fail-safe gate `STAGE_UNKNOWN_AWD_WITHHELD` engages, permanently suspending AWD water savings and forcing conservative continuous flooding. | Operational AWD water-savings during non-critical vegetative stages. | Establishing weekly field scouting protocol (50–85 days post-transplanting) to manually toggle flowering stage in dashboard. | Operational protocol | **HIGH** |
| *Actuation Reference Code* (`edge/actuation.py`, `edge/flow.py`) | Retained as validated reference implementations. | Actuation hardware is out of scope. Tested by 6 tests in `tests/test_edge.py`. | Closed-loop valve actuation (future scope). | None (advisory-only deployment). | Out of scope | N/A |

---

## Subsystem 7 — Networking, Radio & WiFi Subsystem

> **Architecture Decision (Confirmed 12–15 Sep 2026)**:
> 1. **LoRa (SX1278 433 MHz) is DESCOPED**: Physical radio dropped in favor of an all-WiFi architecture (`ans for vitthal.md §0, §4`). `edge/lora.py` packet framing and CRC logic is retained solely as non-runtime reference.
> 2. **Single Atheros AR9271 USB WiFi Dongle**: Jetson Nano pod carries a single AR9271 dongle (RTL8188EUS removed entirely). Concurrent AP+STA on this chipset is explicitly untested/unsupported per driver documentation (`wireless.docs.kernel.org`, `ath9k_htc`). Therefore, **sequential mode-switching is the confirmed and only supported operating architecture**.
> 3. **Operating States**:
>    * **Normal State**: AP mode hosting SSID `SIH-FIELD` (WPA2-PSK password *Ahsan123*, static IP `192.168.4.1:8080`, `dnsmasq` DHCP) for the farmer's smartphone app.
>    * **Mast Sync State**: When brought near the mast station, the radio switches to STA mode, connects to mast SoftAP `SIH-NODE-01` (`192.168.9.1`), pulls stored sensor telemetry and ESP32-CAM trap JPEGs over HTTP, and switches back to AP mode.
>    * **Accepted Drop**: A 20–30s connectivity drop for the phone occurs during sync. This is accepted system design, not a defect.
> 4. **Gateway Software Architecture Requirement**: The `gateway/` service orchestrates this sequential mode transition and exposes an explicit `"syncing"` state to the app rather than silently dropping the interface.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **USB WiFi Dongle (Atheros AR9271, External Antenna)** | Selected/ordered, not yet arrived | Jetson Nano is currently on development network; cannot host standalone field AP. | Hosting `SIH-FIELD` AP on pod and executing sequential STA sync with mast node. | Delivery, plugging into Jetson USB port, verifying interface enumeration (`wlan0`). | ~₹600–1,200 (**ESTIMATED**) | **CRITICAL** |
| **Driver Firmware (`/lib/firmware/ath9k_htc/htc_9271.fw`)** | Unverified on flashed L4T rootfs | If firmware blob is missing from rootfs, AR9271 will fail to initialize on USB plug (`dmesg: Failed to load firmware`). | Driver initialization for Atheros AR9271. | Run `ls /lib/firmware/ath9k_htc/htc_9271.fw` on Nano via SSH. If absent, copy from `github.com/qca/open-ath9k-htc-firmware`. | Checklist action (Zero cost) | **HIGH (Checklist)** |
| **Gateway Sequential AP/STA Manager** | `edge/wifi_switch.py` implemented with safe finally cleanup | Mode switch tested in simulation; physical radio testing pending AR9271 arrival. | Smooth 20–30s mast sync without persistent phone disconnect errors. | Bench test switching between `SIH-FIELD` AP and `SIH-NODE-01` STA once dongle arrives. | Hardware test | **HIGH** |
| **ESP32 Mast SoftAP (`SIH-NODE-01` at 192.168.9.1)** | Firmware ready in `firmware/node_n01` | Mast node cannot serve stored data until flashed. | Pulling unattended mast data into pod. | Flashing ESP32 firmware with `WiFi.softAPConfig()` on subnet `192.168.9.0/24`. | Firmware flashing | **HIGH** |
| **SX1278 Ra-02 433 MHz LoRa Transceiver** | **DESCOPED (12 Sep 2026)** | `edge/lora.py` tested in software; physical SPI transmission unverified and excluded from runtime pipeline. | None. Radio link dropped in favor of WiFi AP/STA sync. | Hardware eliminated from scope. | None (Descoped) | **DESCOPED** |

---

## Subsystem 8 — Mast Mechanical & Power Infrastructure

Consolidated ground station: 3 m pole, 1.5 m horizontal boom arm, solar power system, and IP65 enclosure.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **Mast Pole & Horizontal Boom Arm (3 m pole, 1.5 m boom)** | Not built / constructed | Field geometry constraints documented. | Field mounting of MLX90614, SHT31, sticky trap, and solar panel; maintaining horizontal standoff so sticky card stays out of thermal FOV. | Procuring 3 m GI pipe / treated timber, 1.5 m GI pipe boom arm, clamps, and erecting on farm test plot. | ₹900–2,200 total (**ESTIMATED**) | **HIGH** |
| **Solar Power System (10–20 W panel, PWM controller, 12V 7Ah SLA / 3S battery)** | Not purchased | Autonomous off-grid solar operation of field mast station. | Procurement, wiring through LM2596 buck converter to ESP32 node. | ₹1,680–3,750 total (**ESTIMATED**) | **HIGH** |
| **IP65 Weatherproof Enclosure & Cable Glands** | Not purchased | Weatherproofing mast electronics, battery, and radio against monsoon rain and dust. | Procuring IP65 ABS junction box (200×150×100 mm) and cable glands. | ₹700–1,500 total (**ESTIMATED**) | **HIGH** |

---

## Subsystem 9 — Jetson Nano Edge Compute & Storage Constraints

Edge compute hardware is flashed, benchmarked, and operational.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **Jetson Nano 4GB eMMC Storage (~85% full of 14 GB)** | Board flashed and reachable at `nvidia@nvidia.local`. ~2 GB free disk space. SQLite persistence (`edge/storage.py`) ceiling empirically measured at 121 MB (5.9% of free eMMC) for 50,000 retained frames; bounded by WAL checkpoint truncate with zero growth past high-water mark due to freelist page reuse. | If disk reaches 100%, OS crashes, TensorRT cache writes fail, and SQLite logging halts. | Downloading large video evaluation datasets; building heavy packages; multi-day sensor log storage. | Mounting high-speed USB 3.0 SSD / microSD card to `/mnt/storage`; moving logging and video datasets off root eMMC. | ~₹800–1,500 for 128GB USB 3.0 drive (**ESTIMATED**) | **MEDIUM** |
| **TensorRT FP16 Engine (`artifacts/engines/model_a_fp16.engine`)** | **VERIFIED ON JETSON NANO**. Pipeline processes 30 frames in 10.6 s with 12 scenes passed and 108 tiles classified; warm TRT inference ~126 ms per 9-tile scene (~14 ms/tile), ~5 s one-time engine init; smoke test 10/10; soak of 60 consecutive runs at ~16 s each with 14 MB memory drift, swap 0, 24–28 °C, no thermal throttling; zero queue drops, rejections balancing exactly. | Verified with real FP16 engine on Maxwell GPU. | None (Pipeline inference fully operational). | Already completed and verified on-device. | None | **VERIFIED** |

---

## Subsystem 10 — Field Video Test Data

Validation test clip verified; extended real-world field video recommended for multi-hectare sweeps.

| Item / Parameter | Current State | Code Guard & Failure Mode If Left As-Is | Blocks | Unblocked By | Price Tag / Lead Time | Priority |
|---|---|---|---|---|---|---|
| **Real Field Walk Video Footage (1080p nadir crop scanning)** | Multi-class validation video clip (`test_video_from_dataset_images.mp4`) verified on Nano (30 frames, 12 scenes passed, 18 rejected balancing exactly). Extended field walk footage pending. | `edge/frame_gate.py` qualified on validation sequence; rejection accounting verified. | Long-duration field walk validation of the 92–97% frame rejection target under varied sun glint and operator walking pacing. | Recording 1080p nadir walk video (1–1.5 m/s walk speed with pod on rod at 1.0–1.8 m height) over real crop canopy; placing file at `data/video/test_video.mp4`. | Field video recording walk | **MEDIUM** |

---

## Critical Path Execution Priorities

```
[VERIFIED ON HARDWARE BENCH]
  ├── Model A TensorRT FP16 Pipeline (30 frames in 10.6 s, 12 scenes, 108 tiles, ~126 ms latency, 60-run soak)
  ├── Raspberry Pi Camera Module V2 (IMX219) on CSI-0 (stills and 1080p30 H.264 working)
  ├── NEO-6M GPS UART on /dev/ttyTHS1 (background reader thread, 0.001 s non-blocking fix access)
  └── MLX90640 Thermal Array on I2C-1 at 0x33 (driver verified, live tc_c 29.81 °C, PENDING_CALIBRATION)

[OUTSTANDING HARDWARE & BENCH CALIBRATIONS]
  ├── Atheros AR9271 USB WiFi Dongle (not yet received; blocks standalone SIH-FIELD AP & mast sync)
  ├── Gateway over WiFi & SIH-FIELD Hotspot validation with field mobile app
  ├── Production Systemd Services installation (setup_nano_services.sh)
  ├── Wet/Dry Thermal Reference Pad calibration (configs/thermal_refs.json) to unblock CWSI
  ├── MLX90640 absolute temperature scale validation against reference thermometer (~32 °C ceiling, -8.6 °C ice)
  ├── MidOpt DB660/850 Filter import (2–4 week lead time; blocks on-pod multispectral NDVI)
  ├── Raspberry Pi NoIR Camera Module V2 (IMX219) unmixing matrix K^-1 calibration (calibrate_dual_bandpass.py)
  └── Ground Mast Node ESP32 + ESP32-CAM firmware flashing and bench test
```

---

## Maintainer Verification Protocol

Before any future developer or agent claims a hardware-gated item is resolved:
1. Verify physical hardware is present, wired, and communicating (`i2cdetect -y -r 1`, `v4l2-ctl --list-devices`, `lsusb`).
2. Verify empirical calibration data is checked into `configs/` or logged in the repository.
3. Remove the corresponding `RuntimeError` / `NotImplementedError` guard or update `PROVISIONAL_` parameter to calibrated value.
4. Run full unit test regression suite (`pytest -v`) to confirm zero regressions (331 tests passing).
5. Update this register (`PENDING_HARDWARE.md`) reflecting the resolved status.
