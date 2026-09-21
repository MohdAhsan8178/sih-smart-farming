# Payload Contract Specification (v1.0)

**Last Updated:** 21 September 2026  
**Document:** `docs/PAYLOAD_CONTRACT.md`  
**Purpose:** Authoritative wire schema contract for all advisory JSON payloads emitted by `GET /api/v1/advisory/<id_or_seq>`, `GET /api/v1/advisory/latest`, and edge telemetry endpoints.  
**Generation & Validation:** Synchronized with code constants and verified against real synthesized advisory payloads via `scripts/generate_payload_contract.py` and `tests/test_payload_contract.py`.

---

## Gateway Connection & Base URL

The offline mobile API gateway is served at:
```
http://192.168.4.1:8080
```
- **Connection Method:** The farmer's mobile phone connects directly to the Handheld Pod's local WiFi Access Point:
  - **SSID:** `SIH-FIELD`
  - **IP Address:** `192.168.4.1`
  - **Port:** `8080` (plain HTTP, no TLS)
- **Hardware Verification Status:** All REST API endpoints, response schemas, and payload generation have been verified over Ethernet on the Jetson Nano hardware. The standalone `SIH-FIELD` WiFi Access Point hotspot itself is pending deployment of the replacement AR9271 USB WiFi dongle.

---

## 1. Top-Level Advisory Schema Fields

| Field Path | Type | Nullable | Allowed Values / Range | Description |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire contract schema version |
| `advisory_id` | string | No | String identifier (e.g. `2026-09-19T10:00:00Z_F01`) | Unique advisory document identifier |
| `seq` | integer | No | `>= 1` | Monotonic sequential advisory sequence number assigned by SQLite rowid |
| `generated_at_utc` | string | No | ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`) | Timestamp of advisory synthesis |
| `inference_backend` | string | No | `"trt"`, `"onnx"`, `"mock"` | Explicit runtime inference engine backend |
| `replay` | boolean | No | `true`, `false` | Provenance label (`true` for seeded replay / demo data; `false` for live) |
| `scan` | object | No | Object | Session scan summary metrics |
| `crop_health` | object | No | Object | Frame consensus crop health diagnosis |
| `growth_stage` | object | No | Object | Phenological stage estimation (FAO-56 Chapter 5) |
| `vegetation` | object | No | Object | Nadir RGB vegetation cover & relative indices |
| `thermal` | object | No | Object | Canopy thermal status & CWSI availability block |
| `ndvi` | object | No | Object | Dual-bandpass NoIR camera NDVI status block |
| `ndvi_satellite` | object | No | Object | Sentinel-2 L2A satellite NDVI fallback block |
| `irrigation` | object | No | Object | FAO-56 Hargreaves-Samani crop water requirement block |
| `detections` | array | No | List of detection objects | Individual spatial/temporal plant detections with GPS |
| `gps` | object | No | Object | Pod GPS fix status and accuracy metadata |
| `disease` | array | No | List of disease objects | Active disease diagnoses requiring agronomic attention |
| `pest` | array | No | List of pest objects | Sticky trap pest counts and cumulative ETL evaluations |
| `inputs` | array | No | List of sensor node input status objects | Operational status of all sensor inputs |
| `actions` | array | No | List of advisory action objects | Recommended IPM / agronomic actions from rules engine |

---

## 2. Block-Level Field Specifications

### 2.1 Scan Block (`scan`)
| Field | Type | Nullable | Description |
|---|---|---|---|
| `started_utc` | string | No | Session scan start timestamp |
| `ended_utc` | string | No | Session scan end timestamp |
| `mode` | string | No | Operating mode (`"handheld_pod"`) |
| `frames_captured` | integer | No | Total video frames read from capture source |
| `frames_evaluated` | integer | No | Frames passing quality gates 1–4 |
| `tiles_classified` | integer | No | Total 224x224 tiles passed through inference (`frames * 9`) |
| `distance_walked_m` | float | Yes | Distance traversed during pass in meters (or `null`) |
| `distance_reason` | string | Yes | Reason if distance is unavailable (`"GPS_TRACK_NOT_RECORDED"` or `null`) |

### 2.2 Crop Health Block (`crop_health`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `state` | string | No | `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`, `NO_DATA` | Consensus crop health verdict across cells |
| `reason` | string | Yes | `HIGH_UNCERTAINTY`, `MULTIPLE_CROPS_DETECTED`, `UNCONFIRMED_DETECTIONS`, or `null` | Reason if state is non-definitive |
| `crop` | string | Yes | `"rice"`, `"wheat"`, `"sugarcane"`, or `null` | Dominant diagnosed crop species |
| `frames_evaluated` | integer | No | `>= 0` | Total accepted frames analyzed |
| `frames_agreeing` | integer | No | `>= 0` | Agreement count on top diagnosis |
| `frames_rejected_ood` | integer | No | `>= 0` | Frames rejected by open-set energy gate |
| `frames_rejected_not_crop` | integer | No | `>= 0` | Frames classified as soil/weed/not_crop |
| `frames_uncertain` | integer | No | `>= 0` | Frames with uncertain verdicts |
| `source` | string | No | `"measured"` | Provenance of diagnosis |

### 2.3 Growth Stage Block (`growth_stage`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `crop` | string | Yes | `"rice"`, `"wheat"`, `"sugarcane"`, or `null` | Crop evaluated |
| `stage` | string | Yes | `initial`, `development`, `mid_season`, `late_season`, or `null` | Current phenological stage |
| `stage_code` | string | Yes | `"INI"`, `"DEV"`, `"MID"`, `"LATE"`, or `null` | Short stage code |
| `days_since_planting` | integer | Yes | `>= 0` or `null` | Days elapsed since planting / transplanting |
| `total_cycle_days` | integer | Yes | Typical 120–280 days | Assumed total lifecycle duration |
| `cycle_source` | string | Yes | `"default_assumption"`, `"user_override"` | Source of variety cycle length |
| `cycle_verification_status` | string | Yes | `"RECALLED_UNVERIFIED"` | Provenance of variety cycle length |
| `stage_lengths_days` | object | Yes | Dictionary of stage durations | Durations of initial, development, mid_season, late_season |
| `canopy_cover_measured` | float | Yes | `0.0` to `1.0` or `null` | Fractional canopy cover measured during scan |
| `canopy_cover_expected_range` | array | Yes | Pair of floats `[min, max]` or `null` | Expected canopy cover range for stage |
| `kc` | float | Yes | `0.20` to `1.35` or `null` | FAO-56 Table 12 crop coefficient |
| `status` | string | No | `"OK"`, `"DAYS_SINCE_PLANTING_REQUIRED"`, `"AWAITING_PLANTING_DATE"`, `"UNSUPPORTED_CROP"` | Estimation validity status |
| `reason` | string | Yes | `"DAYS_SINCE_PLANTING_REQUIRED"` or `null` | Reason if stage is unestimated |
| `verification_status` | string | No | `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED` | Four-value frozen provenance status |
| `source` | string | No | `"derived"` | Agronomic model origin |
| `document_reference` | string | No | Text citation | Primary FAO-56 document reference |

### 2.4 Vegetation Block (`vegetation`)

#### 2.4.1 Top-Level Vegetation Fields
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `interpretation_mode` | string | No | `"relative"` | Within-scan relative distribution mode |
| `canopy_cover` | object | No | Dictionary | Green canopy coverage statistics |
| `vari` | object | No | Dictionary | Visible Atmospherically Resistant Index statistics |
| `exg` | object | No | Dictionary | Excess Green index statistics |
| `tgi` | object | No | Dictionary | Triangular Greenness Index statistics |
| `dgci` | object | No | Dictionary | Dark Green Color Index statistics |
| `ndvi` | float / null | Yes | `null` (hardware pending) | Dual-bandpass normalized difference vegetation index |
| `ndvi_status` | string | No | `"GATED_HARDWARE_CALIBRATION"`, `"PENDING_HARDWARE_FINALIZATION"` | Hardware gating status |
| `ndvi_reason` | string | Yes | Text explanation | Explanation of hardware reservation |

#### 2.4.2 Canopy Cover Sub-Block (`vegetation.canopy_cover`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `mean` | float | No | `0.0` to `1.0` | Mean fractional green canopy coverage across valid frames |
| `p10` | float | No | `0.0` to `1.0` | 10th percentile canopy coverage |
| `p50` | float | No | `0.0` to `1.0` | Median (50th percentile) canopy coverage |
| `p90` | float | No | `0.0` to `1.0` | 90th percentile canopy coverage |
| `min_fraction_threshold` | float | No | `0.10` | Minimum canopy fraction required for reliable index calculation |
| `status` | string | No | `"OK"`, `"INSUFFICIENT_CANOPY"` | Canopy coverage sufficiency status |
| `threshold_source` | string | No | Text string | Origin note for provisional canopy threshold |
| `threshold_confirmed` | boolean | No | `false` | Calibration confirmation flag |
| `source` | string | No | `"measured"` | Provenance label |

#### 2.4.3 VARI Sub-Block (`vegetation.vari`)
When canopy fraction is sufficient (`mean_canopy >= min_fraction_threshold`):
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `band` | string | No | `"LOWER_TAIL"`, `"BELOW_TYPICAL"`, `"TYPICAL"`, `"ABOVE_TYPICAL"` | Within-scan relative distribution band |
| `band_basis` | string | No | `"within_scan_percentile"` | Basis used for relative VARI band classification |
| `field_median` | float | No | Numeric | Scan median VARI value |
| `mean` | float | Yes | Numeric or `null` | Mean VARI across canopy-covered pixels |
| `p10` | float | Yes | Numeric | 10th percentile VARI |
| `p50` | float | Yes | Numeric | 50th percentile VARI |
| `p90` | float | Yes | Numeric | 90th percentile VARI |
| `source` | string | No | `"measured"` | Provenance label |
| `threshold_confirmed` | boolean | No | `false` | Calibration confirmation flag |
| `threshold_source` | string | No | Text string | Explanatory note: relative within-scan distribution |

*(Note: If canopy fraction is insufficient, `mean` is `null`, `reason` is `"INSUFFICIENT_CANOPY_FRACTION"`, `min_fraction_threshold` is `0.10`, `threshold_source` and `threshold_confirmed` are preserved, and `source` is `"measured"`).*

#### 2.4.4 ExG Sub-Block (`vegetation.exg`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `mean` | float | Yes | Numeric or `null` | Mean Excess Green index across valid canopy pixels |
| `source` | string | No | `"measured"` | Provenance label |
| `threshold_confirmed` | boolean | No | `false` | Calibration confirmation flag |
| `threshold_source` | string | No | Text string | Provisional threshold provenance |

#### 2.4.5 TGI Sub-Block (`vegetation.tgi`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `mean` | float | Yes | Numeric or `null` | Mean Triangular Greenness Index across canopy pixels |
| `source` | string | No | `"measured"` | Provenance label |
| `threshold_confirmed` | boolean | No | `false` | Calibration confirmation flag |
| `threshold_source` | string | No | Text string | Provisional regression provenance |

#### 2.4.6 DGCI Sub-Block (`vegetation.dgci`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `mean` | float | Yes | Numeric or `null` | Mean Dark Green Color Index in `[0, 1]` |
| `out_of_domain_fraction` | float | No | `0.0` to `1.0` | Fraction of pixels with hue outside `[60°, 120°]` |
| `source` | string | No | `"measured"` | Provenance label |
| `threshold_confirmed` | boolean | No | `false` | Calibration confirmation flag |
| `threshold_source` | string | No | Text string | Narrowed foliage domain provenance |

### 2.5 Hardware-Gated & Environmental Blocks (`thermal`, `ndvi`, `ndvi_satellite`, `irrigation`)

#### Thermal Block (`thermal`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `available` | boolean | No | `true`, `false` | True when MLX90640 frame captured and references valid |
| `reason` | string | Yes | `THERMAL_REFS_NOT_CONFIGURED`, `INSUFFICIENT_REFERENCE_GAP`, etc. | Explanation if CWSI unavailable |
| `tc_c` | float | Yes | Numeric (°C) | Median canopy temperature (°C) measured directly from thermal array (populated whenever a valid thermal frame is captured, even when wet/dry references are unconfigured) |
| `twet_c` | float | Yes | Numeric (°C) | Median temperature of wet reference pad |
| `tdry_c` | float | Yes | Numeric (°C) | Median temperature of dry reference pad |
| `cwsi` | float | Yes | Numeric | Raw Crop Water Stress Index $(T_c - T_{wet})/(T_{dry} - T_{wet})$ |
| `flag` | string | Yes | `"NORMAL"`, `"CWSI_BELOW_ZERO"`, `"CWSI_ABOVE_ONE"` | Out-of-bounds flag (unclamped reporting) |
| `thermal_source` | string | No | `"hardware"`, `"mock"` | Origin of thermal data |
| `frame_utc` | string | Yes | ISO-8601 UTC string or `null` | Capture timestamp of thermal frame |

#### NDVI Hardware Probe Block (`ndvi`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `available` | boolean | No | `true`, `false` | True when NoIR camera is detected on CSI port 1 |
| `reason` | string | Yes | Text explanation or `null` | Status reason if camera absent or uncalibrated |

#### Sentinel-2 Satellite NDVI Block (`ndvi_satellite`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `available` | boolean | No | `true`, `false` | True when a clear Sentinel-2 scene is cached |
| `reason` | string | Yes | `NO_SATELLITE_DATA_RECORDED`, `NO_CLEAR_SCENE`, `CREDENTIALS_MISSING`, etc. | Explanation if unavailable |
| `source` | string | No | `"SENTINEL2_L2A_CDSE"` | Origin label for satellite data |
| `scene_date` | string | Yes | ISO Date string (`YYYY-MM-DD`) | Acquisition date of satellite scene |
| `age_days` | float | Yes | `>= 0.0` | Elapsed days since scene acquisition |
| `ndvi_mean` | float | Yes | `-1.0` to `1.0` | Field polygon mean NDVI |
| `ndvi_std` | float | Yes | `>= 0.0` | Field polygon standard deviation |
| `valid_pixel_count` | integer | Yes | `>= 0` | Cloud-free valid 10m pixels evaluated |
| `cloud_masked_fraction` | float | Yes | `0.0` to `1.0` | Fractional cloud/shadow mask coverage |
| `pixel_size_m` | integer | No | `10` | Ground sample distance (10 meters) |
| `reliability_note` | string | Yes | `"UNRELIABLE_SMALL_FIELD (<9 pixels / ~30x30m footprint)"` or `null` | Small field warning |

#### FAO-56 Irrigation Block (`irrigation`)
| Field | Type | Nullable | Allowed Values / Range | Description |
|---|---|---|---|---|
| `available` | boolean | No | `true`, `false` | True when ground mast temperature history is sufficient |
| `reason` | string | Yes | Text reason | Explanation if irrigation computation unavailable |
| `method` | string | Yes | `"fao56_hargreaves_samani"` | Irrigation calculation method |
| `air_temp_c` | float | Yes | `-10.0` to `60.0` | Instantaneous air temperature (°C) |
| `rh_pct` | float | Yes | `0.0` to `100.0` | Relative humidity (%) |
| `t_min_24h_c` | float | Yes | Minimum diurnal air temp (°C) | 24-hour minimum air temperature |
| `t_max_24h_c` | float | Yes | Maximum diurnal air temp (°C) | 24-hour maximum air temperature |
| `t_mean_24h_c` | float | Yes | Mean diurnal air temp (°C) | 24-hour mean air temperature |
| `ra_mj_m2_day` | float | Yes | Numeric | Dynamic FAO-56 Eq. 21 extraterrestrial radiation (MJ/m²/day) |
| `ra_mm_day` | float | Yes | Numeric | $R_a \times 0.408$ equivalent depth (mm/day) |
| `ra_source` | string | Yes | `"GPS"`, `"CONFIG_LATITUDE"` | Origin of latitude coordinate for $R_a$ |
| `ra_latitude_deg` | float | Yes | `-90.0` to `90.0` | Latitude used for $R_a$ computation |
| `day_of_year` | integer | Yes | `1` to `366` | Day of year ($J$) |
| `et0_mm_day` | float | Yes | `0.0` to `15.0` | Reference evapotranspiration ($ET_0$) |
| `kc` | float | Yes | `0.20` to `1.35` | Crop coefficient |
| `crop_et_mm_day` | float | Yes | `>= 0.0` | Crop ET ($ET_c = ET_0 \times K_c$) |
| `soil1_v` | float | Yes | `0.0` to `3.3` | Soil moisture sensor 1 voltage |
| `soil2_v` | float | Yes | `0.0` to `3.3` | Soil moisture sensor 2 voltage |
| `battery_v` | float | Yes | `0.0` to `5.0` | Mast node battery voltage |
| `samples_24h` | integer | Yes | `>= 6` | Readings count in 24h window |
| `source` | string | Yes | `"derived_fao56"` | Provenance label |

### 2.6 Trap Pest Block (`pest[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `target_pest_context` | string | No | Taxon identifier | Evaluated pest species context |
| `count_basis` | string | No | `"watershed_all_blobs"`, `"direct_count"` | Blob counting basis (watershed primary) |
| `count_observed` | float | No | `>= 0.0` | Total observed sticky trap count |
| `days_monitored` | float | No | `1.0` to `7.0` | Days card has been deployed |
| `daily_rate` | float | No | `>= 0.0` | Observed insects per trap daily rate |
| `threshold_value` | float | Yes | Numeric or `null` | Published ICAR/NIPHM economic threshold |
| `threshold_unit` | string | Yes | `"insects_per_trap"`, or `null` | Unit of published threshold |
| `threshold_available` | boolean | No | `true`, `false` | Explicit indicator whether numeric threshold exists |
| `status` | string | No | `BELOW_ETL`, `AT_ETL`, `ABOVE_ETL`, `NO_PUBLISHED_ETL`, `NOT_SAMPLED_BY_STICKY_TRAP`, `UNKNOWN_PEST`, `CARD_SATURATED`, `INVALID_MONITORING_WINDOW`, `MISSING_DEPLOYMENT_TIMESTAMP` | Operational ETL comparison status |
| `threshold_verification_status` | string | No | `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED` | Four-value verification status of threshold |
| `classification_verification_status`| string| No| `"RECALLED_UNVERIFIED"` | Classification provenance status |
| `total_blobs_counted` | integer | Yes | `>= 0` | Deterministic watershed blob count |
| `classification_source` | string | Yes | `"CROSS_DOMAIN_PRETRAINED"` | Model B origin label |

### 2.7 Detections Block (`detections[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `class` | string | No | 29 Model A classes | Diagnosed class label (e.g. `"rice__blast"`) |
| `confidence` | float | No | `0.0` to `1.0` | Softmax probability |
| `cross_source_reliability` | string | No | `TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED` | Cross-source evaluation tier |
| `lat` | float | Yes | Numeric coordinate or `null` | Detection GPS latitude |
| `lon` | float | Yes | Numeric coordinate or `null` | Detection GPS longitude |
| `fix_quality` | integer | Yes | `0`, `1`, `2` or `null` | GPS fix quality (`0`=invalid, `1`=GPS, `2`=DGPS) |
| `hdop` | float | Yes | `>= 0.0` or `null` | Horizontal Dilution of Precision |
| `captured_utc` | string | No | ISO-8601 UTC string | Timestamp of frame capture |
| `source` | string | No | `"measured"` | Provenance of detection |

### 2.8 GPS Scan Summary Block (`gps`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `OK`, `ABSENT` | GPS overall coverage status during walk scan |
| `point_count` | integer | No | `>= 0` | Number of geotagged frames recorded in scan |
| `accuracy_note` | string | No | Text disclaimer | Position precision disclaimer ("Point tagging only, approximately 2.5 m CEP. Not a survey-grade position.") |

### 2.9 Disease Diagnoses Block (`disease[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `class` | string | No | Non-healthy Model A classes | Diagnosed disease class label (e.g. `"rice__blast"`) |
| `confidence` | float | No | `0.0` to `1.0` | Softmax probability |
| `media_ids` | array | No | Empty list `[]` | Associated image IDs (retained empty per media retention pruning policy) |
| `source` | string | No | `"measured"` | Provenance of diagnosis |

### 2.10 Actions Block (`actions[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `rank` | integer | No | `>= 1` | Action priority rank |
| `template_id` | string | No | 21 template IDs | Deterministic action template identifier |
| `action` | string | No | <= 240 chars | English action directive |
| `rationale` | string | No | <= 400 chars | Agronomic explanation and threshold comparison |
| `params` | object | No | Dictionary | Parameters for deterministic offline translation |
| `verification_status` | string | No | `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED` | Four-value citation verification status |
| `url` | string | Yes | URL string or `null` | Primary legal / agronomic regulatory source URL |
| `document_reference` | string | Yes | Text string or `null` | Exact publication / bulletin reference citation |
| `offline_source_file` | string | Yes | Path or `null` | Local PDF / Markdown archive citation in `docs/sources/` |
| `confidence` | string | No | `"high"`, `"medium"`, `"low"` | Action confidence tier |
| `advisory_only` | boolean | No | `true` | Explicit disclaimer: advisory recommendation only |
| `generated_by` | string | No | `"template"` | Generator provenance tag (ALWAYS `"template"`) |
| `source` | string | No | `"derived"` | Agronomic origin label |

### 2.11 Sensor Inputs Block (`inputs[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `name` | string | No | `"pod_camera_rgb"`, `"pod_gps"`, `"pod_thermal"` | Sensor input stream identifier |
| `source_node` | string | No | `"POD"`, `"MAST"` | Physical hardware node hosting the sensor |
| `status` | string | No | `OK`, `PENDING_CALIBRATION`, `MOCK_PROVISIONAL`, `ABSENT` | Operational hardware and calibration status (`OK`: operational; `PENDING_CALIBRATION`: sensor hardware present and functional but derived metric unavailable pending calibration; `MOCK_PROVISIONAL`: simulated data; `ABSENT`: hardware disconnected/missing) |

---

## 3. Ground Mast Telemetry Pull Record Schema (Guide §6)

The ground mast ESP32 acts as an HTTP server (`GET /readings?since=&limit=`) pulled by the Jetson Nano client.

| Field | Type | Required | Allowed Values / Range | Description |
|---|---|---|---|---|
| `log_epoch` | integer | Yes | Monotonic boot timestamp | Epoch counter to distinguish reboots |
| `seq` | integer | Yes | `>= 1` | Monotonic sequential reading ID |
| `node_id` | string | Yes | `"SIH-NODE-01"` | Ground mast node identifier |
| `field_id` | string | No | String (e.g. `"F01"`) or null | Field identifier |
| `utc` | string | No | ISO-8601 UTC string or null | RTC timestamp if synchronized |
| `rtc_valid` | boolean | Yes | `true`, `false` | RTC time validity flag |
| `uptime_s` | integer | No | `>= 0` | ESP32 uptime in seconds |
| `air_temp_c` | float | No | `-10.0` to `60.0` | SHT40 air temperature in °C |
| `rh_pct` | float | No | `0.0` to `100.0` | SHT40 relative humidity in % |
| `ir_object_c` | float | No | `-10.0` to `80.0` | MLX90614 surface temperature in °C |
| `ir_ambient_c`| float | No | `-10.0` to `60.0` | MLX90614 ambient temperature in °C |
| `lux` | float | No | `0.0` to `120000.0` | BH1750 ambient illuminance |
| `soil1_v` | float | No | `0.0` to `3.3` | ADS1115 soil moisture probe 1 voltage |
| `soil2_v` | float | No | `0.0` to `3.3` | ADS1115 soil moisture probe 2 voltage |
| `battery_v` | float | No | `0.0` to `5.0` | LiFePO4 battery voltage |
| `status` | object | No | Subsystem status dictionary | Sensor health bits |
| `received_at` | string | Yes | ISO-8601 UTC string | Timestamp recorded by Nano |

---

## 4. Frozen Code Constant Registries

### 4.1 Model A Diagnostic Classes (29 classes)
`rice__normal`, `rice__bacterial_leaf_blight`, `rice__bacterial_leaf_streak`, `rice__bacterial_panicle_blight`, `rice__blast`, `rice__brown_spot`, `rice__downy_mildew`, `rice__tungro`, `rice__hispa`, `rice__leaf_roller`, `rice__yellow_stem_borer`, `sugarcane__healthy`, `sugarcane__dried_leaf`, `sugarcane__mosaic`, `sugarcane__red_rot`, `sugarcane__rust`, `sugarcane__yellow_leaf`, `sugarcane__smut`, `sugarcane__pokkah_boeng`, `sugarcane__grassy_shoot`, `sugarcane__brown_spot`, `sugarcane__banded_chlorosis`, `sugarcane__sett_rot`, `wheat__healthy`, `wheat__yellow_rust`, `wheat__brown_rust`, `wheat__septoria`, `wheat__powdery_mildew`, `not_crop`

### 4.2 Model B Sticky-Trap Morphological Classes
`small_pale_winged`, `larger_insect`, `debris`, `UNCERTAIN_NON_TARGET`

### 4.3 Action Template IDs (21 templates)
`ACT_EXT_OFFICER_CONSULT`, `ACT_IRRIGATE_WATER_DEFICIT`, `ACT_MAINTAIN_ROUTINE`, `ACT_MULTICROP_INVESTIGATE`, `ACT_RESCAN_AMBIGUOUS`, `ACT_TREAT_RICE_BLAST`, `ACT_TREAT_RICE_BLIGHT`, `ACT_TREAT_RICE_BROWN_SPOT`, `ACT_TREAT_RICE_HISPA`, `ACT_TREAT_RICE_LEAF_ROLLER`, `ACT_TREAT_RICE_OTHER_DISEASE`, `ACT_TREAT_RICE_STEM_BORER`, `ACT_TREAT_RICE_TUNGRO`, `ACT_TREAT_SUGARCANE_POKKAH_BOENG`, `ACT_TREAT_SUGARCANE_RED_ROT`, `ACT_TREAT_SUGARCANE_RUST`, `ACT_TREAT_SUGARCANE_SMUT`, `ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC`, `ACT_TREAT_WHEAT_BROWN_RUST`, `ACT_TREAT_WHEAT_POWDERY_MILDEW`, `ACT_TREAT_WHEAT_YELLOW_RUST`

### 4.4 Cross-Source Reliability Tiers (4 tiers)
`TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED`

### 4.5 Frozen Verification Status Enum (4 values)
`VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`

### 4.6 Established Sticky-Trap ETL Comparison Statuses
`BELOW_ETL`, `AT_ETL`, `ABOVE_ETL`

### 4.7 Sensor Input Status Enum (4 values)
`OK`, `PENDING_CALIBRATION`, `MOCK_PROVISIONAL`, `ABSENT`

---

## 5. Gateway Endpoint Response Schemas

### 5.1 Health Endpoint (`GET /api/v1/health`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `device` | string | No | `"sih-pod-01"` | Handheld Nano Pod device identifier |
| `schema_version` | string | No | `"1.0"` | Wire schema version |
| `server_time_utc` | string | No | ISO-8601 UTC string | Current system clock timestamp on the Nano |
| `gps_time_valid` | boolean | No | `true`, `false` | Whether system clock is locked to GPS or RTC |
| `clock_source` | string | No | `"gps"`, `"rtc"`, `"filesystem"` | Source of system clock synchronization |
| `advisory_count` | integer | No | `>= 0` | Total number of advisories stored in SQLite database |
| `latest_seq` | integer | No | `>= 0` | Highest monotonic advisory sequence number |
| `storage_free_kb` | integer | No | `>= 0` | Free disk space available on storage partition (KB) |
| `syncing` | boolean | No | `true`, `false` | AP/STA mode-switch status flag (true when syncing with ground mast) |
| `sync_state` | string | No | `"IDLE"`, `"STA_SYNC"` | AP/STA state seam identifier |

### 5.2 Mast Sync Status Endpoint (`GET /api/v1/sync/status`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `sync_in_progress` | boolean | No | `true`, `false` | Whether a background mast sync pull is currently executing |
| `last_success_utc` | string | Yes | ISO-8601 UTC string or `null` | ISO-8601 UTC timestamp of last successful sync cycle |
| `last_attempt_utc` | string | Yes | ISO-8601 UTC string or `null` | ISO-8601 UTC timestamp of last sync attempt |
| `last_result` | string | Yes | `"OK"`, `"MAST_NOT_FOUND"`, `"PARTIAL"`, `"ERROR"`, or `null` | Result code of last sync cycle |
| `mast_data_age_s` | integer | Yes | `>= 0` or `null` | Age of latest ingested mast telemetry record in seconds |
| `records_pulled` | integer | No | `>= 0` | Count of telemetry records pulled in last sync cycle |
| `trap_images_pulled` | integer | No | `>= 0` | Count of sticky-trap images pulled in last sync cycle |

### 5.3 Advisory Manifest Endpoint (`GET /api/v1/manifest?since=&limit=`)
#### Top-Level Response
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire schema version |
| `count` | integer | No | `>= 0` | Number of advisory items returned in this page |
| `advisories` | array | No | List of summary objects | Monotonically ordered advisory catalog items |
| `truncated` | boolean | No | `true`, `false` | True if additional newer advisories exist beyond limit |

#### Advisory Summary Object (`advisories[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `advisory_id` | string | No | String identifier | Unique advisory document identifier |
| `seq` | integer | No | `>= 1` | Monotonic sequential advisory sequence number |
| `generated_at_utc` | string | No | ISO-8601 UTC string | Timestamp of advisory generation |
| `bytes` | integer | No | `>= 0` | Size of raw JSON advisory document in bytes |
| `replay` | boolean | No | `true`, `false` | Provenance label (`true` for replay, `false` for live) |
| `inference_backend` | string | No | `"trt"`, `"onnx"`, `"mock"` | Model A inference backend used |

### 5.4 Acknowledgment Endpoint (`POST /api/v1/ack`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `"ok"` | Acknowledgement execution status |
| `acked` | string / integer | No | String or integer | Cursor or advisory identifier acknowledged |

### 5.5 Media Pruning Endpoint (`GET /api/v1/media/<id>`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `error` | string | No | `"gone"` | HTTP 410 error identifier |
| `reason` | string | No | `"retention_pruned"` | Explanation of media lifecycle pruning policy |

### 5.6 Mast Sync Trigger Endpoint (`POST /api/v1/sync/trigger`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `"accepted"` | Trigger acceptance status (returns 202) |
| `timestamp` | string | No | ISO-8601 UTC string | Trigger initiation timestamp |
| `expected_ap_downtime_s` | integer | No | `30` | Expected AP downtime during STA mode switch |

### 5.7 Sticky Trap Photo Upload Endpoint (`POST /api/v1/trap/upload`)

> [!NOTE]
> **Hardware Status:** Not yet exercised on real Jetson Nano device (Model B pipeline verified in unit tests and offline simulation).

#### Request
- **Method:** `POST`
- **Path:** `/api/v1/trap/upload?trap_id=<id>&days=<days>&scale=<scale>&allow_provisional=<bool>`
- **Query / Headers:** `X-Trap-Id`, `X-Days-Monitored`, `X-Scale`, `X-Allow-Provisional`
- **Body:** Binary JPEG or `multipart/form-data` with image file

#### Success Response (`200 OK`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `"ok"` | Upload and processing status |
| `job_id` | string | No | String identifier | Unique background trap processing job ID |
| `trap_id` | string | No | String identifier | Sticky trap card identifier (e.g. `"TRAP_01"`) |
| `record_id` | integer | No | `>= 1` | SQLite `sticky_traps` database row ID |
| `advisory_id` | string | Yes | String ID or `null` | ID of updated/synthesized advisory document |
| `total_blobs_counted` | integer | No | `>= 0` | Total morphological blobs segmented |
| `scale_status` | string | No | `"CALIBRATED"`, `"PROVISIONAL"`, `"UNMEASURED"` | Pixel scale calibration status |
| `scale_mm_per_pixel` | float | Yes | Numeric or `null` | Scale factor in mm per pixel |
| `etl_status` | string | No | `"BELOW_ETL"`, `"AT_ETL"`, `"ABOVE_ETL"`, `"NO_PUBLISHED_ETL"`, `"UNKNOWN_PEST"` | Economic threshold comparison status |
| `pest` | array | No | List of pest objects | Model B pest evaluation block (matches `pest[]` schema) |

---

## 6. Gateway HTTP Error Responses

All error responses return `application/json; charset=utf-8` with a standardized JSON envelope:

### 6.1 `400 Bad Request`
Returned when headers, query parameters, or payload violate API constraints:
- **Missing Content-Length:**
  ```json
  {"error": "bad_request", "detail": "Missing Content-Length header"}
  ```
- **Malformed JSON Payload:**
  ```json
  {"error": "bad_request", "detail": "Malformed JSON: <parser error>"}
  ```
- **Missing Required Identifier in Ack Payload:**
  ```json
  {"error": "bad_request", "detail": "Missing 'upto' or 'advisory_id' field in payload"}
  ```
- **Invalid Manifest Pagination Parameters:**
  ```json
  {"error": "bad_request", "detail": "limit must be an integer"}
  ```
  ```json
  {"error": "bad_request", "detail": "limit must be >= 1"}
  ```
- **Uncalibrated Trap Image Scale (without provisional override):**
  ```json
  {"error": "scale_uncalibrated", "detail": "<scale error explanation>"}
  ```

### 6.2 `403 Forbidden`
Returned in production mode (`--allow-mock` not set) when an advisory generated by the simulated/mock inference backend is requested:
```json
{
  "error": "mock_advisory_rejected",
  "detail": "Gateway running in production mode rejecting mock advisory"
}
```

### 6.3 `404 Not Found`
- **Unknown Advisory Document (by UUID string or sequence integer):**
  ```json
  {
    "error": "not_found",
    "advisory_id": "adv_unknown_999"
  }
  ```
- **Unmatched Route:**
  ```json
  {
    "error": "not_found",
    "path": "/api/v1/nonexistent"
  }
  ```

### 6.4 `409 Conflict`
Returned by `POST /api/v1/sync/trigger` when a ground mast synchronization cycle is already actively executing:
```json
{
  "error": "sync_in_progress",
  "detail": "Mast synchronization is already running"
}
```

### 6.5 `410 Gone`
Returned by `GET /api/v1/media/<id>` for raw capture images, indicating that high-resolution assets have been retention-pruned per device storage policy:
```json
{
  "error": "gone",
  "reason": "retention_pruned"
}
```

### 6.6 `503 Service Unavailable`
Returned during the initial device boot window before pipeline services and database readiness are established:
- **HTTP Header:** `Retry-After: 5`
- **Body:**
  ```json
  {
    "error": "not_ready"
  }
  ```
