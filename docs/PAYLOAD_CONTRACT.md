# Payload Contract Specification (v1.0)

**Document:** `docs/PAYLOAD_CONTRACT.md`  
**Purpose:** Authoritative wire schema contract for all advisory JSON payloads emitted by `GET /api/v1/advisory/<id>` and telemetry endpoints.  
**Generation:** Automatically generated and verified from code constants via `scripts/generate_payload_contract.py` and tested by `tests/test_payload_contract.py`.

---

## 1. Top-Level Advisory Schema Fields

| Field Path | Type | Nullable | Allowed Values / Range | Description |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire contract schema version |
| `advisory_id` | string | No | String identifier (e.g. `2026-09-17T06:30:00Z_field1`) | Unique advisory document identifier |
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
| `frames_evaluated` | integer | No | Frames passing quality gates 1-4 |
| `tiles_classified` | integer | No | Total 320x320 tiles passed through inference (frames * 9) |
| `distance_walked_m` | float | Yes | Distance traversed during pass (meters) |
| `distance_reason` | string | Yes | Reason if distance is unavailable (`"GPS_TRACK_NOT_RECORDED"`) |

### 2.2 Crop Health Block (`crop_health`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `state` | string | No | `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`, `NO_DATA` | Consensus crop health verdict across cells |
| `reason` | string | Yes | `HIGH_UNCERTAINTY`, `MULTIPLE_CROPS_DETECTED`, or `null` | Reason if state is non-definitive |
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
| `crop` | string | Yes | `"rice"`, `"wheat"`, `"sugarcane"` | Crop evaluated |
| `stage` | string | Yes | `initial`, `development`, `mid_season`, `late_season`, or `null` | Current phenological stage |
| `stage_code` | string | Yes | `"INI"`, `"DEV"`, `"MID"`, `"LATE"`, or `null` | Short stage code |
| `days_since_planting` | integer | Yes | `>= 0` | Days elapsed since planting / transplanting |
| `total_cycle_days` | integer | Yes | Typical 120-280 days | Assumed total lifecycle duration |
| `kc` | float | Yes | `0.20` to `1.35` | FAO-56 Table 12 crop coefficient |
| `status` | string | No | `"OK"`, `"DAYS_SINCE_PLANTING_REQUIRED"`, `"UNSUPPORTED_CROP"` | Estimation validity status |
| `verification_status` | string | No | `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED` | Four-value frozen provenance status |
| `source` | string | No | `"derived"` | Agronomic model origin |

### 2.4 Vegetation Block (`vegetation`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `interpretation_mode` | string | No | `"relative"` | Within-scan relative distribution mode |
| `canopy_cover.mean` | float | Yes | `0.0` to `1.0` | Fractional green canopy coverage |
| `vari.band` | string | Yes | `LOWER_TAIL`, `BELOW_TYPICAL`, `TYPICAL`, `ABOVE_TYPICAL` | Relative within-scan greenness band |
| `exg.mean` | float | Yes | Numeric | Excess Green index mean |
| `tgi.mean` | float | Yes | Numeric | Triangular Greenness Index mean |
| `dgci.mean` | float | Yes | `0.0` to `1.0` | Dark Green Color Index mean |
| `ndvi` | float / null | Yes | `null` (hardware pending) | Dual-bandpass normalized difference vegetation index |
| `ndvi_status` | string | No | `"GATED_HARDWARE_CALIBRATION"`, `"PENDING_HARDWARE_FINALIZATION"` | Hardware gating status |

### 2.5 Hardware-Gated Blocks (`thermal`, `ndvi`, `irrigation`)
| Block | Field | Type | Value When Hardware Absent | Description |
|---|---|---|---|---|
| `thermal` | `available` | boolean | `false` | `false` when MLX90640 thermal array is absent |
| `thermal` | `reason` | string | Citation to PENDING_HARDWARE.md Subsystem 1 | Machine-readable availability explanation |
| `ndvi` | `available` | boolean | `false` | `false` when IMX219-77IR + DB660/850 is absent |
| `ndvi` | `reason` | string | Citation to PENDING_HARDWARE.md Subsystem 4 | Machine-readable availability explanation |
| `irrigation` | `available` | boolean | `false` (unless mast data present) | FAO-56 Hargreaves-Samani ET0 availability |
| `irrigation` | `method` | string | `"fao56_hargreaves_samani"` (when available) | ET0 calculation methodology |
| `irrigation` | `t_min_24h_c`| float | Minimum air temp over 24h window | Diurnal air temp min (°C) |
| `irrigation` | `t_max_24h_c`| float | Maximum air temp over 24h window | Diurnal air temp max (°C) |
| `irrigation` | `t_mean_24h_c`| float| Mean air temp over 24h window | Diurnal air temp mean (°C) |
| `irrigation` | `ra_mj_m2_day`| float | Dynamic FAO-56 Eq. 21 extraterrestrial radiation (MJ/m²/day) |
| `irrigation` | `ra_mm_day`| float | $R_a \times 0.408$ equivalent depth in mm/day |
| `irrigation` | `ra_source`| string | `"GPS"`, `"CONFIG_LATITUDE"` | Source of latitude for $R_a$ computation |
| `irrigation` | `ra_latitude_deg`| float | Latitude used for $R_a$ calculation (°N/°S) |
| `irrigation` | `day_of_year` | integer | 1 to 366 | Day of year ($J$) used for $R_a$ solar geometry |
| `irrigation` | `et0_mm_day`| float | `0.0` to `15.0` mm/day | Reference evapotranspiration |
| `irrigation` | `kc` | float | `0.20` to `1.35` | Growth-stage crop coefficient |
| `irrigation` | `crop_et_mm_day`| float| `>= 0.0` mm/day | Crop evapotranspiration ($ET_c = ET_0 \times K_c$) |
| `irrigation` | `samples_24h`| integer | `>= 6` | Valid air temperature readings in 24h |

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
| `cell_id` | string | No | String identifier (e.g. `"tile_r0_c0"`) | Spatial grid cell identifier |
| `class_name` | string | No | 29 Model A classes | Diagnosed class label |
| `confidence` | float | No | `0.0` to `1.0` | Softmax probability |
| `energy` | float | No | Numeric | Open-set energy score |
| `state` | string | No | `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP` | Cell health classification |
| `cross_source_reliability` | string | No | `TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED` | Cross-source evaluation tier |
| `latitude` | float | Yes | Numeric coordinate or `null` | Detection GPS latitude |
| `longitude` | float | Yes | Numeric coordinate or `null` | Detection GPS longitude |

### 2.8 Actions Block (`actions[]`)
| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `rank` | integer | `>= 1` | Action priority rank |
| `template_id` | string | `ACT_EXT_OFFICER_CONSULT`, `ACT_IRRIGATE_WATER_DEFICIT`, `ACT_MAINTAIN_ROUTINE`, `ACT_MULTICROP_INVESTIGATE`, `ACT_RESCAN_AMBIGUOUS`, `ACT_TREAT_RICE_BLAST`, `ACT_TREAT_RICE_BLIGHT`, `ACT_TREAT_RICE_BROWN_SPOT`, `ACT_TREAT_RICE_HISPA`, `ACT_TREAT_RICE_LEAF_ROLLER`, `ACT_TREAT_RICE_OTHER_DISEASE`, `ACT_TREAT_RICE_STEM_BORER`, `ACT_TREAT_RICE_TUNGRO`, `ACT_TREAT_SUGARCANE_POKKAH_BOENG`, `ACT_TREAT_SUGARCANE_RED_ROT`, `ACT_TREAT_SUGARCANE_RUST`, `ACT_TREAT_SUGARCANE_SMUT`, `ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC`, `ACT_TREAT_WHEAT_BROWN_RUST`, `ACT_TREAT_WHEAT_POWDERY_MILDEW`, `ACT_TREAT_WHEAT_YELLOW_RUST` | Deterministic action template identifier |
| `action` | string | <= 240 chars | English action directive |
| `rationale` | string | <= 400 chars | Agronomic explanation and threshold comparison |
| `params` | object | Dictionary | Parameters for deterministic offline translation |
| `verification_status` | string | `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED` | Four-value citation verification status |
| `offline_source_file` | string | Path | Local PDF archive citation |
| `advisory_only` | boolean | `true` | Explicit disclaimer: advisory recommendation only |

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

