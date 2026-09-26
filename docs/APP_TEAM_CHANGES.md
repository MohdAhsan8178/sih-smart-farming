# App Team Handoff & Payload Changes Specification

**Last Updated:** 21 September 2026  
**Document:** `docs/APP_TEAM_CHANGES.md`  
**Supersedes:** `data_flow_architecture.md` field definitions and `ans_for_vitthal.md` legacy draft items.  
**Authoritative Reference:** `docs/PAYLOAD_CONTRACT.md` (validated by `tests/test_payload_contract.py`).

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

## 1. Summary of Architecture & Data Flow

This document details all schema, telemetry, and contract changes between the initial `data_flow_architecture.md` draft and the finalized edge runtime implementation.

```
                    ┌──────────────────────────────────────────────┐
                    │            Handheld Jetson Nano Pod          │
                    │                                              │
                    │   • Model A Classifier (29 Classes)          │
                    │   • Model B Sticky-Trap Classifier (3 Class) │
                    │   • Rules Engine (21 Action Templates)       │
                    │   • SQLite WAL Storage (edge.db)             │
                    │   • Local HTTP Gateway (:8080)               │
                    └──────────────┬───────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
         (PULL over WiFi)                      (REST API)
                │                                     │
                ▼                                     ▼
     ┌───────────────────────┐             ┌─────────────────────┐
     │ Ground Mast SIH-NODE  │             │ Farmer Mobile App   │
     │ 192.168.9.1 (ESP32)   │             │ (Offline Gateway)   │
     │ GET /readings         │             │ GET /api/v1/health  │
     │ GET /trap/image       │             │ GET /api/v1/advisory│
     └───────────────────────┘             └─────────────────────┘
```

---

## 2. Complete Wire Contract Fields (`GET /api/v1/advisory/<id_or_seq>`, `GET /api/v1/advisory/latest`)

The advisory document emitted by `GET /api/v1/advisory/<id_or_seq>` and `GET /api/v1/advisory/latest` adheres to schema version `1.0`.

### 2.1 Top-Level Fields

| Field Path | Type | Nullable | Values / Format | Purpose |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire contract schema version |
| `advisory_id` | string | No | String (e.g. `2026-09-19T10:00:00Z_F01`) | Unique advisory document identifier |
| `seq` | integer | No | `>= 1` | Monotonic sequential advisory sequence number assigned by SQLite rowid |
| `generated_at_utc` | string | No | ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`) | Timestamp of advisory synthesis |
| `inference_backend` | string | No | `"trt"`, `"onnx"`, `"mock"` | Explicit runtime inference engine backend. Guarantees no silent fallback occurred. |
| `replay` | boolean | No | `true`, `false` | Provenance label (`true` for seeded replay / demo data; `false` for live capture) |
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

### 2.2 Detailed Subsystem Blocks

#### 1. Scan Block (`scan`)
- `started_utc`: ISO-8601 UTC scan start timestamp
- `ended_utc`: ISO-8601 UTC scan end timestamp
- `mode`: Operating mode (`"handheld_pod"`)
- `frames_captured`: Total video frames read from capture source
- `frames_evaluated`: Frames passing quality gates 1–4
- `tiles_classified`: Total 224x224 tiles passed through inference (`frames * 9`)
- `distance_walked_m`: Traversed distance in meters (float or `null`)
- `distance_reason`: Reason if distance unavailable (`"GPS_TRACK_NOT_RECORDED"` or `null`)

#### 2. Crop Health Block (`crop_health`)
- `state`: Consensus health verdict across grid cells: `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`, `NO_DATA`
- `reason`: Explanation if non-definitive: `HIGH_UNCERTAINTY`, `MULTIPLE_CROPS_DETECTED`, `UNCONFIRMED_DETECTIONS`, or `null`
- `crop`: Dominant diagnosed crop species: `"rice"`, `"wheat"`, `"sugarcane"`, or `null`
- `frames_evaluated`: Total accepted frames analyzed
- `frames_agreeing`: Agreement count on top diagnosis
- `frames_rejected_ood`: Frames rejected by open-set energy gate
- `frames_rejected_not_crop`: Frames classified as soil/weed/not_crop
- `frames_uncertain`: Frames with ambiguous verdicts
- `source`: Provenance of diagnosis (`"measured"`)

#### 3. Growth Stage Block (`growth_stage`)
- `crop`: Crop evaluated (`"rice"`, `"wheat"`, `"sugarcane"`, or `null`)
- `stage`: Phenological stage: `initial`, `development`, `mid_season`, `late_season`, or `null`
- `stage_code`: Short code: `"INI"`, `"DEV"`, `"MID"`, `"LATE"`, or `null`
- `days_since_planting`: Days elapsed since planting / transplanting (integer or `null`)
- `total_cycle_days`: Lifecycle duration (typical 120–280 days)
- `cycle_source`: Origin of cycle length (`"default_assumption"`, `"user_override"`)
- `cycle_verification_status`: Provenance status (`"RECALLED_UNVERIFIED"`)
- `stage_lengths_days`: Dictionary of stage durations (`initial`, `development`, `mid_season`, `late_season`)
- `canopy_cover_measured`: Fractional green canopy coverage measured during scan (float or `null`)
- `canopy_cover_expected_range`: Expected canopy cover range for stage (`[min, max]` or `null`)
- `kc`: FAO-56 Table 12 crop coefficient (`0.20` to `1.35` or `null`)
- `status`: `"OK"`, `"DAYS_SINCE_PLANTING_REQUIRED"`, `"AWAITING_PLANTING_DATE"`, `"UNSUPPORTED_CROP"`
- `reason`: Explanation if stage unestimated (`"DAYS_SINCE_PLANTING_REQUIRED"` or `null`)
- `verification_status`: Four-value frozen provenance status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`
- `source`: Agronomic model origin (`"derived"`)
- `document_reference`: Primary FAO-56 document reference citation

#### 4. Vegetation Block (`vegetation`)
- `interpretation_mode`: `"relative"` (within-scan relative distribution mode)
- `canopy_cover`: Fractional green canopy coverage statistics (`mean`, `p10`, `p50`, `p90`, `min_fraction_threshold`, `status`, `threshold_source`, `threshold_confirmed`, `source`)
- `vari`: Visible Atmospherically Resistant Index statistics (`band`, `band_basis`, `field_median`, `mean`, `p10`, `p50`, `p90`, `source`, `threshold_confirmed`, `threshold_source`; or `mean`: null, `reason`, `min_fraction_threshold`, `threshold_source`, `threshold_confirmed`, `source` when canopy insufficient)
- `exg`: Excess Green index statistics (`mean`, `threshold_source`, `threshold_confirmed`, `source`; or `reason`, `min_fraction_threshold` when canopy insufficient)
- `tgi`: Triangular Greenness Index statistics (`mean`, `threshold_source`, `threshold_confirmed`, `source`; or `reason`, `min_fraction_threshold` when canopy insufficient)
- `dgci`: Dark Green Color Index statistics (`mean`, `out_of_domain_fraction`, `threshold_source`, `threshold_confirmed`, `source`; or `reason`, `threshold` when out of domain)
- `ndvi`: Dual-bandpass normalized difference vegetation index (`null` when hardware pending)
- `ndvi_status`: `"GATED_HARDWARE_CALIBRATION"`, `"PENDING_HARDWARE_FINALIZATION"`
- `ndvi_reason`: Text explanation of optical hardware reservation

#### 5. Thermal & Reference CWSI Block (`thermal`)
- `available`: `true` when MLX90640 frame is captured and reference surfaces are configured/valid; `false` otherwise
- `reason`: Explanation if CWSI unavailable (`REPLAY_THERMAL_NOT_OF_SCENE`, `THERMAL_REFS_NOT_CONFIGURED`, `INSUFFICIENT_REFERENCE_GAP`, `WET_REF_VARIANCE_HIGH`, `DRY_REF_VARIANCE_HIGH`, `HARDWARE_NOT_CONNECTED`, etc.)
  - `REPLAY_THERMAL_NOT_OF_SCENE`: Scan RGB source is a recorded replay file; live MLX90640 reading does not correspond to recorded video scene.
- `tc_c`: Median canopy temperature (°C) measured directly from thermal array (populated whenever a valid thermal frame is captured, even when wet/dry references are unconfigured; `null` on replay)
- `twet_c`: Median temperature of wet reference pad (°C)
- `tdry_c`: Median temperature of dry reference pad (°C)
- `cwsi`: Raw Crop Water Stress Index computed via Jones (1999):
  $$\text{CWSI} = \frac{T_c - T_{\text{wet}}}{T_{\text{dry}} - T_{\text{wet}}}$$
- `flag`: Out-of-bounds flag (unclamped raw reporting): `"NORMAL"`, `"CWSI_BELOW_ZERO"`, `"CWSI_ABOVE_ONE"`
- `thermal_source`: Origin of thermal data: `"hardware"`, `"mock"`
- `frame_utc`: ISO-8601 UTC timestamp of thermal frame capture (or `null`)

#### 6. Sentinel-2 Satellite NDVI Block (`ndvi_satellite`)
- `available`: `true` when a valid, cloud-free Sentinel-2 L2A scene is cached from Copernicus CDSE; `false` otherwise
- `reason`: Explanation if unavailable (`NO_SATELLITE_DATA_RECORDED`, `NO_CLEAR_SCENE`, `CREDENTIALS_MISSING`, `FIELD_CONFIG_MISSING`, `FIELD_NOT_CONFIGURED`, etc.)
- `source`: Origin label (`"SENTINEL2_L2A_CDSE"`)
- `scene_date`: Acquisition date of scene (`YYYY-MM-DD` or `null`)
- `age_days`: Elapsed days since scene acquisition (`>= 0.0` or `null`)
- `ndvi_mean`: Field polygon mean NDVI (`-1.0` to `1.0` or `null`)
- `ndvi_std`: Field polygon standard deviation (`>= 0.0` or `null`)
- `valid_pixel_count`: Number of cloud-free valid 10m pixels evaluated
- `cloud_masked_fraction`: Fractional cloud/shadow mask coverage (`0.0` to `1.0` or `null`)
- `pixel_size_m`: Ground sample distance (`10`)
- `reliability_note`: `"UNRELIABLE_SMALL_FIELD (<9 pixels / ~30x30m footprint)"` or `null`

#### 7. Irrigation Block (`irrigation`)
- `available`: `true` when ground mast temperature history is sufficient; `false` otherwise
- `reason`: Explanation if irrigation computation unavailable
- `method`: `"fao56_hargreaves_samani"` (FAO-56 Eq 52)
- `air_temp_c`: Instantaneous air temperature (°C)
- `rh_pct`: Relative humidity (%)
- `t_min_24h_c`: Minimum diurnal air temperature (°C)
- `t_max_24h_c`: Maximum diurnal air temperature (°C)
- `t_mean_24h_c`: Mean diurnal air temperature (°C)
- `ra_mj_m2_day`: Extraterrestrial radiation (MJ/m²/day) from FAO-56 Eqs. 21–25
- `ra_mm_day`: $R_a \times 0.408$ equivalent depth (mm/day)
- `ra_source`: `"GPS"` or `"CONFIG_LATITUDE"`
- `ra_latitude_deg`: Latitude used for $R_a$ calculation
- `day_of_year`: Day of year ($J$, 1–366)
- `et0_mm_day`: Reference evapotranspiration $ET_0$
- `kc`: Crop coefficient from phenology lookup
- `crop_et_mm_day`: Crop evapotranspiration ($ET_c = ET_0 \times K_c$)
- `soil1_v`: Soil moisture sensor 1 voltage (V)
- `soil2_v`: Soil moisture sensor 2 voltage (V)
- `battery_v`: Mast node battery voltage (V)
- `samples_24h`: Number of ground mast readings in last 24h ($\ge 6$)
- `source`: `"derived_fao56"`

#### 8. Pest Block (`pest[]`)
- `target_pest_context`: Evaluated pest species context (e.g. `aphid`, `whitefly`, `thrips`, `mirid_bug`)
- `count_basis`: Blob counting basis (`"watershed_all_blobs"`, `"direct_count"`)
- `count_observed`: Total observed sticky trap count
- `days_monitored`: Days card has been deployed (`1.0` to `7.0`)
- `daily_rate`: Observed insects per trap daily rate
- `threshold_value`: Published economic threshold value (or `null`)
- `threshold_unit`: `"insects_per_trap"` (or `null`)
- `threshold_available`: `true` if numeric threshold exists; `false` otherwise
- `status`: Standardized 3-way comparison status:
  - `ABOVE_ETL`: Observed count strictly exceeds economic threshold (`count_observed > threshold`).
  - `AT_ETL`: Observed count exactly equals economic threshold (`count_observed == threshold`).
  - `BELOW_ETL`: Observed count is below economic threshold (`count_observed < threshold`).
  - Special conditions: `NO_PUBLISHED_ETL`, `NOT_SAMPLED_BY_STICKY_TRAP`, `UNKNOWN_PEST`, `CARD_SATURATED`, `INVALID_MONITORING_WINDOW`, `MISSING_DEPLOYMENT_TIMESTAMP`.
- `threshold_verification_status`: Four-value status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`
- `classification_verification_status`: `"RECALLED_UNVERIFIED"`
- `total_blobs_counted`: Deterministic watershed blob count
- `classification_source`: `"CROSS_DOMAIN_PRETRAINED"` (Model B)

#### 9. Detections Block (`detections[]`)
- `class`: Model A diagnosed class (29 classes, e.g. `"rice__blast"`)
- `confidence`: Softmax probability (`0.0` to `1.0`)
- `cross_source_reliability`: Reliability tier (`TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED`)
- `lat`: Detection GPS latitude coordinate (float or `null`)
- `lon`: Detection GPS longitude coordinate (float or `null`)
- `fix_quality`: GPS fix quality integer (`0`=invalid, `1`=GPS, `2`=DGPS) or `null`
- `hdop`: Horizontal Dilution of Precision (float or `null`)
- `captured_utc`: ISO-8601 UTC timestamp of frame capture
- `source`: Provenance of detection (`"measured"`)

#### 10. GPS Scan Summary Block (`gps`)
- `status`: GPS overall coverage status during walk scan (`"OK"`, `"ABSENT"`)
- `point_count`: Total number of geotagged frames recorded in scan (`>= 0`)
- `accuracy_note`: Sensor precision disclaimer (`"Point tagging only, approximately 2.5 m CEP. Not a survey-grade position."`)

#### 11. Disease Diagnoses Block (`disease[]`)
- `class`: Standardized disease class (e.g. `"rice__blast"`, `"sugarcane__red_rot"`, `"wheat__yellow_rust"`)
- `confidence`: Softmax probability (`0.0` to `1.0`)
- `media_ids`: Array of associated image IDs (empty list `[]` per media retention pruning policy)
- `source`: Provenance of diagnosis (`"measured"`)

#### 12. Sensor Inputs Block (`inputs[]`)
- `name`: Sensor stream identifier (`"pod_camera_rgb"`, `"pod_gps"`, `"pod_thermal"`)
- `source_node`: Node hosting the sensor (`"POD"`, `"MAST"`)
- `status`: Operational hardware and calibration status (`"OK"`, `"PENDING_CALIBRATION"`, `"MOCK_PROVISIONAL"`, `"ABSENT"`)
  - `OK`: Sensor hardware present, data acquired, and derived values fully operational.
  - `PENDING_CALIBRATION`: Sensor hardware present and functional (e.g. valid canopy temperature `tc_c` acquired), but derived metric (CWSI) is unavailable pending reference calibration (`configs/thermal_refs.json`).
  - `MOCK_PROVISIONAL`: Simulated / mock data source requested explicitly.
  - `ABSENT`: Sensor hardware disconnected or probe failed.

#### 13. Actions Block (`actions[]`)
- `rank`: Priority rank (`>= 1`)
- `template_id`: One of 21 deterministic action templates (`ACT_EXT_OFFICER_CONSULT`, `ACT_IRRIGATE_WATER_DEFICIT`, `ACT_MAINTAIN_ROUTINE`, `ACT_MULTICROP_INVESTIGATE`, `ACT_RESCAN_AMBIGUOUS`, etc.)
- `action`: English action directive (<= 240 chars)
- `rationale`: Agronomic rationale (<= 400 chars)
- `params`: Parameter dictionary for deterministic offline translation
- `verification_status`: Four-value status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`
- `url`: Primary legal / agronomic regulatory source URL (or `null`)
- `document_reference`: Exact publication / bulletin reference citation (or `null`)
- `offline_source_file`: Path to local archived source document in `docs/sources/` (or `null`)
- `confidence`: Action confidence string (`"high"`, `"medium"`, `"low"`)
- `advisory_only`: Boolean disclaimer (`true`)
- `generated_by`: Generator provenance tag (ALWAYS `"template"`)
- `source`: Agronomic origin label (`"derived"`)

---

## 3. Production Mock Advisory Protection & Inference Backend

1. **Production Guard (`allow_mock=False`)**:
   - By default, the Gateway operates in production mode (`allow_mock=False`).
   - `GET /api/v1/advisory/<id>` returns `403 Forbidden` (`{"error": "mock_advisory_rejected", ...}`) if the requested advisory was generated with `inference_backend: "mock"` or `thermal_source: "mock"`.
   - `GET /api/v1/manifest` automatically filters out mock advisories in production mode.
2. **Development / Test Harness Override**:
   - Start Gateway with `--allow-mock` or pass `?allow_mock=true` to view synthetic mock data during offline integration tests.
3. **Inference Backend Identifier**:
   - Emitted payloads explicitly record `inference_backend`: `"trt"` (TensorRT on Maxwell GPU), `"onnx"` (ONNX Runtime CPU fallback), or `"mock"` (synthetic test vector).

---

## 4. Ground Mast Pull Telemetry & Collector Sync Endpoints

The ground mast ESP32 acts strictly as an HTTP server (`192.168.9.1`, AP `SIH-NODE-01`). The Jetson Nano pulls telemetry via `edge/mast_collector.py`.

### 4.1 Sync Management Endpoints

| Method | Path | Response | Description |
|---|---|---|---|
| `POST` | `/api/v1/sync/trigger` | `202 Accepted`<br>`{"status": "accepted", "timestamp": "2026-09-21T05:47:54Z", "expected_ap_downtime_s": 30}` | Schedules asynchronous collector pull. Returns `409 Conflict` (`{"error": "sync_in_progress", "detail": "..."}`) if sync is already in progress. |
| `GET` | `/api/v1/sync/status` | `200 OK`<br>`{"sync_in_progress": false, "last_success_utc": null, "last_attempt_utc": null, "last_result": null, "mast_data_age_s": null, "records_pulled": 0, "trap_images_pulled": 0}` | Returns collector status, timestamps, records transferred, and mast data age in seconds. |

#### Sync Status Field Breakdown (`GET /api/v1/sync/status`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `sync_in_progress` | boolean | No | `true`, `false` | Whether a background mast sync pull is currently executing |
| `last_success_utc` | string | Yes | ISO-8601 UTC string or `null` | ISO-8601 UTC timestamp of last successful sync cycle |
| `last_attempt_utc` | string | Yes | ISO-8601 UTC string or `null` | ISO-8601 UTC timestamp of last sync attempt |
| `last_result` | string | Yes | `"OK"`, `"MAST_NOT_FOUND"`, `"PARTIAL"`, `"ERROR"`, or `null` | Result code of last sync cycle |
| `mast_data_age_s` | integer | Yes | `>= 0` or `null` | Age of latest ingested mast telemetry record in seconds |
| `records_pulled` | integer | No | `>= 0` | Count of telemetry records pulled in last sync cycle |
| `trap_images_pulled` | integer | No | `>= 0` | Count of sticky-trap images pulled in last sync cycle |

### 4.2 Ground Mast Pull Record Schema (`GET /readings`)
- `log_epoch`: Monotonic boot timestamp to detect ESP32 reboots
- `seq`: Monotonic sequence number per boot epoch
- `node_id`: Ground mast node ID (`"SIH-NODE-01"`)
- `field_id`: Field identifier (or `null`)
- `utc`: ISO-8601 UTC timestamp if RTC synchronized (or `null`)
- `rtc_valid`: DS3231 RTC synchronization flag (`true`/`false`)
- `uptime_s`: ESP32 uptime in seconds
- `air_temp_c`: SHT40 air temperature (°C)
- `rh_pct`: SHT40 relative humidity (%)
- `ir_object_c`: MLX90614 object surface temperature (°C)
- `ir_ambient_c`: MLX90614 ambient sensor temperature (°C)
- `lux`: BH1750 ambient light level (lux)
- `soil1_v`: ADS1115 soil moisture probe 1 voltage (V)
- `soil2_v`: ADS1115 soil moisture probe 2 voltage (V)
- `battery_v`: LiFePO4 battery voltage (V)
- `status`: Sensor bus health status dictionary
- `received_at`: Ingestion timestamp recorded by Nano

---

## 5. Complete Mobile Gateway REST API Route Table

| Method | Path | Request Body | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Device liveness, advisory count, latest sequence, storage free KB, GPS clock status, and AP/STA sync state |
| `GET` | `/api/v1/manifest?since=&limit=` | None | Monotonic advisory catalog pagination (mock advisories omitted in production) |
| `GET` | `/api/v1/advisory/latest` | None | Retrieves the most recent synthesized advisory document (ordered by `seq DESC LIMIT 1`) |
| `GET` | `/api/v1/advisory/<id_or_seq>` | None | Complete frozen v1.0 advisory document by UUID string or integer sequence number (returns 403 Forbidden in production if mock) |
| `POST` | `/api/v1/ack` | `{"advisory_id": "<id>"}` | Advisory acknowledgement cursor advancement |
| `POST` | `/api/v1/trap/upload?trap_id=&days=` | Multipart JPG image | Sticky trap card photo for Model B segmentation & classification |
| `GET` | `/api/v1/media/<id>` | None | Returns `410 Gone` (media retention pruned per policy) |
| `POST` | `/api/v1/sync/trigger` | None | Triggers async collector sync against mast node (returns `202 Accepted` or `409 Conflict`) |
| `GET` | `/api/v1/sync/status` | None | Returns collector sync status, timestamps, records transferred, and mast data age in seconds |

#### Health Endpoint Schema (`GET /api/v1/health`)
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

#### Advisory Manifest Endpoint Schema (`GET /api/v1/manifest?since=&limit=`)
##### Top-Level Response
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire schema version |
| `count` | integer | No | `>= 0` | Number of advisory items returned in this page |
| `advisories` | array | No | List of summary objects | Monotonically ordered advisory catalog items |
| `truncated` | boolean | No | `true`, `false` | True if additional newer advisories exist beyond limit |

##### Advisory Summary Object (`advisories[]`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `advisory_id` | string | No | String identifier | Unique advisory document identifier |
| `seq` | integer | No | `>= 1` | Monotonic sequential advisory sequence number |
| `generated_at_utc` | string | No | ISO-8601 UTC string | Timestamp of advisory generation |
| `bytes` | integer | No | `>= 0` | Size of raw JSON advisory document in bytes |
| `replay` | boolean | No | `true`, `false` | Provenance label (`true` for replay, `false` for live) |
| `inference_backend` | string | No | `"trt"`, `"onnx"`, `"mock"` | Model A inference backend used |

#### Acknowledgment Endpoint Schema (`POST /api/v1/ack`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `"ok"` | Acknowledgement execution status |
| `acked` | string / integer | No | String or integer | Cursor or advisory identifier acknowledged |

#### Media Pruning Endpoint Schema (`GET /api/v1/media/<id>`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `error` | string | No | `"gone"` | HTTP 410 error identifier |
| `reason` | string | No | `"retention_pruned"` | Explanation of media lifecycle pruning policy |

#### Mast Sync Trigger Endpoint Schema (`POST /api/v1/sync/trigger`)
| Field | Type | Nullable | Allowed Values | Description |
|---|---|---|---|---|
| `status` | string | No | `"accepted"` | Trigger acceptance status (returns 202) |
| `timestamp` | string | No | ISO-8601 UTC string | Trigger initiation timestamp |
| `expected_ap_downtime_s` | integer | No | `30` | Expected AP downtime during STA mode switch |

#### Sticky Trap Photo Upload Endpoint Schema (`POST /api/v1/trap/upload`)

> [!NOTE]
> **Hardware Status:** Not yet exercised on real Jetson Nano device (Model B pipeline verified in unit tests and offline simulation).

##### Success Response (`200 OK`)
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
