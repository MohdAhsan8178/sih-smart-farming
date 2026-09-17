# App Team Handoff & Payload Changes Specification

**Last Updated:** 17 September 2026  
**Document:** `docs/APP_TEAM_CHANGES.md`  
**Supersedes:** `data_flow_architecture.md` field definitions and `ans_for_vitthal.md` legacy draft items.  
**Authoritative Reference:** `docs/PAYLOAD_CONTRACT.md` (validated by `tests/test_payload_contract.py`).

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

## 2. Complete Wire Contract Fields (`GET /api/v1/advisory/<id>`)

The advisory document emitted by `GET /api/v1/advisory/<id>` and `GET /api/v1/advisory/latest` adheres to schema version `1.0`.

### 2.1 Top-Level Fields

| Field Path | Type | Nullable | Values / Format | Purpose |
|---|---|---|---|---|
| `schema_version` | string | No | `"1.0"` | Wire contract schema version |
| `advisory_id` | string | No | String (e.g. `2026-09-21T08:00:00Z_field1`) | Unique advisory document identifier |
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
- `tiles_classified`: Total 320x320 tiles passed through inference (`frames * 9`)
- `distance_walked_m`: Traversed distance in meters (float or `null`)
- `distance_reason`: Reason if distance unavailable (`"GPS_TRACK_NOT_RECORDED"` or `null`)

#### 2. Crop Health Block (`crop_health`)
- `state`: Consensus health verdict across grid cells: `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`, `NO_DATA`
- `reason`: Explanation if non-definitive: `HIGH_UNCERTAINTY`, `MULTIPLE_CROPS_DETECTED`, or `null`
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
- `kc`: FAO-56 Table 12 crop coefficient (`0.20` to `1.35`)
- `status`: `"OK"`, `"DAYS_SINCE_PLANTING_REQUIRED"`, `"UNSUPPORTED_CROP"`
- `verification_status`: Four-value frozen provenance status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`
- `source`: Agronomic model origin (`"derived"`)

#### 4. Vegetation Block (`vegetation`)
- `interpretation_mode`: `"relative"` (within-scan relative distribution mode)
- `canopy_cover.mean`: Fractional green canopy coverage (`0.0` to `1.0`)
- `vari.band`: Relative greenness band: `LOWER_TAIL`, `BELOW_TYPICAL`, `TYPICAL`, `ABOVE_TYPICAL`
- `exg.mean`: Excess Green index mean
- `tgi.mean`: Triangular Greenness Index mean
- `dgci.mean`: Dark Green Color Index mean (`0.0` to `1.0`)
- `ndvi`: Dual-bandpass normalized difference vegetation index (`null` when hardware pending)
- `ndvi_status`: `"GATED_HARDWARE_CALIBRATION"`, `"PENDING_HARDWARE_FINALIZATION"`

#### 5. Thermal & Reference CWSI Block (`thermal`)
- `available`: `true` when MLX90640 frame is captured and reference surfaces are configured/valid; `false` otherwise
- `reason`: Explanation if CWSI unavailable (`THERMAL_REFS_NOT_CONFIGURED`, `INSUFFICIENT_REFERENCE_GAP`, `WET_REF_VARIANCE_HIGH`, `DRY_REF_VARIANCE_HIGH`, `HARDWARE_NOT_CONNECTED`, etc.)
- `tc_c`: Median canopy temperature excluding reference boxes (°C)
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
- `method`: `"fao56_hargreaves_samani"` (FAO-56 Eq 52)
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
- `samples_24h`: Number of ground mast readings in last 24h ($\ge 6$)

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
- `cell_id`: Spatial grid cell identifier (e.g. `"tile_r0_c0"`)
- `class_name`: Model A diagnosed class (29 classes)
- `confidence`: Softmax probability (`0.0` to `1.0`)
- `energy`: Open-set energy score
- `state`: `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`
- `cross_source_reliability`: `TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED`
- `latitude`: Detection GPS latitude (or `null`)
- `longitude`: Detection GPS longitude (or `null`)

#### 10. GPS Block (`gps`)
- `fix_valid`: Boolean flag indicating GPS lock
- `latitude`: WGS-84 latitude in decimal degrees (or `null`)
- `longitude`: WGS-84 longitude in decimal degrees (or `null`)
- `altitude_m`: Altitude above mean sea level in meters (or `null`)
- `hdop`: Horizontal Dilution of Precision (or `null`)
- `satellites_used`: Number of tracked satellites (or `null`)
- `last_fix_utc`: ISO-8601 UTC timestamp of last valid GPS fix (or `null`)

#### 11. Disease Block (`disease[]`)
- `disease_name`: Standardized disease name (e.g. `"rice_blast"`, `"wheat_rust"`)
- `crop`: Affected crop
- `severity_score`: Diagnosed severity index (`0.0` to `1.0`)
- `active`: Boolean flag indicating active disease state

#### 12. Inputs Status Block (`inputs[]`)
- `sensor_id`: Sensor identifier (`"pod_csi0_rgb"`, `"pod_csi1_noir"`, `"pod_mlx90640_thermal"`, `"pod_gps_uart"`, `"mast_sih_node_01"`)
- `type`: Input type category
- `status`: Health status (`"OK"`, `"ABSENT"`, `"UNCONFIGURED"`, `"UNCALIBRATED"`)
- `last_reading_utc`: Timestamp of last ingested reading

#### 13. Actions Block (`actions[]`)
- `rank`: Priority rank (`>= 1`)
- `template_id`: One of 21 deterministic action templates (`ACT_EXT_OFFICER_CONSULT`, `ACT_IRRIGATE_WATER_DEFICIT`, `ACT_MAINTAIN_ROUTINE`, `ACT_MULTICROP_INVESTIGATE`, `ACT_RESCAN_AMBIGUOUS`, etc.)
- `action`: English action directive (<= 240 chars)
- `rationale`: Agronomic rationale (<= 400 chars)
- `params`: Parameter dictionary for deterministic offline translation
- `verification_status`: Four-value status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`
- `offline_source_file`: Path to local regulatory PDF citation
- `advisory_only`: `true` (explicit non-liability disclaimer)

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
| `POST` | `/api/v1/sync/trigger` | `202 Accepted`<br>`{"status": "SYNC_SCHEDULED", "triggered_utc": "...", "task_id": "...", "expected_ap_downtime_s": 30}` | Schedules asynchronous collector pull. Returns `409 Conflict` if sync is already in progress. |
| `GET` | `/api/v1/sync/status` | `200 OK`<br>`{"status": "IDLE", "sync_in_progress": false, "last_attempt_utc": "...", "last_success_utc": "...", "last_result": "SUCCESS", "records_pulled": 42, "trap_images_pulled": 1, "mast_data_age_s": 120, "sih_collector_version": "1.0"}` | Returns collector status, timestamps, records transferred, and mast data age in seconds. |

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
| `GET` | `/api/v1/health` | None | Device liveness, unacked count, storage free KB, sync state |
| `GET` | `/api/v1/manifest?since=&limit=` | None | Monotonic advisory catalog pagination (mock advisories omitted in production) |
| `GET` | `/api/v1/advisory/<id_or_seq>` | None | Complete frozen v1.0 advisory document (returns 403 Forbidden in production if mock) |
| `POST` | `/api/v1/ack` | `{"advisory_id": "<id>"}` | Advisory acknowledgement cursor advancement |
| `POST` | `/api/v1/trap/upload?trap_id=&days=` | Multipart JPG image | Sticky trap card photo for Model B segmentation & classification |
| `GET` | `/api/v1/media/<id>` | None | Returns `410 Gone` (media retention pruned per policy) |
| `POST` | `/api/v1/sync/trigger` | None | Triggers async collector sync against mast node (returns `202 Accepted` or `409 Conflict`) |
| `GET` | `/api/v1/sync/status` | None | Returns collector sync status, timestamps, records pulled, expected downtime |
