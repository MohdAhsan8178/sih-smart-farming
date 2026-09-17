# App Team Handoff & Payload Changes Specification

**Date:** 18 September 2026  
**Document:** `docs/APP_TEAM_CHANGES_2026-09-18.md`  
**Supersedes:** `data_flow_architecture.md` field definitions and `ans_for_vitthal.md` legacy draft items.  
**Authoritative Reference:** `docs/PAYLOAD_CONTRACT.md` (validated by `tests/test_payload_contract.py`).

---

## 1. Summary of Major Structural Changes

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

## 2. Top-Level Advisory Payload (`GET /api/v1/advisory/<id>`)

### 2.1 Added Fields
| Field Path | Type | Values / Format | Purpose |
|---|---|---|---|
| `inference_backend` | string | `"trt"`, `"onnx"`, `"mock"` | Explicit runtime inference engine identifier. Guarantees no silent fallback occurred during scanning. |

### 2.2 Modified Blocks
1. **`irrigation` Block**:
   - **Method**: Switched to `"fao56_hargreaves_samani"` (FAO-56 Eq 52).
   - **Inputs**: Calculated from 24h diurnal air temperature spread ($T_{min}, T_{max}, T_{mean}$) recorded by ground mast SHT40, and dynamic extraterrestrial radiation $R_a$ (FAO-56 Eqs. 21–25) calculated from day of year and pod GPS latitude (with field config fallback).
   - **New Fields Added**: `t_min_24h_c` (float), `t_max_24h_c` (float), `t_mean_24h_c` (float), `ra_mj_m2_day` (float), `ra_mm_day` (float), `ra_source` (`"GPS"` or `"CONFIG_LATITUDE"`), `ra_latitude_deg` (float), `day_of_year` (int 1-366), `samples_24h` (int >= 6).
   - **Fields Removed**: `canopy_temp_c`, `cwsi`, `water_level_mm`, `paddy_awd` (sensors absent from physical build; see `PENDING_HARDWARE.md`).
   - **Coverage Rule**: Requires $\ge 6$ readings spanning $\ge 6.0$ hours in last 24h. Otherwise returns `"available": false` with explicit explanation.

2. **`crop_health` Block**:
   - Standardized state values: `HEALTHY`, `DISEASE`, `UNCERTAIN`, `NOT_CROP`, `NO_DATA`.
   - Explicit failure reasons: `HIGH_UNCERTAINTY`, `MULTIPLE_CROPS_DETECTED`, or `null`.

3. **`growth_stage` Block**:
   - Standardized 4-stage phenology: `initial`, `development`, `mid_season`, `late_season` (codes `INI`, `DEV`, `MID`, `LATE`).
   - Standardized 4-value provenance status: `VERIFIED`, `WEB_VERIFIED`, `RECALLED_UNVERIFIED`, `UNSOURCED`.

4. **`detections[].cross_source_reliability`**:
   - Standardized 4-tier reliability enum: `TESTED_ROBUST`, `TESTED_WEAK`, `TESTED_FAILED`, `UNTESTED`.

5. **`pest[].status` (Sticky-Trap ETL Status)**:
   - Established 3-way threshold comparison:
     - `ABOVE_ETL`: Observed count strictly exceeds economic threshold (`count_observed > threshold`).
     - `AT_ETL`: Observed count exactly equals economic threshold (`count_observed == threshold`).
     - `BELOW_ETL`: Observed count is below economic threshold (`count_observed < threshold`).
   - Standardized status strings for special conditions: `NO_PUBLISHED_ETL`, `NOT_SAMPLED_BY_STICKY_TRAP`, `UNKNOWN_PEST`, `CARD_SATURATED`, `INVALID_MONITORING_WINDOW`, `MISSING_DEPLOYMENT_TIMESTAMP`.

6. **Production Mock Advisory Guard**:
   - By default, Gateway operates in production mode (`allow_mock=False`).
   - `GET /api/v1/advisory/<id>` returns `403 Forbidden` (`{"error": "mock_advisory_rejected", ...}`) if requested advisory was generated with `inference_backend: "mock"`.
   - `GET /api/v1/manifest` automatically filters out mock advisories in production mode.
   - For offline test harnesses and development, pass `--allow-mock` CLI flag to the Gateway.

---

## 3. Ground Mast Telemetry Architecture: PULL, Not PUSH

### 3.1 Network Topology & Role
- **Previous Specification**: Mast pushed readings to Nano via `POST /api/v1/mast/telemetry`.
- **Current Architecture (Guide §6)**: **PULL ONLY**.
  - Mast ESP32 operates as SoftAP `SIH-NODE-01` (password `sih12345`, IP `192.168.9.1`).
  - Jetson Nano runs `edge/wifi_switch.py` and `edge/mast_collector.py` to temporarily join `SIH-NODE-01`, pull new sensor records and camera trap photos, sync RTC clock from GPS fix, and return to `SIH-FIELD` AP.
  - Endpoint `POST /api/v1/mast/telemetry` has been **removed** from the Gateway.

### 3.2 Ground Mast Record Schema (`GET /readings`)
| Field | Type | Description |
|---|---|---|
| `log_epoch` | integer | Monotonic boot timestamp to detect ESP32 reboots |
| `seq` | integer | Monotonic sequence number per boot epoch |
| `node_id` | string | Mast node ID (`"SIH-NODE-01"`) |
| `field_id` | string / null | Field identifier |
| `utc` | string / null | ISO-8601 UTC timestamp if RTC valid |
| `rtc_valid` | boolean | DS3231 RTC synchronization flag |
| `uptime_s` | integer | ESP32 uptime in seconds |
| `air_temp_c` | float / null | SHT40 air temperature (°C) |
| `rh_pct` | float / null | SHT40 relative humidity (%) |
| `ir_object_c` | float / null | MLX90614 object surface temperature (°C) |
| `ir_ambient_c` | float / null | MLX90614 ambient sensor temperature (°C) |
| `lux` | float / null | BH1750 ambient light level (lux) |
| `soil1_v` | float / null | ADS1115 soil moisture probe 1 voltage (V) |
| `soil2_v` | float / null | ADS1115 soil moisture probe 2 voltage (V) |
| `battery_v` | float / null | LiFePO4 battery voltage (V) |
| `status` | object / null | Sensor bus health status dictionary |
| `received_at` | string | ISO-8601 timestamp recorded upon ingestion into SQLite |

---

## 4. Mobile Gateway Endpoints Summary

| Method | Path | Request Body | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Device liveness, unacked count, storage free KB, sync state |
| `GET` | `/api/v1/manifest?since=&limit=` | None | Monotonic advisory catalog pagination (in production mode, mock advisories omitted) |
| `GET` | `/api/v1/advisory/<id_or_seq>` | None | Complete frozen v1.0 advisory document (returns 403 Forbidden in production mode if advisory was generated with mock backend) |
| `POST` | `/api/v1/ack` | `{"advisory_id": "<id>"}` | Cursor advancement |
| `POST` | `/api/v1/trap/upload?trap_id=&days=` | Multipart JPG image | Sticky trap card photo for Model B segmentation & classification |
| `GET` | `/api/v1/media/<id>` | None | Returns `410 Gone` (media retention pruned per policy) |
| `POST` | `/api/v1/sync/trigger` | None | Triggers async collector sync against mast node (returns `202 Accepted` or `409 Conflict` if sync is already in progress) |
| `GET` | `/api/v1/sync/status` | None | Returns current sync status (`idle`, `connecting`, `pulling`, `success`, `error`), last sync timestamp, records pulled |
