# AEGIS — Scan Control API & Walk Report (spec v1, 1 Oct 2026)

Shared contract between the pod (Jetson Nano, repo `sih-smart-farming`) and the phone app (`Aegis-SIH/mobile`).
All changes are **additive to wire contract 1.0**. Existing endpoints and fields keep working unchanged.
All timestamps: ISO-8601 UTC with `Z`. All endpoints are on the existing gateway, port 8080.

---

## 0. The farmer's flow (what this spec makes possible)

1. Farmer switches the pod on. ~60–90 s later the pod is ready (auto-start, no laptop).
2. App shows **Pod ready** (`GET /api/v1/health` → `pod_ready: true`).
3. Farmer taps **Start scan**, picks field + crop → `POST /api/v1/scan/start`.
4. Farmer walks. App polls `GET /api/v1/scan/status` every 1 s, shows the live screen, vibrates on new alerts,
   and sends the phone's GPS fixes every ~5 s → `POST /api/v1/scan/track`.
5. Farmer taps **Stop** → `POST /api/v1/scan/stop`. Status goes `finalizing` → `done` with an `advisory_id`.
6. App fetches that advisory directly (`GET /api/v1/advisory/<advisory_id>`), acks it, opens the **walk report**.
7. Farmer taps **Shut down pod** → `POST /api/v1/pod/shutdown`.

One walk = one advisory (one report). Healthy areas are counted, not listed. Only problems are listed.

---

## 1. Endpoints

### 1.1 `GET /api/v1/health` (existing — fields added)

```jsonc
{
  // ... all existing fields unchanged ...
  "pod_ready": true,                       // gateway up, camera + engine available
  "scan_state": "idle",                    // idle | starting | scanning | finalizing
  "clock_source": "gps",                   // gps | phone | filesystem   ("phone" is new)
  "storage": { "location": "sd", "free_mb": 46210 }   // sd | internal
}
```

### 1.2 `POST /api/v1/scan/start`

Request:
```jsonc
{
  "field_id": "F01",
  "crop": "wheat",                         // wheat | rice | sugarcane  (declared by the farmer)
  "phone_utc": "2026-10-03T04:30:12Z",     // the phone's current time
  "source": "camera"                       // camera | replay   (replay = demo crop video, labelled replay:true)
}
```
Responses:
- `202` `{"scan_id": "2026-10-03T04:30:12Z_F01", "state": "starting", "replay": false}`
- `409` `{"error": "scan_in_progress", "scan_id": "..."}`
- `400` `{"error": "bad_request", "detail": "..."}` (unknown crop, missing field_id)
- `503` `{"error": "camera_unavailable"}`

Pod behaviour on start:
1. **Time:** if the pod has no GPS time, it adopts `phone_utc` (`clock_source: "phone"`).
2. **Field station:** collects the latest ESP32 readings (max ~5 s; finds the ESP32 on the local network itself).
   If found and the pod has GPS or phone time, it also sets the ESP32 clock. If not found → warning `FIELD_STATION_NOT_FOUND`, scan continues.
3. Starts the scan process. Scanning runs **until stopped** (no frame limit).

### 1.3 `GET /api/v1/scan/status` (poll every ~1 s)

```jsonc
{
  "state": "scanning",                     // idle | starting | scanning | finalizing | done | error
  "scan_id": "2026-10-03T04:30:12Z_F01",
  "field_id": "F01",
  "crop": "wheat",
  "replay": false,
  "started_utc": "2026-10-03T04:30:12Z",
  "elapsed_s": 312,
  "max_duration_s": 1800,                  // safety stop after 30 min
  "counts": {
    "frames_seen": 9360, "frames_used": 410,
    "stretches": 15, "healthy": 11, "need_look": 1, "unclear": 2, "not_crop": 1
  },
  "thermal_c_latest": 29.4,                // null if no thermal reading
  "field_station": { "reachable": true, "readings_collected": 48, "last_reading_utc": "2026-10-03T04:30:00Z" },
  "warnings": [                            // currently active; app shows the wording (see §3)
    { "code": "BLURRY_SLOW_DOWN", "since_utc": "2026-10-03T04:35:01Z" }
  ],
  "alerts": [                              // last 20, newest last; alert_id strictly increasing within a scan
    { "alert_id": 3, "utc": "2026-10-03T04:34:40Z", "class": "wheat__brown_rust",
      "frames_agreeing": 4, "lat": 26.8467, "lon": 80.9462, "pos_accuracy_m": 4.0 }
  ],
  "advisory_id": null,                     // set when state == "done"
  "stop_reason": null                      // user | time_limit | error | interrupted
}
```
The app vibrates once for each `alert_id` it has not seen before.

### 1.4 `POST /api/v1/scan/track` (every ~5 s while scanning)

```jsonc
{
  "scan_id": "2026-10-03T04:30:12Z_F01",
  "fixes": [
    { "utc": "2026-10-03T04:34:38Z", "lat": 26.84671, "lon": 80.94622, "accuracy_m": 4.0, "speed_mps": 0.6 }
  ]
}
```
→ `200 {"accepted": 3}`. Fixes with `accuracy_m > 25` are stored but not used for positions.
Position priority on the pod: **pod GPS fix (when valid) → phone fix → none (null)**. Never interpolated beyond ±3 s.

### 1.5 `POST /api/v1/scan/stop`

`{"scan_id": "..."}` → `202 {"state": "finalizing"}`. Idempotent: stopping an already-finalizing or done scan returns `200` with its current state.
Pod behaviour: stops the camera, collects the ESP32 once more, writes the advisory (~5–15 s), state → `done`.

### 1.6 `POST /api/v1/pod/shutdown`

`{"confirm": true}` → `202 {"shutting_down_in_s": 5}`. If a scan is running it is stopped and finalized first.

---

## 2. The walk report (advisory) — new fields

Everything existing stays. Added:

```jsonc
{
  "scan": {
    // ... existing fields ...
    "mode": "walk",
    "crop_declared": "wheat",
    "duration_s": 742,
    "stop_reason": "user"                  // user | time_limit | error | interrupted
  },
  "time_source": "phone",                  // gps | phone | filesystem
  "summary": { "stretches_total": 37, "healthy": 30, "need_look": 3, "unclear": 3, "not_crop": 1, "no_data": 0 },
  "stretches": [
    { "index": 0, "start_utc": "...", "end_utc": "...", "frames_used": 14,
      "verdict": "HEALTHY",                // HEALTHY | DISEASE | UNCERTAIN | NOT_CROP | NO_DATA
      "top_class": null, "frames_agreeing": 0,
      "thermal_median_c": 28.9,            // null in replay or without thermal
      "lat": 26.8467, "lon": 80.9462, "pos_accuracy_m": 4.0, "pos_source": "phone_gps" }   // nulls if no position
  ],
  "alerts": [ /* every alert raised during the walk, same shape as in status */ ],
  "field_conditions": {
    "available": true, "reason": null, "node_id": "SIH-NODE-01",
    "reading_utc": "...", "age_minutes": 0.4,
    "air_temp_c": 27.1, "rh_pct": 61.2, "lux": 1840.0,
    "soil1_v": 2.41, "soil2_v": 2.38, "battery_v": 3.27,
    "soil_units": "raw_volts_uncalibrated", "source": "measured"
  }                                        // or {"available": false, "reason": "NO_VALID_MAST_READING", values null}
}
```
- `detections[]` get `lat`/`lon` from the same position rule (existing fields — the map already uses them).
- `gps` block: `status` reflects the source used; `source`: `pod_gps` | `phone_gps`.
- A stretch = **20 s of walking**. ~37 stretches for a 12-minute walk; max 90 (30-min cap).
- **Interrupted walks** (power cut): on the next boot the pod turns the saved part into a normal advisory with `stop_reason: "interrupted"`. Nothing up to the cut is lost.

---

## 3. Warnings and alerts — codes and the exact app wording

Pod sends codes. The app owns the wording (English + Hindi).

| Code (from pod) | When (provisional thresholds) | English text in the app |
|---|---|---|
| `TOO_DARK` | ≥60 % of frames in last 5 s rejected as underexposed | Too dark to see the leaves. Scan in daylight or move out of the shade. |
| `TOO_BRIGHT` | ≥60 % overexposed in last 5 s | Too much glare. Tilt the pod away from the sun. |
| `BLURRY_SLOW_DOWN` | ≥50 % blurry in last 5 s, or phone speed > 1.2 m/s | Walk slower. Pictures are coming out blurry. |
| `NOT_SEEING_CROP` | ≥80 % of tiles "not crop" in last 10 s | Not able to see the leaves clearly. Point it towards the crop. |
| `FIELD_STATION_NOT_FOUND` | ESP32 not reachable at start | Field station not found. Check that it is switched on. The scan continues without weather and soil data. |
| `POD_HOT` | Nano temperature > 80 °C | Pod is getting hot. Keep it out of direct sun. |
| `STORAGE_LOW` | < 500 MB free | Pod storage almost full. |
| `STORAGE_CARD_MISSING` | SD card not mounted (saving to internal memory) | Storage card not found. Saving to the pod's internal memory. |
| *(app-side)* pod not answering for 6 s | — | Phone is too far from the pod. The pod is still scanning. Walk closer to the pod to reconnect. |
| *(app-side)* phone battery < 20 % | — | Phone battery low ({n}%). Charge soon. |
| **Alert** (vibrate) | ≥3 frames agree on the same disease of the declared crop within 6 s; 15 s cooldown per disease | Possible {disease}. Check this plant by eye. |

**Declared crop:** a tile whose top class belongs to a *different* crop is counted as "unclear", never as that other crop's disease.
Model confidence and calibration are not changed.

---

## 4. Honesty rules (unchanged principles)

- No value is invented. Missing = `null` + reason.
- Replays stay labelled `replay: true`; thermal in replay stays `REPLAY_THERMAL_NOT_OF_SCENE`.
- Phone time / phone GPS are used automatically when the pod's own clock/GPS are not working. The app does **not** show warnings about this;
  the source is recorded in the data (`time_source`, `pos_source`) and shown neutrally only in "About this scan".
- Alerts always say "possible" and "check by eye".
