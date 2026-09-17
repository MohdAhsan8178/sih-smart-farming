# SIH 2026 Ground Mast & Trap Camera Firmware

Reference firmware for the ground mast ESP32 DevKit node and the ESP32-CAM sticky-trap camera node, exported directly from `docs/HARDWARE_WIRING_GUIDE.md` §6 (rev 2, 17 Sep 2026).

## Toolchain & Build Environment

| Item | Main Mast Node (`node_n01.ino`) | Trap Camera (`esp32cam_trap.ino`) |
| :--- | :--- | :--- |
| **Arduino Board Package** | `esp32` by Espressif, version **2.0.17** | `esp32` by Espressif, version **2.0.17** |
| **Board Selection** | `ESP32 Dev Module` | `AI Thinker ESP32-CAM` |
| **Partition Scheme** | **No OTA (2MB APP/2MB SPIFFS)** | **Huge APP (3MB No OTA)** |
| **Upload Speed** | `921600` (fallback `115200`) | `115200` |
| **Libraries (Library Manager)** | • Adafruit SHT4x Library<br>• Adafruit ADS1X15<br>• Adafruit MLX90614 Library<br>• BH1750 (by Christopher Laws)<br>• RTClib (Adafruit)<br>• ArduinoJson **7.x** | *None* beyond core board package (`esp_camera.h`) |

> [!CAUTION]
> Do NOT use ESP32 board package 3.x: the watchdog API and task WDT timer signatures changed in 3.x. The codebase strictly targets **2.0.17**.

## Sketches

1. **`firmware/node_n01/node_n01.ino`**:
   - Runs on ESP32 DevKit V1 (38-pin).
   - Starts AP `SIH-NODE-01` (password `sih12345`, IP `192.168.9.1`).
   - Samples I²C sensors every 10 minutes (SHT40, MLX90614, BH1750, ADS1115 for soil moisture, DS3231 RTC).
   - Rotates local flash storage in LittleFS (max 200 kB × 2 files).
   - Wakes camera twice daily (04:30 and 09:30 UTC = 10:00 and 15:00 IST) via GPIO25.
   - Serves HTTP API (`/api/v1/health`, `/api/v1/readings`, `/api/v1/trap/list`, `/api/v1/trap/image`, `/api/v1/time`, `/api/v1/trap/upload`).
   - Includes prune-after-auth fix in `hUploadChunk`.

2. **`firmware/esp32cam_trap/esp32cam_trap.ino`**:
   - Runs on AI-Thinker ESP32-CAM.
   - Wakes on EXT0 pulse (GPIO13).
   - Captures UXGA (1600×1200) JPEG trap image (quality 12, warmup 5 frames).
   - Connects to AP `SIH-NODE-01` and uploads image chunked to `http://192.168.9.1/api/v1/trap/upload`.
   - Returns immediately to deep sleep (active time 8–15 s).
