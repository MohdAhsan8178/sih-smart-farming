# SIH Smart Farming -- Edge Software Subsystem Status Document

**Date:** 21 September 2026  
**Document:** `docs/SOFTWARE_STATUS.md`  
**Purpose:** Master deployment status, remaining demo checklist, verification states, provisional constant audit, physical hardware requirements, and single-command testing procedures on the Jetson Nano.

---

## 1. Remaining Before Demo

The following itemized checklist lists all remaining tasks that require physical hardware bench actions or pre-demo configuration:

| Category | Task Description | Target Component | Status / One Command to Validate |
| :--- | :--- | :--- | :--- |
| **`[HARDWARE]`** | **Physical Pod Sensor Assembly:** Connect IMX219-160 RGB camera to CSI-0, MLX90640 FIR array to I2C-1 (Pins 3/5 with 4.7k pullups), Quectel L76K GNSS to UART2 (`/dev/ttyTHS1`, Pins 8/10), and TP-Link TL-WN722N WiFi dongle to USB. | Jetson Nano 40-pin & CSI headers | `python3 -c "from edge.sensors import GPS; g = GPS('/dev/ttyTHS1'); print(g.get_current_fix())"` |
| **`[HARDWARE]`** | **Firmware Flashing & Mast Node Test:** Flash `firmware/node_n01` onto mast ESP32 and `firmware/esp32cam_trap` onto trap ESP32-CAM via CP2102. Verify AP `SIH-NODE-01` (192.168.9.1) and `/readings` endpoint on battery. | Ground Mast ESP32 + ESP32-CAM | `curl -s http://192.168.9.1/api/v1/health` |
| **`[HARDWARE]`** | **Thermal Reference Pads Calibration:** Place physical wet (water pad) and dry (sunlit dry pad) reference surfaces in MLX90640 FOV. Run setup script to write bounding boxes to `configs/thermal_refs.json`. | `configs/thermal_refs.json` | `python3 scripts/thermal_ref_setup.py --capture` |
| **`[HARDWARE]`** | **Live WiFi Switching & Client Skip Test:** Validate `edge/wifi_switch.py` switching between `SIH-FIELD` AP and `SIH-NODE-01` mast STA, ensuring `iw dev <iface> station dump` correctly detects connected mobile phones. | `edge/wifi_switch.py` | `python3 edge/wifi_switch.py --dry-run` |
| **`[HARDWARE]`** | **On-Pod NoIR Camera Bench Calibration (Lead-Time Gated):** If MidOpt DB660/850 filter arrives before demo, run `scripts/calibrate_dual_bandpass.py` before integrating sphere to measure unmixing matrix $K^{-1}$ and lift `NotImplementedError` in `core/ndvi.py`. (Otherwise, Sentinel-2 satellite NDVI serves as primary NDVI source). | `core/ndvi.py` | `python3 scripts/nano_smoke_test.py` |
| **`[SOFTWARE]`** | **CDSE Credentials & Field Polygon Setup:** Populate `/etc/sih/cdse.json` (`chmod 600`) with Copernicus Data Space OAuth credentials and set test field boundary polygon in `configs/field.json`. | `/etc/sih/cdse.json`, `configs/field.json` | `python3 scripts/fetch_satellite_ndvi.py --credentials /etc/sih/cdse.json --field-config configs/field.json` |
| **`[SOFTWARE]`** | **Systemd Production Services Installation:** Execute setup script on the physical Jetson Nano to install and enable `sih-gateway.service`, `sih-collector-boot.service`, and `sih-pipeline.service`. | `/etc/systemd/system/sih-*.service` | `sudo ./scripts/setup_nano_services.sh` |

---

## 2. Subsystem Verification Status Matrix

| Subsystem | State | PROVISIONAL Constants in Use (Name, Value, File:Line) | Physical Measurements Still Required | One Command to Test on Nano |
| :--- | :--- | :--- | :--- | :--- |
| **Model A Pipeline** (`edge/pipeline.py`, `core/rejection.py`) | `TESTED_ON_MAC` | `PROVISIONAL_TAU_BLUR = 100.0` (`configs/train_config.py:39`)<br>`PROVISIONAL_TAU_HEALTHY = 0.50` (`configs/train_config.py:44`)<br>`PROVISIONAL_NOTCROP_FRAC = 0.50` (`configs/train_config.py:45`) | Walking video capture on physical IMX219-160 RGB camera to calibrate motion blur and lighting thresholds. | `python3 edge/pipeline.py --source /dev/video0 --db-path data/edge.db --single-pass` |
| **Model B Trap Job** (`edge/trap_job.py`, `core/trap_segmentation.py`) | `TESTED_ON_MAC` | `PROVISIONAL_MARKER_SPACING_W_MM = 80.0` (`core/trap_segmentation.py:25`)<br>`PROVISIONAL_MARKER_SPACING_H_MM = 55.0` (`core/trap_segmentation.py:26`)<br>`PROVISIONAL_MAX_CARD_SATURATION = 0.30` (`core/trap_segmentation.py:34`) | Physical measurement of printed card dot spacing on procured batch using vernier calipers. | `python3 edge/trap_job.py --image tests/fixtures/sample_trap.jpg --allow-provisional` |
| **Rules Engine** (`edge/rules_engine.py`) | `TESTED_ON_MAC` | None (All 21 action templates locked to DPPQS / ICAR standards in `docs/TEMPLATE_ID_REGISTRY.md`). | None (Deterministic agronomic logic fully verified). | `pytest -v tests/test_rules_engine.py` |
| **Storage & Sync** (`edge/storage.py`) | `TESTED_ON_MAC` | None (SQLite WAL schema v1.0 frozen). | Bench stress write test on physical Sandisk High Endurance MicroSD card. | `python3 -c "from edge.storage import EdgeStorage; s = EdgeStorage('data/edge.db'); print(s.get_health())"` |
| **Offline Gateway** (`gateway/server.py`) | `TESTED_ON_MAC` | None (REST wire contract strictly compliant with `docs/PAYLOAD_CONTRACT.md`). | Local loopback testing with Android field tablet on `SIH-FIELD` AP. | `python3 gateway/server.py --host 0.0.0.0 --port 8080 --db-path data/edge.db` |
| **Ground Mast Collector** (`edge/mast_collector.py`) | `TESTED_ON_MAC` | None (Exact pull contract aligned with Guide §6 and firmware). | Live pull over WiFi against physical ESP32 Ground Mast node (`192.168.9.1`). | `python3 edge/mast_collector.py --base-url http://192.168.9.1/api/v1 --db-path data/edge.db` |
| **WiFi AP/STA Switching** (`edge/wifi_switch.py`) | `UNVERIFIED_ON_HARDWARE` | None (nmcli wrapper implemented with finally block safety guarantee). | Hardware validation with TP-Link TL-WN722N (Atheros AR9271) USB WiFi dongle on Nano. | `python3 edge/wifi_switch.py --dry-run` |
| **GPS UART Reader** (`edge/sensors.py`) | `UNVERIFIED_ON_HARDWARE` | None (Pure Python NMEA-0183 GGA/RMC parser verified with synthetic sentences). | Live outdoor test with Quectel L76K GNSS module connected to `/dev/ttyTHS1`. | `python3 -c "from edge.sensors import GPS; g = GPS('/dev/ttyTHS1'); print(g.get_current_fix())"` |
| **Thermal / Reference CWSI** (`edge/thermal_capture.py`, `core/thermal.py`) | `UNVERIFIED_ON_HARDWARE` | `PROVISIONAL_MIN_REF_GAP_C = 1.5` (`core/thermal.py:184`)<br>`PROVISIONAL_MAX_REF_STD_C = 1.5` (`core/thermal.py:185`)<br>`PROVISIONAL_BIMODAL_GAP_C = 4.0` (`core/thermal.py:48`)<br>`PROVISIONAL_OTSU_MIN_INTERCLASS_VARIANCE_RATIO = 0.85` (`core/thermal.py:67`) | Bench test with physical MLX90640 FIR array on `/dev/i2c-1` (400 kHz) with wet/dry calibration pads. | `python3 scripts/thermal_ref_setup.py --capture` |
| **On-Pod NoIR NDVI** (`core/ndvi.py`) | `HARDWARE_GATED` | `PROVISIONAL_PANEL_REFLECTANCES = (0.05, 0.50, 0.84)` (`core/ndvi.py:67`)<br>`calib_matrix = None` (`core/ndvi.py:75, 101` raises `NotImplementedError`) | Spectrophotometer / integrating sphere measurement of IMX219-77IR + MidOpt DB660/850 filter unmixing matrix $K^{-1}$. | `python3 scripts/nano_smoke_test.py` |
| **Sentinel-2 Satellite NDVI** (`edge/ndvi_satellite.py`, `scripts/fetch_satellite_ndvi.py`) | `TESTED_ON_MAC` | `PROVISIONAL_MIN_VALID_PIXEL_FRACTION = 0.50` (`edge/ndvi_satellite.py:27`) | Internet connectivity to Copernicus Data Space Ecosystem (CDSE) with valid OAuth credentials in `/etc/sih/cdse.json`. | `python3 scripts/fetch_satellite_ndvi.py --credentials /etc/sih/cdse.json --field-config configs/field.json` |
| **Irrigation (FAO-56 HS & Ra)** (`edge/irrigation_model.py`) | `TESTED_ON_MAC` | `PROVISIONAL_PADDY_PERCOLATION_MM_DAY = 5.0` (`edge/irrigation_model.py:119`)<br>`PROVISIONAL_PADDY_SATURATION_MM = 200.0` (`edge/irrigation_model.py:117`) | Infiltrometer percolation measurement on specific field soil after puddling. | `pytest -v tests/test_irrigation_wiring.py` |
| **Growth Stage & Phenology** (`core/growth_stage.py`) | `TESTED_ON_MAC` | `PROVISIONAL_DEFAULT_CYCLE_DAYS`: rice 150d, wheat 120d, sugarcane 280d (`configs/train_config.py:53`) | User variety selection in mobile app to replace generic FAO-56 default cycle days. | `pytest -v tests/test_pipeline_runtime.py` |
| **Systemd Services & Boot** (`scripts/setup_nano_services.sh`) | `UNVERIFIED_ON_HARDWARE` | `TIMER_INTERVAL = 60min` (`scripts/setup_nano_services.sh:159`) | Boot cycle execution and service inspection on physical Jetson Nano. | `sudo ./scripts/setup_nano_services.sh` |
| **Mast & Camera Firmware** (`firmware/node_n01`, `firmware/esp32cam_trap`) | `TESTED_ON_MAC` | None (Both sketches compile cleanly under `arduino-cli` with zero errors). | Flashing firmware onto physical ESP32 and ESP32-CAM via CP2102 USB-UART programmer. | `arduino-cli compile --fqbn esp32:esp32:esp32 firmware/node_n01` |

---

## 3. Summary of Physical Next Steps for Hardware Team

1. **Jetson Nano Pod Assembly:** Connect IMX219-160 RGB camera to CSI-0, MLX90640 to I2C-1 (Pins 3/5 with 4.7k pullups), Quectel L76K GPS to UART (Pins 8/10), and TP-Link AR9271 WiFi dongle to USB.
2. **Thermal Calibration:** Run `python3 scripts/thermal_ref_setup.py --capture`, inspect PNG, and enter wet/dry bounding box pixel coordinates.
3. **Satellite NDVI Setup:** Place Copernicus CDSE OAuth client ID/secret in `/etc/sih/cdse.json` (mode 600) and define field boundary in `configs/field.json`.
4. **Service Startup:** Run `sudo ./scripts/setup_nano_services.sh` and enable `sih-gateway.service`, `sih-collector-boot.service`, and `sih-pipeline.service`.
