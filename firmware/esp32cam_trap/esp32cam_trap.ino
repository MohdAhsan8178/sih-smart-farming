/*
 * esp32cam_trap.ino - SIH 2026 sticky-trap camera N01-CAM (AI-Thinker ESP32-CAM)
 * Core: esp32 2.0.17 | Board: AI Thinker ESP32-CAM | Partition: Huge APP (3MB No OTA)
 * Works with OV2640 or OV3660 modules (auto-detected, reported in X-Sensor).
 * No on-device inference. No timekeeping (the mast ESP32 stamps the time).
 * STATUS: reference firmware, NOT yet compiled or bench-tested.
 */
#include "esp_camera.h"
#include <WiFi.h>
#include "driver/rtc_io.h"

// ================= CONFIGURATION =================
const char* WIFI_SSID  = "SIH-NODE-01";
const char* WIFI_PASS  = "sih12345";          // must match node_n01.ino
const char* NODE_HOST  = "192.168.9.1";
const uint16_t NODE_PORT = 80;
const char* DEVICE_ID  = "N01-CAM";

const bool        CAPTURE_ON_POWER_ON = true;    // one test shot at power-up (bench aid)
const gpio_num_t  WAKE_PIN   = GPIO_NUM_13;      // from mast ESP32 GPIO25 via 1k
const gpio_num_t  FLASH_PIN  = GPIO_NUM_4;       // on-board flash LED, kept OFF
const framesize_t FRAME      = FRAMESIZE_UXGA;   // 1600x1200. NEVER change after calibration
const char*       FRAME_NAME = "UXGA_1600x1200";
const int         JPEG_QUALITY  = 12;            // 10..63, lower = larger file; keep < 450 kB
const int         WARMUP_FRAMES = 5;             // lets auto-exposure settle
const uint32_t    WIFI_TIMEOUT_MS = 15000;
const uint32_t    FALLBACK_SLEEP_S = 600;        // used only if the wake line is stuck HIGH

// ================= AI-THINKER PIN MAP =================
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

bool initCamera(String& sensorName) {
  if (!psramFound()) { Serial.println("NO PSRAM: this board cannot capture 1600x1200"); return false; }
  camera_config_t c;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAME;
  c.jpeg_quality = JPEG_QUALITY;
  c.fb_count     = 1;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;

  sensor_t* s = esp_camera_sensor_get();
  if (s->id.PID == OV2640_PID)      sensorName = "OV2640";
  else if (s->id.PID == OV3660_PID) sensorName = "OV3660";
  else                              sensorName = "PID_0x" + String(s->id.PID, HEX);
  s->set_framesize(s, FRAME);        // OV3660 starts at a larger native size; force UXGA
  return true;
}

bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_TIMEOUT_MS) delay(200);
  bool ok = WiFi.status() == WL_CONNECTED;
  Serial.println(ok ? "wifi OK, ip " + WiFi.localIP().toString() : String("wifi FAILED"));
  return ok;
}

bool uploadJpeg(camera_fb_t* fb, const String& sensorName, const String& wake) {
  WiFiClient client;
  client.setTimeout(10);
  if (!client.connect(NODE_HOST, NODE_PORT)) { Serial.println("connect FAILED"); return false; }

  const String boundary = "----sihTrapBoundary7f3a";
  const String head = "--" + boundary + "\r\n"
                      "Content-Disposition: form-data; name=\"image\"; filename=\"trap.jpg\"\r\n"
                      "Content-Type: image/jpeg\r\n\r\n";
  const String tail = "\r\n--" + boundary + "--\r\n";
  size_t total = head.length() + fb->len + tail.length();

  client.print(String("POST /api/v1/trap/upload HTTP/1.1\r\n") +
               "Host: " + NODE_HOST + "\r\n" +
               "X-Device-Id: " + DEVICE_ID + "\r\n" +
               "X-Sensor: " + sensorName + "\r\n" +
               "X-Frame: " + FRAME_NAME + "\r\n" +
               "X-Wake: " + wake + "\r\n" +
               "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n" +
               "Content-Length: " + String(total) + "\r\n" +
               "Connection: close\r\n\r\n");
  client.print(head);
  size_t sent = 0;
  while (sent < fb->len) {
    size_t n = fb->len - sent; if (n > 4096) n = 4096;
    size_t w = client.write(fb->buf + sent, n);
    if (w == 0) { Serial.println("write FAILED"); client.stop(); return false; }
    sent += w;
  }
  client.print(tail);

  uint32_t t0 = millis();
  while (!client.available() && client.connected() && millis() - t0 < 15000) delay(20);
  String status = client.readStringUntil('\n');     // e.g. "HTTP/1.1 201 Created"
  Serial.println("node replied: " + status);
  client.stop();
  return status.indexOf(" 201") > 0;
}

void captureAndUpload(const String& wake) {
  String sensorName;
  if (!initCamera(sensorName)) { Serial.println("camera init FAILED"); return; }
  for (int i = 0; i < WARMUP_FRAMES; i++) {
    camera_fb_t* f = esp_camera_fb_get();
    if (f) esp_camera_fb_return(f);
    delay(200);
  }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("capture FAILED"); esp_camera_deinit(); return; }
  Serial.printf("captured %u bytes, sensor %s\n", (unsigned)fb->len, sensorName.c_str());
  if (fb->len > 450000) Serial.println("WARNING: image > 450 kB, node will reject it; raise JPEG_QUALITY");

  bool ok = false;
  if (connectWiFi()) {
    for (int attempt = 0; attempt < 2 && !ok; attempt++) {
      ok = uploadJpeg(fb, sensorName, wake);
      if (!ok) delay(1500);
    }
  }
  Serial.println(ok ? "upload OK" : "upload FAILED");
  esp_camera_fb_return(fb);
  esp_camera_deinit();
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
}

void goToSleep() {
  // Wait for the wake line to drop, otherwise ext0 would wake us again at once.
  pinMode(WAKE_PIN, INPUT_PULLDOWN);
  uint32_t t0 = millis();
  while (digitalRead(WAKE_PIN) == HIGH && millis() - t0 < 15000) delay(50);
  bool lineStuckHigh = digitalRead(WAKE_PIN) == HIGH;

  digitalWrite(FLASH_PIN, LOW);
  rtc_gpio_hold_en(FLASH_PIN);                       // keep the flash LED off while asleep

  if (lineStuckHigh) {
    Serial.println("wake line stuck HIGH - timer sleep instead");
    esp_sleep_enable_timer_wakeup((uint64_t)FALLBACK_SLEEP_S * 1000000ULL);
  } else {
    rtc_gpio_pullup_dis(WAKE_PIN);
    rtc_gpio_pulldown_en(WAKE_PIN);                  // idle LOW even if the wire comes loose
    esp_sleep_enable_ext0_wakeup(WAKE_PIN, 1);       // wake when the line goes HIGH
  }
  Serial.println("sleeping");
  Serial.flush();
  esp_deep_sleep_start();
}

void setup() {
  Serial.begin(115200);
  delay(200);
  rtc_gpio_hold_dis(FLASH_PIN);
  pinMode(FLASH_PIN, OUTPUT);
  digitalWrite(FLASH_PIN, LOW);

  esp_sleep_wakeup_cause_t why = esp_sleep_get_wakeup_cause();
  bool woke = (why == ESP_SLEEP_WAKEUP_EXT0);
  String wake = woke ? "ext0" : (why == ESP_SLEEP_WAKEUP_TIMER ? "timer" : "power_on");
  Serial.println("\nN01-CAM boot, wake=" + wake);

  if (woke || (wake == "power_on" && CAPTURE_ON_POWER_ON)) captureAndUpload(wake);
  goToSleep();
}

void loop() {}   // never reached
