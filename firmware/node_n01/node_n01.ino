/*
 * node_n01.ino - SIH 2026 Fixed Mast Node N01 (ESP32 DevKit V1)
 * Core: esp32 2.0.17 | Board: ESP32 Dev Module | Partition: No OTA (2MB APP/2MB SPIFFS)
 * Libs: Adafruit SHT4x, Adafruit ADS1X15, Adafruit MLX90614, BH1750 (claws),
 *       RTClib (Adafruit), ArduinoJson 7.x
 * STATUS: reference firmware rev 2, still NOT compiled or bench-tested.
 */
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LittleFS.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <RTClib.h>
#include <Adafruit_SHT4x.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MLX90614.h>
#include <BH1750.h>
#include <esp_task_wdt.h>

// ================= CONFIGURATION =================
#define FW_VERSION "n01-1.0.0"
const char* NODE_ID   = "N01";
const char* FIELD_ID  = "F01";
const char* CAM_ID    = "N01-CAM";
const char* AP_SSID   = "SIH-NODE-01";
const char* AP_PASS   = "sih12345";          // must match Nano collector + camera
const IPAddress AP_IP(192, 168, 9, 1);
const IPAddress AP_MASK(255, 255, 255, 0);

const int PIN_SDA = 21, PIN_SCL = 22, PIN_CAM_WAKE = 25, PIN_LED = 2;
const uint32_t I2C_HZ = 50000;                  // slow bus for long cables

const uint32_t SAMPLE_INTERVAL_MS = 10UL * 60UL * 1000UL;   // 10 min
const int   ADS_AVG      = 8;
const float BATT_DIVIDER = (100.0f + 22.0f) / 22.0f;        // tune against a multimeter
const float BATT_LOW_V   = 9.9f;

// Trap capture slots in UTC: 04:30Z = 10:00 IST, 09:30Z = 15:00 IST
const uint8_t TRAP_SLOTS[][2] = {{4, 30}, {9, 30}};
const int      N_SLOTS        = 2;
const uint32_t CAM_PULSE_MS   = 3000;
const uint32_t UPLOAD_WAIT_MS = 90000;
const uint32_t RETRY_DELAY_MS = 5UL * 60UL * 1000UL;

const size_t MAX_JPEG_BYTES   = 450000;
const int    MAX_TRAP_IMAGES  = 3;
const size_t LOG_ROTATE_BYTES = 200000;
const char*  LOG_CUR = "/log_cur.jsonl";
const char*  LOG_OLD = "/log_old.jsonl";
const char*  UP_TMP  = "/trap/upload.tmp";
const int    MAX_RECORDS_PER_REQ = 500;

// ================= GLOBALS =================
WebServer server(80);
Preferences prefs;
RTC_DS3231 rtc;
Adafruit_SHT4x sht4;
Adafruit_ADS1115 ads;
Adafruit_MLX90614 mlx;
BH1750 lightMeter;

bool okSht = false, okAds = false, okMlx = false, okBh = false, okRtc = false;
bool rtcValid = false;
float lastBattV = NAN;
uint32_t lastSampleMs = 0;
uint32_t logEpoch = 0;

enum CamState { CAM_IDLE, CAM_PULSING, CAM_WAITING, CAM_RETRY_WAIT };
CamState camState = CAM_IDLE;
uint32_t camT0 = 0;
bool camRetried = false;
bool camManual = false;
String trapStatus = "NONE";   // NONE | OK | NO_UPLOAD | RTC_INVALID | LOW_BATTERY

File upFile;
size_t upBytes = 0;
int upError = 0;              // 0 ok, 1 too big, 2 no space, 3 fs error, 4 unknown device

// ================= HELPERS =================
String isoNow() {
  DateTime t = rtc.now();
  char b[25];
  snprintf(b, sizeof(b), "%04d-%02d-%02dT%02d:%02d:%02dZ",
           t.year(), t.month(), t.day(), t.hour(), t.minute(), t.second());
  return String(b);
}

void putUtc(JsonDocument& d, const char* key) {
  if (okRtc) d[key] = isoNow(); else d[key] = nullptr;
}

void putNum(JsonDocument& d, const char* key, float v, int dec) {
  if (isnan(v)) d[key] = nullptr;
  else d[key] = serialized(String(v, dec));
}

const char* okStr(bool ok) { return ok ? "OK" : "ERR"; }

const char* camStateName() {
  switch (camState) {
    case CAM_PULSING:    return "PULSING";
    case CAM_WAITING:    return "WAITING_UPLOAD";
    case CAM_RETRY_WAIT: return "RETRY_WAIT";
    default:             return "IDLE";
  }
}

void sendJson(int code, JsonDocument& d) {
  String out;
  serializeJson(d, out);
  server.send(code, "application/json", out);
}

void sendErr(int code, const char* msg) {
  JsonDocument d;
  d["error"] = msg;
  sendJson(code, d);
}

uint32_t parseSeq(const String& line) {
  if (!line.startsWith("{\"seq\":")) return 0;
  return strtoul(line.c_str() + 7, nullptr, 10);
}

// ================= SENSORS =================
void initSensors() {
  if (!okSht) {
    okSht = sht4.begin(&Wire);
    if (okSht) { sht4.setPrecision(SHT4X_HIGH_PRECISION); sht4.setHeater(SHT4X_NO_HEATER); }
  }
  if (!okAds) {
    okAds = ads.begin(0x48, &Wire);
    if (okAds) ads.setGain(GAIN_ONE);                 // +/-4.096 V range
  }
  if (!okMlx) okMlx = mlx.begin();
  if (!okBh) {
    okBh = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire);
    if (okBh) lightMeter.setMTreg(31);                // extends range to full sunlight
  }
  if (!okRtc) okRtc = rtc.begin(&Wire);
  Wire.setClock(I2C_HZ);
}

float readAdsVolts(int ch) {
  if (!okAds) return NAN;
  long sum = 0;
  for (int i = 0; i < ADS_AVG; i++) sum += ads.readADC_SingleEnded(ch);
  return ads.computeVolts((int16_t)(sum / ADS_AVG));
}

void takeSample() {
  initSensors();                                      // re-tries anything that failed
  JsonDocument d;
  uint32_t seq = prefs.getUInt("seq", 0) + 1;
  d["seq"] = seq;                                     // MUST stay the first key
  d["node_id"] = NODE_ID;
  d["field_id"] = FIELD_ID;
  putUtc(d, "utc");
  d["rtc_valid"] = rtcValid;
  d["uptime_s"] = millis() / 1000;

  sensors_event_t hum, temp;
  if (okSht && sht4.getEvent(&hum, &temp)) {
    putNum(d, "air_temp_c", temp.temperature, 2);
    putNum(d, "rh_pct", hum.relative_humidity, 2);
  } else { okSht = false; d["air_temp_c"] = nullptr; d["rh_pct"] = nullptr; }

  float irObj = NAN, irAmb = NAN;
  if (okMlx) {
    irObj = mlx.readObjectTempC();
    irAmb = mlx.readAmbientTempC();
    bool plausible = !isnan(irObj) && !isnan(irAmb) &&
                     irObj > -40 && irObj < 125 && irAmb > -40 && irAmb < 125;
    if (!plausible) { okMlx = false; irObj = NAN; irAmb = NAN; }
  }
  putNum(d, "ir_object_c", irObj, 2);
  putNum(d, "ir_ambient_c", irAmb, 2);

  float lux = NAN;
  if (okBh) {
    lux = lightMeter.readLightLevel();
    if (lux < 0) { okBh = false; lux = NAN; }
  }
  putNum(d, "lux", lux, 1);

  float s1 = readAdsVolts(0), s2 = readAdsVolts(1), vb = readAdsVolts(2);
  if (okAds && (s1 < -0.1 || s1 > 4.0)) { okAds = false; s1 = s2 = vb = NAN; }
  putNum(d, "soil1_v", s1, 4);
  putNum(d, "soil2_v", s2, 4);
  lastBattV = isnan(vb) ? NAN : vb * BATT_DIVIDER;
  putNum(d, "battery_v", lastBattV, 2);

  JsonObject st = d["status"].to<JsonObject>();
  st["sht40"] = okStr(okSht);
  st["mlx90614"] = okStr(okMlx);
  st["bh1750"] = okStr(okBh);
  st["ads1115"] = okStr(okAds);
  st["rtc"] = okRtc ? (rtcValid ? "OK" : "NOT_SET") : "ERR";

  String line;
  serializeJson(d, line);
  File f = LittleFS.open(LOG_CUR, FILE_APPEND);
  if (f) {
    f.print(line); f.print('\n');
    size_t sz = f.size();
    f.close();
    if (sz > LOG_ROTATE_BYTES) {
      LittleFS.remove(LOG_OLD);
      LittleFS.rename(LOG_CUR, LOG_OLD);
    }
    prefs.putUInt("seq", seq);
  } else {
    Serial.println("LOG WRITE FAILED");
  }
  Serial.println(line);
  digitalWrite(PIN_LED, HIGH); delay(50); digitalWrite(PIN_LED, LOW);
}

// ================= TRAP CAMERA CONTROL =================
bool startCapture(bool manual) {
  if (camState != CAM_IDLE) return false;
  camManual = manual;
  digitalWrite(PIN_CAM_WAKE, HIGH);
  camState = CAM_PULSING;
  camT0 = millis();
  Serial.println(manual ? "trap: manual capture" : "trap: scheduled capture");
  return true;
}

void camTick() {
  uint32_t now = millis();
  switch (camState) {
    case CAM_PULSING:
      if (now - camT0 >= CAM_PULSE_MS) {
        digitalWrite(PIN_CAM_WAKE, LOW);
        camState = CAM_WAITING; camT0 = now;
      }
      break;
    case CAM_WAITING:
      if (now - camT0 >= UPLOAD_WAIT_MS) {
        if (!camRetried) { camRetried = true; camState = CAM_RETRY_WAIT; camT0 = now; }
        else { trapStatus = "NO_UPLOAD"; camState = CAM_IDLE; camRetried = false; }
      }
      break;
    case CAM_RETRY_WAIT:
      if (now - camT0 >= RETRY_DELAY_MS) { camState = CAM_IDLE; startCapture(camManual); }
      break;
    default: break;
  }
}

void scheduleTick() {
  static uint32_t lastCheck = 0;
  if (millis() - lastCheck < 30000) return;
  lastCheck = millis();
  if (!okRtc || !rtcValid) {                 // no trusted time -> no automatic captures
    if (trapStatus == "NONE") trapStatus = "RTC_INVALID";
    return;
  }
  DateTime t = rtc.now();
  uint32_t lastKey = prefs.getUInt("trapKey", 0);
  int nowMin = t.hour() * 60 + t.minute();
  for (int i = 0; i < N_SLOTS; i++) {
    uint32_t key = (uint32_t)(t.year() % 100) * 100000UL + t.month() * 1000UL + t.day() * 10UL + i;
    int slotMin = TRAP_SLOTS[i][0] * 60 + TRAP_SLOTS[i][1];
    // 60-minute window so a reboot hours later does not fire an old slot
    if (nowMin >= slotMin && nowMin < slotMin + 60 && key > lastKey) {
      prefs.putUInt("trapKey", key);
      if (!isnan(lastBattV) && lastBattV < BATT_LOW_V) { trapStatus = "LOW_BATTERY"; return; }
      camRetried = false;
      startCapture(false);
      return;
    }
  }
}

void pruneTrapImages(int keep) {
  while (true) {
    int count = 0; uint32_t minId = UINT32_MAX;
    File root = LittleFS.open("/trap");
    File f = root.openNextFile();
    while (f) {
      String n = f.name();                   // e.g. "T12.jpg"
      if (n.startsWith("T") && n.endsWith(".jpg")) {
        uint32_t id = strtoul(n.c_str() + 1, nullptr, 10);
        count++; if (id < minId) minId = id;
      }
      f = root.openNextFile();
    }
    root.close();
    if (count <= keep || minId == UINT32_MAX) return;
    LittleFS.remove("/trap/T" + String(minId) + ".jpg");
    LittleFS.remove("/trap/T" + String(minId) + ".json");
  }
}

// ================= HTTP HANDLERS =================
void hHealth() {
  JsonDocument d;
  d["node_id"] = NODE_ID;
  d["field_id"] = FIELD_ID;
  d["fw_version"] = FW_VERSION;
  d["log_epoch"] = logEpoch;
  putUtc(d, "utc");
  d["rtc_valid"] = rtcValid;
  d["uptime_s"] = millis() / 1000;
  putNum(d, "battery_v", lastBattV, 2);
  d["last_seq"] = prefs.getUInt("seq", 0);
  d["sample_interval_s"] = SAMPLE_INTERVAL_MS / 1000;
  JsonObject st = d["sensors"].to<JsonObject>();
  st["sht40"] = okStr(okSht);
  st["mlx90614"] = okStr(okMlx);
  st["bh1750"] = okStr(okBh);
  st["ads1115"] = okStr(okAds);
  st["rtc"] = okRtc ? (rtcValid ? "OK" : "NOT_SET") : "ERR";
  JsonObject tr = d["trap"].to<JsonObject>();
  tr["status"] = trapStatus;
  tr["cam_state"] = camStateName();
  tr["last_trap_id"] = prefs.getUInt("trapId", 0);
  d["fs_used_b"] = LittleFS.usedBytes();
  d["fs_total_b"] = LittleFS.totalBytes();
  d["wifi_clients"] = WiFi.softAPgetStationNum();
  sendJson(200, d);
}

void hReadings() {
  uint32_t since = server.hasArg("since") ? strtoul(server.arg("since").c_str(), nullptr, 10) : 0;
  int limit = server.hasArg("limit") ? server.arg("limit").toInt() : MAX_RECORDS_PER_REQ;
  if (limit <= 0 || limit > MAX_RECORDS_PER_REQ) limit = MAX_RECORDS_PER_REQ;

  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "application/json", "");
  server.sendContent(String("{\"node_id\":\"") + NODE_ID + "\",\"log_epoch\":" + String(logEpoch) + ",\"records\":[");
  int n = 0; uint32_t lastSeq = since; bool more = false; bool first = true;
  const char* files[2] = {LOG_OLD, LOG_CUR};           // oldest first -> ascending seq
  for (int k = 0; k < 2 && !more; k++) {
    File fh = LittleFS.open(files[k], FILE_READ);
    if (!fh) continue;
    while (fh.available()) {
      String line = fh.readStringUntil('\n');
      uint32_t s = parseSeq(line);
      if (s == 0 || s <= since) continue;
      if (n >= limit) { more = true; break; }
      if (!first) server.sendContent(",");
      first = false;
      server.sendContent(line);
      n++; lastSeq = s;
      esp_task_wdt_reset();
    }
    fh.close();
  }
  server.sendContent("],\"count\":" + String(n) + ",\"next_since\":" + String(lastSeq) +
                     ",\"truncated\":" + (more ? "true" : "false") + "}");
  server.sendContent("");                               // end of chunked response
}

void hTrapList() {
  JsonDocument d;
  JsonArray arr = d["images"].to<JsonArray>();
  File root = LittleFS.open("/trap");
  File f = root.openNextFile();
  while (f) {
    String n = f.name();
    if (n.endsWith(".json")) {
      JsonDocument m;
      if (!deserializeJson(m, f)) arr.add(m);
    }
    f = root.openNextFile();
  }
  root.close();
  sendJson(200, d);
}

void hTrapLatest() {
  uint32_t id = prefs.getUInt("trapId", 0);
  File f = LittleFS.open("/trap/T" + String(id) + ".json", FILE_READ);
  if (id == 0 || !f) { sendErr(404, "no trap image yet"); return; }
  server.streamFile(f, "application/json");
  f.close();
}

void hTrapImage() {
  if (!server.hasArg("id")) { sendErr(400, "missing id"); return; }
  String path = "/trap/T" + String(strtoul(server.arg("id").c_str(), nullptr, 10)) + ".jpg";
  File f = LittleFS.open(path, FILE_READ);
  if (!f) { sendErr(404, "image not found"); return; }
  server.streamFile(f, "image/jpeg");
  f.close();
}

void hTrapTrigger() {
  if (startCapture(true)) {
    camRetried = true;                                 // manual captures do not auto-retry
    JsonDocument d; d["started"] = true; sendJson(202, d);
  } else sendErr(409, "capture already in progress");
}

void hTime() {
  if (!server.hasArg("utc")) { sendErr(400, "missing utc"); return; }
  uint32_t e = strtoul(server.arg("utc").c_str(), nullptr, 10);
  if (e < 1735689600UL || e > 4102444800UL) { sendErr(400, "implausible time"); return; }
  if (!okRtc) { sendErr(503, "rtc not found"); return; }
  rtc.adjust(DateTime(e));                             // also clears the lost-power flag
  rtcValid = true;
  if (trapStatus == "RTC_INVALID") trapStatus = "NONE";
  JsonDocument d;
  d["ok"] = true; putUtc(d, "utc"); d["rtc_valid"] = true;
  sendJson(200, d);
}

void hUploadChunk() {
  HTTPUpload& u = server.upload();
  if (u.status == UPLOAD_FILE_START) {
    upBytes = 0; upError = 0;
    if (server.header("X-Device-Id") != CAM_ID) {   // unauthorised: open no file, prune nothing
      upError = 4;
      return;
    }
    pruneTrapImages(MAX_TRAP_IMAGES - 1);             // authorised only; must stay before the write (flash is tight)
    LittleFS.remove(UP_TMP);
    if (LittleFS.totalBytes() - LittleFS.usedBytes() < MAX_JPEG_BYTES + 20000) upError = 2;
    else { upFile = LittleFS.open(UP_TMP, FILE_WRITE); if (!upFile) upError = 3; }
  } else if (u.status == UPLOAD_FILE_WRITE) {
    upBytes += u.currentSize;
    if (upBytes > MAX_JPEG_BYTES) upError = 1;
    if (upError == 0 && upFile) upFile.write(u.buf, u.currentSize);
    esp_task_wdt_reset();
  } else if (u.status == UPLOAD_FILE_END || u.status == UPLOAD_FILE_ABORTED) {
    if (upFile) upFile.close();
    if (u.status == UPLOAD_FILE_ABORTED) upError = 3;
  }
}

void hUploadDone() {
  if (upError == 4 || server.header("X-Device-Id") != CAM_ID) { sendErr(403, "unknown device"); return; }
  if (upError == 1) { LittleFS.remove(UP_TMP); sendErr(413, "image too large"); return; }
  if (upError != 0 || upBytes == 0) { LittleFS.remove(UP_TMP); sendErr(400, "upload failed"); return; }

  uint32_t id = prefs.getUInt("trapId", 0) + 1;
  String jp = "/trap/T" + String(id) + ".jpg";
  String js = "/trap/T" + String(id) + ".json";
  if (!LittleFS.rename(UP_TMP, jp)) { sendErr(500, "store failed"); return; }

  JsonDocument m;
  m["trap_id"] = id;
  m["node_id"] = NODE_ID;
  putUtc(m, "received_utc");
  m["rtc_valid"] = rtcValid;
  m["bytes"] = upBytes;
  m["sensor"] = server.header("X-Sensor");
  m["frame"] = server.header("X-Frame");
  m["cam_wake"] = server.header("X-Wake");
  m["trigger"] = (camState == CAM_IDLE) ? "unsolicited" : (camManual ? "manual" : "scheduled");
  File f = LittleFS.open(js, FILE_WRITE);
  if (f) { serializeJson(m, f); f.close(); }
  prefs.putUInt("trapId", id);

  camState = CAM_IDLE; camRetried = false; trapStatus = "OK";
  sendJson(201, m);
}

// ================= SERIAL BENCH COMMANDS =================
void serialTick() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd == "S") takeSample();
  else if (cmd == "C") { if (startCapture(true)) camRetried = true; else Serial.println("busy"); }
  else if (cmd == "H") {
    Serial.printf("rtcOk=%d rtcValid=%d sht=%d ads=%d mlx=%d bh=%d batt=%.2f seq=%u trap=%s cam=%s\n",
                  okRtc, rtcValid, okSht, okAds, okMlx, okBh, lastBattV,
                  prefs.getUInt("seq", 0), trapStatus.c_str(), camStateName());
    if (okRtc) Serial.println(isoNow());
  } else if (cmd.startsWith("T") && okRtc) {
    uint32_t e = strtoul(cmd.c_str() + 1, nullptr, 10);
    if (e > 1735689600UL) { rtc.adjust(DateTime(e)); rtcValid = true; Serial.println("RTC set: " + isoNow()); }
    else Serial.println("bad time");
  }
}

// ================= SETUP / LOOP =================
void setup() {
  pinMode(PIN_CAM_WAKE, OUTPUT);
  digitalWrite(PIN_CAM_WAKE, LOW);                    // camera must never see a stray wake
  pinMode(PIN_LED, OUTPUT);
  Serial.begin(115200);
  delay(300);
  Serial.println("\nSIH mast node " FW_VERSION);

  prefs.begin("n01", false);
  logEpoch = prefs.getUInt("epoch", 0);
  if (logEpoch == 0) { logEpoch = esp_random() | 1; prefs.putUInt("epoch", logEpoch); }

  if (!LittleFS.begin(true)) Serial.println("LittleFS mount FAILED");
  LittleFS.mkdir("/trap");
  LittleFS.remove(UP_TMP);

  Wire.begin(PIN_SDA, PIN_SCL, I2C_HZ);
  initSensors();
  if (okRtc) {
    rtcValid = !rtc.lostPower() && rtc.now().year() >= 2025;
  }
  Serial.printf("sht=%d ads=%d mlx=%d bh=%d rtc=%d rtcValid=%d\n",
                okSht, okAds, okMlx, okBh, okRtc, rtcValid);

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(AP_IP, AP_IP, AP_MASK);
  WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 4);             // channel 6, visible, max 4 clients
  Serial.println("AP up: " + WiFi.softAPIP().toString());

  const char* hdrs[] = {"X-Device-Id", "X-Sensor", "X-Frame", "X-Wake"};
  server.collectHeaders(hdrs, 4);
  server.on("/api/v1/health", HTTP_GET, hHealth);
  server.on("/api/v1/readings", HTTP_GET, hReadings);
  server.on("/api/v1/trap/list", HTTP_GET, hTrapList);
  server.on("/api/v1/trap/latest", HTTP_GET, hTrapLatest);
  server.on("/api/v1/trap/image", HTTP_GET, hTrapImage);
  server.on("/api/v1/trap/trigger", HTTP_POST, hTrapTrigger);
  server.on("/api/v1/time", HTTP_POST, hTime);
  server.on("/api/v1/trap/upload", HTTP_POST, hUploadDone, hUploadChunk);
  server.onNotFound([]() { sendErr(404, "not found"); });
  server.begin();

  esp_task_wdt_init(30, true);                          // reboot if the loop hangs 30 s
  esp_task_wdt_add(NULL);

  lastSampleMs = millis() - SAMPLE_INTERVAL_MS + 5000;  // first sample 5 s after boot
}

void loop() {
  esp_task_wdt_reset();
  server.handleClient();
  camTick();
  scheduleTick();
  if (millis() - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = millis();
    takeSample();
  }
  serialTick();
  delay(2);
}
