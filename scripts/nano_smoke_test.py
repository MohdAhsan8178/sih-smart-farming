#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/nano_smoke_test.py

Automated hardware & runtime smoke test for NVIDIA Jetson Nano deployment.
STRICT PYTHON 3.6 COMPATIBLE:
- No third-party imports at module top level.
- Each third-party module imported inside try/except.
- Writes all output to both stdout and a timestamped log file.
"""
import sys
import os
import time
import shutil
import sqlite3
from datetime import datetime

# Determine repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, "nano_smoke_{}.log".format(TIMESTAMP))


class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def print_check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    pad = 55 - len(name)
    if pad < 2:
        pad = 2
    dots = "." * pad
    line = "  {} {} [{}]".format(name, dots, status)
    if detail:
        line += " - {}".format(detail)
    print(line)
    return passed


def check_python_version():
    major = sys.version_info[0]
    minor = sys.version_info[1]
    micro = sys.version_info[2]
    ver_str = "{}.{}.{}".format(major, minor, micro)
    passed = (major == 3 and minor == 6)
    detail = "Found {}".format(ver_str)
    return print_check("Python 3.6.x Runtime", passed, detail)


def check_imports():
    results = []
    
    # numpy
    try:
        import numpy as np
        results.append(print_check("import numpy", True, "v" + str(np.__version__)))
    except Exception as e:
        results.append(print_check("import numpy", False, str(e)))

    # cv2
    try:
        import cv2
        results.append(print_check("import cv2", True, "v" + str(cv2.__version__)))
    except Exception as e:
        results.append(print_check("import cv2", False, str(e)))

    # tensorrt
    try:
        import tensorrt as trt
        results.append(print_check("import tensorrt", True, "v" + str(trt.__version__)))
    except Exception as e:
        results.append(print_check("import tensorrt", False, str(e)))

    # pycuda.autoinit
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        results.append(print_check("import pycuda.autoinit", True, "CUDA initialized"))
    except Exception as e:
        results.append(print_check("import pycuda.autoinit", False, str(e)))

    # onnxruntime
    try:
        import onnxruntime as ort
        results.append(print_check("import onnxruntime", True, "v" + str(ort.__version__)))
    except Exception as e:
        results.append(print_check("import onnxruntime", False, str(e)))

    # edge.trap_job
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import edge.trap_job as trap_job
        results.append(print_check("import edge.trap_job", True, "Model B trap job loaded"))
    except Exception as e:
        results.append(print_check("import edge.trap_job", False, str(e)))

    # gateway.server
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import gateway.server as gw_server
        results.append(print_check("import gateway.server", True, "Gateway server loaded"))
    except Exception as e:
        results.append(print_check("import gateway.server", False, str(e)))

    return all(results)



def check_firmware():
    fw_path = "/lib/firmware/ath9k_htc/htc_9271.fw"
    exists = os.path.exists(fw_path)
    detail = "Found at {}".format(fw_path) if exists else "Missing at {}".format(fw_path)
    return print_check("TP-Link TL-WN722N Firmware", exists, detail)


def check_gps_uart(port="/dev/ttyTHS1", timeout_s=10.0):
    """
    K2.3: NEO-6M GPS UART check on Jetson Nano 40-pin header UART (/dev/ttyTHS1).
    Opens the port, reads for up to timeout_s (10s), reports NMEA line count and fix status.
    """
    if not os.path.exists(port):
        return print_check("NEO-6M GPS UART ({})".format(port), False, "Device node absent (Simulation Mode / Hardware not connected)")

    try:
        import serial
    except ImportError:
        return print_check("NEO-6M GPS UART ({})".format(port), False, "pyserial not installed (pip install pyserial==3.5)")

    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from edge.sensors import parse_nmea_sentence

        line_count = 0
        valid_fix = False
        satellites = 0
        t0 = time.time()

        with serial.Serial(port, 9600, timeout=1.0) as ser:
            while time.time() - t0 < timeout_s:
                raw = ser.readline()
                if raw:
                    line_count += 1
                    try:
                        line = raw.decode("ascii", errors="replace")
                        parsed = parse_nmea_sentence(line)
                        if parsed and parsed.get("valid"):
                            valid_fix = True
                            if "satellites" in parsed:
                                satellites = parsed["satellites"]
                    except Exception:
                        pass
                if valid_fix and line_count >= 10:
                    break

        elapsed = time.time() - t0
        detail = "Read {} lines in {:.1f}s, fix={}, sats={}".format(line_count, elapsed, "VALID" if valid_fix else "NO_FIX", satellites)
        return print_check("NEO-6M GPS UART ({})".format(port), line_count > 0, detail)
    except Exception as e:
        return print_check("NEO-6M GPS UART ({})".format(port), False, "Error opening {}: {}".format(port, e))


def check_model_a_engine():
    engine_candidates = [
        os.path.join(REPO_ROOT, "artifacts", "trt", "model_a.engine"),
        os.path.join(REPO_ROOT, "artifacts", "engine", "model_a_fused.engine"),
        os.path.join(REPO_ROOT, "artifacts", "engine", "model_a.engine")
    ]
    engine_path = None
    for p in engine_candidates:
        if os.path.exists(p):
            engine_path = p
            break

    if engine_path is None:
        return print_check("Model A TensorRT Engine Exists & Runs", False, "No engine found in artifacts/trt/ or artifacts/engine/")

    try:
        import numpy as np
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return print_check("Model A TensorRT Engine Deserialization", False, "Failed to deserialize")

        context = engine.create_execution_context()
        
        # Allocate buffers for batch=9, 224x224x3 BGR float32 -> (9, 29)
        h_in = cuda.pagelocked_empty((9, 224, 224, 3), np.float32)
        h_out = cuda.pagelocked_empty((9, 29), np.float32)
        d_in = cuda.mem_alloc(h_in.nbytes)
        d_out = cuda.mem_alloc(h_out.nbytes)
        stream = cuda.Stream()
        bindings = [int(d_in), int(d_out)]

        t0 = time.time()
        cuda.memcpy_htod_async(d_in, h_in, stream)
        if hasattr(context, "execute_async_v2"):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        else:
            context.execute_async(batch_size=9, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        latency_ms = (time.time() - t0) * 1000.0

        passed = (h_out.shape == (9, 29) and not np.isnan(h_out).any())
        detail = "Output shape: {}, Latency: {:.2f}ms".format(h_out.shape, latency_ms)
        return print_check("Model A TensorRT Engine Dummy Inference", passed, detail)
    except Exception as e:
        return print_check("Model A TensorRT Engine Dummy Inference", False, str(e))


def check_model_b_onnx():
    onnx_path = os.path.join(REPO_ROOT, "artifacts", "onnx", "model_b.onnx")
    if not os.path.exists(onnx_path):
        return print_check("Model B ONNX Model Exists", False, "Missing at {}".format(onnx_path))

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        # Dummy batch of 4 crops, shape (4, 3, 64, 64)
        dummy_in = np.zeros((4, 3, 64, 64), dtype=np.float32)
        t0 = time.time()
        out = sess.run([out_name], {in_name: dummy_in})[0]
        latency_ms = (time.time() - t0) * 1000.0

        passed = (out.shape == (4, 3) and not np.isnan(out).any())
        detail = "Output shape: {}, No NaN: True, Latency: {:.2f}ms".format(out.shape, latency_ms)
        return print_check("Model B ONNX Runtime Dummy Inference", passed, detail)
    except Exception as e:
        return print_check("Model B ONNX Runtime Dummy Inference", False, str(e))


def check_class_dimension():
    classes_path = os.path.join(REPO_ROOT, "configs", "classes.py")
    if not os.path.exists(classes_path):
        return print_check("Taxonomy Output Dimension Match", False, "configs/classes.py missing")
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from configs.classes import CLASS_NAMES
        count = len(CLASS_NAMES)
        expected = 29
        passed = (count == expected)
        detail = "Found {} classes, expected {}".format(count, expected)
        return print_check("Taxonomy Output Dimension Match", passed, detail)
    except Exception as e:
        return print_check("Taxonomy Output Dimension Match", False, str(e))

def check_sqlite():
    db_path = os.path.join(REPO_ROOT, "artifacts", "storage", "smoke_test.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("CREATE TABLE IF NOT EXISTS smoke_test (id INTEGER PRIMARY KEY, msg TEXT, ts REAL);")
        t_now = time.time()
        cur.execute("INSERT INTO smoke_test (msg, ts) VALUES (?, ?);", ("nano_smoke_ok", t_now))
        conn.commit()
        cur.execute("SELECT msg, ts FROM smoke_test WHERE ts = ?;", (t_now,))
        row = cur.fetchone()
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        passed = (row is not None and row[0] == "nano_smoke_ok")
        detail = "WAL mode write and read verified"
        return print_check("SQLite Storage Read/Write (WAL)", passed, detail)
    except Exception as e:
        return print_check("SQLite Storage Read/Write (WAL)", False, str(e))


def check_system_resources():
    # Disk space
    try:
        total, used, free = shutil.disk_usage(REPO_ROOT)
        free_gb = free / (1024 ** 3)
        disk_ok = (free_gb >= 1.0)
        disk_detail = "{:.2f} GB free".format(free_gb)
        print_check("Disk Space Availability (>=1GB)", disk_ok, disk_detail)
    except Exception as e:
        disk_ok = False
        print_check("Disk Space Availability (>=1GB)", False, str(e))

    # RAM (from /proc/meminfo if on Linux)
    ram_ok = True
    ram_detail = "N/A"
    if os.path.exists("/proc/meminfo"):
        try:
            mem_free_kb = 0
            mem_avail_kb = 0
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail_kb = int(line.split()[1])
                        break
                    elif line.startswith("MemFree:"):
                        mem_free_kb = int(line.split()[1])
            avail_mb = (mem_avail_kb if mem_avail_kb > 0 else mem_free_kb) / 1024.0
            ram_ok = (avail_mb >= 300.0)
            ram_detail = "{:.1f} MB available".format(avail_mb)
        except Exception as e:
            ram_detail = "Error reading /proc/meminfo: {}".format(e)
            ram_ok = False
    else:
        ram_detail = "Non-Linux host; /proc/meminfo absent"

    print_check("RAM Availability (>=300MB)", ram_ok, ram_detail)
    return disk_ok and ram_ok


def check_mlx90640_hardware():
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from edge.hardware_detect import detect_mlx90640
        ok, reason = detect_mlx90640(bus_num=1, address=0x33)
        return print_check("MLX90640 Thermal Array I2C (/dev/i2c-1:0x33)", ok, reason)
    except Exception as e:
        return print_check("MLX90640 Thermal Array I2C (/dev/i2c-1:0x33)", False, str(e))


def main():
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    print("=" * 80)
    print("NVIDIA JETSON NANO DEPLOYMENT SMOKE TEST")
    print("Timestamp : {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")))
    print("Log Path  : {}".format(LOG_FILE))
    print("Repo Root : {}".format(REPO_ROOT))
    print("=" * 80)

    checks = []
    print("\n[1] Runtime & Python Environment:")
    checks.append(check_python_version())

    print("\n[2] Third-Party Library Imports:")
    checks.append(check_imports())

    print("\n[3] Hardware Firmware & Peripheral Drivers:")
    checks.append(check_firmware())
    checks.append(check_gps_uart())
    checks.append(check_mlx90640_hardware())

    print("\n[4] AI Inference Engines:")
    checks.append(check_model_a_engine())
    checks.append(check_model_b_onnx())

    print("\n[5] Classification Taxonomy & Config Integrity:")
    checks.append(check_class_dimension())

    print("\n[6] Edge Database Storage Subsystem:")
    checks.append(check_sqlite())

    print("\n[7] System Resource Margins:")
    checks.append(check_system_resources())

    print("\n" + "=" * 80)
    total_checks = len(checks)
    passed_checks = sum(1 for c in checks if c)
    print("SUMMARY: {}/{} CHECKS PASSED".format(passed_checks, total_checks))
    if passed_checks == total_checks:
        print("OVERALL VERDICT: PASS (Deployment ready)")
    else:
        print("OVERALL VERDICT: FAIL/WARN (Check individual item diagnostics above)")
    print("Smoke test log saved to: {}".format(LOG_FILE))
    print("=" * 80)

    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
