#!/usr/bin/env python3
"""
edge/thermal_capture.py -- Melexis MLX90640 32x24 Thermal Far-Infrared Array Driver (L6.1, L6.2).

Features:
  1. Pure Python smbus2 implementation of MLX90640 EEPROM calibration parameter decoding.
  2. Full RAM subpage conversion (subpages 0 & 1) to a 24x32 float array in degrees Celsius.
  3. Ambient temperature (Ta) and object temperature (To) calculation per Melexis application notes.
  4. Configurable refresh rates (0.5 Hz to 64 Hz, default 2 Hz).
  5. Mock / simulation mode for continuous integration and unit testing without physical I2C bus.
  6. Reference-based CWSI computation requiring explicit wet/dry reference surfaces (L6.3).

Python 3.6 compatible.
"""

import datetime
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class MLX90640(object):
    """
    Driver for Melexis MLX90640 32x24 FIR thermal array sensor over I2C.
    Compliant with official Melexis MLX90640 datasheet and calibration algorithm.
    """

    I2C_ADDR = 0x33
    EEPROM_START = 0x2400
    EEPROM_WORDS = 832
    RAM_START = 0x0400
    RAM_WORDS = 832
    STATUS_REG = 0x8000
    CTRL_REG = 0x800D

    def __init__(self, bus_num=1, address=0x33, fps=2.0, mock=False):
        # type: (int, int, float, bool) -> None
        self.bus_num = int(bus_num)
        self.address = int(address)
        self.fps = float(fps)
        self.mock = bool(mock)
        self.emissivity = 0.95  # Standard agricultural crop canopy emissivity
        self.params = {}        # type: Dict[str, Any]
        self._init_error = None  # type: Optional[str]
        self._frame_buffer = np.zeros((24, 32), dtype=np.float32)

        if self.mock:
            self._bus = None
            print("[THERMAL] Explicit MOCK thermal source requested (--thermal-source mock).")
            self._init_mock_params()
        else:
            try:
                import smbus2
                self.smbus2 = smbus2
                self._bus = smbus2.SMBus(self.bus_num)
                self._load_eeprom()
            except Exception as e:
                # No silent fallback! Record error; hardware requested must fail clearly if absent.
                self._bus = None
                self._init_error = str(e)

    def _read_words(self, start_addr, num_words):
        # type: (int, int) -> List[int]
        """Reads 16-bit words from MLX90640 memory space using 16-bit address."""
        if self._bus is None:
            return [0] * num_words

        words = []
        # SMBus block size limit is 32 bytes (16 words)
        chunk_size = 16
        for offset in range(0, num_words, chunk_size):
            n = min(chunk_size, num_words - offset)
            curr_addr = start_addr + offset
            # MLX90640 expects big-endian address: [addr_msb, addr_lsb]
            addr_bytes = [(curr_addr >> 8) & 0xFF, curr_addr & 0xFF]
            try:
                # Write address then read raw bytes
                msg_w = self.smbus2.i2c_msg.write(self.address, addr_bytes)
                msg_r = self.smbus2.i2c_msg.read(self.address, n * 2)
                self._bus.i2c_rdwr(msg_w, msg_r)
                raw_bytes = list(msg_r)
                for i in range(0, len(raw_bytes), 2):
                    msb = raw_bytes[i]
                    lsb = raw_bytes[i + 1]
                    word = (msb << 8) | lsb
                    words.append(word)
            except Exception as ex:
                raise IOError("Failed reading MLX90640 at 0x%04x: %s" % (curr_addr, ex))

        return words

    def _write_word(self, reg_addr, value):
        # type: (int, int) -> None
        """Writes a 16-bit word to MLX90640 register space."""
        if self._bus is None:
            return
        addr_bytes = [(reg_addr >> 8) & 0xFF, reg_addr & 0xFF]
        data_bytes = [(value >> 8) & 0xFF, value & 0xFF]
        try:
            msg = self.smbus2.i2c_msg.write(self.address, addr_bytes + data_bytes)
            self._bus.i2c_rdwr(msg)
        except Exception as ex:
            raise IOError("Failed writing MLX90640 at 0x%04x: %s" % (reg_addr, ex))

    def _load_eeprom(self):
        """Decodes calibration constants from the 832-word EEPROM block."""
        eeprom = self._read_words(self.EEPROM_START, self.EEPROM_WORDS)
        if len(eeprom) < self.EEPROM_WORDS:
            raise ValueError("EEPROM read returned %d words (expected %d)" % (len(eeprom), self.EEPROM_WORDS))

        self.params = self._decode_eeprom(eeprom)

    def _decode_eeprom(self, eeprom):
        # type: (List[int]) -> Dict[str, Any]
        """
        Parses EEPROM registers per official Melexis MLX90640 datasheet.
        Extracts Vdd, Ta, Gain, TGC, KsTa, KsTo, CP, and per-pixel offset/alpha/kta/kv.
        """
        p = {}

        # --- 1. VDD Parameters (EEPROM[51] = 0x2433) ---
        kVdd = (eeprom[51] >> 8) & 0xFF
        if kVdd > 127:
            kVdd -= 256
        kVdd = kVdd * 32.0

        vdd25 = eeprom[51] & 0xFF
        if vdd25 > 127:
            vdd25 -= 256
        vdd25 = vdd25 * 32.0 - 8192.0

        p["kVdd"] = kVdd
        p["vdd25"] = vdd25

        # --- 2. Ta / PTAT Parameters ---
        # 0x2432 = index 50
        kvPTAT = (eeprom[50] >> 10) & 0x3F
        if kvPTAT > 31:
            kvPTAT -= 64
        kvPTAT = kvPTAT / 4096.0

        ktPTAT = eeprom[50] & 0x3FF
        if ktPTAT > 511:
            ktPTAT -= 1024
        ktPTAT = ktPTAT / 8.0

        # 0x2431 = index 49
        vPTAT25 = eeprom[49]
        if vPTAT25 > 32767:
            vPTAT25 -= 65536

        # 0x2430 = index 48
        alphaPTAT = ((eeprom[48] >> 12) & 0x0F) / 4.0 + 8.0

        p["kvPTAT"] = kvPTAT
        p["ktPTAT"] = ktPTAT
        p["vPTAT25"] = float(vPTAT25)
        p["alphaPTAT"] = alphaPTAT

        # --- 3. Gain (0x2430 = index 48) ---
        gain = eeprom[48]
        if gain > 32767:
            gain -= 65536
        p["gain"] = float(gain)

        # --- 4. Resolution & Scales (0x2438 = index 56) ---
        resEE = (eeprom[56] >> 12) & 0x03
        kvScale = (eeprom[56] >> 8) & 0x0F
        ktaScale1 = (eeprom[56] >> 4) & 0x0F
        ktaScale2 = eeprom[56] & 0x0F
        p["resEE"] = resEE
        p["kvScale"] = kvScale
        p["ktaScale1"] = ktaScale1
        p["ktaScale2"] = ktaScale2

        # --- 5. TGC and KsTa (0x243C = index 60) ---
        tgc = eeprom[60] & 0xFF
        if tgc > 127:
            tgc -= 256
        tgc = tgc / 32.0

        ksTa = (eeprom[60] >> 8) & 0xFF
        if ksTa > 127:
            ksTa -= 256
        ksTa = ksTa / 8192.0

        p["tgc"] = tgc
        p["ksTa"] = ksTa

        # --- 6. KsTo and CT ranges (0x243D = index 61, 0x243E = 62, 0x243F = 63) ---
        ksToScale = (eeprom[61] & 0x0F) + 8
        step = ((eeprom[61] >> 12) & 0x03) * 10
        ct2 = ((eeprom[61] >> 4) & 0x0F) * 10
        ct3 = ct2 + step
        ct4 = ((eeprom[61] >> 8) & 0x0F) * 10 + ct3
        ct1 = -40.0

        p["CT"] = [ct1, float(ct2), float(ct3), float(ct4)]

        ksTo1 = eeprom[62] & 0xFF
        if ksTo1 > 127:
            ksTo1 -= 256
        ksTo2 = (eeprom[62] >> 8) & 0xFF
        if ksTo2 > 127:
            ksTo2 -= 256
        ksTo3 = eeprom[63] & 0xFF
        if ksTo3 > 127:
            ksTo3 -= 256
        ksTo4 = (eeprom[63] >> 8) & 0xFF
        if ksTo4 > 127:
            ksTo4 -= 256

        scale_ksto = 2.0 ** ksToScale
        p["ksTo"] = [
            ksTo1 / scale_ksto,
            ksTo2 / scale_ksto,
            ksTo3 / scale_ksto,
            ksTo4 / scale_ksto,
        ]

        # --- 7. Compensation Pixel (CP) Parameters ---
        # 0x243A = index 58
        offCP_SP0 = eeprom[58] & 0x03FF
        if offCP_SP0 > 511:
            offCP_SP0 -= 1024
        offCP_diff = (eeprom[58] >> 10) & 0x3F
        if offCP_diff > 31:
            offCP_diff -= 64
        offCP_SP1 = offCP_SP0 + offCP_diff

        p["offCP"] = [float(offCP_SP0), float(offCP_SP1)]

        # 0x2439 = index 57: ktaCP is low byte, kvCP is high byte
        ktaCP = eeprom[57] & 0xFF
        if ktaCP > 127:
            ktaCP -= 256
        ktaCP = ktaCP / (2.0 ** (ktaScale1 + 8))

        kvCP = (eeprom[57] >> 8) & 0xFF
        if kvCP > 127:
            kvCP -= 256
        kvCP = kvCP / (2.0 ** kvScale)

        p["ktaCP"] = ktaCP
        p["kvCP"] = kvCP

        # 0x2420 = index 32: alpha scale
        acc_rem_scale = (eeprom[32] >> 4) & 0x0F
        acc_col_scale = (eeprom[32] >> 8) & 0x0F
        acc_row_scale = (eeprom[32] >> 12) & 0x0F
        alpha_scale = (eeprom[32] & 0x0F) + 30
        p["alpha_scale"] = alpha_scale

        # 0x243B = index 59
        alpha_scale_cp = (eeprom[32] & 0x0F) + 27
        alphaCP_SP0 = (eeprom[59] & 0x03FF)
        if alphaCP_SP0 > 511:
            alphaCP_SP0 -= 1024
        alphaCP_SP0 = alphaCP_SP0 / (2.0 ** alpha_scale_cp)

        alphaCP_diff = (eeprom[59] >> 10) & 0x3F
        if alphaCP_diff > 31:
            alphaCP_diff -= 64
        alphaCP_SP1 = (1.0 + alphaCP_diff / 64.0) * alphaCP_SP0

        p["alphaCP"] = [alphaCP_SP0, alphaCP_SP1]

        # --- 8. Offsets: Reference, Rows, Cols ---
        # 0x2410 = index 16
        scale_occ_row = (eeprom[16] >> 8) & 0x0F
        scale_occ_col = (eeprom[16] >> 4) & 0x0F
        scale_occ_rem = eeprom[16] & 0x0F

        # 0x2421 = index 33
        offset_ref = eeprom[33]
        if offset_ref > 32767:
            offset_ref -= 65536

        row_occ = []
        for r in range(24):
            nibble = (eeprom[18 + r // 4] >> ((r % 4) * 4)) & 0x0F
            if nibble > 7:
                nibble -= 16
            row_occ.append(nibble * (1 << scale_occ_row))

        col_occ = []
        for c in range(32):
            nibble = (eeprom[24 + c // 4] >> ((c % 4) * 4)) & 0x0F
            if nibble > 7:
                nibble -= 16
            col_occ.append(nibble * (1 << scale_occ_col))

        row_acc = []
        for r in range(24):
            nibble = (eeprom[34 + r // 4] >> ((r % 4) * 4)) & 0x0F
            if nibble > 7:
                nibble -= 16
            row_acc.append(nibble * (1 << acc_row_scale))

        col_acc = []
        for c in range(32):
            nibble = (eeprom[40 + c // 4] >> ((c % 4) * 4)) & 0x0F
            if nibble > 7:
                nibble -= 16
            col_acc.append(nibble * (1 << acc_col_scale))

        # --- 9. Kv Corners Table ---
        # 0x2434 = index 52
        kv_ro_co = (eeprom[52] >> 12) & 0x0F
        if kv_ro_co > 7:
            kv_ro_co -= 16
        kv_re_co = (eeprom[52] >> 8) & 0x0F
        if kv_re_co > 7:
            kv_re_co -= 16
        kv_ro_ce = (eeprom[52] >> 4) & 0x0F
        if kv_ro_ce > 7:
            kv_ro_ce -= 16
        kv_re_ce = eeprom[52] & 0x0F
        if kv_re_ce > 7:
            kv_re_ce -= 16

        scale_kv = 2.0 ** kvScale
        kv_table = [
            kv_re_ce / scale_kv,  # row even, col even
            kv_re_co / scale_kv,  # row even, col odd
            kv_ro_ce / scale_kv,  # row odd, col even
            kv_ro_co / scale_kv,  # row odd, col odd
        ]

        # --- 10. Kta RC Table ---
        # 0x2436 = index 54, 0x2437 = index 55
        kta_re_ce = (eeprom[54] >> 8) & 0xFF
        if kta_re_ce > 127:
            kta_re_ce -= 256
        kta_re_co = eeprom[54] & 0xFF
        if kta_re_co > 127:
            kta_re_co -= 256
        kta_ro_ce = (eeprom[55] >> 8) & 0xFF
        if kta_ro_ce > 127:
            kta_ro_ce -= 256
        kta_ro_co = eeprom[55] & 0xFF
        if kta_ro_co > 127:
            kta_ro_co -= 256

        kta_rc = [kta_re_ce, kta_re_co, kta_ro_ce, kta_ro_co]

        # --- 11. Build 24x32 Per-Pixel Arrays (offset, alpha, kta, kv) ---
        pixels_offset = np.zeros((24, 32), dtype=np.float32)
        pixels_alpha = np.zeros((24, 32), dtype=np.float32)
        pixels_kta = np.zeros((24, 32), dtype=np.float32)
        pixels_kv = np.zeros((24, 32), dtype=np.float32)

        scale_alpha = 2.0 ** alpha_scale
        scale_kta_denom = 2.0 ** (ktaScale1 + 8)
        scale_kta_num = 2.0 ** ktaScale2
        scale_occ_rem_val = 1 << scale_occ_rem
        scale_acc_rem_val = 1 << acc_rem_scale

        for i in range(768):
            r = i // 32
            c = i % 32
            rc_idx = (r % 2) * 2 + (c % 2)
            p_word = eeprom[64 + i]

            # Offset
            off_rem = (p_word >> 10) & 0x3F
            if off_rem > 31:
                off_rem -= 64
            pixels_offset[r, c] = offset_ref + row_occ[r] + col_occ[c] + off_rem * scale_occ_rem_val

            # Alpha
            alpha_rem = (p_word >> 4) & 0x3F
            if alpha_rem > 31:
                alpha_rem -= 64
            pixels_alpha[r, c] = (row_acc[r] + col_acc[c] + alpha_rem * scale_acc_rem_val) / scale_alpha

            # Kta
            kta_rem = (p_word >> 1) & 0x07
            if kta_rem > 3:
                kta_rem -= 8
            pixels_kta[r, c] = (kta_rc[rc_idx] + kta_rem * scale_kta_num) / scale_kta_denom

            # Kv
            pixels_kv[r, c] = kv_table[rc_idx]

        p["pixels_offset"] = pixels_offset
        p["pixels_alpha"] = pixels_alpha
        p["pixels_kta"] = pixels_kta
        p["pixels_kv"] = pixels_kv

        return p

    def _calculate_temperatures(self, ram_words, subpage=0, return_intermediates=False):
        # type: (List[int], int, bool) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict[str, Any]]]
        """
        Converts 832 RAM words to ambient temperature (Ta) and 24x32 object temperatures (To)
        according to Melexis MLX90640 datasheet formulas.
        """
        p = self.params
        res_ee = p.get("resEE", 2)
        res_ram = 2
        if len(ram_words) > 832 and ram_words[832] != 0:
            res_ram = (ram_words[832] & 0x0C00) >> 10

        res_corr = 2.0 ** (res_ee - res_ram)

        # 1. Vdd Calculation (RAM[810] = 0x072A)
        vdd_ram = ram_words[810]
        if vdd_ram > 32767:
            vdd_ram -= 65536
        kvdd = p["kVdd"] if p["kVdd"] != 0 else 1.0
        delta_vdd = (res_corr * vdd_ram - p["vdd25"]) / kvdd
        vdd = delta_vdd + 3.3

        # 2. Ta (Ambient Temperature) Calculation
        # RAM[800] = 0x0720 (Ta_PTAT)
        vptat_ram = ram_words[800]
        if vptat_ram > 32767:
            vptat_ram -= 65536

        # RAM[768] = 0x0700 (Ta_Vbe)
        vbe_ram = ram_words[768]
        if vbe_ram > 32767:
            vbe_ram -= 65536

        alpha_ptat = p.get("alphaPTAT", 9.0)
        vptat_denom = vptat_ram * alpha_ptat + vbe_ram
        if abs(vptat_denom) < 1e-9:
            vptat_denom = 1.0
        vptat_art = (vptat_ram / vptat_denom) * (2.0 ** 18)

        vptat_comp = vptat_art / (1.0 + p["kvPTAT"] * delta_vdd)
        ktptat = p["ktPTAT"] if abs(p["ktPTAT"]) > 1e-9 else 1.0
        delta_ta = (vptat_comp - p["vPTAT25"]) / ktptat
        ta = delta_ta + 25.0

        # 3. Gain Calculation (RAM[778] = 0x070A)
        gain_ram = ram_words[778]
        if gain_ram > 32767:
            gain_ram -= 65536
        k_gain = p["gain"] / (gain_ram if gain_ram != 0 else 1.0)

        # 4. Compensation Pixel (CP) Calculation (CP0=RAM[776] / 0x0708, CP1=RAM[808] / 0x0728)
        cp0_ram = ram_words[776]
        if cp0_ram > 32767:
            cp0_ram -= 65536
        cp1_ram = ram_words[808]
        if cp1_ram > 32767:
            cp1_ram -= 65536

        cp_raw = cp0_ram if subpage == 0 else cp1_ram
        cp_gain = cp_raw * k_gain

        off_cp = p["offCP"][subpage]
        alpha_cp = p["alphaCP"][subpage]
        kta_cp = p["ktaCP"]
        kv_cp = p["kvCP"]

        v_cp_comp = cp_gain - (off_cp * (1.0 + kta_cp * (ta - 25.0)) * (1.0 + kv_cp * delta_vdd))

        # 5. Per-Pixel IR and To Calculation (24 x 32)
        tgc = p["tgc"]
        ks_ta = p["ksTa"]
        emiss = self.emissivity
        ks_to2 = p["ksTo"][1] if len(p.get("ksTo", [])) > 1 else 0.0

        tr = ta - 8.0  # Open-air standard reflected shift (-8 degC)
        ta_k = ta + 273.15
        tr_k = tr + 273.15
        ta_k4 = ta_k ** 4
        tr_k4 = tr_k ** 4
        ta_tr = tr_k4 - (tr_k4 - ta_k4) / (emiss if emiss > 0 else 0.95)

        to_array = np.zeros((24, 32), dtype=np.float32)

        pixels_offset = p["pixels_offset"]
        pixels_kta = p["pixels_kta"]
        pixels_kv = p["pixels_kv"]
        pixels_alpha = p["pixels_alpha"]

        for i in range(768):
            r = i // 32
            c = i % 32
            pix_raw = ram_words[i]
            if pix_raw > 32767:
                pix_raw -= 65536
            pix_gain = pix_raw * k_gain

            # Offset compensation
            v_pix_offset_comp = pix_gain - (pixels_offset[r, c] * (1.0 + pixels_kta[r, c] * (ta - 25.0)) * (1.0 + pixels_kv[r, c] * delta_vdd))
            # TGC compensation
            v_ir_comp = v_pix_offset_comp - tgc * v_cp_comp

            # Sensitivity compensation
            alpha_comp = (pixels_alpha[r, c] - tgc * alpha_cp) * (1.0 + ks_ta * (ta - 25.0))
            alpha_comp_emiss = alpha_comp * emiss
            if alpha_comp_emiss <= 0:
                alpha_comp_emiss = 1e-12

            # Basic To calculation with standard 4th root
            sx_inner = (alpha_comp_emiss ** 3) * (v_ir_comp + alpha_comp_emiss * ta_tr)
            if sx_inner > 0:
                sx = ks_to2 * math.pow(sx_inner, 0.25)
            else:
                sx = 0.0

            denom = alpha_comp_emiss * (1.0 - ks_to2 * 273.15) + sx
            if abs(denom) < 1e-15:
                denom = 1e-15

            to_k4 = (v_ir_comp / denom) + ta_tr
            if to_k4 > 0:
                to_val = math.pow(to_k4, 0.25) - 273.15
            else:
                to_val = ta

            to_array[r, c] = float(to_val)

        if return_intermediates:
            intermediates = {
                "gain_ram": int(gain_ram),
                "vdd_pix": int(vdd_ram),
                "vptat": int(vptat_ram),
                "vbe": int(vbe_ram),
                "cp0": int(cp0_ram),
                "cp1": int(cp1_ram),
                "k_gain": float(k_gain),
                "delta_vdd": float(delta_vdd),
                "vdd": float(vdd),
                "vptat_art": float(vptat_art),
                "vptat_comp": float(vptat_comp),
                "delta_ta": float(delta_ta),
                "ta": float(ta),
                "to_min": float(np.min(to_array)),
                "to_max": float(np.max(to_array)),
                "to_mean": float(np.mean(to_array)),
                "to_median": float(np.median(to_array)),
            }
            return to_array, float(ta), intermediates

        return to_array, float(ta)

    def _init_mock_params(self):
        """Initializes calibration parameters for mock mode."""
        self.params = {
            "kVdd": -3200.0,
            "vdd25": -13000.0,
            "kvPTAT": 0.005,
            "ktPTAT": 25.0,
            "vPTAT25": 12000.0,
            "alphaPTAT": 9.0,
            "gain": 6000.0,
            "tgc": 0.0,
            "ksTa": 0.0,
            "ksTo": [0.0, 0.0, 0.0, 0.0],
            "CT": [-40.0, 0.0, 40.0, 80.0],
            "offCP": [0.0, 0.0],
            "ktaCP": 0.0,
            "kvCP": 0.0,
            "alphaCP": [1e-7, 1e-7],
            "pixels_alpha": np.ones((24, 32), dtype=np.float32) * 1e-7,
            "pixels_offset": np.zeros((24, 32), dtype=np.float32),
            "pixels_kta": np.zeros((24, 32), dtype=np.float32),
            "pixels_kv": np.zeros((24, 32), dtype=np.float32),
        }

    def capture_frame(self, target_ambient_c=28.5, target_canopy_c=26.8):
        # type: (float, float) -> Dict[str, Any]
        """
        Captures one complete thermal frame (24 rows x 32 columns).
        Returns dictionary with temperature_array (°C), ambient_temp_c (°C), and metadata.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if self.mock:
            # Generate synthetic 24x32 temperature array with realistic canopy + soil hot spots
            np.random.seed(int(time.time() * 100) % 10000)
            # Baseline canopy temperature: ~26.8°C with subtle leaf-angle variance
            thermal = np.random.normal(loc=target_canopy_c, scale=0.6, size=(24, 32)).astype(np.float32)
            # Add a warm sunlit soil patch in upper-right corner (~36°C)
            thermal[0:8, 24:32] += np.random.normal(loc=9.5, scale=0.8, size=(8, 8))
            # Add shaded cool leaves (~25.2°C)
            thermal[14:20, 4:12] -= 1.4

            return {
                "available": True,
                "thermal_source": "mock",
                "temperature_array": thermal,
                "ambient_temp_c": round(float(target_ambient_c), 2),
                "mean_temp_c": round(float(np.mean(thermal)), 2),
                "min_temp_c": round(float(np.min(thermal)), 2),
                "max_temp_c": round(float(np.max(thermal)), 2),
                "timestamp_utc": now_utc,
                "rows": 24,
                "cols": 32,
                "emissivity": self.emissivity,
            }

        # Real hardware path check
        if self._init_error is not None:
            return {
                "available": False,
                "thermal_source": "hardware",
                "reason": "HARDWARE_NOT_CONNECTED: %s" % self._init_error,
                "timestamp_utc": now_utc,
                "temperature_array": None,
            }

        # Real hardware capture path
        try:
            # Capture two subpages to guarantee a fresh, full 24x32 frame
            subpages_captured = 0
            start_wait = time.time()
            last_subpage = -1
            to_array = None
            ta = 25.0

            # Read up to 2 subpages (timeout 1.0s)
            while subpages_captured < 2 and (time.time() - start_wait < 1.0):
                status_words = self._read_words(self.STATUS_REG, 1)
                status = status_words[0] if status_words else 0
                subpage = status & 0x01

                # Read RAM data words (0x0400 to 0x073F)
                ram_words = self._read_words(self.RAM_START, self.RAM_WORDS)
                # Clear data ready bit
                try:
                    self._write_word(self.STATUS_REG, 0x0000)
                except Exception:
                    pass

                to_array, ta = self._calculate_temperatures(ram_words, subpage=subpage)

                # Update frame buffer for pixels of this subpage (Chess Mode)
                for i in range(768):
                    r = i // 32
                    c = i % 32
                    if (r + c) % 2 == subpage:
                        self._frame_buffer[r, c] = to_array[r, c]

                if subpage != last_subpage:
                    subpages_captured += 1
                    last_subpage = subpage

                time.sleep(0.05)

            if to_array is None:
                raise IOError("No RAM data received from MLX90640")

            # If only 1 subpage could be read within timeout, fill uninitialized pixels from to_array
            if np.all(self._frame_buffer == 0.0):
                self._frame_buffer = to_array.copy()
            else:
                zero_mask = (self._frame_buffer == 0.0)
                if np.any(zero_mask):
                    self._frame_buffer[zero_mask] = to_array[zero_mask]

            thermal = self._frame_buffer.copy()

            return {
                "available": True,
                "thermal_source": "hardware",
                "temperature_array": thermal,
                "ambient_temp_c": round(float(ta), 2),
                "mean_temp_c": round(float(np.mean(thermal)), 2),
                "min_temp_c": round(float(np.min(thermal)), 2),
                "max_temp_c": round(float(np.max(thermal)), 2),
                "timestamp_utc": now_utc,
                "rows": 24,
                "cols": 32,
                "emissivity": self.emissivity,
            }
        except Exception as ex:
            return {
                "available": False,
                "thermal_source": "hardware",
                "reason": "HARDWARE_CAPTURE_FAILED: %s" % ex,
                "timestamp_utc": now_utc,
                "temperature_array": None,
            }

    def dump_intermediates(self, ram_words=None, subpage=0):
        # type: (Optional[List[int]], int) -> Dict[str, Any]
        """
        Dumps all raw 16-bit registers and computed intermediate variables
        from a live sensor read or provided RAM words.
        """
        if self.mock or (ram_words is None and self._bus is None):
            if ram_words is None:
                # Synthetic realistic RAM buffer
                ram_words = [0] * 834
                ram_words[768] = 16000  # Vbe
                ram_words[776] = 0      # CP0
                ram_words[778] = 6000   # Gain
                ram_words[800] = 1350   # PTAT
                ram_words[808] = 0      # CP1
                ram_words[810] = -13000 # Vdd
                ram_words[832] = 0x0800 # Res=2
            res = self._calculate_temperatures(ram_words, subpage=subpage, return_intermediates=True)
            return res[2]  # type: ignore

        if ram_words is None:
            ram_words = self._read_words(self.RAM_START, self.RAM_WORDS)
            status_words = self._read_words(self.STATUS_REG, 1)
            ctrl_words = self._read_words(self.CTRL_REG, 1)
            if ctrl_words:
                ram_words.append(ctrl_words[0])
            if status_words:
                subpage = status_words[0] & 0x01

        res = self._calculate_temperatures(ram_words, subpage=subpage, return_intermediates=True)
        return res[2]  # type: ignore


def load_thermal_refs(config_path="configs/thermal_refs.json"):
    # type: (str) -> Dict[str, Any]
    """Loads and returns the thermal reference regions configuration."""
    import json
    if not os.path.exists(config_path):
        return {"status": "NOT_CONFIGURED", "wet_ref": None, "dry_ref": None}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"status": "NOT_CONFIGURED", "wet_ref": None, "dry_ref": None}


def calculate_cwsi_reference_based(tc, t_wet, t_dry, allow_unclamped=False):
    # type: (float, Optional[float], Optional[float], bool) -> Any
    """Delegates to core/thermal.py reference implementation."""
    from core.thermal import calculate_cwsi_reference_based as core_calc
    return core_calc(tc, t_wet, t_dry, allow_unclamped=allow_unclamped)
