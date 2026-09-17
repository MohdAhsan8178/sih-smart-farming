#!/usr/bin/env python3
"""
edge/wifi_switch.py -- WiFi AP/Client Mode Switching for Handheld Nano Pod (K1.3).

Controls network interfaces to switch from local mobile AP (SIH-FIELD)
to the ESP32 Mast AP (SIH-NODE-01) for pulling telemetry readings and trap photos,
and guarantees restoration of SIH-FIELD AP on completion or exception.

STATUS: UNVERIFIED ON HARDWARE (Tested in simulation and mock; bench validation pending AR9271 / WiFi dongle arrival).
Python 3.6 compatible.
"""

import argparse
import logging
import subprocess
import sys
import time
from typing import Callable, Optional

logger = logging.getLogger("edge.wifi_switch")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WiFiSwitch(object):
    """
    Manages switching NetworkManager WiFi connections via nmcli.
    """

    def __init__(
        self,
        field_ap_name="SIH-FIELD",
        mast_ssid="SIH-NODE-01",
        mast_password="sih12345",
        dry_run=False,
    ):
        self.field_ap_name = field_ap_name
        self.mast_ssid = mast_ssid
        self.mast_password = mast_password
        self.dry_run = bool(dry_run)

    def _run_cmd(self, cmd_args):
        cmd_str = " ".join(cmd_args)
        t0 = time.time()
        if self.dry_run:
            logger.info("[DRY-RUN] Would execute: %s", cmd_str)
            return True, 0, "[DRY-RUN] Success", 0.0

        logger.info("Executing: %s", cmd_str)
        try:
            res = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=20,
            )
            elapsed = time.time() - t0
            success = (res.returncode == 0)
            if success:
                logger.info("Command succeeded in %.2fs: %s", elapsed, cmd_str)
            else:
                logger.warning("Command returned %d in %.2fs: %s (stderr: %s)", res.returncode, elapsed, cmd_str, res.stderr.strip())
            return success, res.returncode, res.stdout + res.stderr, elapsed
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("Command failed with exception in %.2fs: %s (%s)", elapsed, cmd_str, e)
            return False, -1, str(e), elapsed

    def stop_field_ap(self):
        """Stops the farmer-facing SIH-FIELD AP connection."""
        logger.info("Step 1: Disconnecting AP '%s'...", self.field_ap_name)
        cmd = ["nmcli", "connection", "down", self.field_ap_name]
        return self._run_cmd(cmd)

    def join_mast_ap(self):
        """Connects to the ground mast AP (SIH-NODE-01)."""
        logger.info("Step 2: Connecting to Ground Mast AP '%s'...", self.mast_ssid)
        cmd = ["nmcli", "device", "wifi", "connect", self.mast_ssid, "password", self.mast_password]
        return self._run_cmd(cmd)

    def restore_field_ap(self):
        """Restores the farmer-facing SIH-FIELD AP connection."""
        logger.info("Step 4: Restoring farmer AP '%s'...", self.field_ap_name)
        cmd = ["nmcli", "connection", "up", self.field_ap_name]
        return self._run_cmd(cmd)

    def run_with_mast_connection(self, task_fn):
        """
        Context wrapper that stops AP, connects to mast, runs task_fn,
        and ALWAYS restores the field AP in a finally block.
        """
        t_total_0 = time.time()
        logger.info("Beginning WiFi mode switch to Mast AP '%s'...", self.mast_ssid)
        try:
            self.stop_field_ap()
            self.join_mast_ap()
            return task_fn()
        finally:
            logger.info("Ensuring field AP restoration...")
            self.restore_field_ap()
            logger.info("WiFi mode switch cycle complete in %.2fs", time.time() - t_total_0)


def main():
    parser = argparse.ArgumentParser(description="WiFi Switch CLI for SIH Smart Farming Nano Pod (UNVERIFIED ON HARDWARE)")
    parser.add_argument("--dry-run", action="store_true", help="Print nmcli commands without executing them")
    parser.add_argument("--field-ap", default="SIH-FIELD", help="Field AP Connection Name")
    parser.add_argument("--mast-ssid", default="SIH-NODE-01", help="Mast AP SSID")
    parser.add_argument("--mast-password", default="sih12345", help="Mast AP Password")
    args = parser.parse_args()

    switcher = WiFiSwitch(
        field_ap_name=args.field_ap,
        mast_ssid=args.mast_ssid,
        mast_password=args.mast_password,
        dry_run=args.dry_run,
    )

    def _sample_task():
        print("Task executed while connected to Mast AP!")
        return True

    switcher.run_with_mast_connection(_sample_task)


if __name__ == "__main__":
    main()
