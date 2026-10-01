#!/usr/bin/env bash
set -euo pipefail

# scripts/install_pod_services.sh — Idempotent Systemd & Sudoers Installer for AEGIS Pod
# Target Hardware: NVIDIA Jetson Nano 4GB (JetPack 4.6.1 / Ubuntu 18.04 LTS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$EUID" -ne 0 ]; then
    echo "Error: This installer must be executed as root (e.g. sudo bash scripts/install_pod_services.sh)" >&2
    exit 1
fi

echo "=========================================================================="
echo "AEGIS HANDHELD NANO POD SERVICE & HELPER INSTALLER"
echo "=========================================================================="
echo "Repository Root: ${REPO_ROOT}"

# 1. Install root helper scripts to /usr/local/sbin/
echo "[1/4] Installing root helper scripts to /usr/local/sbin/..."
install -m 0755 -o root -g root "${REPO_ROOT}/scripts/helpers/aegis-set-time" /usr/local/sbin/aegis-set-time
install -m 0755 -o root -g root "${REPO_ROOT}/scripts/helpers/aegis-shutdown" /usr/local/sbin/aegis-shutdown
echo "  -> Installed /usr/local/sbin/aegis-set-time (mode 0755)"
echo "  -> Installed /usr/local/sbin/aegis-shutdown (mode 0755)"

# 2. Install and validate sudoers configuration
echo "[2/4] Installing restricted sudoers rule to /etc/sudoers.d/aegis-pod..."
SUDOERS_TMP="$(mktemp /tmp/aegis_sudoers_XXXXXX)"
cp "${REPO_ROOT}/scripts/helpers/aegis_sudoers" "${SUDOERS_TMP}"
chmod 0440 "${SUDOERS_TMP}"
chown root:root "${SUDOERS_TMP}"

if visudo -cf "${SUDOERS_TMP}"; then
    mv "${SUDOERS_TMP}" /etc/sudoers.d/aegis-pod
    chmod 0440 /etc/sudoers.d/aegis-pod
    chown root:root /etc/sudoers.d/aegis-pod
    echo "  -> /etc/sudoers.d/aegis-pod validated with visudo -cf and installed."
else
    rm -f "${SUDOERS_TMP}"
    echo "Error: visudo validation failed for aegis_sudoers. Aborting." >&2
    exit 2
fi

# 3. Install systemd service units
echo "[3/4] Installing systemd units to /etc/systemd/system/..."
cp "${REPO_ROOT}/services/aegis-maxn.service" /etc/systemd/system/aegis-maxn.service
cp "${REPO_ROOT}/services/aegis-gateway.service" /etc/systemd/system/aegis-gateway.service
chmod 0644 /etc/systemd/system/aegis-maxn.service /etc/systemd/system/aegis-gateway.service
chown root:root /etc/systemd/system/aegis-maxn.service /etc/systemd/system/aegis-gateway.service

# 4. Reload systemd and enable services
echo "[4/4] Reloading systemd daemon and enabling services..."
systemctl daemon-reload
systemctl enable aegis-maxn.service
systemctl enable aegis-gateway.service
echo "  -> Enabled aegis-maxn.service"
echo "  -> Enabled aegis-gateway.service"

echo "=========================================================================="
echo "INSTALLATION COMPLETE"
echo "=========================================================================="
echo "To test and inspect services on Jetson Nano, run the following commands:"
echo ""
echo "  # Start services manually:"
echo "  sudo systemctl start aegis-maxn.service"
echo "  sudo systemctl start aegis-gateway.service"
echo ""
echo "  # Check service statuses:"
echo "  systemctl status aegis-maxn.service"
echo "  systemctl status aegis-gateway.service"
echo ""
echo "  # Inspect live gateway logs:"
echo "  journalctl -u aegis-gateway.service -f"
echo ""
echo "  # Verify phone-time helper (dry-run/test):"
echo "  sudo /usr/local/sbin/aegis-set-time 2026-10-01T14:32:00Z"
echo ""
echo "  # Safe shutdown test:"
echo "  curl -X POST http://192.168.4.1:8080/api/v1/pod/shutdown \\"
echo "       -H 'Content-Type: application/json' -d '{\"confirm\": true}'"
echo ""
echo "  # Operational Runbook Note:"
echo "  # If CSI camera driver locks up between reboots (GStreamer capture failure):"
echo "  #   sudo systemctl restart nvargus-daemon"
echo "=========================================================================="
