#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_nano_services.sh
#
# Idempotent Jetson Nano Service & Network Configuration (Round L / L7.1).
#
# Configures:
#   1. AR9271 WiFi Hotspot "SIH-FIELD" (192.168.4.1/24, WPA2-PSK)
#   2. WiFi Client Profile for ESP32 Ground Mast "SIH-NODE-01" (sih12345)
#   3. Systemd Services & Timer:
#      - sih-gateway.service: Offline HTTP API Gateway (port 8080)
#      - sih-pipeline.service: Model A Edge Inference Daemon
#      - sih-collector.service: Ground Mast Telemetry Puller (mutex locked)
#      - sih-collector.timer: Periodic 30-minute collection trigger
#
# Usage:
#   sudo ./scripts/setup_nano_services.sh
# ==============================================================================

set -euo pipefail

# 1. Root check
if [[ $EUID -ne 0 ]]; then
   echo "[-] ERROR: This script must be run as root (sudo)." >&2
   exit 1
fi

echo "========================================================================"
echo " JETSON NANO SYSTEMD & NETWORK SERVICES SETUP (L7.1)"
echo "========================================================================"

INSTALL_DIR="/home/nvidia/sih-smart-farming"
SIH_CONF_DIR="/etc/sih"
RUN_LOCK_DIR="/run/lock"
mkdir -p "${SIH_CONF_DIR}" "${RUN_LOCK_DIR}"

# 2. Configure WiFi Credentials
AP_CONF="${SIH_CONF_DIR}/wifi_ap.conf"
if [[ ! -f "${AP_CONF}" ]]; then
    AP_PASS="${SIH_AP_PASSWORD:-sihfield12345}"
    echo "AP_SSID=SIH-FIELD" > "${AP_CONF}"
    echo "AP_PASSWORD=${AP_PASS}" >> "${AP_CONF}"
    echo "AP_IP=192.168.4.1/24" >> "${AP_CONF}"
    chmod 600 "${AP_CONF}"
    echo "[+] Created ${AP_CONF}"
fi

source "${AP_CONF}"

# 3. Configure NetworkManager Connections (idempotent)
if command -v nmcli &> /dev/null; then
    echo "[*] Configuring NetworkManager profiles..."

    # Check/create SIH-FIELD Hotspot
    if ! nmcli connection show "SIH-FIELD" &> /dev/null; then
        echo "[+] Creating Hotspot profile SIH-FIELD..."
        nmcli connection add type wifi ifname wlan0 con-name "SIH-FIELD" autoconnect yes ssid "${AP_SSID}" 2>/dev/null || true
        nmcli connection modify "SIH-FIELD" 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared ipv4.addresses "${AP_IP}" 2>/dev/null || true
        nmcli connection modify "SIH-FIELD" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "${AP_PASSWORD}" 2>/dev/null || true
    else
        echo "[.] Hotspot profile SIH-FIELD already exists."
    fi

    # Check/create SIH-NODE-01 Client Profile
    if ! nmcli connection show "SIH-NODE-01" &> /dev/null; then
        echo "[+] Creating Client profile SIH-NODE-01..."
        nmcli connection add type wifi ifname wlan0 con-name "SIH-NODE-01" autoconnect no ssid "SIH-NODE-01" 2>/dev/null || true
        nmcli connection modify "SIH-NODE-01" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "sih12345" 2>/dev/null || true
    else
        echo "[.] Client profile SIH-NODE-01 already exists."
    fi
else
    echo "[!] nmcli not found; skipping live NetworkManager profile creation."
fi

# 4. Create Systemd Unit Files

# 4.1 Gateway Service
cat << EOF > /etc/systemd/system/sih-gateway.service
[Unit]
Description=SIH Smart Farming Offline HTTP API Gateway Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/sih-smart-farming
ExecStart=/usr/bin/python3 gateway/server.py --host 0.0.0.0 --port 8080 --db-path data/edge.db
Restart=always
RestartSec=5
KillMode=process
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
echo "[+] Wrote /etc/systemd/system/sih-gateway.service"

# 4.2 Edge Pipeline Service (starts after gateway)
cat << EOF > /etc/systemd/system/sih-pipeline.service
[Unit]
Description=SIH Smart Farming Model A Edge Inference Pipeline Daemon
After=sih-gateway.service
Requires=sih-gateway.service

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/sih-smart-farming
ExecStart=/usr/bin/python3 -u edge/pipeline.py --source /dev/video0 --db-path data/edge.db
Restart=on-failure
RestartSec=10
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
echo "[+] Wrote /etc/systemd/system/sih-pipeline.service"

# 4.3 Mast Collector Service (with flock mutex)
cat << EOF > /etc/systemd/system/sih-collector.service
[Unit]
Description=SIH Smart Farming Ground Mast Telemetry & Trap Pull Collector
After=network-online.target

[Service]
Type=oneshot
User=nvidia
WorkingDirectory=/home/nvidia/sih-smart-farming
ExecStart=/usr/bin/flock -n /run/lock/sih-collector.lock /usr/bin/python3 edge/mast_collector.py --db-path data/edge.db
StandardOutput=journal
StandardError=journal
EOF
echo "[+] Wrote /etc/systemd/system/sih-collector.service"

# 4.4 Mast Collector Boot Service (M3.1 oneshot after gateway is up)
cat << EOF > /etc/systemd/system/sih-collector-boot.service
[Unit]
Description=SIH Smart Farming Initial Post-Boot Mast Collection
After=sih-gateway.service
Requires=sih-gateway.service

[Service]
Type=oneshot
User=nvidia
WorkingDirectory=/home/nvidia/sih-smart-farming
ExecStart=/usr/bin/flock -n /run/lock/sih-collector.lock /usr/bin/python3 edge/mast_collector.py --db-path data/edge.db
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "[+] Wrote /etc/systemd/system/sih-collector-boot.service"

# 4.5 Mast Collector Periodic Timer (M3.1, M3.2: DISABLED by default, minimum 60min interval)
TIMER_INTERVAL="${SIH_COLLECTOR_INTERVAL_MIN:-60min}"
cat << EOF > /etc/systemd/system/sih-collector.timer
[Unit]
Description=Trigger SIH Ground Mast Collection Periodically (Default 60 min, Disabled by default)

[Timer]
OnBootSec=10min
OnUnitActiveSec=${TIMER_INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
EOF
echo "[+] Wrote /etc/systemd/system/sih-collector.timer"

# 5. Reload Systemd Daemon
if command -v systemctl &> /dev/null; then
    systemctl daemon-reload
    echo "[+] Systemd daemon reloaded."
    echo "[*] To enable and start active production services on the Nano:"
    echo "      sudo systemctl enable --now sih-gateway.service"
    echo "      sudo systemctl enable --now sih-collector-boot.service"
    echo "      sudo systemctl enable sih-pipeline.service"
    echo "      # Note: sih-collector.timer is DISABLED by default (app trigger is primary)."
    echo "      # To enable 60-min background polling: sudo systemctl enable --now sih-collector.timer"
fi

echo "========================================================================"
echo " SETUP COMPLETE: Services and timers successfully configured."
echo "========================================================================"
