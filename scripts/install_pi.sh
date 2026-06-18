#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/marcelvw/securitycamera}"
APP_USER="${APP_USER:-marcelvw}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this script as ${APP_USER}, not with sudo."
  exit 1
fi

if [ "$(id -un)" != "${APP_USER}" ]; then
  echo "Warning: running as $(id -un), but APP_USER is ${APP_USER}."
  echo "Set APP_USER=$(id -un) if this is intentional."
fi

cd "${APP_DIR}"

echo "Installing system packages..."
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg libgl1 libglib2.0-0

echo "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
  "${PYTHON_BIN}" -m venv .venv
fi

echo "Installing Python dependencies..."
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing systemd units..."
sudo cp deploy/systemd/security-camera-headless.service /etc/systemd/system/
sudo cp deploy/systemd/security-camera-start.timer /etc/systemd/system/
sudo cp deploy/systemd/security-camera-stop.service /etc/systemd/system/
sudo cp deploy/systemd/security-camera-stop.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now security-camera-start.timer
sudo systemctl enable --now security-camera-stop.timer

echo
echo "Install complete."
echo
echo "Next, run first-time setup if headless_config.json does not exist:"
echo "  cd ${APP_DIR}"
echo "  . .venv/bin/activate"
echo "  python headless.py"
echo
echo "Manual service test:"
echo "  sudo systemctl start security-camera-headless.service"
echo "  journalctl -u security-camera-headless.service -f"
