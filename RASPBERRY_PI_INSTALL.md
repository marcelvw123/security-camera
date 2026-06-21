# Raspberry Pi Install

These steps install the headless version on a Raspberry Pi. They assume the app will live at:

```bash
/home/marcelvw/securitycamera
```

Use Raspberry Pi OS 64-bit if possible. Ethernet is strongly recommended for the Pi and DVR/NVR.

## 1. Copy This Version To The Pi

From your Mac:

```bash
./scripts/package_pi.sh
scp /private/tmp/securitycamera-pi.tar.gz marcelvw@raspberrypi.local:/home/marcelvw/
```

If `raspberrypi.local` does not resolve, use the Pi IP address:

```bash
scp /private/tmp/securitycamera-pi.tar.gz marcelvw@192.168.1.123:/home/marcelvw/
```

On the Pi:

```bash
mkdir -p /home/marcelvw/securitycamera
tar -xzf /home/marcelvw/securitycamera-pi.tar.gz -C /home/marcelvw/securitycamera
cd /home/marcelvw/securitycamera
```

## 2. Install The App

Run the installer:

```bash
cd /home/marcelvw/securitycamera
./scripts/install_pi.sh
```

The installer:

- installs OS packages
- creates `.venv`
- installs `requirements-headless.txt`
- installs the systemd service and timers
- enables the 22:00 start timer and 05:00 stop timer

## Manual Install Alternative

If you do not want to use `scripts/install_pi.sh`, run these manually.

### Install System Packages

On the Pi:

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  python3-opencv \
  python3-numpy \
  python3-pil \
  python3-yaml \
  python3-requests \
  python3-scipy \
  python3-matplotlib \
  python3-pandas \
  python3-psutil \
  python3-tqdm \
  python3-torch \
  python3-torchvision
```

### Create Python Environment

```bash
cd /home/marcelvw/securitycamera
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --no-deps -r requirements-headless.txt
pip install --no-cache-dir --no-deps polars ultralytics-thop
```

If you previously hit a no-space error, clean partial downloads before retrying:

```bash
rm -rf /home/marcelvw/securitycamera/.venv
rm -rf ~/.cache/pip
sudo apt clean
df -h
```

## 3. First Interactive Setup

Run once from a terminal:

```bash
cd /home/marcelvw/securitycamera
. .venv/bin/activate
python headless.py
```

The app will prompt for:

- DVR/NVR IP
- username
- password
- streams to monitor
- sensitivity per stream

It will save this into:

```bash
/home/marcelvw/securitycamera/headless_config.json
```

The file is written with owner-only permissions. Confirm:

```bash
ls -l headless_config.json
```

Expected permissions:

```text
-rw-------
```

Stop the test run with `Ctrl+C`.

## 4. Install systemd Timers

The included timers start the app at 22:00 and stop it at 05:00.

```bash
cd /home/marcelvw/securitycamera
sudo cp deploy/systemd/security-camera-headless.service /etc/systemd/system/
sudo cp deploy/systemd/security-camera-start.timer /etc/systemd/system/
sudo cp deploy/systemd/security-camera-stop.service /etc/systemd/system/
sudo cp deploy/systemd/security-camera-stop.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now security-camera-start.timer
sudo systemctl enable --now security-camera-stop.timer
```

For a manual test start:

```bash
sudo systemctl start security-camera-headless.service
```

View logs:

```bash
journalctl -u security-camera-headless.service -f
```

Stop manually:

```bash
sudo systemctl stop security-camera-headless.service
```

Check timer schedule:

```bash
systemctl list-timers "security-camera-*"
```

## 5. Where Files Are Saved

Clips:

```bash
~/SecurityCamera/clips
```

Debug frames:

```bash
~/SecurityCamera/debug_frames
```

Scenario frames:

```bash
~/SecurityCamera/scenario_frames
```

Headless config:

```bash
/home/marcelvw/securitycamera/headless_config.json
```

Headless log file:

```bash
/home/marcelvw/securitycamera/security_camera_headless.log
```

## 6. RTSP Stability Settings

If streams drop, add these to `/home/marcelvw/securitycamera/.env`:

```env
RTSP_OPEN_TIMEOUT_MS=5000
RTSP_READ_TIMEOUT_MS=8000
RTSP_READ_FAILURES_BEFORE_RECONNECT=20
RTSP_RECONNECT_DELAY_SECONDS=2
RECORDING_FPS=20
DETECT_DURING_RECORDING=false
```

Then restart:

```bash
sudo systemctl restart security-camera-headless.service
```
