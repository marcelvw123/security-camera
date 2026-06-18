#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PATH="${PACKAGE_PATH:-/private/tmp/securitycamera-pi.tar.gz}"

tar \
  --exclude="./.git" \
  --exclude="./.github" \
  --exclude="./.idea" \
  --exclude="./.DS_Store" \
  --exclude="./.venv" \
  --exclude="./__pycache__" \
  --exclude="./*/__pycache__" \
  --exclude="./clips" \
  --exclude="./dist" \
  --exclude="./build" \
  --exclude="./*.spec" \
  --exclude="./*.mp4" \
  --exclude="./*.avi" \
  --exclude="./*.mov" \
  --exclude="./*.log" \
  --exclude="./.env" \
  --exclude="./.env.*" \
  --exclude="./headless_config.json" \
  --exclude="./headless_streams.json" \
  --exclude="./yolov8n.pt" \
  --exclude="./securitycamera-pi.tar.gz" \
  -czf "${PACKAGE_PATH}" .

echo "Created ${PACKAGE_PATH}"
