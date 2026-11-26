#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

python annotator_server.py \
  --slides-dir "${SLIDES_DIR:-./slides}" \
  --storage "${STORAGE_BACKEND:-sqlite}" \
  --checkpoint "${CHECKPOINT:-./weights/multihead_classifier.pth}" \
  --mitotic_checkpoint "${MITOTIC_CHECKPOINT:-./weights/mitotic_classifier.pth}" \
  --mitotic_threshold "${MITOTIC_THRESHOLD:-0.9}" \
  --max-slide-dim "${MAX_SLIDE_DIM:-7000}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-5006}"
