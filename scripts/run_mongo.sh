#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

python annotator_server.py \
  --slides-dir "${SLIDES_DIR:-./slides}" \
  --storage mongo \
  --mongo_db "${MONGO_DB:-annotator_db}" \
  --checkpoint "${CHECKPOINT:-./weights/multihead_classifier.pth}" \
  --mitotic_checkpoint "${MITOTIC_CHECKPOINT:-./weights/mitotic_classifier.pth}" \
  --mitotic_threshold "${MITOTIC_THRESHOLD:-0.9}" \
  --max-slide-dim "${MAX_SLIDE_DIM:-7000}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-5006}"
