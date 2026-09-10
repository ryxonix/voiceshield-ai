#!/usr/bin/env bash
# Train AASIST-L on Indian voices and deploy INT8 ONNX to the backend.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-data}"
EPOCHS="${2:-12}"

python -m training.train --data "$DATA_DIR" --epochs "$EPOCHS" --out training/weights

mkdir -p backend/inference/model
cp training/weights/aasist_l_int8.onnx backend/inference/model/aasist_l.onnx 2>/dev/null \
  || cp training/weights/aasist_l.onnx backend/inference/model/aasist_l.onnx

echo "Model deployed to backend/inference/model/aasist_l.onnx"
