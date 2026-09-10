# AASIST-L ONNX Weights

The runtime auto-loads an ONNX AASIST-L from this directory (override with `AASIST_ONNX_PATH`).

## Two paths to weights

**1. Train your own on Indian voices (recommended for the college submission):**

```bash
# 1. Collect data (all free):
#    - Common Voice en/hi/kn clips (bonafide)
#    - edge-tts generated en-IN / hi-IN / kn-IN clips (synthetic)
#    - Kaggle "Indian Deepfake Voice" / ASVspoof 2019 LA (synthetic)
# 2. Lay out:
data/en/bonafide/*.wav   data/en/synthetic/*.wav
data/hi/bonafide/*.wav   data/hi/synthetic/*.wav
data/kn/bonafide/*.wav   data/kn/synthetic/*.wav

# 3. Train (CPU fine, ~1–2 h for a few thousand clips):
pip install torch librosa onnxruntime onnx
python -m training.train --data data --epochs 12

# 4. Deploy:
cp training/weights/aasist_l_int8.onnx backend/inference/model/aasist_l.onnx
```

**2. Use pretrained AASIST weights:**
Download the AASIST checkpoint (Jung et al., ICASSP 2022 — MIT license) from the
official repo, convert with `training/aasist_model.py` as the architecture target,
then export + quantize the same way.

Until weights exist, the backend runs a deterministic spectral-heuristic fallback
so the full pipeline (WS streaming, XAI, fusion, alerts, reports) is testable.
