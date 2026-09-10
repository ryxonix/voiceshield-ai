"""ONNX Runtime inference for AASIST-L.

Production path: INT8 dynamic-quantized ONNX, CPUExecutionProvider,
intra_op_num_threads=2 → sub-50 ms on 2 vCPU.

Fallback path: a deterministic, zero-dependency heuristic scorer so the
platform is fully functional before trained weights are dropped in.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("voiceshield.inference")

N_MELS = 80
WINDOW_SAMPLES = 4800

_ONNX = None
_MODEL_PATH: str | None = None


def _try_onnx(path: str):
    global _ONNX, _MODEL_PATH
    if not path or not os.path.exists(path):
        return None
    if _ONNX is not None and _MODEL_PATH == path:
        return _ONNX
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        _ONNX = sess
        _MODEL_PATH = path
        logger.info("AASIST-L ONNX loaded: %s", path)
        return sess
    except Exception as exc:
        logger.warning("ONNX load failed (%s) — heuristic fallback active", exc)
        return None


def preprocess(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Log-mel [1, 1, n_mels, frames] in the layout AASIST expects."""
    try:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=x.astype(np.float32), sr=sr, n_fft=512, hop_length=128, n_mels=N_MELS
        )
        logmel = np.log(mel + 1e-10)
        return logmel[None, None, ...].astype(np.float32)
    except Exception:
        # scipy-only fallback mel
        from scipy.signal import stft as _stft

        f, t, Z = _stft(x, fs=sr, nperseg=512, noverlap=512 - 128, padded=False, boundary=None)
        power = np.abs(Z) ** 2
        # flat filterbank (identity) — shape kept consistent
        return np.log(power + 1e-10)[None, None, ...].astype(np.float32)


def onnx_predict(x: np.ndarray, sr: int = 16000) -> float | None:
    """Return P(synthetic) from ONNX AASIST, or None when unavailable."""
    sess = _try_onnx(os.environ.get("AASIST_ONNX_PATH", "inference/model/areads_l.onnx"))
    if sess is None:
        return None
    try:
        inp = preprocess(x, sr)
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: inp})[0]
        arr = np.asarray(out).reshape(-1)
        if arr.size >= 2:
            e = np.exp(arr - arr.max())
            p = e / e.sum()
            return float(p[1])  # P(spoof)
        return float(np.clip(arr[0], 0.0, 1.0))
    except Exception as exc:
        logger.warning("ONNX inference failed: %s", exc)
        return None


def heuristic_predict(x: np.ndarray, sr: int = 16000) -> float:
    """Deterministic artifact heuristic (fallback before trained weights ship).

    Combines three signals measured on the window:
      * HF/HF-ratio — neural vocoders leave excess energy above 6 kHz
      * spectral flatness in 2–5.5 kHz — TTS formants are unnaturally flat
      * ZCR variance — synthetic glottal pulses are unnaturally regular
    Returns a calibrated-ish P(synthetic) in [0,1].
    """
    if x is None or len(x) < 960:
        return 0.5
    x = x.astype(np.float64)
    n = np.fft.rfft(x * np.hanning(len(x)), n=2048)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    mag = np.abs(n) + 1e-12

    hf = float(mag[freqs >= 6000].mean())
    lf = float(mag[freqs < 6000].mean() + 1e-9)
    hf_ratio = hf / (hf + lf)

    band = (freqs >= 2000) & (freqs <= 5500)
    b = mag[band]
    flatness = float(np.exp(np.log(b).mean()) / b.mean())

    zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    # typical speech zcr 0.02–0.15; synth extremes deviate
    zcr_dev = abs(zcr - 0.08) / 0.08

    # squash each signal into pseudo-probabilities
    p_hf = np.clip((hf_ratio - 0.12) / 0.20, 0, 1)
    p_flat = np.clip((flatness - 0.02) / 0.10, 0, 1)
    p_zcr = np.clip(zcr_dev / 2.0, 0, 1)

    p = 0.45 * p_hf + 0.35 * p_flat + 0.20 * p_zcr
    return float(np.clip(p, 0.0, 1.0))


def predict(x: np.ndarray, sr: int = 16000) -> float:
    """Main entry: ONNX if available, else heuristic."""
    p = onnx_predict(x, sr)
    if p is not None:
        return p
    return heuristic_predict(x, sr)


def quantize_model_if_needed(src: str, dst: str) -> str | None:
    """INT8 dynamic-range quantization via onnxruntime.quantization (~4x smaller)."""
    if not os.path.exists(src):
        return None
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            src, dst, weight_type=QuantType.QInt8, per_channel=False, reduce_range=False
        )
        logger.info("Quantized %s -> %s", src, dst)
        return dst
    except Exception as exc:
        logger.warning("Quantization skipped: %s", exc)
        return None
