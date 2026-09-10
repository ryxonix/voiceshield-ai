"""VoiceShield AI — AASIST-L ONNX Runtime inference wrapper."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger("voiceshield.inference")

# Fallback: if no ONNX model is present, use a deterministic heuristic
_MODEL_AVAILABLE = False
_session: Optional[object] = None


def _load_model() -> None:
    """Load the ONNX model lazily on first inference call."""
    global _session, _MODEL_AVAILABLE  # noqa: PLW0603

    model_path = Path(settings.ONNX_MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            "ONNX model not found at %s — using heuristic fallback. "
            "Place your trained aasist_l.onnx in inference/model/",
            model_path,
        )
        return

    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.ONNX_NUM_THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        _MODEL_AVAILABLE = True
        logger.info("AASIST-L ONNX model loaded from %s", model_path)
    except Exception as e:
        logger.error("Failed to load ONNX model: %s", e)


def predict(window: np.ndarray) -> tuple[float, float]:
    """
    Run inference on a 300ms window (4800 samples, float32).

    Returns:
        (model_probability, inference_time_ms)
        model_probability is in [0.0, 1.0] — higher = more likely synthetic.
    """
    if not _MODEL_AVAILABLE:
        _load_model()

    t0 = time.perf_counter()

    if _MODEL_AVAILABLE and _session is not None:
        prob = _run_onnx(window)
    else:
        prob = _heuristic_fallback(window)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return prob, elapsed_ms


def _run_onnx(window: np.ndarray) -> float:
    """Run the real ONNX model inference."""
    import onnxruntime as ort

    assert _session is not None  # noqa: S101

    # Prepare input — assume model expects (1, 1, 4800) or (1, 4800)
    input_meta = _session.get_inputs()  # type: ignore[union-attr]
    input_name = input_meta[0].name
    input_shape = input_meta[0].shape

    if len(input_shape) == 3:
        data = window.reshape(1, 1, -1).astype(np.float32)
    elif len(input_shape) == 2:
        data = window.reshape(1, -1).astype(np.float32)
    else:
        data = window.astype(np.float32)

    outputs = _session.run(None, {input_name: data})  # type: ignore[union-attr]
    raw = float(outputs[0].flatten()[0])

    # Convert logit to probability
    prob = 1.0 / (1.0 + np.exp(-raw)) if abs(raw) < 30 else (1.0 if raw > 0 else 0.0)
    return float(np.clip(prob, 0.0, 1.0))


def _heuristic_fallback(window: np.ndarray) -> float:
    """
    Deterministic heuristic when no model is present.
    Uses spectral flatness as a proxy for synthetic likelihood.
    Synthetic audio tends to have flatter spectra than natural speech.
    """
    fft = np.abs(np.fft.rfft(window))
    fft = np.maximum(fft, 1e-10)

    log_mean = np.mean(np.log(fft))
    log_amean = np.mean(fft)
    flatness = np.exp(log_mean) / log_amean if log_amean > 0 else 0.5

    # Natural speech flatness is typically 0.01–0.15; synthetic often > 0.2
    score = np.clip((flatness - 0.15) / 0.35, 0.0, 1.0)
    return float(score)
