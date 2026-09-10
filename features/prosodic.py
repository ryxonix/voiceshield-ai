"""VoiceShield AI — prosodic feature extraction (per 300 ms window).

Spec formulas, computed on transient NumPy arrays only (nothing persists):
- Jitter_local (%)  = mean(|T_i - T_{i+1}|) / mean(T_i) * 100
- Shimmer_local (%) = mean(|A_i - A_{i+1}|) / mean(A_i) * 100
- Pitch Stability   = max(0, 100 - min(jitter_pct * 40, 100))
f0 tracking uses librosa.pyin (probabilistic YIN), 65-400 Hz.
"""
from __future__ import annotations

import numpy as np

_sr: int = 16000


def bind_sample_rate(sr: int) -> None:
    global _sr
    _sr = sr


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 1e-9 else 0.0


def extract_prosodic(window: np.ndarray, sr: int) -> dict:
    """Return jitter_pct, shimmer_pct, pitch_stability, f0_mean."""
    window = np.asarray(window, dtype=np.float32)
    if window.size < 200:
        return {"jitter_pct": 0.0, "shimmer_pct": 0.0, "pitch_stability": 100.0, "f0_mean": 0.0}

    # ---- 1) f0 via YIN -> jitter (cycle-to-cycle period instability) ----
    f0_mean = 0.0
    jitter_pct = 0.0
    try:
        import librosa

        frame = np.pad(window, (0, max(0, 1024 - window.size)))
        f0, _vflag, _vprob = librosa.pyin(
            frame, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=160
        )
        f0_voiced = f0[~np.isnan(f0)]
        if f0_voiced.size >= 4:
            periods = 1.0 / f0_voiced
            jitter_pct = _safe_ratio(
                float(np.mean(np.abs(np.diff(periods)))), float(np.mean(periods))
            ) * 100.0
            f0_mean = float(np.mean(f0_voiced))
    except Exception:  # noqa: BLE001
        jitter_pct = 0.0

    # ---- 2) shimmer (glottal-cycle peak amplitude instability) ----
    shimmer_pct = 0.0
    try:
        pos = window - float(window.min())
        peaks, _props = _find_peaks(pos, prominence=float(np.std(pos)) * 0.5)
        if peaks.size >= 4:
            amps = pos[peaks].astype(np.float64)
            shimmer_pct = _safe_ratio(
                float(np.mean(np.abs(np.diff(amps)))), float(np.mean(amps))
            ) * 100.0
    except Exception:  # noqa: BLE001
        shimmer_pct = 0.0

    pitch_stability = max(0.0, 100.0 - min(jitter_pct * 40.0, 100.0))
    return {
        "jitter_pct": round(jitter_pct, 4),
        "shimmer_pct": round(shimmer_pct, 4),
        "pitch_stability": round(pitch_stability, 4),
        "f0_mean": round(f0_mean, 2),
    }


def _find_peaks(x: np.ndarray, prominence: float):
    from scipy.signal import find_peaks

    return find_peaks(x, prominence=prominence)
