"""VoiceShield AI — spectral phase continuity (group-delay coherence).

Spec: STFT with n_fft=512, hop=128. For adjacent frames:
  phi_cont = 1 - mean_k( |angle(X_t,k) - angle(X_{t-1},k) - dphi_expected_k| / pi )
where dphi_expected_k = 2*pi*k*hop/n_fft is the linear-phase advance a
time-shifted stationary signal would produce. Bins are energy-weighted so
silence does not dominate the score. Returns value in [0, 1].
"""
from __future__ import annotations

import numpy as np


def phase_continuity(window: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 128) -> float:
    del sr  # kept for API symmetry
    x = np.asarray(window, dtype=np.float32)
    if x.size < n_fft + hop:
        return 1.0

    frames = _frames(x, n_fft, hop)
    if frames.shape[0] < 2:
        return 1.0

    window_fn = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * window_fn[None, :], axis=1)  # [T, K]
    mag = np.abs(spec)
    phase = np.angle(spec)

    floor = np.median(mag, axis=0) + 1e-9
    dphi_expected = 2.0 * np.pi * hop * np.arange(spec.shape[1]) / n_fft

    # wrap-safe phase difference against the expected advance
    dp = np.angle(np.exp(1j * (phase[1:] - phase[:-1] - dphi_expected[None, :])))
    deviation = np.abs(dp) / np.pi  # [0, 1]

    w = np.maximum(mag[1:], mag[:-1]) / (floor[None, :] * 50.0)
    w = np.clip(w, 0.0, 1.0)

    den = float(np.sum(w))
    cont = 1.0 - (float(np.sum(deviation * w)) / den if den > 1e-9 else 0.0)
    return round(float(np.clip(cont, 0.0, 1.0)), 4)


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    count = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]
