"""Spectral phase continuity XAI feature.

φ_cont = 1 - (1/K) * Σ_k |∠X_t[k] - ∠X_{t-1}[k] - Δφ_expected[k]| / π

Measures whether inter-frame phase evolution matches what a natural glottal
source + vocal tract would produce — neural vocoders often violate this.

Also: background noise-floor dropout detection (§4.2) — neural synthesis
occasionally produces frames whose noise floor collapses far below the
surrounding floor; physical recordings never drop that abruptly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

N_FFT = 512
HOP = 128


def noise_floor_dropouts(
    x: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 10,
    drop_db: float = 12.0,
    silence_floor_db: float = -62.0,
) -> int:
    """Count frames where the noise floor drops unnaturally (spec §4.2).

    Per-frame RMS is measured in dBFS over ~10 ms frames. A frame is flagged
    when it falls more than `drop_db` below the running median of the previous
    ~300 ms *while still being above absolute silence* (so natural speech
    pauses and true digital silence are not counted). Physical captures
    (room + mic chain) never step down that fast; vocoder/codecs do.
    """
    if x is None or len(x) < sr * frame_ms // 1000 * 12:
        return 0
    x = np.asarray(x, dtype=np.float64)
    frame = max(1, int(sr * frame_ms / 1000))
    hop = frame // 2
    n_frames = 1 + (len(x) - frame) // hop
    if n_frames < 14:
        return 0

    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[np.clip(idx, 0, len(x) - 1)]
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    dbfs = 20.0 * np.log10(rms)

    med_window = 30  # ~300 ms of history
    med = np.full(n_frames, -np.inf)
    for i in range(12, n_frames):  # need >= 12 prior frames before judging
        med[i] = np.median(dbfs[max(0, i - med_window) : i])

    active = dbfs > silence_floor_db  # ignore true silence gaps
    drops = (dbfs < med - drop_db) & active & (med > silence_floor_db)
    return int(np.count_nonzero(drops))


def phase_continuity(x: np.ndarray, sr: int = 16000) -> float:
    """Return φ_cont in [0, 1] — 1 means perfectly natural phase progression."""
    if x is None or len(x) < N_FFT * 2:
        return 0.5  # neutral
    x = x.astype(np.float64)
    if np.max(np.abs(x)) < 1e-4:
        return 0.5

    f, t, Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", padded=False, boundary=None)
    phase = np.angle(Z)  # [freq, time]
    if phase.shape[1] < 2:
        return 0.5

    # Expected phase advance for each bin over one hop of Δt seconds
    dt = HOP / sr
    k = np.arange(phase.shape[0])
    omega = 2 * np.pi * f
    delta_expected = omega * dt  # modulo 2π inherently in the residual

    diffs = phase[:, 1:] - phase[:, :-1] - delta_expected[:, None]
    # wrap to [-π, π]
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    K = diffs.shape[0]
    mean_resid = np.mean(np.abs(diffs)) / np.pi
    phi_cont = 1.0 - mean_resid
    return float(np.clip(phi_cont, 0.0, 1.0))
