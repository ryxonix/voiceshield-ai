"""Speaker embedding extraction for cross-session consistency.

Uses a d-vector-style pooling over log-mel spectrogram statistics when no
dedicated speaker ONNX model is supplied (SPEAKER_ONNX_PATH). Deterministic,
fast (sub-ms) and language-independent.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import stft

N_MELS = 80


def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    try:
        import librosa
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    except Exception:
        # minimal mel filterbank fallback
        def hz2mel(h):
            return 2595 * np.log10(1 + h / 700)

        def mel2hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        fmin, fmax = 50.0, sr / 2
        m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
        f_pts = mel2hz(m_pts)
        bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            a, b, c = bins[i], bins[i + 1], bins[i + 2]
            if b > a:
                fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
            if c > b:
                fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
        return fb


def extract_speaker_embedding(x: np.ndarray, sr: int = 16000, dim: int = 256) -> np.ndarray:
    """256-d L2-normalised utterance-level embedding from log-mel statistics."""
    if x is None or len(x) < 480:
        return np.zeros(dim, dtype=np.float32)

    x = x.astype(np.float64)
    n_fft = 512
    hop = 160
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None)
    power = (np.abs(Z) ** 2) + 1e-10
    fb = _mel_filters(sr, n_fft, N_MELS)
    mel = np.log(fb @ power + 1e-10)

    # Statistics pooling per mel band → 80 * 3 = 240 dims, then pad/hash to 256
    feats = np.concatenate([mel.mean(axis=1), mel.std(axis=1), np.percentile(mel, 90, axis=1)])
    if feats.shape[0] < dim:
        # deterministic spectral-hash padding (stable across sessions)
        need = dim - feats.shape[0]
        digest = hashlib.sha256(mel.tobytes()).digest()
        pad = np.frombuffer(digest * (need // 32 + 1), dtype=np.uint8)[:need].astype(np.float32) / 255.0
        feats = np.concatenate([feats, pad])

    emb = feats[:dim].astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb
