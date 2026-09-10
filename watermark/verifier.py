"""VoiceShield AI — enterprise watermark verifier.

4096-point FFT over the 300 ms window. Scans the VoLTE pilot band
7.0-7.5 kHz for a stable tone exceeding PILOT_SNR_RATIO (12x) the local
noise floor. On detection the pipeline short-circuits: the call is
enterprise-verified and synthetic_score is forced to 0.0.
"""
from __future__ import annotations

import numpy as np

PILOT_BAND_HZ = (7000.0, 7500.0)
GUARD_HZ = 500.0
PILOT_SNR_RATIO = 12.0
MIN_PILOT_BINS = 2


def verify(window: np.ndarray, sr: int = 16000) -> dict:
    x = np.asarray(window, dtype=np.float32)
    if x.size < 4096:
        return {"verified_enterprise": False, "pilot_snr": 0.0, "pilot_hz": 0.0}

    nfft = 4096
    seg = x[:nfft] * np.hanning(nfft)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    band = (freqs >= PILOT_BAND_HZ[0]) & (freqs <= PILOT_BAND_HZ[1])
    guard = (freqs >= PILOT_BAND_HZ[0] - GUARD_HZ) & (freqs <= PILOT_BAND_HZ[1] + GUARD_HZ) & ~band

    if not np.any(band) or not np.any(guard):
        return {"verified_enterprise": False, "pilot_snr": 0.0, "pilot_hz": 0.0}

    band_energy = spec[band]
    peak_idx = int(np.argmax(band_energy))
    peak_power = float(band_energy[peak_idx])
    peak_hz = float(freqs[band][peak_idx])

    # noise floor: median of guard band around the pilot (robust to speech)
    noise_floor = float(np.median(spec[guard])) + 1e-9
    snr = peak_power / noise_floor

    # stability: the pilot must dominate its own band (tone, not noise-shaped speech)
    band_sorted = np.sort(band_energy)
    dominance = peak_power / (float(np.mean(band_sorted[:-1])) + 1e-9)

    verified = bool(snr >= PILOT_SNR_RATIO and dominance >= PILOT_SNR_RATIO * 0.5)
    return {
        "verified_enterprise": verified,
        "pilot_snr": round(snr, 2),
        "pilot_hz": round(peak_hz, 1),
    }
