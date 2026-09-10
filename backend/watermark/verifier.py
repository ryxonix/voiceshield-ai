"""Enterprise watermark verifier.

4096-point FFT over the 300 ms window; scans the VoLTE-optimized 7.0–7.5 kHz
band for a stable pilot tone exceeding 12x the local noise floor. On detection
the pipeline short-circuits: verified_enterprise, synthetic_score = 0.0.
"""
from __future__ import annotations

import numpy as np

FFT_SIZE = 4096
BAND_LO = 7000.0
BAND_HI = 7500.0
SNR_RATIO = 12.0


def verify_watermark(x: np.ndarray, sr: int = 16000) -> dict:
    """Check for enterprise pilot tone. Returns {hit, snr, freq_hz}.

    Note: at sr=16 kHz the Nyquist is 8 kHz so the 7.0–7.5 kHz band is
    representable. Enterprise deployments would run this on 48 kHz taps.
    """
    out = {"hit": False, "snr": 0.0, "freq_hz": 0.0}
    if x is None or len(x) < FFT_SIZE:
        # zero-pad short windows up to 4096
        if x is None or len(x) == 0:
            return out
        x = np.pad(x, (0, FFT_SIZE - len(x)))
    x = x.astype(np.float64)

    # Remove DC and apply Hann to reduce leakage
    x = x - x.mean()
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win, n=FFT_SIZE))
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sr)

    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    if not band.any():
        return out

    band_spec = spec[band]
    if len(band_spec) == 0:
        return out

    peak_idx = np.argmax(band_spec)
    peak = float(band_spec[peak_idx])
    peak_freq = float(freqs[band][peak_idx])

    # Noise floor: median of full spectrum excluding DC and the band itself
    mask = np.ones(len(spec), dtype=bool)
    mask[:3] = False
    mask[band] = False
    noise = float(np.median(spec[mask]))
    if noise <= 0:
        noise = 1e-12

    snr = peak / noise
    out["snr"] = round(snr, 2)
    out["freq_hz"] = round(peak_freq, 1)
    out["hit"] = snr >= SNR_RATIO
    return out
