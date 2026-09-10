"""Prosodic feature extraction — Jitter, Shimmer, Pitch Stability (librosa YIN).

Strictly sub-phonemic / glottal-cycle metrics — no linguistic tokens — making the
features language- and accent-invariant (English, Hindi, Kannada...).
"""
from __future__ import annotations

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:  # pragma: no cover
    _HAS_LIBROSA = False

try:
    import parselmouth  # praat-parselmouth, optional but far more robust
    _HAS_PM = True
    _PM_JITTER_LOCAL = True
except Exception:  # pragma: no cover
    _HAS_PM = False

MIN_F0 = 65.0
MAX_F0 = 400.0


def _jitter_from_periods(periods: np.ndarray) -> float:
    """Jitter_local = mean(|T_i - T_{i+1}|) / mean(T) * 100 %."""
    if len(periods) < 2:
        return 0.0
    diffs = np.abs(np.diff(periods))
    mean_t = float(np.mean(periods))
    if mean_t <= 0:
        return 0.0
    return float(np.mean(diffs) / mean_t * 100.0)


def _shimmer_from_amps(amps: np.ndarray) -> float:
    """Shimmer_local = mean(|A_i - A_{i+1}|) / mean(A) * 100 %."""
    if len(amps) < 2:
        return 0.0
    diffs = np.abs(np.diff(amps))
    mean_a = float(np.mean(amps))
    if mean_a <= 0:
        return 0.0
    return float(np.mean(diffs) / mean_a * 100.0)


def extract_prosodic_features(x: np.ndarray, sr: int = 16000) -> dict:
    """Return jitter_pct, shimmer_pct, pitch_stability, f0_mean."""
    out = {"jitter_pct": 0.0, "shimmer_pct": 0.0, "pitch_stability": 0.0, "f0_mean": 0.0}
    if x is None or len(x) < 480:
        return out

    if _HAS_PM:
        try:
            snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.0, pitch_floor=MIN_F0, pitch_ceiling=MAX_F0)
            periods = pitch.selected_array["frequency"]
            voiced = periods[periods > 0]
            if len(voiced) >= 2:
                periods_s = 1.0 / voiced
                out["jitter_pct"] = _jitter_from_periods(periods_s)
                # amplitude per glottal cycle via intensity of pulse neighborhoods
                pulses = call_pulses(snd)
                amps = pulse_amplitudes(snd, pulses)
                out["shimmer_pct"] = _shimmer_from_amps(amps)
                out["f0_mean"] = float(np.mean(voiced))
        except Exception:
            pass

    if out["jitter_pct"] == 0.0 and _HAS_LIBROSA:
        try:
            f0, vflag, vprob = librosa.pyin(
                x.astype(np.float32),
                fmin=MIN_F0,
                fmax=MAX_F0,
                sr=sr,
                frame_length=1024,
                center=True,
            )
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) >= 3:
                periods = 1.0 / voiced_f0
                out["jitter_pct"] = _jitter_from_periods(periods)
                out["f0_mean"] = float(np.mean(voiced_f0))
                rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
                # align rms frames to voiced pyin frames (same hop when frame_length=1024? pyin uses 1024/1? use nearest)
                m = min(len(rms), len(voiced_f0))
                amps = rms[:m]
                amps = amps[amps > 1e-4]
                if len(amps) >= 3:
                    out["shimmer_pct"] = _shimmer_from_amps(amps)
        except Exception:
            pass

    jitter = max(0.0, out["jitter_pct"])
    out["pitch_stability"] = max(0.0, 100.0 - min(jitter * 40.0, 100.0))
    return out


def call_pulses(snd):
    try:
        from parselmouth.praat import call
        return call(snd, "To PointProcess (periodic, cc)", MIN_F0, MAX_F0)
    except Exception:
        return None


def pulse_amplitudes(snd, pulses):
    try:
        from parselmouth.praat import call
        if pulses is None:
            return np.asarray([])
        n = call(pulses, "Get number of points")
        if n < 3:
            return np.asarray([])
        vals = []
        for i in range(1, int(n) + 1):
            t = call(pulses, "Get time from index", i)
            try:
                a = call(snd, "Get value at time", t, "cubic")
            except Exception:
                a = 0.0
            vals.append(float(a))
        return np.asarray(vals)
    except Exception:
        return np.asarray([])
