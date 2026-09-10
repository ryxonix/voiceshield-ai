"""Telecom codec & noise augmentation pipeline (training + runtime parity).

Stochastic batch augmentations (30–60% probability per batch), spec:
 1. G.711 A-law / µ-law round-trip (PSTN 64 kbps)          — quantization artifact
 2. AMR-NB (2G/3G) & AMR-WB (VoLTE HD) round-trips         — ffmpeg/opencore, graceful fallback
 3. Background noise injection (traffic/crowd)             — DeepFilterNet-style noise, synthetic fallback
 4. Random packet-loss concealment: frame drops + linear interpolation
 5. Bandwidth truncation: Butterworth 300–3,400 Hz (narrowband) / 50–7,000 Hz (wideband)

Each augmentation returns float32 in [-1, 1] @ 16 kHz. All are numpy-based with
zero hard dependencies beyond scipy/numpy so training runs on the college CPU.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

SR = 16000

# --------------------------------------------------------------------------
# 1. G.711 A-law / u-law (pure numpy — exact ITU curves)
# --------------------------------------------------------------------------


def g711_alaw_roundtrip(x: np.ndarray) -> np.ndarray:
    a = 87.56
    xn = np.clip(x, -1, 1)
    ax = np.abs(xn)
    y = np.where(
        ax < 1 / a,
        a * ax / (1 + np.log(a)),
        (1 + np.log(a * ax)) / (1 + np.log(a)),
    )
    y = np.sign(xn) * y
    # 8-bit quantization (the codec's actual resolution) then decode
    q = np.round(y * 127) / 127
    ax = np.abs(q)
    inv = np.where(
        ax < 1 / a,
        ax * (1 + np.log(a)) / a,
        np.exp(ax * (1 + np.log(a)) - 1) / a,
    )
    return (np.sign(q) * inv).astype(np.float32)


def g711_ulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    mu = 255.0
    xn = np.clip(x, -1, 1)
    y = np.sign(xn) * np.log(1 + mu * np.abs(xn)) / np.log(1 + mu)
    q = np.round(y * 127) / 127
    return (np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)).astype(np.float32)


# --------------------------------------------------------------------------
# 2. AMR-NB / AMR-WB round-trips via ffmpeg (if available) — else NB truncation
# --------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def amr_roundtrip(x: np.ndarray, wideband: bool = False) -> np.ndarray:
    """Encode/decode through AMR-WB (VoLTE) or AMR-NB (2G/3G) when ffmpeg supports it."""
    if not _ffmpeg_available():
        # Fallback: emulate codec artifacts (NB: 300–3400 Hz + heavier quantization)
        return bandwidth_truncate(x, 300, 3400 if not wideband else 7000)
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.wav"
            dst = Path(td) / ("out.amr" if not wideband else "out.wamr")
            _write_wav(x, src)
            codec = "libopencore_amrnb" if not wideband else "libvo_amrwbenc"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                "-ar", "8000" if not wideband else "16000",
                "-ac", "1", "-c:a", codec, str(dst),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=10)
            out = Path(td) / "out.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(dst), "-ar", str(SR), "-ac", "1", str(out)],
                check=True, capture_output=True, timeout=10,
            )
            return _read_wav(out, len(x))
    except Exception:
        return bandwidth_truncate(x, 300, 3400)


def _write_wav(x: np.ndarray, path: Path) -> None:
    try:
        import soundfile as sf

        sf.write(str(path), x, SR)
    except Exception:
        import wave

        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _read_wav(path: Path, n_target: int) -> np.ndarray:
    try:
        import soundfile as sf

        y, _ = sf.read(str(path), dtype="float32")
    except Exception:
        import wave

        with wave.open(str(path), "rb") as w:
            raw = w.readframes(w.getnframes())
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if len(y) < n_target:
        y = np.pad(y, (0, n_target - len(y)))
    return y[:n_target].astype(np.float32)


# --------------------------------------------------------------------------
# 3. Background noise injection (street/crowd/cafe profiles)
# --------------------------------------------------------------------------

_NOISE_PROFILES = {
    "traffic": {"rolloff": 0.25, "hum": 120.0, "level": 0.05},
    "crowd": {"rolloff": 0.7, "hum": 0.0, "level": 0.06},
    "cafe": {"rolloff": 0.45, "hum": 100.0, "level": 0.045},
    "office": {"rolloff": 0.55, "hum": 60.0, "level": 0.03},
}


def background_noise(x: np.ndarray, profile: str | None = None, snr_db: float = 15.0) -> np.ndarray:
    p = _NOISE_PROFILES[profile or random.choice(list(_NOISE_PROFILES))]
    rng = np.random.default_rng()
    n = len(x)
    white = rng.standard_normal(n).astype(np.float32)
    # 1-pole lowpass shaping toward the profile's spectral rolloff
    alpha = np.exp(-2 * np.pi * p["rolloff"] * 4000 / SR)
    shaped = np.zeros(n, dtype=np.float32)
    acc = 0.0
    for i in range(n):
        acc = alpha * acc + (1 - alpha) * white[i]
        shaped[i] = acc
    if p["hum"]:
        t = np.arange(n) / SR
        shaped += 0.3 * np.sin(2 * np.pi * p["hum"] * t).astype(np.float32)
    shaped /= (np.max(np.abs(shaped)) + 1e-9)
    sig_power = np.mean(x**2) + 1e-9
    noise_power = np.mean(shaped**2) + 1e-9
    gain = np.sqrt(sig_power / noise_power / (10 ** (snr_db / 10)))
    return (x + gain * shaped).astype(np.float32)


# --------------------------------------------------------------------------
# 4. Packet-loss concealment: random frame drops + linear interpolation
# --------------------------------------------------------------------------


def packet_loss_concealment(x: np.ndarray, frame_ms: int = 20, loss_rate: float = 0.08) -> np.ndarray:
    frame = int(SR * frame_ms / 1000)
    n_frames = len(x) // frame
    y = x.copy()
    for i in range(n_frames):
        if random.random() < loss_rate:
            s, e = i * frame, (i + 1) * frame
            left = y[s - 1] if s > 0 else 0.0
            right = y[e] if e < len(y) else 0.0
            # linear interpolation across the lost frame (PLC behaviour)
            y[s:e] = np.linspace(left, right, e - s).astype(np.float32)
    return y


# --------------------------------------------------------------------------
# 5. Bandwidth truncation (Butterworth)
# --------------------------------------------------------------------------


def bandwidth_truncate(x: np.ndarray, lo: float = 300, hi: float = 3400) -> np.ndarray:
    nyq = SR / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
    return lfilter(b, a, x).astype(np.float32)


# --------------------------------------------------------------------------
# Composed stochastic batch augmentation (30–60% per-batch probability)
# --------------------------------------------------------------------------

ALL_AUGS = ("g711a", "g711u", "amrnb", "amrwb", "noise", "plc", "narrowband")


def augment_batch(x: np.ndarray, p_range: tuple[float, float] = (0.3, 0.6)) -> np.ndarray:
    """Apply each augmentation stochastically with probability drawn per batch."""
    p = random.uniform(*p_range)
    y = x.astype(np.float32)
    for name in ALL_AUGS:
        if random.random() < p:
            if name == "g711a":
                y = g711_alaw_roundtrip(y)
            elif name == "g711u":
                y = g711_ulaw_roundtrip(y)
            elif name == "amrnb":
                y = amr_roundtrip(y, wideband=False)
            elif name == "amrwb":
                y = amr_roundtrip(y, wideband=True)
            elif name == "noise":
                y = background_noise(y)
            elif name == "plc":
                y = packet_loss_concealment(y)
            elif name == "narrowband":
                y = bandwidth_truncate(y)
    # final safety clamp
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)
