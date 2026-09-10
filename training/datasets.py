"""Dataset loaders for Indian-voice fine-tuning (English, Hindi, Kannada).

Priority of sources (all free):
 * Common Voice (en/hi/kn) — Mozilla, free download
 * OpenSLR SLR 104/105 (Kannada/Hindi) — free
 * IndicVoices / AI4Bharat corpora — free for research
 * Any local folder: data/<lang>/{bonafide,synthetic}/*.wav|mp3|flac

Synthetic side: any TTS you can generate free (e.g., edge-tts en-IN/hi-IN/kn-IN,
google TTS samples) — the loader accepts a `synthetic/` folder next to bonafide.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment_batch

SR = 16000


@dataclass
class Utterance:
    path: Path
    label: int  # 0 bonafide, 1 synthetic
    lang: str


def scan_dir(root: Path) -> list[Utterance]:
    items: list[Utterance] = []
    if not root.exists():
        return items
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name.lower()
        for label_name, label in (("bonafide", 0), ("synthetic", 1)):
            d = lang_dir / label_name
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                    items.append(Utterance(f, label, lang))
    return items


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def logmel(y: np.ndarray, sr: int = SR, n_mels: int = 80) -> np.ndarray:
    import librosa

    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=n_mels)
    return np.log(m + 1e-10).astype(np.float32)


class IndianSpoofDataset(Dataset):
    """Audio dataset with on-the-fly telecom augmentation and fixed 3 s crops."""

    def __init__(self, root: str | Path, augment: bool = True, crop_s: float = 3.0, langs: tuple = ("en", "hi", "kn")):
        self.items = [u for u in scan_dir(Path(root)) if u.lang in langs]
        self.augment = augment
        self.crop = int(crop_s * SR)
        if not self.items:
            raise RuntimeError(
                f"No audio found under {root}. Expected data/<en|hi|kn>/{{bonafide,synthetic}}/*.wav"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        u = self.items[idx]
        try:
            y = load_audio(u.path)
        except Exception:
            y = np.zeros(self.crop, dtype=np.float32)
        if self.augment and u.label == 1:
            # augment both classes, but heavier on synthetic to mimic telecom
            y = augment_batch(y)
        elif self.augment and random.random() < 0.3:
            y = augment_batch(y)
        # fixed-length crop / pad
        if len(y) >= self.crop:
            off = random.randint(0, len(y) - self.crop)
            y = y[off : off + self.crop]
        else:
            y = np.pad(y, (0, self.crop - len(y)))
        mel = logmel(y)
        # time-crop to 380 frames (~3 s @ hop 128)
        if mel.shape[1] > 380:
            mel = mel[:, :380]
        elif mel.shape[1] < 380:
            mel = np.pad(mel, ((0, 0), (0, 380 - mel.shape[1])))
        return torch.from_numpy(mel)[None, ...], torch.tensor(u.label, dtype=torch.long)


def make_loaders(root: str | Path, batch_size: int = 16, val_split: float = 0.15):
    from torch.utils.data import DataLoader, random_split

    ds = IndianSpoofDataset(root)
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )
