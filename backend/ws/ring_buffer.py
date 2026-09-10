"""300 ms sliding-window ring buffer — 4,800 samples @ 16 kHz with 1,600-sample hop.

Push raw int16 LE PCM chunks; emits analysis-ready float32 windows on every hop
boundary, tagged with the window's end-time in milliseconds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass
class Window:
    samples: np.ndarray  # float32 in [-1, 1]
    t_ms: int            # end time of the window relative to stream start


@dataclass
class RingBuffer:
    sample_rate: int = 16000
    window_samples: int = 4800          # 300 ms @ 16 kHz
    hop_samples: int = 1600             # 100 ms hop
    max_frame_bytes: int = 192000

    _buf: Deque[float] = field(init=False, repr=False, default_factory=deque)
    _since_hop: int = field(init=False, default=0)
    total_samples: int = field(init=False, default=0)

    @property
    def t_ms(self) -> int:
        return int(self.total_samples * 1000 / self.sample_rate)

    def push(self, pcm_bytes: bytes) -> list[Window]:
        """Push raw little-endian int16 PCM; returns windows ready for analysis."""
        n_new = len(pcm_bytes) // 2
        if n_new == 0:
            return []
        arr = np.frombuffer(pcm_bytes[: n_new * 2], dtype="<i2").astype(np.float32) / 32768.0
        self._buf.extend(arr.tolist())
        self.total_samples += n_new
        self._since_hop += n_new

        windows: list[Window] = []
        while len(self._buf) >= self.window_samples and self._since_hop >= self.hop_samples:
            self._since_hop -= self.hop_samples
            while len(self._buf) > self.window_samples:
                self._buf.popleft()
            samples = np.asarray(self._buf, dtype=np.float32)
            windows.append(Window(samples=samples, t_ms=self.t_ms))
        return windows
