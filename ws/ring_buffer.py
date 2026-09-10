"""VoiceShield AI — Thread-safe ring buffer for 300ms sliding windows."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

# 300ms at 16 kHz = 4800 samples, hop = 100ms = 1600 samples
WINDOW_SIZE = 4800
HOP_SIZE = 1600
SAMPLE_RATE = 16000


class RingBuffer:
    """
    Lock-free circular buffer holding the most recent PCM samples.
    When enough new samples accumulate (HOP_SIZE), a new window is available.
    """

    def __init__(self, capacity: int = WINDOW_SIZE * 3) -> None:
        self._capacity = max(capacity, WINDOW_SIZE * 2)
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> int:
        """Push new PCM samples. Returns number of complete windows available."""
        n = len(samples)
        with self._lock:
            # Write samples into circular buffer
            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = samples
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = samples[:first]
                self._buf[: n - first] = samples[first:]
            self._write_pos = end % self._capacity
            self._total_written += n

        windows = self._total_written // HOP_SIZE
        return max(0, windows - getattr(self, "_emitted_windows", 0))

    def get_window(self) -> Optional[np.ndarray]:
        """Return the most recent WINDOW_SIZE samples, or None if not enough data."""
        with self._lock:
            if self._total_written < WINDOW_SIZE:
                return None
            start = (self._write_pos - WINDOW_SIZE) % self._capacity
            if start + WINDOW_SIZE <= self._capacity:
                window = self._buf[start : start + WINDOW_SIZE].copy()
            else:
                first = self._capacity - start
                window = np.concatenate(
                    [self._buf[start :], self._buf[: WINDOW_SIZE - first]]
                ).copy()
            return window

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write_pos = 0
            self._total_written = 0
            self._emitted_windows = 0  # type: ignore[attr-defined]

    @property
    def sample_count(self) -> int:
        return self._total_written

    @property
    def window_count(self) -> int:
        return self._total_written // HOP_SIZE
