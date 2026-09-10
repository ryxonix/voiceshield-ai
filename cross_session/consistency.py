"""VoiceShield AI — cross-session speaker consistency (pgvector).

Opt-in per session (?compare=true&caller_id=+91...). Once the session has
CROSS_SESSION_MIN_WINDOWS of audio (default 300 = 30 s), the session-level
speaker embedding (mean of window embeddings, L2-normalised) is compared
against the caller's stored genuine samples via pgvector cosine distance.
If the nearest distance exceeds the threshold, an anomaly (0..1) feeds the
0.15 bonus in the fusion engine. Absent DB or samples => anomaly None.
"""
from __future__ import annotations

import logging

import numpy as np

import db
from config import get_settings

log = logging.getLogger("voiceshield.crosssession")


class CrossSessionTracker:
    """Accumulates per-window embeddings and runs the check once enough audio."""

    def __init__(self, call_id: str, caller_id: str | None, compare_enabled: bool) -> None:
        self.call_id = call_id
        self.caller_id = caller_id
        self.compare_enabled = bool(compare_enabled and caller_id and db.enabled())
        self.vectors: list[list[float]] = []
        self.checked = False
        self.result: dict | None = None

    def add(self, vec: list[float]) -> None:
        if not self.compare_enabled or self.checked:
            return
        self.vectors.append(vec)
        if len(self.vectors) >= get_settings().cross_session_min_windows:
            self.checked = True  # single async check per session

    def session_embedding(self) -> list[float] | None:
        if not self.vectors:
            return None
        arr = np.asarray(self.vectors, dtype=np.float64)
        mean = arr.mean(axis=0)
        n = float(np.linalg.norm(mean))
        return [float(v) for v in (mean / n if n > 1e-9 else mean)]

    async def run_check(self) -> dict | None:
        """Async, non-blocking — never on the hot path."""
        if not self.checked or self.result is not None:
            return self.result
        emb = self.session_embedding()
        if emb is None:
            return None
        try:
            matches = await db.nearest_genuine(self.caller_id, emb, top=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-session check failed: %s", exc)
            matches = None
        if not matches:
            self.result = {"status": "no_reference", "anomaly": None}
            return self.result
        nearest = matches[0]
        distance = float(nearest.get("distance", 1.0))
        threshold = get_settings().cross_session_similarity_threshold
        anomaly = max(0.0, min(1.0, (distance - threshold) / max(0.1, threshold)))
        self.result = {
            "status": "anomaly" if anomaly > 0 else "consistent",
            "anomaly": round(anomaly, 4),
            "cosine_distance": round(distance, 4),
            "matched_sample_id": nearest.get("id"),
            "matched_label": nearest.get("label"),
            "samples_compared": len(matches),
        }
        return self.result
