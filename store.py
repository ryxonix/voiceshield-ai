"""VoiceShield AI — in-memory session registry.

Authoritative live state for the dashboard. Durable history goes to
PostgreSQL when configured; this store is always present so the product
works with zero cloud accounts (DPDP posture: no audio ever held here,
only scalar scores).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionState:
    call_id: str
    role: str = "adult"
    caller_id: str | None = None
    compare_enabled: bool = False
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    peak_score: float = 0.0
    peak_snapshot: dict = field(default_factory=dict)
    alert_triggered: bool = False
    alert_count: int = 0
    last_alert_at: datetime | None = None
    window_count: int = 0
    last_scores: list = field(default_factory=list)  # last 60 snapshots for UI sparkline
    cross_session: dict | None = None
    incident_id: str | None = None

    def as_dict(self, include_history: bool = True) -> dict:
        return {
            "call_id": self.call_id,
            "role": self.role,
            "caller_id": self.caller_id,
            "compare_enabled": self.compare_enabled,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "peak_score": self.peak_score,
            "peak_snapshot": self.peak_snapshot,
            "alert_triggered": self.alert_triggered,
            "window_count": self.window_count,
            "cross_session": self.cross_session,
            "incident_id": self.incident_id,
            "last_scores": self.last_scores if include_history else [],
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._incidents: list[dict] = []
        self._lock = threading.Lock()

    # -- sessions --
    def create(self, call_id: str, role: str, caller_id: str | None, compare_enabled: bool) -> SessionState:
        s = SessionState(call_id=call_id, role=role, caller_id=caller_id, compare_enabled=compare_enabled)
        with self._lock:
            self._sessions[call_id] = s
        return s

    def get(self, call_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(call_id)

    def end(self, call_id: str) -> SessionState | None:
        with self._lock:
            s = self._sessions.pop(call_id, None)
        if s:
            s.ended_at = _now()
            self._archive(s)
        return s

    def active(self) -> list[SessionState]:
        with self._lock:
            return [s for s in self._sessions.values() if s.ended_at is None]

    def all_recent(self, limit: int = 50) -> list[SessionState]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sessions[:limit]

    def record_snapshot(self, s: SessionState, snap: dict, score: float) -> None:
        s.window_count += 1
        s.last_scores.append({"t": snap.get("t"), "score": round(score, 4)})
        del s.last_scores[:-60]
        if score > s.peak_score:
            s.peak_score = score
            s.peak_snapshot = snap

    # -- incidents (in-memory fallback when DB absent) --
    def add_incident(self, incident: dict) -> None:
        with self._lock:
            self._incidents.insert(0, incident)
            del self._incidents[200:]

    def incidents(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._incidents[:limit])

    def get_incident(self, incident_id: str) -> dict | None:
        with self._lock:
            for inc in self._incidents:
                if inc.get("id") == incident_id:
                    return inc
        return None

    def _archive(self, s: SessionState) -> None:
        # Keep ended sessions briefly for the dashboard "recent" list.
        with self._lock:
            del self._sessions[s.call_id] if s.call_id in self._sessions else None


store = SessionStore()


def summary() -> dict:
    settings = get_settings()
    active = store.active()
    scores = [s.peak_score for s in active]
    return {
        "active_sessions": len(active),
        "roles": {
            "adult": sum(1 for s in active if s.role == "adult"),
            "child": sum(1 for s in active if s.role == "child"),
        },
        "highest_live_score": max(scores) if scores else 0.0,
        "thresholds": {
            "adult": settings.adult_threshold,
            "child": settings.child_threshold,
        },
    }
