"""Role-aware mitigation dispatcher.

Thresholds: child >= 0.70 · adult >= 0.85 (configurable via env).
On trigger, builds an incident and hands it to the async alert dispatcher.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import get_settings
from ..fusion.risk_engine import recommendation, risk_band

logger = logging.getLogger("voiceshield.mitigation")
settings = get_settings()

# Cooldown so one bad call does not spam channels every 100 ms
_LAST_ALERT_AT: dict[str, float] = {}
COOLDOWN_S = 45.0


def thresholds_for(role: str) -> float:
    return settings.THRESHOLD_CHILD if role == "child" else settings.THRESHOLD_ADULT


def evaluate_trigger(score: float, role: str) -> dict:
    band = risk_band(score, role)
    t = thresholds_for(role)
    return {
        "band": band,
        "triggered": score >= t,
        "threshold": t,
        "recommendation": recommendation(band, role),
    }


async def maybe_dispatch(
    db_session,
    session_id: str,
    call_id: str,
    role: str,
    language: str,
    score: float,
    triggers: list[str],
    speaker_mismatch: bool,
) -> dict | None:
    """Create an Incident + fire async alerts when threshold crossed & cooldown passed."""
    import time

    from ..models import Incident

    ev = evaluate_trigger(score, role)
    if not ev["triggered"]:
        return None

    now = time.time()
    last = _LAST_ALERT_AT.get(call_id, 0.0)
    if now - last < COOLDOWN_S:
        return None
    _LAST_ALERT_AT[call_id] = now

    severity = "critical" if score >= 0.92 else "high" if score >= 0.85 else "medium"
    incident = Incident(
        session_id=session_id,
        severity=severity,
        score=score,
        role=role,
        language=language,
        triggers=triggers[:8],
        speaker_mismatch=speaker_mismatch,
    )
    db_session.add(incident)
    await db_session.commit()
    await db_session.refresh(incident)

    payload = {
        "incident_id": incident.id,
        "call_id": call_id,
        "severity": severity,
        "score": score,
        "role": role,
        "language": language,
        "triggers": triggers[:8],
        "speaker_mismatch": speaker_mismatch,
        "recommendation": ev["recommendation"],
        "timestamp_ist": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S IST"),
    }

    # Non-blocking alerts — never fail the WS loop
    try:
        from .alerts import dispatch_alerts

        await dispatch_alerts(payload)
    except Exception as exc:
        logger.warning("alert dispatch failed: %s", exc)

    return {"incident_id": incident.id, **payload}
