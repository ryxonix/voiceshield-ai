"""VoiceShield AI — Role-aware mitigation dispatcher."""

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from models import MitigationAction, MitigationMessage

logger = logging.getLogger("voiceshield.mitigation")


def evaluate_mitigation(
    synthetic_score: float,
    role: str,
    peak_score: float = 0.0,
    window_index: int = 0,
) -> Optional[MitigationMessage]:
    """
    Evaluate whether mitigation action should be triggered based on
    the synthetic score, session role, and thresholds.

    Returns a MitigationMessage if threshold is exceeded, None otherwise.
    """
    threshold = (
        settings.CHILD_THRESHOLD
        if role == "child"
        else settings.ADULT_THRESHOLD
    )

    if synthetic_score < threshold:
        return None

    logger.warning(
        "Threshold exceeded: score=%.4f >= threshold=%.4f (role=%s, peak=%.4f)",
        synthetic_score,
        threshold,
        role,
        peak_score,
    )

    if role == "child":
        return MitigationMessage(
            action=MitigationAction.CHILD_SHIELD,
            severity="critical",
            overlay=True,
            mute=True,
            message=(
                f"⚠️ CHILD SHIELD ACTIVATED — Potential deepfake detected "
                f"(score: {synthetic_score:.1%}). Audio muted. "
                f"Contact a guardian immediately."
            ),
        )
    else:
        severity = "critical" if synthetic_score >= 0.9 else "high"
        return MitigationMessage(
            action=MitigationAction.RISK_BANNER,
            severity=severity,
            overlay=False,
            mute=False,
            message=(
                f"⚠️ HIGH RISK — Potential voice impersonation detected "
                f"(score: {synthetic_score:.1%}). "
                f"Verify caller identity through secondary channels before proceeding."
            ),
        )


def should_generate_forensic_report(
    peak_score: float,
    threshold: float,
) -> bool:
    """Determine if a forensic PDF report should be generated for this session."""
    return peak_score >= threshold
