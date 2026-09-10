"""Risk fusion engine — XAI risk, fused synthetic_score, cross-session bonus."""
from __future__ import annotations

from typing import Optional


def xai_risk(phase_continuity: float, jitter_pct: float, shimmer_pct: float) -> float:
    """xai_risk = 0.5*(1-φ_cont) + 0.3*max(0,1-jitter) + 0.2*max(0,1-shimmer/3)."""
    a = (1.0 - float(phase_continuity)) * 0.5
    b = max(0.0, 1.0 - float(jitter_pct) / 1.0) * 0.3
    c = max(0.0, 1.0 - float(shimmer_pct) / 3.0) * 0.2
    return a + b + c


def fuse(
    model_prob: float,
    xai: float,
    speaker_mismatch: bool = False,
    cs_bonus: float = 0.10,
) -> float:
    """synthetic_score = clip(0.7*model + 0.3*clip(xai,0,1) [+ cs_bonus], 0, 1)."""
    base = 0.7 * float(model_prob) + 0.3 * float(np_clip(xai, 0.0, 1.0))
    if speaker_mismatch:
        base += cs_bonus
    return float(np_clip(base, 0.0, 1.0))


def np_clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def risk_band(score: float, role: str, thresholds: Optional[dict] = None) -> str:
    """Band labels used by the dashboard gauge."""
    thresholds = thresholds or {}
    t = thresholds.get(role, 0.85 if role == "child" else 0.85)
    if score >= t:
        return "critical"
    if score >= t * 0.8:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def recommendation(band: str, role: str) -> str:
    if band == "critical":
        if role == "child":
            return "BLOCK sensitive actions · alert guardian · initiate call-back verification"
        return "BLOCK sensitive actions · require multi-factor call-back before approvals"
    if band == "high":
        if role == "child":
            return "Flag for guardian review · pause transactions · verify via trusted channel"
        return "Hold transaction · verify via call-back or MFA before proceeding"
    if band == "medium":
        return "Enable warning banner · monitor session · recommend secondary verification"
    return "No action · continue monitoring"
