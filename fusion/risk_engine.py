"""VoiceShield AI — risk fusion engine.

XAI risk (spec formula):
  xai_risk = (1 - phase_continuity)*0.5
           + max(0, 1 - jitter_pct/1.0)*0.3
           + max(0, 1 - shimmer_pct/3.0)*0.2

Fused score (spec formula + cross-session bonus):
  synthetic_score = clip(0.7*model_prob + 0.3*clip(xai_risk,0,1) + anomaly_bonus, 0, 1)
  anomaly_bonus   = 0.15 * anomaly (0 when cross-session is off)
"""
from __future__ import annotations


def xai_risk(jitter_pct: float, shimmer_pct: float, phase_continuity: float) -> float:
    j = max(0.0, 1.0 - (jitter_pct / 1.0)) * 0.3
    s = max(0.0, 1.0 - (shimmer_pct / 3.0)) * 0.2
    p = (1.0 - phase_continuity) * 0.5
    return float(j + s + p)


def fuse(model_probability: float, xai: float, anomaly: float | None = None) -> float:
    anomaly = 0.0 if anomaly is None else max(0.0, min(1.0, anomaly))
    bonus = 0.15 * anomaly
    score = 0.7 * max(0.0, min(1.0, model_probability)) + 0.3 * max(0.0, min(1.0, xai)) + bonus
    return float(max(0.0, min(1.0, score)))
