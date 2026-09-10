"""VoiceShield AI — Pydantic schemas for API responses."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class WindowResult(BaseModel):
    """Per-window result emitted over the WebSocket (scores only, never audio)."""
    type: str = "score"
    call_id: str
    t: float
    synthetic_score: float
    model_probability: Optional[float] = None
    detector_mode: str = "heuristic"  # "onnx" | "heuristic"
    jitter_pct: float = 0.0
    shimmer_pct: float = 0.0
    phase_continuity: float = 0.0
    pitch_stability: float = 0.0
    xai_risk: float = 0.0
    cross_session_anomaly: Optional[float] = None
    verified_enterprise: bool = False
    triggered: bool = False
    role: str = "adult"
    threshold: float = 0.85
    action: Optional[dict] = None
    latency_ms: float = 0.0


class SessionOut(BaseModel):
    call_id: str
    role: str
    caller_id: Optional[str] = None
    compare_enabled: bool = False
    started_at: str
    peak_score: float
    alert_triggered: bool
    window_count: int
    last_scores: list = []
    cross_session: Optional[dict] = None


class IncidentOut(BaseModel):
    id: str
    session_call_id: Optional[str] = None
    role: str
    detected_at: str
    peak_score: float
    action_taken: str
    alerts_sent: dict = {}
    report_available: bool = True
    peak_snapshot: dict = {}


class HealthOut(BaseModel):
    status: str
    detector_mode: str
    onnx_loaded: bool
    database: str
    alert_channels: dict
    thresholds: dict


class GenuineSampleIn(BaseModel):
    caller_id: str
    label: str = "genuine"
    embedding: list[float]


class AlertRequestIn(BaseModel):
    title: str = "VoiceShield test alert"
    message: str = "Manual test from the dashboard."
