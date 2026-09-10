"""VoiceShield AI — Pydantic data models."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field

class SessionRole(str, Enum):
    ADULT = "adult"
    CHILD = "child"

class MitigationAction(str, Enum):
    CHILD_SHIELD = "child_shield"
    RISK_BANNER = "risk_banner"
    NONE = "none"

class RiskSnapshot(BaseModel):
    model_config = {"protected_namespaces": ()}
    call_id: str
    window_index: int
    model_prob: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    phase_continuity: float = 0.0
    pitch_stability: float = 0.0
    xai_risk: float = 0.0
    cross_session_anomaly: Optional[float] = None
    synthetic_score: float = 0.0
    verified_enterprise: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_ws_json(self) -> Dict[str, Any]:
        return {
            "type": "risk_update",
            "call_id": self.call_id,
            "window_index": self.window_index,
            "scores": {
                "model_prob": round(self.model_prob, 4),
                "jitter_pct": round(self.jitter, 4),
                "shimmer_pct": round(self.shimmer, 4),
                "phase_continuity": round(self.phase_continuity, 4),
                "pitch_stability": round(self.pitch_stability, 4),
                "xai_risk": round(self.xai_risk, 4),
                "cross_session_anomaly": round(self.cross_session_anomaly, 4) if self.cross_session_anomaly is not None else None,
                "synthetic_score": round(self.synthetic_score, 4),
            },
            "verified_enterprise": self.verified_enterprise,
            "timestamp": self.timestamp.isoformat(),
        }

class MitigationMessage(BaseModel):
    action: MitigationAction
    severity: str = "low"
    overlay: bool = False
    mute: bool = False
    message: str = ""

    def to_ws_json(self) -> Dict[str, Any]:
        return {
            "type": "mitigation",
            "action": self.action.value,
            "severity": self.severity,
            "overlay": self.overlay,
            "mute": self.mute,
            "message": self.message,
        }

class SessionCreate(BaseModel):
    call_id: str
    caller_id: Optional[str] = None
    role: SessionRole = SessionRole.ADULT
    compare_enabled: bool = False

class SessionResponse(BaseModel):
    id: UUID
    call_id: str
    caller_id: Optional[str] = None
    role: str
    compare_enabled: bool
    started_at: datetime
    ended_at: Optional[datetime] = None
    peak_score: float = 0.0
    alert_triggered: bool = False

class IncidentResponse(BaseModel):
    id: UUID
    session_id: UUID
    detected_at: datetime
    peak_score: float
    role: str
    action_taken: str
    alerts_sent: Dict[str, Any] = {}

class PrivacyResponse(BaseModel):
    statement: str = (
        "VoiceShield AI processes audio ephemerally in RAM only. "
        "No raw audio is ever stored on disk, transmitted to cloud servers, "
        "or persisted in any database. Only mathematical sub-scores and "
        "scalar risk values are retained. Speaker embeddings are one-way "
        "transforms that cannot be inverted to reconstruct audio."
    )
    data_retention: str = "Session metadata and scores retained for 30 days. Audio overwritten every 100ms."
    compliance_framework: str = "DPDP Act 2023, IT Act 2000"
    contact: str = "Report concerns to cybercrime.gov.in"
