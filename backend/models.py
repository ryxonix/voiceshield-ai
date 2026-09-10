"""SQLAlchemy ORM models + Pydantic API schemas for VoiceShield AI."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import IS_POSTGRES

if IS_POSTGRES:
    try:
        from pgvector.sqlalchemy import Vector
    except Exception:  # pragma: no cover
        Vector = None
else:
    Vector = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base:
    pass


from sqlalchemy.orm import DeclarativeBase


class _B(DeclarativeBase):
    pass


Base = _B


class CallSession(Base):
    __tablename__ = "call_sessions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=new_id)
    call_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(16), default="adult")  # adult | child
    language: Mapped[str] = mapped_column(String(16), default="en")  # en | hi | kn
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|closed
    window_count: Mapped[int] = mapped_column(Integer, default=0)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0)
    last_score: Mapped[float] = mapped_column(Float, default=0.0)
    enterprise_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)

    windows: Mapped[list["WindowAnalysis"]] = relationship(back_populates="session", cascade="all,delete")
    incidents: Mapped[list["Incident"]] = relationship(back_populates="session", cascade="all,delete")


class WindowAnalysis(Base):
    __tablename__ = "window_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("call_sessions.id"), index=True)
    t_ms: Mapped[int] = mapped_column(Integer, default=0)
    model_prob: Mapped[float] = mapped_column(Float, default=0.0)
    xai_risk: Mapped[float] = mapped_column(Float, default=0.0)
    synthetic_score: Mapped[float] = mapped_column(Float, default=0.0)
    jitter_pct: Mapped[float] = mapped_column(Float, default=0.0)
    shimmer_pct: Mapped[float] = mapped_column(Float, default=0.0)
    phase_continuity: Mapped[float] = mapped_column(Float, default=0.0)
    pitch_stability: Mapped[float] = mapped_column(Float, default=0.0)
    watermark_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    verdict: Mapped[str] = mapped_column(String(24), default="benign")
    features: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[CallSession] = relationship(back_populates="windows")


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("call_sessions.id"), index=True)
    severity: Mapped[str] = mapped_column(String(12), default="medium")  # low|medium|high|critical
    score: Mapped[float] = mapped_column(Float, default=0.0)
    role: Mapped[str] = mapped_column(String(16), default="adult")
    language: Mapped[str] = mapped_column(String(16), default="en")
    triggers: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    speaker_mismatch: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    report_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[CallSession] = relationship(back_populates="incidents")


class SpeakerRef(Base):
    """Stored genuine speaker embeddings for cross-session consistency (pgvector)."""
    __tablename__ = "speaker_refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(64), index=True)  # speaker identity label
    language: Mapped[str] = mapped_column(String(16), default="en")
    if IS_POSTGRES and Vector is not None:
        embedding = mapped_column(Vector(256))
    else:
        embedding: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)  # {"v": [...]}
    frame_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ----------------------------- Pydantic schemas ------------------------------


class WindowOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    t_ms: int
    model_prob: float
    xai_risk: float
    synthetic_score: float
    jitter_pct: float
    shimmer_pct: float
    phase_continuity: float
    pitch_stability: float
    watermark_hit: bool
    verdict: str


class SessionOut(BaseModel):
    id: str
    call_id: str
    role: str
    language: str
    status: str
    window_count: int
    max_score: float
    avg_score: float
    last_score: float
    enterprise_verified: bool
    started_at: datetime

    class Config:
        from_attributes = True


class IncidentOut(BaseModel):
    id: str
    session_id: str
    severity: str
    score: float
    role: str
    language: str
    triggers: list
    speaker_mismatch: bool
    acknowledged: bool
    report_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MitigationEvent(BaseModel):
    """Client-side mitigation directive (spec §4.4/§4.5)."""
    model_config = {"protected_namespaces": ()}

    type: str = "mitigation"
    action: str              # child_shield | risk_banner
    severity: str            # critical | high
    overlay: bool            # render full-screen protective banner
    mute: bool               # client should mute/pause audio output
    score: float
    threshold: float
    role: str
    message: str
    incident_id: Optional[str] = None


class AnalysisEvent(BaseModel):
    """WebSocket JSON event emitted per window."""
    model_config = {"protected_namespaces": ()}

    type: str = "analysis"
    call_id: str
    t_ms: int
    synthetic_score: float
    model_prob: float
    xai_risk: float
    latency_ms: float
    prosody: dict
    watermark_hit: bool
    verdict: str
    risk_band: str
    recommendation: str
    session_id: str
    speaker_mismatch: Optional[bool] = None
