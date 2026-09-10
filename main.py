"""VoiceShield AI — FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from config import settings
from database import init_db, get_session
from models import (
    IncidentResponse,
    PrivacyResponse,
    SessionResponse,
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("voiceshield")

IST = timezone(timedelta(hours=5, minutes=30))

# ── Sentry (optional) ─────────────────────────────────────────
if settings.SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.5,
    )
    logger.info("Sentry initialized")


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    if settings.DATABASE_URL:
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.error("Database init failed: %s", e)
    else:
        logger.warning("DATABASE_URL not set — running without database")
    yield


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceShield AI",
    description="Real-time audio deepfake detection and mitigation platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket router ──────────────────────────────────────────
from ws.handler import router as ws_router

app.include_router(ws_router)


# ── REST Endpoints ────────────────────────────────────────────

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "VoiceShield AI",
        "version": "1.0.0",
        "timestamp": datetime.now(IST).isoformat(),
    }


@app.get("/api/privacy", response_model=PrivacyResponse)
async def privacy_policy() -> PrivacyResponse:
    """DPDP Act compliance statement."""
    return PrivacyResponse()


@app.get("/api/sessions")
async def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent sessions."""
    if not settings.DATABASE_URL:
        return []
    try:
        async with get_session() as db:
            result = await db.execute(
                text(
                    """
                    SELECT id, call_id, caller_id, role, compare_enabled,
                           started_at, ended_at, peak_score, alert_triggered
                    FROM sessions
                    ORDER BY started_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "id": str(row[0]),
                    "call_id": row[1],
                    "caller_id": row[2],
                    "role": row[3],
                    "compare_enabled": row[4],
                    "started_at": row[5].isoformat() if row[5] else None,
                    "ended_at": row[6].isoformat() if row[6] else None,
                    "peak_score": row[7],
                    "alert_triggered": row[8],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        return []


@app.get("/api/incidents")
async def list_incidents(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent incidents."""
    if not settings.DATABASE_URL:
        return []
    try:
        async with get_session() as db:
            result = await db.execute(
                text(
                    """
                    SELECT i.id, i.session_id, i.detected_at, i.peak_score,
                           i.role, i.action_taken, i.alerts_sent,
                           s.call_id
                    FROM incidents i
                    JOIN sessions s ON s.id = i.session_id
                    ORDER BY i.detected_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "id": str(row[0]),
                    "session_id": str(row[1]),
                    "detected_at": row[2].isoformat() if row[2] else None,
                    "peak_score": row[3],
                    "role": row[4],
                    "action_taken": row[5],
                    "alerts_sent": row[6] if isinstance(row[6], dict) else {},
                    "call_id": row[7],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error("Failed to list incidents: %s", e)
        return []


@app.get("/api/incidents/{incident_id}/report")
async def download_report(incident_id: str) -> Response:
    """Generate and download a forensic PDF report for an incident."""
    if not settings.DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with get_session() as db:
            # Fetch incident
            result = await db.execute(
                text(
                    """
                    SELECT i.id, i.session_id, i.detected_at, i.peak_score,
                           i.role, i.action_taken,
                           s.call_id, s.caller_id, s.started_at, s.ended_at,
                           s.peak_score AS session_peak
                    FROM incidents i
                    JOIN sessions s ON s.id = i.session_id
                    WHERE i.id = :id
                    """
                ),
                {"id": incident_id},
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Incident not found")

            session_data = {
                "call_id": row[6],
                "caller_id": row[7],
                "role": row[4],
                "peak_score": row[3],
                "action_taken": row[5],
                "started_at": row[8],
                "ended_at": row[9],
                "detected_at": row[2],
            }

            # Fetch peak risk snapshot
            snap_result = await db.execute(
                text(
                    """
                    SELECT model_prob, jitter, shimmer, phase_continuity,
                           pitch_stability, xai_risk, cross_session_anomaly, synthetic_score
                    FROM risk_snapshots
                    WHERE session_id = :sid
                    ORDER BY synthetic_score DESC
                    LIMIT 1
                    """
                ),
                {"sid": str(row[1])},
            )
            snap = snap_result.fetchone()
            peak_snapshot = {
                "model_prob": snap[0] if snap else 0.0,
                "jitter": snap[1] if snap else 0.0,
                "shimmer": snap[2] if snap else 0.0,
                "phase_continuity": snap[3] if snap else 0.0,
                "pitch_stability": snap[4] if snap else 0.0,
                "xai_risk": snap[5] if snap else 0.0,
                "cross_session_anomaly": snap[6] if snap else None,
                "synthetic_score": snap[7] if snap else 0.0,
            }

        # Generate PDF
        from forensics.pdf_report import generate_forensic_report
        pdf_bytes = generate_forensic_report(session_data, peak_snapshot)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="voiceshield_report_{incident_id[:8]}.pdf"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate report: %s", e)
        raise HTTPException(status_code=500, detail="Report generation failed")


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get aggregate statistics for the dashboard."""
    if not settings.DATABASE_URL:
        return {"sessions": 0, "incidents": 0, "avg_score": 0}
    try:
        async with get_session() as db:
            sessions = await db.execute(text("SELECT COUNT(*) FROM sessions"))
            incidents = await db.execute(text("SELECT COUNT(*) FROM incidents"))
            avg = await db.execute(text("SELECT COALESCE(AVG(peak_score), 0) FROM sessions"))

            s_count = sessions.scalar() or 0
            i_count = incidents.scalar() or 0
            a_score = float(avg.scalar() or 0)

            return {
                "total_sessions": int(s_count),
                "total_incidents": int(i_count),
                "avg_peak_score": round(a_score, 4),
                "timestamp": datetime.now(IST).isoformat(),
            }
    except Exception as e:
        logger.error("Failed to get stats: %s", e)
        return {"total_sessions": 0, "total_incidents": 0, "avg_peak_score": 0}


# ── Serve frontend static files ───────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
