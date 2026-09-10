"""VoiceShield AI — WebSocket handler for real-time audio stream processing."""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import text

from config import settings
from database import get_session
from features.prosodic import extract_prosodic
from features.spectral import extract_spectral
from features.speaker_embedding import extract_speaker_embedding
from fusion.risk_engine import compute_xai_risk, fuse_scores
from inference.aasist import predict
from mitigation.dispatcher import evaluate_mitigation, should_generate_forensic_report
from mitigation.alerts import dispatch_alerts
from models import RiskSnapshot
from watermark import verifier as watermark
from cross_session import consistency

from .ring_buffer import RingBuffer, WINDOW_SIZE, HOP_SIZE, SAMPLE_RATE

logger = logging.getLogger("voiceshield.ws")
IST = timezone(timedelta(hours=5, minutes=30))

router = APIRouter()


class SessionState:
    """Per-connection state for one WebSocket session."""

    def __init__(
        self,
        session_db_id: UUID,
        call_id: str,
        role: str,
        caller_id: Optional[str],
        compare_enabled: bool,
    ) -> None:
        self.session_db_id = session_db_id
        self.call_id = call_id
        self.role = role
        self.caller_id = caller_id
        self.compare_enabled = compare_enabled

        self.buffer = RingBuffer()
        self.window_index = 0
        self.peak_score = 0.0
        self.peak_snapshot: Optional[dict] = None
        self.alert_triggered = False
        self.started_at = datetime.now(timezone.utc)
        self.prev_window: Optional[np.ndarray] = None
        self.window_embeddings: list[np.ndarray] = []
        self.total_audio_seconds = 0.0


@router.websocket("/ws/stream/{call_id}")
async def stream_handler(
    websocket: WebSocket,
    call_id: str,
    role: str = Query(default="adult"),
    caller_id: Optional[str] = Query(default=None),
    compare: bool = Query(default=False),
) -> None:
    """
    WebSocket endpoint accepting binary PCM 16kHz mono frames.
    Each 300ms window triggers the full detection pipeline.
    """
    await websocket.accept()
    logger.info("WS connected: call_id=%s role=%s", call_id, role)

    # Create session record in database
    session_db_id = uuid4()
    try:
        async with get_session() as db:
            await db.execute(
                text(
                    """
                    INSERT INTO sessions (id, call_id, caller_id, role, compare_enabled, started_at)
                    VALUES (:id, :call_id, :caller_id, :role, :compare, NOW())
                    """
                ),
                {
                    "id": str(session_db_id),
                    "call_id": call_id,
                    "caller_id": caller_id,
                    "role": role,
                    "compare": compare,
                },
            )
    except Exception as e:
        logger.error("Failed to create session record: %s", e)

    state = SessionState(
        session_db_id=session_db_id,
        call_id=call_id,
        role=role,
        caller_id=caller_id,
        compare_enabled=compare,
    )

    try:
        while True:
            data = await websocket.receive_bytes()

            # Convert PCM bytes to float32 samples
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            state.buffer.push(samples)
            state.total_audio_seconds += len(samples) / SAMPLE_RATE

            # Check if a new window is available
            window = state.buffer.get_window()
            if window is None or len(window) < WINDOW_SIZE:
                continue

            # Process the window
            result = await _process_window(state, window)

            if result is not None:
                # Send risk update to client
                await websocket.send_json(result.to_ws_json())

                # Check mitigation
                mitigation = evaluate_mitigation(
                    result.synthetic_score,
                    state.role,
                    state.peak_score,
                    state.window_index,
                )
                if mitigation is not None and not state.alert_triggered:
                    state.alert_triggered = True
                    await websocket.send_json(mitigation.to_ws_json())

            state.window_index += 1
            state.prev_window = window.copy()

    except WebSocketDisconnect:
        logger.info("WS disconnected: call_id=%s", call_id)
    except Exception as e:
        logger.error("WS error for call_id=%s: %s", call_id, e)
    finally:
        await _on_disconnect(state)


async def _process_window(state: SessionState, window: np.ndarray) -> Optional[RiskSnapshot]:
    """Run the full detection pipeline on one 300ms window."""
    try:
        # ── 1. Watermark check (short-circuits if detected) ─────
        wm = watermark.check_watermark(window)
        if wm.detected:
            snapshot = RiskSnapshot(
                call_id=state.call_id,
                window_index=state.window_index,
                verified_enterprise=True,
                synthetic_score=0.0,
            )
            return snapshot

        # ── 2. ONNX inference ───────────────────────────────────
        model_prob, inference_ms = predict(window)

        # ── 3. Prosodic features ────────────────────────────────
        prosodic = extract_prosodic(window)

        # ── 4. Spectral features ────────────────────────────────
        spectral = extract_spectral(window, state.prev_window)

        # ── 5. Speaker embedding ────────────────────────────────
        if state.compare_enabled:
            embedding = extract_speaker_embedding(window)
            if embedding is not None:
                state.window_embeddings.append(embedding)

        # ── 6. Cross-session consistency ────────────────────────
        cross_anomaly = 0.0
        if (
            state.compare_enabled
            and state.caller_id
            and len(state.window_embeddings) >= 30
            and state.total_audio_seconds >= 30.0
            and len(state.window_embeddings) % 30 == 0
        ):
            # Compute session-level embedding (mean of recent windows)
            recent = np.mean(state.window_embeddings[-30:], axis=0)
            cross_anomaly = await consistency.check_cross_session(
                state.caller_id, recent
            )

        # ── 7. XAI risk ─────────────────────────────────────────
        xai_risk = compute_xai_risk(
            spectral.phase_continuity,
            prosodic.jitter_pct,
            prosodic.shimmer_pct,
        )

        # ── 8. Risk fusion ──────────────────────────────────────
        fused = fuse_scores(model_prob, xai_risk, cross_anomaly)

        # ── 9. Update peak score ────────────────────────────────
        if fused.synthetic_score > state.peak_score:
            state.peak_score = fused.synthetic_score
            state.peak_snapshot = {
                "model_prob": fused.model_prob,
                "jitter": prosodic.jitter_pct,
                "shimmer": prosodic.shimmer_pct,
                "phase_continuity": spectral.phase_continuity,
                "pitch_stability": prosodic.pitch_stability,
                "xai_risk": fused.xai_risk,
                "cross_session_anomaly": cross_anomaly,
                "synthetic_score": fused.synthetic_score,
            }

        snapshot = RiskSnapshot(
            call_id=state.call_id,
            window_index=state.window_index,
            model_prob=fused.model_prob,
            jitter=prosodic.jitter_pct,
            shimmer=prosodic.shimmer_pct,
            phase_continuity=spectral.phase_continuity,
            pitch_stability=prosodic.pitch_stability,
            xai_risk=fused.xai_risk,
            cross_session_anomaly=cross_anomaly if cross_anomaly > 0 else None,
            synthetic_score=fused.synthetic_score,
        )

        return snapshot

    except Exception as e:
        logger.error("Pipeline error for call_id=%s window=%d: %s",
                     state.call_id, state.window_index, e)
        return None


async def _on_disconnect(state: SessionState) -> None:
    """Handle session end: store snapshot, incident, and dispatch alerts if needed."""
    try:
        async with get_session() as db:
            # Update session end
            await db.execute(
                text(
                    """
                    UPDATE sessions
                    SET ended_at = NOW(), peak_score = :peak, alert_triggered = :alert
                    WHERE id = :id
                    """
                ),
                {"id": str(state.session_db_id), "peak": state.peak_score, "alert": state.alert_triggered},
            )

            # Store final risk snapshot if peak exists
            if state.peak_snapshot:
                await db.execute(
                    text(
                        """
                        INSERT INTO risk_snapshots
                        (session_id, model_prob, jitter, shimmer, phase_continuity,
                         pitch_stability, xai_risk, cross_session_anomaly, synthetic_score)
                        VALUES (:sid, :mp, :jit, :shi, :pc, :ps, :xai, :csa, :ss)
                        """
                    ),
                    {
                        "sid": str(state.session_db_id),
                        "mp": state.peak_snapshot["model_prob"],
                        "jit": state.peak_snapshot["jitter"],
                        "shi": state.peak_snapshot["shimmer"],
                        "pc": state.peak_snapshot["phase_continuity"],
                        "ps": state.peak_snapshot["pitch_stability"],
                        "xai": state.peak_snapshot["xai_risk"],
                        "csa": state.peak_snapshot.get("cross_session_anomaly"),
                        "ss": state.peak_snapshot["synthetic_score"],
                    },
                )

            # Create incident if threshold was triggered
            if state.alert_triggered:
                threshold = (
                    settings.CHILD_THRESHOLD if state.role == "child"
                    else settings.ADULT_THRESHOLD
                )
                action = "child_shield" if state.role == "child" else "risk_banner"

                result = await db.execute(
                    text(
                        """
                        INSERT INTO incidents (session_id, peak_score, role, action_taken)
                        VALUES (:sid, :peak, :role, :action)
                        RETURNING id
                        """
                    ),
                    {"sid": str(state.session_db_id), "peak": state.peak_score,
                     "role": state.role, "action": action},
                )
                incident_row = result.fetchone()
                incident_id = str(incident_row[0]) if incident_row else None

                # Dispatch alerts (non-blocking)
                await dispatch_alerts(
                    call_id=state.call_id,
                    synthetic_score=state.peak_score,
                    role=state.role,
                    peak_score=state.peak_score,
                    action_taken=action,
                    session_start=state.started_at,
                )

                # Store final alerts_sent
                if incident_id:
                    await db.execute(
                        text("UPDATE incidents SET alerts_sent = :alerts WHERE id = :id"),
                        {"id": incident_id, "alerts": "{}"},
                    )

        # Store final session-level embedding
        if state.window_embeddings and len(state.window_embeddings) > 0:
            avg_embedding = np.mean(state.window_embeddings, axis=0)
            await consistency.store_embedding(
                state.session_db_id, state.caller_id, avg_embedding, is_genuine=False
            )

    except Exception as e:
        logger.error("Disconnect handler error for call_id=%s: %s", state.call_id, e)
