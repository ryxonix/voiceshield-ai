"""VoiceShield AI — optional PostgreSQL (Neon) persistence layer.

Design rules:
- Runs only when DATABASE_URL is set; otherwise every function is a no-op.
- Failure-safe: DB problems must never break the real-time pipeline.
- Stores ONLY scalar scores, metadata and speaker embeddings — never audio.
- Uses pgvector cosine distance for cross-session speaker comparison.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import asyncpg

from config import get_settings

log = logging.getLogger("voiceshield.db")

_pool: asyncpg.Pool | None = None
_init_done = False


def enabled() -> bool:
    return get_settings().database_url is not None


async def init() -> bool:
    """Create pool + schema. Returns True when the DB is live."""
    global _pool, _init_done
    if not enabled():
        log.info("DATABASE_URL not set — running in zero-cloud local mode (in-memory store).")
        return False
    if _pool is not None and _init_done:
        return True
    try:
        dsn = get_settings().database_url
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg does not understand the SQLAlchemy driver scheme
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3, command_timeout=15)
        async with _pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY,
                    call_id TEXT UNIQUE,
                    caller_id TEXT,
                    role TEXT NOT NULL DEFAULT 'adult',
                    compare_enabled BOOLEAN DEFAULT FALSE,
                    started_at TIMESTAMPTZ,
                    ended_at TIMESTAMPTZ,
                    peak_score REAL DEFAULT 0,
                    alert_triggered BOOLEAN DEFAULT FALSE
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_snapshots (
                    id SERIAL PRIMARY KEY,
                    session_id UUID REFERENCES sessions(id),
                    ts TIMESTAMPTZ DEFAULT now(),
                    model_prob REAL, jitter REAL, shimmer REAL,
                    phase_continuity REAL, pitch_stability REAL,
                    xai_risk REAL, cross_session_anomaly REAL,
                    synthetic_score REAL
                )
                """
            )
            dim = get_settings().embedding_dim
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS speaker_embeddings (
                    id UUID PRIMARY KEY,
                    session_id UUID REFERENCES sessions(id),
                    caller_id TEXT,
                    embedding vector({dim}),
                    is_genuine BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS genuine_samples (
                    id UUID PRIMARY KEY,
                    caller_id TEXT NOT NULL,
                    embedding vector({dim}),
                    label TEXT DEFAULT 'genuine',
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id UUID PRIMARY KEY,
                    session_id UUID REFERENCES sessions(id),
                    detected_at TIMESTAMPTZ DEFAULT now(),
                    peak_score REAL,
                    role TEXT,
                    action_taken TEXT,
                    alerts_sent JSONB DEFAULT '{}'::jsonb
                )
                """
            )
        _init_done = True
        log.info("Neon PostgreSQL ready (pgvector enabled).")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Database unavailable (%s) — continuing in local mode.", exc)
        _pool = None
        return False


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def upsert_session(call_id: str, role: str, caller_id: str | None, compare_enabled: bool) -> str | None:
    if not _pool:
        return None
    try:
        sid = str(uuid.uuid4())
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, call_id, role, caller_id, compare_enabled, started_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (call_id) DO UPDATE SET role = EXCLUDED.role
                """,
                uuid.UUID(sid), call_id, role, caller_id, compare_enabled,
            )
            row = await conn.fetchrow("SELECT id FROM sessions WHERE call_id = $1", call_id)
            return str(row["id"]) if row else sid
    except Exception as exc:  # noqa: BLE001
        log.warning("upsert_session failed: %s", exc)
        return None


async def end_session_db(session_id: str | None, peak: float, triggered: bool) -> None:
    if not _pool or not session_id:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET ended_at = now(), peak_score = $2, alert_triggered = $3 WHERE id = $1",
                uuid.UUID(session_id), peak, triggered,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("end_session_db failed: %s", exc)


async def insert_risk_snapshot(session_id: str | None, snap: dict) -> None:
    if not _pool or not session_id:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO risk_snapshots
                (session_id, model_prob, jitter, shimmer, phase_continuity,
                 pitch_stability, xai_risk, cross_session_anomaly, synthetic_score)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                uuid.UUID(session_id), snap.get("model_probability"), snap.get("jitter_pct"),
                snap.get("shimmer_pct"), snap.get("phase_continuity"), snap.get("pitch_stability"),
                snap.get("xai_risk"), snap.get("cross_session_anomaly"), snap.get("synthetic_score"),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("insert_risk_snapshot failed: %s", exc)


async def insert_speaker_embedding(session_id: str | None, caller_id: str | None, vec: list[float], is_genuine: bool) -> None:
    if not _pool:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO speaker_embeddings (id, session_id, caller_id, embedding, is_genuine) "
                "VALUES ($1,$2,$3,$4::vector,$5)",
                uuid.uuid4(), uuid.UUID(session_id) if session_id else None, caller_id, _vec_literal(vec), is_genuine,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("insert_speaker_embedding failed: %s", exc)


async def insert_genuine_sample(caller_id: str, vec: list[float], label: str) -> str | None:
    if not _pool:
        return None
    try:
        gid = uuid.uuid4()
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO genuine_samples (id, caller_id, embedding, label) VALUES ($1,$2,$3::vector,$4)",
                gid, caller_id, _vec_literal(vec), label,
            )
        return str(gid)
    except Exception as exc:  # noqa: BLE001
        log.warning("insert_genuine_sample failed: %s", exc)
        return None


async def nearest_genuine(caller_id: str, vec: list[float], top: int = 5) -> list[dict] | None:
    """Top-N nearest genuine embeddings by cosine distance (<=> operator)."""
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text, caller_id, label,
                       embedding <=> $2::vector AS distance
                FROM genuine_samples
                WHERE caller_id = $1
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                caller_id, _vec_literal(vec), top,
            )
            return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("nearest_genuine failed: %s", exc)
        return None


async def insert_incident(session_id: str | None, call_id: str, peak: float, role: str,
                          action: str, alerts_sent: dict) -> str | None:
    if not _pool:
        return None
    try:
        iid = uuid.uuid4()
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO incidents (id, session_id, detected_at, peak_score, role, action_taken, alerts_sent)
                VALUES ($1,$2,now(),$3,$4,$5,$6::jsonb)
                """,
                iid, uuid.UUID(session_id) if session_id else None, peak, role, action,
                json.dumps(alerts_sent),
            )
        return str(iid)
    except Exception as exc:  # noqa: BLE001
        log.warning("insert_incident failed: %s", exc)
        return None


async def list_incidents(limit: int = 100) -> list[dict]:
    if not _pool:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT i.id::text, i.detected_at, i.peak_score, i.role, i.action_taken, i.alerts_sent,
                       s.call_id AS session_call_id
                FROM incidents i LEFT JOIN sessions s ON s.id = i.session_id
                ORDER BY i.detected_at DESC LIMIT $1
                """,
                limit,
            )
            out = []
            for r in rows:
                d = dict(r)
                d["detected_at"] = d["detected_at"].isoformat() if isinstance(d["detected_at"], datetime) else str(d["detected_at"])
                d["alerts_sent"] = json.loads(d["alerts_sent"]) if isinstance(d["alerts_sent"], str) else (d["alerts_sent"] or {})
                out.append(d)
            return out
    except Exception as exc:  # noqa: BLE001
        log.warning("list_incidents failed: %s", exc)
        return []


async def get_incident(incident_id: str) -> dict | None:
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT i.id::text, i.detected_at, i.peak_score, i.role, i.action_taken, i.alerts_sent,
                       s.call_id AS session_call_id, s.started_at, s.ended_at
                FROM incidents i LEFT JOIN sessions s ON s.id = i.session_id
                WHERE i.id = $1
                """,
                uuid.UUID(incident_id),
            )
            if not row:
                return None
            d = dict(row)
            for k in ("detected_at", "started_at", "ended_at"):
                if isinstance(d.get(k), datetime):
                    d[k] = d[k].isoformat()
            if isinstance(d.get("alerts_sent"), str):
                d["alerts_sent"] = json.loads(d["alerts_sent"])
            # Peak XAI snapshot: worst synthetic_score snapshot of that session
            snap = await conn.fetchrow(
                """
                SELECT model_prob, jitter, shimmer, phase_continuity, pitch_stability,
                       xai_risk, cross_session_anomaly, synthetic_score
                FROM risk_snapshots WHERE session_id = $1
                ORDER BY synthetic_score DESC NULLS LAST LIMIT 1
                """,
                uuid.UUID(incident_id),
            )
            d["peak_snapshot"] = dict(snap) if snap else {}
            return d
    except Exception as exc:  # noqa: BLE001
        log.warning("get_incident failed: %s", exc)
        return None
