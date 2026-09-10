"""VoiceShield AI — Async PostgreSQL database with pgvector."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from config import settings

logger = logging.getLogger("voiceshield.db")

_engine = None
_session_factory = None

def _get_engine():
    global _engine, _session_factory
    if _engine is None and settings.DATABASE_URL:
        _engine = create_async_engine(settings.DATABASE_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine

async def init_db() -> None:
    engine = _get_engine()
    if engine is None:
        logger.warning("No DATABASE_URL — skipping DB init")
        return
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                call_id TEXT UNIQUE NOT NULL,
                caller_id TEXT,
                role TEXT NOT NULL DEFAULT 'adult',
                compare_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at TIMESTAMPTZ,
                peak_score FLOAT NOT NULL DEFAULT 0.0,
                alert_triggered BOOLEAN NOT NULL DEFAULT FALSE
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_snapshots (
                id SERIAL PRIMARY KEY,
                session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_prob FLOAT NOT NULL DEFAULT 0.0,
                jitter FLOAT NOT NULL DEFAULT 0.0,
                shimmer FLOAT NOT NULL DEFAULT 0.0,
                phase_continuity FLOAT NOT NULL DEFAULT 0.0,
                pitch_stability FLOAT NOT NULL DEFAULT 0.0,
                xai_risk FLOAT NOT NULL DEFAULT 0.0,
                cross_session_anomaly FLOAT,
                synthetic_score FLOAT NOT NULL DEFAULT 0.0
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS speaker_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
                caller_id TEXT,
                embedding vector(512),
                is_genuine BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS genuine_samples (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                caller_id TEXT NOT NULL,
                embedding vector(512),
                label TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS incidents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
                detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                peak_score FLOAT NOT NULL DEFAULT 0.0,
                role TEXT NOT NULL DEFAULT 'adult',
                action_taken TEXT NOT NULL DEFAULT 'alert',
                alerts_sent JSONB NOT NULL DEFAULT '{}'
            )
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_speaker_emb_cosine
            ON speaker_embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_genuine_emb_cosine
            ON genuine_samples USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        await conn.commit()
    logger.info("Database schema initialized with pgvector indexes")

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized — no DATABASE_URL configured")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
