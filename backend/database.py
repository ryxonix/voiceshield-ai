"""Database bootstrap.

Uses Neon PostgreSQL (asyncpg) when DATABASE_URL is set — required for pgvector
cross-session similarity. Falls back to a local SQLite file (aiosqlite) so the
platform runs with zero external configuration; vector similarity then degrades
to in-Python cosine search.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

logger = logging.getLogger("voiceshield.db")
settings = get_settings()


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


if settings.DATABASE_URL:
    ENGINE_URL = _normalize_url(settings.DATABASE_URL)
    IS_POSTGRES = True
else:
    ENGINE_URL = f"sqlite+aiosqlite:///{settings.SQLITE_PATH}"
    IS_POSTGRES = False

engine = create_async_engine(
    ENGINE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"timeout": 10} if not IS_POSTGRES else {},
)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create tables + enable pgvector when on Postgres."""
    from sqlalchemy import text

    from .models import Base

    async with engine.begin() as conn:
        if IS_POSTGRES:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("pgvector extension enabled")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not enable pgvector (%s) — falling back to JSON embeddings", exc)
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready (postgres=%s)", IS_POSTGRES)


async def db_health() -> dict:
    from sqlalchemy import text

    info = {"database": "postgres" if IS_POSTGRES else "sqlite", "ok": False, "pgvector": False}
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            info["ok"] = True
            if IS_POSTGRES:
                res = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'"))
                info["pgvector"] = res.fetchone() is not None
    except Exception as exc:
        info["error"] = str(exc)[:200]
    return info
