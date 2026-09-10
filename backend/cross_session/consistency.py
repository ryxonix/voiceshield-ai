"""Cross-session speaker consistency via pgvector cosine similarity.

Compares the current window's speaker embedding against stored genuine
references for the same speaker label. When the best similarity falls below
CS_SIMILARITY_FLOOR with enough references, an identity anomaly is raised.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import get_settings

logger = logging.getLogger("voiceshield.cross_session")
settings = get_settings()


async def compare_to_references(
    session,
    label: str,
    embedding: np.ndarray,
    language: str = "en",
) -> dict:
    """Return {mismatch, best_similarity, n_refs}.

    Postgres path uses pgvector cosine distance ORDER BY for speed.
    SQLite path loads JSON embeddings and computes cosine in Python.
    """
    from ..database import IS_POSTGRES
    from ..models import SpeakerRef

    out = {"mismatch": False, "best_similarity": None, "n_refs": 0}
    vec = embedding.astype(float).tolist()

    try:
        if IS_POSTGRES:
            from pgvector.sqlalchemy import Vector  # noqa: F401

            from sqlalchemy import select, text

            rows = (
                await session.execute(
                    text(
                        """
                        SELECT label, 1 - (embedding <=> :v) AS sim
                        FROM speaker_refs
                        WHERE label = :label
                        ORDER BY embedding <=> :v
                        LIMIT 5
                        """
                    ),
                    {"v": "[" + ",".join(f"{x:.6f}" for x in vec) + "]", "label": label},
                )
            ).fetchall()
            sims = [float(r.sim) for r in rows]
            out["n_refs"] = len(sims)
        else:
            from sqlalchemy import select

            res = await session.execute(
                select(SpeakerRef).where(SpeakerRef.label == label).limit(200)
            )
            refs = res.scalars().all()
            sims = []
            q = np.asarray(vec, dtype=np.float32)
            for r in refs:
                try:
                    v = np.asarray(r.embedding.get("v", []), dtype=np.float32)
                    if v.size == q.size:
                        denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
                        sims.append(float(np.dot(q, v) / denom))
                except Exception:
                    continue
            sims.sort(reverse=True)
            sims = sims[:5]
            out["n_refs"] = len(sims)

        if sims:
            best = max(sims)
            out["best_similarity"] = round(best, 4)
            if out["n_refs"] >= settings.CS_MIN_REFERENCES and best < settings.CS_SIMILARITY_FLOOR:
                out["mismatch"] = True
    except Exception as exc:
        logger.warning("cross-session comparison failed: %s", exc)
    return out


async def store_reference(
    session,
    label: str,
    embedding: np.ndarray,
    language: str = "en",
) -> None:
    """Persist a genuine speaker reference (call /api/speakers/register first)."""
    from ..database import IS_POSTGRES
    from ..models import SpeakerRef

    vec = embedding.astype(float).tolist()
    if IS_POSTGRES:
        try:
            from sqlalchemy import text

            await session.execute(
                text(
                    "INSERT INTO speaker_refs (label, language, embedding, frame_count) "
                    "VALUES (:label, :lang, :v, 1)"
                ),
                {
                    "label": label,
                    "lang": language,
                    "v": "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                },
            )
            await session.commit()
            return
        except Exception as exc:
            logger.warning("pgvector insert failed (%s) — storing JSON", exc)
    from ..models import SpeakerRef as SR

    session.add(SR(label=label, language=language, embedding={"v": vec}))
    await session.commit()
