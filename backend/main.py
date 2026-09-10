"""FastAPI application — VoiceShield AI.

REST + WebSocket streaming + static dashboard serving.
gRPC optional (ENABLE_GRPC=true) on :50051 for enterprise/banking integrations.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .config import get_settings
from .database import db_health, get_session, init_db
from .fusion.risk_engine import fuse, risk_band, recommendation, xai_risk
from .models import (
    AnalysisEvent,
    CallSession,
    Incident,
    IncidentOut,
    MitigationEvent,
    SessionOut,
    WindowAnalysis,
    WindowOut,
)
from .ws.ring_buffer import RingBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("voiceshield")
settings = get_settings()

# Optional Sentry (free tier)
if settings.SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            integrations=[StarletteIntegration(), FastApiIntegration()],
        )
        logger.info("Sentry initialized")
    except Exception as exc:
        logger.warning("Sentry init failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("VoiceShield AI v%s ready (postgres=%s)", settings.VERSION, settings.DATABASE_URL is not None)
    if settings.ENABLE_GRPC:
        from .grpc.server import start_grpc

        asyncio.create_task(start_grpc())
    yield


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------ REST API ------------------------------------


@app.get("/api/health")
async def health():
    h = await db_health()
    return {
        "status": "ok" if h["ok"] else "degraded",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        **h,
    }


@app.get("/api/sessions", response_model=list[SessionOut])
async def list_sessions(limit: int = Query(50, le=200)):
    async for db in get_session():
        res = await db.execute(
            select(CallSession).order_by(CallSession.started_at.desc()).limit(limit)
        )
        return res.scalars().all()
    return []


@app.get("/api/sessions/{sid}", response_model=SessionOut)
async def get_session_api(sid: str):
    async for db in get_session():
        obj = await db.get(CallSession, sid)
        if not obj:
            raise HTTPException(404, "session not found")
        return obj
    raise HTTPException(404)


@app.get("/api/sessions/{sid}/windows", response_model=list[WindowOut])
async def get_windows(sid: str, limit: int = Query(500, le=2000)):
    async for db in get_session():
        res = await db.execute(
            select(WindowAnalysis)
            .where(WindowAnalysis.session_id == sid)
            .order_by(WindowAnalysis.t_ms.asc())
            .limit(limit)
        )
        return res.scalars().all()
    return []


@app.get("/api/incidents", response_model=list[IncidentOut])
async def list_incidents(limit: int = Query(100, le=500)):
    async for db in get_session():
        res = await db.execute(
            select(Incident).order_by(Incident.created_at.desc()).limit(limit)
        )
        return res.scalars().all()
    return []


@app.get("/api/incidents/{iid}", response_model=IncidentOut)
async def get_incident(iid: str):
    async for db in get_session():
        obj = await db.get(Incident, iid)
        if not obj:
            raise HTTPException(404, "incident not found")
        return obj
    raise HTTPException(404)


@app.post("/api/incidents/{iid}/ack")
async def ack_incident(iid: str):
    async for db in get_session():
        obj = await db.get(Incident, iid)
        if not obj:
            raise HTTPException(404, "incident not found")
        obj.acknowledged = True
        await db.commit()
        return {"ok": True, "id": iid}
    raise HTTPException(404)


@app.get("/api/incidents/{iid}/report")
async def download_report(iid: str):
    async for db in get_session():
        inc = await db.get(Incident, iid)
        if not inc:
            raise HTTPException(404, "incident not found")
        if not inc.report_path:
            from .forensics.pdf_report import build_pdf

            sess = await db.get(CallSession, inc.session_id)
            wins = (
                await db.execute(
                    select(WindowAnalysis)
                    .where(WindowAnalysis.session_id == inc.session_id)
                    .order_by(WindowAnalysis.t_ms.asc())
                )
            ).scalars().all()
            w_dicts = [
                {
                    "t_ms": w.t_ms,
                    "synthetic_score": w.synthetic_score,
                    "model_prob": w.model_prob,
                    "xai_risk": w.xai_risk,
                    "jitter_pct": w.jitter_pct,
                    "shimmer_pct": w.shimmer_pct,
                    "phase_continuity": w.phase_continuity,
                    "pitch_stability": w.pitch_stability,
                    "watermark_hit": w.watermark_hit,
                }
                for w in wins
            ]
            s_dict = {
                "call_id": sess.call_id,
                "role": sess.role,
                "language": sess.language,
                "started_at": sess.started_at,
                "enterprise_verified": sess.enterprise_verified,
            }
            i_dict = {
                "id": inc.id,
                "score": inc.score,
                "severity": inc.severity,
                "triggers": inc.triggers or [],
                "speaker_mismatch": inc.speaker_mismatch,
                "recommendation": recommendation(risk_band(inc.score, inc.role), inc.role),
            }
            path = build_pdf(i_dict, s_dict, w_dicts, f"reports/incident_{inc.id}.pdf")
            inc.report_path = path
            await db.commit()
        return FileResponse(
            inc.report_path,
            media_type="application/pdf",
            filename=f"voiceshield_incident_{inc.id}.pdf",
        )
    raise HTTPException(404)


# --------------------------- Speaker references ------------------------------


@app.post("/api/speakers/register")
async def register_speaker(label: str = Query(...), language: str = Query("en"), file: UploadFile = File(...)):
    """Upload a genuine voice sample (wav/mp3) to enroll a trusted speaker."""
    import numpy as np

    from .features.speaker_embedding import extract_speaker_embedding

    data = await file.read()
    try:
        import io as _io

        import soundfile as sf

        audio, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != settings.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.SAMPLE_RATE)
            sr = settings.SAMPLE_RATE
    except Exception as exc:
        raise HTTPException(400, f"Unsupported audio: {exc}")

    emb = extract_speaker_embedding(audio, sr, dim=settings.EMBED_DIM)
    async for db in get_session():
        from .cross_session.consistency import store_reference

        await store_reference(db, label, emb, language)
    return {"ok": True, "label": label, "dim": int(emb.shape[0])}


# ------------------------- Standalone analysis API ---------------------------


@app.post("/api/analyze")
async def analyze_file(role: str = Query("adult"), language: str = Query("en"), file: UploadFile = File(...)):
    """One-shot file analysis: returns per-window scores + fused verdict (REST/SDK path)."""
    import numpy as np

    from .features.prosodic import extract_prosodic_features        from .features.spectral import noise_floor_dropouts, phase_continuity

    from .features.speaker_embedding import extract_speaker_embedding
    from .inference import aasist
    from .watermark.verifier import verify_watermark

    data = await file.read()
    try:
        import io as _io

        import soundfile as sf

        audio, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != settings.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.SAMPLE_RATE)
            sr = settings.SAMPLE_RATE
    except Exception as exc:
        raise HTTPException(400, f"Unsupported audio: {exc}")

    rb = RingBuffer(window_samples=settings.window_samples, hop_samples=settings.hop_samples)
    results = []
    idx = 0
    while (idx + settings.window_samples) <= len(audio):
        win = audio[idx : idx + settings.window_samples]
        t_ms = int(idx / sr * 1000)
        pros = extract_prosodic_features(win, sr)
        pc = phase_continuity(win, sr)
        wm = verify_watermark(win, sr)
        if wm["hit"]:
            score, mprob, verdict = 0.0, 0.0, "verified_enterprise"
        else:
            mprob = aasist.predict(win, sr)
            xr = xai_risk(pc, pros["jitter_pct"], pros["shimmer_pct"])
            score = fuse(mprob, xr)
            band = risk_band(score, role)
            verdict = "synthetic" if band in ("high", "critical") else "suspicious" if band == "medium" else "benign"
        results.append({
            "t_ms": t_ms, "model_prob": round(mprob, 4), "xai_risk": round(xr, 4) if not wm["hit"] else 0.0,
            "synthetic_score": round(score, 4), "jitter_pct": round(pros["jitter_pct"], 4),
            "shimmer_pct": round(pros["shimmer_pct"], 4), "phase_continuity": round(pc, 4),
            "pitch_stability": round(pros["pitch_stability"], 2),
            "watermark_hit": wm["hit"], "verdict": verdict,
        })
        idx += settings.hop_samples

    peak = max((r["synthetic_score"] for r in results), default=0.0)
    return {
        "role": role, "language": language, "windows": results,
        "peak_score": round(peak, 4),
        "risk_band": risk_band(peak, role),
        "recommendation": recommendation(risk_band(peak, role), role),
        "windows_analyzed": len(results),
    }


# ------------------------------ WebSocket ------------------------------------


@app.websocket("/ws/stream/{call_id}")
async def ws_stream(ws: WebSocket, call_id: str, role: str = "adult", language: str = "en", speaker: str = ""):
    """Real-time PCM (int16 LE, 16 kHz, mono) streaming analysis.

    Query params: role=adult|child, language=en|hi|kn, speaker=<label for cross-session check>.
    Emits JSON events per window; accepts {"type":"end"} text frames to close a session.
    """
    await ws.accept()
    rb = RingBuffer(window_samples=settings.window_samples, hop_samples=settings.hop_samples)
    from .features.prosodic import extract_prosodic_features
    from .features.spectral import noise_floor_dropouts, phase_continuity
    from .features.speaker_embedding import extract_speaker_embedding
    from .inference import aasist
    from .watermark.verifier import verify_watermark

    session_id: str | None = None
    max_score = 0.0
    score_sum = 0.0
    n_windows = 0
    enterprise_verified = False
    last_spk_check = -10_000.0

    try:
        async for db in get_session():
            sess = CallSession(call_id=call_id, role=role, language=language)
            db.add(sess)
            await db.commit()
            await db.refresh(sess)
            session_id = sess.id
            break

        await ws.send_json({
            "type": "session_start", "session_id": session_id, "call_id": call_id,
            "role": role, "language": language,
            "window_ms": settings.WINDOW_MS, "hop_ms": settings.HOP_MS,
        })

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (text := msg.get("text")) is not None:
                import json as _json

                data = _json.loads(text or "{}")
                if data.get("type") == "end":
                    break
                continue
            data = msg.get("bytes")
            if not data:
                continue
            if len(data) > settings.MAX_FRAME_BYTES:
                continue

            for win in rb.push(data):
                t0 = time.perf_counter()
                t_ms = win.t_ms
                x = win.samples
                pros = extract_prosodic_features(x, settings.SAMPLE_RATE)
                pc = phase_continuity(x, settings.SAMPLE_RATE)
                nf_drops = noise_floor_dropouts(x, settings.SAMPLE_RATE)
                wm = verify_watermark(x, settings.SAMPLE_RATE)

                speaker_mismatch = None
                if wm["hit"]:
                    mprob, xr, score = 0.0, 0.0, 0.0
                    verdict = "verified_enterprise"
                    enterprise_verified = True
                else:
                    mprob = aasist.predict(x, settings.SAMPLE_RATE)
                    xr = xai_risk(pc, pros["jitter_pct"], pros["shimmer_pct"])
                    score = fuse(mprob, xr)
                    band = risk_band(score, role)
                    verdict = "synthetic" if band in ("high", "critical") else "suspicious" if band == "medium" else "benign"

                    # cross-session check every ~2.5 s
                    if speaker and (t_ms - last_spk_check) >= 2500:
                        last_spk_check = t_ms
                        try:
                            emb = extract_speaker_embedding(x, settings.SAMPLE_RATE, dim=settings.EMBED_DIM)
                            async for db in get_session():
                                from .cross_session.consistency import compare_to_references

                                res = await compare_to_references(db, speaker, emb, language)
                                speaker_mismatch = res.get("mismatch")
                                if speaker_mismatch:
                                    score = fuse(mprob, xr, speaker_mismatch=True)
                                break
                        except Exception as exc:
                            logger.debug("speaker check skipped: %s", exc)

                    if speaker_mismatch:
                        verdict = "synthetic+speaker_mismatch"

                latency_ms = (time.perf_counter() - t0) * 1000.0
                band = risk_band(score, role)
                n_windows += 1
                max_score = max(max_score, score)
                score_sum += score

                # persist window + update session aggregates (best-effort)
                try:
                    async for db in get_session():
                        db.add(WindowAnalysis(
                            session_id=session_id, t_ms=t_ms,
                            model_prob=mprob, xai_risk=xr, synthetic_score=score,
                            jitter_pct=pros["jitter_pct"], shimmer_pct=pros["shimmer_pct"],
                            phase_continuity=pc, pitch_stability=pros["pitch_stability"],
                            watermark_hit=wm["hit"], verdict=verdict,
                            features={"snr": wm.get("snr", 0.0), "f0": pros.get("f0_mean", 0.0)},
                        ))
                        sess_row = await db.get(CallSession, session_id)
                        if sess_row:
                            sess_row.window_count = n_windows
                            sess_row.max_score = round(max_score, 4)
                            sess_row.avg_score = round(score_sum / n_windows, 4)
                            sess_row.last_score = round(score, 4)
                            sess_row.enterprise_verified = enterprise_verified
                        await db.commit()
                        break
                except Exception as exc:
                    logger.debug("window persist failed: %s", exc)

                event = AnalysisEvent(
                    call_id=call_id, t_ms=t_ms,
                    synthetic_score=round(score, 4), model_prob=round(mprob, 4),
                    xai_risk=round(xr, 4),
                    prosody={
                        "jitter_pct": round(pros["jitter_pct"], 4),
                        "shimmer_pct": round(pros["shimmer_pct"], 4),
                        "phase_continuity": round(pc, 4),
                        "pitch_stability": round(pros["pitch_stability"], 2),
                        "noise_floor_dropouts": nf_drops,
                        "watermark_snr": wm.get("snr", 0.0),
                    },
                    watermark_hit=wm["hit"], verdict=verdict, risk_band=band,
                    recommendation=recommendation(band, role), session_id=session_id or "",
                    speaker_mismatch=speaker_mismatch,
                    latency_ms=round(latency_ms, 2),
                )
                await ws.send_json(event.model_dump())

                # client-side mitigation directive (spec §4.4): child shield mutes + overlays
                thr = settings.THRESHOLD_CHILD if role == "child" else settings.THRESHOLD_ADULT
                if score >= thr:
                    if role == "child":
                        mitigation = MitigationEvent(
                            action="child_shield", severity="critical", overlay=True, mute=True,
                            score=round(score, 4), threshold=thr, role=role,
                            message="Potential AI-cloned voice detected. Audio paused — contact a trusted guardian before continuing this call.",
                        )
                    else:
                        mitigation = MitigationEvent(
                            action="risk_banner", severity="high", overlay=False, mute=False,
                            score=round(score, 4), threshold=thr, role=role,
                            message="High synthetic-voice risk. Verify the caller via call-back or MFA before any sensitive action.",
                        )
                    await ws.send_json(mitigation.model_dump())

                # mitigation
                try:
                    async for db in get_session():
                        from .mitigation.dispatcher import maybe_dispatch

                        trig = []
                        if score >= settings.THRESHOLD_ADULT:
                            trig.append("high_synthetic_score")
                        if wm["hit"]:
                            trig.append("enterprise_watermark")
                        if speaker_mismatch:
                            trig.append("speaker_mismatch")
                        await maybe_dispatch(db, session_id, call_id, role, language, score, trig, bool(speaker_mismatch))
                        break
                except Exception as exc:
                    logger.debug("dispatch check failed: %s", exc)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("ws error: %s", exc)
    finally:
        try:
            async for db in get_session():
                sess = await db.get(CallSession, session_id)
                if sess:
                    sess.status = "closed"
                    sess.ended_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                    sess.window_count = n_windows
                    sess.max_score = round(max_score, 4)
                    sess.avg_score = round(score_sum / n_windows, 4) if n_windows else 0.0
                    sess.enterprise_verified = enterprise_verified
                    await db.commit()
                break
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


# ------------------------------ Static dashboard -----------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
