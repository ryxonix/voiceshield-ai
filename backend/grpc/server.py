"""gRPC enterprise channel (optional) — mirrors the REST contract for banking integrations.

Enable with ENABLE_GRPC=true (runs on :50051 alongside FastAPI).

Generate stubs with:
    python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/voiceshield.proto

When stubs are absent the server reports unavailability via /api/health instead of crashing.
"""
from __future__ import annotations

import logging
import os

from .config import get_settings

logger = logging.getLogger("voiceshield.grpc")
settings = get_settings()

_STUBS_READY: bool | None = None


def stubs_ready() -> bool:
    global _STUBS_READY
    if _STUBS_READY is None:
        try:
            import backend.grpc.voiceshield_pb2  # noqa: F401
            import backend.grpc.voiceshield_pb2_grpc  # noqa: F401

            _STUBS_READY = True
            _ = os
        except Exception:
            _STUBS_READY = False
    return _STUBS_READY


async def start_grpc() -> None:
    if not stubs_ready():
        logger.warning("gRPC stubs missing — generate them (see module docstring) to enable :50051")
        return
    import grpc
    from concurrent import futures

    import backend.grpc.voiceshield_pb2 as pb
    import backend.grpc.voiceshield_pb2_grpc as pb_grpc

    from .inference import aasist
    from .fusion.risk_engine import fuse, risk_band, recommendation, xai_risk
    from .features.prosodic import extract_prosodic_features
    from .features.spectral import phase_continuity
    from .watermark.verifier import verify_watermark
    from .ws.ring_buffer import RingBuffer

    class VoiceShieldServicer(pb_grpc.VoiceShieldServicer):
        def AnalyzeChunk(self, request, context):
            rb = RingBuffer(window_samples=settings.window_samples, hop_samples=settings.hop_samples)
            events = []
            for win in rb.push(request.pcm_int16):
                x = win.samples
                pros = extract_prosodic_features(x, settings.SAMPLE_RATE)
                pc = phase_continuity(x, settings.SAMPLE_RATE)
                wm = verify_watermark(x, settings.SAMPLE_RATE)
                if wm["hit"]:
                    mprob, xr, score = 0.0, 0.0, 0.0
                else:
                    mprob = aasist.predict(x, settings.SAMPLE_RATE)
                    xr = xai_risk(pc, pros["jitter_pct"], pros["shimmer_pct"])
                    score = fuse(mprob, xr)
                band = risk_band(score, request.role)
                events.append(pb.WindowScore(
                    t_ms=win.t_ms, model_prob=mprob, xai_risk=xr, synthetic_score=score,
                    watermark_hit=wm["hit"], risk_band=band,
                ))
            return pb.AnalyzeReply(call_id=request.call_id, scores=events)

        def AnalyzeFile(self, request, context):
            import base64
            import io

            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(base64.b64decode(request.audio_b64)), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != settings.SAMPLE_RATE:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.SAMPLE_RATE)
                sr = settings.SAMPLE_RATE
            rb = RingBuffer(window_samples=settings.window_samples, hop_samples=settings.hop_samples)
            events = []
            idx = 0
            while (idx + settings.window_samples) <= len(audio):
                x = audio[idx : idx + settings.window_samples]
                pros = extract_prosodic_features(x, sr)
                pc = phase_continuity(x, sr)
                wm = verify_watermark(x, sr)
                if wm["hit"]:
                    mprob, xr, score = 0.0, 0.0, 0.0
                else:
                    mprob = aasist.predict(x, sr)
                    xr = xai_risk(pc, pros["jitter_pct"], pros["shimmer_pct"])
                    score = fuse(mprob, xr)
                events.append(pb.WindowScore(
                    t_ms=int(idx / sr * 1000), model_prob=mprob, xai_risk=xr,
                    synthetic_score=score, watermark_hit=wm["hit"],
                    risk_band=risk_band(score, request.role),
                ))
                idx += settings.hop_samples
            return pb.AnalyzeReply(
                call_id=request.call_id, scores=events,
                recommendation=recommendation(risk_band(max((e.synthetic_score for e in events), default=0.0), request.role), request.role),
            )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_VoiceShieldServicer_to_server(VoiceShieldServicer(), server)
    port = f"[::]:{settings.GRPC_PORT}"
    server.add_insecure_port(port)
    server.start()
    logger.info("gRPC server listening on %s", port)
    server.wait_for_termination()
