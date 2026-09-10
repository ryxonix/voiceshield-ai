"""VoiceShield AI — Async alert dispatch (Telegram + Webhooks)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx

from config import settings

logger = logging.getLogger("voiceshield.alerts")

IST = timezone(timedelta(hours=5, minutes=30))


async def dispatch_alerts(
    call_id: str,
    synthetic_score: float,
    role: str,
    peak_score: float,
    action_taken: str,
    session_start: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Dispatch alerts concurrently via all configured channels.
    Never blocks the main pipeline — all errors are swallowed.
    """
    results: Dict[str, Any] = {}
    tasks = []

    # ── Telegram ───────────────────────────────────────────────
    if settings.has_telegram:
        tasks.append(
            _send_telegram(
                call_id, synthetic_score, role, peak_score, action_taken, session_start
            )
        )

    # ── Webhooks ───────────────────────────────────────────────
    for url in settings.webhook_url_list:
        tasks.append(
            _send_webhook(
                url, call_id, synthetic_score, role, peak_score, action_taken, session_start
            )
        )

    if tasks:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes):
            channel = f"task_{i}"
            if isinstance(outcome, Exception):
                results[channel] = {"status": "error", "error": str(outcome)}
                logger.warning("Alert dispatch failed (%s): %s", channel, outcome)
            else:
                results[channel] = outcome

    return results


async def _send_telegram(
    call_id: str,
    synthetic_score: float,
    role: str,
    peak_score: float,
    action_taken: str,
    session_start: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Send alert via Telegram Bot API."""
    now_ist = datetime.now(IST)
    start_str = session_start.astimezone(IST).strftime("%H:%M:%S IST") if session_start else "N/A"

    text = (
        f"🚨 <b>VoiceShield AI Alert</b>\n\n"
        f"📞 <b>Call ID:</b> <code>{call_id}</code>\n"
        f"👤 <b>Role:</b> {role.upper()}\n"
        f"📊 <b>Score:</b> {synthetic_score:.1%} (peak: {peak_score:.1%})\n"
        f"⚡ <b>Action:</b> {action_taken}\n"
        f"🕐 <b>Time:</b> {now_ist.strftime('%d %b %Y, %H:%M:%S IST')}\n"
        f"📞 <b>Started:</b> {start_str}\n\n"
        f"{'🔴 CHILD SHIELD TRIGGERED' if action_taken == 'child_shield' else '🟠 Risk threshold exceeded'}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                },
            )
            resp.raise_for_status()
            return {"channel": "telegram", "status": "sent", "status_code": resp.status_code}
    except Exception as e:
        return {"channel": "telegram", "status": "error", "error": str(e)}


async def _send_webhook(
    url: str,
    call_id: str,
    synthetic_score: float,
    role: str,
    peak_score: float,
    action_taken: str,
    session_start: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Send alert via POST webhook."""
    now_ist = datetime.now(IST)

    payload = {
        "event": "deepfake_detected",
        "platform": "voiceshield_ai",
        "call_id": call_id,
        "role": role,
        "synthetic_score": round(synthetic_score, 4),
        "peak_score": round(peak_score, 4),
        "action_taken": action_taken,
        "timestamp": now_ist.isoformat(),
        "session_started": session_start.isoformat() if session_start else None,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return {"channel": "webhook", "url": url, "status": "sent", "status_code": resp.status_code}
    except Exception as e:
        return {"channel": "webhook", "url": url, "status": "error", "error": str(e)}
