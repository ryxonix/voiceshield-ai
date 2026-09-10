"""Async non-blocking alert dispatch: Telegram Bot API + Resend email + generic webhook + Twilio SMS.

All channels are free-tier where possible:
 * Telegram Bot API      — fully free
 * Resend                — 3,000 emails/month free (email/alerts@resend.dev sandbox)
 * Twilio SMS            — test credentials / trial; production SMS costs
 * Generic webhook       — self-hosted / enterprise sink
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from ..config import get_settings

logger = logging.getLogger("voiceshield.alerts")
settings = get_settings()


def _fmt(payload: dict) -> str:
    flags = ", ".join(payload.get("triggers", [])) or "synthetic-voice"
    mm = "YES" if payload.get("speaker_mismatch") else "no"
    return (
        f"🚨 VoiceShield AI — {payload.get('severity', 'high').upper()} ALERT\n"
        f"Call {payload.get('call_id')} | role={payload.get('role')} lang={payload.get('language')}\n"
        f"Synthetic score: {payload.get('score', 0):.2f}\n"
        f"Speaker mismatch: {mm}\n"
        f"Triggers: {flags}\n"
        f"Action: {payload.get('recommendation', '')}\n"
        f"At: {payload.get('timestamp_ist', '')}"
    )


async def _telegram(payload: dict) -> Optional[str]:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=settings.ALERT_TIMEOUT_S) as client:
        r = await client.post(
            url,
            json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": _fmt(payload)},
        )
    return "telegram" if r.status_code == 200 else None


async def _email(payload: dict) -> Optional[str]:
    if not settings.RESEND_API_KEY or not settings.ALERT_TO_EMAILS:
        return None
    to = [e.strip() for e in settings.ALERT_TO_EMAILS.split(",") if e.strip()]
    async with httpx.AsyncClient(timeout=settings.ALERT_TIMEOUT_S) as client:
        r = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={
                "from": settings.ALERT_FROM_EMAIL or "voiceshield@resend.dev",
                "to": to,
                "subject": f"[VoiceShield] {payload.get('severity','high').upper()} synthetic-voice alert — call {payload.get('call_id')}",
                "text": _fmt(payload),
            },
        )
    return "email" if r.status_code in (200, 201) else None


async def _webhook(payload: dict) -> Optional[str]:
    if not settings.GENERIC_WEBHOOK_URL:
        return None
    async with httpx.AsyncClient(timeout=settings.ALERT_TIMEOUT_S) as client:
        r = await client.post(settings.GENERIC_WEBHOOK_URL, json=payload)
    return "webhook" if r.status_code < 300 else None


async def _sms(payload: dict) -> Optional[str]:
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN or not settings.TWILIO_FROM_NUMBER:
        return None
    if not settings.TWILIO_TO_NUMBER:
        return None
    try:
        from twilio.rest import Client
    except Exception as exc:
        logger.warning("twilio import failed (install twilio): %s", exc)
        return None
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    try:
        msg = client.messages.create(
            body=_fmt_sms(payload),
            from_=settings.TWILIO_FROM_NUMBER,
            to=settings.TWILIO_TO_NUMBER,
        )
    except Exception as exc:
        logger.warning("twilio send failed: %s", exc)
        return None
    return "sms" if getattr(msg, "sid", None) else None


def _fmt_sms(payload: dict) -> str:
    flags = (payload.get("triggers") or ["synthetic-voice"])
    mm = "YES" if payload.get("speaker_mismatch") else "no"
    return (
        f"[VoiceShield] {payload.get('severity','high').upper()} ALERT — Call {payload.get('call_id')}\n"
        f"Synthetic score {(payload.get('score') or 0):.2f}. Speaker mismatch: {mm}. "
        f"Triggers: {', '.join(flags)}. {payload.get('recommendation','')}"
    )


async def dispatch_alerts(payload: dict) -> list[str]:
    """Fire all configured channels concurrently; return delivered channel names."""
    tasks = [t for t in (_telegram(payload), _email(payload), _sms(payload), _webhook(payload))]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    delivered = []
    for r in results:
        if isinstance(r, str):
            delivered.append(r)
        elif isinstance(r, Exception):
            logger.warning("alert channel error: %s", r)
    if delivered:
        logger.info("alerts delivered via %s", delivered)
    else:
        logger.info("no alert channels configured or all failed (free-tier keys pending)")
    return delivered
