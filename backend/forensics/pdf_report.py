"""Forensic PDF reports — I4C-format (Indian Cyber Crime Coordination Centre).

Includes metadata table, per-window XAI breakdown, analyst recommendations and
next steps. All timestamps in IST (UTC+5:30) as mandated for Indian filings.
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timedelta, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..config import get_settings

logger = logging.getLogger("voiceshield.forensics")
settings = get_settings()

IST = timezone(timedelta(hours=5, minutes=30))


def ist_now() -> datetime:
    return datetime.now(IST)


def ist_str(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%d-%m-%Y %H:%M:%S IST")


def build_pdf(
    incident: dict,
    session: dict,
    windows: list[dict],
    out_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=18, spaceAfter=4)
    sub = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="VoiceShield Forensic Report",
        author="VoiceShield AI",
    )

    story: list = []
    story.append(Paragraph("VoiceShield AI — Voice Integrity Forensic Report", h1))
    story.append(Paragraph("Format aligned to I4C (Indian Cyber Crime Coordination Centre) filing requirements", sub))
    story.append(Spacer(1, 6))

    # ---- Incident summary ----
    story.append(Paragraph("1. Incident Summary", h2))
    rows = [
        ["Incident ID", incident.get("id", "-")],
        ["Call ID", session.get("call_id", "-")],
        ["Generated At (IST)", ist_str(ist_now())],
        ["Call Start (IST)", ist_str(session.get("started_at")) if session.get("started_at") else "-"],
        ["Role", str(session.get("role", "-"))],
        ["Language", str(session.get("language", "-"))],
        ["Peak Synthetic Score", f"{float(incident.get('score', 0)):.3f}"],
        ["Severity", str(incident.get("severity", "-")).upper()],
        ["Enterprise Watermark Verified", "YES — genuine enterprise caller" if session.get("enterprise_verified") else "No"],
        ["Speaker Mismatch (cross-session)", "YES" if incident.get("speaker_mismatch") else "No"],
        ["Detected Triggers", ", ".join(incident.get("triggers", []) or ["synthetic-voice"])],
        ["Recommended Action", incident.get("recommendation", "Initiate call-back verification via trusted channel")],
    ]
    t = Table(rows, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # ---- Score timeline chart ----
    if windows:
        story.append(Paragraph("2. Synthetic-Score Timeline", h2))
        img = _timeline_chart(windows)
        if img:
            story.append(img)
        story.append(Spacer(1, 8))

    # ---- XAI window table ----
    if windows:
        story.append(Paragraph("3. Per-Window XAI Breakdown", h2))
        head = ["t(ms)", "score", "model", "xai", "jitter%", "shimmer%", "φ_cont", "pitch stab", "WM"]
        data = [head]
        for w in windows[-60:]:
            data.append([
                str(int(w.get("t_ms", 0))),
                f"{float(w.get('synthetic_score', 0)):.2f}",
                f"{float(w.get('model_prob', 0)):.2f}",
                f"{float(w.get('xai_risk', 0)):.2f}",
                f"{float(w.get('jitter_pct', 0)):.3f}",
                f"{float(w.get('shimmer_pct', 0)):.3f}",
                f"{float(w.get('phase_continuity', 0)):.2f}",
                f"{float(w.get('pitch_stability', 0)):.1f}",
                "Y" if w.get("watermark_hit") else "-",
            ])
        tw = Table(data, colWidths=[18 * mm] + [17 * mm] * 8, repeatRows=1)
        tw.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        story.append(tw)

    # ---- Methodology + next steps ----
    story.append(Paragraph("4. Methodology", h2))
    story.append(Paragraph(
        "Scores combine an AASIST-L ONNX acoustic model (graph-attention spectro-temporal artifacts), "
        "prosodic XAI (jitter/shimmer via glottal-cycle analysis), spectral phase-continuity deviation, "
        "enterprise watermark verification (4096-pt FFT, 7.0–7.5 kHz pilot @ 12x noise floor) and "
        "cross-session speaker-consistency checks (pgvector cosine similarity). "
        "Fusion: synthetic_score = clip(0.7·P_model + 0.3·XAI_risk [+ 0.10 speaker-mismatch bonus], 0, 1).",
        styles["Normal"],
    ))

    story.append(Paragraph("5. Next Steps for the Analyst", h2))
    for i, step in enumerate([
        "Preserve the original call recording and this report for I4C submission (helpline 1930 / cybercrime.gov.in).",
        "Verify caller identity via a secondary trusted channel (call-back on registered number, MFA).",
        "Cross-check transaction requests raised during the flagged window and freeze pending approvals.",
        "If speaker mismatch is flagged, attach historical genuine voice samples for expert review.",
        "File within 24 hours of financial fraud for highest recovery probability (RBI/DoT guidance).",
    ], 1):
        story.append(Paragraph(f"{i}. {step}", styles["Normal"]))

    footer = ParagraphStyle("F", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generated by VoiceShield AI v{settings.VERSION} · Report ID {incident.get('id','-')} · "
        f"All times IST (UTC+05:30) · Confidential — for authorized fraud-analysis use only.",
        footer,
    ))

    doc.build(story)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(buf.getvalue())
    return out_path


def _timeline_chart(windows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ys = [float(w.get("synthetic_score", 0)) for w in windows]
        xs = [int(w.get("t_ms", 0)) / 1000.0 for w in windows]
        fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=120)
        ax.plot(xs, ys, color="#0ea5e9", lw=1.5)
        ax.fill_between(xs, ys, color="#0ea5e9", alpha=0.15)
        ax.axhline(0.85, color="#ef4444", ls="--", lw=0.8)
        ax.axhline(0.70, color="#f59e0b", ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("synthetic score")
        ax.set_title("Per-window synthetic likelihood", fontsize=9)
        ax.grid(alpha=0.25)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image(buf, width=170 * mm, height=61 * mm)
    except Exception as exc:
        logger.info("timeline chart skipped: %s", exc)
        return None
