"""VoiceShield AI — I4C-formatted forensic PDF report generator."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    HRFlowable,
)

logger = logging.getLogger("voiceshield.forensics")

IST = timezone(timedelta(hours=5, minutes=30))

# ── Styles ────────────────────────────────────────────────────

_styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "VSTitle",
    parent=_styles["Title"],
    fontSize=18,
    spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
)

HEADING_STYLE = ParagraphStyle(
    "VSHeading",
    parent=_styles["Heading2"],
    fontSize=13,
    spaceBefore=12,
    spaceAfter=6,
    textColor=colors.HexColor("#16213e"),
)

BODY_STYLE = ParagraphStyle(
    "VSBody",
    parent=_styles["BodyText"],
    fontSize=10,
    leading=14,
    spaceAfter=4,
)

SMALL_STYLE = ParagraphStyle(
    "VSSmall",
    parent=_styles["BodyText"],
    fontSize=8,
    leading=10,
    textColor=colors.grey,
)


def generate_forensic_report(
    session_data: Dict[str, Any],
    peak_snapshot: Dict[str, Any],
    cross_session: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a formal I4C-ready forensic incident report as PDF bytes.

    Contents:
    - Header with VoiceShield AI branding
    - Metadata table (Call ID, Role, Duration, Scores, Timestamps)
    - Peak Risk XAI Snapshot table
    - Cross-Session Consistency Result (if available)
    - Legal/Compliant Next Steps for victims
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ── Header ─────────────────────────────────────────────────
    story.append(Paragraph("VoiceShield AI", TITLE_STYLE))
    story.append(Paragraph("Deepfake Detection Incident Report", HEADING_STYLE))
    story.append(Paragraph("Formatted for I4C (Indian Cyber Crime Coordination Centre)", SMALL_STYLE))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e94560")))
    story.append(Spacer(1, 6 * mm))

    # ── Metadata Table ─────────────────────────────────────────
    story.append(Paragraph("1. Incident Metadata", HEADING_STYLE))

    call_id = session_data.get("call_id", "N/A")
    role = session_data.get("role", "adult")
    peak_score = session_data.get("peak_score", 0.0)
    action_taken = session_data.get("action_taken", "alert")
    started = _fmt_ist(session_data.get("started_at"))
    ended = _fmt_ist(session_data.get("ended_at"))
    detected = _fmt_ist(session_data.get("detected_at"))
    duration = session_data.get("duration_seconds", "N/A")

    meta_data = [
        ["Field", "Value"],
        ["Call ID", call_id],
        ["Session Role", role.upper()],
        ["Duration", f"{duration}s" if isinstance(duration, (int, float)) else duration],
        ["Peak Synthetic Score", f"{peak_score:.1%}"],
        ["Action Taken", action_taken.upper()],
        ["Session Started", started],
        ["Session Ended", ended],
        ["Alert Triggered At", detected],
        ["Report Generated", _fmt_ist(datetime.now(IST))],
    ]
    story.append(_make_table(meta_data))
    story.append(Spacer(1, 6 * mm))

    # ── Peak Risk XAI Snapshot ─────────────────────────────────
    story.append(Paragraph("2. Peak Risk XAI Analysis", HEADING_STYLE))

    xai_data = [
        ["Metric", "Value"],
        ["Model Probability", f"{peak_snapshot.get('model_prob', 0.0):.4f}"],
        ["Pitch Jitter", f"{peak_snapshot.get('jitter', 0.0):.2f}%"],
        ["Amplitude Shimmer", f"{peak_snapshot.get('shimmer', 0.0):.2f}%"],
        ["Spectral Phase Continuity", f"{peak_snapshot.get('phase_continuity', 0.0):.4f}"],
        ["Pitch Stability", f"{peak_snapshot.get('pitch_stability', 0.0):.1f}%"],
        ["XAI Risk Score", f"{peak_snapshot.get('xai_risk', 0.0):.4f}"],
        ["Cross-Session Anomaly", f"{peak_snapshot.get('cross_session_anomaly', 'N/A')}"],
        ["Fused Synthetic Score", f"{peak_snapshot.get('synthetic_score', 0.0):.4f}"],
    ]
    story.append(_make_table(xai_data))
    story.append(Spacer(1, 6 * mm))

    # ── Cross-Session Consistency ──────────────────────────────
    if cross_session:
        story.append(Paragraph("3. Cross-Session Consistency", HEADING_STYLE))
        cs_data = [
            ["Metric", "Value"],
            ["Caller ID", cross_session.get("caller_id", "N/A")],
            ["Matched Genuine Samples", str(cross_session.get("matched_count", 0))],
            ["Average Similarity", f"{cross_session.get('avg_similarity', 0.0):.4f}"],
            ["Anomaly Score", f"{cross_session.get('anomaly_score', 0.0):.4f}"],
            ["Anomaly Detected", "YES" if cross_session.get("anomaly_detected", False) else "NO"],
        ]
        story.append(_make_table(cs_data))
        story.append(Spacer(1, 6 * mm))

    # ── Legal / Next Steps ─────────────────────────────────────
    story.append(Paragraph("4. Legal & Compliant Next Steps", HEADING_STYLE))

    steps = [
        "<b>Step 1:</b> File a complaint at <b>https://cybercrime.gov.in</b> under the IT Act, 2000 "
        "(Sections 66D, 66E, 66F) and the Bharatiya Nyaya Sanhita (Sections 318, 319, 320).",
        "<b>Step 2:</b> Contact the <b>I4C Helpline: 1930</b> (toll-free) for immediate assistance.",
        "<b>Step 3:</b> Preserve this PDF report and any call recordings as evidence.",
        "<b>Step 4:</b> Report to your bank's fraud department if financial transactions were initiated.",
        "<b>Step 5:</b> Under the DPDP Act 2023, you have the right to demand data erasure from the platform operator.",
        "<b>Step 6:</b> Consult a cybercrime lawyer for civil remedies under the IT Act.",
    ]
    for step in steps:
        story.append(Paragraph(step, BODY_STYLE))
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e94560")))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "This report was generated by VoiceShield AI — a real-time audio deepfake detection platform. "
            "All analysis is performed on acoustic and spectral properties only; no raw audio is stored or transmitted.",
            SMALL_STYLE,
        )
    )

    doc.build(story)
    return buf.getvalue()


def _make_table(data: list[list[str]]) -> Table:
    """Create a styled table from 2-column data."""
    t = Table(data, colWidths=[6 * cm, 10 * cm])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f0f0f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return t


def _fmt_ist(dt: Optional[datetime]) -> str:
    """Format a datetime in IST."""
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%d %b %Y, %H:%M:%S IST")
