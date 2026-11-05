"""PDF rendering via ReportLab."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.lib.styles import (  # type: ignore[import-untyped]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    BaseDocTemplate,
    NextPageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths


def render_pdf(
    store: Store,
    paths: WorkspacePaths,
    per_reel: bool = False,
) -> Path:
    """Write ``outputs/reel-vault.pdf`` with cover, TOC and one section per reel."""
    out_dir = paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reel-vault.pdf"

    reels = store.list_reels(limit=10000)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=24, spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor="#555555"
    )

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="reel-vault archive",
        author="reel-vault",
    )

    # Page templates: cover (no frame content needed) and body
    from reportlab.platypus import Frame, PageTemplate

    cover_frame = Frame(
        0.8 * inch, 0.8 * inch, letter[0] - 1.6 * inch, letter[1] - 1.6 * inch, id="cover"
    )
    body_frame = Frame(
        0.8 * inch, 0.8 * inch, letter[0] - 1.6 * inch, letter[1] - 1.6 * inch, id="body"
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[cover_frame]),
            PageTemplate(id="body", frames=[body_frame]),
        ]
    )

    story = []

    # Cover
    story.append(Paragraph("reel-vault", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Personal Instagram archive", h2))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total reels: {len(reels)}", body))
    story.append(Paragraph(f"Generated: {_now()}", body))
    story.append(NextPageTemplate("body"))
    from reportlab.platypus import PageBreak

    story.append(PageBreak())

    # TOC
    story.append(Paragraph("Contents", h1))
    story.append(Spacer(1, 12))
    for reel in reels:
        shortcode = str(reel["shortcode"])
        story.append(Paragraph(f"• {shortcode}", body))
    story.append(PageBreak())

    # One section per reel
    for reel in reels:
        shortcode = str(reel["shortcode"])
        metadata = store.get_metadata(shortcode)
        triage = store.get_triage(shortcode)
        extraction = store.get_extraction(shortcode)
        classification = store.get_classification(shortcode)

        story.append(Paragraph(shortcode, ParagraphStyle("ReelTitle", parent=h1, fontSize=14)))
        story.append(
            Paragraph(f"URL: <a href='{reel.get('url', '')}'>{reel.get('url', '')}</a>", small)
        )
        story.append(
            Paragraph(
                f"Status: <b>{reel.get('status')}</b> &nbsp;|&nbsp; "
                f"Decision: <b>{(triage or {}).get('decision') or '—'}</b> &nbsp;|&nbsp; "
                f"Score: <b>{(triage or {}).get('value_score') or 0}</b>",
                small,
            )
        )
        if classification and classification.get("primary"):
            story.append(Paragraph(f"Category: {classification['primary']}", small))
        story.append(Spacer(1, 6))

        caption = (metadata or {}).get("caption") or ""
        if caption:
            story.append(Paragraph("Caption", h2))
            story.append(Paragraph(caption, body))

        if triage and triage.get("reason"):
            story.append(Paragraph("Why it was kept", h2))
            story.append(Paragraph(str(triage["reason"]), body))

        if extraction and extraction.get("items"):
            story.append(Paragraph("Extracted items", h2))
            item_rows = [["Field", "Text", "Source"]]
            for item in extraction["items"][:12]:
                item_rows.append(
                    [
                        str(item.get("field", "")),
                        str(item.get("text", ""))[:100],
                        str(item.get("source", "")),
                    ]
                )
            item_table = Table(item_rows, colWidths=[0.9 * inch, 4.2 * inch, 1.0 * inch])
            item_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
                        ("GRID", (0, 0), (-1, -1), 0.5, "#cccccc"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(item_table)

        story.append(Spacer(1, 12))

    doc.build(story)
    return out_path


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
