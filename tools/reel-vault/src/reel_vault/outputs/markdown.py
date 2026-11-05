"""Per-reel Markdown rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths


def render_reel_markdown(
    store: Store,
    paths: WorkspacePaths,
    shortcode: str,
) -> Path | None:
    """Write ``reels/<shortcode>/outputs/reel.md`` and return its path."""
    reel = store.get_reel(shortcode)
    if reel is None:
        return None

    metadata = store.get_metadata(shortcode)
    triage = store.get_triage(shortcode)
    extraction = store.get_extraction(shortcode)
    classification = store.get_classification(shortcode)
    transcript = store.get_transcript(shortcode)

    reel_paths = paths.reel(shortcode)
    out_dir = reel_paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reel.md"

    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f"url: {reel['url']}")
    lines.append(f"author: {reel.get('author') or ''}")
    lines.append(f"saved_date: {(reel.get('saved_at') or '')[:10]}")
    lines.append(f"collections: {json.dumps(reel.get('collections') or [])}")
    lines.append(f"categories: {json.dumps((classification or {}).get('secondary') or [])}")
    lines.append(f"score: {(triage or {}).get('value_score') or 0}")
    lines.append(f"decision: {(triage or {}).get('decision') or '—'}")
    lines.append(f"status: {reel['status']}")
    tags = _tags(triage, classification)
    lines.append(f"tags: {json.dumps(tags)}")
    lines.append("---")
    lines.append("")

    # Title
    caption = (metadata or {}).get("caption") or (metadata or {}).get("description") or ""
    lines.append(f"# {reel['shortcode']}")
    lines.append("")

    if caption:
        lines.append(caption)
        lines.append("")

    # Summary / reason
    if triage and triage.get("reason"):
        lines.append("## Why it was kept")
        lines.append("")
        lines.append(str(triage["reason"]))
        lines.append("")

    if triage and triage.get("archive_reason"):
        lines.append("## Archive reason")
        lines.append("")
        lines.append(str(triage["archive_reason"]))
        lines.append("")

    # Extracted items
    if extraction and extraction.get("items"):
        lines.append("## Extracted items")
        lines.append("")
        for item in extraction["items"]:
            lines.append(
                f"- **{item.get('field', '')}**: {item.get('text', '')}  "
                f"_(source: {item.get('source', '')})_"
            )
        lines.append("")

    # Links
    if extraction and extraction.get("links"):
        lines.append("## Links")
        lines.append("")
        for link in extraction["links"]:
            title = link.get("title") or link.get("url")
            lines.append(f"- [{title}]({link.get('url', '')})")
        lines.append("")

    # Transcript excerpt
    if transcript and transcript.get("text"):
        text = str(transcript["text"])
        excerpt = text[:500] + ("…" if len(text) > 500 else "")
        lines.append("## Transcript excerpt")
        lines.append("")
        lines.append(excerpt)
        lines.append("")

    # Keyframes / screenshots
    if extraction and extraction.get("keyframes"):
        lines.append("## Keyframes")
        lines.append("")
        for kf in extraction["keyframes"]:
            lines.append(f"- ![frame]({kf.get('path', '')})")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _tags(triage: dict[str, Any] | None, classification: dict[str, Any] | None) -> list[str]:
    tags: list[str] = []
    if triage:
        dt = triage.get("decision")
        if dt:
            tags.append(f"decision:{dt}")
        ct = triage.get("content_type")
        if ct:
            tags.append(f"type:{ct}")
        if triage.get("needs_review"):
            tags.append("needs_review")
    if classification:
        primary = classification.get("primary")
        if primary:
            tags.append(f"category:{primary}")
    return tags
