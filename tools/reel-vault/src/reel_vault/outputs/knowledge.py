"""knowledge.json — versioned knowledge base with provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths

KNOWLEDGE_VERSION = 1


def render_knowledge_json(
    store: Store,
    paths: WorkspacePaths,
) -> Path:
    """Write ``outputs/knowledge.json`` with all extracted items and provenance."""
    out_dir = paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "knowledge.json"

    reels = store.list_reels(limit=10000)

    items: list[dict[str, Any]] = []
    for reel in reels:
        shortcode = str(reel["shortcode"])
        extraction = store.get_extraction(shortcode)
        triage = store.get_triage(shortcode)
        classification = store.get_classification(shortcode)
        transcript = store.get_transcript(shortcode)

        if not extraction or not extraction.get("items"):
            continue

        for item in extraction["items"]:
            items.append(
                {
                    "shortcode": shortcode,
                    "url": reel.get("url"),
                    "field": item.get("field"),
                    "text": item.get("text"),
                    "source": item.get("source"),
                    "decision": (triage or {}).get("decision"),
                    "score": (triage or {}).get("value_score"),
                    "primary_category": (classification or {}).get("primary"),
                    "secondary_categories": (classification or {}).get("secondary") or [],
                    "transcript_id": _segment_id(item.get("source"), transcript),
                    "created_at": extraction.get("created_at"),
                }
            )

    knowledge = {
        "version": KNOWLEDGE_VERSION,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_items": len(items),
        "items": items,
    }

    out_path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _segment_id(source: str | None, transcript: dict[str, Any] | None) -> int | None:
    if not source or not source.startswith("transcript:"):
        return None
    if not transcript:
        return None
    try:
        return int(source.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
