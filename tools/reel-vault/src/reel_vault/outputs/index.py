"""Category index and stubs.md rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths


def render_category_index(
    store: Store,
    paths: WorkspacePaths,
) -> Path:
    """Write ``outputs/category-index.md`` grouped by primary category."""
    out_dir = paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "category-index.md"

    reels = store.list_reels(limit=10000)

    by_category: dict[str, list[dict[str, Any]]] = {}
    uncategorized: list[dict[str, Any]] = []
    for reel in reels:
        classification = store.get_classification(reel["shortcode"])
        primary = (classification or {}).get("primary")
        if primary:
            by_category.setdefault(primary, []).append(reel)
        else:
            uncategorized.append(reel)

    lines: list[str] = []
    lines.append("# Category index")
    lines.append("")
    lines.append(f"Total reels: {len(reels)}")
    lines.append("")

    for category in sorted(by_category):
        items = by_category[category]
        items.sort(key=lambda r: str(r.get("saved_at") or ""), reverse=True)
        lines.append(f"## {category} ({len(items)})")
        lines.append("")
        for reel in items:
            triage = store.get_triage(reel["shortcode"])
            score = (triage or {}).get("value_score") or 0
            decision = (triage or {}).get("decision") or "—"
            saved = (reel.get("saved_at") or "")[:10]
            lines.append(
                f"- [{reel['shortcode']}](#{reel['shortcode']}) — score {score}, "
                f"{decision}, saved {saved}"
            )
        lines.append("")

    if uncategorized:
        lines.append("## Uncategorized")
        lines.append("")
        for reel in uncategorized:
            lines.append(f"- {reel['shortcode']} — {reel['status']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def render_stubs(
    store: Store,
    paths: WorkspacePaths,
) -> Path:
    """Write ``outputs/stubs.md`` with every low-value/unavailable/failed reel."""
    out_dir = paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stubs.md"

    reels = store.list_reels(limit=10000)

    stubby_statuses = {
        "stub",
        "unavailable",
        "triage_failed",
        "extraction_failed",
        "classification_failed",
        "transcript_failed",
        "media_failed",
        "failed",
    }

    lines: list[str] = []
    lines.append("# Stubs")
    lines.append("")
    lines.append("Low-value, unavailable or failed reels kept for metadata only.")
    lines.append("")

    count = 0
    for reel in reels:
        if reel["status"] not in stubby_statuses:
            continue
        triage = store.get_triage(reel["shortcode"])
        reason = (triage or {}).get("reason") or "no reason recorded"
        score = (triage or {}).get("value_score") or 0
        lines.append(f"## {reel['shortcode']}")
        lines.append("")
        lines.append(f"- Status: {reel['status']}")
        lines.append(f"- Score: {score}")
        lines.append(f"- Reason: {reason}")
        url = reel.get("url")
        if url:
            lines.append(f"- URL: {url}")
        lines.append("")
        count += 1

    lines.insert(2, f"Total stubs: {count}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
