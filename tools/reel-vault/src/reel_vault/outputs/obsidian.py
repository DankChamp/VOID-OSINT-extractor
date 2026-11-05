"""Obsidian vault rendering — folder per category, YAML frontmatter, wikilinks."""

from __future__ import annotations

from pathlib import Path

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths


def render_obsidian_vault(
    store: Store,
    paths: WorkspacePaths,
) -> Path:
    """Write an Obsidian vault under ``outputs/obsidian/``."""
    vault_dir = paths.outputs / "obsidian"
    vault_dir.mkdir(parents=True, exist_ok=True)

    reels = store.list_reels(limit=10000)

    for reel in reels:
        shortcode = str(reel["shortcode"])
        classification = store.get_classification(shortcode)
        triage = store.get_triage(shortcode)
        extraction = store.get_extraction(shortcode)

        primary = (classification or {}).get("primary") or "uncategorized"
        category_dir = vault_dir / _safe_dir(primary)
        category_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append("---")
        lines.append(f"shortcode: {shortcode}")
        lines.append(f"url: {reel.get('url') or ''}")
        lines.append(f"author: {reel.get('author') or ''}")
        lines.append(f"saved_at: {(reel.get('saved_at') or '')[:10]}")
        lines.append(f"status: {reel.get('status')}")
        lines.append(f"decision: {(triage or {}).get('decision') or '—'}")
        lines.append(f"score: {(triage or {}).get('value_score') or 0}")
        lines.append(f"category: {primary}")
        if classification and classification.get("secondary"):
            lines.append(f"secondary: {classification['secondary']}")
        if triage and triage.get("content_type"):
            lines.append(f"content_type: {triage['content_type']}")
        lines.append("---")
        lines.append("")

        # Title
        caption = (store.get_metadata(shortcode) or {}).get("caption") or ""
        lines.append(f"# {shortcode}")
        lines.append("")

        if caption:
            lines.append(caption)
            lines.append("")

        if triage and triage.get("reason"):
            lines.append(f"> {triage['reason']}")
            lines.append("")

        # Wikilinks to related reels by category
        related = _related_reels(store, shortcode, primary)
        if related:
            lines.append("## Related")
            lines.append("")
            for other in related[:5]:
                lines.append(f"- [[{other}]]")
            lines.append("")

        # Extracted items
        if extraction and extraction.get("items"):
            lines.append("## Extracted items")
            lines.append("")
            for item in extraction["items"]:
                lines.append(f"- **{item.get('field', '')}**: {item.get('text', '')}")
            lines.append("")

        # Links
        if extraction and extraction.get("links"):
            lines.append("## Links")
            lines.append("")
            for link in extraction["links"]:
                title = link.get("title") or link.get("url")
                lines.append(f"- [{title}]({link.get('url', '')})")
            lines.append("")

        out_path = category_dir / f"{shortcode}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")

    # Index note
    index_lines = ["# reel-vault", "", f"Total reels: {len(reels)}", ""]
    cats: dict[str, int] = {}
    for reel in reels:
        classification = store.get_classification(str(reel["shortcode"]))
        primary = (classification or {}).get("primary") or "uncategorized"
        cats[primary] = cats.get(primary, 0) + 1
    for cat in sorted(cats):
        index_lines.append(f"- [[{cat}]] ({cats[cat]})")
    (vault_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    # Category notes
    for cat in cats:
        cat_lines = [f"# {cat}", "", f"Reels in this category: {cats[cat]}", ""]
        for reel in reels:
            shortcode = str(reel["shortcode"])
            classification = store.get_classification(shortcode)
            if (classification or {}).get("primary") == cat:
                cat_lines.append(f"- [[{shortcode}]]")
        (vault_dir / f"{_safe_dir(cat)}.md").write_text("\n".join(cat_lines), encoding="utf-8")

    return vault_dir


def _safe_dir(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_ ") else "-" for c in name)
    return safe.strip() or "uncategorized"


def _related_reels(store: Store, shortcode: str, primary: str) -> list[str]:
    related: list[str] = []
    reels = store.list_reels(limit=10000)
    for reel in reels:
        other = str(reel["shortcode"])
        if other == shortcode:
            continue
        classification = store.get_classification(other)
        if (classification or {}).get("primary") == primary:
            related.append(other)
    return related
