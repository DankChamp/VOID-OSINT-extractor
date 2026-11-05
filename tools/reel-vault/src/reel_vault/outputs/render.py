"""Output rendering — exports the dataset to Markdown, JSON, HTML, PDF and Obsidian."""

from __future__ import annotations

from pathlib import Path

from reel_vault.cli.context import AppContext
from reel_vault.outputs.dashboard import render_dashboard_html
from reel_vault.outputs.index import render_category_index, render_stubs
from reel_vault.outputs.knowledge import render_knowledge_json
from reel_vault.outputs.markdown import render_reel_markdown
from reel_vault.outputs.obsidian import render_obsidian_vault
from reel_vault.outputs.pdf import render_pdf
from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths

SUPPORTED_FORMATS = ("md", "json", "html", "pdf", "obsidian")


def export_all(
    context: AppContext,
    *,
    formats: str,
    out: Path | None = None,
) -> int:
    """Render the current dataset to the requested formats.

    ``formats`` is a comma-separated list: md, json, html, pdf, obsidian.
    ``out`` overrides the configured output root.
    """
    requested = [f.strip().lower() for f in formats.split(",") if f.strip()]
    invalid = [f for f in requested if f not in SUPPORTED_FORMATS]
    if invalid:
        from rich.console import Console

        Console().print(f"[red]unknown output format(s): {', '.join(invalid)}[/red]")
        return 1

    data_dir = context.config.data_dir
    db_path = context.config.db_path
    media_dir = context.config.media_dir
    paths = WorkspacePaths(data_dir, db_path, media_dir)

    store = Store(paths.database)
    with store:
        if "md" in requested:
            reels = store.list_reels(limit=10000)
            for reel in reels:
                render_reel_markdown(store, paths, str(reel["shortcode"]))
        if "json" in requested:
            render_knowledge_json(store, paths)
        if "html" in requested:
            render_dashboard_html(store, paths)
        if "pdf" in requested:
            render_pdf(store, paths)
        if "obsidian" in requested:
            render_obsidian_vault(store, paths)
        render_category_index(store, paths)
        render_stubs(store, paths)

    return 0


__all__ = ["SUPPORTED_FORMATS", "export_all"]
