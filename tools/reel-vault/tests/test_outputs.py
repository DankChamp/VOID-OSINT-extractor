"""Tests for Phase 11 output rendering."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.outputs.dashboard import render_dashboard_html
from reel_vault.outputs.index import render_category_index, render_stubs
from reel_vault.outputs.knowledge import render_knowledge_json
from reel_vault.outputs.markdown import render_reel_markdown
from reel_vault.outputs.obsidian import render_obsidian_vault
from reel_vault.outputs.pdf import render_pdf
from reel_vault.outputs.render import SUPPORTED_FORMATS, export_all
from reel_vault.schemas.source import SourceReel
from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths

_STORES: list[Store] = []


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while _STORES:
        _STORES.pop().close()


def _workspace(tmp_path: Path, codes: list[str]) -> tuple[Store, WorkspacePaths, Any]:
    overrides: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "db_path": str(tmp_path / "data" / "state.db"),
        "media_dir": str(tmp_path / "media"),
        "run": {"request_delay_seconds": 0.0},
    }
    config = load_config(overrides=overrides)
    paths = WorkspacePaths(config.data_dir, config.db_path, config.media_dir).ensure()
    store = Store(paths.database)
    _STORES.append(store)
    report = ImportReport(source=str(tmp_path / "export"))
    report.reels = [
        SourceReel(
            shortcode=code,
            url=f"https://www.instagram.com/reel/{code}/",
            source_import=report.imported_at,
        )
        for code in codes
    ]
    store.import_report(report)
    return store, paths, config


def _seed_full(
    store: Store,
    shortcode: str,
    *,
    caption: str = "how to focus while studying",
    decision: str = "extract",
    score: int = 75,
    category: str = "learning & study tricks",
) -> None:
    from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus
    from reel_vault.schemas.extraction import (
        ExtractionItem,
        ExtractionResult,
        Keyframe,
        ResolvedLink,
    )
    from reel_vault.schemas.triage import TriageMethod

    metadata = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        fetch_status=FetchStatus.OK,
        description=caption,
        author="creator",
        duration=30.0,
    )
    store.save_metadata(shortcode, metadata.model_dump(mode="json"), "ok")

    store.save_triage(
        shortcode,
        {
            "shortcode": shortcode,
            "decision": decision,
            "value_score": score,
            "content_type": "tutorial",
            "reason": "useful tutorial",
            "actions": ["transcript", "links"],
            "fields": ["steps"],
            "categories": [],
            "archive_reason": None,
            "method": TriageMethod.AI,
            "model": "test",
            "needs_review": False,
            "prompt_version": "triage-v1",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
    )

    extraction = ExtractionResult(
        shortcode=shortcode,
        decision=decision,
        items=[ExtractionItem(field="steps", text="click the button", source="transcript:0")],
        links=[ResolvedLink(url="https://example.com", title="Example", status="ok")],
        keyframes=[Keyframe(path="keyframes/keyframe-00.jpg", timestamp=5.0)],
        method=TriageMethod.AI,
    )
    store.save_extraction(shortcode, extraction.model_dump(mode="json"))

    store.save_classification(
        shortcode,
        primary_label=category,
        secondary=[],
        confidence=0.9,
        method="ai",
    )


def test_supported_formats() -> None:
    assert "md" in SUPPORTED_FORMATS
    assert "json" in SUPPORTED_FORMATS
    assert "html" in SUPPORTED_FORMATS
    assert "pdf" in SUPPORTED_FORMATS
    assert "obsidian" in SUPPORTED_FORMATS


def test_render_reel_markdown(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_reel_markdown(store, paths, "AAA111")
    assert result is not None
    assert result.is_file()
    content = result.read_text(encoding="utf-8")
    assert "# AAA111" in content
    assert "how to focus while studying" in content
    assert "useful tutorial" in content
    assert "click the button" in content
    assert "https://example.com" in content


def test_render_category_index(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_category_index(store, paths)
    assert result.is_file()
    content = result.read_text(encoding="utf-8")
    assert "# Category index" in content
    assert "learning & study tricks" in content


def test_render_stubs(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111", decision="stub", score=10)

    result = render_stubs(store, paths)
    assert result.is_file()
    content = result.read_text(encoding="utf-8")
    assert "# Stubs" in content


def test_render_knowledge_json(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_knowledge_json(store, paths)
    assert result.is_file()
    import json

    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["total_items"] >= 1
    assert data["items"][0]["shortcode"] == "AAA111"


def test_render_dashboard_html(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_dashboard_html(store, paths)
    assert result.is_file()
    content = result.read_text(encoding="utf-8")
    assert "reel-vault dashboard" in content
    assert "AAA111" in content


def test_render_pdf(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_pdf(store, paths)
    assert result.is_file()
    assert result.stat().st_size > 0


def test_render_obsidian_vault(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    result = render_obsidian_vault(store, paths)
    assert result.is_dir()
    md_files = list(result.rglob("*.md"))
    assert len(md_files) >= 1


def test_export_all(tmp_path: Path) -> None:
    store, paths, _config = _workspace(tmp_path, ["AAA111"])
    _seed_full(store, "AAA111")

    from reel_vault.cli.context import AppContext

    context = AppContext(config=_config, paths=paths)
    code = export_all(context, formats="md,json,html,pdf,obsidian")
    assert code == 0

    outputs_dir = paths.outputs
    assert (outputs_dir / "category-index.md").is_file()
    assert (outputs_dir / "stubs.md").is_file()
    assert (outputs_dir / "knowledge.json").is_file()
    assert (outputs_dir / "dashboard.html").is_file()
    assert (outputs_dir / "reel-vault.pdf").is_file()
    assert (outputs_dir / "obsidian").is_dir()


def test_export_all_invalid_format(tmp_path: Path) -> None:
    _, paths, _config = _workspace(tmp_path, ["AAA111"])

    from reel_vault.cli.context import AppContext

    context = AppContext(config=_config, paths=paths)
    code = export_all(context, formats="md,invalid")
    assert code == 1


__all__ = []
