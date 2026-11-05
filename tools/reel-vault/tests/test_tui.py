"""Tests for the review TUI data logic."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.review.tui import REVIEW_FILTERS, ReelSummary
from reel_vault.schemas.common import ReelStatus
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


def test_review_filters_are_valid() -> None:
    assert "all" in REVIEW_FILTERS
    assert "needs_review" in REVIEW_FILTERS
    assert "low_value" in REVIEW_FILTERS
    assert "failed" in REVIEW_FILTERS
    assert "archive" in REVIEW_FILTERS


def test_reel_summary_dataclass() -> None:
    reel = ReelSummary(
        shortcode="AAA111",
        score=75,
        category="learning & study tricks",
        status="triaged",
        decision="extract",
        caption="how to focus while studying",
        url="https://www.instagram.com/reel/AAA111/",
    )
    assert reel.shortcode == "AAA111"
    assert reel.score == 75
    assert reel.category == "learning & study tricks"
    assert reel.decision == "extract"


def test_tui_loads_reels_from_store(tmp_path: Path) -> None:
    _, _paths, _config = _workspace(tmp_path, ["AAA111", "BBB222"])
    from reel_vault.cli.context import AppContext
    from reel_vault.review.tui import ReviewScreen

    context = AppContext(config=_config, paths=_paths)
    screen = ReviewScreen(context)
    # Access the store through the screen's context
    with screen.store as s:
        reels = s.list_reels(limit=1000)
        assert len(reels) == 2
        shortcodes = {str(r["shortcode"]) for r in reels}
        assert shortcodes == {"AAA111", "BBB222"}


def test_tui_reel_status_filter(tmp_path: Path) -> None:
    store, _paths, _config = _workspace(tmp_path, ["AAA111"])
    store.set_status("AAA111", ReelStatus.TRIAGE_FAILED)
    with store as s:
        reels = s.list_reels(limit=1000)
        failed = [r for r in reels if r["status"] == "triage_failed"]
        assert len(failed) == 1
        assert str(failed[0]["shortcode"]) == "AAA111"


__all__ = []
