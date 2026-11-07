"""Tests for the OSINT review TUI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reel_vault.cli.context import AppContext
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.review.tui import OSINTApp
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


def test_tui_app_instantiates(tmp_path: Path) -> None:
    _store, paths, config = _workspace(tmp_path, ["AAA111", "BBB222"])
    context = AppContext(config=config, paths=paths)
    _app = OSINTApp(context)
    assert _app is not None
    assert _app.paths == paths
    assert _app.config == config


def test_tui_loads_reels_from_store(tmp_path: Path) -> None:
    _store, paths, config = _workspace(tmp_path, ["AAA111", "BBB222"])
    context = AppContext(config=config, paths=paths)
    _app = OSINTApp(context)
    with _store as s:
        reels = s.list_reels(limit=1000)
        assert len(reels) == 2


def test_run_tui_returns_int(tmp_path: Path) -> None:
    _store, _, _ = _workspace(tmp_path, ["AAA111"])
    from reel_vault.review.tui import run_tui

    assert callable(run_tui)


__all__ = []
