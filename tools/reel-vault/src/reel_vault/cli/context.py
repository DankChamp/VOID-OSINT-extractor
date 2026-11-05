"""CLI application context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from reel_vault.config.env_file import load_env_file
from reel_vault.config.loader import ConfigError, load_config
from reel_vault.config.models import AppConfig
from reel_vault.logging_setup import setup_logging
from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths


@dataclass(frozen=True)
class AppContext:
    config: AppConfig
    paths: WorkspacePaths

    def store(self) -> Store:
        return Store(self.paths.database)


def build_context(
    *,
    config_path: Path | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> AppContext:
    load_env_file()
    config = load_config(explicit_path=config_path)
    paths = WorkspacePaths(config.data_dir, config.db_path, config.media_dir).ensure()
    level = "DEBUG" if verbose else config.log_level
    setup_logging(level=level, log_file=paths.logs / "reel-vault.log", quiet=quiet)
    return AppContext(config=config, paths=paths)


def context_from_typer(ctx: typer.Context) -> AppContext:
    data: dict[str, Any] = ctx.obj or {}
    try:
        return build_context(
            config_path=data.get("config"),
            verbose=bool(data.get("verbose")),
            quiet=bool(data.get("quiet")),
        )
    except ConfigError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


__all__ = ["AppContext", "build_context", "context_from_typer"]
