"""Shared CLI context: configuration, logging, paths and state store (§16, §17)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import typer

from video_intel.config.env_file import load_env_file
from video_intel.config.loader import ConfigError, load_config
from video_intel.config.models import AppConfig
from video_intel.logging_setup import SECRETS, setup_logging
from video_intel.providers.registry import ProviderRegistry
from video_intel.storage.db import StateStore
from video_intel.storage.paths import ProjectPaths, project_id_for


@dataclass
class AppContext:
    """Everything a command needs, assembled once per invocation."""

    config: AppConfig
    verbose: bool = False
    quiet: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    _registry: ProviderRegistry | None = None

    def registry(self) -> ProviderRegistry:
        """Provider registry for this run, created on first use."""
        if self._registry is None:
            self._registry = ProviderRegistry(self.config)
        return self._registry

    @property
    def data_dir(self) -> Path:
        return Path(self.config.data_dir).expanduser().resolve()

    @property
    def project_paths(self) -> ProjectPaths:
        return ProjectPaths(self.data_dir, project_id_for(self.config.project_name))

    @property
    def store(self) -> StateStore:
        return StateStore(self.project_paths.database)

    @property
    def log_level(self) -> str:
        if self.verbose:
            return "DEBUG"
        return self.config.log_level


def build_context(
    *,
    config_path: Path | None,
    verbose: bool,
    quiet: bool,
    overrides: dict[str, Any] | None = None,
    log_to_file: bool = False,
) -> AppContext:
    """Load configuration, register secrets for redaction, and configure logging.

    ``log_to_file`` stays off for read-only commands so that ``providers`` or
    ``config show`` never create directories in the current working directory.
    """
    # Provider keys may live in a .env file; real environment variables win.
    load_env_file()
    config = load_config(explicit_path=config_path, overrides=overrides or {})

    SECRETS.clear()
    for provider in config.providers.values():
        key = provider.resolve_api_key()
        SECRETS.add(key.get_secret_value() if key else None)

    log_file: Path | None = None
    if log_to_file:
        log_dir = ProjectPaths(Path(config.data_dir).expanduser().resolve(), "").logs
        log_file = log_dir / "video-intel.log"

    setup_logging(
        level="DEBUG" if verbose else config.log_level,
        log_format=config.log_format,
        log_file=log_file,
        quiet=quiet,
    )
    return AppContext(config=config, verbose=verbose, quiet=quiet, overrides=overrides or {})


def fail_config(exc: ConfigError) -> NoReturn:
    """Render a configuration failure without exposing secrets."""
    typer.secho(f"configuration error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)
