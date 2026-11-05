"""Output rendering (implemented in a later phase)."""

from __future__ import annotations

from pathlib import Path

from reel_vault.cli.context import AppContext


def export_all(context: AppContext, *, formats: str, out: Path | None = None) -> int:
    raise NotImplementedError("outputs are implemented in phase 11")
