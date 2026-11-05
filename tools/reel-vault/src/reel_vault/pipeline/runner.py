"""Pipeline orchestration (implemented in a later phase)."""

from __future__ import annotations

from dataclasses import dataclass

from reel_vault.cli.context import AppContext


@dataclass(frozen=True)
class RunOptions:
    limit: int | None = None
    dry_run: bool = False
    depth: str = "adaptive"


def run_pipeline(context: AppContext, options: RunOptions) -> int:
    raise NotImplementedError("the pipeline is implemented in phase 3")
