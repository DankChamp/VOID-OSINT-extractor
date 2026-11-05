"""Media protocols and errors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from reel_vault.schemas.enrichment import EnrichmentResult


class DriveUnavailable(RuntimeError):
    """The media root disappeared. The run must stop cleanly and resume later."""


class MediaDownloadError(RuntimeError):
    """One reel's media could not be downloaded or verified."""


@runtime_checkable
class MediaDownloader(Protocol):
    def download(self, result: EnrichmentResult, destination: Path, *, quality: str) -> list[Path]:
        """Download media for ``result`` into ``destination`` and return the files."""
        ...


@runtime_checkable
class MediaVerifier(Protocol):
    def verify(self, path: Path, *, kind: str) -> bool:
        """Return True when ``path`` is a readable, non-empty file of ``kind``."""
        ...


__all__ = ["DriveUnavailable", "MediaDownloadError", "MediaDownloader", "MediaVerifier"]
