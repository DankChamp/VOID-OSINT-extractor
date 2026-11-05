"""Fetcher protocols.

The pipeline depends on these protocols rather than on ``yt-dlp`` or ``httpx`` so that
unit tests can inject fakes and never open a network connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus


class FetchError(RuntimeError):
    """A fetch failed in a way the caller can classify and maybe retry."""

    def __init__(self, status: FetchStatus, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = (
            status in {FetchStatus.RATE_LIMITED, FetchStatus.ERROR}
            if retryable is None
            else retryable
        )


@runtime_checkable
class MetadataFetcher(Protocol):
    def fetch(self, url: str, *, shortcode: str) -> EnrichmentResult:
        """Return metadata for ``url`` without downloading the media."""
        ...


@runtime_checkable
class ThumbnailFetcher(Protocol):
    def download(self, url: str, destination: Path) -> int:
        """Download ``url`` to ``destination`` atomically; return bytes written."""
        ...


__all__ = ["FetchError", "MetadataFetcher", "ThumbnailFetcher"]
