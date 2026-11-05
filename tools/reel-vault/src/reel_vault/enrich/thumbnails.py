"""Thumbnail download.

Thumbnails are small, so they are buffered and written with the shared atomic writer.
A thumbnail failure is never fatal: the reel keeps its metadata and is enriched later.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from reel_vault.storage.io import write_bytes_atomic

MAX_THUMBNAIL_BYTES = 10 * 1024 * 1024


class HttpThumbnailFetcher:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def download(self, url: str, destination: Path) -> int:
        with httpx.Client(
            follow_redirects=True,
            timeout=self.timeout_seconds,
            headers={"User-Agent": "reel-vault/0.1 (+personal archive)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.lower().startswith("image/"):
                raise ValueError(f"expected an image, got {content_type}")
            data = response.content
        if not data:
            raise ValueError("thumbnail response was empty")
        if len(data) > MAX_THUMBNAIL_BYTES:
            raise ValueError(f"thumbnail is too large ({len(data)} bytes)")
        return write_bytes_atomic(destination, data)


__all__ = ["MAX_THUMBNAIL_BYTES", "HttpThumbnailFetcher"]
