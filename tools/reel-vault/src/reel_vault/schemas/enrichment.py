"""Metadata fetched for one saved post.

The export only knows *that* a post was saved. Enrichment fills in what the post is:
caption, author, duration, thumbnail and the media URLs needed by the download stage.

``fetch_status`` is deliberately separate from ``ReelStatus``: it says why a fetch
succeeded or failed, while the reel status says where the reel is in the pipeline.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import iso_now


class FetchStatus(StrEnum):
    OK = "ok"
    PRIVATE = "private"
    DELETED = "deleted"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    UNSUPPORTED = "unsupported"
    ERROR = "error"

    @property
    def unavailable(self) -> bool:
        """Terminal states: retrying cannot make the post public again."""
        return self in {FetchStatus.PRIVATE, FetchStatus.DELETED, FetchStatus.UNSUPPORTED}


class MediaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    ext: str | None = None
    width: int | None = None
    height: int | None = None
    filesize: int | None = None
    thumbnail: str | None = None
    index: int = 0


class EnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    url: str
    fetch_status: FetchStatus = FetchStatus.OK
    title: str | None = None
    description: str | None = None
    author: str | None = None
    duration: float | None = None
    upload_date: str | None = None
    thumbnail: str | None = None
    media: list[MediaItem] = Field(default_factory=list)
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    extractor: str | None = None
    webpage_url: str | None = None
    is_carousel: bool = False
    item_count: int = 1
    error: str | None = None
    thumbnail_path: str | None = None
    fetched_at: str = Field(default_factory=iso_now)

    @property
    def caption(self) -> str | None:
        return self.description or self.title


__all__ = ["EnrichmentResult", "FetchStatus", "MediaItem"]
