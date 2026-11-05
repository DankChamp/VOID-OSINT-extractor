"""One saved post, as it arrives from the official export."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import MediaType, ReelStatus


class SourceReel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    url: str
    media_type: MediaType = MediaType.UNKNOWN
    author: str | None = None
    saved_at: datetime | None = None
    collections: list[str] = Field(default_factory=list)
    caption: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    source_import: str
    status: ReelStatus = ReelStatus.NEW


__all__ = ["SourceReel"]
