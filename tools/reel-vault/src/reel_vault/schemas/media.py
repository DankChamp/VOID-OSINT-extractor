"""Media artifacts for one reel."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import iso_now

MediaKind = Literal["video", "audio", "image", "other"]


class MediaFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: MediaKind = "other"
    bytes: int = 0
    sha256: str | None = None


class MediaManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    mode: str
    quality: str
    files: list[MediaFile] = Field(default_factory=list)
    expected_items: int = 1
    verified: bool = False
    bytes: int = 0
    error: str | None = None
    downloaded_at: str = Field(default_factory=iso_now)

    @property
    def ok(self) -> bool:
        return self.verified and self.error is None and bool(self.files)


__all__ = ["MediaFile", "MediaKind", "MediaManifest"]
