"""Timed transcript for one reel."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import iso_now


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    language: str | None = None
    model: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    text: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    chunk_count: int = 0
    no_audio: bool = False
    error: str | None = None
    created_at: str = Field(default_factory=iso_now)

    @property
    def ok(self) -> bool:
        return self.error is None


__all__ = ["Transcript", "TranscriptSegment"]
