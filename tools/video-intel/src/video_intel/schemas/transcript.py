"""Transcript data models (§4, §5).

Three representations are kept separately on purpose:

* ``RawTranscript``      - exactly what was acquired.
* ``NormalizedTranscript`` - cleaned text with source timing preserved.
* ``TranscriptSegment``  - the atomic, timestamped unit everything else cites.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    SegmentId,
    StrictModel,
    TranscriptKind,
    utc_now,
)


class VideoMetadata(StrictModel):
    """Metadata for one source video (§4)."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    url: str
    title: str
    channel: str | None = None
    channel_id: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    upload_date: str | None = None
    language: str | None = None
    available_languages: list[str] = Field(default_factory=list)
    description: str | None = None
    source_detail: dict[str, str] = Field(
        default_factory=dict,
        description="How each field was obtained; unknown fields stay None rather than guessed.",
    )
    retrieved_at: datetime = Field(default_factory=utc_now)


class TranscriptSegment(StrictModel):
    """A single timestamped transcript passage (§5).

    ``segment_id`` is assigned by ingestion and is the citation key used by every
    downstream item. Timestamps are measured source values, never model output.
    """

    segment_id: SegmentId
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str = Field(min_length=1)
    raw_text: str | None = Field(
        default=None, description="Original segment text before normalization."
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"segment {self.segment_id}: end_seconds "
                f"({self.end_seconds}) precedes start_seconds ({self.start_seconds})"
            )
        if not self.text.strip():
            raise ValueError(f"segment {self.segment_id}: text is blank")
        return self

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def original_text(self) -> str:
        return self.raw_text if self.raw_text is not None else self.text


class RawTranscript(StrictModel):
    """The transcript exactly as acquired (§3). Never edited, never dropped."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    language: str | None = None
    detected_language: str | None = None
    kind: TranscriptKind = TranscriptKind.UNKNOWN
    segments: list[TranscriptSegment] = Field(min_length=1)
    source: str = Field(description="Acquisition route, e.g. 'youtube-transcript-api'.")
    source_detail: dict[str, str] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=utc_now)

    @field_validator("segments")
    @classmethod
    def _unique_segment_ids(cls, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ids = [segment.segment_id for segment in segments]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate segment_id values: {duplicates}")
        return segments

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def duration_seconds(self) -> float:
        """End time of the last segment, measured from the source timing."""
        return max((segment.end_seconds for segment in self.segments), default=0.0)

    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


class NormalizationStep(StrictModel):
    """One applied normalization rule, recorded so cleaning stays auditable."""

    name: str
    description: str
    segments_changed: int = Field(default=0, ge=0)


class NormalizedTranscript(StrictModel):
    """Cleaned transcript that keeps source timing (§3).

    Cleaning must not change meaning and must not delete repeated passages: repeated
    explanations are consolidated later, in the knowledge representation only.
    """

    schema_version: str = SCHEMA_VERSION
    video_id: str
    source_transcript_hash: str = Field(description="sha256 of the raw transcript text.")
    segments: list[TranscriptSegment] = Field(min_length=1)
    steps: list[NormalizationStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("segments")
    @classmethod
    def _unique_segment_ids(cls, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ids = [segment.segment_id for segment in segments]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate segment_id values: {duplicates}")
        return segments

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def duration_seconds(self) -> float:
        return max((segment.end_seconds for segment in self.segments), default=0.0)

    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


def hash_transcript(text: str) -> str:
    """Stable content hash used for stage caching and change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
