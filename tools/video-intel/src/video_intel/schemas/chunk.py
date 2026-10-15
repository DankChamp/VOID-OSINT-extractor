"""Chunk data models (§6).

Chunking is deterministic for a given configuration and transcript, so a chunk set
can be cached and reused across runs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ChunkId,
    SegmentId,
    StrictModel,
    utc_now,
)


class ChunkingConfig(StrictModel):
    """Chunking parameters (§6). All fields are part of the chunk cache key."""

    max_chars: int = Field(default=6000, gt=100, description="Upper bound on chunk size.")
    overlap_chars: int = Field(default=600, ge=0)
    min_chars: int = Field(default=500, ge=0)
    respect_sentences: bool = True
    estimated_chars_per_token: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be smaller than max_chars")
        return self

    @property
    def estimated_tokens(self) -> int:
        return max(1, self.max_chars // self.estimated_chars_per_token)


class Chunk(StrictModel):
    """A timestamped processing unit that cites the segments it contains."""

    schema_version: str = SCHEMA_VERSION
    chunk_id: ChunkId
    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str = Field(min_length=1)
    segment_ids: list[SegmentId] = Field(min_length=1)
    overlaps_previous: bool = False

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError(f"chunk {self.chunk_id}: end_seconds precedes start_seconds")
        return self

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.text) // 4)


class ChunkSet(StrictModel):
    """The complete, ordered chunk list for one video."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    config: ChunkingConfig
    transcript_hash: str
    chunks: list[Chunk]
    cache_key: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def total_chars(self) -> int:
        return sum(chunk.char_count for chunk in self.chunks)


def chunk_set_hash(config: ChunkingConfig, transcript_hash: str) -> str:
    """Cache key for a chunked transcript: same config + transcript -> same chunks."""
    payload = f"{config.model_dump_json()}|{transcript_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
