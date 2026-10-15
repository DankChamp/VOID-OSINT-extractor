"""Extraction data models (§8, §9).

Models return the ``ExtractionResponse`` contract; responses that fail validation are
kept verbatim in ``RawExtractionRecord`` and never enter the knowledge dataset.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ChunkId,
    Confidence,
    ItemId,
    ItemType,
    PipelineId,
    ProviderKind,
    SegmentId,
    StageState,
    StrictModel,
    utc_now,
)


class ModelRef(StrictModel):
    """Identifies the model that produced a record (§5)."""

    provider: ProviderKind
    provider_name: str
    model: str


class ItemProvenance(StrictModel):
    """Where an item came from: pipeline, model, chunks, segments (§5)."""

    pipeline_id: PipelineId
    pass_index: int = Field(default=1, ge=1)
    model: ModelRef
    chunk_ids: list[ChunkId] = Field(default_factory=list)
    source_segment_ids: list[SegmentId] = Field(default_factory=list)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    prompt_version: str = "1"

    @model_validator(mode="after")
    def _interval_consistent(self) -> ItemProvenance:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("provenance end_seconds precedes start_seconds")
        return self


class ExtractedItem(StrictModel):
    """One structured extraction. Timestamps come from cited segments only."""

    item_id: ItemId
    type: ItemType
    statement: str = Field(min_length=3)
    details: list[str] = Field(default_factory=list)
    source_segment_ids: list[SegmentId] = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    topic_hint: str | None = None
    provenance: ItemProvenance | None = None

    @property
    def text(self) -> str:
        return " ".join([self.statement, *self.details]).strip()


class ExtractionResponse(StrictModel):
    """The JSON contract a model must satisfy (§9).

    ``extra='forbid'`` is intentional: unexpected fields trigger the repair/retry
    path instead of being silently absorbed into the dataset.
    """

    schema_version: str = SCHEMA_VERSION
    pipeline_id: PipelineId
    items: list[ExtractedItem] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("items")
    @classmethod
    def _unique_item_ids(cls, items: list[ExtractedItem]) -> list[ExtractedItem]:
        ids = [item.item_id for item in items]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate item_id values: {duplicates}")
        return items


class RawExtractionRecord(StrictModel):
    """A raw model response plus the outcome of parsing it (§9, §15)."""

    schema_version: str = SCHEMA_VERSION
    record_id: str
    video_id: str
    pipeline_id: PipelineId
    pass_index: int = Field(default=1, ge=1, description="Extraction pass this record belongs to.")
    model: ModelRef
    chunk_ids: list[ChunkId] = Field(default_factory=list)
    prompt_version: str = "1"
    raw_response: str = Field(description="Verbatim model output, stored unmodified.")
    parsed: ExtractionResponse | None = None
    state: StageState = StageState.PENDING
    attempts: int = Field(default=0, ge=0)
    error: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def valid(self) -> bool:
        return self.state is StageState.COMPLETED and self.parsed is not None
