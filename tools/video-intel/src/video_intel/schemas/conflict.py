"""Conflict models (§12).

Models may read the same passage differently. Conflicts are recorded with their
original claims and evidence, resolved only when the transcript itself decides,
otherwise left open for review. Majority agreement is never a resolution.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ItemId,
    SegmentId,
    StrictModel,
    VerificationStatus,
    utc_now,
)
from video_intel.schemas.verification import Evidence


class ConflictRecord(StrictModel):
    """Two or more items that cannot both be true of the transcript."""

    schema_version: str = SCHEMA_VERSION
    conflict_id: str
    claim_ids: list[ItemId] = Field(min_length=2)
    description: str = Field(min_length=3)
    status: VerificationStatus = VerificationStatus.NEEDS_REVIEW
    resolution: str | None = Field(
        default=None,
        description="Only set when the transcript itself clearly settles the conflict.",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    segment_ids: list[SegmentId] = Field(default_factory=list)
    rationale: str | None = None
    detected_at: datetime = Field(default_factory=utc_now)

    @field_validator("claim_ids")
    @classmethod
    def _unique_claims(cls, claim_ids: list[ItemId]) -> list[ItemId]:
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("conflict claim_ids must be distinct")
        return claim_ids

    @property
    def resolved(self) -> bool:
        return self.resolution is not None and self.status is VerificationStatus.SUPPORTED


class ConflictSet(StrictModel):
    """All conflicts found for one video."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def unresolved(self) -> list[ConflictRecord]:
        return [conflict for conflict in self.conflicts if not conflict.resolved]
