"""Evidence verification models (§10).

An extracted item and a transcript-supported fact are different things. Each item is
paired with an explicit verification state so unsupported output is never presented
as something the speaker said.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    Confidence,
    ItemId,
    SegmentId,
    StrictModel,
    VerificationStatus,
    utc_now,
)
from video_intel.schemas.extraction import ExtractedItem, ModelRef


class Evidence(StrictModel):
    """A cited transcript passage and how closely it backs a claim."""

    segment_id: SegmentId
    quote: str = Field(min_length=1)
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _interval(self) -> Evidence:
        if self.end_seconds < self.start_seconds:
            raise ValueError("evidence end_seconds precedes start_seconds")
        return self


class VerificationResult(StrictModel):
    """Verification outcome for exactly one item (§10)."""

    item_id: ItemId
    status: VerificationStatus
    evidence: list[Evidence] = Field(default_factory=list)
    supported_parts: list[str] = Field(default_factory=list)
    unsupported_parts: list[str] = Field(default_factory=list)
    rationale: str | None = None
    verified_by: str = Field(description="'deterministic', a verifier model id, or 'none'.")
    verifier_model: ModelRef | None = None
    verified_at: datetime = Field(default_factory=utc_now)

    @property
    def is_reliable(self) -> bool:
        return self.status is VerificationStatus.SUPPORTED

    @property
    def needs_review(self) -> bool:
        return self.status in {
            VerificationStatus.PARTIALLY_SUPPORTED,
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.NEEDS_REVIEW,
        }


class VerifiedItem(StrictModel):
    """An extraction item paired with its verification record."""

    schema_version: str = SCHEMA_VERSION
    item: ExtractedItem
    verification: VerificationResult
    retained: bool = Field(
        default=True,
        description="False when the item is kept for review but excluded from outputs.",
    )
    review_reason: str | None = None

    @model_validator(mode="after")
    def _ids_match(self) -> VerifiedItem:
        if self.item.item_id != self.verification.item_id:
            raise ValueError(
                f"verification item_id {self.verification.item_id!r} does not match "
                f"item {self.item.item_id!r}"
            )
        return self


class VerificationSummary(StrictModel):
    """Aggregate counts so a dataset can state how complete verification is (§19)."""

    total: int = Field(ge=0)
    supported: int = Field(default=0, ge=0)
    partially_supported: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)
    needs_review: int = Field(default=0, ge=0)
    unverified: int = Field(default=0, ge=0)
    overall_confidence: Confidence = Confidence.MEDIUM

    @property
    def verified_fraction(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.supported + self.partially_supported) / self.total

    @property
    def complete(self) -> bool:
        """True only when every item carries a definitive verification state."""
        return self.unverified == 0 and self.needs_review == 0
