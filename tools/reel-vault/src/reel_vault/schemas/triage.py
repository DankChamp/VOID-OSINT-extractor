"""Validated triage output.

The model returns a small JSON object. ``TriageOutput`` is that object; ``TriageResult``
adds the audit fields the pipeline persists: which method decided, which model ran, what
it cost and whether a human should look at it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import Decision, iso_now


class ContentType(StrEnum):
    TUTORIAL = "tutorial"
    NEWS = "news"
    COMEDY = "comedy"
    STORY = "story"
    OPINION = "opinion"
    RECIPE = "recipe"
    AD = "ad"
    OTHER = "other"


class TriageAction(StrEnum):
    """Work the extraction stage may perform."""

    TRANSCRIPT = "transcript"
    OCR = "ocr"
    KEYFRAMES = "keyframes"
    LINKS = "links"
    DOWNLOAD_MEDIA = "download_media"


class ExtractionField(StrEnum):
    """Knowledge fields the extraction stage may fill."""

    STEPS = "steps"
    TRICKS = "tricks"
    TOOLS = "tools"
    LINKS = "links"
    CLAIMS = "claims"
    QUOTES = "quotes"
    RECIPES = "recipes"
    PEOPLE = "people"
    BOOKS = "books"
    PLACES = "places"
    CODE = "code"


class TriageMethod(StrEnum):
    AI = "ai"
    RULE = "rule"
    FALLBACK = "fallback"
    HEURISTIC = "heuristic"
    MANUAL = "manual"
    UNAVAILABLE = "unavailable"


class CategoryHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TriageOutput(BaseModel):
    """Exactly what the triage model is asked to return."""

    # Extra keys are ignored rather than failing the whole call: models routinely add a
    # helpful-but-unasked field, and one repair round is expensive.
    model_config = ConfigDict(extra="ignore")

    decision: Decision
    value_score: int = Field(ge=0, le=100)
    content_type: ContentType = ContentType.OTHER
    reason: str = Field(min_length=1, max_length=2000)
    actions: list[TriageAction] = Field(default_factory=list)
    fields: list[ExtractionField] = Field(default_factory=list)
    categories: list[CategoryHint] = Field(default_factory=list)
    archive_reason: str | None = None


class TriageResult(TriageOutput):
    """Triage output plus provenance and review state."""

    model_config = ConfigDict(extra="forbid")

    shortcode: str
    method: TriageMethod = TriageMethod.AI
    model: str | None = None
    needs_review: bool = False
    rule_reason: str | None = None
    prompt_version: str = "triage-v1"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    created_at: str = Field(default_factory=iso_now)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


__all__ = [
    "CategoryHint",
    "ContentType",
    "ExtractionField",
    "TriageAction",
    "TriageMethod",
    "TriageOutput",
    "TriageResult",
]
