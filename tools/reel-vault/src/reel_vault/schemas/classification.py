"""Classification output for one reel.

A reel receives one optional primary label plus any number of secondary labels, each
with a confidence. Labels come from an editable taxonomy; rules prefilter obvious cases
and the LLM handles the rest. Manual labels in ``review.json`` always override.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import iso_now


class ClassificationMethod(StrEnum):
    RULE = "rule"
    AI = "ai"
    FALLBACK = "fallback"
    MANUAL = "manual"


class CategoryLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    primary: CategoryLabel | None = None
    secondary: list[CategoryLabel] = Field(default_factory=list)
    method: ClassificationMethod = ClassificationMethod.AI
    needs_review: bool = False
    rule_reason: str | None = None
    model: str | None = None
    error: str | None = None
    raw_output: str | None = None
    prompt_version: str = "classify-v1"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: str = Field(default_factory=iso_now)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def all_labels(self) -> list[CategoryLabel]:
        labels: list[CategoryLabel] = []
        if self.primary is not None:
            labels.append(self.primary)
        labels.extend(self.secondary)
        return labels

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.all_labels]


__all__ = [
    "CategoryLabel",
    "ClassificationMethod",
    "ClassificationResult",
]
