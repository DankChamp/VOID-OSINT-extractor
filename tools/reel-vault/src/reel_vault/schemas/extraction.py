"""Grounded extraction output for one reel.

Every knowledge item carries a citation back to the caption, a transcript segment or an
OCR frame. The extraction service drops anything it cannot ground, so a downstream
document never contains an unsourced claim.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import Decision, iso_now
from reel_vault.schemas.triage import ExtractionField

LinkStatus = Literal["ok", "redirected", "unreachable", "blocked", "invalid", "error", "skipped"]


class ExtractionMethod(StrEnum):
    AI = "ai"
    RULE = "rule"
    ARCHIVE = "archive"
    FALLBACK = "fallback"


class ExtractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: ExtractionField
    text: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=200)


class ResolvedLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str | None = None
    title: str | None = None
    status: LinkStatus = "ok"
    source: str | None = None
    error: str | None = None


class Keyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    timestamp: float | None = None
    source: str | None = None


class OcrLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    frame: str
    confidence: float | None = None


class OcrResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    engine: str = "rapidocr"
    available: bool = False
    frames: int = 0
    lines: list[OcrLine] = Field(default_factory=list)
    text: str = ""
    error: str | None = None
    created_at: str = Field(default_factory=iso_now)

    @property
    def ok(self) -> bool:
        return self.available and self.error is None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortcode: str
    decision: Decision
    summary: str | None = None
    items: list[ExtractionItem] = Field(default_factory=list)
    links: list[ResolvedLink] = Field(default_factory=list)
    keyframes: list[Keyframe] = Field(default_factory=list)
    ocr: OcrResult | None = None
    keyframe_error: str | None = None
    model: str | None = None
    method: ExtractionMethod = ExtractionMethod.RULE
    needs_review: bool = False
    error: str | None = None
    raw_output: str | None = None
    prompt_version: str = "extract-v1"
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


__all__ = [
    "ExtractionItem",
    "ExtractionMethod",
    "ExtractionResult",
    "Keyframe",
    "LinkStatus",
    "OcrLine",
    "OcrResult",
    "ResolvedLink",
]
