"""Shared primitives: schema versioning, identifiers, enums, timestamp helpers.

Every persisted artifact in this project carries ``schema_version`` so that a later
version can extend a dataset without breaking older projects.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION: Final[str] = "1.0.0"

# Identifier constraints. IDs are stable, human readable, and filesystem safe.
ID_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
ItemId = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
SegmentId = Annotated[str, StringConstraints(pattern=r"^segment_\d{3,}$")]
ChunkId = Annotated[str, StringConstraints(pattern=r"^chunk_\d{4,}$")]
TopicId = Annotated[str, StringConstraints(pattern=r"^topic_[0-9A-Za-z._:-]{1,120}$")]


class StrictModel(BaseModel):
    """Base model for every schema: rejects unknown fields, validates on assignment."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def utc_now() -> datetime:
    """Return the current time as an aware UTC timestamp."""
    return datetime.now(UTC)


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``.

    Timestamps in the spec come from real transcript timing, so this helper formats
    measured values only; it never estimates or invents a time.
    """
    if seconds < 0:
        raise ValueError(f"negative timestamp: {seconds}")
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, max_length: int = 48) -> str:
    """Lowercase, hyphenated slug; empty result falls back to ``unknown``."""
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or "unknown"


class ProviderKind(StrEnum):
    """Supported provider families. Adapters ship in Phase 4."""

    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    GEMINI = "gemini"
    NOTEBOOKLM_HANDOFF = "notebooklm_handoff"


class ExecutionMode(StrEnum):
    """Where transcript data is allowed to be sent (§17)."""

    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"


class PipelineId(StrEnum):
    """Extraction pipeline categories (§8)."""

    CONCEPT = "concept"
    DEFINITION = "definition"
    FACT = "fact"
    EXAMPLE = "example"
    FRAMEWORK = "framework"
    PROCESS = "process"
    COMPARISON = "comparison"
    COVERAGE_AUDIT = "coverage_audit"


class ItemType(StrEnum):
    """Category of a single extracted knowledge item (§5, §9)."""

    CONCEPT = "concept"
    DEFINITION = "definition"
    FACT = "fact"
    EXAMPLE = "example"
    FRAMEWORK = "framework"
    PROCESS = "process"
    COMPARISON = "comparison"
    CLAIM = "claim"
    RELATIONSHIP = "relationship"
    MISSED = "missed"


class VerificationStatus(StrEnum):
    """Evidence verification states (§10)."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    NEEDS_REVIEW = "needs_review"


class TranscriptKind(StrEnum):
    """How the transcript was produced, when the source reports it (§4)."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    UNKNOWN = "unknown"


class StageState(StrEnum):
    """Processing state for a cached pipeline stage (§15, §19)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageId(StrEnum):
    """Cacheable processing stages. Stage identity drives invalidation (§15)."""

    METADATA = "metadata"
    RAW_TRANSCRIPT = "raw_transcript"
    NORMALIZED_TRANSCRIPT = "normalized_transcript"
    CHUNKS = "chunks"
    EXTRACTIONS = "extractions"
    NORMALIZE_EXTRACTIONS = "normalize_extractions"
    DEDUPLICATE = "deduplicate"
    VERIFY = "verify"
    CONFLICTS = "conflicts"
    KNOWLEDGE_DATASET = "knowledge_dataset"
    OUTPUT_NOTES = "output_notes"
    OUTPUT_JSON = "output_json"
    OUTPUT_PPT = "output_ppt"
    OUTPUT_SUMMARY = "output_summary"


class OutputMode(StrEnum):
    """Analysis modes exposed by the CLI (§16)."""

    DEEP = "deep"
    PPT = "ppt"
    QUICK = "quick"


class Confidence(StrEnum):
    """Confidence label used where a numeric score is not meaningful."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


NON_NEG_SECONDS = Field(ge=0.0, description="Seconds from the start of the video.")


NOTEBOOKLM_HANDOFF = "notebooklm_handoff"


def timestamp_str(seconds: float) -> str:
    """Alias kept for readability at call sites."""
    return format_timestamp(seconds)
