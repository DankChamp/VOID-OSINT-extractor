"""Extraction pipelines (§5 of Phase 5)."""

from video_intel.extraction.engine import (
    ExtractionEngine,
    ExtractionOutcome,
    ExtractionSettings,
    SegmentResolver,
    records_path,
)
from video_intel.extraction.prompts import (
    PROMPT_VERSION,
    build_prompt,
    build_repair_prompt,
    build_system_prompt,
)

__all__ = [
    "PROMPT_VERSION",
    "ExtractionEngine",
    "ExtractionOutcome",
    "ExtractionSettings",
    "SegmentResolver",
    "build_prompt",
    "build_repair_prompt",
    "build_system_prompt",
    "records_path",
]
