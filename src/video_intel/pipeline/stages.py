"""Canonical stage order (§15).

One place decides what runs before what, what ``--until`` may name, and which stages are
cacheable. Both the ingestion and analysis stage modules read this list.
"""

from __future__ import annotations

from video_intel.schemas.common import StageId

INGESTION_STAGES: tuple[StageId, ...] = (
    StageId.METADATA,
    StageId.RAW_TRANSCRIPT,
    StageId.NORMALIZED_TRANSCRIPT,
    StageId.CHUNKS,
)

ANALYSIS_STAGES: tuple[StageId, ...] = (
    StageId.EXTRACTIONS,
    StageId.KNOWLEDGE_DATASET,
)

OUTPUT_STAGES: tuple[StageId, ...] = (
    StageId.OUTPUT_NOTES,
    StageId.OUTPUT_JSON,
    StageId.OUTPUT_PPT,
    StageId.OUTPUT_SUMMARY,
)

STAGE_ORDER: tuple[StageId, ...] = (*INGESTION_STAGES, *ANALYSIS_STAGES, *OUTPUT_STAGES)


def is_known(stage: StageId) -> bool:
    return stage in STAGE_ORDER


def stages_up_to(stage: StageId | None) -> tuple[StageId, ...]:
    """Stages to run for a run that stops at ``stage`` (all stages when ``None``)."""
    if stage is None:
        return STAGE_ORDER
    if stage not in STAGE_ORDER:
        raise ValueError(
            f"stage {stage.value!r} is not implemented; runnable stages: "
            + ", ".join(item.value for item in STAGE_ORDER)
        )
    return STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]


def index_of(stage: StageId) -> int:
    return STAGE_ORDER.index(stage)


__all__ = [
    "ANALYSIS_STAGES",
    "INGESTION_STAGES",
    "OUTPUT_STAGES",
    "STAGE_ORDER",
    "index_of",
    "is_known",
    "stages_up_to",
]
