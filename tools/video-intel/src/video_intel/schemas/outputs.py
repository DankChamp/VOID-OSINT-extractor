"""Output models (§14).

Every artifact is generated from the consolidated dataset and records the dataset
version and input hash it came from, so changing an output format never re-triggers
AI work.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

from pydantic import Field, field_validator, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    OutputMode,
    StageId,
    StageState,
    StrictModel,
    utc_now,
)

OUTPUT_STAGES_BY_MODE: Final[dict[OutputMode, frozenset[StageId]]] = {
    OutputMode.DEEP: frozenset(
        {
            StageId.OUTPUT_NOTES,
            StageId.OUTPUT_JSON,
            StageId.OUTPUT_PPT,
            StageId.OUTPUT_SUMMARY,
        }
    ),
    OutputMode.PPT: frozenset({StageId.OUTPUT_PPT, StageId.OUTPUT_JSON}),
    OutputMode.QUICK: frozenset({StageId.OUTPUT_SUMMARY, StageId.OUTPUT_JSON}),
}


class OutputRequest(StrictModel):
    """What the user asked to be generated."""

    mode: OutputMode
    outputs: set[StageId] = Field(default_factory=set)

    @model_validator(mode="after")
    def _fill_outputs(self) -> OutputRequest:
        if not self.outputs:
            self.outputs = set(OUTPUT_STAGES_BY_MODE[self.mode])
        return self


class OutputArtifact(StrictModel):
    """A generated file on disk."""

    schema_version: str = SCHEMA_VERSION
    stage_id: StageId
    filename: str
    path: str
    dataset_hash: str = Field(description="Hash of the dataset the artifact was built from.")
    dataset_version: int = Field(default=1, ge=1)
    bytes_written: int = Field(default=0, ge=0)
    state: StageState = StageState.COMPLETED
    generated_at: datetime = Field(default_factory=utc_now)

    @field_validator("path")
    @classmethod
    def _relative_path(cls, path: str) -> str:
        """Reject absolute paths on any platform, not just POSIX ones.

        A leading-slash test lets ``C:\\data\\knowledge.json`` and
        ``\\\\nas\\share\\knowledge.json`` through. Both spellings are checked so that a
        dataset written on one machine cannot embed a location that is meaningless on
        another.
        """
        if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
            raise ValueError("artifact path must be relative to the video directory")
        return path


class SlideOutline(StrictModel):
    """One slide of the presentation outline (§14 B)."""

    slide_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    objective: str | None = None
    key_points: list[str] = Field(default_factory=list)
    supporting_items: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    speaker_notes: str | None = None
    topic_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)


class PresentationOutline(StrictModel):
    """A presentation outline. Slide count is driven by material, not by a target."""

    schema_version: str = SCHEMA_VERSION
    title: str
    slides: list[SlideOutline] = Field(default_factory=list)
    target_slides: int = 10
    meets_target: bool = False
    limitation_note: str | None = None
