"""Processing-state models (§15, §19).

Stage records are what make caching, invalidation and resume possible: a stage is
reused only when its inputs, configuration and recorded outputs still match. State
lives here and is mirrored into SQLite; the JSON manifest is the human-readable view.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from video_intel.schemas.common import SCHEMA_VERSION, StageId, StageState, StrictModel, utc_now


class StageRecord(StrictModel):
    """Execution state and cache identity for one processing stage."""

    stage_id: StageId
    state: StageState = StageState.PENDING
    attempts: int = Field(default=0, ge=0)
    input_hash: str | None = None
    output_hash: str | None = None
    config_hash: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error: str | None = None
    detail: dict[str, str] = Field(default_factory=dict)

    @property
    def reusable(self) -> bool:
        """A completed stage can be reused only if it produced an output hash."""
        return self.state is StageState.COMPLETED and self.output_hash is not None

    def is_stale(self, *, input_hash: str | None, config_hash: str | None) -> bool:
        """True when cached output no longer matches the current inputs."""
        if not self.reusable:
            return True
        if input_hash is not None and self.input_hash != input_hash:
            return True
        return config_hash is not None and self.config_hash != config_hash


class StageManifest(StrictModel):
    """All stage records for one video."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    stages: list[StageRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def get(self, stage_id: StageId) -> StageRecord | None:
        return next((stage for stage in self.stages if stage.stage_id == stage_id), None)

    def is_complete(self, stage_id: StageId) -> bool:
        record = self.get(stage_id)
        return record is not None and record.state is StageState.COMPLETED

    def upsert(self, record: StageRecord) -> StageManifest:
        """Insert or replace a stage record, keeping order stable."""
        for index, existing in enumerate(self.stages):
            if existing.stage_id == record.stage_id:
                self.stages[index] = record
                break
        else:
            self.stages.append(record)
        self.updated_at = utc_now()
        return self

    def failed_stages(self) -> list[StageId]:
        return [stage.stage_id for stage in self.stages if stage.state is StageState.FAILED]


class ProjectManifest(StrictModel):
    """Project-level bookkeeping: videos, config hash, output requests."""

    schema_version: str = SCHEMA_VERSION
    project_id: str
    name: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    config_hash: str | None = None
    video_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
