"""Output requests and stage manifests: cache identity and staleness."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from video_intel.schemas.common import OutputMode, StageId, StageState, utc_now
from video_intel.schemas.manifest import ProjectManifest, StageManifest, StageRecord
from video_intel.schemas.outputs import OutputArtifact, OutputRequest


def test_deep_mode_requests_every_output() -> None:
    request = OutputRequest(mode=OutputMode.DEEP)
    assert request.outputs == {
        StageId.OUTPUT_NOTES,
        StageId.OUTPUT_JSON,
        StageId.OUTPUT_PPT,
        StageId.OUTPUT_SUMMARY,
    }


def test_quick_mode_is_smaller_than_deep() -> None:
    quick = OutputRequest(mode=OutputMode.QUICK).outputs
    deep = OutputRequest(mode=OutputMode.DEEP).outputs
    assert quick < deep


def test_explicit_outputs_override_mode_defaults() -> None:
    request = OutputRequest(mode=OutputMode.DEEP, outputs={StageId.OUTPUT_SUMMARY})
    assert request.outputs == {StageId.OUTPUT_SUMMARY}


def test_artifact_path_must_be_relative() -> None:
    with pytest.raises(ValidationError, match="must be relative"):
        OutputArtifact(
            stage_id=StageId.OUTPUT_NOTES,
            filename="full_notes.md",
            path="/absolute/full_notes.md",
            dataset_hash="abc",
        )


def test_artifact_binds_to_dataset_version() -> None:
    artifact = OutputArtifact(
        stage_id=StageId.OUTPUT_PPT,
        filename="ppt_outline.md",
        path="outputs/ppt_outline.md",
        dataset_hash="deadbeef",
        dataset_version=2,
    )
    assert artifact.dataset_version == 2
    assert artifact.stage_id is StageId.OUTPUT_PPT


def _record(**overrides: object) -> StageRecord:
    payload: dict[str, object] = {
        "stage_id": StageId.CHUNKS,
        "state": StageState.COMPLETED,
        "input_hash": "input-1",
        "output_hash": "output-1",
        "config_hash": "config-1",
    }
    payload.update(overrides)
    return StageRecord.model_validate(payload)


def test_completed_stage_with_output_is_reusable() -> None:
    assert _record().reusable is True


def test_stage_without_output_hash_is_not_reusable() -> None:
    assert _record(output_hash=None).reusable is False


def test_changed_transcript_marks_stage_stale() -> None:
    record = _record()
    assert record.is_stale(input_hash="input-2", config_hash="config-1") is True
    assert record.is_stale(input_hash="input-1", config_hash="config-1") is False


def test_changed_chunking_config_marks_stage_stale() -> None:
    assert _record().is_stale(input_hash="input-1", config_hash="config-2") is True


def test_failed_stage_is_always_stale() -> None:
    failed = _record(state=StageState.FAILED, output_hash=None)
    assert failed.is_stale(input_hash="input-1", config_hash="config-1") is True


def test_manifest_upsert_replaces_and_keeps_single_entry() -> None:
    manifest = StageManifest(video_id="v1")
    manifest.upsert(_record())
    manifest.upsert(_record(state=StageState.RUNNING, output_hash=None))
    assert len(manifest.stages) == 1
    assert manifest.is_complete(StageId.CHUNKS) is False
    manifest.upsert(_record())
    assert manifest.is_complete(StageId.CHUNKS) is True


def test_manifest_tracks_failed_stages() -> None:
    manifest = StageManifest(video_id="v1")
    manifest.upsert(_record(state=StageState.FAILED, error="timeout"))
    assert manifest.failed_stages() == [StageId.CHUNKS]


def test_manifest_updated_at_advances() -> None:
    manifest = StageManifest(video_id="v1")
    before = manifest.updated_at
    manifest.upsert(_record())
    assert manifest.updated_at - before >= timedelta(0)
    assert manifest.created_at <= manifest.updated_at


def test_project_manifest_tracks_videos() -> None:
    manifest = ProjectManifest(project_id="default", name="default", video_ids=["v1"])
    assert manifest.video_ids == ["v1"]
    assert manifest.created_at <= utc_now()
