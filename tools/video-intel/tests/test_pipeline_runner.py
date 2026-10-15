"""Stage runner: cache reuse, staleness, atomic artifacts, failure and resume (§15)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel
from video_intel.pipeline.runner import StageContext, StageRunner, payload_hash
from video_intel.schemas.common import StageId, StageState
from video_intel.schemas.manifest import StageManifest
from video_intel.schemas.transcript import RawTranscript
from video_intel.storage.db import StateStore
from video_intel.storage.paths import ProjectPaths

VIDEO_ID = "synthetic01"


class Payload(BaseModel):
    value: str
    version: int = 1


@pytest.fixture
def context(tmp_path: Path) -> StageContext:
    project_paths = ProjectPaths(tmp_path / "data", "default").ensure()
    paths = project_paths.video(VIDEO_ID).ensure()
    store = StateStore(project_paths.database)
    store.initialize()
    store.upsert_project("default", "Default", now="2026-01-01T00:00:00+00:00")
    store.upsert_video("default", VIDEO_ID, now="2026-01-01T00:00:00+00:00")
    return StageContext(project_id="default", video_id=VIDEO_ID, paths=paths, store=store)


def _run(
    context: StageContext,
    *,
    value: str = "a",
    input_hash: str = "input-1",
    config_hash: str | None = None,
    force: bool = False,
) -> tuple[StageRunner, object, int]:
    calls = {"n": 0}

    def compute() -> Payload:
        calls["n"] += 1
        return Payload(value=value, version=calls["n"])

    runner = StageRunner(context)
    outcome = runner.run(
        StageId.METADATA,
        compute,
        artifact_path=context.paths.metadata,
        input_hash=input_hash,
        config_hash=config_hash,
        force=force,
    )
    return runner, outcome, calls["n"]


def test_stage_writes_artifact_and_records_state(context: StageContext) -> None:
    runner, outcome, calls = _run(context)
    assert calls == 1
    assert outcome.state is StageState.COMPLETED
    assert outcome.reused is False
    assert outcome.path is not None and outcome.path.is_file()
    assert outcome.output_hash

    record = runner.load_manifest().get(StageId.METADATA)
    assert record is not None
    assert record.state is StageState.COMPLETED
    assert record.output_hash == outcome.output_hash
    assert record.started_at is not None and record.completed_at is not None


def test_second_run_reuses_the_cached_artifact(context: StageContext) -> None:
    _run(context)
    _runner, outcome, calls = _run(context)
    assert calls == 0
    assert outcome.reused is True
    assert outcome.state is StageState.COMPLETED


def test_changed_input_invalidates_the_cache(context: StageContext) -> None:
    _run(context, value="first", input_hash="input-1")
    _runner, outcome, calls = _run(context, value="second", input_hash="input-2")
    assert calls == 1
    assert outcome.reused is False
    assert "second" in outcome.path.read_text(encoding="utf-8")


def test_changed_config_invalidates_the_cache(context: StageContext) -> None:
    _run(context, config_hash="cfg-1")
    _runner, outcome, calls = _run(context, value="new", config_hash="cfg-2")
    assert calls == 1
    assert outcome.reused is False


def test_force_recomputes_even_when_cached(context: StageContext) -> None:
    _run(context)
    _runner, outcome, calls = _run(context, force=True)
    assert calls == 1
    assert outcome.reused is False


def test_disabling_reuse_always_recomputes(context: StageContext) -> None:
    _run(context)
    calls = {"n": 0}

    def compute() -> Payload:
        calls["n"] += 1
        return Payload(value="x")

    runner = StageRunner(context, reuse=False)
    outcome = runner.run(
        StageId.METADATA,
        compute,
        artifact_path=context.paths.metadata,
        input_hash="input-1",
    )
    assert calls["n"] == 1
    assert outcome.reused is False


def test_deleted_artifact_forces_recomputation(context: StageContext) -> None:
    _run(context)
    context.paths.metadata.unlink()
    _runner, outcome, calls = _run(context)
    assert calls == 1
    assert outcome.reused is False


def test_failure_is_recorded_and_propagates(context: StageContext) -> None:
    runner = StageRunner(context)

    def boom() -> Payload:
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        runner.run(
            StageId.METADATA,
            boom,
            artifact_path=context.paths.metadata,
            input_hash="input-1",
        )

    record = runner.load_manifest().get(StageId.METADATA)
    assert record is not None
    assert record.state is StageState.FAILED
    assert "network down" in (record.error or "")
    assert record.reusable is False
    assert not context.paths.metadata.exists()


def test_retry_after_failure_succeeds_and_replaces_the_record(context: StageContext) -> None:
    runner = StageRunner(context)
    attempts = {"n": 0}

    def flaky() -> Payload:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return Payload(value="recovered")

    with pytest.raises(RuntimeError):
        runner.run(StageId.METADATA, flaky, artifact_path=context.paths.metadata, input_hash="i")
    outcome = runner.run(
        StageId.METADATA, flaky, artifact_path=context.paths.metadata, input_hash="i"
    )
    record = runner.load_manifest().get(StageId.METADATA)
    assert outcome.state is StageState.COMPLETED
    assert record is not None and record.state is StageState.COMPLETED
    assert record.attempts >= 2


def test_state_file_and_database_agree(context: StageContext) -> None:
    _run(context)
    assert context.paths.state.is_file()

    disk = StageManifest.model_validate_json(context.paths.state.read_text(encoding="utf-8"))
    from_db = context.store.load_manifest("default", VIDEO_ID)
    assert disk.is_complete(StageId.METADATA)
    assert from_db.is_complete(StageId.METADATA)


def test_manifest_merges_database_and_file_records(context: StageContext) -> None:
    _run(context)
    from video_intel.schemas.manifest import StageRecord

    manifest = StageManifest(video_id=VIDEO_ID)
    manifest.upsert(StageRecord(stage_id=StageId.CHUNKS, state=StageState.COMPLETED))
    context.store.upsert_stage("default", VIDEO_ID, manifest.stages[0])

    merged = StageRunner(context).load_manifest()
    assert merged.get(StageId.METADATA) is not None
    assert merged.get(StageId.CHUNKS) is not None


def test_input_hash_is_content_addressed() -> None:
    assert payload_hash("abc") == payload_hash("abc")
    assert payload_hash("abc") != payload_hash(b"abc\x00")


def test_transcript_artifact_roundtrips_through_the_runner(
    context: StageContext, transcript: RawTranscript
) -> None:
    runner = StageRunner(context)
    outcome = runner.run(
        StageId.RAW_TRANSCRIPT,
        lambda: transcript,
        artifact_path=context.paths.raw_transcript,
        input_hash=payload_hash("fetch"),
    )
    assert outcome.path is not None

    from video_intel.storage.io import read_model

    restored = read_model(outcome.path, RawTranscript)
    assert [s.segment_id for s in restored.segments] == [s.segment_id for s in transcript.segments]
    assert [(s.start_seconds, s.end_seconds) for s in restored.segments] == [
        (s.start_seconds, s.end_seconds) for s in transcript.segments
    ]
