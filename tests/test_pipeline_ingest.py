"""Phase 2 stages end to end: artifacts on disk, cache reuse, interruption recovery."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes import (
    FakeMetadataSource,
    FakeTranscriptSource,
    transcript_unavailable,
    video_unavailable,
)

from video_intel.config.loader import load_config
from video_intel.config.models import AppConfig
from video_intel.ingest.errors import TranscriptUnavailable, VideoUnavailable
from video_intel.ingest.service import IngestRequest, TranscriptIngestor
from video_intel.pipeline.ingest_stages import IMPLEMENTED_STAGES, ingest_stages
from video_intel.pipeline.runner import StageContext, StageRunner, payload_hash
from video_intel.schemas.common import StageId, StageState
from video_intel.schemas.transcript import (
    NormalizedTranscript,
    RawTranscript,
    VideoMetadata,
    hash_transcript,
)
from video_intel.storage.db import StateStore
from video_intel.storage.io import read_model
from video_intel.storage.paths import ProjectPaths

VIDEO_ID = "synthetic01"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
REPEATED = (
    ("Let me define it clearly.", 0.0, 3.0),
    ("Risk assessment is the process of evaluating risk.", 3.0, 7.0),
    ("Let me define it clearly.", 7.0, 10.0),
    ("Risk assessment is the process of evaluating risk.", 10.0, 14.0),
)


@pytest.fixture
def project(tmp_path: Path) -> tuple[ProjectPaths, AppConfig]:
    paths = ProjectPaths(tmp_path / "data", "default").ensure()
    config = load_config(
        project_dir=Path("/nonexistent"), overrides={"data_dir": str(paths.data_dir)}
    )
    return paths, config


@pytest.fixture
def runner(project: tuple[ProjectPaths, AppConfig]) -> StageRunner:
    paths, _ = project
    video_paths = paths.video(VIDEO_ID).ensure()
    store = StateStore(paths.database)
    store.initialize()
    store.upsert_project("default", "Default", now="2026-01-01T00:00:00+00:00")
    store.upsert_video("default", VIDEO_ID, url=URL, now="2026-01-01T00:00:00+00:00")
    return StageRunner(
        StageContext(project_id="default", video_id=VIDEO_ID, paths=video_paths, store=store)
    )


def _ingestor(transcript_source: FakeTranscriptSource | None = None) -> TranscriptIngestor:
    return TranscriptIngestor(transcript_source or FakeTranscriptSource(), FakeMetadataSource())


def test_implemented_stages_are_ordered() -> None:
    assert IMPLEMENTED_STAGES == (
        StageId.METADATA,
        StageId.RAW_TRANSCRIPT,
        StageId.NORMALIZED_TRANSCRIPT,
        StageId.CHUNKS,
    )


def test_ingestion_writes_every_artifact(runner: StageRunner, project: object) -> None:
    paths, config = project
    outcome = ingest_stages(
        runner=runner,
        ingestor=_ingestor(),
        request=IngestRequest(url=URL, languages=("en",)),
        config=config,
    )

    video_paths = paths.video(VIDEO_ID)
    assert video_paths.metadata.is_file()
    assert video_paths.raw_transcript.is_file()
    assert video_paths.normalized_transcript.is_file()
    assert video_paths.state.is_file()
    assert outcome.metadata.title.startswith("Synthetic")
    assert len(outcome.transcript.segments) == 4
    assert len(outcome.normalized.segments) == 4


def test_artifacts_are_validated_schemas_on_disk(runner: StageRunner, project: object) -> None:
    paths, config = project
    ingest_stages(
        runner=runner,
        ingestor=_ingestor(),
        request=IngestRequest(url=URL),
        config=config,
    )
    video_paths = paths.video(VIDEO_ID)

    metadata = read_model(video_paths.metadata, VideoMetadata)
    transcript = read_model(video_paths.raw_transcript, RawTranscript)
    normalized = read_model(video_paths.normalized_transcript, NormalizedTranscript)

    assert metadata.duration_seconds == 312.0
    assert transcript.source == "fake"
    assert transcript.segments[0].segment_id == "segment_0001"
    assert normalized.source_transcript_hash == hash_transcript(transcript.text())
    assert [s.start_seconds for s in normalized.segments] == [
        s.start_seconds for s in transcript.segments
    ]


def test_second_run_reuses_all_stages(runner: StageRunner, project: object) -> None:
    _paths, config = project
    request = IngestRequest(url=URL)
    first = ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=config)
    assert first.executed_stages == ("metadata", "raw_transcript", "normalized_transcript")
    assert first.reused_stages == ()

    second = ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=config)
    assert second.reused_stages == ("metadata", "raw_transcript", "normalized_transcript")
    assert second.executed_stages == ()


def test_force_recomputes_every_stage(runner: StageRunner, project: object) -> None:
    _paths, config = project
    request = IngestRequest(url=URL)
    ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=config)
    forced = ingest_stages(
        runner=runner, ingestor=_ingestor(), request=request, config=config, force=True
    )
    assert forced.reused_stages == ()


def test_caption_source_is_not_called_when_cached(runner: StageRunner, project: object) -> None:
    _paths, config = project
    request = IngestRequest(url=URL)
    ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=config)

    counting = FakeTranscriptSource()
    ingest_stages(runner=runner, ingestor=_ingestor(counting), request=request, config=config)
    assert counting.calls == []


def test_changing_language_preference_refetches(runner: StageRunner, project: object) -> None:
    _paths, config = project
    ingest_stages(
        runner=runner,
        ingestor=_ingestor(),
        request=IngestRequest(url=URL, languages=("en",)),
        config=config,
    )
    outcome = ingest_stages(
        runner=runner,
        ingestor=_ingestor(),
        request=IngestRequest(url=URL, languages=("de", "en")),
        config=config,
    )
    assert "raw_transcript" in outcome.executed_stages


def test_failed_transcript_stage_is_recorded_and_recoverable(
    runner: StageRunner, project: object
) -> None:
    paths, config = project
    broken = _ingestor(FakeTranscriptSource(failure=transcript_unavailable()))

    with pytest.raises(TranscriptUnavailable):
        ingest_stages(runner=runner, ingestor=broken, request=IngestRequest(url=URL), config=config)

    video_paths = paths.video(VIDEO_ID)
    assert not video_paths.raw_transcript.exists()
    record = runner.load_manifest().get(StageId.RAW_TRANSCRIPT)
    assert record is not None and record.state is StageState.FAILED

    recovered = ingest_stages(
        runner=runner, ingestor=_ingestor(), request=IngestRequest(url=URL), config=config
    )
    assert len(recovered.transcript.segments) == 4
    assert runner.load_manifest().is_complete(StageId.RAW_TRANSCRIPT)


def test_unavailable_video_does_not_leave_a_partial_transcript(
    runner: StageRunner, project: object
) -> None:
    paths, config = project
    broken = _ingestor(FakeTranscriptSource(failure=video_unavailable()))
    with pytest.raises(VideoUnavailable):
        ingest_stages(runner=runner, ingestor=broken, request=IngestRequest(url=URL), config=config)
    video_paths = paths.video(VIDEO_ID)
    assert not video_paths.raw_transcript.exists()
    assert not video_paths.normalized_transcript.exists()


def test_repeated_explanations_survive_ingestion(runner: StageRunner, project: object) -> None:
    _paths, config = project
    from video_intel.ingest.sources import CaptionSnippet

    source = FakeTranscriptSource(
        snippets=tuple(CaptionSnippet(text=t, start=s, duration=e - s) for t, s, e in REPEATED)
    )
    outcome = ingest_stages(
        runner=runner,
        ingestor=_ingestor(source),
        request=IngestRequest(url=URL),
        config=config,
    )
    texts = [segment.text for segment in outcome.normalized.segments]
    assert texts.count("Let me define it clearly.") == 2
    assert texts.count("Risk assessment is the process of evaluating risk.") == 2


def test_normalized_stage_is_invalidated_by_a_config_change(
    runner: StageRunner, project: object
) -> None:
    _paths, config = project
    request = IngestRequest(url=URL)
    ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=config)

    changed = config.model_copy(
        update={"chunking": config.chunking.model_copy(update={"max_chars": 9000})}
    )
    outcome = ingest_stages(runner=runner, ingestor=_ingestor(), request=request, config=changed)
    assert "metadata" in outcome.reused_stages
    assert "raw_transcript" in outcome.reused_stages
    assert "normalized_transcript" in outcome.executed_stages


def test_interrupted_run_resumes_without_refetching(runner: StageRunner, project: object) -> None:
    """Simulate a crash after metadata by running stages one at a time."""
    paths, config = project
    request = IngestRequest(url=URL)
    ingestor = _ingestor()
    video_paths = paths.video(VIDEO_ID)

    runner.run(
        StageId.METADATA,
        lambda: ingestor.fetch_metadata(VIDEO_ID),
        artifact_path=video_paths.metadata,
        input_hash=payload_hash(VIDEO_ID),
    )

    resumed = ingest_stages(runner=runner, ingestor=ingestor, request=request, config=config)
    assert resumed.reused_stages == ("metadata",)
    assert resumed.executed_stages == ("raw_transcript", "normalized_transcript")
    assert len(resumed.transcript.segments) == 4
