"""Reliability and cache behaviour (Phase 9, §15).

These tests pin the cache rules the specification asks for: work is reused only when it
belongs to the same configuration, a changed configuration invalidates exactly the
affected stage, and an interrupted run resumes without repeating completed work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeMetadataSource, FakeProvider, FakeTranscriptSource
from typer.testing import CliRunner

from video_intel.cli.main import app
from video_intel.config.loader import load_config
from video_intel.config.models import AppConfig, PipelineConfig
from video_intel.ingest.service import TranscriptIngestor
from video_intel.ingest.sources import TrackInfo
from video_intel.pipeline.analysis_stages import _cache_mismatch, run_extractions
from video_intel.pipeline.runner import StageContext, StageRunner
from video_intel.providers.registry import ProviderRegistry
from video_intel.schemas.chunk import Chunk, ChunkingConfig, ChunkSet
from video_intel.schemas.common import PipelineId, StageId, StageState
from video_intel.schemas.extraction import RawExtractionRecord
from video_intel.schemas.transcript import NormalizedTranscript, TranscriptSegment
from video_intel.storage.db import StateStore
from video_intel.storage.paths import ProjectPaths

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


VIDEO_ID = "synthetic01"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

SEGMENT = TranscriptSegment(
    segment_id="segment_0001",
    start_seconds=0.0,
    end_seconds=8.0,
    text="Risk assessment evaluates the likelihood of threats.",
)
NORMALIZED = NormalizedTranscript(
    video_id=VIDEO_ID,
    source_transcript_hash="hash-1",
    segments=[SEGMENT],
)
CHUNK_SET = ChunkSet(
    video_id=VIDEO_ID,
    config=ChunkingConfig(),
    transcript_hash="hash-1",
    chunks=[
        Chunk(
            chunk_id="chunk_0001",
            index=0,
            start_seconds=0.0,
            end_seconds=8.0,
            text=SEGMENT.text,
            segment_ids=[SEGMENT.segment_id],
        )
    ],
    cache_key="chunk-key",
)

CONTRACT = json.dumps(
    {
        "items": [
            {
                "type": "fact",
                "statement": "Risk assessment evaluates the likelihood of threats.",
                "source_segment_ids": ["segment_0001"],
                "details": [],
                "uncertainties": [],
            }
        ]
    }
)


def pipeline(model: str = "model-a", provider: str = "fake") -> PipelineConfig:
    return PipelineConfig(pipeline_id=PipelineId.FACT, provider=provider, model=model, passes=1)


@pytest.fixture
def stage_context(tmp_path: Path) -> StageContext:
    project = ProjectPaths(tmp_path / "data", "default").ensure()
    paths = project.video(VIDEO_ID).ensure()
    store = StateStore(project.database)
    store.initialize()
    store.upsert_project("default", "Default", now="2026-01-01T00:00:00+00:00")
    store.upsert_video("default", VIDEO_ID, now="2026-01-01T00:00:00+00:00")
    return StageContext(project_id="default", video_id=VIDEO_ID, paths=paths, store=store)


def config(tmp_path: Path) -> AppConfig:
    return load_config(
        project_dir=Path("/nonexistent"), overrides={"data_dir": str(tmp_path / "data")}
    )


# ---------------------------------------------------------------- cache mismatch
def _record(
    model: str = "model-a", provider: str = "fake", prompt: str = "1"
) -> RawExtractionRecord:
    from video_intel.schemas.common import ProviderKind
    from video_intel.schemas.extraction import ModelRef

    return RawExtractionRecord(
        record_id="fact_p1_chunk_0001",
        video_id=VIDEO_ID,
        pipeline_id=PipelineId.FACT,
        pass_index=1,
        model=ModelRef(provider=ProviderKind.OLLAMA, provider_name=provider, model=model),
        prompt_version=prompt,
        raw_response=CONTRACT,
        state=StageState.COMPLETED,
    )


def test_matching_record_is_reusable() -> None:
    assert (
        _cache_mismatch(
            _record(), pipeline=pipeline(), pass_index=1, model="model-a", provider_name="fake"
        )
        is None
    )


def test_different_model_invalidates_the_record() -> None:
    reason = _cache_mismatch(
        _record(),
        pipeline=pipeline("model-b"),
        pass_index=1,
        model="model-b",
        provider_name="fake",
    )
    assert reason is not None and "model" in reason


def test_different_provider_invalidates_the_record() -> None:
    reason = _cache_mismatch(
        _record(), pipeline=pipeline(), pass_index=1, model="model-a", provider_name="other"
    )
    assert reason is not None and "provider" in reason


def test_different_prompt_version_invalidates_the_record() -> None:
    reason = _cache_mismatch(
        _record(prompt="1"),
        pipeline=pipeline(),
        pass_index=1,
        model="model-a",
        provider_name="fake",
    )
    assert reason is None
    other = PipelineConfig(
        pipeline_id=PipelineId.FACT, provider="fake", model="model-a", prompt_version="2"
    )
    reason = _cache_mismatch(
        _record(prompt="1"),
        pipeline=other,
        pass_index=1,
        model="model-a",
        provider_name="fake",
    )
    assert reason is not None and "prompt" in reason


def test_different_pass_invalidates_the_record() -> None:
    reason = _cache_mismatch(
        _record(), pipeline=pipeline(), pass_index=2, model="model-a", provider_name="fake"
    )
    assert reason is not None and "pass" in reason


# --------------------------------------------------------------- stage caching
async def test_second_extraction_run_reuses_records(
    stage_context: StageContext, tmp_path: Path
) -> None:
    provider = FakeProvider(default_response=CONTRACT)
    first = await run_extractions(
        runner=StageRunner(stage_context),
        provider=provider,
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
    )
    assert first.records_written == 1
    assert len(provider.prompts) == 1

    second_provider = FakeProvider(default_response=CONTRACT)
    second = await run_extractions(
        runner=StageRunner(stage_context),
        provider=second_provider,
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
    )
    assert second.reused_records == 1
    assert second_provider.prompts == []
    assert len(second.outcome.items) == 1


async def test_switching_model_reextracts(stage_context: StageContext, tmp_path: Path) -> None:
    await run_extractions(
        runner=StageRunner(stage_context),
        provider=FakeProvider(default_response=CONTRACT),
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline("model-a")],
    )
    provider = FakeProvider(default_response=CONTRACT)
    result = await run_extractions(
        runner=StageRunner(stage_context),
        provider=provider,
        model="model-b",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline("model-b")],
    )
    assert len(provider.prompts) == 1, "a model change must invalidate stored records"
    assert result.outcome.items[0].provenance is not None
    assert result.outcome.items[0].provenance.model.model == "model-b"


async def test_force_reextracts_everything(stage_context: StageContext, tmp_path: Path) -> None:
    await run_extractions(
        runner=StageRunner(stage_context),
        provider=FakeProvider(default_response=CONTRACT),
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
    )
    provider = FakeProvider(default_response=CONTRACT)
    result = await run_extractions(
        runner=StageRunner(stage_context),
        provider=provider,
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
        force=True,
    )
    assert len(provider.prompts) == 1
    assert result.reused_records == 0


async def test_extraction_stage_state_is_recorded(
    stage_context: StageContext, tmp_path: Path
) -> None:
    runner = StageRunner(stage_context)
    await run_extractions(
        runner=runner,
        provider=FakeProvider(default_response=CONTRACT),
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
    )
    record = runner.load_manifest().get(StageId.EXTRACTIONS)
    assert record is not None
    assert record.state is StageState.COMPLETED
    assert record.output_hash
    manifest = stage_context.store.load_manifest("default", VIDEO_ID)
    assert manifest.is_complete(StageId.EXTRACTIONS)


async def test_failed_extraction_is_recorded_as_failed(
    stage_context: StageContext, tmp_path: Path
) -> None:
    from video_intel.providers.base import ProviderError

    provider = FakeProvider(default_response="{}", fail_times=5, errors=[ProviderError("down")])
    runner = StageRunner(stage_context)
    result = await run_extractions(
        runner=runner,
        provider=provider,
        model="model-a",
        chunk_set=CHUNK_SET,
        normalized=NORMALIZED,
        pipelines=[pipeline()],
    )
    assert result.outcome.ok_records == []
    record = runner.load_manifest().get(StageId.EXTRACTIONS)
    assert record is not None and record.state is StageState.FAILED
    assert record.error


# ---------------------------------------------------------------- cli resume
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nchunking:\n  max_chars: 4000\n  overlap_chars: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "video_intel.cli.main.build_ingestor",
        lambda cfg: TranscriptIngestor(
            FakeTranscriptSource(
                tracks=(TrackInfo(language="English", language_code="en", is_generated=True),)
            ),
            FakeMetadataSource(),
        ),
    )
    return config_file


def _install_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    """Patch the composition root so the CLI uses the given fake provider."""

    def fake_registry(self: Any) -> ProviderRegistry:
        registry = ProviderRegistry(self.config)
        monkeypatch.setattr(registry, "build", lambda name, model=None: provider)
        return registry

    monkeypatch.setattr("video_intel.cli.context.AppContext.registry", fake_registry)


def test_interrupted_run_resumes_without_repeating_work(
    cli_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """Kill the run after extraction, then resume: no provider call may repeat."""
    provider = FakeProvider(default_response=CONTRACT)
    _install_provider(monkeypatch, provider)

    app_result = runner.invoke(
        app, ["analyze", URL, "--config", str(cli_env), "--until", "extractions"]
    )
    assert app_result.exit_code in (0, 3), app_result.output
    calls = len(provider.prompts)
    assert calls > 0

    resumed = FakeProvider(default_response=CONTRACT)
    _install_provider(monkeypatch, resumed)
    result = runner.invoke(app, ["analyze", URL, "--config", str(cli_env)])
    assert result.exit_code == 0, result.output
    assert resumed.prompts == []


def test_state_json_survives_a_deleted_database(
    cli_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """The manifest on disk is enough to skip completed stages."""
    _install_provider(monkeypatch, FakeProvider(default_response=CONTRACT))
    runner.invoke(app, ["analyze", URL, "--config", str(cli_env)])

    (tmp_path / "data" / "state.db").unlink(missing_ok=True)
    provider = FakeProvider(default_response=CONTRACT)
    _install_provider(monkeypatch, provider)
    result = runner.invoke(app, ["analyze", URL, "--config", str(cli_env)])
    assert result.exit_code == 0
    assert provider.prompts == []


def test_outputs_are_regenerated_when_only_the_dataset_changes(
    cli_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    _install_provider(monkeypatch, FakeProvider(default_response=CONTRACT))
    runner.invoke(app, ["analyze", URL, "--config", str(cli_env)])

    outputs = tmp_path / "data" / "projects" / "default" / VIDEO_ID / "outputs"
    before = (outputs / "full_notes.md").stat().st_mtime_ns

    provider = FakeProvider(default_response=CONTRACT)
    _install_provider(monkeypatch, provider)
    runner.invoke(app, ["analyze", URL, "--config", str(cli_env), "-m", "ppt"])
    assert (outputs / "ppt_outline.md").is_file()
    assert provider.prompts == []
    assert (outputs / "full_notes.md").stat().st_mtime_ns == before  # untouched output


def test_log_file_is_written_for_real_runs(
    cli_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    _install_provider(monkeypatch, FakeProvider(default_response=CONTRACT))
    runner.invoke(app, ["analyze", URL, "--config", str(cli_env)])
    log = tmp_path / "data" / "logs" / "video-intel.log"
    assert log.is_file()
    contents = log.read_text(encoding="utf-8")
    assert "stage=chunks" in contents
    assert "api_key" not in contents.lower()
