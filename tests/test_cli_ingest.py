"""CLI ingestion path: real command flow with synthetic sources injected."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import (
    FakeMetadataSource,
    FakeTranscriptSource,
    transcript_unavailable,
    video_unavailable,
)
from typer.testing import CliRunner

from video_intel.cli.main import app
from video_intel.ingest.service import TranscriptIngestor
from video_intel.ingest.sources import TrackInfo
from video_intel.storage.io import read_json

VIDEO_URL = "https://www.youtube.com/watch?v=synthetic01"
VIDEO_ID = "synthetic01"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "video-intel.yaml"
    path.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    return path


@pytest.fixture
def inject_fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace the composition root so the CLI runs against synthetic sources."""

    def _install(
        transcript_source: FakeTranscriptSource | None = None,
        metadata_source: FakeMetadataSource | None = None,
    ) -> FakeTranscriptSource:
        source = transcript_source or FakeTranscriptSource()
        ingestor = TranscriptIngestor(source, metadata_source or FakeMetadataSource())
        monkeypatch.setattr("video_intel.cli.main.build_ingestor", lambda config: ingestor)
        return source

    return _install


def test_analyze_ingests_transcript_and_chunks(
    runner: CliRunner, config_file: Path, inject_fake: Any, tmp_path: Path
) -> None:
    source = inject_fake()
    result = runner.invoke(
        app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"]
    )

    assert result.exit_code == EXIT_OK
    output = result.stdout + (result.stderr if result.stderr_bytes else "")
    assert "ingestion complete" in output
    assert "stopped before analysis" in output

    video_dir = tmp_path / "data" / "projects" / "default" / VIDEO_ID
    assert (video_dir / "metadata.json").is_file()
    assert (video_dir / "raw_transcript.json").is_file()
    assert (video_dir / "normalized_transcript.json").is_file()
    assert source.calls == [(VIDEO_ID, ("en",))]


def test_analyze_reports_transcript_details(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    inject_fake()
    result = runner.invoke(
        app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"]
    )
    output = result.stdout + (result.stderr if result.stderr_bytes else "")
    assert "Synthetic risk assessment lecture" in output
    assert "4 segments" in output
    assert "language=en" in output
    assert "kind=automatic" in output


def test_second_analyze_reuses_cached_stages(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    inject_fake()
    runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"])

    second_source = inject_fake()
    result = runner.invoke(
        app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"]
    )
    output = result.stdout + (result.stderr if result.stderr_bytes else "")
    assert second_source.calls == []
    assert "reused" in output


def test_force_refetches_transcript(runner: CliRunner, config_file: Path, inject_fake: Any) -> None:
    inject_fake()
    runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"])

    forced_source = inject_fake()
    result = runner.invoke(
        app,
        ["analyze", VIDEO_URL, "--config", str(config_file), "--force", "--until", "chunks"],
    )
    assert forced_source.calls == [(VIDEO_ID, ("en",))]
    assert result.exit_code == EXIT_OK


def test_language_option_selects_that_track(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    tracks = (
        TrackInfo(language="English", language_code="en", is_generated=True),
        TrackInfo(language="Deutsch", language_code="de", is_generated=True),
    )
    source = inject_fake(FakeTranscriptSource(tracks=tracks))
    result = runner.invoke(
        app,
        [
            "analyze",
            VIDEO_URL,
            "--config",
            str(config_file),
            "-l",
            "de",
            "-l",
            "en",
            "--until",
            "chunks",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert source.calls == [(VIDEO_ID, ("de",))]
    assert "language=de" in (result.stdout + (result.stderr if result.stderr_bytes else ""))


def test_until_metadata_skips_transcript_stages(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    source = inject_fake()
    result = runner.invoke(
        app,
        ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "metadata"],
    )
    assert result.exit_code == EXIT_OK
    assert source.calls == []

    video_dir = config_file.parent / "data" / "projects" / "default" / VIDEO_ID
    assert (video_dir / "metadata.json").is_file()
    assert not (video_dir / "raw_transcript.json").exists()


def test_until_rejects_unimplemented_stage(runner: CliRunner, config_file: Path) -> None:
    result = runner.invoke(
        app,
        ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "verify"],
    )
    assert result.exit_code == EXIT_CONFIG
    assert "not implemented" in (result.stdout + result.stderr)


def test_analyze_writes_chunks(
    runner: CliRunner, config_file: Path, inject_fake: Any, tmp_path: Path
) -> None:
    inject_fake()
    result = runner.invoke(
        app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"]
    )
    assert result.exit_code == EXIT_OK
    video_dir = tmp_path / "data" / "projects" / "default" / VIDEO_ID
    chunks = json.loads((video_dir / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks["chunks"]) >= 1
    assert chunks["chunks"][0]["chunk_id"] == "chunk_0001"
    covered = {sid for chunk in chunks["chunks"] for sid in chunk["segment_ids"]}
    transcript = json.loads((video_dir / "raw_transcript.json").read_text(encoding="utf-8"))
    assert covered == {segment["segment_id"] for segment in transcript["segments"]}


def test_dry_run_writes_nothing(
    runner: CliRunner, config_file: Path, inject_fake: Any, tmp_path: Path
) -> None:
    source = inject_fake()
    result = runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--dry-run"])
    assert result.exit_code == EXIT_OK
    assert source.calls == []
    assert not (tmp_path / "data" / "projects").exists()


def test_missing_captions_reports_the_problem_and_a_next_step(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    inject_fake(FakeTranscriptSource(failure=transcript_unavailable()))
    result = runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file)])
    output = result.stdout + (result.stderr if result.stderr_bytes else "")
    assert result.exit_code == EXIT_ERROR
    assert "no usable transcript" in output
    assert "speech-to-text" in output
    assert "content is ever fabricated" in re.sub(r"\s+", " ", output)


def test_unavailable_video_is_reported(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    inject_fake(FakeTranscriptSource(failure=video_unavailable()))
    result = runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file)])
    output = result.stdout + (result.stderr if result.stderr_bytes else "")
    assert result.exit_code == EXIT_ERROR
    assert "unavailable" in output


def test_inspect_shows_ingested_artifacts(
    runner: CliRunner, config_file: Path, inject_fake: Any
) -> None:
    inject_fake()
    runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"])
    result = runner.invoke(app, ["inspect", VIDEO_ID, "--config", str(config_file)])
    assert result.exit_code == EXIT_OK
    assert "raw_transcript.json" in result.stdout
    assert "normalized_transcript.json" in result.stdout
    assert "completed" in result.stdout


def test_state_file_records_stage_hashes(
    runner: CliRunner, config_file: Path, inject_fake: Any, tmp_path: Path
) -> None:
    inject_fake()
    runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"])
    state = read_json(tmp_path / "data" / "projects" / "default" / VIDEO_ID / "state.json")
    stages = {stage["stage_id"]: stage for stage in state["stages"]}
    assert set(stages) == {
        "metadata",
        "raw_transcript",
        "normalized_transcript",
        "chunks",
    }
    for stage in stages.values():
        assert stage["state"] == "completed"
        assert stage["output_hash"]
        assert stage["input_hash"]


def test_transcript_artifact_keeps_timestamps(
    runner: CliRunner, config_file: Path, inject_fake: Any, tmp_path: Path
) -> None:
    inject_fake()
    runner.invoke(app, ["analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks"])
    transcript = json.loads(
        (tmp_path / "data" / "projects" / "default" / VIDEO_ID / "raw_transcript.json").read_text(
            encoding="utf-8"
        )
    )
    timings = [(s["start_seconds"], s["end_seconds"]) for s in transcript["segments"]]
    assert timings[0] == (0.0, 3.2)
    assert timings[-1] == (8.0, 12.5)
