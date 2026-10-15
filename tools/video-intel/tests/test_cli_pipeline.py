"""End-to-end CLI runs: ingest, chunk, extract, verify, assemble, render (§14, §22).

The provider is injected at the composition root, so the whole pipeline runs offline and
deterministically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from video_intel.cli.main import app
from video_intel.ingest.service import TranscriptIngestor
from video_intel.ingest.sources import TrackInfo
from video_intel.providers.registry import ProviderRegistry
from video_intel.testing.fakes import FakeMetadataSource, FakeProvider, FakeTranscriptSource

VIDEO_URL = "https://www.youtube.com/watch?v=synthetic01"
VIDEO_ID = "synthetic01"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2026", ""))


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "video-intel.yaml"
    path.write_text(
        f"data_dir: {tmp_path / 'data'}\n"
        "chunking:\n"
        "  max_chars: 600\n"
        "  overlap_chars: 100\n"
        "  min_chars: 100\n"
        "execution_mode: local_only\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def video_dir(config_file: Path, tmp_path: Path) -> Path:
    return tmp_path / "data" / "projects" / "default" / VIDEO_ID


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    provider: FakeProvider,
    *,
    tracks: tuple[TrackInfo, ...] | None = None,
) -> FakeTranscriptSource:
    """Inject transcript, metadata and provider fakes into the CLI."""
    transcript_source = FakeTranscriptSource(
        tracks=tracks
        or (
            TrackInfo(language="English", language_code="en", is_generated=True),
            TrackInfo(language="Deutsch", language_code="de", is_generated=True),
        )
    )
    ingestor = TranscriptIngestor(transcript_source, FakeMetadataSource())
    monkeypatch.setattr("video_intel.cli.main.build_ingestor", lambda config: ingestor)

    def fake_registry(self: Any) -> ProviderRegistry:
        registry = ProviderRegistry(self.config)
        monkeypatch.setattr(registry, "build", lambda name, model=None: provider)
        return registry

    monkeypatch.setattr("video_intel.cli.context.AppContext.registry", fake_registry)
    return transcript_source


def contract(segment_id: str, statement: str, item_type: str = "fact") -> str:
    return json.dumps(
        {
            "items": [
                {
                    "type": item_type,
                    "statement": statement,
                    "details": [],
                    "source_segment_ids": [segment_id],
                    "uncertainties": [],
                }
            ]
        }
    )


def run(runner: CliRunner, *args: str) -> Any:
    return runner.invoke(app, list(args))


def test_full_run_produces_every_output(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract(
            "segment_0001", "Risk assessment is the process of evaluating risk."
        )
    )
    install_fakes(monkeypatch, provider)

    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    assert result.exit_code == EXIT_OK, result.output

    outputs = video_dir / "outputs"
    for name in ("full_notes.md", "summary.md", "ppt_outline.md", "knowledge.json"):
        assert (outputs / name).is_file(), f"{name} missing"

    dataset = json.loads((video_dir / "knowledge.json").read_text(encoding="utf-8"))
    assert dataset["video_id"] == VIDEO_ID
    assert dataset["schema_version"] == "1.0.0"
    assert dataset["items"], "dataset should contain extracted items"
    assert (video_dir / "raw_transcript.json").is_file()
    assert (video_dir / "chunks.json").is_file()
    assert list((video_dir / "raw_extractions").rglob("*.json"))


def test_dataset_items_carry_provenance_and_timestamps(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "Welcome to the channel.", item_type="concept")
    )
    install_fakes(monkeypatch, provider)
    assert run(runner, "analyze", VIDEO_URL, "--config", str(config_file)).exit_code == EXIT_OK

    dataset = json.loads((video_dir / "outputs" / "knowledge.json").read_text(encoding="utf-8"))
    for item in dataset["items"]:
        assert item["source_segment_ids"]
        assert item["start_seconds"] is not None
        assert item["provenance"]
        assert item["provenance"][0]["pipeline_id"]
        assert item["provenance"][0]["model"]["model"]


def test_unsupported_claims_are_excluded_and_reported(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract(
            "segment_0001",
            "Quantum tunnelling makes risk assessment entirely optional in practice.",
        )
    )
    install_fakes(monkeypatch, provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    assert result.exit_code == EXIT_OK

    dataset = json.loads((video_dir / "outputs" / "knowledge.json").read_text(encoding="utf-8"))
    assert dataset["items"] == []
    notes = flat((video_dir / "outputs" / "full_notes.md").read_text(encoding="utf-8"))
    assert "No verified knowledge items" in notes


def test_second_run_reuses_extraction(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "It identifies potential threats.")
    )
    install_fakes(monkeypatch, provider)
    assert run(runner, "analyze", VIDEO_URL, "--config", str(config_file)).exit_code == EXIT_OK
    calls_after_first = len(provider.prompts)
    assert calls_after_first > 0

    second = FakeProvider(default_response=provider.default_response)
    install_fakes(monkeypatch, second)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert second.prompts == [], "cached extraction records must be reused, not refetched"


def test_changing_output_format_does_not_call_the_model(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§20.6: only the output format changed, so no AI call may happen."""
    provider = FakeProvider(
        default_response=contract("segment_0001", "It identifies potential threats.")
    )
    install_fakes(monkeypatch, provider)
    assert run(
        runner, "analyze", VIDEO_URL, "--config", str(config_file), "-m", "deep"
    ).exit_code == (EXIT_OK)
    baseline = len(provider.prompts)

    quick_provider = FakeProvider(default_response=provider.default_response)
    install_fakes(monkeypatch, quick_provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file), "-m", "quick")
    assert result.exit_code == EXIT_OK
    assert quick_provider.prompts == []
    assert len(provider.prompts) == baseline

    outputs = video_dir / "outputs"
    assert (outputs / "summary.md").is_file()
    assert (outputs / "ppt_outline.md").is_file()  # kept from the deep run


def test_force_reruns_extraction(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "It identifies potential threats.")
    )
    install_fakes(monkeypatch, provider)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    first = len(provider.prompts)

    forced = FakeProvider(default_response=provider.default_response)
    install_fakes(monkeypatch, forced)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file), "--force")
    assert len(forced.prompts) == first > 0


def test_total_provider_failure_is_not_reported_as_success(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_intel.providers.base import ProviderError

    provider = FakeProvider(
        default_response="{}",
        fail_times=99,
        errors=[ProviderError("ollama is not reachable")],
    )
    install_fakes(monkeypatch, provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    output = flat(result.stdout + (result.stderr if result.stderr_bytes else ""))
    assert result.exit_code == EXIT_ERROR
    assert "no provider returned usable output" in output


def test_partial_provider_failure_keeps_working(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_intel.providers.base import ProviderError

    provider = FakeProvider(
        default_response=contract("segment_0001", "Welcome to the channel.", "concept"),
        fail_times=1,
        errors=[ProviderError("transient failure")],
    )
    install_fakes(monkeypatch, provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    output = flat(result.stdout + (result.stderr if result.stderr_bytes else ""))
    assert result.exit_code == EXIT_OK
    assert "chunk/pipeline call(s) failed" in output

    dataset = json.loads((video_dir / "outputs" / "knowledge.json").read_text(encoding="utf-8"))
    assert dataset["items"], "results from the successful calls must be kept"
    assert StageFailed(dataset) is True


def StageFailed(dataset: dict[str, Any]) -> bool:
    return "extractions" in dataset["completeness"]["failed_stages"]


def test_ppt_outline_reports_short_material(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "It identifies potential threats.")
    )
    install_fakes(monkeypatch, provider)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file), "-m", "ppt")
    outline = (video_dir / "outputs" / "ppt_outline.md").read_text(encoding="utf-8")
    assert "Presentation outline" in outline
    assert "Limitation" in outline  # this fixture cannot fill ten slides
    assert "Nothing was invented to reach the target" in outline or (
        "No topics were extracted" in outline
    )


def test_notes_include_timestamps_and_sources(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "Welcome to the channel.", item_type="concept")
    )
    install_fakes(monkeypatch, provider)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    notes = (video_dir / "outputs" / "full_notes.md").read_text(encoding="utf-8")
    assert "Source:" in notes
    assert "segment_0001" in notes
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", notes)


def test_summary_is_generated_from_the_dataset(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "Welcome to the channel.", item_type="concept")
    )
    install_fakes(monkeypatch, provider)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file))
    summary = (video_dir / "outputs" / "summary.md").read_text(encoding="utf-8")
    assert "# Summary" in summary
    assert "Welcome to the channel." in summary


def test_until_stops_before_extraction(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(default_response="{}")
    install_fakes(monkeypatch, provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file), "--until", "chunks")
    assert result.exit_code == EXIT_OK
    assert provider.prompts == []
    assert (video_dir / "chunks.json").is_file()
    assert not (video_dir / "knowledge.json").exists()


def test_inspect_reports_analysis_state(
    runner: CliRunner, config_file: Path, video_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        default_response=contract("segment_0001", "It identifies potential threats.")
    )
    install_fakes(monkeypatch, provider)
    run(runner, "analyze", VIDEO_URL, "--config", str(config_file))

    result = run(runner, "inspect", VIDEO_ID, "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    rendered = flat(result.stdout)
    assert "knowledge.json" in rendered
    assert "completed" in rendered

    state = json.loads((video_dir / "state.json").read_text(encoding="utf-8"))
    stages = {entry["stage_id"]: entry["state"] for entry in state["stages"]}
    assert stages["extractions"] == "completed"
    assert stages["knowledge_dataset"] == "completed"
    assert stages["output_notes"] == "completed"


def test_dry_run_makes_no_calls(
    runner: CliRunner, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(default_response="{}")
    install_fakes(monkeypatch, provider)
    result = run(runner, "analyze", VIDEO_URL, "--config", str(config_file), "--dry-run")
    assert result.exit_code == EXIT_OK
    assert provider.prompts == []
    assert not (tmp_path / "data" / "projects").exists()


def test_local_only_refuses_cloud_provider_override(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(default_response="{}")
    install_fakes(monkeypatch, provider)
    result = run(
        runner,
        "analyze",
        VIDEO_URL,
        "--config",
        str(config_file),
        "--provider",
        "openai",
        "--dry-run",
    )
    assert result.exit_code == EXIT_CONFIG
