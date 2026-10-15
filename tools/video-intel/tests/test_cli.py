"""CLI behaviour, including honest failure before Phases 2-9 exist."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from video_intel.schemas.common import ProviderKind

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_NOT_IMPLEMENTED = 3


def flat(text: str) -> str:
    """Collapse rich's wrapping and padding so assertions can match content."""
    return re.sub(r"\s+", " ", text.replace("\u2026", ""))


def test_version_flag(invoke: Any) -> None:
    result = invoke("--version")
    assert result.exit_code == EXIT_OK
    assert "video-intel" in result.stdout


def test_help_lists_the_documented_commands(invoke: Any) -> None:
    result = invoke("--help")
    assert result.exit_code == EXIT_OK
    for command in ("analyze", "providers", "pipelines", "inspect", "init", "config"):
        assert command in result.stdout


def test_providers_table_reports_adapter_status(invoke: Any) -> None:
    result = invoke("providers")
    assert result.exit_code == EXIT_OK
    rendered = flat(result.stdout)
    assert "ollama" in rendered
    assert "openai" in rendered
    assert "local" in rendered and "cloud" in rendered


def test_providers_json_reports_per_adapter_status(invoke: Any) -> None:
    from video_intel.providers.registry import IMPLEMENTED_KINDS

    result = invoke("providers", "--json")
    payload = json.loads(result.stdout)
    by_name = {entry["name"]: entry for entry in payload}
    assert by_name["ollama"]["adapter_implemented"] is True
    assert by_name["gemini"]["adapter_implemented"] is False
    assert by_name["notebooklm_handoff"]["adapter_implemented"] is False
    assert ProviderKind("ollama") in IMPLEMENTED_KINDS
    assert all(entry["base_url"] or entry["kind"] for entry in payload)


def test_providers_json_never_exposes_key_values(
    invoke: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-secret-value")
    result = invoke("providers", "--json")
    assert result.exit_code == EXIT_OK
    payload = json.loads(result.stdout)
    openai = next(entry for entry in payload if entry["name"] == "openai")
    assert openai["api_key_present"] is True
    assert "sk-cli-test-secret-value" not in result.stdout


def test_providers_notes_notebooklm_is_not_an_api(invoke: Any) -> None:
    result = invoke("providers")
    rendered = flat(result.stdout)
    assert "notebooklm_handoff" in rendered
    assert "not an automated API integration" in rendered


def test_pipelines_lists_all_eight_categories(invoke: Any) -> None:
    result = invoke("pipelines")
    assert result.exit_code == EXIT_OK
    for pipeline in (
        "concept",
        "definition",
        "fact",
        "example",
        "framework",
        "process",
        "comparison",
        "coverage_audit",
    ):
        assert pipeline in result.stdout
    assert "maximum-coverage" in result.stdout


def test_pipelines_json_structure(invoke: Any) -> None:
    result = invoke("pipelines", "--json")
    assert result.exit_code == EXIT_OK
    payload = json.loads(result.stdout)
    assert payload["pipelines"]["fact"]["provider"] == "ollama"
    assert "deep" in payload["modes"]
    assert "maximum-coverage" in payload["presets"]


def test_config_validate_reports_sources(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text("project_name: cli-test\n", encoding="utf-8")
    result = invoke("config", "validate", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "configuration is valid" in result.stdout
    assert str(config_file) in result.stdout


def test_config_validate_rejects_bad_file(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("unknown_setting: 1\n", encoding="utf-8")
    result = invoke("config", "validate", "--config", str(config_file))
    assert result.exit_code == EXIT_CONFIG
    assert "configuration error" in result.output


def test_config_show_hides_secret_values(
    invoke: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-cli-secret-value")
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    result = invoke("config", "show", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "gsk-cli-secret-value" not in result.stdout
    assert "GROQ_API_KEY" in result.stdout


def test_config_path_lists_precedence(invoke: Any) -> None:
    result = invoke("config", "path")
    assert result.exit_code == EXIT_OK
    assert "packaged" in result.stdout
    assert "VIDEO_INTEL_" in result.stdout


def test_analyze_rejects_non_youtube_url(invoke: Any) -> None:
    result = invoke("analyze", "https://example.com/watch?v=abc")
    assert result.exit_code == EXIT_ERROR
    assert "not a recognisable YouTube video URL" in result.output


def test_analyze_dry_run_succeeds_without_processing(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    result = invoke(
        "analyze", VIDEO_URL, "--mode", "deep", "--dry-run", "--config", str(config_file)
    )
    assert result.exit_code == EXIT_OK
    assert "dry run: nothing was fetched, processed or written" in result.stdout
    assert not (tmp_path / "data" / "projects").exists()


def test_analyze_plan_shows_pipelines_and_outputs(invoke: Any) -> None:
    result = invoke("analyze", VIDEO_URL, "--dry-run")
    assert result.exit_code == EXIT_OK
    assert "Pipeline plan" in result.stdout
    assert "full_notes.md" in result.stdout
    assert "knowledge.json" in result.stdout


def test_analyze_rejects_unknown_provider(invoke: Any) -> None:
    result = invoke("analyze", VIDEO_URL, "--provider", "does-not-exist", "--dry-run")
    assert result.exit_code == EXIT_CONFIG
    assert "unknown provider" in result.output


def test_analyze_rejects_unknown_preset(invoke: Any) -> None:
    result = invoke("analyze", VIDEO_URL, "--pipeline", "nope", "--dry-run")
    assert result.exit_code == EXIT_CONFIG
    assert "unknown pipeline preset" in result.output


def test_analyze_preset_selection(invoke: Any) -> None:
    result = invoke("analyze", VIDEO_URL, "--pipeline", "maximum-coverage", "--dry-run")
    assert result.exit_code == EXIT_OK
    assert "coverage_audit" in result.stdout


def test_analyze_local_only_rejects_cloud_override(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text("execution_mode: local_only\n", encoding="utf-8")
    result = invoke(
        "analyze", VIDEO_URL, "--provider", "openai", "--dry-run", "--config", str(config_file)
    )
    assert result.exit_code == EXIT_CONFIG
    assert "local_only" in result.output


def test_analyze_allows_cloud_provider_when_explicitly_requested(
    invoke: Any, tmp_path: Path
) -> None:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text("execution_mode: cloud_only\n", encoding="utf-8")
    result = invoke(
        "analyze",
        VIDEO_URL,
        "--provider",
        "openai",
        "--dry-run",
        "--config",
        str(config_file),
    )
    assert result.exit_code == EXIT_OK
    assert "openai" in flat(result.stdout)


def test_init_creates_layout_and_database(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    data_dir = tmp_path / "data"
    config_file.write_text(f"data_dir: {data_dir}\n", encoding="utf-8")
    result = invoke("init", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert (data_dir / "state.db").is_file()
    assert (data_dir / "projects" / "default" / "project.json").is_file()
    assert (data_dir / "logs").is_dir()


def test_inspect_reports_missing_video(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    result = invoke("inspect", "dQw4w9WgXcQ", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "no stored data for this video yet" in result.stdout


def test_inspect_lists_artifacts(invoke: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "video-intel.yaml"
    data_dir = tmp_path / "data"
    config_file.write_text(f"data_dir: {data_dir}\n", encoding="utf-8")
    invoke("init", "--config", str(config_file))

    video_dir = data_dir / "projects" / "default" / "dQw4w9WgXcQ"
    (video_dir / "outputs").mkdir(parents=True)
    (video_dir / "knowledge.json").write_text("{}", encoding="utf-8")
    (video_dir / "outputs" / "summary.md").write_text("# summary", encoding="utf-8")

    result = invoke("inspect", "dQw4w9WgXcQ", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "knowledge.json" in result.stdout
    assert "summary.md" in result.stdout


def test_read_only_commands_do_not_create_directories(invoke: Any, tmp_path: Path) -> None:
    workdir = tmp_path / "cwd"
    workdir.mkdir()

    previous = Path.cwd()
    os.chdir(workdir)
    try:
        assert invoke("providers").exit_code == EXIT_OK
        assert invoke("config", "show").exit_code == EXIT_OK
        assert list(workdir.iterdir()) == []
    finally:
        os.chdir(previous)


# ------------------------------------------------------- device-transfer commands


def _project(invoke: Any, tmp_path: Path) -> tuple[Path, Path]:
    """An initialised project, plus the config file that points at it."""
    data_dir = tmp_path / "data"
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(f"data_dir: {data_dir}\nproject_name: default\n", encoding="utf-8")
    assert invoke("init", "--config", str(config_file)).exit_code == EXIT_OK
    return data_dir, config_file


def test_doctor_reports_paths_and_survives_an_uninitialised_workspace(
    invoke: Any, tmp_path: Path
) -> None:
    """The diagnostic must work on a machine that is not set up yet."""
    config_file = tmp_path / "video-intel.yaml"
    config_file.write_text(
        f"data_dir: {tmp_path / 'elsewhere'}\nproject_name: default\n", encoding="utf-8"
    )
    result = invoke("doctor", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "no state database yet" in result.stdout
    assert "providers" in result.stdout


def test_doctor_does_not_create_directories(invoke: Any, tmp_path: Path) -> None:
    """Read-only, like every other diagnostic."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        assert invoke("doctor").exit_code == EXIT_OK
        assert list(workdir.iterdir()) == []
    finally:
        os.chdir(previous)


def test_checkpoint_makes_the_database_self_contained(invoke: Any, tmp_path: Path) -> None:
    """A copied state.db must carry its own schema, not depend on a -wal sidecar."""
    import sqlite3

    data_dir, config_file = _project(invoke, tmp_path)
    database = data_dir / "state.db"

    result = invoke("checkpoint", "--config", str(config_file))
    assert result.exit_code == EXIT_OK
    assert "checkpointed" in result.stdout

    destination = tmp_path / "copied.db"
    destination.write_bytes(database.read_bytes())
    copied = sqlite3.connect(destination)
    try:
        tables = {
            row[0] for row in copied.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        copied.close()
    assert {"projects", "videos", "stages", "artifacts"} <= tables


def test_checkpoint_is_safe_to_repeat(invoke: Any, tmp_path: Path) -> None:
    _data_dir, config_file = _project(invoke, tmp_path)
    assert invoke("checkpoint", "--config", str(config_file)).exit_code == EXIT_OK
    assert invoke("checkpoint", "--config", str(config_file)).exit_code == EXIT_OK
