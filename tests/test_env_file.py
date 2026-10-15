"""``.env`` loading for provider credentials (§17)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_intel.config.env_file import find_env_file, load_env_file, parse_env_text


def test_parses_plain_assignments() -> None:
    parsed = parse_env_text("GROQ_API_KEY=gsk-123\nOTHER=value\n")
    assert parsed == {"GROQ_API_KEY": "gsk-123", "OTHER": "value"}


def test_parses_export_prefix_and_quotes() -> None:
    parsed = parse_env_text("export A=\"quoted value\"\nB='single'\n")
    assert parsed == {"A": "quoted value", "B": "single"}


def test_ignores_comments_and_blank_lines() -> None:
    parsed = parse_env_text("# comment\n\n  \nKEY=value  # trailing\n")
    assert parsed == {"KEY": "value  # trailing"}


def test_ignores_lines_without_equals() -> None:
    assert parse_env_text("just text\nKEY=v\n") == {"KEY": "v"}


def test_values_may_contain_equals() -> None:
    assert parse_env_text("KEY=a=b=c\n") == {"KEY": "a=b=c"}


def test_empty_value_is_allowed() -> None:
    assert parse_env_text("KEY=\n") == {"KEY": ""}


def test_load_fills_missing_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")

    applied = load_env_file()
    assert applied == {"GROQ_API_KEY": "from-file"}
    assert os.environ["GROQ_API_KEY"] == "from-file"


def test_real_environment_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")

    assert load_env_file() == {}
    assert os.environ["GROQ_API_KEY"] == "from-shell"


def test_explicit_env_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDEO_INTEL_ENV_FILE", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    custom = tmp_path / "secrets.env"
    custom.write_text("GROQ_API_KEY=explicit\n", encoding="utf-8")
    monkeypatch.setenv("VIDEO_INTEL_ENV_FILE", str(custom))

    assert find_env_file() == custom
    assert load_env_file() == {"GROQ_API_KEY": "explicit"}


def test_missing_file_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIDEO_INTEL_ENV_FILE", raising=False)
    assert find_env_file() is None
    assert load_env_file() == {}


def test_unreadable_file_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIDEO_INTEL_ENV_FILE", raising=False)
    env_file = tmp_path / ".env"
    env_file.mkdir()  # a directory where a file is expected
    assert load_env_file() == {}


def test_cli_reports_the_key_as_present_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a key in .env is visible to config, and never echoed."""
    import json

    from typer.testing import CliRunner

    from video_intel.cli.main import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-super-secret-value-123\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["providers", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    groq = next(entry for entry in payload if entry["name"] == "groq")
    assert groq["api_key_present"] is True
    assert "gsk-super-secret-value-123" not in result.stdout
