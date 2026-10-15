"""Configuration layering, validation, secret handling and cache hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
from video_intel.config.loader import (
    ConfigError,
    EnvSettings,
    deep_merge,
    discover_config_files,
    load_config,
    load_yaml,
)
from video_intel.config.models import AppConfig, ProviderConfig
from video_intel.schemas.common import ExecutionMode, OutputMode, PipelineId, ProviderKind


def _write_yaml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_defaults_load() -> None:
    config = load_config(project_dir=Path("/nonexistent"))
    assert config.project_name == "default"
    assert config.execution_mode is ExecutionMode.LOCAL_ONLY
    assert set(config.providers) >= {"ollama", "openai", "deepseek", "groq", "gemini"}
    assert set(config.pipelines) == set(PipelineId)
    assert config.modes[OutputMode.DEEP].pipelines
    assert "maximum-coverage" in config.presets


def test_default_pipelines_use_the_local_provider() -> None:
    config = load_config(project_dir=Path("/nonexistent"))
    for pipeline in config.enabled_pipelines():
        assert config.provider(pipeline.provider).is_local() is True


def test_provider_implementation_status_matches_the_registry() -> None:
    """Configuration may not claim an adapter exists when none is built."""
    from video_intel.providers.registry import IMPLEMENTED_KINDS

    config = load_config(project_dir=Path("/nonexistent"))
    for provider in config.providers.values():
        assert provider.implemented == (provider.kind in IMPLEMENTED_KINDS)


def test_project_config_overrides_packaged_defaults(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "video-intel.yaml",
        """
        project_name: research
        log_level: DEBUG
        chunking:
          max_chars: 8000
        """,
    )
    config = load_config(project_dir=tmp_path)
    assert config.project_name == "research"
    assert config.log_level == "DEBUG"
    assert config.chunking.max_chars == 8000
    assert config.chunking.overlap_chars == 600  # untouched default survives the merge


def test_user_config_has_lower_precedence_than_project_config(tmp_path: Path) -> None:
    config_home = tmp_path / "xdg"
    _write_yaml(config_home / "video-intel" / "config.yaml", "project_name: from-user\n")
    _write_yaml(tmp_path / "project" / "video-intel.yaml", "project_name: from-project\n")
    config = load_config(
        project_dir=tmp_path / "project",
        env=EnvSettings(_env_file=None),
    )
    assert config.project_name == "from-project"


def test_environment_overrides_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_yaml(tmp_path / "video-intel.yaml", "project_name: from-file\n")
    monkeypatch.setenv("VIDEO_INTEL_PROJECT_NAME", "from-env")
    monkeypatch.setenv("VIDEO_INTEL_LOG_FORMAT", "json")
    config = load_config(project_dir=tmp_path)
    assert config.project_name == "from-env"
    assert config.log_format == "json"


def test_explicit_overrides_win_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDEO_INTEL_PROJECT_NAME", "from-env")
    config = load_config(project_dir=tmp_path, overrides={"project_name": "from-cli"})
    assert config.project_name == "from-cli"


def test_string_paths_are_accepted(tmp_path: Path) -> None:
    """Callers pass plain strings from the CLI and the web layer."""
    path = _write_yaml(tmp_path / "video-intel.yaml", "project_name: from-string\n")
    assert load_config(explicit_path=str(path)).project_name == "from-string"
    assert load_config(project_dir=str(tmp_path)).project_name == "from-string"
    assert discover_config_files(explicit=str(path)) == [path]


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(explicit_path=tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported_clearly(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "video-intel.yaml", "project_name: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(project_dir=tmp_path, explicit_path=path)


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "video-intel.yaml", "not_a_real_setting: 1\n")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config(project_dir=tmp_path)


def test_invalid_chunking_is_rejected(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "video-intel.yaml", "chunking:\n  max_chars: 10\n")
    with pytest.raises(ConfigError):
        load_config(project_dir=tmp_path)


def test_local_only_refuses_remote_pipelines(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "video-intel.yaml",
        """
        execution_mode: local_only
        pipelines:
          fact:
            provider: openai
            model: gpt-4o-mini
        """,
    )
    with pytest.raises(ConfigError, match="local_only"):
        load_config(project_dir=tmp_path)


def test_hybrid_allows_remote_pipelines(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "video-intel.yaml",
        """
        execution_mode: hybrid
        pipelines:
          fact:
            provider: openai
            model: gpt-4o-mini
        """,
    )
    config = load_config(project_dir=tmp_path)
    assert config.execution_mode is ExecutionMode.HYBRID
    assert config.pipelines[PipelineId.FACT].provider == "openai"


def test_unknown_provider_lookup_lists_alternatives(config: AppConfig) -> None:
    with pytest.raises(KeyError, match="unknown provider 'nope'"):
        config.provider("nope")


def test_unknown_preset_lookup_lists_alternatives(config: AppConfig) -> None:
    with pytest.raises(KeyError, match="unknown pipeline preset"):
        config.preset("does-not-exist")


def test_api_key_is_read_from_environment_not_config(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig
) -> None:
    provider = config.provider("openai")
    assert provider.api_key_env == "OPENAI_API_KEY"
    assert provider.has_api_key() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value-123456")
    assert provider.has_api_key() is True
    key = provider.resolve_api_key()
    assert key is not None and key.get_secret_value() == "sk-test-value-123456"


def test_config_dump_never_contains_secret_values(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value-999")
    provider = config.provider("openai")
    key = provider.resolve_api_key()
    assert key is not None
    provider.api_key_env = "OPENAI_API_KEY"
    dumped = str(config.redaction_safe_dump())
    assert "sk-super-secret-value-999" not in dumped
    assert "OPENAI_API_KEY" in dumped
    assert config.hash() != ""


def test_redacted_dump_marks_key_presence(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-example-key-value")
    providers = config.redaction_safe_dump()["providers"]
    assert providers["groq"]["api_key_present"] is True
    assert providers["ollama"]["api_key_present"] is False


def test_config_hash_is_stable_across_loads(tmp_path: Path) -> None:
    first = load_config(project_dir=tmp_path)
    second = load_config(project_dir=tmp_path)
    assert first.hash() == second.hash()


def test_config_hash_changes_with_settings(tmp_path: Path) -> None:
    baseline = load_config(project_dir=tmp_path)
    changed = load_config(project_dir=tmp_path, overrides={"chunking": {"max_chars": 9000}})
    assert baseline.hash() != changed.hash()


def test_config_hash_ignores_the_source_file_list(tmp_path: Path) -> None:
    bare = load_config(project_dir=Path("/nonexistent"))
    with_file = load_config(project_dir=tmp_path)
    assert bare.hash() == with_file.hash()


def test_deep_merge_replaces_lists_and_merges_maps() -> None:
    merged = deep_merge(
        {"a": {"x": 1, "y": 2}, "list": [1, 2]},
        {"a": {"y": 3, "z": 4}, "list": [9]},
    )
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "list": [9]}


def test_load_yaml_accepts_empty_file(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "empty.yaml", "\n")
    assert load_yaml(path) == {}


def test_provider_config_defaults_model_to_first_entry() -> None:
    provider = ProviderConfig(name="demo", kind=ProviderKind.OPENAI_COMPATIBLE, models=["m1", "m2"])
    assert provider.default_model == "m1"


def test_source_files_are_recorded(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "video-intel.yaml", "project_name: traced\n")
    config = load_config(project_dir=tmp_path)
    assert any(path.endswith("defaults/config.yaml") for path in config.source_files)
    assert any(path.endswith("video-intel.yaml") for path in config.source_files)
