"""Configuration loading and layering.

Precedence, lowest to highest:

1. packaged defaults (this module)
2. user config (``$XDG_CONFIG_HOME/reel-vault/config.yaml``)
3. workspace config (``reel-vault.yaml`` at the discovered root)
4. ``REEL_VAULT_*`` environment variables
5. explicit CLI overrides
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from reel_vault.config.models import AppConfig
from reel_vault.config.root import find_project_root, resolve_path

ENV_PREFIX = "REEL_VAULT_"
PROJECT_CONFIG_NAMES = ("reel-vault.yaml", "reel-vault.yml", "config/reel-vault.yaml")
USER_CONFIG_RELATIVE = Path("reel-vault") / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "./data/reels",
    "db_path": "./data/reels/state.db",
    "log_level": "INFO",
    "log_format": "text",
    "download": {
        "mode": "archive",
        "quality": "best",
        "keep_audio": True,
        "keep_keyframes": True,
        "verify": True,
        "retries": 1,
        "drive_wait_seconds": 60.0,
    },
    "enrich": {
        "retries": 2,
        "backoff_seconds": 5.0,
        "timeout_seconds": 30.0,
        "thumbnails": True,
        "max_media_items": 50,
    },
    "run": {
        "unattended": False,
        "max_hours": 0.0,
        "max_cost_usd": 5.0,
        "concurrency": 1,
        "request_delay_seconds": 3.0,
        "rate_limit_cooldown_minutes": 30,
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "triage_model": "openai/gpt-oss-120b",
        "extraction_model": "openai/gpt-oss-120b",
        "whisper_model": "whisper-large-v3-turbo",
        "whisper_cost_per_hour": 0.04,
        "chat_input_usd_per_million": 0.15,
        "chat_output_usd_per_million": 0.60,
        "retry_backoff_seconds": 5.0,
        "timeout_seconds": 120.0,
        "max_retries": 4,
    },
    "triage": {
        "extract_threshold": 40,
        "needs_review_threshold": 60,
        "fallback": "standard",
        "max_input_chars": 6000,
    },
    "transcription": {
        "enabled": True,
        "language": None,
        "chunk_seconds": 300,
        "max_chunk_bytes": 24 * 1024 * 1024,
    },
    "ocr": {"enabled": True, "max_keyframes": 6},
}


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    data_dir: Path | None = None
    db_path: Path | None = None
    media_dir: Path | None = None
    log_level: str | None = None
    log_format: str | None = None


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def user_config_paths() -> list[Path]:
    env = os.environ
    home = Path.home()
    candidates: list[Path] = []

    def add(base: Path) -> None:
        candidate = base / USER_CONFIG_RELATIVE
        if candidate not in candidates:
            candidates.append(candidate)

    for variable in ("APPDATA", "XDG_CONFIG_HOME"):
        value = env.get(variable)
        if value:
            add(Path(value))
    add(home)
    add(home / "Library" / "Application Support")
    return candidates


def user_config_path() -> Path | None:
    for candidate in user_config_paths():
        if candidate.is_file():
            return candidate
    return None


def project_config_paths(start: Path | str | None = None) -> list[Path]:
    base = find_project_root(start)
    return [base / name for name in PROJECT_CONFIG_NAMES]


def discover_config_files(
    *, explicit: Path | str | None = None, project_dir: Path | str | None = None
) -> list[Path]:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return [path]

    found: list[Path] = []
    user = user_config_path()
    if user is not None:
        found.append(user)
    for candidate in project_config_paths(project_dir):
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path} is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def env_overrides(settings: EnvSettings) -> dict[str, Any]:
    values = settings.model_dump(exclude_none=True)
    return {
        key: (str(value) if isinstance(value, Path) else value) for key, value in values.items()
    }


def load_config(
    *,
    explicit_path: Path | str | None = None,
    project_dir: Path | str | None = None,
    env: EnvSettings | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    root = find_project_root(project_dir)
    sources: list[Path] = []
    sources.extend(discover_config_files(explicit=explicit_path, project_dir=root))

    merged: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    for path in sources:
        merged = deep_merge(merged, load_yaml(path))

    settings = env if env is not None else EnvSettings()
    merged = deep_merge(merged, env_overrides(settings))
    if overrides:
        merged = deep_merge(merged, overrides)

    data_dir = resolve_path(merged["data_dir"], base=root)
    db_path = resolve_path(merged["db_path"], base=root)
    media_raw = merged.get("media_dir")
    media_dir = resolve_path(media_raw, base=root) if media_raw else data_dir / "media"

    taxonomy_raw = merged.get("taxonomy_file")
    taxonomy_file: Path | None
    if taxonomy_raw:
        taxonomy_file = resolve_path(taxonomy_raw, base=root)
    else:
        candidate = root / "reel-vault-taxonomy.yaml"
        taxonomy_file = candidate if candidate.is_file() else None

    rules_raw = merged.get("archive_rules_file")
    archive_rules_file: Path | None
    if rules_raw:
        archive_rules_file = resolve_path(rules_raw, base=root)
    else:
        candidate = root / "reel-vault-archive-rules.yaml"
        archive_rules_file = candidate if candidate.is_file() else None

    merged.update(
        {
            "data_dir": str(data_dir),
            "db_path": str(db_path),
            "media_dir": str(media_dir),
            "taxonomy_file": str(taxonomy_file) if taxonomy_file else None,
            "archive_rules_file": str(archive_rules_file) if archive_rules_file else None,
            "source_files": [str(path) for path in sources],
        }
    )

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


__all__ = [
    "DEFAULT_CONFIG",
    "ENV_PREFIX",
    "ConfigError",
    "EnvSettings",
    "deep_merge",
    "discover_config_files",
    "load_config",
    "load_yaml",
    "project_config_paths",
    "user_config_path",
]
