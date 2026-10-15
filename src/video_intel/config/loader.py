"""Configuration loading and layering (§17, §18).

Order of precedence, lowest to highest:

1. ``video_intel/config/defaults/config.yaml`` (packaged, version controlled)
2. user config (``$XDG_CONFIG_HOME/video-intel/config.yaml`` or ``~/.config/...``)
3. project config (``./video-intel.yaml`` or ``./config/video-intel.yaml``)
4. environment variables (``VIDEO_INTEL_*``)
5. explicit CLI overrides

The merged result is validated as an ``AppConfig``. Secrets are never read from files:
only the *name* of an environment variable is stored.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from video_intel.config.models import AppConfig
from video_intel.config.root import find_project_root, resolve_path
from video_intel.schemas.common import ExecutionMode

ENV_PREFIX = "VIDEO_INTEL_"
PACKAGED_CONFIG = Path(__file__).with_name("defaults") / "config.yaml"
PROJECT_CONFIG_NAMES = ("video-intel.yaml", "video-intel.yml", "config/video-intel.yaml")
USER_CONFIG_RELATIVE = Path("video-intel") / "config.yaml"


class EnvSettings(BaseSettings):
    """Environment-level settings. Only operational values, never API keys."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    data_dir: Path | None = None
    project_name: str | None = None
    execution_mode: ExecutionMode | None = None
    log_level: str | None = None
    log_format: str | None = None
    max_extraction_passes: int | None = None


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def user_config_paths() -> list[Path]:
    """Every location checked for a user config, in precedence order.

    Honours the native convention (``%APPDATA%`` on Windows, ``$XDG_CONFIG_HOME``
    elsewhere) and also probes the macOS location, so the user layer is not silently
    skipped on any of the three desktop platforms.
    """
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
    """The first existing user config file, or ``None`` when the user has none.

    Returning ``None`` rather than a synthesised path matters: an absent user layer must
    not be reported as an error, and ``config path`` should say so plainly.
    """
    for candidate in user_config_paths():
        if candidate.is_file():
            return candidate
    return None


def project_config_paths(start: Path | str | None = None) -> list[Path]:
    """Candidate project config files, resolved against the discovered project root.

    Anchoring to the root rather than the working directory means the configuration is
    found identically from the project root, from a subdirectory, and from a shell that
    was started somewhere else entirely.
    """
    base = find_project_root(start) if start is not None else find_project_root()
    return [base / name for name in PROJECT_CONFIG_NAMES]


def discover_config_files(
    *, explicit: Path | str | None = None, project_dir: Path | str | None = None
) -> list[Path]:
    """Return existing config files in precedence order (excluding packaged defaults)."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return [path]

    found: list[Path] = []
    user = user_config_path()
    if user is not None:
        found.append(user)
    for candidate in project_config_paths(Path(project_dir) if project_dir else None):
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping, tolerating a UTF-8 BOM.

    ``utf-8-sig`` accepts both BOM-prefixed and plain UTF-8. Windows editors (Notepad
    since Windows 10 1903) write a BOM by default, and without this a saved
    ``video-intel.yaml`` crashes the CLI with an uncaught ``UnicodeDecodeError``.
    """
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
    """Recursive dict merge. Mappings merge; lists and scalars are replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def env_overrides(settings: EnvSettings) -> dict[str, Any]:
    """Environment values as plain config keys (paths become strings)."""
    overrides = {
        key: value
        for key, value in settings.model_dump(exclude_none=True).items()
        if value is not None
    }
    return {
        key: (str(value) if isinstance(value, Path) else value) for key, value in overrides.items()
    }


def load_config(
    *,
    explicit_path: Path | str | None = None,
    project_dir: Path | str | None = None,
    env: EnvSettings | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Build the effective configuration.

    Raises:
        ConfigError: if a file is missing, malformed, or the merged result is invalid.
    """
    sources: list[Path] = [PACKAGED_CONFIG]
    sources.extend(
        discover_config_files(
            explicit=Path(explicit_path) if explicit_path is not None else None,
            project_dir=Path(project_dir) if project_dir is not None else None,
        )
    )

    merged: dict[str, Any] = {}
    for path in sources:
        merged = deep_merge(merged, load_yaml(path))

    settings = env if env is not None else EnvSettings()
    merged = deep_merge(merged, env_overrides(settings))
    if overrides:
        merged = deep_merge(merged, overrides)

    merged["source_files"] = [str(path) for path in sources]

    # A relative data_dir is interpreted against the discovered project root, never
    # against the working directory. Without this, running from a subdirectory silently
    # creates and reads a second, empty data tree.
    if isinstance(merged.get("data_dir"), (str, Path)):
        merged["data_dir"] = str(resolve_path(merged["data_dir"]))

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


def redacted_config(config: AppConfig) -> dict[str, Any]:
    """Config data that is safe to display or log."""
    return config.redaction_safe_dump()
