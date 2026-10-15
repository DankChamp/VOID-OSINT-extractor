"""Configuration subsystem: validated models, YAML layering and environment overrides."""

from video_intel.config.env_file import find_env_file, load_env_file, parse_env_text
from video_intel.config.loader import (
    ConfigError,
    EnvSettings,
    discover_config_files,
    load_config,
    load_yaml,
    project_config_paths,
    user_config_path,
)
from video_intel.config.models import (
    AppConfig,
    IngestSettings,
    ModeConfig,
    PipelineConfig,
    PresetConfig,
    ProviderConfig,
    TranscriptionSettings,
    VerificationSettings,
)

__all__ = [
    "AppConfig",
    "ConfigError",
    "EnvSettings",
    "IngestSettings",
    "ModeConfig",
    "PipelineConfig",
    "PresetConfig",
    "ProviderConfig",
    "TranscriptionSettings",
    "VerificationSettings",
    "discover_config_files",
    "find_env_file",
    "load_config",
    "load_env_file",
    "load_yaml",
    "parse_env_text",
    "project_config_paths",
    "user_config_path",
]
