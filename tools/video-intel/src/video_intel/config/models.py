"""Configuration models (§17).

Configuration is layered: packaged defaults < user config file < project config <
environment variables < CLI overrides. Secrets are referenced by environment
variable name only; values are resolved at call time and never serialized.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from video_intel.schemas.chunk import ChunkingConfig
from video_intel.schemas.common import ExecutionMode, OutputMode, PipelineId, ProviderKind


class ProviderConfig(BaseModel):
    """One provider entry. Adapters are implemented in Phase 4."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    kind: ProviderKind
    enabled: bool = True
    base_url: str | None = None
    api_key_env: str | None = Field(
        default=None,
        description="Env var holding the key. The value itself is never stored here.",
    )
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    concurrency: int = Field(default=2, ge=1, le=32)
    local: bool = Field(default=False, description="True for on-device providers.")
    implemented: bool = Field(
        default=False,
        description="Adapter status. False means declared but not yet available.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.default_model is None and self.models:
            object.__setattr__(self, "default_model", self.models[0])
        return self

    def is_local(self) -> bool:
        return self.local or self.kind is ProviderKind.OLLAMA

    def has_api_key(self) -> bool:
        return bool(self.api_key_env and os.environ.get(self.api_key_env))

    def resolve_api_key(self) -> SecretStr | None:
        """Read the key from the environment at the moment it is needed."""
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        return SecretStr(value) if value else None

    def redacted(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": str(self.kind),
            "enabled": self.enabled,
            "local": self.is_local(),
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_present": self.has_api_key(),
            "models": self.models,
            "default_model": self.default_model,
            "adapter_implemented": self.implemented,
            "notes": self.notes,
        }


class PipelineConfig(BaseModel):
    """Binding of one extraction pipeline to a provider and model (§8)."""

    model_config = ConfigDict(extra="forbid")

    pipeline_id: PipelineId
    enabled: bool = True
    provider: str
    model: str
    passes: int = Field(default=1, ge=1, le=10)
    prompt_version: str = "1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    description: str = ""
    max_items_per_chunk: int = Field(default=40, ge=1, le=500)

    def cache_key(self) -> str:
        payload = (
            f"{self.pipeline_id}|{self.provider}|{self.model}|{self.passes}|{self.prompt_version}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ModeConfig(BaseModel):
    """Which pipelines and outputs a CLI mode activates."""

    model_config = ConfigDict(extra="forbid")

    mode: OutputMode
    pipelines: list[PipelineId] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    description: str = ""

    def is_default_pipeline(self, pipeline_id: PipelineId) -> bool:
        return pipeline_id in self.pipelines


class PresetConfig(BaseModel):
    """A named set of pipelines selected with ``--pipeline`` (§16)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    pipelines: list[PipelineId] = Field(min_length=1)
    mode: OutputMode | None = None
    description: str = ""

    def covers(self, pipeline_id: PipelineId) -> bool:
        return pipeline_id in self.pipelines


class IngestSettings(BaseModel):
    """Transcript acquisition preferences (§4)."""

    model_config = ConfigDict(extra="forbid")

    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="Caption languages in descending priority.",
    )
    prefer_manual_tracks: bool = Field(
        default=True, description="Prefer human captions over automatic ones at equal language."
    )
    use_watch_page_metadata: bool = True
    proxy: str | None = Field(default=None, description="Proxy for caption requests.")
    request_timeout_seconds: float = Field(default=20.0, gt=0)

    def language_tuple(self) -> tuple[str, ...]:
        return tuple(code.strip().lower() for code in self.languages if code.strip())


def _portable_path_label(value: Path | str) -> str:
    """Reduce a path to its final component, independent of separators and location.

    Both ``/home/alice/data`` and ``C:\\Users\\bob\\data`` become ``data``, which is what
    makes the configuration hash stable across machines and operating systems. The drive
    or root is intentionally discarded: it carries no behavioural meaning.
    """
    text = str(value).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] or text or "/"


class AppConfig(BaseModel):
    """The effective, validated configuration for a run."""

    model_config = ConfigDict(extra="forbid")

    config_version: str = "1.0"
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".video-intel")
    project_name: str = "default"
    execution_mode: ExecutionMode = ExecutionMode.LOCAL_ONLY
    log_level: str = "INFO"
    log_format: str = "text"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    pipelines: dict[PipelineId, PipelineConfig] = Field(default_factory=dict)
    modes: dict[OutputMode, ModeConfig] = Field(default_factory=dict)
    presets: dict[str, PresetConfig] = Field(default_factory=dict)
    max_extraction_passes: int = Field(default=2, ge=1, le=10)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    verification: VerificationSettings = Field(default_factory=lambda: VerificationSettings())
    transcription: TranscriptionSettings = Field(default_factory=lambda: TranscriptionSettings())
    source_files: list[str] = Field(
        default_factory=list, description="Config files merged, in order, for this result."
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_keys_as_ids(cls, data: Any) -> Any:
        """Let YAML use natural mappings: the key becomes the entry's id."""
        if not isinstance(data, dict):
            return data
        for field, id_field in (
            ("providers", "name"),
            ("pipelines", "pipeline_id"),
            ("modes", "mode"),
        ):
            section = data.get(field)
            if not isinstance(section, dict):
                continue
            filled = {
                key: (
                    {id_field: key, **value}
                    if isinstance(value, dict) and id_field not in value
                    else value
                )
                for key, value in section.items()
            }
            data[field] = filled
        return data

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        for name, provider in self.providers.items():
            if provider.name != name:
                raise ValueError(
                    f"provider key {name!r} does not match provider name {provider.name!r}"
                )
        for pipeline_id, pipeline in self.pipelines.items():
            if pipeline.pipeline_id != pipeline_id:
                raise ValueError(
                    f"pipeline key {pipeline_id.value!r} does not match "
                    f"pipeline_id {pipeline.pipeline_id.value!r}"
                )
        if self.execution_mode is ExecutionMode.LOCAL_ONLY:
            offenders = [
                pipeline.pipeline_id.value
                for pipeline in self.pipelines.values()
                if pipeline.enabled and not self._is_local(pipeline.provider)
            ]
            if offenders:
                raise ValueError(
                    "execution_mode=local_only but these pipelines use remote providers: "
                    + ", ".join(sorted(offenders))
                    + ". Set execution_mode to hybrid or cloud_only, or disable them."
                )
        for mode in self.modes.values():
            unknown = [p for p in mode.pipelines if p not in self.pipelines]
            if unknown:
                raise ValueError(
                    f"mode {mode.mode} references unknown pipelines: "
                    + ", ".join(sorted(p.value for p in unknown))
                )
        for preset in self.presets.values():
            unknown = [p for p in preset.pipelines if p not in self.pipelines]
            if unknown:
                raise ValueError(
                    f"preset {preset.name} references unknown pipelines: "
                    + ", ".join(sorted(p.value for p in unknown))
                )
            missing_providers = sorted(
                {
                    self.pipelines[p].provider
                    for p in preset.pipelines
                    if p in self.pipelines and self.pipelines[p].provider not in self.providers
                }
            )
            if missing_providers:
                raise ValueError(
                    f"preset {preset.name} references unconfigured providers: "
                    + ", ".join(missing_providers)
                )
        return self

    def _is_local(self, provider_name: str) -> bool:
        provider = self.providers.get(provider_name)
        return provider is not None and provider.is_local()

    def enabled_pipelines(self) -> list[PipelineConfig]:
        return [pipeline for pipeline in self.pipelines.values() if pipeline.enabled]

    def pipelines_for_mode(self, mode: OutputMode) -> list[PipelineConfig]:
        mode_config = self.modes.get(mode)
        if mode_config is None:
            return self.enabled_pipelines()
        return [
            self.pipelines[pipeline_id]
            for pipeline_id in mode_config.pipelines
            if pipeline_id in self.pipelines and self.pipelines[pipeline_id].enabled
        ]

    def outputs_for_mode(self, mode: OutputMode) -> list[str]:
        mode_config = self.modes.get(mode)
        return list(mode_config.outputs) if mode_config else []

    def preset(self, name: str) -> PresetConfig:
        try:
            return self.presets[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.presets)) or "none"
            raise KeyError(f"unknown pipeline preset {name!r}; available: {known}") from exc

    def pipelines_for_preset(self, name: str) -> list[PipelineConfig]:
        preset = self.preset(name)
        return [
            self.pipelines[pipeline_id]
            for pipeline_id in preset.pipelines
            if pipeline_id in self.pipelines
        ]

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.providers)) or "none"
            raise KeyError(f"unknown provider {name!r}; configured providers: {known}") from exc

    def redaction_safe_dump(self) -> dict[str, Any]:
        """Configuration as plain data, safe to print or log.

        Intended for humans: it carries absolute paths, because those are what a user
        needs to see in ``config show``. For cache keys use :meth:`hash`, which
        deliberately strips location-dependent values.
        """
        return {
            "config_version": self.config_version,
            "data_dir": str(self.data_dir),
            "project_name": self.project_name,
            "execution_mode": str(self.execution_mode),
            "log_level": self.log_level,
            "log_format": self.log_format,
            "chunking": self.chunking.model_dump(),
            "providers": {
                name: provider.redacted() for name, provider in sorted(self.providers.items())
            },
            "pipelines": {
                pipeline_id.value: pipeline.model_dump(mode="json")
                for pipeline_id, pipeline in sorted(
                    self.pipelines.items(), key=lambda item: item[0].value
                )
            },
            "modes": {
                mode.value: mode_config.model_dump(mode="json")
                for mode, mode_config in sorted(self.modes.items(), key=lambda kv: kv[0].value)
            },
            "presets": {
                name: preset.model_dump(mode="json")
                for name, preset in sorted(self.presets.items())
            },
            "max_extraction_passes": self.max_extraction_passes,
            "ingest": self.ingest.model_dump(mode="json"),
            "verification": self.verification.model_dump(mode="json"),
            "transcription": self.transcription.model_dump(mode="json"),
            "source_files": self.source_files,
        }

    def portable_dump(self) -> dict[str, Any]:
        """Location-independent view of the configuration.

        The cache key must survive the project directory being moved, renamed, copied to
        another device, or checked out under a different user account. Absolute paths do
        not: ``/home/alice/code/video-intel`` and ``C:\\Users\\bob\\video-intel`` describe
        the same configuration but would otherwise hash differently, silently invalidating
        the transcript, chunk and dataset caches for every new machine.

        Paths are therefore reduced to a stable label - the final path component, plus the
        index for ordering - while every behavioural setting is preserved exactly.
        """
        data = self.redaction_safe_dump()
        data["data_dir"] = _portable_path_label(self.data_dir)
        data["source_files"] = [
            f"{index}:{_portable_path_label(path)}" for index, path in enumerate(self.source_files)
        ]
        return data

    def hash(self) -> str:
        """Stable, location-independent hash of the effective configuration."""
        payload = json.dumps(self.portable_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VerificationSettings(BaseModel):
    """Evidence verification behaviour (§10)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    deterministic_checks: bool = True
    verifier_provider: str | None = None
    verifier_model: str | None = None
    context_window_segments: int = Field(default=1, ge=0, le=10)
    drop_unsupported_from_outputs: bool = True
    flag_unsupported_for_review: bool = True


class TranscriptionSettings(BaseModel):
    """Optional speech-to-text fallback (§4). Disabled by default: it needs an
    explicit opt-in because it downloads and processes audio the user must be
    authorized to process."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: str | None = None
    model: str | None = None
    language: str | None = None
    require_explicit_opt_in: bool = True
