"""Configuration models.

Secrets are never stored in configuration files. A provider entry names the environment
variable that holds its key; the value is read at call time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DownloadMode = Literal["none", "archive", "all"]
Quality = Literal["best", "1080", "720"]
FallbackDepth = Literal["light", "standard", "full"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DownloadConfig(_Model):
    mode: DownloadMode = "archive"
    quality: Quality = "best"
    keep_audio: bool = True
    keep_keyframes: bool = True
    verify: bool = True
    retries: int = Field(default=1, ge=0, le=5)
    drive_wait_seconds: float = Field(default=60.0, ge=0.0)


class RunConfig(_Model):
    unattended: bool = False
    max_hours: float = 0.0
    max_cost_usd: float = 5.0
    concurrency: int = Field(default=1, ge=1, le=8)
    request_delay_seconds: float = Field(default=3.0, ge=0.0)
    rate_limit_cooldown_minutes: int = Field(default=30, ge=0)


class EnrichConfig(_Model):
    retries: int = Field(default=2, ge=0, le=10)
    backoff_seconds: float = Field(default=5.0, ge=0.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    thumbnails: bool = True
    max_media_items: int = Field(default=50, ge=1, le=500)


class GroqConfig(_Model):
    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    triage_model: str = "openai/gpt-oss-120b"
    extraction_model: str = "openai/gpt-oss-120b"
    whisper_model: str = "whisper-large-v3-turbo"
    whisper_cost_per_hour: float = Field(default=0.04, ge=0.0)
    retry_backoff_seconds: float = Field(default=5.0, ge=0.0)
    timeout_seconds: float = 120.0
    max_retries: int = Field(default=4, ge=0, le=10)


class TriageConfig(_Model):
    extract_threshold: int = Field(default=40, ge=0, le=100)
    needs_review_threshold: int = Field(default=60, ge=0, le=100)
    fallback: FallbackDepth = "standard"


class TranscriptionConfig(_Model):
    enabled: bool = True
    language: str | None = None
    chunk_seconds: int = Field(default=300, ge=10, le=3600)
    max_chunk_bytes: int = Field(default=24 * 1024 * 1024, ge=1024 * 1024)


class OcrConfig(_Model):
    enabled: bool = True
    max_keyframes: int = Field(default=6, ge=1, le=30)


class AppConfig(_Model):
    data_dir: Path
    db_path: Path
    media_dir: Path
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    taxonomy_file: Path | None = None
    archive_rules_file: Path | None = None
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    source_files: list[str] = Field(default_factory=list)

    def redaction_safe_dump(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"source_files"})


__all__ = [
    "AppConfig",
    "DownloadConfig",
    "DownloadMode",
    "EnrichConfig",
    "FallbackDepth",
    "GroqConfig",
    "OcrConfig",
    "Quality",
    "RunConfig",
    "TranscriptionConfig",
    "TriageConfig",
]
