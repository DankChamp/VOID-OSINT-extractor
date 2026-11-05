"""Transcription protocols and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.transcript import TranscriptSegment


class NoAudioError(RuntimeError):
    """The media has no audio track, so transcription is skipped."""


class TranscriptionError(RuntimeError):
    """Transcription failed; the reel is marked for a later retry."""


@dataclass
class AudioExtraction:
    chunks: list[Path]
    duration_seconds: float
    has_audio: bool = True


@dataclass
class WhisperResult:
    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    duration_seconds: float = 0.0


@runtime_checkable
class AudioExtractor(Protocol):
    def extract(
        self, source: Path, workdir: Path, *, chunk_seconds: int, max_chunk_bytes: int
    ) -> AudioExtraction:
        """Extract mono 16 kHz audio and split it under the provider limit."""
        ...


@runtime_checkable
class WhisperClient(Protocol):
    def transcribe(self, audio: Path, *, model: str, language: str | None = None) -> WhisperResult:
        """Transcribe one audio chunk with timestamps."""
        ...


@runtime_checkable
class AudioSource(Protocol):
    def prepare(self, result: EnrichmentResult, workdir: Path) -> Path:
        """Return a media file with audio, downloading audio-only if necessary."""
        ...


__all__ = [
    "AudioExtraction",
    "AudioExtractor",
    "AudioSource",
    "NoAudioError",
    "TranscriptionError",
    "WhisperClient",
    "WhisperResult",
]
