"""Transcription orchestration: audio, Whisper, retries and cost logging."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from reel_vault.config.models import AppConfig
from reel_vault.logging_setup import get_logger
from reel_vault.schemas.common import JobState, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.transcript import Transcript, TranscriptSegment
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_model_atomic
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.transcribe.audio import FfmpegAudioExtractor
from reel_vault.transcribe.base import (
    AudioExtractor,
    AudioSource,
    NoAudioError,
    TranscriptionError,
    WhisperClient,
)
from reel_vault.transcribe.groq import GroqWhisperClient
from reel_vault.transcribe.source import ReelMediaAudioSource

logger = get_logger(__name__)

TRANSCRIBE_STATUSES = frozenset({ReelStatus.FETCHED.value, ReelStatus.TRANSCRIPT_FAILED.value})


@dataclass
class TranscriptionSummary:
    processed: int = 0
    transcribed: int = 0
    no_audio: int = 0
    skipped: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    cost_usd: float = 0.0
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "transcribed": self.transcribed,
            "no_audio": self.no_audio,
            "skipped": self.skipped,
            "failed": self.failed,
            "audio_seconds": round(self.audio_seconds, 2),
            "cost_usd": round(self.cost_usd, 6),
            "failures": dict(sorted(self.failures.items())),
        }


class TranscriptionService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        audio_source: AudioSource | None = None,
        extractor: AudioExtractor | None = None,
        client: WhisperClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.model = model or config.groq.whisper_model
        self.language = language if language is not None else config.transcription.language
        self.audio_source = audio_source or ReelMediaAudioSource(
            paths, timeout_seconds=config.groq.timeout_seconds
        )
        self.extractor = extractor or FfmpegAudioExtractor()
        self.client = client or GroqWhisperClient(config.groq, sleep=sleep)

    def transcribe_pending(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
    ) -> TranscriptionSummary:
        summary = TranscriptionSummary()
        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            reels = [reel for reel in reels if str(reel["shortcode"]) in wanted]
        elif not force:
            reels = [reel for reel in reels if str(reel["status"]) in TRANSCRIBE_STATUSES]
        if limit is not None:
            reels = reels[: max(0, limit)]

        if not self.config.transcription.enabled:
            summary.skipped = len(reels)
            return summary

        for reel in reels:
            shortcode = str(reel["shortcode"])
            transcript = self.transcribe_one(shortcode)
            summary.processed += 1
            if transcript.no_audio:
                summary.no_audio += 1
            elif transcript.error is not None:
                summary.failed += 1
                summary.failures[shortcode] = transcript.error
            else:
                summary.transcribed += 1
                summary.audio_seconds += transcript.duration_seconds
                summary.cost_usd += transcript.cost_usd
        return summary

    def transcribe_one(self, shortcode: str) -> Transcript:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise LookupError(f"unknown shortcode: {shortcode}")
        metadata = self.store.get_metadata(shortcode)
        if metadata is None or metadata.get("fetch_status") != "ok":
            transcript = Transcript(
                shortcode=shortcode,
                model=self.model,
                error="metadata is missing or the post is unavailable",
            )
            self._persist(shortcode, transcript)
            return transcript

        payload = {key: value for key, value in metadata.items() if key != "_row"}
        result = EnrichmentResult.model_validate(payload)
        self.store.set_job(shortcode, "transcribe", JobState.RUNNING)

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"transcribe-{shortcode}-", dir=self.paths.data_dir
            ) as temp:
                workdir = Path(temp)
                source = self.audio_source.prepare(result, workdir)
                extraction = self.extractor.extract(
                    source,
                    workdir,
                    chunk_seconds=self.config.transcription.chunk_seconds,
                    max_chunk_bytes=self.config.transcription.max_chunk_bytes,
                )
                transcript = self._call_whisper(
                    shortcode, extraction.chunks, extraction.duration_seconds
                )
        except NoAudioError as exc:
            logger.info("transcribe %s: no audio (%s)", shortcode, exc)
            transcript = Transcript(
                shortcode=shortcode,
                model=self.model,
                no_audio=True,
            )
        except TranscriptionError as exc:
            logger.warning("transcribe %s failed: %s", shortcode, exc)
            transcript = Transcript(
                shortcode=shortcode,
                model=self.model,
                error=str(exc),
            )
        except Exception as exc:
            logger.warning("transcribe %s crashed: %s", shortcode, exc)
            transcript = Transcript(
                shortcode=shortcode,
                model=self.model,
                error=f"unexpected transcription failure: {exc}",
            )

        self._persist(shortcode, transcript)
        return transcript

    def _call_whisper(
        self, shortcode: str, chunks: list[Path], total_duration: float
    ) -> Transcript:
        model = self.model
        language = self.language
        per_second = self.config.groq.whisper_cost_per_hour / 3600
        fallback_chunk_duration = total_duration / len(chunks) if chunks and total_duration else 0.0

        segments: list[TranscriptSegment] = []
        texts: list[str] = []
        detected_language: str | None = None
        offset = 0.0
        billed_seconds = 0.0

        for chunk in chunks:
            response = self.client.transcribe(chunk, model=model, language=language)
            chunk_duration = response.duration_seconds or fallback_chunk_duration
            for segment in response.segments:
                segments.append(
                    TranscriptSegment(
                        id=len(segments),
                        start=round(segment.start + offset, 3),
                        end=round(segment.end + offset, 3),
                        text=segment.text,
                    )
                )
            if not response.segments and response.text:
                segments.append(
                    TranscriptSegment(
                        id=len(segments),
                        start=round(offset, 3),
                        end=round(offset + chunk_duration, 3),
                        text=response.text,
                    )
                )
            if response.text:
                texts.append(response.text)
            detected_language = detected_language or response.language
            offset += chunk_duration
            billed_seconds += chunk_duration

        duration = total_duration or offset
        cost = round(billed_seconds * per_second, 6)
        transcript = Transcript(
            shortcode=shortcode,
            language=detected_language,
            model=model,
            segments=segments,
            text=" ".join(texts).strip(),
            duration_seconds=round(duration, 3),
            cost_usd=cost,
            chunk_count=len(chunks),
        )
        logger.info("transcribe %s: %.1fs audio, $%.6f", shortcode, duration, cost)
        return transcript

    def _persist(self, shortcode: str, transcript: Transcript) -> None:
        payload = transcript.model_dump(mode="json")
        self.store.save_transcript(shortcode, payload)
        write_model_atomic(self.paths.reel(shortcode).transcript, transcript)

        if transcript.error is not None:
            self.store.set_status(shortcode, ReelStatus.TRANSCRIPT_FAILED)
            self.store.set_job(shortcode, "transcribe", JobState.FAILED, error=transcript.error)
            return

        self.store.set_status(shortcode, ReelStatus.TRANSCRIBED)
        self.store.set_job(shortcode, "transcribe", JobState.DONE)
        if transcript.cost_usd > 0:
            self.store.record_cost(
                shortcode,
                "groq",
                transcript.model,
                units=transcript.duration_seconds,
                usd=transcript.cost_usd,
            )


__all__ = ["TranscriptionService", "TranscriptionSummary"]
