"""Phase 5: audio extraction and Groq Whisper transcription."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.schemas.common import ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.source import SourceReel
from reel_vault.schemas.transcript import Transcript, TranscriptSegment
from reel_vault.storage.db import Store
from reel_vault.storage.io import read_model
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.transcribe.audio import FfmpegAudioExtractor
from reel_vault.transcribe.base import (
    AudioExtraction,
    NoAudioError,
    TranscriptionError,
    WhisperResult,
)
from reel_vault.transcribe.groq import GroqWhisperClient, parse_response
from reel_vault.transcribe.service import TranscriptionService

_STORES: list[Store] = []


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while _STORES:
        _STORES.pop().close()


class FakeSource:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, Path]] = []

    def prepare(self, result: EnrichmentResult, workdir: Path) -> Path:
        self.calls.append((result.shortcode, workdir))
        if self.error is not None:
            raise self.error
        source = workdir / "source.m4a"
        source.write_bytes(b"audio")
        return source


class FakeExtractor:
    def __init__(
        self,
        *,
        duration: float = 120.0,
        chunks: int = 1,
        error: Exception | None = None,
    ) -> None:
        self.duration = duration
        self.chunk_count = chunks
        self.error = error
        self.calls: list[tuple[Path, int, int]] = []

    def extract(
        self, source: Path, workdir: Path, *, chunk_seconds: int, max_chunk_bytes: int
    ) -> AudioExtraction:
        self.calls.append((source, chunk_seconds, max_chunk_bytes))
        if self.error is not None:
            raise self.error
        chunks = []
        for index in range(self.chunk_count):
            chunk = workdir / f"chunk-{index:03d}.m4a"
            chunk.write_bytes(b"chunk")
            chunks.append(chunk)
        return AudioExtraction(chunks=chunks, duration_seconds=self.duration)


class FakeClient:
    def __init__(self, *, error: Exception | None = None, duration: float = 60.0) -> None:
        self.error = error
        self.duration = duration
        self.calls: list[tuple[Path, str, str | None]] = []

    def transcribe(self, audio: Path, *, model: str, language: str | None = None) -> WhisperResult:
        self.calls.append((audio, model, language))
        if self.error is not None:
            raise self.error
        return WhisperResult(
            text=f"words from {audio.name}",
            language="en",
            segments=[TranscriptSegment(id=0, start=0.0, end=10.0, text="hello")],
            duration_seconds=self.duration,
        )


def _workspace(
    tmp_path: Path, codes: list[str], extra: dict[str, Any] | None = None
) -> tuple[Store, WorkspacePaths, Any]:
    overrides: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "db_path": str(tmp_path / "data" / "state.db"),
        "media_dir": str(tmp_path / "media"),
        "run": {"request_delay_seconds": 0.0},
        "groq": {"retry_backoff_seconds": 0.0, "max_retries": 1},
        "transcription": {"chunk_seconds": 300, "max_chunk_bytes": 24 * 1024 * 1024},
    }
    if extra:
        overrides.update(extra)
    config = load_config(overrides=overrides)
    paths = WorkspacePaths(config.data_dir, config.db_path, config.media_dir).ensure()
    store = Store(paths.database)
    _STORES.append(store)
    report = ImportReport(source=str(tmp_path / "export"))
    report.reels = [
        SourceReel(
            shortcode=code,
            url=f"https://www.instagram.com/reel/{code}/",
            source_import=report.imported_at,
        )
        for code in codes
    ]
    store.import_report(report)
    for code in codes:
        store.set_status(code, ReelStatus.FETCHED)
    return store, paths, config


def _seed_metadata(store: Store, shortcode: str) -> None:
    result = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
    )
    store.save_metadata(shortcode, result.model_dump(mode="json"), "ok")


def _service(
    store: Store,
    paths: WorkspacePaths,
    config: Any,
    *,
    source: FakeSource | None = None,
    extractor: FakeExtractor | None = None,
    client: FakeClient | None = None,
) -> TranscriptionService:
    return TranscriptionService(
        store,
        paths,
        config,
        audio_source=source or FakeSource(),
        extractor=extractor or FakeExtractor(),
        client=client or FakeClient(),
        sleep=lambda _: None,
    )


def test_transcription_success_persists_segments_and_cost(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    client = FakeClient(duration=120.0)
    service = _service(store, paths, config, client=client)

    summary = service.transcribe_pending()

    assert summary.transcribed == 1
    assert summary.audio_seconds == 120.0
    expected_cost = round(120.0 * 0.04 / 3600, 6)
    assert summary.cost_usd == expected_cost
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRANSCRIBED.value  # type: ignore[index]

    record = store.get_transcript("AAA111")
    assert record is not None
    assert record["text"] == "words from chunk-000.m4a"
    assert record["segments"][0]["text"] == "hello"
    assert record["chunk_count"] == 1

    artifact = read_model(paths.reel("AAA111").transcript, Transcript)
    assert artifact.model == config.groq.whisper_model
    assert artifact.cost_usd == expected_cost
    assert store.total_cost() == expected_cost
    job = store.get_job("AAA111", "transcribe")
    assert job is not None and job["state"] == "done"


def test_multiple_chunks_are_offset_and_reindexed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    extractor = FakeExtractor(duration=120.0, chunks=2)
    client = FakeClient(duration=60.0)
    service = _service(store, paths, config, extractor=extractor, client=client)

    transcript = service.transcribe_one("AAA111")

    assert transcript.chunk_count == 2
    assert [segment.id for segment in transcript.segments] == [0, 1]
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[1].start == 60.0
    assert transcript.text == "words from chunk-000.m4a words from chunk-001.m4a"
    assert len(client.calls) == 2


def test_no_audio_is_skipped_cleanly(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    extractor = FakeExtractor(error=NoAudioError("no audio track"))
    service = _service(store, paths, config, extractor=extractor)

    summary = service.transcribe_pending()

    assert summary.no_audio == 1
    assert summary.failed == 0
    assert summary.cost_usd == 0.0
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRANSCRIBED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").transcript, Transcript)
    assert artifact.no_audio is True
    assert artifact.error is None


def test_client_failure_marks_transcript_failed_and_can_retry(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    client = FakeClient(error=TranscriptionError("Groq is down"))
    service = _service(store, paths, config, client=client)

    summary = service.transcribe_pending()

    assert summary.failed == 1
    assert "Groq is down" in summary.failures["AAA111"]
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRANSCRIPT_FAILED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").transcript, Transcript)
    assert artifact.error == "Groq is down"
    job = store.get_job("AAA111", "transcribe")
    assert job is not None and job["state"] == "failed"

    client.error = None
    assert service.transcribe_pending().transcribed == 1
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRANSCRIBED.value  # type: ignore[index]


def test_finished_reels_are_skipped_unless_forced(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    service = _service(store, paths, config)

    assert service.transcribe_pending().transcribed == 1
    assert service.transcribe_pending().processed == 0
    assert service.transcribe_pending(force=True).transcribed == 1


def test_missing_metadata_marks_transcript_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    service = _service(store, paths, config)

    summary = service.transcribe_pending()

    assert summary.failed == 1
    assert "metadata" in summary.failures["AAA111"]
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRANSCRIPT_FAILED.value  # type: ignore[index]


def test_disabled_transcription_skips(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"], {"transcription": {"enabled": False}})
    _seed_metadata(store, "AAA111")
    service = _service(store, paths, config)

    summary = service.transcribe_pending()

    assert summary.skipped == 1
    assert summary.processed == 0


def test_unknown_shortcode_raises(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, [])
    service = _service(store, paths, config)
    with pytest.raises(LookupError):
        service.transcribe_one("NOPE")


def test_parse_response_falls_back_to_segments() -> None:
    result = parse_response(
        {
            "language": "en",
            "duration": 3.5,
            "segments": [
                {"start": 0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 3.5, "text": "world"},
            ],
        }
    )
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration_seconds == 3.5
    assert [segment.id for segment in result.segments] == [0, 1]


def _groq_config() -> Any:
    return load_config(overrides={"groq": {"retry_backoff_seconds": 1.0, "max_retries": 2}})


def test_groq_client_retries_429_with_backoff(tmp_path: Path) -> None:
    config = _groq_config()
    audio = tmp_path / "chunk.m4a"
    audio.write_bytes(b"audio")
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "text": "transcribed",
                "language": "en",
                "duration": 12.0,
                "segments": [{"start": 0, "end": 12, "text": "transcribed"}],
            },
        )

    client = GroqWhisperClient(
        config.groq,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    result = client.transcribe(audio, model="whisper-large-v3-turbo")

    assert result.text == "transcribed"
    assert calls["n"] == 2
    assert sleeps == [1.0]


def test_groq_client_does_not_retry_400(tmp_path: Path) -> None:
    config = _groq_config()
    audio = tmp_path / "chunk.m4a"
    audio.write_bytes(b"audio")
    sleeps: list[float] = []

    client = GroqWhisperClient(
        config.groq,
        api_key="test-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text="bad file")),
        sleep=sleeps.append,
    )
    with pytest.raises(TranscriptionError, match="400"):
        client.transcribe(audio, model="whisper-large-v3-turbo")
    assert sleeps == []


def test_groq_client_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = _groq_config()
    client = GroqWhisperClient(config.groq)
    with pytest.raises(TranscriptionError, match="GROQ_API_KEY"):
        client.transcribe(tmp_path / "chunk.m4a", model="whisper-large-v3-turbo")


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_extracts_and_splits_real_audio(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )

    extraction = FfmpegAudioExtractor().extract(
        source, tmp_path / "work", chunk_seconds=1, max_chunk_bytes=24 * 1024 * 1024
    )

    assert extraction.has_audio is True
    assert extraction.duration_seconds >= 2.9
    assert len(extraction.chunks) >= 2
    assert all(chunk.stat().st_size > 0 for chunk in extraction.chunks)
