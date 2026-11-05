"""Groq Whisper client with timestamps, retries and cost accounting."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from reel_vault.config.models import GroqConfig
from reel_vault.schemas.transcript import TranscriptSegment
from reel_vault.transcribe.base import TranscriptionError, WhisperResult

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_ERROR_BODY = 500


def _segments(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw = payload.get("segments")
    if not isinstance(raw, list):
        return []
    segments: list[TranscriptSegment] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
        except (TypeError, ValueError):
            continue
        segments.append(TranscriptSegment(id=index, start=start, end=end, text=text.strip()))
    return segments


def parse_response(payload: dict[str, Any]) -> WhisperResult:
    segments = _segments(payload)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        text = " ".join(segment.text for segment in segments)
    language = payload.get("language")
    try:
        duration = float(payload.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return WhisperResult(
        text=text.strip(),
        language=language if isinstance(language, str) else None,
        segments=segments,
        duration_seconds=duration,
    )


class GroqWhisperClient:
    def __init__(
        self,
        config: GroqConfig,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.api_key = api_key or os.environ.get(config.api_key_env)
        self.transport = transport
        self.sleep = sleep

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/audio/transcriptions"

    def transcribe(self, audio: Path, *, model: str, language: str | None = None) -> WhisperResult:
        if not self.api_key:
            raise TranscriptionError(
                f"{self.config.api_key_env} is not set; transcription needs a Groq key"
            )
        if not audio.is_file():
            raise TranscriptionError(f"audio chunk is missing: {audio}")

        attempts = self.config.max_retries + 1
        last_error = "transcription failed"
        for attempt in range(1, attempts + 1):
            try:
                with (
                    httpx.Client(
                        timeout=self.config.timeout_seconds, transport=self.transport
                    ) as client,
                    audio.open("rb") as handle,
                ):
                    data = {
                        "model": model,
                        "response_format": "verbose_json",
                        "temperature": "0",
                    }
                    if language:
                        data["language"] = language
                    response = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files={"file": (audio.name, handle, "audio/mp4")},
                        data=data,
                    )
            except httpx.RequestError as exc:
                last_error = f"Groq request failed: {exc}"
                if attempt < attempts:
                    self.sleep(self.config.retry_backoff_seconds * (2.0 ** (attempt - 1)))
                    continue
                raise TranscriptionError(last_error) from exc

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise TranscriptionError("Groq returned malformed JSON") from exc
                if not isinstance(payload, dict):
                    raise TranscriptionError("Groq returned an unexpected payload")
                return parse_response(payload)

            body = response.text[:MAX_ERROR_BODY]
            last_error = f"Groq HTTP {response.status_code}: {body}"
            if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                self.sleep(self.config.retry_backoff_seconds * (2.0 ** (attempt - 1)))
                continue
            raise TranscriptionError(last_error)

        raise TranscriptionError(last_error)


__all__ = ["RETRYABLE_STATUS", "GroqWhisperClient", "parse_response"]
