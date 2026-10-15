"""Synthetic sources used by tests.

These fakes stand in for YouTube: they return fixed caption cues and metadata, or the
typed errors a real failure would produce. No test in this suite touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from video_intel.ingest.errors import (
    MetadataRetrievalFailed,
    TranscriptRetrievalFailed,
    TranscriptUnavailable,
    VideoUnavailable,
)
from video_intel.ingest.sources import CaptionSnippet, CaptionTrack, TrackInfo
from video_intel.providers.base import ProviderError, ProviderRequest, ProviderResponse
from video_intel.schemas.common import ProviderKind
from video_intel.schemas.extraction import ModelRef
from video_intel.schemas.transcript import VideoMetadata

CUE_TEXTS: tuple[tuple[str, float, float], ...] = (
    ("Welcome to the channel.", 0.0, 3.2),
    ("Today we discuss  risk\u00a0assessment .", 3.2, 6.0),
    ("", 6.0, 6.4),
    ("[Music]", 6.4, 8.0),
    ("Risk assessment evaluates likelihood and impact.", 8.0, 12.5),
)


@dataclass
class FakeTranscriptSource:
    """In-memory caption source."""

    tracks: tuple[TrackInfo, ...] = (
        TrackInfo(language="English (auto-generated)", language_code="en", is_generated=True),
    )
    snippets: tuple[CaptionSnippet, ...] = field(
        default_factory=lambda: tuple(
            CaptionSnippet(text=text, start=start, duration=end - start)
            for text, start, end in CUE_TEXTS
        )
    )
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    failure: Exception | None = None

    def available_tracks(self, video_id: str) -> tuple[TrackInfo, ...]:
        if self.failure is not None:
            raise self.failure
        return self.tracks

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> CaptionTrack:
        self.calls.append((video_id, languages))
        if self.failure is not None:
            raise self.failure
        return CaptionTrack(
            video_id=video_id,
            language="English",
            language_code=languages[0] if languages else "en",
            is_generated=any(t.is_generated for t in self.tracks),
            snippets=self.snippets,
            source="fake",
        )


@dataclass
class FakeMetadataSource:
    """In-memory metadata source."""

    title: str = "Synthetic risk assessment lecture"
    channel: str = "Synthetic Channel"
    duration_seconds: float | None = 312.0
    available_languages: tuple[str, ...] = ("en",)
    failure: Exception | None = None

    def fetch(self, video_id: str) -> VideoMetadata:
        if self.failure is not None:
            raise self.failure
        return VideoMetadata(
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=self.title,
            channel=self.channel,
            duration_seconds=self.duration_seconds,
            available_languages=list(self.available_languages),
            source_detail={"metadata_source": "fake"},
        )


def transcript_unavailable(video_id: str = "synthetic01") -> TranscriptUnavailable:
    return TranscriptUnavailable(video_id, reason="captions are disabled")


def video_unavailable(video_id: str = "synthetic01") -> VideoUnavailable:
    return VideoUnavailable(video_id, reason="VideoUnplayable")


def retrieval_failed(video_id: str = "synthetic01") -> TranscriptRetrievalFailed:
    return TranscriptRetrievalFailed(video_id, reason="RequestBlocked")


def metadata_failed(video_id: str = "synthetic01") -> MetadataRetrievalFailed:
    return MetadataRetrievalFailed(video_id, "oEmbed request failed")


# --------------------------------------------------------------------- providers
@dataclass
class FakeProvider:
    """Scripted provider for tests: returns canned responses in call order."""

    name: str = "fake"
    kind: str = "ollama"
    local: bool = True
    responses: list[str] = field(default_factory=list)
    default_response: str = '{"items": []}'
    prompts: list[str] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)
    fail_times: int = 0

    def model_ref(self, model: str | None = None) -> ModelRef:
        return ModelRef(
            provider=ProviderKind.OLLAMA, provider_name=self.name, model=model or "fake-model"
        )

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.prompts.append(request.prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            error = self.errors.pop(0) if self.errors else ProviderError("scripted failure")
            raise error
        text = self.responses.pop(0) if self.responses else self.default_response
        return ProviderResponse(
            text=text, model=request.model, provider=self.name, provider_kind=self.kind
        )

    async def aclose(self) -> None:
        return None


def extraction_response_json(items: list[dict[str, object]]) -> str:
    """Well-formed contract payload for a fake model."""
    return json.dumps({"items": items})
