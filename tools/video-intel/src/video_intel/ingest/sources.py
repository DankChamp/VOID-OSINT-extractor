"""Source abstractions for captions and metadata (§4).

Protocols, not concrete classes, so the pipeline depends on "a source of transcripts"
rather than on a scraping library. Unit tests inject synthetic sources; the CLI uses
the network-backed implementations in :mod:`video_intel.ingest.clients`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from video_intel.schemas.transcript import VideoMetadata


@dataclass(frozen=True)
class CaptionSnippet:
    """One timed caption cue exactly as the source reported it."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return max(0.0, self.start + max(0.0, self.duration))


@dataclass(frozen=True)
class TrackInfo:
    """Description of one available caption track (no content)."""

    language: str
    language_code: str
    is_generated: bool

    @property
    def kind(self) -> str:
        return "automatic" if self.is_generated else "manual"


@dataclass(frozen=True)
class CaptionTrack:
    """A fetched caption track with its cues and provenance."""

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: tuple[CaptionSnippet, ...]
    source: str

    @property
    def duration_seconds(self) -> float:
        return max((snippet.end for snippet in self.snippets), default=0.0)


@runtime_checkable
class TranscriptSource(Protocol):
    """Provides captions for a video."""

    def available_tracks(self, video_id: str) -> tuple[TrackInfo, ...]:
        """List caption tracks without downloading their content."""

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> CaptionTrack:
        """Fetch the best track matching ``languages`` in priority order.

        Raises:
            TranscriptUnavailable: when no matching track exists.
            TranscriptRetrievalFailed: when retrieval itself failed.
            VideoUnavailable: when the video cannot be read at all.
        """


@runtime_checkable
class MetadataSource(Protocol):
    """Provides video metadata."""

    def fetch(self, video_id: str) -> VideoMetadata:
        """Return metadata, leaving unknown fields as ``None`` rather than guessing.

        Raises:
            VideoUnavailable: when the video cannot be read at all.
            MetadataRetrievalFailed: when retrieval failed.
        """
