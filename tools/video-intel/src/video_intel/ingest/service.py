"""Transcript ingestion service (§4).

Orchestrates metadata retrieval, caption acquisition, segment construction and
normalization. Nothing here invents content: if captions are missing the failure is
reported and an optional, explicitly enabled speech-to-text path is offered.
"""

from __future__ import annotations

from dataclasses import dataclass

from video_intel.ingest.errors import TranscriptUnavailable
from video_intel.ingest.normalize import TranscriptNormalizer
from video_intel.ingest.sources import (
    CaptionTrack,
    MetadataSource,
    TrackInfo,
    TranscriptSource,
)
from video_intel.ingest.youtube import ParsedVideo, parse_youtube_url
from video_intel.logging_setup import get_logger
from video_intel.schemas.common import TranscriptKind
from video_intel.schemas.transcript import (
    NormalizedTranscript,
    RawTranscript,
    TranscriptSegment,
    VideoMetadata,
    hash_transcript,
)

logger = get_logger(__name__)

SEGMENT_ID_TEMPLATE = "segment_{:04d}"


def segment_id_for(index: int) -> str:
    """Stable, zero-padded segment id used as the citation key everywhere."""
    return SEGMENT_ID_TEMPLATE.format(index)


def build_segments(track: CaptionTrack) -> list[TranscriptSegment]:
    """Convert caption cues into timestamped segments, preserving source timing.

    Cue timing is copied, never recomputed. Cues with no words cannot be represented
    as segments (a segment must carry text), so they are skipped and counted; the
    count is recorded in the transcript's provenance so nothing disappears silently.
    """
    cues = [snippet for snippet in track.snippets if snippet.text.strip()]
    segments = [
        TranscriptSegment(
            segment_id=segment_id_for(index),
            start_seconds=round(snippet.start, 3),
            end_seconds=round(snippet.end, 3),
            text=snippet.text.strip(),
        )
        for index, snippet in enumerate(cues, start=1)
    ]
    if not segments:
        raise TranscriptUnavailable(track.video_id, reason="every caption cue was empty")
    return segments


def empty_cue_count(track: CaptionTrack, segments: list[TranscriptSegment]) -> int:
    """Number of source cues that carried no text."""
    return max(0, len(track.snippets) - len(segments))


@dataclass(frozen=True)
class IngestRequest:
    """One ingestion request."""

    url: str
    languages: tuple[str, ...] = ("en",)

    def parsed(self) -> ParsedVideo:
        return parse_youtube_url(self.url)


@dataclass(frozen=True)
class IngestResult:
    """Everything produced by a successful ingestion."""

    video: ParsedVideo
    metadata: VideoMetadata
    transcript: RawTranscript
    tracks_available: tuple[TrackInfo, ...]
    selected_track: TrackInfo

    @property
    def transcript_hash(self) -> str:
        return hash_transcript(self.transcript.text())

    @property
    def segment_count(self) -> int:
        return len(self.transcript.segments)

    @property
    def duration_seconds(self) -> float:
        if self.metadata.duration_seconds is not None:
            return self.metadata.duration_seconds
        return self.transcript.segments[-1].end_seconds


class TranscriptIngestor:
    """Builds transcripts from injected sources.

    Use it as a context manager (``with TranscriptIngestor(...) as ingestor:``) so
    network-backed sources release their connections.
    """

    def __init__(
        self,
        transcript_source: TranscriptSource,
        metadata_source: MetadataSource,
        *,
        normalizer: TranscriptNormalizer | None = None,
    ) -> None:
        self.transcript_source = transcript_source
        self.metadata_source = metadata_source
        self.normalizer = normalizer or TranscriptNormalizer()

    def __enter__(self) -> TranscriptIngestor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any source that owns resources (HTTP clients, sessions)."""
        for source in (self.transcript_source, self.metadata_source):
            closer = getattr(source, "close", None)
            if callable(closer):
                closer()

    # ------------------------------------------------------------------ metadata
    def fetch_metadata(self, video_id: str) -> VideoMetadata:
        return self.metadata_source.fetch(video_id)

    # ----------------------------------------------------------------- transcript
    def available_tracks(self, video_id: str) -> tuple[TrackInfo, ...]:
        return self.transcript_source.available_tracks(video_id)

    def select_track(
        self, video_id: str, tracks: tuple[TrackInfo, ...], languages: tuple[str, ...]
    ) -> TrackInfo:
        """Choose the best track: language priority first, manual before automatic."""
        if not tracks:
            raise TranscriptUnavailable(video_id, reason="no caption tracks are published")

        wanted = [code.lower() for code in languages if code]
        available = sorted({track.language_code.lower() for track in tracks})
        for code in wanted:
            candidates = [track for track in tracks if track.language_code.lower() == code]
            if candidates:
                return next((t for t in candidates if not t.is_generated), candidates[0])
        if wanted:
            raise TranscriptUnavailable(
                video_id,
                reason=(
                    f"no track in requested languages {', '.join(wanted)}; "
                    f"available: {', '.join(available) or 'none'}"
                ),
            )
        manual = [track for track in tracks if not track.is_generated]
        return (manual or list(tracks))[0]

    def fetch_transcript(
        self, video_id: str, languages: tuple[str, ...]
    ) -> tuple[RawTranscript, TrackInfo]:
        """Fetch captions and build the raw transcript plus the track description."""
        tracks = self.available_tracks(video_id)
        selected = self.select_track(video_id, tracks, languages)
        track = self.transcript_source.fetch(video_id, (selected.language_code,))

        segments = build_segments(track)
        transcript = RawTranscript(
            video_id=video_id,
            language=track.language or selected.language,
            detected_language=track.language_code or selected.language_code,
            kind=TranscriptKind.AUTOMATIC if track.is_generated else TranscriptKind.MANUAL,
            segments=segments,
            source=track.source,
            source_detail={
                "language_code": track.language_code or selected.language_code,
                "is_generated": str(track.is_generated).lower(),
                "available_languages": ",".join(sorted({t.language_code for t in tracks})),
                "cue_count": str(len(track.snippets)),
                "empty_cues_dropped": str(empty_cue_count(track, segments)),
            },
        )
        logger.info(
            "video=%s segments=%d language=%s kind=%s",
            video_id,
            len(segments),
            transcript.detected_language,
            transcript.kind.value,
        )
        return transcript, selected

    # --------------------------------------------------------------------- run
    def ingest(
        self, request: IngestRequest, *, metadata: VideoMetadata | None = None
    ) -> IngestResult:
        """Run metadata and transcript acquisition for one request."""
        video = request.parsed()
        metadata = metadata if metadata is not None else self.fetch_metadata(video.video_id)
        tracks = self.available_tracks(video.video_id)
        transcript, selected = self.fetch_transcript(video.video_id, request.languages)
        return IngestResult(
            video=video,
            metadata=metadata,
            transcript=transcript,
            tracks_available=tracks,
            selected_track=selected,
        )

    def normalize(self, transcript: RawTranscript) -> NormalizedTranscript:
        """Apply the normalization rules, keeping raw text and timing intact."""
        return self.normalizer.normalize(transcript)


__all__ = [
    "IngestRequest",
    "IngestResult",
    "TranscriptIngestor",
    "build_segments",
    "empty_cue_count",
    "segment_id_for",
]
