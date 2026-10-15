"""Transcript ingestion: URL validation, caption and metadata sources, normalization."""

from video_intel.ingest.clients import HttpMetadataSource, YouTubeCaptionSource
from video_intel.ingest.errors import (
    IngestError,
    InvalidVideoUrl,
    MetadataRetrievalFailed,
    TranscriptionDisabled,
    TranscriptRetrievalFailed,
    TranscriptUnavailable,
    VideoUnavailable,
)
from video_intel.ingest.normalize import (
    DEFAULT_RULES,
    NormalizationRule,
    TranscriptNormalizer,
)
from video_intel.ingest.service import (
    IngestRequest,
    IngestResult,
    TranscriptIngestor,
    build_segments,
    segment_id_for,
)
from video_intel.ingest.sources import (
    CaptionSnippet,
    CaptionTrack,
    MetadataSource,
    TrackInfo,
    TranscriptSource,
)
from video_intel.ingest.youtube import ParsedVideo, extract_video_id, parse_youtube_url

__all__ = [
    "DEFAULT_RULES",
    "CaptionSnippet",
    "CaptionTrack",
    "HttpMetadataSource",
    "IngestError",
    "IngestRequest",
    "IngestResult",
    "InvalidVideoUrl",
    "MetadataRetrievalFailed",
    "MetadataSource",
    "NormalizationRule",
    "ParsedVideo",
    "TrackInfo",
    "TranscriptIngestor",
    "TranscriptNormalizer",
    "TranscriptRetrievalFailed",
    "TranscriptSource",
    "TranscriptUnavailable",
    "TranscriptionDisabled",
    "VideoUnavailable",
    "YouTubeCaptionSource",
    "build_segments",
    "extract_video_id",
    "parse_youtube_url",
    "segment_id_for",
]
