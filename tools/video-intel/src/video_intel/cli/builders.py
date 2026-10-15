"""Composition root: turn configuration into concrete ingestion objects.

Keeping construction here means the CLI stays thin and tests can build the same
objects from fakes without patching module internals.
"""

from __future__ import annotations

from video_intel.config.models import AppConfig
from video_intel.ingest.clients import HttpMetadataSource, YouTubeCaptionSource
from video_intel.ingest.service import TranscriptIngestor
from video_intel.ingest.sources import MetadataSource, TranscriptSource


def build_ingestor(
    config: AppConfig,
    *,
    transcript_source: TranscriptSource | None = None,
    metadata_source: MetadataSource | None = None,
) -> TranscriptIngestor:
    """Assemble the ingestor described by ``config``.

    Either source can be injected, which is how tests run the full pipeline without
    network access.
    """
    ingest_settings = config.ingest
    return TranscriptIngestor(
        transcript_source or YouTubeCaptionSource(proxy=ingest_settings.proxy),
        metadata_source
        or HttpMetadataSource(
            timeout=ingest_settings.request_timeout_seconds,
            use_watch_page=ingest_settings.use_watch_page_metadata,
        ),
    )


__all__ = ["build_ingestor"]
