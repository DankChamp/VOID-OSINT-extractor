"""Typed ingestion errors (§4, §19).

Every failure mode the specification calls out has its own type so callers can react
precisely, and so the CLI can report what failed without inventing a cause. No error
path ever substitutes a different source or fabricates transcript content.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for all ingestion failures."""

    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        return f"{self.message}\nhint: {self.hint}" if self.hint else self.message


class InvalidVideoUrl(IngestError):
    """The supplied URL is not a recognisable YouTube video URL."""

    def __init__(self, url: str, hint: str | None = None) -> None:
        super().__init__(
            f"not a recognisable YouTube video URL: {url!r}",
            hint=hint
            or "expected forms: youtube.com/watch?v=<id>, youtu.be/<id>, "
            "/shorts/<id>, /embed/<id>, /live/<id>",
        )


class VideoUnavailable(IngestError):
    """The video does not exist, is private, or is region/age restricted."""

    def __init__(self, video_id: str, reason: str | None = None) -> None:
        super().__init__(
            f"video {video_id} is unavailable"
            + (f": {reason}" if reason else "")
            + (
                ". The original video stays the reference for resolving ambiguities; "
                "no substitute source is used."
            )
        )


class TranscriptUnavailable(IngestError):
    """No usable caption track exists for this video."""

    def __init__(self, video_id: str, *, reason: str | None = None) -> None:
        super().__init__(
            f"no usable transcript for video {video_id}" + (f": {reason}" if reason else ""),
            hint=(
                "captions may be disabled or absent. An optional speech-to-text path exists "
                "but is disabled by default and requires explicit opt-in "
                "(transcription.enabled in configuration, and audio you are authorised to "
                "process). No transcript content is ever fabricated."
            ),
        )


class TranscriptRetrievalFailed(IngestError):
    """The caption request failed for a reason that may be transient."""

    def __init__(self, video_id: str, reason: str) -> None:
        super().__init__(
            f"transcript retrieval failed for video {video_id}: {reason}",
            hint="this is usually rate limiting or a blocked request; retry later or use "
            "another provider network",
        )


class MetadataRetrievalFailed(IngestError):
    """Video metadata could not be retrieved and no partial result is trustworthy."""

    def __init__(self, video_id: str, reason: str) -> None:
        super().__init__(f"metadata retrieval failed for video {video_id}: {reason}")


class TranscriptionDisabled(IngestError):
    """Speech-to-text was requested but is not enabled."""

    def __init__(self, video_id: str) -> None:
        super().__init__(
            f"speech-to-text fallback requested for {video_id} but transcription is disabled",
            hint="enable it explicitly in configuration; it processes audio you must be "
            "authorised to use, so it stays off by default",
        )
