"""Real clients are exercised offline: mocked HTTP, real library exception types."""

from __future__ import annotations

import json

import httpx
import pytest
from video_intel.ingest.clients import HttpMetadataSource, YouTubeCaptionSource
from video_intel.ingest.errors import (
    MetadataRetrievalFailed,
    TranscriptRetrievalFailed,
    TranscriptUnavailable,
    VideoUnavailable,
)

VIDEO_ID = "synthetic01"

OEMBED = {
    "title": "Risk assessment explained",
    "author_name": "Synthetic Channel",
    "author_url": "https://www.youtube.com/channel/UC1234567890",
}
WATCH_HTML = (
    "<html><head><title>Risk assessment explained - YouTube</title></head>"
    "<body><script>var ytInitialPlayerResponse = "
    + json.dumps(
        {
            "videoDetails": {
                "title": "Risk assessment explained",
                "author": "Synthetic Channel",
                "channelId": "UC1234567890",
                "lengthSeconds": "312",
            },
            "microformat": {"playerMicroformatRenderer": {"uploadDate": "2026-01-15"}},
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {"languageCode": "en", "kind": "asr"},
                        {"languageCode": "es"},
                    ]
                }
            },
        }
    )
    + ";</script></body></html>"
)


def _client(handler) -> HttpMetadataSource:
    return HttpMetadataSource(httpx.Client(transport=httpx.MockTransport(handler)))


# ------------------------------------------------------------------ metadata
def test_metadata_from_watch_page_is_complete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "oembed" in request.url.host + str(request.url.path):
            return httpx.Response(200, json=OEMBED)
        return httpx.Response(200, text=WATCH_HTML)

    metadata = _client(handler).fetch(VIDEO_ID)
    assert metadata.title == "Risk assessment explained"
    assert metadata.channel == "Synthetic Channel"
    assert metadata.channel_id == "UC1234567890"
    assert metadata.duration_seconds == 312.0
    assert metadata.upload_date == "2026-01-15"
    assert metadata.available_languages == ["en", "es"]
    assert metadata.source_detail["caption_kinds"] == "asr,manual"


def test_metadata_falls_back_to_oembed_without_watch_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OEMBED)

    metadata = HttpMetadataSource(
        httpx.Client(transport=httpx.MockTransport(handler)), use_watch_page=False
    ).fetch(VIDEO_ID)
    assert metadata.title == "Risk assessment explained"
    assert metadata.duration_seconds is None
    assert metadata.upload_date is None
    assert metadata.source_detail["metadata_source"] == "oembed"


def test_missing_duration_stays_none_and_is_not_estimated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OEMBED)

    metadata = _client(handler).fetch(VIDEO_ID)
    assert metadata.duration_seconds is None
    assert metadata.upload_date is None
    assert "watch_page_error" in metadata.source_detail


def test_oembed_failure_is_reported_as_metadata_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(MetadataRetrievalFailed, match="metadata retrieval failed"):
        _client(handler).fetch(VIDEO_ID)


def test_transport_error_is_reported_as_metadata_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(MetadataRetrievalFailed, match="oEmbed request failed"):
        _client(handler).fetch(VIDEO_ID)


def test_unparsable_oembed_body_is_handled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(MetadataRetrievalFailed, match="no title could be retrieved"):
        _client(handler).fetch(VIDEO_ID)


# ------------------------------------------------------------------- captions
@pytest.mark.parametrize(
    ("library_error", "expected"),
    [
        ("TranscriptsDisabled", TranscriptUnavailable),
        ("NoTranscriptFound", TranscriptUnavailable),
        ("VideoUnavailable", VideoUnavailable),
        ("VideoUnplayable", VideoUnavailable),
        ("AgeRestricted", VideoUnavailable),
        ("InvalidVideoId", VideoUnavailable),
        ("IpBlocked", TranscriptRetrievalFailed),
        ("RequestBlocked", TranscriptRetrievalFailed),
        ("PoTokenRequired", TranscriptRetrievalFailed),
    ],
)
def test_library_exceptions_map_to_typed_errors(library_error: str, expected: type) -> None:
    import youtube_transcript_api as lib

    error_class = getattr(lib, library_error)
    if library_error == "NoTranscriptFound":
        raw: Exception = error_class(VIDEO_ID, ["en"], [])
    elif library_error == "VideoUnplayable":
        raw = error_class(VIDEO_ID, "This video is unavailable", [])
    elif library_error == "CouldNotRetrieveTranscript":
        raw = error_class(VIDEO_ID, [])
    elif library_error == "YouTubeRequestFailed":
        raw = error_class(VIDEO_ID)
    else:
        raw = error_class(VIDEO_ID)

    translated = YouTubeCaptionSource._translate(raw, VIDEO_ID, languages=("en",))
    assert isinstance(translated, expected)
    assert translated.message


def test_unknown_library_exception_becomes_retrieval_failure() -> None:
    translated = YouTubeCaptionSource._translate(RuntimeError("boom"), VIDEO_ID)
    assert isinstance(translated, TranscriptRetrievalFailed)
    assert "RuntimeError" in translated.message


def test_translation_is_idempotent_for_typed_errors() -> None:
    original = TranscriptUnavailable(VIDEO_ID, reason="captions disabled")
    assert YouTubeCaptionSource._translate(original, VIDEO_ID) is original


class _StubApi:
    """Stands in for ``YouTubeTranscriptApi`` without touching the network."""

    def __init__(self, snippets: list[object] | None = None, tracks: object | None = None) -> None:
        self._snippets = snippets or []
        self._tracks = tracks
        self.fetch_calls: list[dict[str, object]] = []

    def fetch(self, video_id: str, languages: object = None, **kwargs: object) -> object:
        self.fetch_calls.append({"video_id": video_id, "languages": languages})

        class _Fetched(list):
            language = "English"
            language_code = "en"
            is_generated = False

        return _Fetched(self._snippets)

    def list(self, video_id: str) -> object:
        if self._tracks is None:
            from youtube_transcript_api import NoTranscriptFound

            raise NoTranscriptFound(video_id, ["en"], [])
        return self._tracks


def test_empty_track_is_reported_as_unavailable() -> None:
    source = YouTubeCaptionSource()
    source._api = _StubApi(snippets=[])
    with pytest.raises(TranscriptUnavailable, match="empty"):
        source.fetch(VIDEO_ID, ("en",))


def test_fetch_passes_language_priority_and_copies_timing() -> None:
    from youtube_transcript_api import FetchedTranscriptSnippet

    snippets = [FetchedTranscriptSnippet(text=" hello ", start=1.5, duration=2.0)]
    source = YouTubeCaptionSource()
    stub = _StubApi(snippets=snippets)
    source._api = stub

    track = source.fetch(VIDEO_ID, ("en", "es"))
    assert stub.fetch_calls[0]["languages"] == ["en", "es"]
    assert track.snippets[0].text == " hello "
    assert track.snippets[0].start == 1.5
    assert track.snippets[0].end == pytest.approx(3.5)
    assert track.source == "youtube-transcript-api"


def test_available_tracks_requires_a_library_transcript_list() -> None:
    source = YouTubeCaptionSource()
    source._api = _StubApi()
    with pytest.raises(TranscriptUnavailable, match="no usable transcript"):
        source.available_tracks(VIDEO_ID)
