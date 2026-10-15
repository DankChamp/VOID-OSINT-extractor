"""Network-backed ingestion clients (§4).

Two clients, both dependency-injected:

* :class:`YouTubeCaptionSource` wraps ``youtube-transcript-api`` (verified against the
  1.x API: ``YouTubeTranscriptApi().list(video_id)`` and ``.fetch(video_id, languages=…)``
  returning objects with ``language_code``, ``language``, ``is_generated`` and snippets
  carrying ``text``/``start``/``duration``). Library exceptions are translated into this
  application's typed errors rather than leaking through.
* :class:`HttpMetadataSource` reads metadata over httpx: the official oEmbed endpoint
  for title and channel, plus the watch page's player response for duration, upload date
  and the caption track list. If the watch page cannot be parsed, oEmbed data is kept and
  the unavailable fields stay ``None`` — nothing is estimated or invented.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from video_intel.ingest.errors import (
    IngestError,
    MetadataRetrievalFailed,
    TranscriptRetrievalFailed,
    TranscriptUnavailable,
    VideoUnavailable,
)
from video_intel.ingest.sources import CaptionSnippet, CaptionTrack, TrackInfo
from video_intel.schemas.transcript import VideoMetadata

OEMBED_URL = "https://www.youtube.com/oembed"
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
PLAYER_RESPONSE_RE = re.compile(
    r"ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;\s*(?:var|</script>)", re.DOTALL
)
DURATION_RE = re.compile(r'"lengthSeconds"\s*:\s*"(\d+)"')


class YouTubeCaptionSource:
    """Caption source backed by ``youtube-transcript-api``."""

    def __init__(self, *, proxy: str | None = None) -> None:
        self._proxy = proxy
        self._api: Any | None = None

    def _client(self) -> Any:
        # Imported lazily so the dependency stays optional for pure-offline use.
        if self._api is None:
            from youtube_transcript_api import YouTubeTranscriptApi

            proxy_config = None
            if self._proxy:
                from youtube_transcript_api.proxies import GenericProxyConfig

                proxy_config = GenericProxyConfig(http_url=self._proxy, https_url=self._proxy)
            self._api = YouTubeTranscriptApi(proxy_config=proxy_config)
        return self._api

    def available_tracks(self, video_id: str) -> tuple[TrackInfo, ...]:
        try:
            transcript_list = self._client().list(video_id)
        except Exception as exc:  # library raises many types; all are translated below
            raise self._translate(exc, video_id) from exc

        tracks: list[TrackInfo] = []
        # The library stores these as dicts keyed by language code; the values hold
        # the Transcript objects with language names and codes.
        for group, is_generated in (
            (getattr(transcript_list, "_manually_created_transcripts", None), False),
            (getattr(transcript_list, "_generated_transcripts", None), True),
        ):
            for track in (group or {}).values():
                tracks.append(_track_info(track, is_generated=is_generated))
        if not tracks:
            raise TranscriptUnavailable(video_id, reason="no caption tracks are published")
        return tuple(tracks)

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> CaptionTrack:
        try:
            fetched = self._client().fetch(video_id, languages=list(languages) or ["en"])
        except Exception as exc:  # library raises many types; all are translated below
            raise self._translate(exc, video_id, languages=languages) from exc

        snippets = tuple(
            CaptionSnippet(
                text=snippet.text,
                start=float(snippet.start),
                duration=float(snippet.duration),
            )
            for snippet in fetched
        )
        if not snippets:
            raise TranscriptUnavailable(video_id, reason="the caption track is empty")
        return CaptionTrack(
            video_id=video_id,
            language=getattr(fetched, "language", "") or "",
            language_code=getattr(fetched, "language_code", "")
            or (languages[0] if languages else ""),
            is_generated=bool(getattr(fetched, "is_generated", False)),
            snippets=snippets,
            source="youtube-transcript-api",
        )

    @staticmethod
    def _translate(exc: Exception, video_id: str, *, languages: tuple[str, ...] = ()) -> Exception:
        """Map library exceptions onto this application's typed errors."""
        if isinstance(exc, IngestError):
            return exc  # already translated; translation is idempotent

        from youtube_transcript_api import (
            AgeRestricted,
            CouldNotRetrieveTranscript,
            InvalidVideoId,
            IpBlocked,
            NoTranscriptFound,
            PoTokenRequired,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnplayable,
            YouTubeRequestFailed,
        )
        from youtube_transcript_api import (
            VideoUnavailable as LibVideoUnavailable,
        )

        if isinstance(exc, (LibVideoUnavailable, VideoUnplayable, AgeRestricted, InvalidVideoId)):
            return VideoUnavailable(video_id, reason=type(exc).__name__)
        if isinstance(exc, (TranscriptsDisabled, NoTranscriptFound)):
            wanted = ", ".join(languages) if languages else None
            return TranscriptUnavailable(
                video_id,
                reason=f"{type(exc).__name__}" + (f" (requested: {wanted})" if wanted else ""),
            )
        if isinstance(exc, (IpBlocked, RequestBlocked, PoTokenRequired)):
            return TranscriptRetrievalFailed(video_id, reason=type(exc).__name__)
        if isinstance(exc, CouldNotRetrieveTranscript):
            return TranscriptRetrievalFailed(video_id, reason=type(exc).__name__)
        if isinstance(exc, YouTubeRequestFailed):
            return TranscriptRetrievalFailed(video_id, reason=str(exc) or type(exc).__name__)
        return TranscriptRetrievalFailed(video_id, reason=f"{type(exc).__name__}: {exc}")


def _track_info(track: Any, *, is_generated: bool) -> TrackInfo:
    return TrackInfo(
        language=getattr(track, "language", "") or "",
        language_code=getattr(track, "language_code", "") or "",
        is_generated=is_generated,
    )


class HttpMetadataSource:
    """Metadata source over httpx: oEmbed plus watch-page player response.

    The underlying client is created lazily and must be released with ``close()`` (or by
    using the object as a context manager); otherwise a run leaks sockets.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: float = 20.0,
        use_watch_page: bool = True,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._use_watch_page = use_watch_page

    def __enter__(self) -> HttpMetadataSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the client, but only if this object created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout, headers=DEFAULT_HEADERS, follow_redirects=True
            )
            self._owns_client = True
        return self._client

    def fetch(self, video_id: str) -> VideoMetadata:
        source_detail: dict[str, str] = {}
        title, channel, channel_id = self._oembed(video_id, source_detail)
        duration: float | None = None
        upload_date: str | None = None
        languages: list[str] = []
        caption_kinds: str | None = None

        if self._use_watch_page:
            player = self._watch_page(video_id, source_detail)
            if player:
                details = player.get("videoDetails") or {}
                micro = ((player.get("microformat") or {}).get("playerMicroformatRenderer")) or {}
                title = details.get("title") or title
                channel = details.get("author") or channel
                channel_id = details.get("channelId") or channel_id
                length = details.get("lengthSeconds")
                if isinstance(length, (str, int)) and str(length).isdigit():
                    duration = float(length)
                upload_date = micro.get("uploadDate") or micro.get("publishDate")
                tracks = (
                    (player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
                ).get("captionTracks") or []
                languages = sorted(
                    {
                        str(track.get("languageCode"))
                        for track in tracks
                        if isinstance(track, dict) and track.get("languageCode")
                    }
                )
                if tracks:
                    kinds = {
                        str(track.get("kind") or "manual")
                        for track in tracks
                        if isinstance(track, dict)
                    }
                    caption_kinds = ",".join(sorted(kinds))

        if not title:
            raise MetadataRetrievalFailed(video_id, "no title could be retrieved")

        return VideoMetadata(
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=title,
            channel=channel,
            channel_id=channel_id,
            duration_seconds=duration,
            upload_date=upload_date,
            available_languages=languages,
            source_detail={
                **source_detail,
                **({"caption_kinds": caption_kinds} if caption_kinds else {}),
            },
        )

    def _oembed(
        self, video_id: str, source_detail: dict[str, str]
    ) -> tuple[str | None, str | None, str | None]:
        """Title, channel name and channel id from the official oEmbed endpoint."""
        params = {"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"}
        try:
            response = self._http().get(OEMBED_URL, params=params)
        except httpx.HTTPError as exc:
            raise MetadataRetrievalFailed(video_id, f"oEmbed request failed: {exc}") from exc
        if response.status_code != 200:
            source_detail["oembed_status"] = str(response.status_code)
            return (None, None, None)
        try:
            payload = response.json()
        except json.JSONDecodeError:
            source_detail["oembed_status"] = "unparsable"
            return (None, None, None)
        source_detail["metadata_source"] = "oembed"
        author_url = str(payload.get("author_url") or "")
        channel_id = author_url.rsplit("/", 1)[-1] if author_url.startswith("/channel/") else None
        return (payload.get("title"), payload.get("author_name"), channel_id)

    def _watch_page(self, video_id: str, source_detail: dict[str, str]) -> dict[str, Any] | None:
        try:
            response = self._http().get(WATCH_URL.format(video_id=video_id))
        except httpx.HTTPError as exc:
            source_detail["watch_page_error"] = type(exc).__name__
            return None
        if response.status_code != 200:
            source_detail["watch_page_status"] = str(response.status_code)
            return None
        html = response.text
        match = PLAYER_RESPONSE_RE.search(html)
        if match:
            try:
                parsed: dict[str, Any] = json.loads(match.group(1))
            except json.JSONDecodeError:
                source_detail["watch_page_error"] = "unparsable player response"
                return None
            else:
                return parsed
        duration_match = DURATION_RE.search(html)
        if duration_match:
            source_detail["metadata_source"] = "watch_page"
            source_detail["duration_only"] = "true"
            return {"videoDetails": {"lengthSeconds": duration_match.group(1)}}
        source_detail["watch_page_error"] = "player response not found"
        return None


__all__ = [
    "DEFAULT_HEADERS",
    "OEMBED_URL",
    "WATCH_URL",
    "HttpMetadataSource",
    "YouTubeCaptionSource",
]
