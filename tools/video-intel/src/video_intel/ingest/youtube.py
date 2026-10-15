"""YouTube URL parsing and validation (§4).

Accepts the common watch, short, embed, shorts and live forms plus bare 11-character
ids. Anything else is rejected with :class:`InvalidVideoUrl` before any network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from video_intel.ingest.errors import InvalidVideoUrl

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ACCEPTED_HOSTS = {
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "www.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
_PATH_PREFIXES = ("/embed/", "/v/", "/shorts/", "/live/")


@dataclass(frozen=True)
class ParsedVideo:
    """A validated YouTube video reference."""

    video_id: str
    url: str
    canonical_url: str

    def __str__(self) -> str:
        return self.canonical_url


def extract_video_id(url: str) -> str | None:
    """Return the 11-character video id, or ``None`` when the URL is not YouTube."""
    candidate = (url or "").strip()
    if not candidate:
        return None
    if VIDEO_ID_RE.match(candidate):
        return candidate

    parsed = urlparse(candidate if "//" in candidate else f"https://{candidate}")
    host = (parsed.netloc or "").lower()
    if host not in _ACCEPTED_HOSTS:
        return None

    path = parsed.path.rstrip("/")
    if host in {"youtu.be", "www.youtu.be"}:
        found = path.lstrip("/").split("/")[0]
    elif any(path.startswith(prefix) for prefix in _PATH_PREFIXES):
        found = path.split("/")[2]
    else:
        found = (parse_qs(parsed.query).get("v") or [""])[0]

    return found if VIDEO_ID_RE.match(found) else None


def parse_youtube_url(url: str) -> ParsedVideo:
    """Validate a YouTube URL and return the parsed reference.

    Raises:
        InvalidVideoUrl: when the URL does not identify a video.
    """
    video_id = extract_video_id(url)
    if video_id is None:
        raise InvalidVideoUrl(url)
    canonical = f"https://www.youtube.com/watch?v={video_id}"
    return ParsedVideo(video_id=video_id, url=url, canonical_url=canonical)
