"""Anonymous metadata fetching through ``yt-dlp``.

Only public data is requested: no cookies, no account, no session. ``yt-dlp`` is
imported lazily so that ``reel-vault doctor`` and the test suite stay fast and usable
when the optional binary is absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from reel_vault.enrich.base import FetchError
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus, MediaItem

_MEDIA_EXTS = frozenset({"mp4", "m4v", "webm", "mov", "m4a", "mp3", "jpg", "jpeg", "png", "webp"})
_PLAYLIST_TYPES = frozenset({"playlist", "multi_video"})


def classify_error(message: str) -> FetchStatus:
    """Map an extractor error message onto a status a human can act on."""
    text = message.lower()
    if ("rate" in text and "limit" in text) or "429" in text or "too many requests" in text:
        return FetchStatus.RATE_LIMITED
    if "private" in text:
        return FetchStatus.PRIVATE
    if "login" in text or "cookies" in text or "blocked" in text or "not a bot" in text:
        return FetchStatus.BLOCKED
    if any(
        phrase in text
        for phrase in ("not available", "deleted", "removed", "does not exist", "no longer")
    ):
        return FetchStatus.DELETED
    if "unsupported" in text or "no video" in text:
        return FetchStatus.UNSUPPORTED
    return FetchStatus.ERROR


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _upload_date(value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC).date().isoformat()
        except ValueError:
            continue
    return None


def _media_item(raw: dict[str, Any], *, index: int) -> MediaItem | None:
    url = _as_text(raw.get("url"))
    if not url:
        return None
    ext = _as_text(raw.get("ext"))
    if ext is not None:
        ext = ext.lower()
        if ext not in _MEDIA_EXTS:
            return None
    return MediaItem(
        url=url,
        ext=ext,
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        filesize=_as_int(raw.get("filesize") or raw.get("filesize_approx")),
        thumbnail=_as_text(raw.get("thumbnail")),
        index=index,
    )


def _media_items(info: dict[str, Any], *, limit: int) -> list[MediaItem]:
    items: list[MediaItem] = []
    seen: set[str] = set()

    entries = info.get("entries")
    if isinstance(entries, list):
        candidates = [entry for entry in entries if isinstance(entry, dict)]
    else:
        candidates = [info]

    for candidate in candidates:
        direct = _media_item(candidate, index=len(items))
        if direct is not None and direct.url not in seen:
            items.append(direct)
            seen.add(direct.url)
        formats = candidate.get("formats")
        if not isinstance(formats, list):
            continue
        for raw_format in formats:
            if not isinstance(raw_format, dict):
                continue
            item = _media_item(raw_format, index=len(items))
            if item is not None and item.url not in seen:
                items.append(item)
                seen.add(item.url)
            if len(items) >= limit:
                return items
    return items


def normalise_info(
    info: dict[str, Any], *, shortcode: str, url: str, limit: int
) -> EnrichmentResult:
    """Convert one ``yt-dlp`` info dict into the stored schema."""
    entries = info.get("entries")
    is_carousel = isinstance(entries, list) and len(entries) > 1
    first_entry = entries[0] if isinstance(entries, list) and entries else None
    primary = first_entry if isinstance(first_entry, dict) else info

    return EnrichmentResult(
        shortcode=shortcode,
        url=url,
        fetch_status=FetchStatus.OK,
        title=_as_text(primary.get("title")) or _as_text(info.get("title")),
        description=_as_text(primary.get("description")) or _as_text(info.get("description")),
        author=(
            _as_text(info.get("uploader"))
            or _as_text(info.get("channel"))
            or _as_text(primary.get("uploader"))
        ),
        duration=_as_float(primary.get("duration")) or _as_float(info.get("duration")),
        upload_date=_upload_date(primary.get("upload_date") or info.get("upload_date")),
        thumbnail=_as_text(primary.get("thumbnail")) or _as_text(info.get("thumbnail")),
        media=_media_items(info, limit=limit),
        view_count=_as_int(primary.get("view_count")),
        like_count=_as_int(primary.get("like_count")),
        comment_count=_as_int(primary.get("comment_count")),
        extractor=_as_text(info.get("extractor_key")) or _as_text(info.get("extractor")),
        webpage_url=_as_text(info.get("webpage_url")) or url,
        is_carousel=is_carousel,
        item_count=len(entries) if isinstance(entries, list) and entries else 1,
    )


class YtDlpFetcher:
    """Metadata-only ``yt-dlp`` fetcher. Never passes cookies or credentials."""

    def __init__(self, *, timeout_seconds: float = 30.0, max_media_items: int = 50) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_media_items = max_media_items

    def _options(self) -> dict[str, Any]:
        return {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": False,
            "socket_timeout": self.timeout_seconds,
            "extract_flat": False,
            "ignoreerrors": False,
            "no_color": True,
            "nocheckcertificate": False,
            # Never read or write a cookie jar.
            "cookiefile": None,
        }

    def fetch(self, url: str, *, shortcode: str) -> EnrichmentResult:
        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise FetchError(FetchStatus.ERROR, f"yt-dlp is not installed: {exc}") from exc

        try:
            with yt_dlp.YoutubeDL(self._options()) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            message = str(exc)
            raise FetchError(classify_error(message), message) from exc
        except Exception as exc:
            raise FetchError(FetchStatus.ERROR, str(exc)) from exc

        if not isinstance(info, dict):
            raise FetchError(FetchStatus.ERROR, "yt-dlp returned no metadata")
        return normalise_info(info, shortcode=shortcode, url=url, limit=self.max_media_items)


__all__ = ["YtDlpFetcher", "classify_error", "normalise_info"]
