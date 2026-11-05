"""Anonymous media download through ``yt-dlp``.

Downloads land in a caller-provided temporary directory. The service verifies them
before moving anything into the reel's permanent folder, so an interrupted download
never leaves a partial file where the archive expects a complete one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reel_vault.media.base import MediaDownloadError
from reel_vault.media.verify import kind_for
from reel_vault.schemas.enrichment import EnrichmentResult

_QUALITY_FORMATS: dict[str, str] = {
    "best": "best[ext=mp4]/best",
    "1080": "best[height<=1080][ext=mp4]/best[height<=1080]/best",
    "720": "best[height<=720][ext=mp4]/best[height<=720]/best",
}

_SKIP_SUFFIXES = frozenset({".part", ".ytdl", ".temp", ".json"})


def _downloaded_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() not in _SKIP_SUFFIXES
    )


def canonical_name(path: Path, counters: dict[str, int]) -> str:
    """Give downloaded files stable, human-readable names."""
    kind = kind_for(path)
    suffix = path.suffix.lower()
    counters[kind] = counters.get(kind, 0) + 1
    number = counters[kind]
    if kind == "video":
        return f"video{suffix}" if number == 1 else f"video-{number:02d}{suffix}"
    if kind == "audio":
        return f"audio{suffix}" if number == 1 else f"audio-{number:02d}{suffix}"
    if kind == "image":
        return f"image-{number:02d}{suffix}"
    return f"file-{number:02d}{suffix or '.bin'}"


class YtDlpMediaDownloader:
    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    def _options(self, destination: Path, quality: str) -> dict[str, Any]:
        return {
            "quiet": True,
            "no_warnings": True,
            "no_color": True,
            "noplaylist": False,
            "format": _QUALITY_FORMATS.get(quality, _QUALITY_FORMATS["best"]),
            "outtmpl": str(destination / "%(autonumber)03d-%(id)s.%(ext)s"),
            "socket_timeout": self.timeout_seconds,
            "retries": 3,
            "fragment_retries": 3,
            "overwrites": True,
            "cookiefile": None,
            "writethumbnail": False,
            "writesubtitles": False,
            "ignoreerrors": False,
        }

    def download(self, result: EnrichmentResult, destination: Path, *, quality: str) -> list[Path]:
        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise MediaDownloadError(f"yt-dlp is not installed: {exc}") from exc

        destination.mkdir(parents=True, exist_ok=True)
        try:
            with yt_dlp.YoutubeDL(self._options(destination, quality)) as ydl:
                ydl.download([result.url])
        except Exception as exc:
            raise MediaDownloadError(str(exc)) from exc

        files = _downloaded_files(destination)
        if not files:
            raise MediaDownloadError("yt-dlp downloaded no media files")
        return files


__all__ = ["YtDlpMediaDownloader", "canonical_name"]
