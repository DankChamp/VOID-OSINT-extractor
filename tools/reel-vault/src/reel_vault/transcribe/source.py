"""Find or fetch an audio-bearing file for a reel.

When media was archived, the local video is reused. Otherwise only the audio track is
downloaded into a temporary directory, so ``--download none`` still produces a
transcript without keeping a video.
"""

from __future__ import annotations

from pathlib import Path

from reel_vault.media.verify import kind_for
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.transcribe.base import NoAudioError, TranscriptionError


def _audio_capable(paths: list[Path]) -> Path | None:
    for path in sorted(paths):
        if path.is_file() and kind_for(path) in {"video", "audio"}:
            return path
    return None


class ReelMediaAudioSource:
    def __init__(self, paths: WorkspacePaths, *, timeout_seconds: float = 60.0) -> None:
        self.paths = paths
        self.timeout_seconds = timeout_seconds

    def prepare(self, result: EnrichmentResult, workdir: Path) -> Path:
        existing = _audio_capable(list(self.paths.media(result.shortcode).glob("*")))
        if existing is not None:
            return existing

        workdir.mkdir(parents=True, exist_ok=True)
        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise TranscriptionError(f"yt-dlp is not installed: {exc}") from exc

        options = {
            "quiet": True,
            "no_warnings": True,
            "no_color": True,
            "format": "bestaudio/best",
            "outtmpl": str(workdir / "source.%(ext)s"),
            "noplaylist": True,
            "socket_timeout": self.timeout_seconds,
            "cookiefile": None,
            "overwrites": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([result.url])
        except Exception as exc:
            raise TranscriptionError(f"cannot fetch audio: {exc}") from exc

        downloaded = _audio_capable(list(workdir.glob("source.*")))
        if downloaded is None:
            raise NoAudioError("the post has no downloadable audio")
        return downloaded


__all__ = ["ReelMediaAudioSource"]
