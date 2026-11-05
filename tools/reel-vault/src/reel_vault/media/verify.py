"""Media verification: size, container readability and checksums."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reel_vault.schemas.media import MediaKind

VIDEO_EXTS = frozenset({".mp4", ".m4v", ".webm", ".mov", ".mkv"})
AUDIO_EXTS = frozenset({".m4a", ".mp3", ".aac", ".ogg", ".opus", ".wav"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CHECKSUM_CHUNK = 1024 * 1024


def kind_for(path: Path) -> MediaKind:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in AUDIO_EXTS:
        return "audio"
    if suffix in IMAGE_EXTS:
        return "image"
    return "other"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHECKSUM_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path, binary: str, timeout: float) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                binary,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class FfprobeVerifier:
    """Verify video/audio with ffprobe; images only need a non-empty file."""

    def __init__(
        self,
        *,
        ffprobe: str = "ffprobe",
        timeout_seconds: float = 30.0,
        run: Callable[[Path, str, float], dict[str, Any] | None] = _probe,
    ) -> None:
        self.ffprobe = ffprobe
        self.timeout_seconds = timeout_seconds
        self._run = run

    @property
    def available(self) -> bool:
        return shutil.which(self.ffprobe) is not None

    def verify(self, path: Path, *, kind: str) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if kind not in {"video", "audio"}:
            return True
        if not self.available:
            # ffprobe missing: doctor already warns; size is the best we can do.
            return True
        payload = self._run(path, self.ffprobe, self.timeout_seconds)
        if payload is None:
            return False
        streams = payload.get("streams")
        if not isinstance(streams, list):
            return False
        wanted = "video" if kind == "video" else "audio"
        return any(
            isinstance(stream, dict) and stream.get("codec_type") == wanted for stream in streams
        )


__all__ = [
    "AUDIO_EXTS",
    "IMAGE_EXTS",
    "VIDEO_EXTS",
    "FfprobeVerifier",
    "kind_for",
    "sha256_file",
]
