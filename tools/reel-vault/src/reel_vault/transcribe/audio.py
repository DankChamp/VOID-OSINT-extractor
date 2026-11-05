"""Audio extraction with ffmpeg.

Whisper wants mono 16 kHz audio. Chunks stay well under Groq's 25 MB request limit;
at the default 64 kbps a five-minute chunk is about 2.4 MB.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from reel_vault.transcribe.base import AudioExtraction, NoAudioError, TranscriptionError

AUDIO_BITRATE = "64k"
SAMPLE_RATE = "16000"
CHUNK_PATTERN = "chunk-%03d.m4a"


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> tuple[float, bool]:
    """Return ``(duration_seconds, has_audio)`` for ``path``."""
    if shutil.which(ffprobe) is None:
        raise TranscriptionError(f"{ffprobe} is required to inspect media")
    completed = subprocess.run(
        [
            ffprobe,
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
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise TranscriptionError(f"ffprobe cannot read {path.name}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"ffprobe returned malformed JSON for {path.name}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"ffprobe returned no metadata for {path.name}")

    duration = 0.0
    raw_format = payload.get("format")
    if isinstance(raw_format, dict):
        try:
            duration = float(raw_format.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    streams = payload.get("streams")
    has_audio = False
    if isinstance(streams, list):
        has_audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
        )
    return duration, has_audio


def _run(args: list[str]) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise TranscriptionError(detail[-1] if detail else "ffmpeg failed")


class FfmpegAudioExtractor:
    def extract(
        self, source: Path, workdir: Path, *, chunk_seconds: int, max_chunk_bytes: int
    ) -> AudioExtraction:
        if not source.is_file() or source.stat().st_size <= 0:
            raise TranscriptionError(f"media file is missing or empty: {source}")
        duration, has_audio = probe_media(source)
        if not has_audio:
            raise NoAudioError(f"{source.name} has no audio track")

        workdir.mkdir(parents=True, exist_ok=True)
        audio = workdir / "audio.m4a"
        _run(
            [
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                SAMPLE_RATE,
                "-c:a",
                "aac",
                "-b:a",
                AUDIO_BITRATE,
                str(audio),
            ]
        )
        if not audio.is_file() or audio.stat().st_size <= 0:
            raise TranscriptionError("ffmpeg produced no audio")

        if duration <= chunk_seconds and audio.stat().st_size <= max_chunk_bytes:
            return AudioExtraction(chunks=[audio], duration_seconds=duration)

        _run(
            [
                "-i",
                str(audio),
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
                str(workdir / CHUNK_PATTERN),
            ]
        )
        chunks = sorted(workdir.glob("chunk-*.m4a"))
        if not chunks:
            raise TranscriptionError("ffmpeg produced no audio chunks")
        oversized = [chunk.name for chunk in chunks if chunk.stat().st_size > max_chunk_bytes]
        if oversized:
            raise TranscriptionError(
                f"audio chunk exceeds the provider limit: {', '.join(oversized)}"
            )
        return AudioExtraction(chunks=chunks, duration_seconds=duration)


__all__ = ["AUDIO_BITRATE", "FfmpegAudioExtractor", "probe_media"]
