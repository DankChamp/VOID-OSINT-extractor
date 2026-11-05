"""Keyframe selection by scene change.

Picks a bounded set of representative frames from a video so OCR and screenshots have
something to look at without scanning every second.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Keyframe:
    path: Path
    timestamp: float


class KeyframeError(RuntimeError):
    """Could not select keyframes from a video."""


def _parse_timestamps(stderr: str) -> list[float]:
    timestamps: list[float] = []
    for line in stderr.splitlines():
        if "pts_time" not in line:
            continue
        for token in line.split():
            if token.startswith("pts_time:"):
                with contextlib.suppress(ValueError):
                    timestamps.append(float(token.split(":", 1)[1]))
                break
    return timestamps


def _duration(video: Path, ffprobe: str, run: Callable[..., Any]) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        completed = run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return float(completed.stdout.strip() or 0.0)
    except (ValueError, subprocess.SubprocessError, OSError):
        return 0.0


def select_keyframes(
    video: Path,
    workdir: Path,
    *,
    max_keyframes: int = 6,
    scene_threshold: float = 0.3,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    run: Callable[..., Any] = subprocess.run,
) -> list[Keyframe]:
    """Return up to ``max_keyframes`` frames chosen by scene change."""
    if not video.is_file():
        raise KeyframeError(f"video not found: {video}")
    workdir.mkdir(parents=True, exist_ok=True)

    # Detect scene changes; ffmpeg's select filter emits a metadata score per frame.
    detect_cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video),
        "-filter_complex",
        f"select='gt(score\\,{scene_threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    completed = run(
        detect_cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    timestamps = _parse_timestamps(completed.stderr) if completed.returncode == 0 else []

    if not timestamps:
        # Fallback: evenly spaced frames across the duration.
        duration = _duration(video, ffprobe, run)
        if duration <= 0:
            return []
        step = max(1.0, duration / 6.0)
        timestamps = [
            min(duration - 0.5, i * step) for i in range(min(6, max(1, int(duration // step or 1))))
        ]

    # Cap and deduplicate, preferring early frames.
    chosen: list[float] = []
    for ts in timestamps:
        if all(abs(ts - kept) >= 1.0 for kept in chosen):
            chosen.append(ts)
        if len(chosen) >= max_keyframes:
            break

    frames: list[Keyframe] = []
    for index, ts in enumerate(chosen):
        out = workdir / f"keyframe-{index:02d}.jpg"
        extract_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-ss",
            str(ts),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        completed = run(extract_cmd, capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            frames.append(Keyframe(path=out, timestamp=ts))
    return frames


__all__ = ["Keyframe", "KeyframeError", "select_keyframes"]
