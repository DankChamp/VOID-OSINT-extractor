"""Shared enums and small helpers."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum


class MediaType(StrEnum):
    REEL = "reel"
    POST = "post"
    CAROUSEL = "carousel"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ReelStatus(StrEnum):
    NEW = "new"
    FETCHED = "fetched"
    TRANSCRIBED = "transcribed"
    TRIAGED = "triaged"
    EXTRACTED = "extracted"
    CLASSIFIED = "classified"
    DONE = "done"
    STUB = "stub"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"


class Decision(StrEnum):
    EXTRACT = "extract"
    ARCHIVE = "archive"
    STUB = "stub"


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "unknown"


__all__ = [
    "Decision",
    "JobState",
    "MediaType",
    "ReelStatus",
    "iso_now",
    "slugify",
    "utc_now",
]
