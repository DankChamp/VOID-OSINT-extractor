"""Enrichment orchestration.

Fetches metadata one reel at a time, classifies failures, retries transient errors and
persists every outcome. A failure marks one reel; it never aborts the run.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from reel_vault.config.models import AppConfig
from reel_vault.enrich.base import FetchError, MetadataFetcher, ThumbnailFetcher
from reel_vault.enrich.thumbnails import HttpThumbnailFetcher
from reel_vault.enrich.ytdlp import YtDlpFetcher
from reel_vault.logging_setup import get_logger
from reel_vault.schemas.common import JobState, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_json_atomic
from reel_vault.storage.paths import WorkspacePaths, relative_key

logger = get_logger(__name__)

RETRYABLE_REEL_STATUS = frozenset({ReelStatus.NEW, ReelStatus.RATE_LIMITED, ReelStatus.FAILED})

_THUMBNAIL_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

_FAILURE_REEL_STATUS: dict[FetchStatus, ReelStatus] = {
    FetchStatus.PRIVATE: ReelStatus.UNAVAILABLE,
    FetchStatus.DELETED: ReelStatus.UNAVAILABLE,
    FetchStatus.UNSUPPORTED: ReelStatus.UNAVAILABLE,
    FetchStatus.BLOCKED: ReelStatus.FAILED,
    FetchStatus.RATE_LIMITED: ReelStatus.RATE_LIMITED,
    FetchStatus.ERROR: ReelStatus.FAILED,
}


@dataclass
class FetchSummary:
    processed: int = 0
    ok: int = 0
    unavailable: int = 0
    failed: int = 0
    skipped: int = 0
    statuses: dict[str, int] = field(default_factory=dict)

    def record(self, result: EnrichmentResult) -> None:
        self.processed += 1
        self.statuses[result.fetch_status.value] = (
            self.statuses.get(result.fetch_status.value, 0) + 1
        )
        if result.fetch_status is FetchStatus.OK:
            self.ok += 1
        elif result.fetch_status.unavailable:
            self.unavailable += 1
        else:
            self.failed += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "ok": self.ok,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "skipped": self.skipped,
            "statuses": dict(sorted(self.statuses.items())),
        }


def thumbnail_filename(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return f"thumb{suffix if suffix in _THUMBNAIL_SUFFIXES else '.jpg'}"


_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def hashtags_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    found: dict[str, str] = {}
    for match in _HASHTAG_RE.findall(text):
        found.setdefault(match.lower(), f"#{match}")
    return sorted(found.values())


class EnrichmentService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        fetcher: MetadataFetcher | None = None,
        thumbnail_fetcher: ThumbnailFetcher | None = None,
        sleep: Callable[[float], None] = time.sleep,
        request_delay: float | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.fetcher = fetcher or YtDlpFetcher(
            timeout_seconds=config.enrich.timeout_seconds,
            max_media_items=config.enrich.max_media_items,
        )
        self.thumbnail_fetcher = thumbnail_fetcher or HttpThumbnailFetcher(
            timeout_seconds=config.enrich.timeout_seconds
        )
        self.sleep = sleep
        self.request_delay = (
            config.run.request_delay_seconds if request_delay is None else max(0.0, request_delay)
        )
        self._requests = 0

    # ------------------------------------------------------------------ public API

    def fetch_pending(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
    ) -> FetchSummary:
        summary = FetchSummary()
        requested = (
            None if shortcodes is None else list(dict.fromkeys(str(code) for code in shortcodes))
        )
        selected = self._select(shortcodes=requested, force=force)
        if limit is not None:
            selected = selected[: max(0, limit)]
        known = {str(reel["shortcode"]) for reel in self.store.list_reels()}
        if requested is not None:
            summary.skipped += sum(1 for code in requested if code not in known)

        for reel in selected:
            shortcode = str(reel["shortcode"])
            result = self.fetch_one(shortcode)
            summary.record(result)
            logger.info("fetch %s: %s", shortcode, result.fetch_status.value)
        return summary

    def fetch_one(self, shortcode: str) -> EnrichmentResult:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise LookupError(f"unknown shortcode: {shortcode}")

        self.store.set_job(shortcode, "fetch", JobState.RUNNING)
        result = self._fetch_with_retries(shortcode, str(reel["url"]))

        if result.fetch_status is FetchStatus.OK:
            self._attach_thumbnail(shortcode, result)
            self.store.update_reel_enrichment(
                shortcode,
                author=result.author,
                caption=result.caption,
                media_type="carousel" if result.is_carousel else None,
                hashtags=hashtags_from_text(result.caption),
            )
            self.store.set_status(shortcode, ReelStatus.FETCHED)
            self.store.set_job(shortcode, "fetch", JobState.DONE)
        else:
            self.store.set_status(shortcode, _FAILURE_REEL_STATUS[result.fetch_status])
            self.store.set_job(
                shortcode,
                "fetch",
                JobState.DONE if result.fetch_status.unavailable else JobState.FAILED,
                error=result.error,
            )

        self._persist(shortcode, result)
        return result

    # ------------------------------------------------------------------ internals

    def _select(self, *, shortcodes: Iterable[str] | None, force: bool) -> list[dict[str, object]]:
        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            return [reel for reel in reels if str(reel["shortcode"]) in wanted]
        if force:
            return reels
        return [reel for reel in reels if str(reel["status"]) in RETRYABLE_REEL_STATUS]

    def _fetch_with_retries(self, shortcode: str, url: str) -> EnrichmentResult:
        attempts = self.config.enrich.retries + 1
        last: FetchError | None = None

        for attempt in range(1, attempts + 1):
            self._throttle()
            try:
                return self.fetcher.fetch(url, shortcode=shortcode)
            except FetchError as exc:
                last = exc
                logger.warning(
                    "fetch %s failed (attempt %d/%d): %s: %s",
                    shortcode,
                    attempt,
                    attempts,
                    exc.status.value,
                    exc,
                )
                if not exc.retryable or attempt == attempts:
                    break
                self.sleep(self._backoff(attempt))
            except Exception as exc:
                last = FetchError(FetchStatus.ERROR, str(exc))
                logger.warning(
                    "fetch %s crashed (attempt %d/%d): %s", shortcode, attempt, attempts, exc
                )
                if attempt == attempts:
                    break
                self.sleep(self._backoff(attempt))

        assert last is not None
        return EnrichmentResult(
            shortcode=shortcode,
            url=url,
            fetch_status=last.status,
            error=str(last),
        )

    def _backoff(self, attempt: int) -> float:
        base: float = self.config.enrich.backoff_seconds
        return base * (2.0 ** (attempt - 1))

    def _throttle(self) -> None:
        if self._requests > 0 and self.request_delay > 0:
            self.sleep(self.request_delay)
        self._requests += 1

    def _attach_thumbnail(self, shortcode: str, result: EnrichmentResult) -> None:
        if not self.config.enrich.thumbnails or not result.thumbnail:
            return
        destination = self.paths.media(shortcode) / thumbnail_filename(result.thumbnail)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.thumbnail_fetcher.download(result.thumbnail, destination)
        except Exception as exc:
            logger.warning("thumbnail for %s failed: %s", shortcode, exc)
            return
        result.thumbnail_path = relative_key(self.paths.media_dir, destination)

    def _persist(self, shortcode: str, result: EnrichmentResult) -> None:
        payload = result.model_dump(mode="json")
        self.store.save_metadata(shortcode, payload, result.fetch_status.value)
        write_json_atomic(self.paths.reel(shortcode).metadata, payload)


__all__ = ["EnrichmentService", "FetchSummary", "hashtags_from_text", "thumbnail_filename"]
