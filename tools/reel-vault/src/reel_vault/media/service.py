"""Media storage: download, verify, move into place and record the manifest."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from reel_vault.config.models import AppConfig, DownloadMode, Quality
from reel_vault.logging_setup import get_logger
from reel_vault.media.base import (
    DriveUnavailable,
    MediaDownloader,
    MediaDownloadError,
    MediaVerifier,
)
from reel_vault.media.downloader import YtDlpMediaDownloader, canonical_name
from reel_vault.media.drive import check_reachable, require_reachable
from reel_vault.media.verify import FfprobeVerifier, kind_for, sha256_file
from reel_vault.schemas.common import JobState, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.media import MediaFile, MediaManifest
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_model_atomic
from reel_vault.storage.paths import WorkspacePaths, relative_key

logger = get_logger(__name__)


@dataclass
class MediaSummary:
    processed: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "bytes": self.bytes,
            "failures": dict(sorted(self.failures.items())),
        }


class MediaService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        downloader: MediaDownloader | None = None,
        verifier: MediaVerifier | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.downloader = downloader or YtDlpMediaDownloader(
            timeout_seconds=max(60.0, config.enrich.timeout_seconds)
        )
        self.verifier = verifier or FfprobeVerifier()
        self.sleep = sleep
        self.clock = clock

    # ------------------------------------------------------------------ selection

    def should_download(self, shortcode: str, mode: DownloadMode) -> bool:
        if mode == "none":
            return False
        if mode == "all":
            return True
        triage = self.store.get_triage(shortcode)
        return bool(triage and triage.get("decision") == "archive")

    def download_pending(
        self,
        *,
        mode: DownloadMode | None = None,
        quality: Quality | None = None,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
    ) -> MediaSummary:
        mode = mode or self.config.download.mode
        quality = quality or self.config.download.quality
        summary = MediaSummary()

        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            reels = [reel for reel in reels if str(reel["shortcode"]) in wanted]
        if limit is not None:
            reels = reels[: max(0, limit)]

        if mode == "none":
            summary.skipped = len(reels)
            return summary

        for reel in reels:
            shortcode = str(reel["shortcode"])
            if not force and not self.should_download(shortcode, mode):
                summary.skipped += 1
                continue
            if not force and self._already_downloaded(shortcode):
                summary.skipped += 1
                continue

            require_reachable(
                self.paths.media_dir,
                wait_seconds=self.config.download.drive_wait_seconds,
                sleep=self.sleep,
                clock=self.clock,
            )
            summary.processed += 1
            try:
                manifest = self.download_one(shortcode, mode=mode, quality=quality, force=force)
            except DriveUnavailable:
                raise
            except MediaDownloadError as exc:
                if not check_reachable(self.paths.media_dir):
                    raise DriveUnavailable(
                        f"media directory disappeared during download: {self.paths.media_dir}"
                    ) from exc
                summary.failed += 1
                summary.failures[shortcode] = str(exc)
                logger.warning("media %s failed: %s", shortcode, exc)
                continue
            summary.downloaded += 1
            summary.bytes += manifest.bytes
        return summary

    # ------------------------------------------------------------------ one reel

    def download_one(
        self, shortcode: str, *, mode: DownloadMode, quality: Quality, force: bool = False
    ) -> MediaManifest:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise MediaDownloadError(f"unknown shortcode: {shortcode}")
        metadata = self.store.get_metadata(shortcode)
        if metadata is None or metadata.get("fetch_status") != "ok":
            raise MediaDownloadError("metadata is missing or the post is unavailable")

        payload = {key: value for key, value in metadata.items() if key != "_row"}
        result = EnrichmentResult.model_validate(payload)
        expected = max(1, result.item_count)
        self.store.set_job(shortcode, "media", JobState.RUNNING)

        attempts = self.config.download.retries + 1
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                manifest = self._attempt(shortcode, result, mode=mode, quality=quality)
            except MediaDownloadError as exc:
                last_error = str(exc)
                logger.warning(
                    "media %s attempt %d/%d failed: %s", shortcode, attempt, attempts, exc
                )
                continue
            if manifest.verified and manifest.expected_items <= len(manifest.files):
                self.store.save_media(
                    shortcode,
                    manifest.model_dump(mode="json"),
                    bytes_written=manifest.bytes,
                    verified=True,
                )
                self.store.set_job(shortcode, "media", JobState.DONE)
                return manifest
            last_error = manifest.error or "verification failed"

        self.store.set_status(shortcode, ReelStatus.MEDIA_FAILED)
        self.store.set_job(shortcode, "media", JobState.FAILED, error=last_error)
        failed = MediaManifest(
            shortcode=shortcode,
            mode=mode,
            quality=quality,
            expected_items=expected,
            error=last_error or "verification failed",
        )
        write_model_atomic(self.paths.reel(shortcode).media_manifest, failed)
        raise MediaDownloadError(last_error or "verification failed")

    def _attempt(
        self, shortcode: str, result: EnrichmentResult, *, mode: DownloadMode, quality: Quality
    ) -> MediaManifest:
        final_dir = self.paths.media(shortcode)
        temp_dir = self.paths.media_dir / f".tmp-{shortcode}-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise DriveUnavailable(
                f"media directory is not writable: {self.paths.media_dir}"
            ) from exc
        try:
            downloaded = self.downloader.download(result, temp_dir, quality=quality)
            if not downloaded:
                raise MediaDownloadError("downloader returned no files")

            expected = max(1, result.item_count)
            if result.is_carousel and len(downloaded) < expected:
                raise MediaDownloadError(
                    f"carousel expected {expected} items, downloaded {len(downloaded)}"
                )

            files: list[MediaFile] = []
            counters: dict[str, int] = {}
            for path in downloaded:
                kind = kind_for(path)
                if not self.verifier.verify(path, kind=kind):
                    raise MediaDownloadError(f"verification failed for {path.name}")
                size = path.stat().st_size
                checksum = sha256_file(path) if self.config.download.verify else None
                files.append(
                    MediaFile(
                        path=canonical_name(path, counters),
                        kind=kind,
                        bytes=size,
                        sha256=checksum,
                    )
                )

            try:
                final_dir.mkdir(parents=True, exist_ok=True)
                for existing in final_dir.iterdir():
                    if existing.is_file() and existing.name.startswith(
                        ("video", "audio", "image-", "file-")
                    ):
                        existing.unlink(missing_ok=True)
                for source, item in zip(downloaded, files, strict=True):
                    os.replace(source, final_dir / item.path)
                    item.path = relative_key(self.paths.media_dir, final_dir / item.path)
            except OSError as exc:
                if not check_reachable(self.paths.media_dir):
                    raise DriveUnavailable(
                        f"media directory disappeared while moving files: {self.paths.media_dir}"
                    ) from exc
                raise MediaDownloadError(f"cannot move media into place: {exc}") from exc

            manifest = MediaManifest(
                shortcode=shortcode,
                mode=mode,
                quality=quality,
                files=files,
                expected_items=expected,
                verified=True,
                bytes=sum(item.bytes for item in files),
            )
            write_model_atomic(self.paths.reel(shortcode).media_manifest, manifest)
            return manifest
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _already_downloaded(self, shortcode: str) -> bool:
        record = self.store.get_media(shortcode)
        if record is None:
            return False
        if not record.get("_row", {}).get("verified"):
            return False
        files = record.get("files")
        if not isinstance(files, list) or not files:
            return False
        return all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and (self.paths.media_dir / item["path"]).is_file()
            for item in files
        )


__all__ = ["MediaService", "MediaSummary"]
