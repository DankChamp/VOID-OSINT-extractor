"""Phase 4: media download, verification and external-drive safety."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.media.base import DriveUnavailable, MediaDownloadError
from reel_vault.media.downloader import canonical_name
from reel_vault.media.drive import check_reachable
from reel_vault.media.service import MediaService
from reel_vault.media.verify import kind_for, sha256_file
from reel_vault.schemas.common import ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.media import MediaManifest
from reel_vault.schemas.source import SourceReel
from reel_vault.storage.db import Store
from reel_vault.storage.io import read_model
from reel_vault.storage.paths import WorkspacePaths

_STORES: list[Store] = []


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while _STORES:
        _STORES.pop().close()


class FakeDownloader:
    def __init__(self, files: tuple[str, ...] = ("abc.mp4",), *, fail: bool = False) -> None:
        self.files = files
        self.fail = fail
        self.calls: list[tuple[str, Path, str]] = []

    def download(self, result: EnrichmentResult, destination: Path, *, quality: str) -> list[Path]:
        self.calls.append((result.shortcode, destination, quality))
        if self.fail:
            raise MediaDownloadError("network exploded")
        written: list[Path] = []
        for name in self.files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"media-bytes")
            written.append(target)
        return written


class FakeVerifier:
    def __init__(self, results: list[bool] | None = None, *, default: bool = True) -> None:
        self.results = list(results or [])
        self.default = default
        self.calls: list[tuple[Path, str]] = []

    def verify(self, path: Path, *, kind: str) -> bool:
        self.calls.append((path, kind))
        if self.results:
            return self.results.pop(0)
        return self.default and path.is_file() and path.stat().st_size > 0


def _workspace(tmp_path: Path, codes: list[str]) -> tuple[Store, WorkspacePaths, Any]:
    config = load_config(
        overrides={
            "data_dir": str(tmp_path / "data"),
            "db_path": str(tmp_path / "data" / "state.db"),
            "media_dir": str(tmp_path / "media"),
            "run": {"request_delay_seconds": 0.0},
            "download": {"retries": 1, "drive_wait_seconds": 0.0, "verify": True},
            "enrich": {"thumbnails": False},
        }
    )
    paths = WorkspacePaths(config.data_dir, config.db_path, config.media_dir).ensure()
    store = Store(paths.database)
    _STORES.append(store)
    report = ImportReport(source=str(tmp_path / "export"))
    report.reels = [
        SourceReel(
            shortcode=code,
            url=f"https://www.instagram.com/reel/{code}/",
            source_import=report.imported_at,
        )
        for code in codes
    ]
    store.import_report(report)
    return store, paths, config


def _seed_metadata(
    store: Store, shortcode: str, *, item_count: int = 1, is_carousel: bool = False
) -> None:
    result = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        item_count=item_count,
        is_carousel=is_carousel,
    )
    store.save_metadata(shortcode, result.model_dump(mode="json"), "ok")


def _service(
    store: Store,
    paths: WorkspacePaths,
    config: Any,
    downloader: FakeDownloader,
    verifier: FakeVerifier | None = None,
) -> MediaService:
    return MediaService(
        store,
        paths,
        config,
        downloader=downloader,
        verifier=verifier or FakeVerifier(),
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )


def test_none_mode_skips_everything(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    downloader = FakeDownloader()
    service = _service(store, paths, config, downloader)

    summary = service.download_pending(mode="none")

    assert summary.skipped == 1
    assert summary.downloaded == 0
    assert downloader.calls == []
    assert not paths.media("AAA111").exists()


def test_all_mode_downloads_verifies_and_records_manifest(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    downloader = FakeDownloader(("abc.mp4",))
    verifier = FakeVerifier()
    service = _service(store, paths, config, downloader, verifier)

    summary = service.download_pending(mode="all", quality="720")

    assert summary.downloaded == 1
    assert summary.bytes == len(b"media-bytes")
    assert downloader.calls[0][2] == "720"
    final = paths.media("AAA111") / "video.mp4"
    assert final.is_file()
    assert verifier.calls[0][1] == "video"

    manifest = read_model(paths.reel("AAA111").media_manifest, MediaManifest)
    assert manifest.verified is True
    assert manifest.files[0].path == "AAA111/video.mp4"
    assert manifest.files[0].kind == "video"
    assert manifest.files[0].bytes == len(b"media-bytes")
    assert manifest.files[0].sha256 == sha256_file(final)

    record = store.get_media("AAA111")
    assert record is not None
    assert record["_row"]["verified"] == 1
    assert record["bytes"] == len(b"media-bytes")
    job = store.get_job("AAA111", "media")
    assert job is not None and job["state"] == "done"
    assert not [path for path in paths.media_dir.iterdir() if path.name.startswith(".tmp-")]


def test_archive_mode_requires_a_triage_decision(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    downloader = FakeDownloader()
    service = _service(store, paths, config, downloader)

    assert service.download_pending(mode="archive").skipped == 1
    assert downloader.calls == []

    store.save_triage("AAA111", {"decision": "archive", "value_score": 70})
    assert service.download_pending(mode="archive").downloaded == 1
    assert len(downloader.calls) == 1


def test_carousel_count_mismatch_fails_after_one_retry(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111", item_count=2, is_carousel=True)
    downloader = FakeDownloader(("one.jpg",))
    service = _service(store, paths, config, downloader)

    summary = service.download_pending(mode="all")

    assert summary.failed == 1
    assert len(downloader.calls) == 2
    assert store.get_reel("AAA111")["status"] == ReelStatus.MEDIA_FAILED.value  # type: ignore[index]
    job = store.get_job("AAA111", "media")
    assert job is not None and job["state"] == "failed"
    failed = read_model(paths.reel("AAA111").media_manifest, MediaManifest)
    assert failed.verified is False
    assert failed.error is not None and "carousel" in failed.error
    assert not [path for path in paths.media_dir.iterdir() if path.name.startswith(".tmp-")]


def test_verification_failure_retries_once_then_succeeds(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    downloader = FakeDownloader()
    verifier = FakeVerifier(results=[False, True])
    service = _service(store, paths, config, downloader, verifier)

    summary = service.download_pending(mode="all")

    assert summary.downloaded == 1
    assert len(downloader.calls) == 2
    assert (paths.media("AAA111") / "video.mp4").is_file()


def test_downloader_error_marks_failure_and_continues(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["BAD1", "GOOD2"])
    _seed_metadata(store, "BAD1")
    _seed_metadata(store, "GOOD2")
    downloader = FakeDownloader()
    service = _service(store, paths, config, downloader)
    # First reel always fails; the second succeeds.
    original = downloader.download
    calls = {"n": 0}

    def flaky(result: EnrichmentResult, destination: Path, *, quality: str) -> list[Path]:
        calls["n"] += 1
        if result.shortcode == "BAD1":
            raise MediaDownloadError("network exploded")
        return original(result, destination, quality=quality)

    downloader.download = flaky  # type: ignore[method-assign]
    summary = service.download_pending(mode="all")

    assert summary.failed == 1
    assert summary.downloaded == 1
    assert "BAD1" in summary.failures
    assert store.get_reel("BAD1")["status"] == ReelStatus.MEDIA_FAILED.value  # type: ignore[index]
    assert store.get_reel("GOOD2")["status"] == ReelStatus.NEW.value  # type: ignore[index]


def test_force_redownloads_existing_media(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    downloader = FakeDownloader()
    service = _service(store, paths, config, downloader)

    assert service.download_pending(mode="all").downloaded == 1
    assert service.download_pending(mode="all").skipped == 1
    assert service.download_pending(mode="all", force=True).downloaded == 1
    assert len(downloader.calls) == 2


def test_missing_media_file_is_redownloaded(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    downloader = FakeDownloader()
    service = _service(store, paths, config, downloader)

    assert service.download_pending(mode="all").downloaded == 1
    (paths.media("AAA111") / "video.mp4").unlink()
    assert service.download_pending(mode="all").downloaded == 1
    assert len(downloader.calls) == 2


def test_missing_metadata_is_reported(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    service = _service(store, paths, config, FakeDownloader())

    summary = service.download_pending(mode="all")

    assert summary.failed == 1
    assert "metadata" in summary.failures["AAA111"]


def test_unreachable_drive_raises_drive_unavailable(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_metadata(store, "AAA111")
    shutil.rmtree(paths.media_dir)
    service = _service(store, paths, config, FakeDownloader())

    with pytest.raises(DriveUnavailable):
        service.download_pending(mode="all")


def test_check_reachable(tmp_path: Path) -> None:
    assert check_reachable(tmp_path)
    missing = tmp_path / "missing"
    assert not check_reachable(missing)


def test_canonical_names_are_stable() -> None:
    counters: dict[str, int] = {}
    assert canonical_name(Path("a.mp4"), counters) == "video.mp4"
    assert canonical_name(Path("b.mp4"), counters) == "video-02.mp4"
    counters = {}
    assert canonical_name(Path("a.jpg"), counters) == "image-01.jpg"
    assert canonical_name(Path("b.jpg"), counters) == "image-02.jpg"
    counters = {}
    assert canonical_name(Path("a.m4a"), counters) == "audio.m4a"
    assert canonical_name(Path("a.xyz"), counters) == "file-01.xyz"


def test_kind_detection() -> None:
    assert kind_for(Path("a.mp4")) == "video"
    assert kind_for(Path("a.M4A")) == "audio"
    assert kind_for(Path("a.webp")) == "image"
    assert kind_for(Path("a.txt")) == "other"
