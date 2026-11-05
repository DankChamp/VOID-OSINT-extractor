"""Phase 3: anonymous metadata enrichment."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reel_vault.config.loader import load_config
from reel_vault.enrich.base import FetchError
from reel_vault.enrich.service import EnrichmentService, hashtags_from_text, thumbnail_filename
from reel_vault.enrich.ytdlp import classify_error, normalise_info
from reel_vault.importer import ImportReport
from reel_vault.schemas.common import ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus, MediaItem
from reel_vault.schemas.source import SourceReel
from reel_vault.storage.db import Store
from reel_vault.storage.io import read_model
from reel_vault.storage.paths import WorkspacePaths


class FakeFetcher:
    def __init__(self, outcomes: dict[str, list[EnrichmentResult | FetchError]]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str]] = []

    def fetch(self, url: str, *, shortcode: str) -> EnrichmentResult:
        self.calls.append((url, shortcode))
        queue = self.outcomes[shortcode]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, FetchError):
            raise outcome
        return outcome


_STORES: list[Store] = []


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while _STORES:
        _STORES.pop().close()


class FakeThumbnails:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, destination: Path) -> int:
        self.calls.append((url, destination))
        if self.fail:
            raise OSError("thumbnail host is down")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-image")
        return len(b"fake-image")


def _result(
    shortcode: str,
    *,
    description: str | None = None,
    author: str | None = None,
    thumbnail: str | None = None,
    is_carousel: bool = False,
) -> EnrichmentResult:
    return EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        description=description,
        author=author,
        thumbnail=thumbnail,
        is_carousel=is_carousel,
        media=[MediaItem(url="https://cdn.example/video.mp4", ext="mp4", width=1080, height=1920)],
    )


def _workspace(tmp_path: Path, codes: list[str]) -> tuple[Store, WorkspacePaths, Any]:
    config = load_config(
        overrides={
            "data_dir": str(tmp_path / "data"),
            "db_path": str(tmp_path / "data" / "state.db"),
            "media_dir": str(tmp_path / "media"),
            "run": {"request_delay_seconds": 0.0},
            "enrich": {"retries": 2, "backoff_seconds": 2.0, "thumbnails": True},
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


def _service(
    store: Store,
    paths: WorkspacePaths,
    config: Any,
    fetcher: FakeFetcher,
    *,
    thumbnails: FakeThumbnails | None = None,
    delay: float = 0.0,
) -> tuple[EnrichmentService, list[float]]:
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    service = EnrichmentService(
        store,
        paths,
        config,
        fetcher=fetcher,
        thumbnail_fetcher=thumbnails or FakeThumbnails(),
        sleep=record_sleep,
        request_delay=delay,
    )
    return service, sleeps


def test_success_persists_metadata_thumbnail_and_hashtags(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher(
        {
            "AAA111": [
                _result(
                    "AAA111",
                    description="Three study tricks #study #focus",
                    author="study_coach",
                    thumbnail="https://cdn.example/thumb.jpg",
                )
            ]
        }
    )
    thumbnails = FakeThumbnails()
    service, _ = _service(store, paths, config, fetcher, thumbnails=thumbnails)

    summary = service.fetch_pending()

    assert summary.as_dict() == {
        "processed": 1,
        "ok": 1,
        "unavailable": 0,
        "failed": 0,
        "skipped": 0,
        "statuses": {"ok": 1},
    }
    reel = store.get_reel("AAA111")
    assert reel is not None
    assert reel["status"] == ReelStatus.FETCHED.value
    assert reel["author"] == "study_coach"
    assert reel["caption"] == "Three study tricks #study #focus"
    assert reel["hashtags"] == ["#focus", "#study"]

    metadata = store.get_metadata("AAA111")
    assert metadata is not None
    assert metadata["fetch_status"] == "ok"
    assert metadata["thumbnail_path"] == "AAA111/thumb.jpg"
    assert metadata["_row"]["fetch_status"] == "ok"

    artifact = read_model(paths.reel("AAA111").metadata, EnrichmentResult)
    assert artifact.thumbnail_path == "AAA111/thumb.jpg"
    assert paths.media("AAA111") / "thumb.jpg" == thumbnails.calls[0][1]
    assert thumbnails.calls[0][1].is_file()

    job = store.get_job("AAA111", "fetch")
    assert job is not None and job["state"] == "done"


def test_failure_marks_one_reel_and_continues(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["PRIVATE1", "GOOD2"])
    fetcher = FakeFetcher(
        {
            "PRIVATE1": [FetchError(FetchStatus.PRIVATE, "This account is private")],
            "GOOD2": [_result("GOOD2", description="keep me")],
        }
    )
    service, _ = _service(store, paths, config, fetcher)

    summary = service.fetch_pending()

    assert summary.ok == 1
    assert summary.unavailable == 1
    assert store.get_reel("PRIVATE1")["status"] == ReelStatus.UNAVAILABLE.value  # type: ignore[index]
    assert store.get_reel("GOOD2")["status"] == ReelStatus.FETCHED.value  # type: ignore[index]
    failed_metadata = store.get_metadata("PRIVATE1")
    assert failed_metadata is not None
    assert failed_metadata["fetch_status"] == "private"
    assert failed_metadata["error"] == "This account is private"
    assert fetcher.calls == [
        ("https://www.instagram.com/reel/PRIVATE1/", "PRIVATE1"),
        ("https://www.instagram.com/reel/GOOD2/", "GOOD2"),
    ]


def test_rate_limit_retries_with_exponential_backoff(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher(
        {
            "AAA111": [
                FetchError(FetchStatus.RATE_LIMITED, "429 too many requests"),
                _result("AAA111", description="finally"),
            ]
        }
    )
    service, sleeps = _service(store, paths, config, fetcher)

    summary = service.fetch_pending()

    assert summary.ok == 1
    assert sleeps == [2.0]
    assert len(fetcher.calls) == 2
    assert store.get_reel("AAA111")["status"] == ReelStatus.FETCHED.value  # type: ignore[index]


def test_permanent_rate_limit_marks_reel_rate_limited(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher(
        {"AAA111": [FetchError(FetchStatus.RATE_LIMITED, "429 too many requests")]}
    )
    service, sleeps = _service(store, paths, config, fetcher)

    summary = service.fetch_pending()

    assert summary.failed == 1
    assert sleeps == [2.0, 4.0]
    assert len(fetcher.calls) == 3
    assert store.get_reel("AAA111")["status"] == ReelStatus.RATE_LIMITED.value  # type: ignore[index]
    job = store.get_job("AAA111", "fetch")
    assert job is not None and job["state"] == "failed"
    assert job["attempts"] == 2


def test_request_delay_is_applied_between_reels(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111", "BBB222"])
    fetcher = FakeFetcher({"AAA111": [_result("AAA111")], "BBB222": [_result("BBB222")]})
    service, sleeps = _service(store, paths, config, fetcher, delay=3.0)

    summary = service.fetch_pending()

    assert summary.ok == 2
    assert sleeps == [3.0]


def test_force_refetches_and_plain_run_skips_finished(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher({"AAA111": [_result("AAA111", description="first")]})
    service, _ = _service(store, paths, config, fetcher)

    assert service.fetch_pending().processed == 1
    assert service.fetch_pending().processed == 0

    fetcher.outcomes["AAA111"] = [_result("AAA111", description="second")]
    assert service.fetch_pending(force=True).processed == 1
    assert store.get_reel("AAA111")["caption"] == "second"  # type: ignore[index]


def test_explicit_shortcode_fetches_even_when_fetched(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher({"AAA111": [_result("AAA111")]})
    service, _ = _service(store, paths, config, fetcher)

    assert service.fetch_pending().processed == 1
    assert service.fetch_pending(shortcodes=["AAA111"]).processed == 1
    assert service.fetch_pending(shortcodes=["NOPE"]).skipped == 1


def test_thumbnail_failure_is_not_fatal(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    fetcher = FakeFetcher(
        {"AAA111": [_result("AAA111", thumbnail="https://cdn.example/thumb.jpg")]}
    )
    service, _ = _service(store, paths, config, fetcher, thumbnails=FakeThumbnails(fail=True))

    summary = service.fetch_pending()

    assert summary.ok == 1
    assert store.get_reel("AAA111")["status"] == ReelStatus.FETCHED.value  # type: ignore[index]
    metadata = store.get_metadata("AAA111")
    assert metadata is not None and metadata["thumbnail_path"] is None


def test_normalise_info_handles_carousels_and_formats() -> None:
    info: dict[str, Any] = {
        "extractor_key": "Instagram",
        "webpage_url": "https://www.instagram.com/p/CAROUSEL/",
        "entries": [
            {
                "title": "first slide",
                "description": "caption #one",
                "uploader": "creator",
                "duration": 12.5,
                "upload_date": "20260910",
                "thumbnail": "https://cdn.example/one.jpg",
                "url": "https://cdn.example/one.jpg",
                "ext": "jpg",
                "width": 1080,
                "height": 1350,
                "formats": [
                    {"url": "https://cdn.example/one.mp4", "ext": "mp4", "filesize": 100},
                    {"url": "https://cdn.example/skip.m3u8", "ext": "m3u8"},
                ],
            },
            {"url": "https://cdn.example/two.jpg", "ext": "jpg"},
        ],
    }

    result = normalise_info(
        info, shortcode="CAROUSEL", url="https://www.instagram.com/p/CAROUSEL/", limit=10
    )

    assert result.fetch_status is FetchStatus.OK
    assert result.is_carousel is True
    assert result.author == "creator"
    assert result.description == "caption #one"
    assert result.upload_date == "2026-09-10"
    assert result.duration == 12.5
    assert result.media[0].url == "https://cdn.example/one.jpg"
    assert {item.url for item in result.media} == {
        "https://cdn.example/one.jpg",
        "https://cdn.example/one.mp4",
        "https://cdn.example/two.jpg",
    }


@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("HTTP Error 429: Too Many Requests", FetchStatus.RATE_LIMITED),
        ("This account is private", FetchStatus.PRIVATE),
        ("Sign in to confirm you are not a bot", FetchStatus.BLOCKED),
        ("Video is not available anymore", FetchStatus.DELETED),
        ("Unsupported URL", FetchStatus.UNSUPPORTED),
        ("something odd happened", FetchStatus.ERROR),
    ],
)
def test_classify_error(message: str, status: FetchStatus) -> None:
    assert classify_error(message) is status


def test_thumbnail_filename_and_hashtags() -> None:
    assert thumbnail_filename("https://cdn.example/a.webp?x=1") == "thumb.webp"
    assert thumbnail_filename("https://cdn.example/a") == "thumb.jpg"
    assert hashtags_from_text("hello #One and #two #one") == ["#One", "#two"]
    assert hashtags_from_text(None) == []


def test_service_rejects_unknown_reel(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, [])
    service, _ = _service(store, paths, config, FakeFetcher({}))
    with pytest.raises(LookupError):
        service.fetch_one("NOPE")


def test_sleep_type_is_used(tmp_path: Path) -> None:
    """Guard against a real ``time.sleep`` sneaking into tests."""
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    service = EnrichmentService(
        store,
        paths,
        config,
        fetcher=FakeFetcher({"AAA111": [_result("AAA111")]}),
        thumbnail_fetcher=FakeThumbnails(),
        sleep=fake_sleep,
        request_delay=1.5,
    )
    service.fetch_pending()
    service.fetch_pending(force=True)
    assert calls == [1.5]


def test_summary_counts_unavailable_separately() -> None:
    from reel_vault.enrich.service import FetchSummary

    summary = FetchSummary()
    summary.record(_result("A"))
    summary.record(
        EnrichmentResult(
            shortcode="B",
            url="https://www.instagram.com/reel/B/",
            fetch_status=FetchStatus.DELETED,
        )
    )
    summary.record(
        EnrichmentResult(
            shortcode="C",
            url="https://www.instagram.com/reel/C/",
            fetch_status=FetchStatus.ERROR,
        )
    )
    assert summary.as_dict()["ok"] == 1
    assert summary.as_dict()["unavailable"] == 1
    assert summary.as_dict()["failed"] == 1
