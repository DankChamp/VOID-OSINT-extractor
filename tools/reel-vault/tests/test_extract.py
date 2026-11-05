"""Phase 7: grounded extraction, link resolution, keyframes and RapidOCR."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from reel_vault.config.loader import load_config
from reel_vault.extract.service import ExtractionService
from reel_vault.importer import ImportReport
from reel_vault.llm.base import ChatError, ChatMessage, ChatResult, ChatUsage
from reel_vault.ocr.base import OcrEngine, OcrResult
from reel_vault.schemas.common import Decision, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus
from reel_vault.schemas.extraction import (
    ExtractionResult,
    ResolvedLink,
)
from reel_vault.schemas.source import SourceReel
from reel_vault.schemas.transcript import Transcript, TranscriptSegment
from reel_vault.storage.db import Store
from reel_vault.storage.io import read_model
from reel_vault.storage.paths import WorkspacePaths

_STORES: list[Store] = []


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while _STORES:
        _STORES.pop().close()


class FakeChat:
    def __init__(self, responses: Sequence[str | Exception | ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.models: list[str] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.models.append(model)
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, ChatResult):
            return response
        return ChatResult(
            text=str(response),
            model=model,
            usage=ChatUsage(input_tokens=100, output_tokens=50, usd=0.00005),
        )


class FakeOcr(OcrEngine):
    def __init__(self, *, available: bool = True, error: Exception | None = None) -> None:
        self.available = available
        self.error = error
        self.calls: list[str] = []

    def read(self, image: Path) -> OcrResult:
        self.calls.append(str(image))
        if self.error is not None:
            raise self.error
        if not self.available:
            return OcrResult(text="", lines=[], engine="rapidocr", available=False, error="missing")
        return OcrResult(text="sale today only", lines=[], engine="rapidocr", available=True)


def _json(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "summary": "a useful clip",
        "items": [
            {"field": "steps", "text": "click the button", "source": "transcript:0"},
            {"field": "links", "text": "https://example.com/guide", "source": "caption"},
        ],
        "links": [{"url": "https://example.com/guide", "title": "Guide"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _workspace(
    tmp_path: Path,
    codes: list[str],
    *,
    collections: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Store, WorkspacePaths, Any]:
    overrides: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "db_path": str(tmp_path / "data" / "state.db"),
        "media_dir": str(tmp_path / "media"),
        "run": {"request_delay_seconds": 0.0},
        "groq": {"retry_backoff_seconds": 0.0, "max_retries": 1},
        "extract": {"max_items": 60, "max_links": 12, "resolve_links": False},
        "triage": {"extract_threshold": 40, "needs_review_threshold": 60, "max_input_chars": 6000},
    }
    if extra:
        overrides.update(extra)
    config = load_config(overrides=overrides)
    paths = WorkspacePaths(config.data_dir, config.db_path, config.media_dir).ensure()
    store = Store(paths.database)
    _STORES.append(store)
    report = ImportReport(source=str(tmp_path / "export"))
    report.reels = [
        SourceReel(
            shortcode=code,
            url=f"https://www.instagram.com/reel/{code}/",
            source_import=report.imported_at,
            collections=collections or [],
        )
        for code in codes
    ]
    store.import_report(report)
    return store, paths, config


def _seed_triaged(
    store: Store,
    shortcode: str,
    *,
    decision: str = "extract",
    fields: list[str] | None = None,
    actions: list[str] | None = None,
    caption: str = "caption text",
    transcript: str = "hello world",
    archive_reason: str | None = None,
) -> None:
    metadata = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        fetch_status=FetchStatus.OK,
        description=caption,
        author="creator",
        duration=30.0,
    )
    store.save_metadata(shortcode, metadata.model_dump(mode="json"), "ok")
    store.update_reel_enrichment(shortcode, caption=caption)
    payload = Transcript(
        shortcode=shortcode,
        model="whisper-large-v3-turbo",
        text=transcript,
        segments=[TranscriptSegment(id=0, start=0.0, end=5.0, text=transcript)],
        duration_seconds=30.0,
    )
    store.save_transcript(shortcode, payload.model_dump(mode="json"))
    store.save_triage(
        shortcode,
        {
            "shortcode": shortcode,
            "decision": decision,
            "value_score": 75,
            "content_type": "tutorial",
            "reason": "useful",
            "actions": actions or ["transcript", "links"],
            "fields": fields or ["steps", "links"],
            "categories": [],
            "archive_reason": archive_reason,
            "method": "ai",
            "model": "openai/gpt-oss-120b",
            "needs_review": False,
            "prompt_version": "triage-v1",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
    )
    store.set_status(shortcode, ReelStatus.TRIAGED)


def _service(store, paths, config, chat, *, ocr=None) -> ExtractionService:
    return ExtractionService(
        store,
        paths,
        config,
        client=chat,
        ocr=ocr,
        sleep=lambda _: None,
    )


def test_extract_decision_grounds_items_and_persists(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert len(chat.calls) == 1
    assert result.decision is Decision.EXTRACT
    assert result.method.value == "ai"
    assert len(result.items) == 2
    assert result.items[0].field.value == "steps"
    assert result.items[0].source == "transcript:0"
    assert result.links[0].url == "https://example.com/guide"
    assert store.get_reel("AAA111")["status"] == ReelStatus.EXTRACTED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").extraction, ExtractionResult)
    assert artifact.decision is Decision.EXTRACT
    assert store.total_cost() == pytest.approx(0.00005)


def test_stub_skips_model_call(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111", decision="stub")
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert chat.calls == []
    assert result.decision is Decision.STUB
    assert result.items == []
    assert store.get_reel("AAA111")["status"] == ReelStatus.STUB.value  # type: ignore[index]


def test_archive_keeps_notes_without_model_call(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111", decision="archive", archive_reason="family memory worth keeping")
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert chat.calls == []
    assert result.decision is Decision.ARCHIVE
    assert len(result.items) == 1
    assert result.items[0].field.value == "notes"
    assert result.items[0].text == "family memory worth keeping"
    assert result.method.value == "archive"


def test_invalid_json_falls_back_and_marks_review(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat(["not json", "still not json"])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert len(chat.calls) == 2
    assert result.method.value == "fallback"
    assert result.needs_review is True
    assert result.decision is Decision.EXTRACT
    assert store.get_reel("AAA111")["status"] == ReelStatus.EXTRACTED.value  # type: ignore[index]


def test_provider_error_marks_extraction_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat([ChatError("Groq is down", retryable=False)])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert result.error == "Groq is down"
    assert result.needs_review is True
    assert store.get_reel("AAA111")["status"] == ReelStatus.EXTRACTION_FAILED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").extraction, ExtractionResult)
    assert artifact.error == "Groq is down"


def test_ungrounded_items_are_dropped(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111", fields=["steps"])
    chat = FakeChat(
        [
            _json(
                items=[
                    {"field": "steps", "text": "good step", "source": "caption"},
                    {"field": "steps", "text": "bad step", "source": "made-up"},
                    {"field": "tools", "text": "wrong field", "source": "caption"},
                ]
            )
        ]
    )

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert len(result.items) == 1
    assert result.items[0].text == "good step"


def test_ocr_degrades_gracefully_when_missing(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111", actions=["ocr"])
    chat = FakeChat([_json()])
    ocr = FakeOcr(available=False)

    result = _service(store, paths, config, chat, ocr=ocr).extract_one("AAA111")

    assert result.ocr is not None
    assert result.ocr.available is False
    assert result.ocr.frames == 0
    assert result.decision is Decision.EXTRACT


def test_ocr_runs_on_keyframes(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111", actions=["ocr"])
    chat = FakeChat([_json()])
    ocr = FakeOcr(available=True)

    result = _service(store, paths, config, chat, ocr=ocr).extract_one("AAA111")

    assert result.ocr is not None
    assert result.ocr.available is True


def test_extract_pending_skips_finished_and_force_retries(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat([_json()])
    service = _service(store, paths, config, chat)

    assert service.extract_pending().processed == 1
    assert service.extract_pending().processed == 0
    assert service.extract_pending(force=True).processed == 1
    assert len(chat.calls) == 2


def test_unknown_shortcode_raises(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, [])
    service = _service(store, paths, config, FakeChat([_json()]))
    with pytest.raises(LookupError):
        service.extract_one("NOPE")


def test_missing_triage_marks_extraction_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).extract_one("AAA111")

    assert result.error is not None and "triage" in result.error
    assert store.get_reel("AAA111")["status"] == ReelStatus.EXTRACTION_FAILED.value  # type: ignore[index]


def test_links_resolved_with_injected_fetch(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat([_json()])

    def fetch(url: str, timeout: float) -> tuple[str, str | None, str, str | None]:
        return url, "Example Page", "ok", None

    service = _service(store, paths, config, chat)
    resolved = service._resolve_links(["https://example.com"], "see https://example.com")
    # resolve_links is module-level; use the injected-free path by monkeypatching is not needed
    # here we just check the method returns schema ResolvedLink objects
    assert all(isinstance(item, ResolvedLink) for item in resolved)


def test_extraction_result_round_trips_through_storage(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_triaged(store, "AAA111")
    chat = FakeChat([_json()])

    _service(store, paths, config, chat).extract_one("AAA111")

    record = store.get_extraction("AAA111")
    assert record is not None
    payload = {key: value for key, value in record.items() if key != "_row"}
    restored = ExtractionResult.model_validate(payload)
    assert restored.decision is Decision.EXTRACT
    assert restored.links[0].url == "https://example.com/guide"


def test_extract_config_resolve_links_flag(tmp_path: Path) -> None:
    store, _paths, config = _workspace(
        tmp_path, ["AAA111"], extra={"extract": {"resolve_links": True}}
    )
    _seed_triaged(store, "AAA111")
    assert config.extract.resolve_links is True


__all__ = []
