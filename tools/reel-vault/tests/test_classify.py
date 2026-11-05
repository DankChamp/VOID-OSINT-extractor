"""Phase 8: classification with taxonomy rules + LLM labeling."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from reel_vault.classification.rules import prefilter
from reel_vault.classification.service import ClassificationService
from reel_vault.classification.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
)
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.llm.base import ChatError, ChatMessage, ChatResult, ChatUsage
from reel_vault.schemas.classification import (
    ClassificationMethod,
    ClassificationResult,
)
from reel_vault.schemas.common import ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus
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


def _json(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "primary": {"name": "learning & study tricks", "confidence": 0.9},
        "secondary": [{"name": "tech & AI", "confidence": 0.6}],
        "reason": "a study tutorial",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _workspace(
    tmp_path: Path,
    codes: list[str],
    *,
    collections: list[str] | None = None,
    hashtags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Store, WorkspacePaths, Any]:
    overrides: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "db_path": str(tmp_path / "data" / "state.db"),
        "media_dir": str(tmp_path / "media"),
        "run": {"request_delay_seconds": 0.0},
        "groq": {"retry_backoff_seconds": 0.0, "max_retries": 1},
        "classify": {
            "needs_review_threshold": 0.60,
            "rule_confidence_threshold": 0.75,
            "max_secondary": 4,
        },
        "triage": {"max_input_chars": 6000},
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
            hashtags=hashtags or [],
        )
        for code in codes
    ]
    store.import_report(report)
    return store, paths, config


def _seed_classifiable(
    store: Store,
    shortcode: str,
    *,
    status: str = ReelStatus.TRIAGED.value,
    caption: str = "how to focus while studying",
    transcript: str | None = "here is how you focus",
    fetch_status: str = "ok",
) -> None:
    metadata = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        fetch_status=FetchStatus.OK,
        description=caption,
        author="creator",
        duration=30.0,
    )
    store.save_metadata(shortcode, metadata.model_dump(mode="json"), fetch_status)
    store.update_reel_enrichment(shortcode, caption=caption)
    store.set_status(shortcode, status)
    if transcript is not None:
        from reel_vault.schemas.transcript import Transcript, TranscriptSegment

        payload = Transcript(
            shortcode=shortcode,
            model="whisper-large-v3-turbo",
            text=transcript,
            segments=[TranscriptSegment(id=0, start=0.0, end=5.0, text=transcript)],
            duration_seconds=30.0,
        )
        store.save_transcript(shortcode, payload.model_dump(mode="json"))


def _service(
    store: Store,
    paths: WorkspacePaths,
    config: Any,
    chat: FakeChat,
    *,
    taxonomy: Taxonomy | None = None,
) -> ClassificationService:
    return ClassificationService(
        store,
        paths,
        config,
        client=chat,
        taxonomy=taxonomy,
        sleep=lambda _: None,
    )


def test_rules_prefilter_matches_keywords() -> None:
    taxonomy = load_taxonomy()
    matches = prefilter(
        taxonomy,
        caption="feeling sad and lonely tonight",
        hashtags=["#depressed"],
    )
    assert len(matches) >= 1
    assert matches[0].category == "psychology & mind"
    assert matches[0].confidence >= 0.8


def test_rules_prefilter_matches_collections() -> None:
    taxonomy = load_taxonomy()
    matches = prefilter(
        taxonomy,
        caption="neutral caption",
        collections=["Psych"],
    )
    assert any(match.category == "psychology & mind" for match in matches)


def test_rules_prefilter_returns_empty_on_no_match() -> None:
    taxonomy = load_taxonomy()
    matches = prefilter(taxonomy, caption="just a random clip", collections=[])
    assert matches == []


def test_taxonomy_default_has_starter_categories() -> None:
    taxonomy = load_taxonomy()
    names = taxonomy.names()
    assert "psychology & mind" in names
    assert "learning & study tricks" in names
    assert "other" in names


def test_taxonomy_file_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        "categories:\n  - name: custom cat\n    description: x\n    keywords: [foo]\n",
        encoding="utf-8",
    )
    taxonomy = load_taxonomy(path)
    assert taxonomy.names() == ("custom cat",)


def test_taxonomy_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError):
        load_taxonomy(tmp_path / "missing.yaml")


def test_classify_uses_rule_prefilter_without_model(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"], collections=["Psych"])
    _seed_classifiable(store, "AAA111", caption="feeling sad and lonely tonight")
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert chat.calls == []
    assert result.method is ClassificationMethod.RULE
    assert result.primary is not None
    assert result.primary.name == "psychology & mind"
    assert result.primary.confidence >= 0.75
    assert store.get_reel("AAA111")["status"] == ReelStatus.CLASSIFIED.value  # type: ignore[index]


def test_classify_uses_llm_when_no_rule_match(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip about nothing")
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert len(chat.calls) == 1
    assert result.method is ClassificationMethod.AI
    assert result.primary is not None
    assert result.primary.name == "learning & study tricks"
    assert result.primary.confidence == 0.9
    assert len(result.secondary) == 1
    assert result.secondary[0].name == "tech & AI"


def test_classify_manual_override_wins_without_model(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="how to focus while studying")
    store.save_review(
        "AAA111",
        {
            "primary": "comedy & funny",
            "secondary": ["tech & AI"],
            "notes": "actually a meme",
        },
    )
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert chat.calls == []
    assert result.method is ClassificationMethod.MANUAL
    assert result.primary is not None
    assert result.primary.name == "comedy & funny"
    assert result.primary.confidence == 1.0
    assert result.secondary[0].name == "tech & AI"
    assert result.needs_review is False


def test_classify_needs_review_below_threshold(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip")
    chat = FakeChat(
        [
            _json(
                primary={"name": "other", "confidence": 0.4},
                secondary=[],
                reason="unsure",
            )
        ]
    )

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert result.method is ClassificationMethod.AI
    assert result.needs_review is True
    assert result.primary is not None
    assert result.primary.name == "other"


def test_classify_invalid_json_falls_back(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip")
    chat = FakeChat(["not json", _json()])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert len(chat.calls) == 2
    assert result.method is ClassificationMethod.AI
    assert result.primary is not None
    assert store.get_reel("AAA111")["status"] == ReelStatus.CLASSIFIED.value  # type: ignore[index]


def test_classify_provider_error_marks_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip")
    chat = FakeChat([ChatError("Groq is down", retryable=False)])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert result.error == "Groq is down"
    assert result.needs_review is True
    assert store.get_reel("AAA111")["status"] == ReelStatus.CLASSIFICATION_FAILED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").classification, ClassificationResult)
    assert artifact.error == "Groq is down"


def test_classify_pending_skips_finished_and_force_retries(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip about nothing")
    chat = FakeChat([_json()])
    service = _service(store, paths, config, chat)

    assert service.classify_pending().processed == 1
    assert service.classify_pending().processed == 0
    assert service.classify_pending(force=True).processed == 1
    assert len(chat.calls) == 2


def test_unknown_shortcode_raises(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, [])
    service = _service(store, paths, config, FakeChat([_json()]))
    with pytest.raises(LookupError):
        service.classify_one("NOPE")


def test_missing_metadata_marks_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert result.error is not None and "metadata" in result.error
    assert store.get_reel("AAA111")["status"] == ReelStatus.CLASSIFICATION_FAILED.value  # type: ignore[index]


def test_classification_result_round_trips_through_storage(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption="random clip about nothing")
    chat = FakeChat([_json()])

    _service(store, paths, config, chat).classify_one("AAA111")

    record = store.get_classification("AAA111")
    assert record is not None
    assert record["primary"] == "learning & study tricks"
    assert record["method"] == "ai"
    assert record["confidence"] == 0.9

    restored = read_model(paths.reel("AAA111").classification, ClassificationResult)
    assert restored.primary is not None
    assert restored.primary.name == "learning & study tricks"
    assert restored.secondary[0].name == "tech & AI"


def test_classify_uses_transcript_when_caption_empty(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed_classifiable(store, "AAA111", caption=None, transcript="random stuff here")
    chat = FakeChat([_json(primary={"name": "tech & AI", "confidence": 0.95})])

    result = _service(store, paths, config, chat).classify_one("AAA111")

    assert result.method is ClassificationMethod.AI
    assert result.primary is not None
    assert result.primary.name == "tech & AI"
    user_message = chat.calls[0][1].content
    assert "random stuff here" in user_message


def test_canonical_label_matches_case_insensitively() -> None:
    taxonomy = load_taxonomy()
    names = taxonomy.names()
    aliases = {name.casefold(): name for name in names}
    assert aliases["learning & study tricks"] == "learning & study tricks"
    assert aliases["LEARNING & STUDY TRICKS".casefold()] == "learning & study tricks"


__all__ = []
