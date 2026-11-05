"""Phase 6: triage rules, model validation, fallback and manual overrides."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from reel_vault.config.loader import load_config
from reel_vault.importer import ImportReport
from reel_vault.llm.base import ChatError, ChatMessage, ChatResult, ChatUsage
from reel_vault.llm.groq import GroqChatClient
from reel_vault.schemas.common import Decision, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult, FetchStatus
from reel_vault.schemas.source import SourceReel
from reel_vault.schemas.transcript import Transcript, TranscriptSegment
from reel_vault.schemas.triage import (
    ContentType,
    ExtractionField,
    TriageAction,
    TriageMethod,
    TriageResult,
)
from reel_vault.storage.db import Store
from reel_vault.storage.io import read_model
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.triage.prompts import TriageContext, build_messages, repair_messages
from reel_vault.triage.rules import (
    ArchiveRules,
    RuleError,
    load_rules,
)
from reel_vault.triage.service import (
    FALLBACK_ACTIONS,
    TriageService,
    extract_json_object,
    normalise_output,
)

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
        "decision": "extract",
        "value_score": 75,
        "content_type": "tutorial",
        "reason": "several actionable steps",
        "actions": ["transcript", "links"],
        "fields": ["steps", "links"],
        "categories": [{"name": "learning & study", "confidence": 0.9}],
        "archive_reason": None,
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
        "triage": {
            "extract_threshold": 40,
            "needs_review_threshold": 60,
            "max_input_chars": 6000,
        },
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


def _seed(
    store: Store,
    shortcode: str,
    caption: str | None = None,
    *,
    transcript: str | None = None,
    fetch_status: str = "ok",
) -> None:
    metadata = EnrichmentResult(
        shortcode=shortcode,
        url=f"https://www.instagram.com/reel/{shortcode}/",
        fetch_status=FetchStatus(fetch_status),
        description=caption,
        author="creator",
        duration=30.0,
    )
    store.save_metadata(shortcode, metadata.model_dump(mode="json"), fetch_status)
    store.update_reel_enrichment(shortcode, caption=caption)
    if transcript is not None:
        payload = Transcript(
            shortcode=shortcode,
            model="whisper-large-v3-turbo",
            text=transcript,
            segments=[TranscriptSegment(id=0, start=0.0, end=30.0, text=transcript)],
            duration_seconds=30.0,
        )
        store.save_transcript(shortcode, payload.model_dump(mode="json"))
    store.set_status(shortcode, ReelStatus.FETCHED)


def _service(
    store: Store,
    paths: WorkspacePaths,
    config: Any,
    chat: FakeChat,
    *,
    rules: ArchiveRules | None = None,
) -> TriageService:
    return TriageService(
        store,
        paths,
        config,
        client=chat,
        rules=rules,
        sleep=lambda _: None,
    )


def test_emotional_rule_confirmed_archives(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"], collections=["psychology & mind"])
    _seed(store, "AAA111", "feeling sad and lonely tonight")
    chat = FakeChat(
        [
            _json(
                decision="archive",
                value_score=20,
                content_type="story",
                reason="personal reflection",
                actions=[],
                fields=["links"],
                categories=[{"name": "psychology & mind", "confidence": 0.8}],
                archive_reason="emotional post worth keeping",
            )
        ]
    )

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.ARCHIVE
    assert result.method is TriageMethod.RULE
    assert result.fields == []
    assert TriageAction.DOWNLOAD_MEDIA in result.actions
    assert TriageAction.OCR in result.actions
    assert result.archive_reason == "emotional post worth keeping"
    assert result.rule_reason is not None and "archive confirmed" in result.rule_reason
    assert result.needs_review is False
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRIAGED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").triage, TriageResult)
    assert artifact.decision is Decision.ARCHIVE
    assert store.total_cost() == pytest.approx(0.00005)
    job = store.get_job("AAA111", "triage")
    assert job is not None and job["state"] == "done"


def test_emotional_rule_not_confirmed_falls_to_score(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "I love this song so much")
    chat = FakeChat([_json(decision="extract", value_score=70, content_type="opinion")])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.method is TriageMethod.AI
    assert result.rule_reason is not None
    assert "not confirmed" in result.rule_reason
    assert result.needs_review is False


def test_tutorial_rule_forces_extract_and_fields(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "How to focus: step 1, step 2")
    chat = FakeChat(
        [
            _json(
                decision="stub",
                value_score=10,
                content_type="other",
                reason="not sure",
                actions=[],
                fields=[],
            )
        ]
    )

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.method is TriageMethod.RULE
    assert result.value_score == 40
    assert ExtractionField.STEPS in result.fields
    assert ExtractionField.TOOLS in result.fields
    assert ExtractionField.LINKS in result.fields
    assert TriageAction.TRANSCRIPT in result.actions
    assert TriageAction.KEYFRAMES in result.actions
    assert result.needs_review is True


def test_news_rule_forces_links_and_claims(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "Breaking: a new study reports a surprising result")
    chat = FakeChat([_json(decision="stub", value_score=15, content_type="other")])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.method is TriageMethod.RULE
    assert ExtractionField.LINKS in result.fields
    assert ExtractionField.CLAIMS in result.fields


def test_score_at_threshold_extracts_and_marks_review(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat([_json(decision="extract", value_score=40)])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.needs_review is True
    assert result.method is TriageMethod.AI


def test_low_score_stubs_and_clears_actions(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat([_json(decision="extract", value_score=39, actions=["ocr"], fields=["tools"])])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.STUB
    assert result.actions == []
    assert result.fields == []
    assert result.needs_review is False
    assert store.get_reel("AAA111")["status"] == ReelStatus.STUB.value  # type: ignore[index]
    assert store.get_job("AAA111", "triage")["state"] == "done"  # type: ignore[index]


def test_archive_rule_outranks_tutorial_forcing_when_confirmed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "how to cope when you feel sad and lonely")
    chat = FakeChat(
        [
            _json(
                decision="archive",
                value_score=30,
                content_type="story",
                reason="personal reflection with a tip",
                actions=[],
                fields=[],
                archive_reason="emotional post worth keeping",
            )
        ]
    )

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.ARCHIVE
    assert result.method is TriageMethod.RULE
    assert result.fields == []
    assert TriageAction.DOWNLOAD_MEDIA in result.actions
    assert result.rule_reason is not None and "archive confirmed" in result.rule_reason


def test_archive_rule_unconfirmed_still_forces_tutorial_extract(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "how to cope when you feel sad and lonely")
    chat = FakeChat(
        [
            _json(
                decision="extract",
                value_score=80,
                content_type="tutorial",
                reason="actionable coping steps",
                actions=["transcript"],
                fields=["steps"],
            )
        ]
    )

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.method is TriageMethod.RULE
    assert ExtractionField.STEPS in result.fields
    assert result.rule_reason is not None and "forced extract" in result.rule_reason


def test_rule_forced_extract_high_score_is_not_flagged(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "How to focus: step 1, step 2")
    chat = FakeChat([_json(decision="extract", value_score=75)])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.decision is Decision.EXTRACT
    assert result.needs_review is False


def test_manual_override_wins_without_a_model_call(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "How to focus: step 1")
    store.save_review("AAA111", {"decision": "stub", "notes": "seen this already"})
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert chat.calls == []
    assert result.decision is Decision.STUB
    assert result.method is TriageMethod.MANUAL
    assert result.reason == "Manual override: seen this already"
    assert store.get_reel("AAA111")["status"] == ReelStatus.STUB.value  # type: ignore[index]


def test_manual_archive_keeps_media_and_notes(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    store.save_review(
        "AAA111",
        {
            "decision": "archive",
            "notes": "family memory",
            "categories": [{"name": "personal", "confidence": 0.9}],
        },
    )
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert chat.calls == []
    assert result.decision is Decision.ARCHIVE
    assert result.archive_reason == "family memory"
    assert TriageAction.DOWNLOAD_MEDIA in result.actions
    assert result.categories[0].name == "personal"


def test_invalid_json_gets_one_repair(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat(["not json at all", _json(value_score=75)])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert len(chat.calls) == 2
    assert len(chat.calls[1]) == 4
    assert chat.calls[1][2].role == "assistant"
    assert "not valid" in chat.calls[1][3].content
    assert result.decision is Decision.EXTRACT
    assert result.error is None
    assert store.total_cost() == pytest.approx(0.0001)


def test_invalid_json_twice_falls_back_to_standard_plan(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat(["nope", "still nope"])

    summary = _service(store, paths, config, chat).triage_pending()

    result = summary.plans[0]
    assert result.decision is Decision.EXTRACT
    assert result.method is TriageMethod.FALLBACK
    assert result.needs_review is True
    assert result.actions == list(FALLBACK_ACTIONS)
    assert result.fields == [ExtractionField.LINKS]
    assert result.error is None
    assert summary.extracted == 1
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRIAGED.value  # type: ignore[index]
    assert store.total_cost() == pytest.approx(0.0001)


def test_provider_error_marks_triage_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat([ChatError("Groq is down", retryable=False)])

    summary = _service(store, paths, config, chat).triage_pending()

    assert summary.failed == 1
    assert "Groq is down" in summary.failures["AAA111"]
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRIAGE_FAILED.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").triage, TriageResult)
    assert artifact.error == "Groq is down"
    assert store.get_job("AAA111", "triage")["state"] == "failed"  # type: ignore[index]


def test_unavailable_post_is_not_sent_to_the_model(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "gone", fetch_status="private")
    chat = FakeChat([_json()])

    summary = _service(store, paths, config, chat).triage_pending()

    assert chat.calls == []
    assert summary.unavailable == 1
    assert summary.stubbed == 0
    assert store.get_reel("AAA111")["status"] == ReelStatus.UNAVAILABLE.value  # type: ignore[index]
    artifact = read_model(paths.reel("AAA111").triage, TriageResult)
    assert artifact.method is TriageMethod.UNAVAILABLE


def test_missing_metadata_marks_triage_failed(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    chat = FakeChat([_json()])

    result = _service(store, paths, config, chat).triage_one("AAA111")

    assert result.error is not None and "metadata" in result.error
    assert chat.calls == []
    assert store.get_reel("AAA111")["status"] == ReelStatus.TRIAGE_FAILED.value  # type: ignore[index]


def test_dry_run_is_deterministic_and_has_no_side_effects(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "How to focus: step 1, step 2")
    chat = FakeChat([_json()])
    service = _service(store, paths, config, chat)

    summary = service.triage_pending(dry_run=True)

    assert chat.calls == []
    assert summary.plans[0].method is TriageMethod.HEURISTIC
    assert summary.plans[0].decision is Decision.EXTRACT
    assert ExtractionField.STEPS in summary.plans[0].fields
    assert store.get_triage("AAA111") is None
    assert store.get_reel("AAA111")["status"] == ReelStatus.FETCHED.value  # type: ignore[index]
    assert store.total_cost() == 0.0
    assert store.get_job("AAA111", "triage") is None


def test_dry_run_archive_rule_plans_full_media(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "feeling sad tonight")
    chat = FakeChat([_json()])

    summary = _service(store, paths, config, chat).triage_pending(dry_run=True)

    plan = summary.plans[0]
    assert plan.decision is Decision.ARCHIVE
    assert TriageAction.DOWNLOAD_MEDIA in plan.actions
    assert plan.method is TriageMethod.HEURISTIC


def test_plain_run_skips_finished_and_force_retriages(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat([_json()])
    service = _service(store, paths, config, chat)

    assert service.triage_pending().processed == 1
    assert service.triage_pending().processed == 0
    assert service.triage_pending(force=True).processed == 1
    assert len(chat.calls) == 2


def test_transcript_is_included_in_the_prompt(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "caption text", transcript="secret transcript words")
    chat = FakeChat([_json()])

    _service(store, paths, config, chat).triage_one("AAA111")

    user_message = chat.calls[0][1].content
    assert "caption text" in user_message
    assert "secret transcript words" in user_message


def test_unknown_shortcode_raises(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, [])
    service = _service(store, paths, config, FakeChat([_json()]))
    with pytest.raises(LookupError):
        service.triage_one("NOPE")


def test_rules_default_matching_is_word_boundary_aware() -> None:
    rules = ArchiveRules()

    hit = rules.detect("I love this trick")
    assert "love" in hit.archive_keywords
    assert "trick" in hit.tutorial_keywords
    assert hit.forced_extract is True

    miss = rules.detect("warm gloves and a lovely day")
    assert "love" not in miss.archive_keywords

    category = rules.detect("neutral caption", ["psychology & mind"])
    assert "psychology & mind" in category.archive_categories
    assert category.archive_triggered is True


def test_rules_file_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "\n".join(
            [
                "confirm_with_llm: false",
                "archive:",
                "  keywords: [grief, mourning]",
                "  categories: [memory]",
                "extract:",
                "  tutorial_keywords: [lesson]",
                "  news_keywords: [bulletin]",
            ]
        ),
        encoding="utf-8",
    )

    rules = load_rules(path)

    assert rules.confirm_with_llm is False
    assert rules.archive_keywords == ("grief", "mourning")
    assert rules.archive_categories == ("memory",)
    assert rules.tutorial_keywords == ("lesson",)
    assert rules.news_keywords == ("bulletin",)
    assert rules.source == str(path)


def test_rules_file_missing_or_malformed_raises(tmp_path: Path) -> None:
    with pytest.raises(RuleError):
        load_rules(tmp_path / "missing.yaml")

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("archive: [not, a, mapping]", encoding="utf-8")
    with pytest.raises(RuleError):
        load_rules(malformed)


def test_rule_can_force_archive_without_model_confirmation(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "grief and mourning")
    rules = ArchiveRules(
        confirm_with_llm=False,
        archive_keywords=("grief",),
        archive_categories=(),
        tutorial_keywords=(),
        news_keywords=(),
    )
    chat = FakeChat([_json(decision="stub", value_score=10)])

    result = _service(store, paths, config, chat, rules=rules).triage_one("AAA111")

    assert result.decision is Decision.ARCHIVE
    assert result.method is TriageMethod.RULE
    assert result.rule_reason is not None and "archive forced" in result.rule_reason


def test_normalise_output_handles_model_quirks() -> None:
    payload = normalise_output(
        {
            "Decision": "KEEP",
            "Value_Score": "88",
            "Content_Type": "not-a-type",
            "Reason": "  useful   stuff ",
            "Actions": ["download", "transcript", "bogus"],
            "Fields": ["step", "tool", "link", "bogus"],
            "Categories": ["learning", {"name": "focus", "confidence": 2}],
            "Archive_Reason": "",
        }
    )

    assert payload["decision"] == "archive"
    assert payload["value_score"] == 88
    assert payload["content_type"] == "other"
    assert payload["reason"] == "useful stuff"
    assert payload["actions"] == [TriageAction.DOWNLOAD_MEDIA, TriageAction.TRANSCRIPT]
    assert payload["fields"] == [
        ExtractionField.STEPS,
        ExtractionField.TOOLS,
        ExtractionField.LINKS,
    ]
    assert [category.name for category in payload["categories"]] == ["learning", "focus"]
    assert payload["categories"][1].confidence == 1.0
    assert payload["archive_reason"] is None


def test_extract_json_object_handles_fences_and_prose() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here you go: {"a": 2} thanks') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json_object("no object here")


def test_prompt_includes_hints_and_truncates() -> None:
    context = TriageContext(
        shortcode="AAA111",
        caption="caption " * 1000,
        transcript="transcript " * 2000,
        hashtags=["#one"],
        collections=["learning"],
        rule_hints=["tutorial keywords matched: how to"],
        max_chars=2000,
    )

    messages = build_messages(context)

    assert messages[0].role == "system"
    assert "untrusted data" in messages[0].content
    assert messages[1].role == "user"
    assert "AAA111" in messages[1].content
    assert "tutorial keywords matched" in messages[1].content
    assert "[truncated]" in messages[1].content
    assert len(messages[1].content) <= 2300

    repaired = repair_messages(messages, "bad output", "missing decision")
    assert len(repaired) == 4
    assert repaired[2].role == "assistant"
    assert "missing decision" in repaired[3].content


def _groq_config() -> Any:
    return load_config(
        overrides={
            "groq": {
                "retry_backoff_seconds": 1.0,
                "max_retries": 2,
                "chat_input_usd_per_million": 0.15,
                "chat_output_usd_per_million": 0.60,
            }
        }
    )


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="user"),
    ]


def test_groq_chat_client_returns_content_and_cost() -> None:
    config = _groq_config()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    client = GroqChatClient(
        config.groq,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    result = client.complete(_messages(), model="openai/gpt-oss-120b")

    assert result.text == '{"ok": true}'
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 500
    assert result.usage.usd == pytest.approx(0.00045)
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["messages"][0]["role"] == "system"


def test_groq_chat_client_retries_429_with_backoff() -> None:
    config = _groq_config()
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = GroqChatClient(
        config.groq,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    result = client.complete(_messages(), model="m")

    assert result.text == "{}"
    assert calls["n"] == 2
    assert sleeps == [1.0]


def test_groq_chat_client_does_not_retry_400() -> None:
    config = _groq_config()
    sleeps: list[float] = []
    client = GroqChatClient(
        config.groq,
        api_key="test-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text="bad request")),
        sleep=sleeps.append,
    )

    with pytest.raises(ChatError, match="400") as excinfo:
        client.complete(_messages(), model="m")

    assert excinfo.value.retryable is False
    assert sleeps == []


def test_groq_chat_client_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = _groq_config()
    client = GroqChatClient(config.groq)

    with pytest.raises(ChatError, match="GROQ_API_KEY"):
        client.complete(_messages(), model="m")


def test_triage_result_round_trips_through_storage(tmp_path: Path) -> None:
    store, paths, config = _workspace(tmp_path, ["AAA111"])
    _seed(store, "AAA111", "a neutral clip")
    chat = FakeChat([_json(value_score=75)])

    _service(store, paths, config, chat).triage_one("AAA111")

    record = store.get_triage("AAA111")
    assert record is not None
    payload = {key: value for key, value in record.items() if key != "_row"}
    restored = TriageResult.model_validate(payload)
    assert restored.decision is Decision.EXTRACT
    assert restored.content_type is ContentType.TUTORIAL
    assert restored.categories[0].name == "learning & study"
