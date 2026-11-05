"""Triage orchestration.

Decision order:

1. unavailable post -> ``unavailable``
2. emotional/personal rule confirmed by the model -> ``archive``
3. tutorial/news rule -> forced ``extract`` with the fields that matter
4. model score against ``extract_threshold``
5. manual review override, which always wins

Invalid model JSON gets one repair attempt and then a deterministic Standard plan.
A provider outage is different: the reel is marked ``triage_failed`` and retried later.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import ValidationError

from reel_vault.config.models import AppConfig
from reel_vault.llm.base import ChatClient, ChatError, ChatUsage
from reel_vault.llm.groq import GroqChatClient
from reel_vault.logging_setup import get_logger
from reel_vault.schemas.common import Decision, JobState, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.triage import (
    CategoryHint,
    ContentType,
    ExtractionField,
    TriageAction,
    TriageMethod,
    TriageOutput,
    TriageResult,
)
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_model_atomic
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.triage.prompts import (
    PROMPT_VERSION,
    TriageContext,
    build_messages,
    repair_messages,
)
from reel_vault.triage.rules import ArchiveRules, RuleMatches, load_rules

logger = get_logger(__name__)

TRIAGE_STATUSES = frozenset(
    {
        ReelStatus.FETCHED.value,
        ReelStatus.TRANSCRIBED.value,
        ReelStatus.TRANSCRIPT_FAILED.value,
        ReelStatus.MEDIA_FAILED.value,
        ReelStatus.TRIAGE_FAILED.value,
    }
)

UNAVAILABLE_FETCH_STATUSES = frozenset({"private", "deleted", "unsupported", "blocked"})

ARCHIVE_ACTIONS: tuple[TriageAction, ...] = (
    TriageAction.TRANSCRIPT,
    TriageAction.OCR,
    TriageAction.KEYFRAMES,
    TriageAction.DOWNLOAD_MEDIA,
)
TUTORIAL_FIELDS: tuple[ExtractionField, ...] = (
    ExtractionField.STEPS,
    ExtractionField.TOOLS,
    ExtractionField.LINKS,
)
NEWS_FIELDS: tuple[ExtractionField, ...] = (
    ExtractionField.LINKS,
    ExtractionField.CLAIMS,
)
FALLBACK_ACTIONS: tuple[TriageAction, ...] = (
    TriageAction.TRANSCRIPT,
    TriageAction.LINKS,
    TriageAction.KEYFRAMES,
)

_ACTION_ALIASES: dict[str, str] = {
    "download": TriageAction.DOWNLOAD_MEDIA.value,
    "media": TriageAction.DOWNLOAD_MEDIA.value,
    "video": TriageAction.DOWNLOAD_MEDIA.value,
    "frames": TriageAction.KEYFRAMES.value,
    "keyframe": TriageAction.KEYFRAMES.value,
    "text": TriageAction.TRANSCRIPT.value,
    "speech": TriageAction.TRANSCRIPT.value,
    "ocr_text": TriageAction.OCR.value,
}
_FIELD_ALIASES: dict[str, str] = {
    "step": ExtractionField.STEPS.value,
    "trick": ExtractionField.TRICKS.value,
    "tool": ExtractionField.TOOLS.value,
    "link": ExtractionField.LINKS.value,
    "claim": ExtractionField.CLAIMS.value,
    "quote": ExtractionField.QUOTES.value,
    "recipe": ExtractionField.RECIPES.value,
    "person": ExtractionField.PEOPLE.value,
    "book": ExtractionField.BOOKS.value,
    "place": ExtractionField.PLACES.value,
    "product": ExtractionField.PRODUCTS.value,
    "snippet": ExtractionField.CODE.value,
    "note": ExtractionField.NOTES.value,
    "summary": ExtractionField.NOTES.value,
}
_DECISION_ALIASES: dict[str, str] = {
    "keep": Decision.ARCHIVE.value,
    "save": Decision.ARCHIVE.value,
    "skip": Decision.STUB.value,
    "ignore": Decision.STUB.value,
    "none": Decision.STUB.value,
    "low_value": Decision.STUB.value,
    "notes": Decision.EXTRACT.value,
    "knowledge": Decision.EXTRACT.value,
    "process": Decision.EXTRACT.value,
}

_T = TypeVar("_T", bound=StrEnum)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_STEP_RE = re.compile(r"(?im)^\s*(?:\d+[.)]|step\s+\d+|first,|then,|finally,)")


class _InvalidOutput(RuntimeError):
    """The model returned unusable JSON twice."""

    def __init__(self, message: str, usage: ChatUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or ChatUsage()


@dataclass
class TriageSummary:
    processed: int = 0
    extracted: int = 0
    archived: int = 0
    stubbed: int = 0
    unavailable: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0
    decisions: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    plans: list[TriageResult] = field(default_factory=list)

    def record(self, result: TriageResult) -> None:
        self.processed += 1
        self.cost_usd += result.cost_usd
        self.decisions[result.decision.value] = self.decisions.get(result.decision.value, 0) + 1
        self.plans.append(result)
        if result.error is not None:
            self.failed += 1
            self.failures[result.shortcode] = result.error
        elif result.method is TriageMethod.UNAVAILABLE:
            self.unavailable += 1
        elif result.decision is Decision.ARCHIVE:
            self.archived += 1
        elif result.decision is Decision.STUB:
            self.stubbed += 1
        else:
            self.extracted += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "extracted": self.extracted,
            "archived": self.archived,
            "stubbed": self.stubbed,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "skipped": self.skipped,
            "cost_usd": round(self.cost_usd, 6),
            "decisions": dict(sorted(self.decisions.items())),
            "failures": dict(sorted(self.failures.items())),
        }


def _dedupe(values: Iterable[_T]) -> list[_T]:
    seen: set[_T] = set()
    result: list[_T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _coerce_enum(value: Any, enum_type: type[_T], aliases: dict[str, str]) -> _T | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = aliases.get(key, key)
    try:
        return enum_type(key)
    except ValueError:
        return None


def _enum_list(values: Any, enum_type: type[_T], aliases: dict[str, str]) -> list[_T]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    result: list[_T] = []
    for value in values:
        coerced = _coerce_enum(value, enum_type, aliases)
        if coerced is not None:
            result.append(coerced)
    return _dedupe(result)


def _categories(values: Any) -> list[CategoryHint]:
    if not isinstance(values, list):
        return []
    merged: dict[str, CategoryHint] = {}
    for value in values:
        name: Any
        confidence: float
        if isinstance(value, str):
            name, confidence = value, 0.5
        elif isinstance(value, dict):
            name = value.get("name")
            try:
                confidence = float(value.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
        else:
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned = " ".join(name.split())[:120]
        hint = CategoryHint(name=cleaned, confidence=max(0.0, min(1.0, confidence)))
        key = cleaned.casefold()
        current = merged.get(key)
        if current is None or hint.confidence > current.confidence:
            merged[key] = hint
    return list(merged.values())[:5]


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a model response, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the response")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("the response must be a JSON object")
    return payload


def normalise_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce the usual model quirks into the schema before validation."""
    data = {str(key).strip().lower(): value for key, value in payload.items()}

    decision = data.get("decision")
    if isinstance(decision, str):
        key = decision.strip().lower().replace("-", "_")
        data["decision"] = _DECISION_ALIASES.get(key, key)

    content_type = data.get("content_type")
    if isinstance(content_type, str):
        key = content_type.strip().lower()
        data["content_type"] = key if key in {item.value for item in ContentType} else "other"
    else:
        data["content_type"] = "other"

    try:
        score = int(float(data.get("value_score", 0)))
    except (TypeError, ValueError):
        score = 0
    data["value_score"] = max(0, min(100, score))

    reason = data.get("reason")
    data["reason"] = (
        " ".join(str(reason).split())[:2000] if reason is not None else "No reason provided."
    ) or "No reason provided."

    data["actions"] = _enum_list(data.get("actions"), TriageAction, _ACTION_ALIASES)
    data["fields"] = _enum_list(data.get("fields"), ExtractionField, _FIELD_ALIASES)
    data["categories"] = _categories(data.get("categories"))

    archive_reason = data.get("archive_reason")
    data["archive_reason"] = (
        " ".join(str(archive_reason).split())[:1000]
        if isinstance(archive_reason, str) and archive_reason.strip()
        else None
    )
    return data


def _strip_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "_row"}


class TriageService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        client: ChatClient | None = None,
        rules: ArchiveRules | None = None,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.model = model or config.groq.triage_model
        self.rules = rules if rules is not None else load_rules(config.archive_rules_file)
        self.client = client or GroqChatClient(config.groq, sleep=sleep)

    # ------------------------------------------------------------------ public API

    def triage_pending(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
        dry_run: bool = False,
    ) -> TriageSummary:
        summary = TriageSummary()
        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            reels = [reel for reel in reels if str(reel["shortcode"]) in wanted]
        elif not force:
            reels = [reel for reel in reels if str(reel["status"]) in TRIAGE_STATUSES]
        if limit is not None:
            reels = reels[: max(0, limit)]

        for reel in reels:
            shortcode = str(reel["shortcode"])
            result = self.triage_one(shortcode, dry_run=dry_run)
            summary.record(result)
            logger.info(
                "triage %s: %s score=%d method=%s",
                shortcode,
                result.decision.value,
                result.value_score,
                result.method.value,
            )
        return summary

    def triage_one(self, shortcode: str, *, dry_run: bool = False) -> TriageResult:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise LookupError(f"unknown shortcode: {shortcode}")

        metadata = self.store.get_metadata(shortcode)
        if metadata is None:
            result = self._error_result(
                shortcode, "metadata is missing; run `reel-vault fetch` first"
            )
            if not dry_run:
                self._persist(shortcode, result)
            return result

        fetch_status = str(metadata.get("fetch_status") or "")
        if (
            fetch_status in UNAVAILABLE_FETCH_STATUSES
            or str(reel["status"]) == ReelStatus.UNAVAILABLE.value
        ):
            result = self._unavailable_result(shortcode)
            if not dry_run:
                self._persist(shortcode, result)
            return result
        if fetch_status != "ok":
            result = self._error_result(
                shortcode, f"metadata fetch status is {fetch_status or 'unknown'}"
            )
            if not dry_run:
                self._persist(shortcode, result)
            return result

        payload = _strip_row(metadata)
        try:
            enrichment = EnrichmentResult.model_validate(payload)
        except ValidationError as exc:
            result = self._error_result(shortcode, f"metadata is invalid: {exc}")
            if not dry_run:
                self._persist(shortcode, result)
            return result

        transcript = self._transcript_text(shortcode)
        caption = enrichment.caption
        text = "\n".join(part for part in (caption, transcript) if part)
        matches = self.rules.detect(text, list(reel["collections"]))

        context = TriageContext(
            shortcode=shortcode,
            caption=caption,
            transcript=transcript,
            hashtags=list(reel["hashtags"]),
            collections=list(reel["collections"]),
            author=enrichment.author,
            duration=enrichment.duration,
            media_type=str(reel["media_type"]) if reel["media_type"] else None,
            rule_hints=matches.hints(),
            max_chars=self.config.triage.max_input_chars,
        )

        manual = self._manual_result(shortcode, self.store.get_review(shortcode), matches)
        if manual is not None:
            if not dry_run:
                self._persist(shortcode, manual)
            return manual

        if dry_run:
            return self._heuristic(shortcode, context, matches)

        self.store.set_job(shortcode, "triage", JobState.RUNNING)
        try:
            output, usage = self._ask(context)
        except ChatError as exc:
            logger.warning("triage %s provider failure: %s", shortcode, exc)
            result = self._error_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result
        except _InvalidOutput as exc:
            logger.warning("triage %s invalid model output: %s", shortcode, exc)
            result = self._fallback_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result

        result = self._decide(shortcode, output, matches, usage, method=TriageMethod.AI)
        self._persist(shortcode, result)
        return result

    # ------------------------------------------------------------------ model call

    def _ask(self, context: TriageContext) -> tuple[TriageOutput, ChatUsage]:
        messages = build_messages(context)
        usage = ChatUsage()
        try:
            first = self.client.complete(messages, model=self.model)
        except ChatError:
            raise
        usage.input_tokens += first.usage.input_tokens
        usage.output_tokens += first.usage.output_tokens
        usage.usd += first.usage.usd

        try:
            return self._parse_output(first.text), usage
        except (ValueError, ValidationError) as exc:
            first_error = str(exc)
            raw = first.text

        try:
            second = self.client.complete(
                repair_messages(messages, raw, first_error), model=self.model
            )
        except ChatError as exc:
            raise ChatError(
                str(exc),
                retryable=exc.retryable,
                status_code=exc.status_code,
                usage=usage,
            ) from exc
        usage.input_tokens += second.usage.input_tokens
        usage.output_tokens += second.usage.output_tokens
        usage.usd += second.usage.usd

        try:
            return self._parse_output(second.text), usage
        except (ValueError, ValidationError) as exc:
            raise _InvalidOutput(
                f"the model did not return valid JSON after one repair: {exc}", usage
            ) from exc

    @staticmethod
    def _parse_output(text: str) -> TriageOutput:
        payload = extract_json_object(text)
        return TriageOutput.model_validate(normalise_output(payload))

    # ------------------------------------------------------------------ decisions

    def _decide(
        self,
        shortcode: str,
        output: TriageOutput,
        matches: RuleMatches,
        usage: ChatUsage,
        *,
        method: TriageMethod,
    ) -> TriageResult:
        decision = output.decision
        score = output.value_score
        actions = list(output.actions)
        fields = list(output.fields)
        reason = output.reason
        archive_reason = output.archive_reason
        rule_reason: str | None = None
        needs_review = False

        # The emotional/personal rule outranks the tutorial/news rule when both fire:
        # a post that is genuinely personal is kept whole, not mined for steps.
        archive_by_rule = matches.archive_triggered and not self.rules.confirm_with_llm
        archive_confirmed = decision is Decision.ARCHIVE and matches.archive_triggered

        if archive_by_rule or archive_confirmed:
            decision = Decision.ARCHIVE
            actions = _dedupe([*actions, *ARCHIVE_ACTIONS])
            fields = []
            if archive_confirmed:
                rule_reason = self._rule_reason(matches, "archive confirmed")
            else:
                rule_reason = self._rule_reason(matches, "archive forced by rules")
            archive_reason = archive_reason or rule_reason
            method = TriageMethod.RULE
        elif matches.forced_extract:
            decision = Decision.EXTRACT
            score = max(score, self.config.triage.extract_threshold)
            forced_fields: list[ExtractionField] = []
            if matches.tutorial_triggered:
                forced_fields.extend(TUTORIAL_FIELDS)
            if matches.news_triggered:
                forced_fields.extend(NEWS_FIELDS)
            fields = _dedupe([*fields, *forced_fields])
            actions = _dedupe(
                [
                    *actions,
                    TriageAction.TRANSCRIPT,
                    TriageAction.LINKS,
                    TriageAction.KEYFRAMES,
                ]
            )
            rule_reason = self._rule_reason(matches, "forced extract")
            reason = f"{reason} [{rule_reason}]"
            needs_review = output.value_score < self.config.triage.needs_review_threshold
            method = TriageMethod.RULE
        elif decision is Decision.ARCHIVE:
            actions = _dedupe([*actions, *ARCHIVE_ACTIONS])
            fields = []
            archive_reason = archive_reason or "the model judged this post worth keeping whole"
        else:
            if matches.archive_triggered:
                rule_reason = self._rule_reason(matches, "archive trigger not confirmed by model")
            decision = (
                Decision.EXTRACT if score >= self.config.triage.extract_threshold else Decision.STUB
            )
            if decision is Decision.EXTRACT:
                actions = _dedupe([*actions, TriageAction.TRANSCRIPT, TriageAction.LINKS])
                needs_review = score < self.config.triage.needs_review_threshold
            else:
                actions = []
                fields = []

        if decision is Decision.ARCHIVE:
            needs_review = False
        if decision is Decision.STUB:
            actions = []
            fields = []

        return TriageResult(
            shortcode=shortcode,
            decision=decision,
            value_score=score,
            content_type=output.content_type,
            reason=reason,
            actions=actions,
            fields=fields,
            categories=output.categories,
            archive_reason=archive_reason,
            method=method,
            model=self.model,
            needs_review=needs_review,
            rule_reason=rule_reason,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    def _manual_result(
        self, shortcode: str, review: dict[str, Any] | None, matches: RuleMatches
    ) -> TriageResult | None:
        if not review:
            return None
        raw_decision = review.get("decision") or review.get("force") or review.get("force_decision")
        if not isinstance(raw_decision, str):
            return None
        decision = _coerce_enum(raw_decision, Decision, _DECISION_ALIASES)
        if decision is None:
            return None

        try:
            score = int(float(review.get("value_score", 50)))
        except (TypeError, ValueError):
            score = 50
        score = max(0, min(100, score))
        content_type = (
            _coerce_enum(review.get("content_type", ContentType.OTHER.value), ContentType, {})
            or ContentType.OTHER
        )
        fields = _enum_list(review.get("fields", []), ExtractionField, _FIELD_ALIASES)
        actions = _enum_list(review.get("actions", []), TriageAction, _ACTION_ALIASES)
        categories = _categories(review.get("categories", []))
        notes = review.get("notes")
        note_text = " ".join(str(notes).split())[:1000] if notes else ""
        reason = f"Manual override: {note_text}" if note_text else "Manual override."

        if decision is Decision.ARCHIVE:
            actions = _dedupe([*actions, *ARCHIVE_ACTIONS])
            fields = []
            archive_reason = note_text or "manually kept whole"
        elif decision is Decision.EXTRACT:
            actions = _dedupe(
                [*actions, TriageAction.TRANSCRIPT, TriageAction.LINKS, TriageAction.KEYFRAMES]
            )
            fields = fields or [ExtractionField.LINKS]
            archive_reason = None
        else:
            actions = []
            fields = []
            archive_reason = None

        return TriageResult(
            shortcode=shortcode,
            decision=decision,
            value_score=score,
            content_type=content_type,
            reason=reason,
            actions=actions,
            fields=fields,
            categories=categories,
            archive_reason=archive_reason,
            method=TriageMethod.MANUAL,
            model=None,
            needs_review=False,
            rule_reason=self._rule_reason(matches, "ignored because of a manual override")
            if matches.triggered
            else None,
            prompt_version=PROMPT_VERSION,
        )

    def _heuristic(
        self, shortcode: str, context: TriageContext, matches: RuleMatches
    ) -> TriageResult:
        text = "\n".join(part for part in (context.caption, context.transcript) if part)
        score = 10
        score += min(20, len(text) // 120)
        score += min(24, 8 * len(_URL_RE.findall(text)))
        score += min(12, 4 * len(context.hashtags))
        if matches.tutorial_triggered:
            score += 25
        if matches.news_triggered:
            score += 20
        if _STEP_RE.search(text):
            score += 10
        if any(char.isdigit() for char in text):
            score += 5
        score = max(0, min(100, score))

        if matches.archive_triggered:
            decision = Decision.ARCHIVE
            score = max(score, 30)
        elif matches.forced_extract:
            decision = Decision.EXTRACT
        else:
            decision = (
                Decision.EXTRACT if score >= self.config.triage.extract_threshold else Decision.STUB
            )

        if matches.tutorial_triggered:
            content_type = ContentType.TUTORIAL
        elif matches.news_triggered:
            content_type = ContentType.NEWS
        elif "recipe" in text.casefold():
            content_type = ContentType.RECIPE
        else:
            content_type = ContentType.OTHER

        if decision is Decision.ARCHIVE:
            actions = list(ARCHIVE_ACTIONS)
            fields: list[ExtractionField] = []
            archive_reason = self._rule_reason(matches, "archive trigger")
        elif decision is Decision.EXTRACT:
            actions = [TriageAction.TRANSCRIPT, TriageAction.LINKS, TriageAction.KEYFRAMES]
            fields = []
            if matches.tutorial_triggered:
                fields.extend(TUTORIAL_FIELDS)
            if matches.news_triggered:
                fields.extend(NEWS_FIELDS)
            if not fields:
                fields = [ExtractionField.LINKS]
            archive_reason = None
        else:
            actions = []
            fields = []
            archive_reason = None

        return TriageResult(
            shortcode=shortcode,
            decision=decision,
            value_score=score,
            content_type=content_type,
            reason="Dry-run heuristic; no model was called.",
            actions=actions,
            fields=_dedupe(fields),
            categories=[],
            archive_reason=archive_reason,
            method=TriageMethod.HEURISTIC,
            model=None,
            needs_review=False,
            rule_reason=self._rule_reason(matches, "dry-run rule match")
            if matches.triggered
            else None,
            prompt_version=PROMPT_VERSION,
        )

    # ------------------------------------------------------------------ results

    def _error_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> TriageResult:
        text = " ".join(message.split())[:2000] or "triage failed"
        usage = usage or ChatUsage()
        return TriageResult(
            shortcode=shortcode,
            decision=Decision.STUB,
            value_score=0,
            content_type=ContentType.OTHER,
            reason=f"Triage failed: {text}",
            method=TriageMethod.FALLBACK,
            model=self.model,
            needs_review=True,
            error=text,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    def _fallback_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> TriageResult:
        text = " ".join(message.split())[:2000] or "invalid model output"
        usage = usage or ChatUsage()
        return TriageResult(
            shortcode=shortcode,
            decision=Decision.EXTRACT,
            value_score=self.config.triage.extract_threshold,
            content_type=ContentType.OTHER,
            reason=f"Triage fallback: {text}. A standard plan was applied; please review.",
            actions=list(FALLBACK_ACTIONS),
            fields=[ExtractionField.LINKS],
            method=TriageMethod.FALLBACK,
            model=self.model,
            needs_review=True,
            error=None,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    def _unavailable_result(self, shortcode: str) -> TriageResult:
        return TriageResult(
            shortcode=shortcode,
            decision=Decision.STUB,
            value_score=0,
            content_type=ContentType.OTHER,
            reason="Post is unavailable (private, deleted or blocked).",
            method=TriageMethod.UNAVAILABLE,
            model=None,
            needs_review=False,
            prompt_version=PROMPT_VERSION,
        )

    @staticmethod
    def _rule_reason(matches: RuleMatches, action: str) -> str:
        parts: list[str] = []
        if matches.archive_keywords:
            parts.append("keywords: " + ", ".join(matches.archive_keywords))
        if matches.archive_categories:
            parts.append("collections: " + ", ".join(matches.archive_categories))
        if matches.tutorial_keywords:
            parts.append("tutorial keywords: " + ", ".join(matches.tutorial_keywords))
        if matches.news_keywords:
            parts.append("news keywords: " + ", ".join(matches.news_keywords))
        detail = "; ".join(parts) if parts else "no rule detail"
        return f"{action} ({detail})"

    # ------------------------------------------------------------------ persistence

    def _transcript_text(self, shortcode: str) -> str | None:
        record = self.store.get_transcript(shortcode)
        if not record or record.get("error"):
            return None
        text = record.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _persist(self, shortcode: str, result: TriageResult) -> None:
        payload = result.model_dump(mode="json")
        self.store.save_triage(shortcode, payload)
        write_model_atomic(self.paths.reel(shortcode).triage, result)

        if result.error is not None:
            self.store.set_status(shortcode, ReelStatus.TRIAGE_FAILED)
            self.store.set_job(shortcode, "triage", JobState.FAILED, error=result.error)
        elif result.method is TriageMethod.UNAVAILABLE:
            self.store.set_status(shortcode, ReelStatus.UNAVAILABLE)
            self.store.set_job(shortcode, "triage", JobState.DONE)
        elif result.decision is Decision.STUB:
            self.store.set_status(shortcode, ReelStatus.STUB)
            self.store.set_job(shortcode, "triage", JobState.DONE)
        else:
            self.store.set_status(shortcode, ReelStatus.TRIAGED)
            self.store.set_job(shortcode, "triage", JobState.DONE)

        if result.cost_usd > 0:
            self.store.record_cost(
                shortcode,
                "groq",
                result.model or self.model,
                units=float(result.total_tokens),
                usd=result.cost_usd,
            )


__all__ = [
    "ARCHIVE_ACTIONS",
    "FALLBACK_ACTIONS",
    "TRIAGE_STATUSES",
    "TriageService",
    "TriageSummary",
    "extract_json_object",
    "normalise_output",
]
