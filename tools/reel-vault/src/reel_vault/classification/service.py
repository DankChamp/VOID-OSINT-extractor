"""Classification orchestration.

Decision order:

1. manual labels in ``review.json`` always override -> ``manual``
2. taxonomy rules prefilter obvious cases -> ``rule``
3. otherwise the LLM picks labels from the taxonomy -> ``ai``
4. invalid model JSON gets one repair; then a deterministic fallback -> ``fallback``

A reel is flagged ``needs_review`` when its primary confidence is below
``classify.needs_review_threshold`` (default 0.60) or when no primary label could be
chosen.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from pydantic import ValidationError

from reel_vault.classification.prompts import (
    PROMPT_VERSION,
    ClassificationContext,
    build_messages,
    repair_messages,
)
from reel_vault.classification.rules import CategoryMatch, prefilter
from reel_vault.classification.taxonomy import Taxonomy, load_taxonomy
from reel_vault.config.models import AppConfig
from reel_vault.llm.base import ChatClient, ChatError, ChatUsage
from reel_vault.llm.groq import GroqChatClient
from reel_vault.logging_setup import get_logger
from reel_vault.schemas.classification import (
    CategoryLabel,
    ClassificationMethod,
    ClassificationResult,
)
from reel_vault.schemas.common import JobState, ReelStatus
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_model_atomic
from reel_vault.storage.paths import WorkspacePaths

logger = get_logger(__name__)

CLASSIFY_STATUSES = frozenset(
    {
        ReelStatus.TRIAGED.value,
        ReelStatus.EXTRACTED.value,
        ReelStatus.STUB.value,
        ReelStatus.CLASSIFICATION_FAILED.value,
    }
)

_T = TypeVar("_T")


class _InvalidOutput(RuntimeError):
    def __init__(self, message: str, usage: ChatUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or ChatUsage()


@dataclass
class ClassificationSummary:
    processed: int = 0
    classified: int = 0
    needs_review: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0
    methods: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    results: list[ClassificationResult] = field(default_factory=list)

    def record(self, result: ClassificationResult) -> None:
        self.processed += 1
        self.cost_usd += result.cost_usd
        self.methods[result.method.value] = self.methods.get(result.method.value, 0) + 1
        self.results.append(result)
        if result.error is not None:
            self.failed += 1
            self.failures[result.shortcode] = result.error
        else:
            self.classified += 1
            if result.needs_review:
                self.needs_review += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "classified": self.classified,
            "needs_review": self.needs_review,
            "failed": self.failed,
            "skipped": self.skipped,
            "cost_usd": round(self.cost_usd, 6),
            "methods": dict(sorted(self.methods.items())),
            "failures": dict(sorted(self.failures.items())),
        }


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
    return result


def _coerce_enum(value: Any, enum_type: type[_T], aliases: dict[str, str]) -> _T | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = aliases.get(key, key)
    try:
        constructor: Callable[[str], _T] = cast(Callable[[str], _T], enum_type)
        return constructor(key)
    except ValueError:
        return None


def _dedupe(values: Sequence[_T]) -> list[_T]:
    seen: set[_T] = set()
    result: list[_T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _label(name: str, confidence: float) -> CategoryLabel:
    return CategoryLabel(name=name, confidence=max(0.0, min(1.0, confidence)))


class ClassificationService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        client: ChatClient | None = None,
        taxonomy: Taxonomy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.model = model or config.groq.classification_model
        self.client = client or GroqChatClient(config.groq, sleep=sleep)
        self.taxonomy = taxonomy if taxonomy is not None else load_taxonomy(config.taxonomy_file)
        self.sleep = sleep

    # ------------------------------------------------------------------ public API

    def classify_pending(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
    ) -> ClassificationSummary:
        summary = ClassificationSummary()
        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            reels = [reel for reel in reels if str(reel["shortcode"]) in wanted]
        elif not force:
            reels = [reel for reel in reels if str(reel["status"]) in CLASSIFY_STATUSES]
        if limit is not None:
            reels = reels[: max(0, limit)]

        for reel in reels:
            shortcode = str(reel["shortcode"])
            result = self.classify_one(shortcode)
            summary.record(result)
            logger.info(
                "classify %s: %s method=%s needs_review=%s",
                shortcode,
                result.primary.name if result.primary else "none",
                result.method.value,
                result.needs_review,
            )
        return summary

    def classify_one(self, shortcode: str) -> ClassificationResult:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise LookupError(f"unknown shortcode: {shortcode}")

        review = self.store.get_review(shortcode)
        manual = self._manual_result(shortcode, review)
        if manual is not None:
            self._persist(shortcode, manual)
            return manual

        metadata = self.store.get_metadata(shortcode)
        if metadata is None:
            result = self._error_result(
                shortcode, "metadata is missing; run `reel-vault fetch` first"
            )
            self._persist(shortcode, result)
            return result

        payload = {key: value for key, value in metadata.items() if key != "_row"}
        from reel_vault.schemas.enrichment import EnrichmentResult

        try:
            enrichment = EnrichmentResult.model_validate(payload)
        except ValidationError as exc:
            result = self._error_result(shortcode, f"metadata is invalid: {exc}")
            self._persist(shortcode, result)
            return result

        rule_matches = prefilter(
            self.taxonomy,
            caption=enrichment.caption,
            transcript=self._transcript_text(shortcode),
            hashtags=list(reel["hashtags"]),
            collections=list(reel["collections"]),
        )

        rule_result = self._rule_result(shortcode, rule_matches, enrichment)
        if rule_result is not None:
            self._persist(shortcode, rule_result)
            return rule_result

        context = self._build_context(shortcode, enrichment, reel, rule_matches)
        self.store.set_job(shortcode, "classify", JobState.RUNNING)
        try:
            output, usage = self._ask(context)
        except ChatError as exc:
            logger.warning("classify %s provider failure: %s", shortcode, exc)
            result = self._error_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result
        except _InvalidOutput as exc:
            logger.warning("classify %s invalid model output: %s", shortcode, exc)
            result = self._fallback_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result

        result = self._decide(shortcode, output, rule_matches, usage, ClassificationMethod.AI)
        self._persist(shortcode, result)
        return result

    # ------------------------------------------------------------------ model call

    def _ask(self, context: ClassificationContext) -> tuple[dict[str, Any], ChatUsage]:
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
    def _parse_output(text: str) -> dict[str, Any]:
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

    # ------------------------------------------------------------------ decisions

    def _decide(
        self,
        shortcode: str,
        output: dict[str, Any],
        rule_matches: list[CategoryMatch],
        usage: ChatUsage,
        method: ClassificationMethod,
    ) -> ClassificationResult:
        category_names = self.taxonomy.names()
        aliases = {name.casefold(): name for name in category_names}

        primary: CategoryLabel | None = None
        raw_primary = output.get("primary")
        if isinstance(raw_primary, dict):
            name = self._canonical_label(raw_primary.get("name"), aliases)
            confidence = self._confidence(raw_primary.get("confidence"))
            if name is not None:
                primary = _label(name, confidence)

        secondary: list[CategoryLabel] = []
        raw_secondary = output.get("secondary")
        if isinstance(raw_secondary, list):
            for item in raw_secondary:
                if not isinstance(item, dict):
                    continue
                name = self._canonical_label(item.get("name"), aliases)
                if name is None:
                    continue
                confidence = self._confidence(item.get("confidence"))
                label = _label(name, confidence)
                if primary is not None and label.name == primary.name:
                    continue
                if any(existing.name == label.name for existing in secondary):
                    continue
                secondary.append(label)
        secondary = secondary[:4]

        needs_review = primary is None or (
            primary.confidence is not None
            and primary.confidence < self.config.classify.needs_review_threshold
        )

        rule_reason = None
        if rule_matches:
            top = rule_matches[0]
            rule_reason = f"rule prefilter matched {top.category} ({top.reason})"

        return ClassificationResult(
            shortcode=shortcode,
            primary=primary,
            secondary=secondary,
            method=method,
            needs_review=needs_review,
            rule_reason=rule_reason,
            model=self.model,
            error=None,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    # ------------------------------------------------------------------ rule prefilter

    def _rule_result(
        self,
        shortcode: str,
        rule_matches: list[CategoryMatch],
        enrichment: Any,
    ) -> ClassificationResult | None:
        if not rule_matches:
            return None
        top = rule_matches[0]
        threshold = self.config.classify.rule_confidence_threshold
        if top.confidence < threshold:
            return None

        primary = _label(top.category, top.confidence)
        secondary: list[CategoryLabel] = []
        for match in rule_matches[1:]:
            if match.category == primary.name:
                continue
            if match.confidence >= threshold * 0.8:
                secondary.append(_label(match.category, match.confidence))
        secondary = secondary[:4]

        needs_review = primary.confidence < self.config.classify.needs_review_threshold

        return ClassificationResult(
            shortcode=shortcode,
            primary=primary,
            secondary=secondary,
            method=ClassificationMethod.RULE,
            needs_review=needs_review,
            rule_reason=f"rule prefilter: {top.reason}",
            model=None,
            error=None,
            prompt_version=PROMPT_VERSION,
        )

    # ------------------------------------------------------------------ manual override

    def _manual_result(
        self, shortcode: str, review: dict[str, Any] | None
    ) -> ClassificationResult | None:
        if not review:
            return None
        raw_primary = review.get("primary") or review.get("primary_label")
        raw_secondary = review.get("secondary") or review.get("secondary_labels")
        category_names = self.taxonomy.names()
        aliases = {name.casefold(): name for name in category_names}

        primary: CategoryLabel | None = None
        if isinstance(raw_primary, str):
            name = self._canonical_label(raw_primary, aliases)
            if name is not None:
                primary = _label(name, self._confidence(review.get("primary_confidence", 1.0)))

        secondary: list[CategoryLabel] = []
        if isinstance(raw_secondary, list):
            for item in raw_secondary:
                if isinstance(item, str):
                    name = self._canonical_label(item, aliases)
                elif isinstance(item, dict):
                    name = self._canonical_label(item.get("name"), aliases)
                else:
                    continue
                if name is None:
                    continue
                label = _label(
                    name,
                    self._confidence(item.get("confidence") if isinstance(item, dict) else 1.0),
                )
                if primary is not None and label.name == primary.name:
                    continue
                if any(existing.name == label.name for existing in secondary):
                    continue
                secondary.append(label)
        secondary = secondary[:4]

        return ClassificationResult(
            shortcode=shortcode,
            primary=primary,
            secondary=secondary,
            method=ClassificationMethod.MANUAL,
            needs_review=False,
            rule_reason=None,
            model=None,
            error=None,
            prompt_version=PROMPT_VERSION,
        )

    # ------------------------------------------------------------------ helpers

    def _canonical_label(self, value: Any, aliases: dict[str, str]) -> str | None:
        if not isinstance(value, str):
            return None
        key = value.strip().casefold()
        return aliases.get(key)

    def _confidence(self, value: Any) -> float:
        if value is None:
            return 0.5
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    def _transcript_text(self, shortcode: str) -> str | None:
        record = self.store.get_transcript(shortcode)
        if not record or record.get("error"):
            return None
        text = record.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _build_context(
        self,
        shortcode: str,
        enrichment: Any,
        reel: dict[str, Any],
        rule_matches: list[CategoryMatch],
    ) -> ClassificationContext:
        return ClassificationContext(
            shortcode=shortcode,
            caption=enrichment.caption,
            transcript=self._transcript_text(shortcode),
            hashtags=list(reel["hashtags"]),
            collections=list(reel["collections"]),
            categories=list(self.taxonomy.names()),
            category_descriptions={
                category.name: category.description for category in self.taxonomy.categories
            },
            rule_matches=[match.reason for match in rule_matches],
            max_chars=self.config.triage.max_input_chars,
        )

    # ------------------------------------------------------------------ result builders

    def _fallback_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> ClassificationResult:
        usage = usage or ChatUsage()
        category_names = self.taxonomy.names()
        primary_name = (
            "other"
            if "other" in category_names
            else (category_names[0] if category_names else None)
        )
        return ClassificationResult(
            shortcode=shortcode,
            primary=_label(primary_name, 0.5) if primary_name else None,
            secondary=[],
            method=ClassificationMethod.FALLBACK,
            needs_review=True,
            rule_reason=None,
            model=self.model,
            error=None,
            raw_output=None,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    def _error_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> ClassificationResult:
        usage = usage or ChatUsage()
        return ClassificationResult(
            shortcode=shortcode,
            method=ClassificationMethod.FALLBACK,
            needs_review=True,
            error=message,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    # ------------------------------------------------------------------ persistence

    def _persist(self, shortcode: str, result: ClassificationResult) -> None:
        primary_name = result.primary.name if result.primary else None
        secondary_names = [label.name for label in result.secondary]
        confidence = result.primary.confidence if result.primary else 0.0
        self.store.save_classification(
            shortcode,
            primary_label=primary_name,
            secondary=secondary_names,
            confidence=confidence,
            method=result.method.value,
        )
        write_model_atomic(self.paths.reel(shortcode).classification, result)

        if result.error is not None:
            self.store.set_status(shortcode, ReelStatus.CLASSIFICATION_FAILED)
            self.store.set_job(shortcode, "classify", JobState.FAILED, error=result.error)
        else:
            self.store.set_status(shortcode, ReelStatus.CLASSIFIED)
            self.store.set_job(shortcode, "classify", JobState.DONE)

        if result.cost_usd > 0:
            self.store.record_cost(
                shortcode,
                "groq",
                result.model or self.model,
                units=float(result.input_tokens + result.output_tokens),
                usd=result.cost_usd,
            )


__all__ = [
    "ClassificationService",
    "ClassificationSummary",
]
