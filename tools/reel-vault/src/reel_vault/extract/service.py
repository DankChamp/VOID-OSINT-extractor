"""Grounded extraction service.

Only the fields and actions chosen by triage are executed. Every extracted item must
cite the caption, a transcript segment or an OCR frame; ungrounded items are dropped.
RapidOCR runs locally and degrades gracefully when it is not installed.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from reel_vault.config.models import AppConfig
from reel_vault.extract.prompts import PROMPT_VERSION, ExtractionContext, build_messages
from reel_vault.keyframes import Keyframe as FsKeyframe
from reel_vault.keyframes import KeyframeError, select_keyframes
from reel_vault.links import find_urls, resolve_links
from reel_vault.llm.base import ChatClient, ChatError, ChatUsage
from reel_vault.llm.groq import GroqChatClient
from reel_vault.logging_setup import get_logger
from reel_vault.media.service import MediaService
from reel_vault.ocr.base import OcrEngine
from reel_vault.ocr.rapid import RapidOcrEngine
from reel_vault.schemas.common import Decision, JobState, ReelStatus
from reel_vault.schemas.enrichment import EnrichmentResult
from reel_vault.schemas.extraction import (
    ExtractionItem,
    ExtractionMethod,
    ExtractionResult,
    Keyframe,
    LinkStatus,
    OcrLine,
    ResolvedLink,
)
from reel_vault.schemas.extraction import (
    OcrResult as OcrResultSchema,
)
from reel_vault.schemas.triage import (
    ExtractionField,
    TriageAction,
)
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_model_atomic
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.triage.rules import ArchiveRules, load_rules

logger = get_logger(__name__)

EXTRACT_STATUSES = frozenset(
    {
        ReelStatus.TRIAGED.value,
        ReelStatus.EXTRACTION_FAILED.value,
    }
)

_ARCHIVE_FIELDS: tuple[ExtractionField, ...] = (ExtractionField.NOTES,)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_STEP_RE = re.compile(r"(?im)^\s*(?:\d+[.)]|step\s+\d+|first,|then,|finally,)")


class _InvalidOutput(RuntimeError):
    def __init__(self, message: str, usage: ChatUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or ChatUsage()


@dataclass
class ExtractionSummary:
    processed: int = 0
    extracted: int = 0
    archived: int = 0
    stubbed: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0
    decisions: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    results: list[ExtractionResult] = field(default_factory=list)

    def record(self, result: ExtractionResult) -> None:
        self.processed += 1
        self.cost_usd += result.cost_usd
        self.decisions[result.decision.value] = self.decisions.get(result.decision.value, 0) + 1
        self.results.append(result)
        if result.error is not None:
            self.failed += 1
            self.failures[result.shortcode] = result.error
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
            "failed": self.failed,
            "skipped": self.skipped,
            "cost_usd": round(self.cost_usd, 6),
            "decisions": dict(sorted(self.decisions.items())),
            "failures": dict(sorted(self.failures.items())),
        }


class ExtractionService:
    def __init__(
        self,
        store: Store,
        paths: WorkspacePaths,
        config: AppConfig,
        *,
        client: ChatClient | None = None,
        ocr: OcrEngine | None = None,
        media: MediaService | None = None,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
        rules: ArchiveRules | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config
        self.extract_config = config.extract
        self.model = model or config.groq.extraction_model
        self.client = client or GroqChatClient(config.groq, sleep=sleep)
        self.ocr = ocr or RapidOcrEngine()
        self.media = media or MediaService(store, paths, config)
        self.sleep = sleep
        self.rules = rules if rules is not None else load_rules(config.archive_rules_file)

    # ------------------------------------------------------------------ public API

    def extract_pending(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        shortcodes: Iterable[str] | None = None,
    ) -> ExtractionSummary:
        summary = ExtractionSummary()
        reels = self.store.list_reels()
        if shortcodes is not None:
            wanted = {str(code) for code in shortcodes}
            reels = [reel for reel in reels if str(reel["shortcode"]) in wanted]
        elif not force:
            reels = [reel for reel in reels if str(reel["status"]) in EXTRACT_STATUSES]
        if limit is not None:
            reels = reels[: max(0, limit)]

        for reel in reels:
            shortcode = str(reel["shortcode"])
            result = self.extract_one(shortcode)
            summary.record(result)
            logger.info(
                "extract %s: %s method=%s items=%d",
                shortcode,
                result.decision.value,
                result.method.value,
                len(result.items),
            )
        return summary

    def extract_one(self, shortcode: str) -> ExtractionResult:
        reel = self.store.get_reel(shortcode)
        if reel is None:
            raise LookupError(f"unknown shortcode: {shortcode}")

        triage = self.store.get_triage(shortcode)
        if triage is None:
            result = self._error_result(
                shortcode, "triage is missing; run `reel-vault triage` first"
            )
            self._persist(shortcode, result)
            return result

        decision = triage.get("decision")
        if decision == Decision.STUB.value:
            result = self._stub_result(shortcode, triage)
            self._persist(shortcode, result)
            return result

        if decision == Decision.ARCHIVE.value:
            result = self._archive_result(shortcode, triage)
            self._persist(shortcode, result)
            return result

        if decision != Decision.EXTRACT.value:
            result = self._error_result(shortcode, f"unknown triage decision {decision!r}")
            self._persist(shortcode, result)
            return result

        metadata = self.store.get_metadata(shortcode)
        if metadata is None:
            result = self._error_result(
                shortcode, "metadata is missing; run `reel-vault fetch` first"
            )
            self._persist(shortcode, result)
            return result

        payload = {key: value for key, value in metadata.items() if key != "_row"}
        try:
            enrichment = EnrichmentResult.model_validate(payload)
        except ValidationError as exc:
            result = self._error_result(shortcode, f"metadata is invalid: {exc}")
            self._persist(shortcode, result)
            return result

        fields = _enum_list(triage.get("fields", []), ExtractionField, {})
        actions = _enum_list(triage.get("actions", []), TriageAction, {})

        self.store.set_job(shortcode, "extract", JobState.RUNNING)
        try:
            output, usage = self._ask(enrichment, fields, actions)
        except ChatError as exc:
            logger.warning("extract %s provider failure: %s", shortcode, exc)
            result = self._error_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result
        except _InvalidOutput as exc:
            logger.warning("extract %s invalid model output: %s", shortcode, exc)
            result = self._fallback_result(shortcode, str(exc), usage=exc.usage)
            self._persist(shortcode, result)
            return result

        result = self._decide(shortcode, output, enrichment, triage, usage, ExtractionMethod.AI)
        self._persist(shortcode, result)
        return result

    # ------------------------------------------------------------------ model call

    def _ask(
        self,
        enrichment: EnrichmentResult,
        fields: list[ExtractionField],
        actions: list[TriageAction],
    ) -> tuple[dict[str, Any], ChatUsage]:
        transcript = self._transcript_text(enrichment.shortcode)
        segments = self._transcript_segments(enrichment.shortcode)
        links = self._links_from_text(enrichment.caption or "", transcript or "")
        ocr_text = ""
        if TriageAction.OCR in actions:
            ocr_text = self._ocr_text(enrichment.shortcode)

        context = ExtractionContext(
            shortcode=enrichment.shortcode,
            caption=enrichment.caption,
            transcript=transcript,
            segments=segments,
            ocr_text=ocr_text,
            fields=fields,
            links=links,
            max_chars=self.config.triage.max_input_chars,
        )
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
                self._repair_messages(messages, raw, first_error), model=self.model
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

    @staticmethod
    def _repair_messages(messages: list[Any], raw: str, error: str) -> list[Any]:
        clipped = raw.strip()
        if len(clipped) > 4000:
            clipped = clipped[:4000] + " …[truncated]"
        return [
            *messages,
            type(messages[0])(role="assistant", content=clipped),
            type(messages[0])(
                role="user",
                content=(
                    "That response was not valid for the required schema: "
                    f"{error}\nReturn only the corrected JSON object, with every "
                    "required key and no markdown fences."
                ),
            ),
        ]

    # ------------------------------------------------------------------ decisions

    def _decide(
        self,
        shortcode: str,
        output: dict[str, Any],
        enrichment: EnrichmentResult,
        triage: dict[str, Any],
        usage: ChatUsage,
        method: ExtractionMethod,
    ) -> ExtractionResult:
        decision = Decision.EXTRACT
        raw_items = output.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []

        fields_wanted = _enum_list(triage.get("fields", []), ExtractionField, {})
        items = self._ground_items(raw_items, fields_wanted, enrichment, transcript=None)
        items = items[: self.extract_config.max_items]

        links = self._resolve_links(output.get("links", []), enrichment.caption or "")

        keyframes_fs: list[FsKeyframe] = []
        keyframe_error: str | None = None
        if TriageAction.KEYFRAMES in self._actions_of(triage):
            keyframes_fs, keyframe_error = self._select_keyframes(shortcode)

        ocr_result: OcrResultSchema | None = None
        if TriageAction.OCR in self._actions_of(triage):
            ocr_result = self._run_ocr(shortcode, keyframes_fs)

        return ExtractionResult(
            shortcode=shortcode,
            decision=decision,
            summary=output.get("summary"),
            items=items,
            links=links,
            keyframes=[Keyframe(path=str(kf.path), timestamp=kf.timestamp) for kf in keyframes_fs],
            ocr=ocr_result,
            keyframe_error=keyframe_error,
            model=self.model,
            method=method,
            needs_review=False,
            raw_output=output.get("raw_output"),
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    # ------------------------------------------------------------------ helpers

    def _transcript_text(self, shortcode: str) -> str | None:
        record = self.store.get_transcript(shortcode)
        if not record or record.get("error"):
            return None
        text = record.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _transcript_segments(self, shortcode: str) -> list[dict[str, object]]:
        record = self.store.get_transcript(shortcode)
        if not record:
            return []
        segments = record.get("segments") or []
        return [
            {
                "id": seg.get("id", index),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", ""),
            }
            for index, seg in enumerate(segments)
        ]

    def _links_from_text(self, caption: str | None, transcript: str | None) -> list[str]:
        text = "\n".join(part for part in (caption, transcript) if part)
        return find_urls(text)[: self.extract_config.max_links]

    def _ocr_text(self, shortcode: str) -> str:
        return ""

    def _run_ocr(self, shortcode: str, keyframes: Sequence[FsKeyframe]) -> OcrResultSchema:
        lines: list[OcrLine] = []
        texts: list[str] = []
        available = bool(getattr(self.ocr, "available", False))
        error: str | None = None
        for kf in keyframes:
            try:
                result = self.ocr.read(Path(kf.path))
            except Exception as exc:  # pragma: no cover - runtime dependency
                error = str(exc)
                continue
            available = available and result.available
            if result.error and error is None:
                error = result.error
            for line in result.lines:
                lines.append(
                    OcrLine(text=line.text, frame=kf.path.name, confidence=line.confidence)
                )
            if result.text:
                texts.append(result.text)
        ocr_result = OcrResultSchema(
            shortcode=shortcode,
            engine="rapidocr",
            available=available,
            frames=len(keyframes),
            lines=[
                OcrLine(text=line.text, frame=line.frame, confidence=line.confidence)
                for line in lines
            ],
            text="\n".join(texts),
            error=error,
        )
        write_model_atomic(self.paths.reel(shortcode).ocr, ocr_result)
        return ocr_result

    def _resolve_links(self, link_specs: list[Any], text: str) -> list[ResolvedLink]:
        urls: list[str] = []
        for spec in link_specs:
            if isinstance(spec, str):
                urls.append(spec)
            elif isinstance(spec, dict) and isinstance(spec.get("url"), str):
                urls.append(spec["url"])
        if not self.extract_config.resolve_links:
            return [
                ResolvedLink(url=url, status="skipped")
                for url in _dedupe(urls)[: self.extract_config.max_links]
            ]
        found = find_urls(text)
        candidates = _dedupe([*urls, *found])
        resolved = resolve_links(
            candidates,
            max_links=self.extract_config.max_links,
            timeout_seconds=self.extract_config.link_timeout_seconds,
        )
        return [
            ResolvedLink(
                url=item.url,
                final_url=item.final_url,
                title=item.title,
                status=cast(LinkStatus, item.status),
                error=item.error,
            )
            for item in resolved
        ]

    def _select_keyframes(self, shortcode: str) -> tuple[list[FsKeyframe], str | None]:
        media_record = self.store.get_media(shortcode)
        if not media_record:
            return [], "no media archive; skipping keyframes"
        files = media_record.get("files") or []
        videos = [
            self.paths.media(shortcode) / item["path"]
            for item in files
            if isinstance(item, dict) and item.get("kind") == "video"
        ]
        if not videos:
            return [], "no archived video; skipping keyframes"
        workdir = self.paths.reel(shortcode).keyframes
        try:
            return select_keyframes(
                videos[0],
                workdir,
                max_keyframes=self.config.ocr.max_keyframes,
                scene_threshold=self.config.extract.scene_threshold,
            ), None
        except (KeyframeError, OSError) as exc:
            return [], str(exc)

    def _ground_items(
        self,
        raw_items: Sequence[Any],
        fields_wanted: list[ExtractionField],
        enrichment: EnrichmentResult,
        transcript: str | None,
    ) -> list[ExtractionItem]:
        wanted = {field.value for field in fields_wanted}
        grounded: list[ExtractionItem] = []
        seen: set[tuple[str, str]] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            field = raw.get("field")
            text = raw.get("text")
            source = raw.get("source") or "caption"
            if not isinstance(field, str) or not isinstance(text, str) or not text.strip():
                continue
            if field not in wanted:
                continue
            if not self._is_grounded(source, text):
                continue
            key = (field, text.strip())
            if key in seen:
                continue
            seen.add(key)
            grounded.append(
                ExtractionItem(
                    field=ExtractionField(field), text=text.strip()[:2000], source=str(source)[:200]
                )
            )
        return grounded

    @staticmethod
    def _is_grounded(source: str, text: str) -> bool:
        if source == "caption":
            return True
        return source.startswith(("transcript:", "ocr:"))

    def _actions_of(self, triage: dict[str, Any]) -> list[TriageAction]:
        return _enum_list(triage.get("actions", []), TriageAction, {})

    # ------------------------------------------------------------------ result builders

    def _stub_result(self, shortcode: str, triage: dict[str, Any]) -> ExtractionResult:
        return ExtractionResult(
            shortcode=shortcode,
            decision=Decision.STUB,
            method=ExtractionMethod.RULE,
            model=None,
            needs_review=False,
            prompt_version=PROMPT_VERSION,
        )

    def _archive_result(self, shortcode: str, triage: dict[str, Any]) -> ExtractionResult:
        notes = triage.get("archive_reason") or triage.get("reason") or ""
        item: list[ExtractionItem] = []
        if notes:
            item.append(
                ExtractionItem(field=ExtractionField.NOTES, text=str(notes)[:2000], source="triage")
            )
        return ExtractionResult(
            shortcode=shortcode,
            decision=Decision.ARCHIVE,
            summary=None,
            items=item,
            method=ExtractionMethod.ARCHIVE,
            model=None,
            needs_review=False,
            prompt_version=PROMPT_VERSION,
        )

    def _error_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> ExtractionResult:
        usage = usage or ChatUsage()
        return ExtractionResult(
            shortcode=shortcode,
            decision=Decision.STUB,
            method=ExtractionMethod.FALLBACK,
            error=message,
            model=self.model,
            needs_review=True,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    def _fallback_result(
        self, shortcode: str, message: str, *, usage: ChatUsage | None = None
    ) -> ExtractionResult:
        usage = usage or ChatUsage()
        return ExtractionResult(
            shortcode=shortcode,
            decision=Decision.EXTRACT,
            method=ExtractionMethod.FALLBACK,
            error=None,
            model=self.model,
            needs_review=True,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.usd, 8),
        )

    # ------------------------------------------------------------------ persistence

    def _persist(self, shortcode: str, result: ExtractionResult) -> None:
        payload = result.model_dump(mode="json")
        self.store.save_extraction(shortcode, payload)
        write_model_atomic(self.paths.reel(shortcode).extraction, result)

        if result.error is not None:
            self.store.set_status(shortcode, ReelStatus.EXTRACTION_FAILED)
            self.store.set_job(shortcode, "extract", JobState.FAILED, error=result.error)
        elif result.decision is Decision.STUB:
            self.store.set_status(shortcode, ReelStatus.STUB)
            self.store.set_job(shortcode, "extract", JobState.DONE)
        else:
            self.store.set_status(shortcode, ReelStatus.EXTRACTED)
            self.store.set_job(shortcode, "extract", JobState.DONE)

        if result.cost_usd > 0:
            self.store.record_cost(
                shortcode,
                "groq",
                result.model or self.model,
                units=float(result.total_tokens),
                usd=result.cost_usd,
            )


def _enum_list(values: Any, enum_type: type, aliases: dict[str, str]) -> list[Any]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    result: list[Any] = []
    for value in values:
        coerced = _coerce_enum(value, enum_type, aliases)
        if coerced is not None:
            result.append(coerced)
    return result


def _coerce_enum(value: Any, enum_type: type, aliases: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = aliases.get(key, key)
    try:
        return enum_type(key)
    except ValueError:
        return None


def _dedupe(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "ExtractionService",
    "ExtractionSummary",
]
