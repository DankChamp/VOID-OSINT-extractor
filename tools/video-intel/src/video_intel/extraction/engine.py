"""Extraction engine (§8, §9).

For every (pipeline, pass, chunk) triple the engine asks a provider for structured JSON,
validates it against the application contract, repairs it once if needed, and stores the
verbatim response separately from the parsed record.

Design points that matter for reliability:

* Raw responses are always kept. An invalid response is recorded with its failure and
  never enters the dataset.
* A model may only cite segment ids that exist in the chunk. Items citing unknown ids
  are repaired by intersecting with the cited ids the model *did* use; if nothing valid
  remains, the item is dropped and counted.
* Timestamps in provenance are copied from the cited segments, never taken from the model.
* One failing chunk does not stop the run: failures are recorded per record.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video_intel.extraction.prompts import (
    PROMPT_VERSION,
    build_prompt,
    build_repair_prompt,
    build_system_prompt,
)
from video_intel.logging_setup import get_logger
from video_intel.providers.base import (
    Provider,
    ProviderError,
    ProviderRequest,
    extract_json,
)
from video_intel.schemas.chunk import Chunk, ChunkSet
from video_intel.schemas.common import ItemType, PipelineId, StageState
from video_intel.schemas.extraction import (
    ExtractedItem,
    ExtractionResponse,
    ItemProvenance,
    ModelRef,
    RawExtractionRecord,
)
from video_intel.schemas.transcript import TranscriptSegment

logger = get_logger(__name__)

ITEM_ID_TEMPLATE = "item_{pipeline}_p{pass_index}_{chunk}_{ordinal:02d}"
TYPE_PREFIXES: dict[ItemType, str] = {
    ItemType.CONCEPT: "concept",
    ItemType.RELATIONSHIP: "relation",
    ItemType.DEFINITION: "definition",
    ItemType.FACT: "fact",
    ItemType.CLAIM: "claim",
    ItemType.EXAMPLE: "example",
    ItemType.FRAMEWORK: "framework",
    ItemType.PROCESS: "process",
    ItemType.COMPARISON: "comparison",
    ItemType.MISSED: "missed",
}


class SegmentResolver:
    """Resolves segment ids to real segments and their timings."""

    def __init__(self, segments: Iterable[TranscriptSegment]) -> None:
        self._by_id: dict[str, TranscriptSegment] = {s.segment_id: s for s in segments}

    def known(self, segment_ids: Iterable[str]) -> list[str]:
        return [sid for sid in segment_ids if sid in self._by_id]

    def unknown(self, segment_ids: Iterable[str]) -> list[str]:
        return [sid for sid in segment_ids if sid not in self._by_id]

    def get(self, segment_id: str) -> TranscriptSegment | None:
        return self._by_id.get(segment_id)

    def span(self, segment_ids: Sequence[str]) -> tuple[float | None, float | None]:
        """Earliest start and latest end across the cited segments."""
        segments = [self._by_id[sid] for sid in segment_ids if sid in self._by_id]
        if not segments:
            return (None, None)
        return (
            min(segment.start_seconds for segment in segments),
            max(segment.end_seconds for segment in segments),
        )


@dataclass
class ExtractionSettings:
    """Per-run extraction knobs, derived from configuration."""

    max_items_per_chunk: int = 40
    max_retries_for_repair: int = 1
    concurrency: int = 2
    allowed_types: dict[PipelineId, tuple[ItemType, ...]] = field(default_factory=dict)


@dataclass
class ExtractionOutcome:
    """What one extraction run produced."""

    records: list[RawExtractionRecord] = field(default_factory=list)
    items: list[ExtractedItem] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok_records(self) -> list[RawExtractionRecord]:
        return [record for record in self.records if record.parsed is not None]

    @property
    def failed_records(self) -> list[RawExtractionRecord]:
        return [record for record in self.records if record.parsed is None]

    def counts_by_pipeline(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            key = item.provenance.pipeline_id.value if item.provenance else "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts


class ExtractionEngine:
    """Runs extraction pipelines over chunks."""

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        video_id: str,
        settings: ExtractionSettings | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.video_id = video_id
        self.settings = settings or ExtractionSettings()
        self._resolver = SegmentResolver([])

    # ------------------------------------------------------------------ helpers
    def bind_transcript(self, segments: Iterable[TranscriptSegment]) -> None:
        """Give the engine the real segment texts and timings for citation checks."""
        self._resolver = SegmentResolver(segments)

    def model_ref(self) -> ModelRef:
        return self.provider.model_ref(self.model)

    # --------------------------------------------------------------------- run
    async def run_pipeline(
        self,
        pipeline_id: PipelineId,
        chunk_set: ChunkSet,
        *,
        pass_index: int = 1,
        temperature: float = 0.0,
        prompt_version: str = PROMPT_VERSION,
    ) -> ExtractionOutcome:
        """Extract one pipeline over every chunk."""
        outcome = ExtractionOutcome()
        semaphore = asyncio.Semaphore(max(1, self.settings.concurrency))

        async def process(chunk: Chunk) -> RawExtractionRecord:
            async with semaphore:
                return await self._extract_one(
                    pipeline_id=pipeline_id,
                    chunk=chunk,
                    pass_index=pass_index,
                    temperature=temperature,
                    prompt_version=prompt_version,
                )

        records = await asyncio.gather(
            *(process(chunk) for chunk in chunk_set.chunks), return_exceptions=True
        )

        for chunk, result in zip(chunk_set.chunks, records, strict=True):
            if isinstance(result, BaseException):
                outcome.records.append(
                    self._failed_record(pipeline_id, chunk, pass_index, prompt_version, str(result))
                )
                outcome.failures.append(f"{pipeline_id.value}/{chunk.chunk_id}: {result}")
                continue
            outcome.records.append(result)
            if result.parsed is None:
                outcome.failures.append(
                    f"{pipeline_id.value}/{chunk.chunk_id}: {result.error or 'invalid response'}"
                )
            else:
                outcome.items.extend(result.parsed.items)

        logger.info(
            "pipeline=%s pass=%d chunks=%d items=%d failures=%d",
            pipeline_id.value,
            pass_index,
            len(chunk_set.chunks),
            len(outcome.items),
            len(outcome.failures),
        )
        return outcome

    # ------------------------------------------------------------------ internals
    async def _extract_one(
        self,
        *,
        pipeline_id: PipelineId,
        chunk: Chunk,
        pass_index: int,
        temperature: float,
        prompt_version: str,
    ) -> RawExtractionRecord:
        segment_texts: dict[str, str] = {}
        for segment_id in chunk.segment_ids:
            segment = self._resolver.get(segment_id)
            if segment is not None:
                segment_texts[segment_id] = segment.text
        prompt = build_prompt(pipeline_id, chunk, segment_texts)
        request = ProviderRequest(
            prompt=prompt,
            system=build_system_prompt(pipeline_id),
            model=self.model,
            temperature=temperature,
        )

        attempts = 0
        raw_text = ""
        error: str | None = None
        parsed: ExtractionResponse | None = None

        max_attempts = 1 + max(0, self.settings.max_retries_for_repair)
        for attempt in range(max_attempts):
            attempts = attempt + 1
            try:
                response = await self.provider.complete(request)
            except ProviderError as exc:
                error = exc.render()
                raw_text = ""
                break

            raw_text = response.text
            try:
                parsed = self._validate(
                    pipeline_id=pipeline_id,
                    chunk=chunk,
                    pass_index=pass_index,
                    prompt_version=prompt_version,
                    payload=extract_json(raw_text),
                )
                error = None
                break
            except (ValidationError, ValueError) as exc:
                error = self._describe(exc)
                if attempt + 1 < max_attempts:
                    request = ProviderRequest(
                        prompt=build_repair_prompt(prompt, error),
                        system=build_system_prompt(pipeline_id),
                        model=self.model,
                        temperature=temperature,
                    )

        record = RawExtractionRecord(
            record_id=self.record_id(pipeline_id, pass_index, chunk),
            video_id=self.video_id,
            pipeline_id=pipeline_id,
            pass_index=pass_index,
            model=self.provider.model_ref(self.model),
            chunk_ids=[chunk.chunk_id],
            prompt_version=prompt_version,
            raw_response=raw_text,
            parsed=parsed,
            state=StageState.COMPLETED if parsed is not None else StageState.FAILED,
            attempts=attempts,
            error=error,
        )
        return record

    def _validate(
        self,
        *,
        pipeline_id: PipelineId,
        chunk: Chunk,
        pass_index: int,
        prompt_version: str,
        payload: object,
    ) -> ExtractionResponse:
        """Validate a decoded payload and attach ids, provenance and timing."""
        if not isinstance(payload, dict):
            raise ValueError("response must be a JSON object")
        items = payload.get("items")
        if items is None:
            raise ValueError("response is missing the 'items' array")
        if not isinstance(items, list):
            raise ValueError("'items' must be an array")

        cleaned: list[dict[str, Any]] = []
        dropped = 0
        stripped = 0
        for ordinal, raw_item in enumerate(items, start=1):
            if not isinstance(raw_item, dict):
                dropped += 1
                continue
            item = self._clean_item(
                raw_item,
                pipeline_id=pipeline_id,
                chunk=chunk,
                pass_index=pass_index,
                prompt_version=prompt_version,
                ordinal=ordinal,
            )
            if item is None:
                dropped += 1
                continue
            stripped += int(item.pop("__stripped_keys__", 0))
            cleaned.append(item)

        notes: list[str] = []
        if dropped:
            notes.append(f"dropped {dropped} unusable item(s)")
        if stripped:
            notes.append(f"ignored {stripped} unexpected key(s) in the response")
        response = ExtractionResponse.model_validate(
            {
                "pipeline_id": pipeline_id.value,
                "items": cleaned,
                "notes": "; ".join(notes) or None,
            }
        )
        return response

    def _clean_item(
        self,
        raw_item: dict[str, object],
        *,
        pipeline_id: PipelineId,
        chunk: Chunk,
        pass_index: int,
        prompt_version: str,
        ordinal: int,
    ) -> dict[str, Any] | None:
        statement = str(raw_item.get("statement") or "").strip()
        if not statement:
            return None

        cited_raw = raw_item.get("source_segment_ids")
        if cited_raw is not None and not isinstance(cited_raw, list):
            raise ValueError("source_segment_ids must be an array of segment ids")
        cited = [str(sid).strip() for sid in cited_raw] if isinstance(cited_raw, list) else []
        valid = [sid for sid in self._resolver.known(cited) if sid in chunk.segment_ids]
        if not valid:
            # A model that cites nothing usable produces an item we cannot ground.
            return None

        details = _string_list(raw_item.get("details"), "details")
        uncertainties = _string_list(raw_item.get("uncertainties"), "uncertainties")
        unknown = self._resolver.unknown(cited)
        if unknown:
            uncertainties.append(
                "model cited unknown segment id(s): " + ", ".join(sorted(set(unknown)))
            )

        item_type = self._coerce_type(raw_item.get("type"), pipeline_id)
        start, end = self._resolver.span(valid)
        provenance = ItemProvenance(
            pipeline_id=pipeline_id,
            pass_index=pass_index,
            model=self.provider.model_ref(self.model),
            chunk_ids=[chunk.chunk_id],
            source_segment_ids=valid,
            start_seconds=start,
            end_seconds=end,
            prompt_version=prompt_version,
        )
        known_keys = {
            "item_id",
            "type",
            "statement",
            "details",
            "source_segment_ids",
            "uncertainties",
            "confidence",
            "topic_hint",
            "provenance",
        }
        unexpected = [key for key in raw_item if key not in known_keys]
        return {
            "__stripped_keys__": len(unexpected),
            "item_id": self.item_id(item_type, pipeline_id, pass_index, chunk, ordinal),
            "type": item_type.value,
            "statement": statement,
            "details": details[: self.settings.max_items_per_chunk],
            "source_segment_ids": valid,
            "uncertainties": uncertainties,
            "confidence": raw_item.get("confidence")
            if raw_item.get("confidence")
            in {
                "high",
                "medium",
                "low",
            }
            else None,
            "topic_hint": (
                str(raw_item["topic_hint"]).strip() if raw_item.get("topic_hint") else None
            ),
            "provenance": provenance.model_dump(mode="json"),
        }

    def _coerce_type(self, value: object, pipeline_id: PipelineId) -> ItemType:
        allowed = self.settings.allowed_types.get(pipeline_id)
        try:
            candidate = ItemType(str(value))
        except ValueError:
            candidate = allowed[0] if allowed else ItemType.CONCEPT
        if allowed and candidate not in allowed:
            return allowed[0]
        return candidate

    @staticmethod
    def record_id(pipeline_id: PipelineId, pass_index: int, chunk: Chunk) -> str:
        return f"{pipeline_id.value}_p{pass_index}_{chunk.chunk_id}"

    @staticmethod
    def item_id(
        item_type: ItemType,
        pipeline_id: PipelineId,
        pass_index: int,
        chunk: Chunk,
        ordinal: int,
    ) -> str:
        prefix = TYPE_PREFIXES.get(item_type, item_type.value)
        return ITEM_ID_TEMPLATE.format(
            pipeline=f"{pipeline_id.value}_{prefix}",
            pass_index=pass_index,
            chunk=chunk.chunk_id,
            ordinal=ordinal,
        )

    def _failed_record(
        self,
        pipeline_id: PipelineId,
        chunk: Chunk,
        pass_index: int,
        prompt_version: str,
        error: str,
    ) -> RawExtractionRecord:
        return RawExtractionRecord(
            record_id=self.record_id(pipeline_id, pass_index, chunk),
            video_id=self.video_id,
            pipeline_id=pipeline_id,
            pass_index=pass_index,
            model=self.provider.model_ref(self.model),
            chunk_ids=[chunk.chunk_id],
            prompt_version=prompt_version,
            raw_response="",
            state=StageState.FAILED,
            attempts=1,
            error=error,
        )

    @staticmethod
    def _describe(error: Exception) -> str:
        if isinstance(error, ValidationError):
            return "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()[:3]
            )
        return str(error)


def _string_list(value: object, field_name: str) -> list[str]:
    """Validate that a field is a list of strings; wrong types are a contract error."""
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return [item.strip() for item in value if item.strip()]


def records_path(root: Path, pipeline_id: PipelineId, pass_index: int, chunk_id: str) -> Path:
    """Where one raw extraction record is stored."""
    safe = chunk_id.replace("/", "_")
    return root / pipeline_id.value / f"pass_{pass_index}" / f"{safe}.json"


__all__ = [
    "ExtractionEngine",
    "ExtractionOutcome",
    "ExtractionSettings",
    "SegmentResolver",
    "records_path",
]
