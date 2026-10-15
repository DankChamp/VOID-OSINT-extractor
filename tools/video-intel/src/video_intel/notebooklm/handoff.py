"""NotebookLM handoff (§7, §8 of Phase 8).

NotebookLM is handled as a **manual** workflow on purpose. No supported public API for
creating notebooks and retrieving generated results is assumed here; this module only
produces the artefacts a person needs, and validates whatever they bring back:

1. ``export``  - a source-grounded packet (timestamped transcript) plus the structured
   prompts and the JSON contract to paste into NotebookLM.
2. ``import``  - reads the user's JSON answer, validates it against the application
   contract, turns it into extraction records, and merges it with the other results.

Imported items are not trusted: they go through the same transcript verification as any
model output, so a claim NotebookLM invented is flagged rather than published.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from video_intel.extraction.prompts import BASE_RULES, CONTRACT_EXAMPLE
from video_intel.schemas.common import (
    ItemType,
    PipelineId,
    ProviderKind,
    StageState,
    format_timestamp,
)
from video_intel.schemas.extraction import (
    ExtractedItem,
    ExtractionResponse,
    ItemProvenance,
    ModelRef,
    RawExtractionRecord,
)
from video_intel.schemas.knowledge import KnowledgeDataset
from video_intel.schemas.transcript import NormalizedTranscript

NOTEBOOKLM_KIND = "notebooklm_handoff"
NOTEBOOKLM_MODEL = "notebooklm"
PACKET_FILENAME = "source_packet.md"
PROMPTS_FILENAME = "extraction_prompts.md"
SCHEMA_FILENAME = "item_schema.json"
README_FILENAME = "HOW_TO.md"
IMPORT_DIR = "notebooklm"

MANUAL_WORKFLOW_NOTICE = """\
NotebookLM integration is a manual handoff.

This project does not assume that NotebookLM offers a supported public API for creating
notebooks or retrieving generated results, and it does not automate private endpoints.
What it does is make the exchange reliable:

* export:  source packet + prompts + JSON schema you paste in yourself
* import:  your JSON answer is validated, merged and verified against the transcript

Anything NotebookLM returns is treated as model output, not as fact: every imported item
is checked against the transcript and flagged when the transcript does not support it.
"""

DEFAULT_PIPELINE = PipelineId.CONCEPT

DEFAULT_PIPELINES: tuple[PipelineId, ...] = (
    PipelineId.CONCEPT,
    PipelineId.DEFINITION,
    PipelineId.FACT,
    PipelineId.EXAMPLE,
)


@dataclass(frozen=True)
class NotebookLMPacket:
    """Files prepared for a manual NotebookLM session."""

    directory: Path
    packet_path: Path
    prompts_path: Path
    schema_path: Path
    readme_path: Path

    def files(self) -> tuple[Path, ...]:
        return (
            self.packet_path,
            self.prompts_path,
            self.schema_path,
            self.readme_path,
        )


def build_source_packet(
    dataset: KnowledgeDataset,
    normalized: NormalizedTranscript,
    *,
    max_chars: int | None = None,
) -> str:
    """A plain-text, timestamped transcript for upload to NotebookLM.

    Every line keeps its segment id and real timestamp, so NotebookLM's answer can be
    mapped back onto the source when it is imported.
    """
    title = dataset.title or dataset.video_id
    lines = [
        f"# {title}",
        "",
        f"Video id: {dataset.video_id}",
        f"Segments: {normalized.segment_count} "
        f"({format_timestamp(normalized.duration_seconds)} of speech)",
        "",
        "Each line is prefixed with [segment_id start-end] so citations can be checked",
        "against the original video afterwards.",
        "",
    ]
    used = 0
    for segment in normalized.segments:
        line = (
            f"[{segment.segment_id} "
            f"{format_timestamp(segment.start_seconds)}-{format_timestamp(segment.end_seconds)}] "
            f"{segment.text}"
        )
        if max_chars is not None and used + len(line) > max_chars:
            lines.append("")
            lines.append(
                f"_packet truncated at {used} characters; raise max_chars to include the rest_"
            )
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) + "\n"


def build_prompt_document(
    dataset: KnowledgeDataset, pipelines: Sequence[PipelineId] = DEFAULT_PIPELINES
) -> str:
    """Prompts plus the contract, ready to paste into a NotebookLM note."""
    from video_intel.extraction.prompts import PROFILES

    title = dataset.title or dataset.video_id
    lines = [
        f"# Extraction prompts — {title}",
        "",
        MANUAL_WORKFLOW_NOTICE,
        "",
        "## Universal rules",
        "",
        BASE_RULES.strip(),
        "",
        "## Required JSON contract",
        "",
        "```json",
        CONTRACT_EXAMPLE.strip(),
        "```",
        "",
    ]
    for pipeline in pipelines:
        profile = PROFILES.get(pipeline)
        if profile is None:
            continue
        allowed = ", ".join(item.value for item in profile.allowed_types)
        lines.extend(
            [
                f"## Task: {pipeline.value}",
                "",
                profile.system,
                "",
                profile.focus,
                "",
                f'Allowed "type" values: {allowed}.',
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def contract_document() -> str:
    """The JSON schema for one extraction response."""
    schema = ExtractionResponse.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "video-intel extraction response"
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def export_packet(
    dataset: KnowledgeDataset,
    normalized: NormalizedTranscript,
    outputs_dir: Path,
    *,
    pipelines: Sequence[PipelineId] = DEFAULT_PIPELINES,
    max_chars: int | None = None,
) -> NotebookLMPacket:
    """Write the packet, prompts, schema and instructions into ``outputs/notebooklm``."""
    from video_intel.storage.io import write_text_atomic

    directory = outputs_dir / IMPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    packet_path = directory / PACKET_FILENAME
    prompts_path = directory / PROMPTS_FILENAME
    schema_path = directory / SCHEMA_FILENAME
    readme_path = directory / README_FILENAME

    write_text_atomic(packet_path, build_source_packet(dataset, normalized, max_chars=max_chars))
    write_text_atomic(prompts_path, build_prompt_document(dataset, pipelines))
    write_text_atomic(schema_path, contract_document())
    write_text_atomic(readme_path, _readme(dataset))

    return NotebookLMPacket(
        directory=directory,
        packet_path=packet_path,
        prompts_path=prompts_path,
        schema_path=schema_path,
        readme_path=readme_path,
    )


def _readme(dataset: KnowledgeDataset) -> str:
    return f"""\
# NotebookLM handoff — {dataset.title or dataset.video_id}

{MANUAL_WORKFLOW_NOTICE}

## Steps

1. Create a NotebookLM notebook (by hand, in the UI).
2. Upload `{PACKET_FILENAME}` as a source.
3. Copy the task prompts from `{PROMPTS_FILENAME}` into the note.
4. Ask NotebookLM for the JSON answer using the contract in `{SCHEMA_FILENAME}`.
5. Save the answer as a `.json` file.
6. Bring it back:

       video-intel notebooklm import path/to/answer.json --video {dataset.video_id}

## What happens on import

* The file is validated against the application contract; invalid files are rejected
  with the reason, and nothing is silently coerced.
* Each item must cite segment ids from the packet. Ids that do not exist are dropped and
  reported.
* Imported items are merged with the other extraction results and verified against the
  transcript exactly like any model output. A claim the transcript does not support is
  flagged `unsupported` and left out of the outputs.
"""


# ---------------------------------------------------------------------- import
@dataclass
class ImportResult:
    """What an import produced."""

    records: list[RawExtractionRecord] = field(default_factory=list)
    items: list[ExtractedItem] = field(default_factory=list)
    rejected_items: int = 0
    unknown_segment_ids: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.items) and not self.errors

    def summary(self) -> str:
        parts = [f"{len(self.items)} item(s) imported"]
        if self.rejected_items:
            parts.append(f"{self.rejected_items} rejected")
        if self.unknown_segment_ids:
            parts.append("unknown segment id(s): " + ", ".join(sorted(self.unknown_segment_ids)))
        if self.errors:
            parts.append("errors: " + "; ".join(self.errors))
        return ", ".join(parts)


class ImportedItem(BaseModel):
    """One item as a human or NotebookLM would write it.

    ``item_id`` and ``provenance`` are assigned by this application: they are derived
    from the file and the cited segments, never trusted from the input.
    """

    model_config = ConfigDict(extra="forbid")

    type: ItemType
    statement: str = Field(min_length=3)
    details: list[str] = Field(default_factory=list)
    source_segment_ids: list[str] = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)


class ImportedResponse(BaseModel):
    """The import-facing contract, mirroring :class:`ExtractionResponse`."""

    model_config = ConfigDict(extra="forbid")

    pipeline_id: PipelineId | None = None
    items: list[ImportedItem] = Field(default_factory=list)
    notes: str | None = None


class ImportValidationError(ValueError):
    """The imported file does not satisfy the contract."""


def parse_import(payload: object, *, video_id: str) -> ImportedResponse:
    """Validate an imported payload against the import contract.

    Raises:
        ImportValidationError: when the payload is not a usable extraction response.
    """
    del video_id  # kept in the signature so callers can pass context later
    if isinstance(payload, list):
        payload = {"items": payload}
    if not isinstance(payload, dict):
        raise ImportValidationError("the imported file must contain a JSON object or array")

    try:
        return ImportedResponse.model_validate(payload)
    except Exception as exc:  # pydantic validation error
        raise ImportValidationError(_describe(exc)) from exc


def _describe(error: Exception) -> str:
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            details = errors()
        except Exception:  # pragma: no cover - defensive
            details = []
        if details:
            return "; ".join(
                f"{'.'.join(str(part) for part in item.get('loc', ()))}: {item.get('msg')}"
                for item in details[:5]
            )
    return str(error)


def import_records(
    path: Path,
    *,
    video_id: str,
    known_segment_ids: set[str],
    pipeline_id: PipelineId | None = None,
    recorded_at: datetime | None = None,
) -> ImportResult:
    """Read, validate and ground an imported NotebookLM answer."""
    from video_intel.storage.io import read_json

    result = ImportResult(source_path=path)
    try:
        payload = read_json(path)
    except (FileNotFoundError, OSError) as exc:
        raise ImportValidationError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise ImportValidationError(str(exc)) from exc

    try:
        response = parse_import(payload, video_id=video_id)
    except ImportValidationError as exc:
        result.errors.append(str(exc))
        return result

    chosen_pipeline = pipeline_id or response.pipeline_id or DEFAULT_PIPELINE
    stem = path.stem.replace("/", "_")
    grounded: list[ExtractedItem] = []

    for ordinal, item in enumerate(response.items, start=1):
        cited = [sid for sid in item.source_segment_ids if sid in known_segment_ids]
        result.unknown_segment_ids.update(set(item.source_segment_ids) - set(cited))
        if not cited:
            result.rejected_items += 1
            continue
        item_id = f"item_notebooklm_{stem}_{ordinal:03d}"
        grounded.append(
            ExtractedItem(
                item_id=item_id,
                type=_as_item_type(item.type),
                statement=item.statement,
                details=list(item.details),
                source_segment_ids=cited,
                uncertainties=list(item.uncertainties),
                provenance=ItemProvenance(
                    pipeline_id=chosen_pipeline,
                    model=ModelRef(
                        provider=ProviderKind.NOTEBOOKLM_HANDOFF,
                        provider_name=NOTEBOOKLM_KIND,
                        model=NOTEBOOKLM_MODEL,
                    ),
                    source_segment_ids=cited,
                    prompt_version="notebooklm-handoff-1",
                ),
            )
        )

    result.items = grounded
    result.records.append(
        RawExtractionRecord(
            record_id=f"notebooklm_{stem}",
            video_id=video_id,
            pipeline_id=chosen_pipeline,
            model=ModelRef(
                provider=ProviderKind.NOTEBOOKLM_HANDOFF,
                provider_name=NOTEBOOKLM_KIND,
                model=NOTEBOOKLM_MODEL,
            ),
            prompt_version="notebooklm-handoff-1",
            raw_response=json.dumps(payload, ensure_ascii=False),
            parsed=_to_extraction_response(chosen_pipeline, grounded, path.name),
            state=StageState.COMPLETED if grounded else StageState.FAILED,
            attempts=1,
            error=None if grounded else "no item cited a segment from the packet",
        )
    )
    return result


def _to_extraction_response(
    pipeline_id: PipelineId, items: list[ExtractedItem], source_name: str
) -> ExtractionResponse:
    return ExtractionResponse(
        pipeline_id=pipeline_id,
        items=items,
        notes=f"imported from {source_name}",
    )


def _as_item_type(value: ItemType | str) -> ItemType:
    if isinstance(value, ItemType):
        return value
    try:
        return ItemType(value)
    except ValueError:
        return ItemType.CONCEPT


def store_import(result: ImportResult, outputs_dir: Path) -> Path:
    """Persist imported records so the next dataset build can merge them."""
    from video_intel.storage.io import write_json_atomic

    directory = outputs_dir / IMPORT_DIR / "imported"
    directory.mkdir(parents=True, exist_ok=True)
    stem = (result.source_path.stem if result.source_path else "import").replace("/", "_")
    target = directory / f"{stem}.json"
    write_json_atomic(
        target, {"records": [record.model_dump(mode="json") for record in result.records]}
    )
    return target


def load_imported_records(outputs_dir: Path) -> list[RawExtractionRecord]:
    """Every record previously imported for this video."""
    directory = outputs_dir / IMPORT_DIR / "imported"
    if not directory.is_dir():
        return []
    records: list[RawExtractionRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload.get("records", []):
                records.append(RawExtractionRecord.model_validate(entry))
        except (ValueError, OSError):
            # A corrupt import file must not take the dataset down; it is skipped and
            # the rest still merge.
            continue
    return records


__all__ = [
    "DEFAULT_PIPELINE",
    "DEFAULT_PIPELINES",
    "IMPORT_DIR",
    "MANUAL_WORKFLOW_NOTICE",
    "ImportResult",
    "ImportValidationError",
    "ImportedItem",
    "ImportedResponse",
    "NotebookLMPacket",
    "build_prompt_document",
    "build_source_packet",
    "contract_document",
    "export_packet",
    "import_records",
    "load_imported_records",
    "parse_import",
    "store_import",
]
