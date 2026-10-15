"""Analysis stages: extraction, verification, dataset assembly, outputs (Phases 5-7).

These stages sit on top of the Phase 2/3 stages and the cache-aware runner:

* ``extractions``  - one provider call per (pipeline, pass, chunk), raw responses stored.
* ``dataset``      - consolidation, verification, conflicts, hierarchy (§6).
* each output stage - rendered from the dataset, keyed by the dataset hash so changing a
  format never re-runs extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from video_intel.aggregation.dataset import DatasetBuilder, DatasetInputs, DatasetResult
from video_intel.config.models import AppConfig, PipelineConfig
from video_intel.extraction.engine import (
    ExtractionEngine,
    ExtractionOutcome,
    ExtractionSettings,
    records_path,
)
from video_intel.extraction.prompts import PROMPT_VERSION
from video_intel.logging_setup import get_logger
from video_intel.outputs.render import RenderedOutput, dataset_hash, render_outputs
from video_intel.pipeline.runner import (
    StageContext,
    StageOutcome,
    StageRunner,
    file_hash,
    payload_hash,
)
from video_intel.pipeline.stages import ANALYSIS_STAGES as ANALYSIS_ORDER
from video_intel.providers.base import Provider
from video_intel.schemas.chunk import Chunk, ChunkSet
from video_intel.schemas.common import (
    PipelineId,
    StageId,
    StageState,
)
from video_intel.schemas.extraction import RawExtractionRecord
from video_intel.schemas.knowledge import ExtractionRun, KnowledgeDataset
from video_intel.schemas.transcript import NormalizedTranscript
from video_intel.storage.io import read_model, write_json_atomic, write_model_atomic

logger = get_logger(__name__)

ANALYSIS_STAGES: tuple[StageId, ...] = ANALYSIS_ORDER

OUTPUT_STAGE_FILES: dict[StageId, str] = {
    StageId.OUTPUT_NOTES: "full_notes.md",
    StageId.OUTPUT_SUMMARY: "summary.md",
    StageId.OUTPUT_PPT: "ppt_outline.md",
    StageId.OUTPUT_JSON: "knowledge.json",
}


def _cache_mismatch(
    record: RawExtractionRecord,
    *,
    pipeline: PipelineConfig,
    pass_index: int,
    model: str,
    provider_name: str,
) -> str | None:
    """Why a stored record cannot be reused for this run, or ``None`` if it can.

    Cache reuse is only valid when the same pipeline, model, provider and prompt
    produced the record; otherwise the result belongs to a different configuration.
    """
    if record.pipeline_id != pipeline.pipeline_id:
        return f"pipeline {record.pipeline_id.value} != {pipeline.pipeline_id.value}"
    if record.model.model != model:
        return f"model {record.model.model!r} != {model!r}"
    if record.model.provider_name != provider_name:
        return f"provider {record.model.provider_name!r} != {provider_name!r}"
    expected_prompt = pipeline.prompt_version or PROMPT_VERSION
    if record.prompt_version != expected_prompt:
        return f"prompt {record.prompt_version!r} != {expected_prompt!r}"
    if record.pass_index != pass_index:
        return f"pass {record.pass_index} != {pass_index}"
    return None


def _allowed_types(pipelines: list[PipelineConfig]) -> dict[PipelineId, tuple[Any, ...]]:
    """Item types each pipeline may produce, taken from its prompt profile."""
    from video_intel.extraction.prompts import PROFILES

    allowed: dict[PipelineId, tuple[Any, ...]] = {}
    for pipeline in pipelines:
        profile = PROFILES.get(pipeline.pipeline_id)
        if profile is not None:
            allowed[pipeline.pipeline_id] = profile.allowed_types
    return allowed


@dataclass
class ExtractionStageResult:
    """Outcome of the extraction stage."""

    outcome: ExtractionOutcome
    records_written: int = 0
    reused_records: int = 0
    runs: list[ExtractionRun] = field(default_factory=list)


async def run_extractions(
    *,
    runner: StageRunner,
    provider: Provider,
    model: str,
    chunk_set: ChunkSet,
    normalized: NormalizedTranscript,
    pipelines: list[PipelineConfig],
    max_passes: int = 1,
    force: bool = False,
    concurrency: int = 2,
    on_progress: Callable[[str], None] | None = None,
) -> ExtractionStageResult:
    """Extract every configured pipeline, reusing per-record results from disk."""
    context = runner.context
    root = context.paths.raw_extractions
    context.paths.ensure()

    settings = ExtractionSettings(
        allowed_types=_allowed_types(pipelines),
        concurrency=max(1, min(8, concurrency)),
    )
    engine = ExtractionEngine(provider, model=model, video_id=context.video_id, settings=settings)
    engine.bind_transcript(normalized.segments)

    aggregate = ExtractionOutcome()
    runs: list[ExtractionRun] = []
    written = 0
    reused = 0

    for pipeline_config in pipelines:
        pipeline_id = pipeline_config.pipeline_id
        passes = min(max(1, pipeline_config.passes), max(1, max_passes))
        items_extracted = 0
        chunks_processed = 0

        for pass_index in range(1, passes + 1):
            pending: list[Chunk] = []
            for chunk in chunk_set.chunks:
                path = records_path(root, pipeline_id, pass_index, chunk.chunk_id)
                if force or not path.is_file():
                    pending.append(chunk)
                    continue

                record = read_model(path, RawExtractionRecord)
                stale_reason = _cache_mismatch(
                    record,
                    pipeline=pipeline_config,
                    pass_index=pass_index,
                    model=model,
                    provider_name=provider.name,
                )
                if stale_reason is not None:
                    # A record produced by a different model, prompt or provider is not
                    # this run's result: re-extract instead of silently reusing it.
                    logger.info(
                        "record %s is stale (%s); re-extracting",
                        record.record_id,
                        stale_reason,
                    )
                    path.unlink(missing_ok=True)
                    pending.append(chunk)
                    continue

                aggregate.records.append(record)
                if record.parsed is not None:
                    aggregate.items.extend(record.parsed.items)
                    items_extracted += len(record.parsed.items)
                else:
                    aggregate.failures.append(
                        f"{pipeline_id.value}/{chunk.chunk_id}: "
                        f"{record.error or 'invalid response'}"
                    )
                chunks_processed += 1
                reused += 1

            if not pending:
                continue

            if on_progress is not None:
                on_progress(f"{pipeline_id.value} pass {pass_index}: {len(pending)} chunk(s)")

            outcome = await engine.run_pipeline(
                pipeline_id,
                chunk_set,
                pass_index=pass_index,
                temperature=pipeline_config.temperature,
                prompt_version=pipeline_config.prompt_version or PROMPT_VERSION,
            )
            aggregate.records.extend(outcome.records)
            aggregate.items.extend(outcome.items)
            aggregate.failures.extend(outcome.failures)
            items_extracted += len(outcome.items)

            for record in outcome.records:
                chunk_id = record.chunk_ids[0]
                write_model_atomic(records_path(root, pipeline_id, pass_index, chunk_id), record)
                written += 1
            chunks_processed += len(outcome.records)

        runs.append(
            ExtractionRun(
                pipeline_id=pipeline_id,
                model=provider.model_ref(model),
                stage_state=StageState.COMPLETED,
                items_extracted=items_extracted,
                items_retained=items_extracted,
                chunks_processed=chunks_processed,
            )
        )

    summary = {
        "video_id": context.video_id,
        "pipelines": [run.model_dump(mode="json") for run in runs],
        "records": len(aggregate.records),
        "failures": aggregate.failures,
    }
    index_path = root / "index.json"
    write_json_atomic(index_path, summary)

    stage_state = StageState.COMPLETED if aggregate.ok_records else StageState.FAILED
    runner.note(
        StageId.EXTRACTIONS,
        stage_state,
        input_hash=payload_hash(
            f"{chunk_set.cache_key}|" + "|".join(sorted(p.cache_key() for p in pipelines))
        ),
        output_hash=file_hash(index_path),
        config_hash=None,
        error=None
        if stage_state is StageState.COMPLETED
        else (f"{len(aggregate.failures)} extraction call(s) failed"),
    )

    return ExtractionStageResult(
        outcome=aggregate, records_written=written, reused_records=reused, runs=runs
    )


def build_dataset(
    *,
    video_id: str,
    title: str | None,
    normalized: NormalizedTranscript,
    extraction: ExtractionOutcome,
    runs: list[ExtractionRun],
    transcript_hash: str | None,
    config_hash: str | None,
    completed: list[StageId],
    failed: list[StageId],
    config: AppConfig,
) -> DatasetResult:
    """Consolidate, verify and assemble the hierarchical dataset."""
    return DatasetBuilder().build(
        DatasetInputs(
            video_id=video_id,
            title=title,
            segments=normalized.segments,
            items=list(extraction.items),
            transcript_hash=transcript_hash,
            config_hash=config_hash,
            extraction_runs=runs,
            completed_stages=completed,
            failed_stages=failed,
            context_window=config.verification.context_window_segments,
            drop_unsupported=config.verification.drop_unsupported_from_outputs,
        )
    )


def dataset_stage(
    *,
    runner: StageRunner,
    dataset: KnowledgeDataset,
    input_hash: str,
    config_hash: str | None,
    force: bool = False,
) -> StageOutcome:
    """Persist the dataset as ``knowledge.json`` with cache metadata."""
    context = runner.context
    return runner.run(
        StageId.KNOWLEDGE_DATASET,
        lambda: dataset,
        artifact_path=context.paths.knowledge,
        input_hash=input_hash,
        config_hash=config_hash,
        force=force,
    )


def load_dataset(context: StageContext) -> KnowledgeDataset:
    if not context.paths.knowledge.is_file():
        raise FileNotFoundError(
            f"no dataset has been built yet: {context.paths.knowledge} is missing"
        )
    return read_model(context.paths.knowledge, KnowledgeDataset)


def output_stages(
    *,
    runner: StageRunner,
    dataset: KnowledgeDataset,
    stages: set[StageId],
    force: bool = False,
) -> dict[StageId, StageOutcome]:
    """Render the requested outputs, each keyed by the dataset hash."""
    rendered: dict[str, RenderedOutput] = render_outputs(dataset)
    digest = dataset_hash(dataset)
    results: dict[StageId, StageOutcome] = {}
    context = runner.context

    for stage_id in sorted(stages, key=lambda stage: stage.value):
        filename = OUTPUT_STAGE_FILES[stage_id]
        output = rendered[filename]

        def render_text(output: RenderedOutput = output) -> str:
            return output.content

        results[stage_id] = runner.run(
            stage_id,
            render_text,
            artifact_path=context.paths.output_file(filename),
            input_hash=digest,
            force=force,
        )
    return results


__all__ = [
    "ANALYSIS_STAGES",
    "OUTPUT_STAGE_FILES",
    "ExtractionStageResult",
    "build_dataset",
    "dataset_stage",
    "load_dataset",
    "output_stages",
    "run_extractions",
]
