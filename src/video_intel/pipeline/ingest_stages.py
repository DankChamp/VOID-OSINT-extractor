"""Ingestion and chunking stages (Phases 2 and 3).

Each stage is a plain function that returns a validated schema object. The runner
handles caching, atomic writes and state recording, so stages stay small and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from video_intel.config.models import AppConfig
from video_intel.ingest.service import IngestRequest, TranscriptIngestor
from video_intel.logging_setup import get_logger
from video_intel.pipeline.chunking import chunk_transcript
from video_intel.pipeline.runner import StageContext, StageOutcome, StageRunner, payload_hash
from video_intel.pipeline.stages import INGESTION_STAGES, stages_up_to
from video_intel.schemas.chunk import ChunkSet
from video_intel.schemas.common import StageId
from video_intel.schemas.transcript import NormalizedTranscript, RawTranscript, VideoMetadata

logger = get_logger(__name__)

IMPLEMENTED_STAGES: tuple[StageId, ...] = INGESTION_STAGES

STAGE_TITLES: dict[StageId, str] = {
    StageId.METADATA: "Video metadata",
    StageId.RAW_TRANSCRIPT: "Raw transcript",
    StageId.NORMALIZED_TRANSCRIPT: "Normalized transcript",
    StageId.CHUNKS: "Chunks",
}


@dataclass(frozen=True)
class IngestionOutcome:
    """Aggregate result of the Phase 2 stage run.

    ``transcript`` and ``normalized`` are ``None`` when the run stopped earlier
    (``--until metadata``), so a partial run can never look complete.
    """

    metadata: VideoMetadata
    transcript: RawTranscript | None
    normalized: NormalizedTranscript | None
    stages: tuple[StageOutcome, ...]
    chunks: ChunkSet | None = None

    @property
    def reused_stages(self) -> tuple[str, ...]:
        return tuple(stage.stage_id.value for stage in self.stages if stage.reused)

    @property
    def executed_stages(self) -> tuple[str, ...]:
        return tuple(stage.stage_id.value for stage in self.stages if not stage.reused)


def selected_stages(until: StageId | None) -> frozenset[StageId]:
    """Stages to execute for a run that stops at ``until``.

    Stages later than the ingestion phase are accepted and simply mean "run everything
    ingestion owns"; only stages this build does not implement are rejected.
    """
    return frozenset(stages_up_to(until)) & frozenset(IMPLEMENTED_STAGES)


def transcript_input_hash(video_id: str, languages: tuple[str, ...]) -> str:
    """Cache identity of a transcript fetch request."""
    return payload_hash(f"{video_id}|{','.join(languages)}")


def ingest_stages(
    *,
    runner: StageRunner,
    ingestor: TranscriptIngestor,
    request: IngestRequest,
    config: AppConfig,
    force: bool = False,
    until: StageId | None = None,
) -> IngestionOutcome:
    """Run the Phase 2 stages up to ``until``, reusing cached work.

    Args:
        until: last stage to run; defaults to the final implemented stage.

    Stages run in order and each one is skipped when a valid cached artifact exists,
    so an interrupted run resumes where it stopped.
    """
    context: StageContext = runner.context
    video_id = context.video_id
    languages = tuple(request.languages) or ("en",)
    selected = selected_stages(until)
    stages: list[StageOutcome] = []

    metadata_outcome = runner.run(
        StageId.METADATA,
        lambda: ingestor.fetch_metadata(video_id),
        artifact_path=context.paths.metadata,
        input_hash=payload_hash(video_id),
        force=force,
    )
    stages.append(metadata_outcome)
    metadata = _load(metadata_outcome.path, VideoMetadata)

    if StageId.RAW_TRANSCRIPT not in selected:
        return IngestionOutcome(
            metadata=metadata, transcript=None, normalized=None, stages=tuple(stages)
        )

    def fetch() -> RawTranscript:
        return ingestor.ingest(request, metadata=metadata).transcript

    transcript_outcome = runner.run(
        StageId.RAW_TRANSCRIPT,
        fetch,
        artifact_path=context.paths.raw_transcript,
        input_hash=transcript_input_hash(video_id, languages),
        force=force,
    )
    stages.append(transcript_outcome)
    transcript = _load(transcript_outcome.path, RawTranscript)

    if StageId.NORMALIZED_TRANSCRIPT not in selected:
        return IngestionOutcome(
            metadata=metadata, transcript=transcript, normalized=None, stages=tuple(stages)
        )

    def normalize() -> NormalizedTranscript:
        return ingestor.normalize(transcript)

    normalized_outcome = runner.run(
        StageId.NORMALIZED_TRANSCRIPT,
        normalize,
        artifact_path=context.paths.normalized_transcript,
        input_hash=payload_hash(transcript.text()),
        config_hash=config.hash(),
        force=force,
    )
    stages.append(normalized_outcome)

    return IngestionOutcome(
        metadata=metadata,
        transcript=transcript,
        normalized=_load(normalized_outcome.path, NormalizedTranscript),
        stages=tuple(stages),
    )


def chunk_stage(
    *,
    runner: StageRunner,
    config: AppConfig,
    force: bool = False,
) -> StageOutcome:
    """Chunk the normalized transcript (Phase 3).

    The stage reads ``normalized_transcript.json`` written by the previous stage, so
    changing the chunk size re-chunks the existing transcript without re-fetching it.
    """
    context = runner.context
    normalized_path = context.paths.normalized_transcript
    if not normalized_path.is_file():
        raise FileNotFoundError(
            f"cannot chunk before normalization has run: {normalized_path} is missing"
        )
    normalized = _load(normalized_path, NormalizedTranscript)

    outcome = runner.run(
        StageId.CHUNKS,
        lambda: chunk_transcript(
            normalized.segments,
            config.chunking,
            video_id=normalized.video_id,
            transcript_hash=normalized.source_transcript_hash,
        ),
        artifact_path=context.paths.chunks,
        input_hash=payload_hash(
            f"{normalized.source_transcript_hash}|{config.chunking.model_dump_json()}"
        ),
        config_hash=config.hash(),
        force=force,
    )
    return outcome


def chunks_after_normalization(
    *,
    runner: StageRunner,
    config: AppConfig,
    until: StageId | None = None,
    force: bool = False,
) -> tuple[ChunkSet | None, StageOutcome | None]:
    """Run the chunk stage when it is part of the selected stage prefix."""
    if StageId.CHUNKS not in selected_stages(until):
        return None, None
    outcome = chunk_stage(runner=runner, config=config, force=force)
    return _load(outcome.path, ChunkSet), outcome


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load(path: Path | None, model: type[ModelT]) -> ModelT:
    """Read a stage artifact back from disk as a validated model."""
    from video_intel.storage.io import read_model

    if path is None or not path.is_file():
        raise RuntimeError(f"stage artifact missing on disk: {path}")
    return read_model(path, model)
