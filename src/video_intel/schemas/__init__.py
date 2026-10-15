"""Versioned data schemas for every persisted artifact.

Import the models from here rather than from individual modules so that schema
locations stay an implementation detail.
"""

from video_intel.schemas.chunk import Chunk, ChunkingConfig, ChunkSet, chunk_set_hash
from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ChunkId,
    Confidence,
    ExecutionMode,
    ItemId,
    ItemType,
    OutputMode,
    PipelineId,
    ProviderKind,
    SegmentId,
    StageId,
    StageState,
    StrictModel,
    TopicId,
    TranscriptKind,
    VerificationStatus,
    format_timestamp,
    slugify,
    utc_now,
)
from video_intel.schemas.conflict import ConflictRecord, ConflictSet
from video_intel.schemas.extraction import (
    ExtractedItem,
    ExtractionResponse,
    ItemProvenance,
    ModelRef,
    RawExtractionRecord,
)
from video_intel.schemas.knowledge import (
    CompletenessReport,
    ExtractionRun,
    KnowledgeDataset,
    KnowledgeItem,
    KnowledgeTopic,
    MergeRecord,
)
from video_intel.schemas.manifest import ProjectManifest, StageManifest, StageRecord
from video_intel.schemas.outputs import (
    OUTPUT_STAGES_BY_MODE,
    OutputArtifact,
    OutputRequest,
    PresentationOutline,
    SlideOutline,
)
from video_intel.schemas.transcript import (
    NormalizationStep,
    NormalizedTranscript,
    RawTranscript,
    TranscriptSegment,
    VideoMetadata,
    hash_transcript,
)

__all__ = [
    "OUTPUT_STAGES_BY_MODE",
    "SCHEMA_VERSION",
    "Chunk",
    "ChunkId",
    "ChunkSet",
    "ChunkingConfig",
    "CompletenessReport",
    "Confidence",
    "ConflictRecord",
    "ConflictSet",
    "ExecutionMode",
    "ExtractedItem",
    "ExtractionResponse",
    "ExtractionRun",
    "ItemId",
    "ItemProvenance",
    "ItemType",
    "KnowledgeDataset",
    "KnowledgeItem",
    "KnowledgeTopic",
    "MergeRecord",
    "ModelRef",
    "NormalizationStep",
    "NormalizedTranscript",
    "OutputArtifact",
    "OutputMode",
    "OutputRequest",
    "PipelineId",
    "PresentationOutline",
    "ProjectManifest",
    "ProviderKind",
    "RawExtractionRecord",
    "RawTranscript",
    "SegmentId",
    "SlideOutline",
    "StageId",
    "StageManifest",
    "StageRecord",
    "StageState",
    "StrictModel",
    "TopicId",
    "TranscriptKind",
    "TranscriptSegment",
    "VerificationStatus",
    "VideoMetadata",
    "chunk_set_hash",
    "format_timestamp",
    "hash_transcript",
    "slugify",
    "utc_now",
]
