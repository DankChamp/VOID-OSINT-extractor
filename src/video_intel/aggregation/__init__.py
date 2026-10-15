"""Aggregation and verification (§6 of Phase 6)."""

from video_intel.aggregation.conflicts import build_conflicts, find_candidates, unresolvable
from video_intel.aggregation.consolidate import (
    Consolidator,
    MergeDecision,
    MergeKind,
    classify_pair,
    normalize_text,
)
from video_intel.aggregation.dataset import (
    DatasetBuilder,
    DatasetInputs,
    DatasetResult,
    empty_dataset,
)
from video_intel.aggregation.verify import (
    EvidenceChecker,
    SegmentIndex,
    VerificationOutcome,
    verify_items,
)

__all__ = [
    "Consolidator",
    "DatasetBuilder",
    "DatasetInputs",
    "DatasetResult",
    "EvidenceChecker",
    "MergeDecision",
    "MergeKind",
    "SegmentIndex",
    "VerificationOutcome",
    "build_conflicts",
    "classify_pair",
    "empty_dataset",
    "find_candidates",
    "normalize_text",
    "unresolvable",
    "verify_items",
]
