"""Conflict detection (§12).

A conflict exists when two retained items cannot both be true of the transcript. A
conflict is recorded, never resolved by majority vote: it is only resolved when the
transcript itself clearly settles it, otherwise it stays ``needs_review``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from video_intel.aggregation.consolidate import (
    MergeKind,
    classify_pair,
    normalize_text,
    numbers_in,
)
from video_intel.schemas.common import VerificationStatus
from video_intel.schemas.conflict import ConflictRecord
from video_intel.schemas.extraction import ExtractedItem
from video_intel.schemas.transcript import TranscriptSegment
from video_intel.schemas.verification import Evidence

CONFLICT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ConflictCandidate:
    """A suspected contradiction, before it is recorded."""

    left: ExtractedItem
    right: ExtractedItem
    reason: str
    score: float


def find_candidates(items: Sequence[ExtractedItem]) -> list[ConflictCandidate]:
    """Pairs that look contradictory on their face."""
    candidates: list[ConflictCandidate] = []
    for left, right in itertools.combinations(items, 2):
        decision = classify_pair(left, right)
        if decision.kind is not MergeKind.CONTRADICTION:
            continue
        candidates.append(
            ConflictCandidate(
                left=left,
                right=right,
                reason=decision.reason,
                score=decision.score,
            )
        )
    return candidates


@dataclass(frozen=True)
class _TranscriptResolution:
    """What the transcript itself says about a disputed pair."""

    segment: TranscriptSegment
    figure: str
    supports_left: bool

    @property
    def statement(self) -> str:
        side = "the first claim" if self.supports_left else "the second claim"
        return f"segment {self.segment.segment_id} states {self.figure}, which backs {side}"


def _resolving_segment(
    candidate: ConflictCandidate, index: dict[str, TranscriptSegment]
) -> _TranscriptResolution | None:
    """Resolve only when the transcript states exactly one of the disputed figures."""
    left_numbers = numbers_in(candidate.left.text)
    right_numbers = numbers_in(candidate.right.text)
    disputed = (left_numbers | right_numbers) - (left_numbers & right_numbers)
    if not disputed:
        return None

    best: tuple[float, TranscriptSegment] | None = None
    for segment_id in {*candidate.left.source_segment_ids, *candidate.right.source_segment_ids}:
        segment = index.get(segment_id)
        if segment is None:
            continue
        spoken = numbers_in(segment.text) & disputed
        if not spoken:
            continue
        coverage = len(spoken) / len(disputed)
        if best is None or coverage > best[0]:
            best = (coverage, segment)
    if best is None:
        return None

    _coverage, segment = best
    spoken = numbers_in(segment.text) & disputed
    if len(spoken) != 1:
        # The transcript is ambiguous here: leave the conflict open for review.
        return None

    figure = sorted(spoken)[0]
    return _TranscriptResolution(
        segment=segment, figure=figure, supports_left=figure in left_numbers
    )


def build_conflicts(
    items: Sequence[ExtractedItem],
    segments: Sequence[TranscriptSegment],
) -> list[ConflictRecord]:
    """Turn candidates into records, resolving only what the transcript settles.

    Only pairs already classified as contradictions reach this point, so no topic filter
    is applied here: two statements about different subjects are never candidates.
    """
    index = {segment.segment_id: segment for segment in segments}
    conflicts: list[ConflictRecord] = []

    for ordinal, candidate in enumerate(find_candidates(items), start=1):
        left, right = candidate.left, candidate.right
        resolution: str | None = None
        status = VerificationStatus.NEEDS_REVIEW
        rationale = f"{candidate.reason}; the transcript does not clearly settle this"

        resolved = _resolving_segment(candidate, index)
        if resolved is not None:
            resolution = resolved.statement
            status = VerificationStatus.SUPPORTED
            rationale = candidate.reason

        conflicts.append(
            ConflictRecord(
                conflict_id=f"conflict_{ordinal:04d}",
                claim_ids=[left.item_id, right.item_id],
                description=(
                    f"'{_shorten(left.statement)}' conflicts with '{_shorten(right.statement)}'"
                ),
                status=status,
                resolution=resolution,
                evidence=[
                    Evidence(
                        segment_id=segment.segment_id,
                        quote=segment.text[:240],
                        match_score=None,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                    )
                    for segment in index.values()
                    if segment.segment_id in set(left.source_segment_ids)
                ],
                segment_ids=sorted(set(left.source_segment_ids) | set(right.source_segment_ids)),
                rationale=rationale,
            )
        )
    return conflicts


def _shorten(text: str, limit: int = 120) -> str:
    text = normalize_text(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def unresolvable(conflicts: Sequence[ConflictRecord]) -> list[ConflictRecord]:
    return [conflict for conflict in conflicts if not conflict.resolved]


__all__ = [
    "CONFLICT_THRESHOLD",
    "ConflictCandidate",
    "build_conflicts",
    "find_candidates",
    "unresolvable",
]
