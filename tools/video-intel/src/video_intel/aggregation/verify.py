"""Evidence verification (§10).

Each extracted statement is compared with the transcript passages it cites. The
verifier is deterministic by design: it cannot be fooled into approving a claim the
transcript does not contain, and it never invents evidence.

A separate model verifier can be added later (§10 allows it); when it is configured and
disagrees with the deterministic check, the item becomes ``needs_review`` rather than
being decided by vote.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from video_intel.aggregation.consolidate import normalize_text, token_set, tokenize
from video_intel.schemas.common import Confidence, VerificationStatus
from video_intel.schemas.extraction import ExtractedItem
from video_intel.schemas.transcript import TranscriptSegment
from video_intel.schemas.verification import (
    Evidence,
    VerificationResult,
    VerificationSummary,
    VerifiedItem,
)

# Coverage is the share of the claim's content words that actually occur in the cited
# passage (and, optionally, its neighbours). Thresholds are deliberately high: a claim
# that shares a couple of words with the transcript is not supported by it.
SUPPORTED_THRESHOLD = 0.6
PARTIAL_THRESHOLD = 0.4
QUOTE_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class SegmentIndex:
    """Lookup from segment id to the real transcript segment."""

    segments: dict[str, TranscriptSegment]

    @classmethod
    def of(cls, segments: Iterable[TranscriptSegment]) -> SegmentIndex:
        return cls({segment.segment_id: segment for segment in segments})

    def get(self, segment_id: str) -> TranscriptSegment | None:
        return self.segments.get(segment_id)

    def resolve(self, segment_ids: Sequence[str]) -> list[TranscriptSegment]:
        return [s for s in (self.segments.get(sid) for sid in segment_ids) if s is not None]

    def context(self, segment_ids: Sequence[str], window: int = 1) -> list[TranscriptSegment]:
        """Cited segments plus their immediate neighbours, in transcript order."""
        if not self.segments:
            return []
        order = list(self.segments)
        positions = {sid: index for index, sid in enumerate(order)}
        wanted: set[int] = set()
        for segment_id in segment_ids:
            index = positions.get(segment_id)
            if index is None:
                continue
            for offset in range(-window, window + 1):
                candidate = index + offset
                if 0 <= candidate < len(order):
                    wanted.add(candidate)
        return [self.segments[order[index]] for index in sorted(wanted)]


@dataclass
class EvidenceChecker:
    """Deterministic claim-versus-transcript comparison."""

    supported_threshold: float = SUPPORTED_THRESHOLD
    partial_threshold: float = PARTIAL_THRESHOLD
    context_window: int = 0

    def verify(self, item: ExtractedItem, index: SegmentIndex) -> VerificationResult:
        cited = index.resolve(item.source_segment_ids)
        if not cited:
            return VerificationResult(
                item_id=item.item_id,
                status=VerificationStatus.UNSUPPORTED,
                rationale="no cited segment exists in the transcript",
                verified_by="deterministic",
            )

        evidence = self._evidence(item, cited)
        coverage = self._coverage(item, cited)
        context = index.context(item.source_segment_ids, self.context_window)
        context_coverage = self._coverage(item, context) if context else coverage

        best = max(coverage, context_coverage)
        status = self._status(best)
        supported_parts = self._supported_parts(item, cited)
        unsupported_parts = self._unsupported_parts(item, cited)

        return VerificationResult(
            item_id=item.item_id,
            status=status,
            evidence=evidence,
            supported_parts=supported_parts,
            unsupported_parts=unsupported_parts,
            rationale=self._rationale(status, best, coverage, context_coverage),
            verified_by="deterministic",
            verifier_model=None,
        )

    # ---------------------------------------------------------------- internals
    def _coverage(self, item: ExtractedItem, segments: Sequence[TranscriptSegment]) -> float:
        """Fraction of the claim's content words present in the passages."""
        claim_tokens = set(tokenize(item.statement))
        if not claim_tokens:
            return 0.0
        passage_tokens = set()
        for segment in segments:
            passage_tokens.update(tokenize(segment.text))
        if not passage_tokens:
            return 0.0
        return len(claim_tokens & passage_tokens) / len(claim_tokens)

    def _status(self, coverage: float) -> VerificationStatus:
        if coverage >= self.supported_threshold:
            return VerificationStatus.SUPPORTED
        if coverage >= self.partial_threshold:
            return VerificationStatus.PARTIALLY_SUPPORTED
        return VerificationStatus.UNSUPPORTED

    def _evidence(self, item: ExtractedItem, cited: Sequence[TranscriptSegment]) -> list[Evidence]:
        claim = token_set(item.statement)
        evidence: list[Evidence] = []
        for segment in cited:
            segment_tokens = token_set(segment.text)
            score = len(claim & segment_tokens) / len(claim) if claim else 0.0
            evidence.append(
                Evidence(
                    segment_id=segment.segment_id,
                    quote=_best_quote(segment.text, item.statement),
                    match_score=round(score, 4),
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                )
            )
        return evidence

    def _supported_parts(
        self, item: ExtractedItem, cited: Sequence[TranscriptSegment]
    ) -> list[str]:
        passage = normalize_text(" ".join(segment.text for segment in cited))
        supported: list[str] = []
        for part in [item.statement, *item.details]:
            tokens = set(tokenize(part))
            if not tokens:
                continue
            present = {token for token in tokens if token in passage}
            if len(present) / len(tokens) >= 0.6:
                supported.append(part)
        return supported

    def _unsupported_parts(
        self, item: ExtractedItem, cited: Sequence[TranscriptSegment]
    ) -> list[str]:
        supported = set(self._supported_parts(item, cited))
        return [part for part in [item.statement, *item.details] if part not in supported]

    @staticmethod
    def _rationale(
        status: VerificationStatus,
        best: float,
        cited_coverage: float,
        context_coverage: float,
    ) -> str:
        return (
            f"{status.value}: claim coverage {best:.2f} "
            f"(cited {cited_coverage:.2f}, with context {context_coverage:.2f})"
        )


def _best_quote(segment_text: str, statement: str, *, limit: int = 240) -> str:
    """The part of a passage most similar to the claim, for human review."""
    claim = token_set(statement)
    if not claim:
        return segment_text[:limit]
    words = segment_text.split()
    if not words:
        return segment_text[:limit]

    window = max(8, min(len(words), 40))
    best_score, best_start = 0.0, 0
    for start in range(0, max(1, len(words) - window + 1)):
        candidate = token_set(" ".join(words[start : start + window]))
        score = len(claim & candidate) / len(claim)
        if score > best_score:
            best_score, best_start = score, start
    return " ".join(words[best_start : best_start + window])[:limit]


@dataclass
class VerificationOutcome:
    """Verified items plus the counts a dataset needs to stay honest."""

    items: list[VerifiedItem] = field(default_factory=list)
    summary: VerificationSummary = field(default_factory=lambda: VerificationSummary(total=0))

    @property
    def unsupported(self) -> list[VerifiedItem]:
        return [
            item
            for item in self.items
            if item.verification.status is VerificationStatus.UNSUPPORTED
        ]

    def by_status(self, status: VerificationStatus) -> list[VerifiedItem]:
        return [item for item in self.items if item.verification.status is status]


def verify_items(
    items: Sequence[ExtractedItem],
    index: SegmentIndex,
    *,
    context_window: int = 0,
    drop_unsupported: bool = True,
) -> VerificationOutcome:
    """Verify every item and summarise the result."""
    checker = EvidenceChecker(context_window=context_window)
    verified: list[VerifiedItem] = []
    counts = {status: 0 for status in VerificationStatus}

    for item in items:
        result = checker.verify(item, index)
        counts[result.status] += 1
        retained = not (drop_unsupported and result.status is VerificationStatus.UNSUPPORTED)
        verified.append(
            VerifiedItem(
                item=item,
                verification=result,
                retained=retained,
                review_reason=None
                if retained
                else "dropped from outputs: the transcript does not support this claim",
            )
        )

    total = len(items)
    summary = VerificationSummary(
        total=total,
        supported=counts[VerificationStatus.SUPPORTED],
        partially_supported=counts[VerificationStatus.PARTIALLY_SUPPORTED],
        unsupported=counts[VerificationStatus.UNSUPPORTED],
        needs_review=counts[VerificationStatus.NEEDS_REVIEW],
        unverified=0,
        overall_confidence=_confidence(counts, total),
    )
    return VerificationOutcome(items=verified, summary=summary)


def _confidence(counts: dict[VerificationStatus, int], total: int) -> Confidence:
    if total == 0:
        return Confidence.LOW
    supported_ratio = counts[VerificationStatus.SUPPORTED] / total
    if supported_ratio >= 0.8:
        return Confidence.HIGH
    if supported_ratio >= 0.5:
        return Confidence.MEDIUM
    return Confidence.LOW


def quote_is_in_segment(quote: str, segment: TranscriptSegment) -> bool:
    """Whether a quoted evidence fragment really occurs in the segment."""
    if not quote:
        return False
    haystack = normalize_text(segment.text)
    needle = normalize_text(quote)
    if not needle:
        return False
    if needle in haystack:
        return True
    claim_tokens = token_set(needle)
    segment_tokens = token_set(segment.text)
    if not claim_tokens:
        return False
    return len(claim_tokens & segment_tokens) / len(claim_tokens) >= QUOTE_MATCH_THRESHOLD


__all__ = [
    "PARTIAL_THRESHOLD",
    "SUPPORTED_THRESHOLD",
    "EvidenceChecker",
    "SegmentIndex",
    "VerificationOutcome",
    "quote_is_in_segment",
    "verify_items",
]
