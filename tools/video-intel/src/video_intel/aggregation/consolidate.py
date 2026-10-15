"""Aggregation: normalization, deduplication and complementary merging (§11).

Rules the specification is explicit about:

* Distinguish exact duplicates, semantic duplicates, complementary details, related but
  distinct concepts, and contradictions.
* Merge complementary statements into one item with several details.
* Do not merge two statements merely because they discuss the same topic.
* Do not discard an item because only one model extracted it.
* Deduplication must be explainable and reversible: every merge is recorded with the
  source item ids and the evidence that justified it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from video_intel.schemas.common import ItemType, VerificationStatus
from video_intel.schemas.extraction import ExtractedItem

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "out",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "also",
        "as",
        "such",
        "may",
        "might",
        "must",
        "shall",
        "should",
    ]
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b", re.IGNORECASE)

# Numbers are often spelled out in speech and in model output ("forty percent").
NUMBER_WORDS: dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}
NUMBER_WORD_RE = re.compile(rf"\b({'|'.join(NUMBER_WORDS)})\b", re.IGNORECASE)
NEGATION_RE = re.compile(
    r"\b(not|never|no|none|cannot|can't|doesn't|doesn't|isn't|wasn't|without|"
    r"neither|nor|unable|refuse[sd]?)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    lowered = text.lower().strip()
    for fancy, plain in (("\u2019", "'"), ("\u2013", "-"), ("\u2014", "-")):
        lowered = lowered.replace(fancy, plain)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", lowered)).strip()


def tokenize(text: str) -> list[str]:
    """Content tokens with stopwords removed."""
    return [token for token in TOKEN_RE.findall(normalize_text(text)) if token not in STOPWORDS]


def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def numbers_in(text: str) -> frozenset[str]:
    """Every quantity in the text, digits and spelled-out words alike.

    Comparisons need one canonical form: "40", "forty" and "forty percent" all yield
    ``{"40"}`` so a figure conflict is detected however it was written.
    """
    found = {match.group(0).lower() for match in NUMBER_RE.finditer(text)}
    for match in NUMBER_WORD_RE.finditer(text):
        found.add(NUMBER_WORDS[match.group(1).lower()])
    return frozenset(found)


def has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def content_of(item: ExtractedItem) -> str:
    return item.text


def statements(item: ExtractedItem) -> list[str]:
    """The item statement split into sentences, for fine-grained comparison."""
    return [part for part in SENTENCE_SPLIT_RE.split(item.statement) if part.strip()]


class MergeKind(StrEnum):
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    COMPLEMENTARY = "complementary"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class MergeDecision:
    """Why two items were, or were not, merged."""

    left_id: str
    right_id: str
    kind: MergeKind | None
    reason: str
    score: float

    @property
    def merged(self) -> bool:
        return self.kind is not None and self.kind is not MergeKind.CONTRADICTION


@dataclass
class ConsolidationResult:
    """Output of running the aggregator over many extractions."""

    items: list[ExtractedItem] = field(default_factory=list)
    merges: list[MergeDecision] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def merged_pairs(self) -> list[MergeDecision]:
        return [decision for decision in self.merges if decision.merged]

    @property
    def contradictions(self) -> list[MergeDecision]:
        return [decision for decision in self.merges if decision.kind is MergeKind.CONTRADICTION]


def exact_key(item: ExtractedItem) -> str:
    return normalize_text(item.text)


def classify_pair(left: ExtractedItem, right: ExtractedItem) -> MergeDecision:
    """Decide how two items relate, without merging on topic alone."""
    left_text, right_text = content_of(left), content_of(right)
    left_tokens, right_tokens = token_set(left_text), token_set(right_text)
    score = jaccard(left_tokens, right_tokens)

    if exact_key(left) == exact_key(right) or (
        left_text.strip().lower() == right_text.strip().lower()
    ):
        return MergeDecision(
            left.item_id,
            right.item_id,
            MergeKind.EXACT_DUPLICATE,
            "identical statement and details",
            1.0,
        )

    left_numbers, right_numbers = numbers_in(left_text), numbers_in(right_text)
    numeric_conflict = bool(left_numbers and right_numbers) and left_numbers != right_numbers
    negation_conflict = has_negation(left_text) != has_negation(right_text)

    if score >= SEMANTIC_THRESHOLD:
        if numeric_conflict or negation_conflict:
            return MergeDecision(
                left.item_id,
                right.item_id,
                MergeKind.CONTRADICTION,
                (
                    f"same wording but conflicting figures {sorted(left_numbers)} vs "
                    f"{sorted(right_numbers)}"
                )
                if numeric_conflict
                else "same wording but opposite polarity",
                score,
            )
        return MergeDecision(
            left.item_id,
            right.item_id,
            MergeKind.SEMANTIC_DUPLICATE,
            f"token overlap {score:.2f} with no conflicting figure or polarity",
            score,
        )

    if negation_conflict and score >= POLARITY_THRESHOLD:
        return MergeDecision(
            left.item_id,
            right.item_id,
            MergeKind.CONTRADICTION,
            f"opposite polarity on the same subject (overlap {score:.2f})",
            score,
        )

    if _is_complementary(left, right, left_tokens, right_tokens):
        return MergeDecision(
            left.item_id,
            right.item_id,
            MergeKind.COMPLEMENTARY,
            f"distinct sentences about the same subject (overlap {score:.2f})",
            score,
        )

    return MergeDecision(
        left.item_id, right.item_id, None, f"distinct statements (overlap {score:.2f})", score
    )


SUBJECT_OVERLAP = 0.3
SEMANTIC_THRESHOLD = 0.6
POLARITY_THRESHOLD = 0.4


def _is_complementary(
    left: ExtractedItem,
    right: ExtractedItem,
    left_tokens: frozenset[str],
    right_tokens: frozenset[str],
) -> bool:
    """True when items say different things about the same thing.

    The bar is deliberately high. Two items qualify when they cite overlapping evidence
    *or* clearly share a subject (high token overlap), and each one contributes wording
    the other lacks. Merely mentioning the same topic is not enough.
    """
    if left.type != right.type:
        return False
    if not left.source_segment_ids or not right.source_segment_ids:
        return False

    overlap = jaccard(left_tokens, right_tokens)
    shared_segments = set(left.source_segment_ids) & set(right.source_segment_ids)
    if not shared_segments and overlap < SUBJECT_OVERLAP:
        return False
    if overlap < 0.12:
        return False

    left_only = token_set(content_of(left)) - token_set(content_of(right))
    right_only = token_set(content_of(right)) - token_set(content_of(left))
    # Each side must add wording of its own, and together they must add real content:
    # one shared sentence plus three distinctive words is complementary, a bare
    # reword of the same words is not.
    return bool(left_only) and bool(right_only) and len(left_only | right_only) >= 3


def merge_items(keep: ExtractedItem, drop: ExtractedItem, decision: MergeDecision) -> ExtractedItem:
    """Fold ``drop`` into ``keep`` without losing evidence or detail."""
    existing_details = list(keep.details)
    existing_details.extend(
        detail
        for detail in drop.details
        if normalize_text(detail) not in {normalize_text(d) for d in existing_details}
    )
    if normalize_text(drop.statement) != normalize_text(keep.statement):
        existing_details.insert(0, drop.statement)

    segments = list(keep.source_segment_ids)
    segments.extend(sid for sid in drop.source_segment_ids if sid not in segments)

    # The consolidated item spans every passage it now cites (§5).
    starts = [
        value.start_seconds
        for value in (keep.provenance, drop.provenance)
        if value is not None and value.start_seconds is not None
    ]
    ends = [
        value.end_seconds
        for value in (keep.provenance, drop.provenance)
        if value is not None and value.end_seconds is not None
    ]
    provenance = keep.provenance
    if provenance is not None and starts and ends:
        provenance = provenance.model_copy(
            update={"start_seconds": min(starts), "end_seconds": max(ends)}
        )

    return keep.model_copy(
        update={
            "details": existing_details,
            "source_segment_ids": segments,
            "uncertainties": list(dict.fromkeys([*keep.uncertainties, *drop.uncertainties])),
            "provenance": provenance,
        }
    )


@dataclass
class Consolidator:
    """Folds duplicate and complementary extractions together, reversibly."""

    semantic_threshold: float = SEMANTIC_THRESHOLD

    def run(self, items: Sequence[ExtractedItem]) -> ConsolidationResult:
        result = ConsolidationResult()
        kept: list[ExtractedItem] = []

        for item in items:
            placed = False
            for index, existing in enumerate(kept):
                decision = classify_pair(existing, item)
                if decision.kind is MergeKind.CONTRADICTION:
                    result.merges.append(decision)
                    continue
                if decision.kind in {
                    MergeKind.EXACT_DUPLICATE,
                    MergeKind.SEMANTIC_DUPLICATE,
                    MergeKind.COMPLEMENTARY,
                }:
                    if decision.kind is MergeKind.SEMANTIC_DUPLICATE and (
                        decision.score < self.semantic_threshold
                    ):
                        continue
                    merged = merge_items(existing, item, decision)
                    kept[index] = merged
                    result.merges.append(decision)
                    placed = True
                    break
            if not placed:
                kept.append(item)

        result.items = kept
        return result


def verification_for_merge(decision: MergeDecision) -> VerificationStatus:
    """Merges do not change verification state; kept items inherit their own."""
    return VerificationStatus.NEEDS_REVIEW if not decision.merged else VerificationStatus.SUPPORTED


def group_by_type(items: Iterable[ExtractedItem]) -> dict[ItemType, list[ExtractedItem]]:
    grouped: dict[ItemType, list[ExtractedItem]] = {}
    for item in items:
        grouped.setdefault(item.type, []).append(item)
    return grouped


__all__ = [
    "STOPWORDS",
    "ConsolidationResult",
    "Consolidator",
    "MergeDecision",
    "MergeKind",
    "classify_pair",
    "content_of",
    "exact_key",
    "group_by_type",
    "has_negation",
    "jaccard",
    "merge_items",
    "normalize_text",
    "numbers_in",
    "statements",
    "token_set",
    "tokenize",
    "verification_for_merge",
]
