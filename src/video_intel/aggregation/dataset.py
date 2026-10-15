"""Dataset assembly (§13).

Turns consolidated and verified extractions into one hierarchical dataset:

* Topics come from the transcript, not from a fixed outline. Concept items become
  candidate topics; items are attached by the segments they cite, so membership is
  evidence-based.
* Every item keeps its evidence, timestamps, verification state and extraction metadata.
* Merges stay reversible: each consolidated item records the item ids it absorbed.
* ``CompletenessReport`` states exactly what was and was not done, so a partial run can
  never be presented as a finished dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from video_intel.aggregation.conflicts import build_conflicts
from video_intel.aggregation.consolidate import Consolidator, MergeDecision, normalize_text
from video_intel.aggregation.verify import SegmentIndex, VerificationOutcome, verify_items
from video_intel.schemas.common import (
    ItemType,
    PipelineId,
    StageId,
    TopicId,
    VerificationStatus,
    utc_now,
)
from video_intel.schemas.conflict import ConflictRecord
from video_intel.schemas.extraction import ExtractedItem
from video_intel.schemas.knowledge import (
    CompletenessReport,
    ExtractionRun,
    KnowledgeDataset,
    KnowledgeItem,
    KnowledgeTopic,
    MergeRecord,
)
from video_intel.schemas.transcript import TranscriptSegment

TYPE_LABELS: dict[ItemType, str] = {
    ItemType.CONCEPT: "Concepts",
    ItemType.DEFINITION: "Definitions",
    ItemType.FACT: "Facts and figures",
    ItemType.CLAIM: "Important claims",
    ItemType.EXAMPLE: "Examples",
    ItemType.FRAMEWORK: "Frameworks and models",
    ItemType.PROCESS: "Processes",
    ItemType.COMPARISON: "Comparisons and trade-offs",
    ItemType.RELATIONSHIP: "Relationships",
    ItemType.MISSED: "Coverage gaps flagged by the audit",
}

TOPIC_EVIDENCE_SHARE = 0.5
TOPIC_TITLE_LIMIT = 120


@dataclass
class DatasetInputs:
    """Everything the builder needs."""

    video_id: str
    title: str | None
    segments: Sequence[TranscriptSegment]
    items: Sequence[ExtractedItem]
    transcript_hash: str | None = None
    config_hash: str | None = None
    extraction_runs: list[ExtractionRun] = field(default_factory=list)
    completed_stages: list[StageId] = field(default_factory=list)
    failed_stages: list[StageId] = field(default_factory=list)
    context_window: int = 0
    drop_unsupported: bool = True


@dataclass
class DatasetResult:
    """The dataset plus the intermediate decisions that produced it."""

    dataset: KnowledgeDataset
    verification: VerificationOutcome
    merges: list[MergeDecision]
    retained: list[KnowledgeItem]
    dropped: list[KnowledgeItem]


class DatasetBuilder:
    """Consolidates, verifies and assembles the final dataset."""

    def __init__(self, *, semantic_threshold: float = 0.6) -> None:
        self.semantic_threshold = semantic_threshold

    def build(self, inputs: DatasetInputs) -> DatasetResult:
        index = SegmentIndex.of(inputs.segments)

        consolidation = Consolidator(semantic_threshold=self.semantic_threshold).run(inputs.items)
        verification = verify_items(
            consolidation.items,
            index,
            context_window=inputs.context_window,
            drop_unsupported=inputs.drop_unsupported,
        )

        merge_records = self._merge_records(consolidation.merges)
        retained: list[KnowledgeItem] = []
        dropped: list[KnowledgeItem] = []

        for verified in verification.items:
            knowledge_item = self._to_item(verified.item, verified.verification.status)
            if verified.retained:
                retained.append(knowledge_item)
            else:
                dropped.append(knowledge_item)

        topics = self._build_topics(retained)
        conflicts = build_conflicts(
            [verified.item for verified in verification.items if verified.retained],
            list(inputs.segments),
        )

        completeness = CompletenessReport(
            completed_stages=list(inputs.completed_stages),
            failed_stages=list(inputs.failed_stages),
            skipped_stages=[
                stage
                for stage in (StageId.VERIFY, StageId.CONFLICTS)
                if stage not in inputs.completed_stages
            ],
            unverified_items=verification.summary.unverified + verification.summary.needs_review,
            unresolved_conflicts=sum(1 for conflict in conflicts if not conflict.resolved),
            missing_items_from_coverage_audit=sum(
                1 for item in retained if item.type == ItemType.MISSED.value
            ),
        )

        dataset = KnowledgeDataset(
            video_id=inputs.video_id,
            title=inputs.title,
            topics=topics,
            items=retained,
            conflicts=conflicts,
            merges=merge_records,
            extractions=inputs.extraction_runs,
            transcript_hash=inputs.transcript_hash,
            config_hash=inputs.config_hash,
            completeness=completeness,
        )
        return DatasetResult(
            dataset=dataset,
            verification=verification,
            merges=consolidation.merges,
            retained=retained,
            dropped=dropped,
        )

    # ---------------------------------------------------------------- internals
    def _to_item(self, item: ExtractedItem, status: VerificationStatus) -> KnowledgeItem:
        starts = [
            provenance.start_seconds
            for provenance in ([item.provenance] if item.provenance else [])
            if provenance is not None and provenance.start_seconds is not None
        ]
        ends = [
            provenance.end_seconds
            for provenance in ([item.provenance] if item.provenance else [])
            if provenance is not None and provenance.end_seconds is not None
        ]
        return KnowledgeItem(
            item_id=item.item_id,
            type=item.type.value,
            statement=item.statement,
            details=list(item.details),
            source_segment_ids=list(item.source_segment_ids),
            start_seconds=min(starts) if starts else None,
            end_seconds=max(ends) if ends else None,
            verification=status,
            uncertainties=list(item.uncertainties),
            cross_refs=[],
            provenance=[item.provenance] if item.provenance else [],
        )

    def _merge_records(self, merges: Sequence[MergeDecision]) -> list[MergeRecord]:
        records: list[MergeRecord] = []
        for decision in merges:
            if not decision.merged:
                continue
            records.append(
                MergeRecord(
                    result_item_id=decision.left_id,
                    source_item_ids=[decision.left_id, decision.right_id],
                    kind=str(decision.kind),
                    detail=decision.reason,
                )
            )
        return records

    def _build_topics(self, items: Sequence[KnowledgeItem]) -> list[KnowledgeTopic]:
        """Attach items to evidence-derived topics, then group by item type.

        Topic assignment is by cited segments: an item belongs to the topic whose
        defining concept it cites most. When the concept pipeline produced nothing, the
        type groups stand on their own instead of inventing a hierarchy.
        """
        concepts = [item for item in items if item.type == ItemType.CONCEPT.value]
        topics: list[KnowledgeTopic] = []
        concept_segments: list[set[str]] = []
        assigned: dict[str, str] = {}

        for index, concept in enumerate(concepts, start=1):
            topic_id = TopicId(f"topic_{index:03d}_{_slug(concept.statement)}")
            if topic_id in {t.topic_id for t in topics}:
                continue
            topic = KnowledgeTopic(
                topic_id=topic_id,
                title=topic_title(concept.statement),
                summary=concept.details[0] if concept.details else None,
                item_ids=[concept.item_id],
            )
            topics.append(topic)
            concept_segments.append(set(concept.source_segment_ids))

        for item in items:
            if item.type == ItemType.CONCEPT.value:
                continue
            item_segments = set(item.source_segment_ids)
            if not item_segments:
                continue
            scored = [
                (len(item_segments & segments), position)
                for position, segments in enumerate(concept_segments)
            ]
            overlap, position = max(scored, default=(0, -1))
            if overlap == 0:
                continue
            # Most of the item's evidence must sit inside the topic.
            if overlap / len(item_segments) < TOPIC_EVIDENCE_SHARE:
                continue
            concept_topic = topics[position]
            concept_topic.item_ids.append(item.item_id)
            assigned[item.item_id] = concept_topic.topic_id

        for pending in items:
            pending_topic_id = assigned.get(pending.item_id)
            if pending_topic_id is None:
                continue
            pending.topic_id = pending_topic_id

        by_type: dict[str, KnowledgeTopic] = {}
        for pending in items:
            if pending.topic_id is not None:
                continue
            label = TYPE_LABELS.get(_as_item_type(pending.type), pending.type)
            group_topic = by_type.get(label)
            if group_topic is None:
                group_topic = KnowledgeTopic(
                    topic_id=TopicId(f"topic_{len(topics) + 1:03d}_{_slug(label)}"),
                    title=label,
                )
                topics.append(group_topic)
                by_type[label] = group_topic
            group_topic.item_ids.append(pending.item_id)

        return topics


def _as_item_type(value: str) -> ItemType:
    try:
        return ItemType(value)
    except ValueError:
        return ItemType.CONCEPT


def topic_title(statement: str, limit: int = TOPIC_TITLE_LIMIT) -> str:
    """A readable topic title, never cut mid-word.

    A statement that already fits is used as-is. A longer one is shortened at its first
    sentence when that fits, otherwise at a word boundary: cutting mid-word produced
    headings like "...as a piece of mat".
    """
    text = " ".join(statement.split())
    if len(text) <= limit:
        return text

    sentence_end = text.find(". ")
    if 0 < sentence_end + 1 <= limit:
        return text[: sentence_end + 1]

    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{clipped}…" if clipped else text[:limit]


def _slug(text: str, limit: int = 40) -> str:
    cleaned = normalize_text(text)
    slug = "".join(char if char.isalnum() else "-" for char in cleaned)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:limit].strip("-") or "topic"


def empty_dataset(video_id: str, title: str | None = None) -> KnowledgeDataset:
    """A dataset that says plainly that nothing was extracted."""
    return KnowledgeDataset(
        video_id=video_id,
        title=title,
        topics=[],
        items=[],
        conflicts=[],
        completeness=CompletenessReport(
            completed_stages=[StageId.METADATA],
            skipped_stages=[
                StageId.RAW_TRANSCRIPT,
                StageId.NORMALIZED_TRANSCRIPT,
                StageId.CHUNKS,
                StageId.EXTRACTIONS,
                StageId.VERIFY,
            ],
        ),
        generated_at=utc_now(),
    )


def stage_report(
    *,
    completed: Sequence[StageId],
    failed: Sequence[StageId],
    verification: VerificationOutcome | None,
    unresolved_conflicts: Sequence[ConflictRecord] = (),
    generated_at: datetime | None = None,
) -> CompletenessReport:
    """Build a completeness report from stage state."""
    summary = verification.summary if verification else None
    return CompletenessReport(
        completed_stages=list(completed),
        failed_stages=list(failed),
        unverified_items=(summary.unverified + summary.needs_review) if summary else 0,
        unresolved_conflicts=sum(1 for conflict in unresolved_conflicts if not conflict.resolved),
    )


def pipeline_ids_used(items: Sequence[KnowledgeItem]) -> set[PipelineId]:
    used: set[PipelineId] = set()
    for item in items:
        for provenance in item.provenance:
            used.add(provenance.pipeline_id)
    return used


__all__ = [
    "TYPE_LABELS",
    "CompletenessReport",
    "DatasetBuilder",
    "DatasetInputs",
    "DatasetResult",
    "empty_dataset",
    "pipeline_ids_used",
    "stage_report",
]
