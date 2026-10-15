"""Knowledge dataset models (§11, §13).

The dataset is the single source for every later output. It records how items were
consolidated so deduplication stays explainable and reversible, and it carries
completeness flags so a partial run is never presented as a finished one.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ItemId,
    PipelineId,
    SegmentId,
    StageId,
    StageState,
    StrictModel,
    TopicId,
    VerificationStatus,
    utc_now,
)
from video_intel.schemas.conflict import ConflictRecord
from video_intel.schemas.extraction import ItemProvenance, ModelRef


class KnowledgeItem(StrictModel):
    """A consolidated, deduplicated item with merged evidence."""

    item_id: ItemId
    type: str = Field(description="Item type, or a dataset-level category label.")
    statement: str = Field(min_length=3)
    details: list[str] = Field(default_factory=list)
    topic_id: TopicId | None = None
    source_segment_ids: list[SegmentId] = Field(min_length=1)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    verification: VerificationStatus = VerificationStatus.NEEDS_REVIEW
    uncertainties: list[str] = Field(default_factory=list)
    cross_refs: list[ItemId] = Field(default_factory=list)
    provenance: list[ItemProvenance] = Field(default_factory=list)
    merged_from: list[ItemId] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join([self.statement, *self.details]).strip()

    @property
    def is_trusted(self) -> bool:
        return self.verification in {
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
        }


class KnowledgeTopic(StrictModel):
    """A node in the topic hierarchy. The actual structure comes from the video."""

    topic_id: TopicId
    title: str = Field(min_length=1)
    summary: str | None = None
    parent_id: TopicId | None = None
    child_ids: list[TopicId] = Field(default_factory=list)
    item_ids: list[ItemId] = Field(default_factory=list)
    cross_refs: list[TopicId] = Field(default_factory=list)


class MergeRecord(StrictModel):
    """Explainable, reversible consolidation history (§11)."""

    result_item_id: ItemId
    source_item_ids: list[ItemId] = Field(min_length=2)
    kind: str = Field(description="'exact_duplicate', 'semantic_duplicate' or 'complementary'.")
    detail: str
    evidence_segment_ids: list[SegmentId] = Field(default_factory=list)

    @property
    def reversible(self) -> bool:
        return bool(self.source_item_ids)


class ExtractionRun(StrictModel):
    """Per-pipeline execution metadata retained in the dataset (§13)."""

    pipeline_id: PipelineId
    model: ModelRef
    stage_state: StageState = StageState.COMPLETED
    items_extracted: int = Field(default=0, ge=0)
    items_retained: int = Field(default=0, ge=0)
    chunks_processed: int = Field(default=0, ge=0)
    error: str | None = None


class CompletenessReport(StrictModel):
    """What was and was not done for this dataset (§19)."""

    completed_stages: list[StageId] = Field(default_factory=list)
    failed_stages: list[StageId] = Field(default_factory=list)
    skipped_stages: list[StageId] = Field(default_factory=list)
    unverified_items: int = Field(default=0, ge=0)
    unresolved_conflicts: int = Field(default=0, ge=0)
    missing_items_from_coverage_audit: int = Field(default=0, ge=0)

    @property
    def complete(self) -> bool:
        return not self.failed_stages and not self.unverified_items


class KnowledgeDataset(StrictModel):
    """The consolidated, hierarchical knowledge dataset (§13)."""

    schema_version: str = SCHEMA_VERSION
    dataset_version: int = Field(default=1, ge=1)
    video_id: str
    title: str | None = None
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    items: list[KnowledgeItem] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    merges: list[MergeRecord] = Field(default_factory=list)
    extractions: list[ExtractionRun] = Field(default_factory=list)
    transcript_hash: str | None = None
    config_hash: str | None = None
    completeness: CompletenessReport = Field(default_factory=CompletenessReport)
    generated_at: datetime = Field(default_factory=utc_now)

    @field_validator("items")
    @classmethod
    def _unique_items(cls, items: list[KnowledgeItem]) -> list[KnowledgeItem]:
        ids = [item.item_id for item in items]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate item_id values in dataset: {duplicates}")
        return items

    @field_validator("topics")
    @classmethod
    def _unique_topics(cls, topics: list[KnowledgeTopic]) -> list[KnowledgeTopic]:
        ids = [topic.topic_id for topic in topics]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate topic_id values in dataset: {duplicates}")
        return topics

    @model_validator(mode="after")
    def _references_resolve(self) -> KnowledgeDataset:
        topic_ids = {topic.topic_id for topic in self.topics}
        item_ids = {item.item_id for item in self.items}
        for topic in self.topics:
            if topic.parent_id is not None and topic.parent_id not in topic_ids:
                raise ValueError(f"topic {topic.topic_id}: unknown parent_id {topic.parent_id}")
            unknown = [i for i in topic.item_ids if i not in item_ids]
            if unknown:
                raise ValueError(f"topic {topic.topic_id}: unknown item_ids {unknown}")
        for item in self.items:
            if item.topic_id is not None and item.topic_id not in topic_ids:
                raise ValueError(f"item {item.item_id}: unknown topic_id {item.topic_id}")
        return self

    def item(self, item_id: ItemId) -> KnowledgeItem | None:
        return next((item for item in self.items if item.item_id == item_id), None)

    def root_topics(self) -> list[KnowledgeTopic]:
        return [topic for topic in self.topics if topic.parent_id is None]

    def trusted_items(self) -> list[KnowledgeItem]:
        return [item for item in self.items if item.is_trusted]

    def items_by_verification(self) -> dict[str, list[KnowledgeItem]]:
        grouped: dict[str, list[KnowledgeItem]] = {}
        for item in self.items:
            grouped.setdefault(str(item.verification), []).append(item)
        return grouped
