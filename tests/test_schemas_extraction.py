"""Extraction contract, provenance, verification, conflict and dataset schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_intel.schemas.common import (
    ItemType,
    PipelineId,
    ProviderKind,
    StageState,
    VerificationStatus,
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
from video_intel.schemas.verification import (
    Evidence,
    VerificationResult,
    VerificationSummary,
    VerifiedItem,
)


# --------------------------------------------------------------------- extraction
def test_fixture_extraction_response_validates(extraction_response: ExtractionResponse) -> None:
    assert extraction_response.pipeline_id is PipelineId.FACT
    assert [item.item_id for item in extraction_response.items] == [
        "item_001",
        "item_002",
        "item_003",
        "item_004",
    ]


def test_contract_forbids_unknown_fields(extraction_payload: dict) -> None:
    extraction_payload["unexpected"] = "should be rejected"
    with pytest.raises(ValidationError, match="unexpected"):
        ExtractionResponse.model_validate(extraction_payload)


def test_contract_rejects_item_without_source_segments(extraction_payload: dict) -> None:
    extraction_payload["items"][0]["source_segment_ids"] = []
    with pytest.raises(ValidationError):
        ExtractionResponse.model_validate(extraction_payload)


def test_contract_rejects_duplicate_item_ids(extraction_payload: dict) -> None:
    extraction_payload["items"][1]["item_id"] = "item_001"
    with pytest.raises(ValidationError, match="duplicate item_id"):
        ExtractionResponse.model_validate(extraction_payload)


def test_contract_keeps_repeated_explanation_details(extraction_payload: dict) -> None:
    response = ExtractionResponse.model_validate(extraction_payload)
    definition = response.items[0]
    assert definition.source_segment_ids == ["segment_002", "segment_010"]
    assert definition.details == ["Risk is the combination of likelihood and impact."]


def test_uncertainty_survives_on_the_item(extraction_response: ExtractionResponse) -> None:
    unsupported = extraction_response.items[3]
    assert unsupported.uncertainties
    assert "ninety percent" in unsupported.statement


def test_item_type_vocabulary() -> None:
    assert "missed" in {item.value for item in ItemType}


def test_provenance_interval_is_validated() -> None:
    model = ModelRef(provider=ProviderKind.OLLAMA, provider_name="ollama", model="llama3.1:8b")
    with pytest.raises(ValidationError, match="precedes start_seconds"):
        ItemProvenance(
            pipeline_id=PipelineId.CONCEPT,
            model=model,
            start_seconds=90.0,
            end_seconds=10.0,
        )


def test_provenance_records_pipeline_model_and_segments() -> None:
    model = ModelRef(provider=ProviderKind.OLLAMA, provider_name="ollama", model="llama3.1:8b")
    provenance = ItemProvenance(
        pipeline_id=PipelineId.CONCEPT,
        model=model,
        chunk_ids=["chunk_0001"],
        source_segment_ids=["segment_001", "segment_002"],
        start_seconds=18.5,
        end_seconds=44.0,
    )
    assert provenance.pipeline_id is PipelineId.CONCEPT
    assert provenance.model.model == "llama3.1:8b"
    assert provenance.source_segment_ids == ["segment_001", "segment_002"]


def test_raw_extraction_record_keeps_response_verbatim(
    extraction_response: ExtractionResponse,
) -> None:
    raw_text = extraction_response.model_dump_json()
    record = RawExtractionRecord(
        record_id="fact_0001",
        video_id="synthetic001",
        pipeline_id=PipelineId.FACT,
        model=ModelRef(provider=ProviderKind.OLLAMA, provider_name="ollama", model="llama3.1:8b"),
        raw_response=raw_text,
        parsed=extraction_response,
        state=StageState.COMPLETED,
        attempts=1,
    )
    assert record.raw_response == raw_text
    assert record.valid is True


def test_invalid_model_response_is_not_treated_as_parsed() -> None:
    record = RawExtractionRecord(
        record_id="fact_0002",
        video_id="synthetic001",
        pipeline_id=PipelineId.FACT,
        model=ModelRef(provider=ProviderKind.OLLAMA, provider_name="ollama", model="llama3.1:8b"),
        raw_response="not json at all",
        attempts=2,
        error="json decode error",
    )
    assert record.parsed is None
    assert record.valid is False


# ------------------------------------------------------------------ verification
def test_verified_item_requires_matching_ids() -> None:
    item = ExtractedItem(
        item_id="item_001",
        type=ItemType.FACT,
        statement="A supported statement.",
        source_segment_ids=["segment_005"],
    )
    result = VerificationResult(
        item_id="item_999",
        status=VerificationStatus.SUPPORTED,
        verified_by="deterministic",
    )
    with pytest.raises(ValidationError, match="does not match"):
        VerifiedItem(item=item, verification=result)


def test_evidence_requires_consistent_interval() -> None:
    with pytest.raises(ValidationError, match="precedes start_seconds"):
        Evidence(segment_id="segment_001", quote="text", start_seconds=20.0, end_seconds=10.0)


def test_verification_states_are_distinct() -> None:
    supported = VerificationResult(
        item_id="item_001", status=VerificationStatus.SUPPORTED, verified_by="deterministic"
    )
    partial = VerificationResult(
        item_id="item_001", status=VerificationStatus.PARTIALLY_SUPPORTED, verified_by="model"
    )
    unsupported = VerificationResult(
        item_id="item_001", status=VerificationStatus.UNSUPPORTED, verified_by="model"
    )
    assert supported.is_reliable and not supported.needs_review
    assert partial.needs_review and not unsupported.is_reliable


def test_summary_reports_verified_fraction() -> None:
    summary = VerificationSummary(total=4, supported=2, partially_supported=1, unsupported=1)
    assert summary.verified_fraction == pytest.approx(0.75)
    assert summary.complete is True


def test_summary_is_incomplete_when_items_are_unverified() -> None:
    summary = VerificationSummary(total=3, supported=1, unverified=2)
    assert summary.complete is False


# --------------------------------------------------------------------- conflicts
def test_conflict_defaults_to_unresolved() -> None:
    conflict = ConflictRecord(
        conflict_id="conflict_001",
        claim_ids=["item_012", "item_047"],
        description="Two models disagree about the treatment options.",
    )
    assert conflict.status is VerificationStatus.NEEDS_REVIEW
    assert conflict.resolution is None
    assert conflict.resolved is False


def test_conflict_requires_distinct_claims() -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        ConflictRecord(
            conflict_id="conflict_001",
            claim_ids=["item_012", "item_012"],
            description="Same claim twice.",
        )


def test_conflict_set_separates_unresolved() -> None:
    resolved = ConflictRecord(
        conflict_id="conflict_001",
        claim_ids=["item_012", "item_047"],
        description="Resolved by transcript evidence.",
        status=VerificationStatus.SUPPORTED,
        resolution="The transcript states four options explicitly.",
    )
    open_conflict = ConflictRecord(
        conflict_id="conflict_002",
        claim_ids=["item_050", "item_051"],
        description="Transcript is unclear.",
    )
    conflicts = ConflictSet(video_id="synthetic001", conflicts=[resolved, open_conflict])
    assert [c.conflict_id for c in conflicts.unresolved] == ["conflict_002"]


# ------------------------------------------------------------- knowledge dataset
def _supported_item(item_id: str = "item_001", **overrides: object) -> KnowledgeItem:
    payload: dict[str, object] = {
        "item_id": item_id,
        "type": "definition",
        "statement": "Risk assessment identifies and evaluates risk.",
        "details": ["Evaluates likelihood."],
        "source_segment_ids": ["segment_002", "segment_010"],
        "start_seconds": 18.5,
        "end_seconds": 243.25,
        "verification": VerificationStatus.SUPPORTED,
    }
    payload.update(overrides)
    return KnowledgeItem.model_validate(payload)


def test_dataset_retains_evidence_from_a_single_model() -> None:
    unique = _supported_item("item_unique")
    dataset = KnowledgeDataset(video_id="synthetic001", items=[unique])
    assert dataset.item("item_unique") is not None
    assert dataset.item("item_unique").source_segment_ids == ["segment_002", "segment_010"]  # type: ignore[union-attr]


def test_consolidation_keeps_every_timestamp_and_detail() -> None:
    merged = _supported_item(
        "item_001",
        details=["Evaluates likelihood.", "Evaluates impact.", "Identifies threats."],
        source_segment_ids=["segment_002", "segment_007", "segment_008", "segment_010"],
        start_seconds=18.5,
        end_seconds=243.25,
        merged_from=["item_001", "item_002", "item_003"],
    )
    assert len(merged.details) == 3
    assert len(merged.source_segment_ids) == 4
    assert merged.merged_from == ["item_001", "item_002", "item_003"]


def test_merge_record_is_reversible() -> None:
    record = MergeRecord(
        result_item_id="item_001",
        source_item_ids=["item_002", "item_003"],
        kind="complementary",
        detail="Both items describe components of risk assessment.",
        evidence_segment_ids=["segment_002", "segment_007"],
    )
    assert record.reversible is True


def test_dataset_rejects_unknown_topic_reference() -> None:
    with pytest.raises(ValidationError, match="unknown topic_id"):
        KnowledgeDataset(
            video_id="v1",
            items=[_supported_item(topic_id="topic_missing")],
        )


def test_dataset_rejects_topic_with_unknown_item() -> None:
    with pytest.raises(ValidationError, match="unknown item_ids"):
        KnowledgeDataset(
            video_id="v1",
            topics=[KnowledgeTopic(topic_id="topic_root", title="Root", item_ids=["item_x"])],
        )


def test_dataset_rejects_duplicate_items() -> None:
    with pytest.raises(ValidationError, match="duplicate item_id"):
        KnowledgeDataset(video_id="v1", items=[_supported_item(), _supported_item()])


def test_dataset_hierarchy_resolves() -> None:
    dataset = KnowledgeDataset(
        video_id="v1",
        topics=[
            KnowledgeTopic(topic_id="topic_root", title="Risk Management"),
            KnowledgeTopic(
                topic_id="topic_assessment",
                title="Risk Assessment",
                parent_id="topic_root",
                item_ids=["item_001"],
            ),
        ],
        items=[_supported_item(topic_id="topic_assessment")],
    )
    assert [t.topic_id for t in dataset.root_topics()] == ["topic_root"]
    assert dataset.topics[1].parent_id == "topic_root"


def test_unsupported_items_are_not_trusted() -> None:
    dataset = KnowledgeDataset(
        video_id="v1",
        items=[
            _supported_item("item_ok"),
            _supported_item("item_bad", verification=VerificationStatus.UNSUPPORTED),
        ],
    )
    assert [item.item_id for item in dataset.trusted_items()] == ["item_ok"]
    assert len(dataset.items_by_verification()["unsupported"]) == 1


def test_completeness_report_flags_incomplete_datasets() -> None:
    dataset = KnowledgeDataset(
        video_id="v1",
        items=[_supported_item(verification=VerificationStatus.NEEDS_REVIEW)],
        completeness=CompletenessReport(
            completed_stages=["chunks"],
            failed_stages=["verify"],
            unverified_items=1,
        ),
    )
    assert dataset.completeness.complete is False
    assert "verify" in dataset.completeness.failed_stages


def test_extraction_run_records_pipeline_outcome() -> None:
    run = ExtractionRun(
        pipeline_id=PipelineId.COVERAGE_AUDIT,
        model=ModelRef(provider=ProviderKind.OLLAMA, provider_name="ollama", model="llama3.1:8b"),
        items_extracted=3,
        items_retained=3,
        chunks_processed=2,
    )
    assert run.pipeline_id is PipelineId.COVERAGE_AUDIT
    assert run.items_extracted == run.items_retained == 3
