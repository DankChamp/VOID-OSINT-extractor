"""Aggregation, verification, conflicts and dataset assembly (§6, §20)."""

from __future__ import annotations

from video_intel.aggregation.conflicts import build_conflicts, find_candidates, unresolvable
from video_intel.aggregation.consolidate import (
    Consolidator,
    MergeKind,
    classify_pair,
    merge_items,
    normalize_text,
)
from video_intel.aggregation.dataset import DatasetBuilder, DatasetInputs
from video_intel.aggregation.verify import SegmentIndex, quote_is_in_segment, verify_items
from video_intel.schemas.common import (
    ItemType,
    PipelineId,
    StageId,
    VerificationStatus,
)
from video_intel.schemas.extraction import (
    ExtractedItem,
    ItemProvenance,
    ModelRef,
)
from video_intel.schemas.transcript import TranscriptSegment

MODEL = ModelRef(provider="ollama", provider_name="ollama", model="llama3.1:8b")

SEGMENTS = [
    TranscriptSegment(
        segment_id="segment_001",
        start_seconds=0.0,
        end_seconds=10.0,
        text=(
            "Risk assessment is the process of identifying, analyzing and evaluating risk "
            "against a set of objectives."
        ),
    ),
    TranscriptSegment(
        segment_id="segment_002",
        start_seconds=10.0,
        end_seconds=25.0,
        text="It identifies potential threats and evaluates their likelihood of occurring.",
    ),
    TranscriptSegment(
        segment_id="segment_003",
        start_seconds=25.0,
        end_seconds=40.0,
        text="It also evaluates the potential impact those threats would have if realised.",
    ),
    TranscriptSegment(
        segment_id="segment_004",
        start_seconds=40.0,
        end_seconds=55.0,
        text="Teams that documented assets recovered forty percent faster after an incident.",
    ),
    TranscriptSegment(
        segment_id="segment_005",
        start_seconds=55.0,
        end_seconds=70.0,
        text="The four treatment options are mitigation, transfer, acceptance and avoidance.",
    ),
]


def item(
    item_id: str,
    statement: str,
    segments: list[str],
    *,
    item_type: ItemType = ItemType.FACT,
    details: list[str] | None = None,
    uncertainties: list[str] | None = None,
    pipeline: PipelineId = PipelineId.FACT,
    start: float = 0.0,
    end: float = 10.0,
) -> ExtractedItem:
    return ExtractedItem(
        item_id=item_id,
        type=item_type,
        statement=statement,
        details=details or [],
        source_segment_ids=segments,
        uncertainties=uncertainties or [],
        provenance=ItemProvenance(
            pipeline_id=pipeline,
            model=MODEL,
            source_segment_ids=segments,
            start_seconds=start,
            end_seconds=end,
        ),
    )


# ------------------------------------------------------------- text utilities
def test_normalize_text_is_comparison_friendly() -> None:
    assert normalize_text("Risk,  Assessment!") == "risk assessment"
    assert normalize_text("don\u2019t") == "don't"


# ---------------------------------------------------------------- classification
def test_identical_items_are_exact_duplicates() -> None:
    a = item("item_a", "Risk is likelihood times impact.", ["segment_002"])
    b = item("item_b", "risk is likelihood times impact", ["segment_002"])
    decision = classify_pair(a, b)
    assert decision.kind is MergeKind.EXACT_DUPLICATE
    assert decision.merged


def test_reworded_same_claim_is_a_semantic_duplicate() -> None:
    a = item("item_a", "Risk assessment evaluates the likelihood of threats.", ["segment_002"])
    b = item(
        "item_b",
        "The risk assessment evaluates how likely the threats are.",
        ["segment_002"],
    )
    decision = classify_pair(a, b)
    assert decision.kind is MergeKind.SEMANTIC_DUPLICATE


def test_complementary_parts_are_merged_not_duplicated() -> None:
    a = item(
        "item_a",
        "Risk assessment evaluates the likelihood of a threat.",
        ["segment_002"],
        details=["Likelihood is rated low, medium or high."],
    )
    b = item(
        "item_b",
        "Risk assessment evaluates the potential impact of a threat.",
        ["segment_003"],
        details=["Impact is rated on the same scale."],
    )
    decision = classify_pair(a, b)
    assert decision.kind is MergeKind.COMPLEMENTARY
    assert decision.merged


def test_same_topic_but_different_content_is_not_merged() -> None:
    a = item("item_a", "Risk assessment is discussed in this video.", ["segment_001"])
    b = item("item_b", "Coffee extraction improves espresso flavour.", ["segment_004"])
    decision = classify_pair(a, b)
    assert decision.kind is None
    assert not decision.merged


def test_conflicting_figures_are_flagged_as_contradiction() -> None:
    a = item(
        "item_a", "Teams recovered forty percent faster after documenting assets.", ["segment_004"]
    )
    b = item(
        "item_b",
        "Teams recovered sixty percent faster after documenting assets.",
        ["segment_004"],
    )
    decision = classify_pair(a, b)
    assert decision.kind is MergeKind.CONTRADICTION
    assert not decision.merged


def test_merge_keeps_details_segments_and_is_reversible() -> None:
    a = item("item_a", "Likelihood is evaluated.", ["segment_002"], details=["Rated low to high."])
    b = item(
        "item_b", "Impact is evaluated.", ["segment_003"], details=["Rated on the same scale."]
    )
    decision = classify_pair(a, b)
    merged = merge_items(a, b, decision)
    assert "Impact is evaluated." in merged.details or "Rated on the same scale." in merged.details
    assert merged.source_segment_ids == ["segment_002", "segment_003"]


# -------------------------------------------------------------- consolidator
def test_consolidator_keeps_a_unique_single_model_item() -> None:
    """§20.1: a fact only one model found survives consolidation."""
    items = [
        item("item_common", "Risk is the combination of likelihood and impact.", ["segment_001"]),
        item("item_common_2", "Risk is the combination of likelihood and impact.", ["segment_001"]),
        item(
            "item_unique",
            "Teams that documented assets recovered forty percent faster.",
            ["segment_004"],
        ),
    ]
    result = Consolidator().run(items)
    ids = {entry.item_id for entry in result.items}
    assert "item_unique" in ids
    assert len(result.items) == 2


def test_consolidator_records_every_merge() -> None:
    items = [
        item("item_a", "Risk is the combination of likelihood and impact.", ["segment_001"]),
        item("item_b", "Risk is the combination of likelihood and impact.", ["segment_001"]),
    ]
    result = Consolidator().run(items)
    assert len(result.items) == 1
    assert result.merged_pairs
    assert result.merged_pairs[0].left_id != result.merged_pairs[0].right_id


def test_consolidator_never_merges_on_topic_alone() -> None:
    items = [
        item("item_a", "Risk assessment is discussed in this video.", ["segment_001"]),
        item("item_b", "Coffee extraction improves espresso flavour.", ["segment_004"]),
    ]
    assert len(Consolidator().run(items).items) == 2


def test_consolidator_preserves_repeated_details() -> None:
    items = [
        item(
            "item_a",
            "Risk assessment evaluates likelihood.",
            ["segment_002"],
            details=["Likelihood is rated low, medium or high."],
        ),
        item(
            "item_b",
            "Risk assessment evaluates impact.",
            ["segment_003"],
            details=["Likelihood is rated low, medium or high."],
        ),
    ]
    merged = Consolidator().run(items).items
    assert len(merged) == 1
    assert merged[0].details.count("Likelihood is rated low, medium or high.") == 1
    assert len(merged[0].source_segment_ids) == 2


# ---------------------------------------------------------------- verification
def test_supported_item_is_verified_as_supported() -> None:
    verified = verify_items(
        [item("item_a", "It identifies potential threats.", ["segment_002"])],
        SegmentIndex.of(SEGMENTS),
    )
    result = verified.items[0]
    assert result.verification.status is VerificationStatus.SUPPORTED
    assert result.verification.evidence[0].segment_id == "segment_002"
    assert result.retained is True


def test_unsupported_claim_is_flagged_and_excluded_from_outputs() -> None:
    """§20.2: a model-invented claim never enters the outputs."""
    invented = item(
        "item_invented",
        "Risk assessment reduces security incidents by ninety percent across every industry.",
        ["segment_002"],
    )
    outcome = verify_items([invented], SegmentIndex.of(SEGMENTS))
    result = outcome.items[0]
    assert result.verification.status is VerificationStatus.UNSUPPORTED
    assert result.retained is False
    assert "not support" in (result.review_reason or "")
    assert outcome.summary.unsupported == 1


def test_partially_supported_item_keeps_its_supported_parts() -> None:
    partial = item(
        "item_partial",
        "It identifies potential threats and improves quarterly revenue by twelve percent.",
        ["segment_002"],
    )
    outcome = verify_items([partial], SegmentIndex.of(SEGMENTS))
    result = outcome.items[0]
    assert result.verification.status in {
        VerificationStatus.PARTIALLY_SUPPORTED,
        VerificationStatus.UNSUPPORTED,
    }
    assert result.verification.evidence


def test_missing_segment_reference_is_unsupported() -> None:
    outcome = verify_items(
        [item("item_x", "A claim citing nothing real.", ["segment_999"])],
        SegmentIndex.of(SEGMENTS),
    )
    assert outcome.items[0].verification.status is VerificationStatus.UNSUPPORTED
    assert "no cited segment" in (outcome.items[0].verification.rationale or "")


def test_context_window_can_rescue_a_split_statement() -> None:
    split = item(
        "item_split",
        "Forty percent faster recovery after documenting assets.",
        ["segment_004"],
        details=["The study was run last year."],
    )
    narrow = verify_items([split], SegmentIndex.of(SEGMENTS), context_window=0)
    wide = verify_items([split], SegmentIndex.of(SEGMENTS), context_window=1)
    assert narrow.summary.total == wide.summary.total == 1


def test_verification_summary_counts_add_up() -> None:
    items = [
        item("item_ok", "It identifies potential threats.", ["segment_002"]),
        item("item_bad", "Totally unrelated statement about quantum tunnelling.", ["segment_003"]),
    ]
    summary = verify_items(items, SegmentIndex.of(SEGMENTS)).summary
    assert summary.total == 2
    assert summary.supported + summary.unsupported == 2


def test_evidence_quote_is_drawn_from_the_cited_segment() -> None:
    outcome = verify_items(
        [item("item_ok", "It identifies potential threats.", ["segment_002"])],
        SegmentIndex.of(SEGMENTS),
    )
    evidence = outcome.items[0].verification.evidence[0]
    segment = next(s for s in SEGMENTS if s.segment_id == "segment_002")
    assert quote_is_in_segment(evidence.quote, segment)


def test_segment_index_context_is_ordered_and_bounded() -> None:
    index = SegmentIndex.of(SEGMENTS)
    context = index.context(["segment_003"], window=1)
    assert [s.segment_id for s in context] == ["segment_002", "segment_003", "segment_004"]
    assert [s.segment_id for s in index.context(["segment_999"], window=2)] == []


# ------------------------------------------------------------------- conflicts
def test_contradictory_claims_produce_a_conflict_record() -> None:
    items = [
        item("item_a", "Teams recovered forty percent faster after incidents.", ["segment_004"]),
        item("item_b", "Teams recovered seventy percent faster after incidents.", ["segment_004"]),
    ]
    conflicts = build_conflicts(items, SEGMENTS)
    assert conflicts
    assert conflicts[0].status is VerificationStatus.SUPPORTED
    assert conflicts[0].resolution and "40" in conflicts[0].resolution


def test_unresolvable_contradiction_stays_open() -> None:
    """§20.5: when the transcript cannot settle it, the conflict stays open."""
    extra = [
        *SEGMENTS,
        TranscriptSegment(
            segment_id="segment_006",
            start_seconds=70.0,
            end_seconds=80.0,
            text="Recovery speed depends entirely on the organisation and cannot be compared.",
        ),
    ]
    items = [
        item(
            "item_a",
            "Mitigation always prevents every security incident entirely.",
            ["segment_005"],
        ),
        item(
            "item_b",
            "Mitigation never prevents any security incident at all.",
            ["segment_005"],
        ),
    ]
    conflicts = build_conflicts(items, extra)
    assert conflicts
    assert conflicts[0].status is VerificationStatus.NEEDS_REVIEW
    assert conflicts[0].resolution is None
    assert unresolvable(conflicts) == conflicts


def test_no_conflicts_for_consistent_items() -> None:
    items = [
        item("item_a", "It identifies potential threats.", ["segment_002"]),
        item("item_b", "It evaluates their likelihood.", ["segment_002"]),
    ]
    assert find_candidates(items) == []
    assert build_conflicts(items, SEGMENTS) == []


# ---------------------------------------------------------------- dataset build
def test_dataset_contains_only_supported_items_by_default() -> None:
    result = DatasetBuilder().build(
        DatasetInputs(
            video_id="synthetic01",
            title="Risk assessment",
            segments=SEGMENTS,
            items=[
                item("item_ok", "It identifies potential threats.", ["segment_002"]),
                item(
                    "item_bad",
                    "Quantum tunnelling makes risk assessment optional.",
                    ["segment_002"],
                ),
            ],
            completed_stages=[StageId.EXTRACTIONS, StageId.VERIFY],
        )
    )
    ids = {entry.item_id for entry in result.dataset.items}
    assert ids == {"item_ok"}
    assert [d.item_id for d in result.dropped] == ["item_bad"]


def test_dataset_retains_unsupported_items_when_configured() -> None:
    result = DatasetBuilder().build(
        DatasetInputs(
            video_id="synthetic01",
            title=None,
            segments=SEGMENTS,
            items=[
                item("item_ok", "It identifies potential threats.", ["segment_002"]),
                item("item_bad", "Quantum tunnelling makes risk optional.", ["segment_002"]),
            ],
            drop_unsupported=False,
        )
    )
    assert len(result.dataset.items) == 2
    unsupported = [i for i in result.dataset.items if i.item_id == "item_bad"]
    assert unsupported[0].verification is VerificationStatus.UNSUPPORTED


def test_multiple_timestamps_survive_consolidation() -> None:
    """§20.4: the specification's example merges while keeping every timestamp."""
    items = [
        item(
            "item_a",
            "Risk assessment identifies potential threats.",
            ["segment_002"],
            start=10.0,
            end=25.0,
        ),
        item(
            "item_b",
            "Risk assessment evaluates the likelihood of those threats.",
            ["segment_003"],
            start=25.0,
            end=40.0,
        ),
        item(
            "item_c",
            "Risk assessment evaluates the potential impact of those threats.",
            ["segment_004"],
            start=40.0,
            end=55.0,
        ),
    ]
    result = DatasetBuilder().build(
        DatasetInputs(
            video_id="synthetic01",
            title=None,
            segments=SEGMENTS,
            items=items,
        )
    )
    assert len(result.dataset.items) == 1
    consolidated = result.dataset.items[0]
    assert consolidated.source_segment_ids == ["segment_002", "segment_003", "segment_004"]
    assert consolidated.start_seconds == 10.0
    assert consolidated.end_seconds == 55.0
    assert len(consolidated.details) == 2  # the two absorbed statements


def test_dataset_hierarchy_is_derived_from_the_transcript() -> None:
    items = [
        item(
            "item_concept",
            "Risk assessment",
            ["segment_001"],
            item_type=ItemType.CONCEPT,
            details=["The speaker's framing of the topic."],
        ),
        item(
            "item_def",
            "Risk is likelihood and impact.",
            ["segment_001", "segment_002"],
            item_type=ItemType.DEFINITION,
        ),
        item("item_fact", "Documented teams recovered forty percent faster.", ["segment_004"]),
    ]
    dataset = (
        DatasetBuilder()
        .build(DatasetInputs(video_id="synthetic01", title=None, segments=SEGMENTS, items=items))
        .dataset
    )

    titles = [topic.title for topic in dataset.topics]
    assert "Risk assessment" in titles
    concept_topic = next(t for t in dataset.topics if t.title == "Risk assessment")
    assert "item_def" in concept_topic.item_ids
    type_titles = {topic.title for topic in dataset.topics}
    assert "Facts and figures" in type_titles


def test_topic_titles_are_cut_at_a_word_boundary() -> None:
    from video_intel.aggregation.dataset import topic_title

    long = (
        "Neural networks and machine learning are relevant to the present and future, "
        "and the speaker aims to show what a neural network is as a piece of math."
    )
    title = topic_title(long)
    assert len(title) <= 121
    assert title.endswith("…")
    assert "mat…" not in title  # never cut mid-word
    assert topic_title("Short one.") == "Short one."
    # Short statements keep their full wording, even with two sentences.
    assert topic_title("First sentence here. Second sentence here.") == (
        "First sentence here. Second sentence here."
    )
    # A long statement is cut at its first sentence when that one is short enough.
    long_two_sentences = "Neural networks learn from data. " + "Extra detail " * 30
    assert topic_title(long_two_sentences) == "Neural networks learn from data."


def test_dataset_without_concepts_falls_back_to_type_groups() -> None:
    dataset = (
        DatasetBuilder()
        .build(
            DatasetInputs(
                video_id="synthetic01",
                title=None,
                segments=SEGMENTS,
                items=[item("item_fact", "It identifies potential threats.", ["segment_002"])],
            )
        )
        .dataset
    )
    assert {topic.title for topic in dataset.topics} == {"Facts and figures"}


def test_completeness_report_records_incompleteness() -> None:
    result = DatasetBuilder().build(
        DatasetInputs(
            video_id="synthetic01",
            title=None,
            segments=SEGMENTS,
            items=[item("item_ok", "It identifies potential threats.", ["segment_002"])],
            completed_stages=[StageId.EXTRACTIONS],
            failed_stages=[StageId.VERIFY],
        )
    )
    report = result.dataset.completeness
    assert report.complete is False
    assert StageId.VERIFY in report.failed_stages


def test_merge_history_is_recorded_in_the_dataset() -> None:
    items = [
        item("item_a", "Risk is likelihood and impact combined.", ["segment_001"]),
        item("item_b", "Risk is likelihood and impact combined.", ["segment_001"]),
    ]
    dataset = (
        DatasetBuilder()
        .build(DatasetInputs(video_id="synthetic01", title=None, segments=SEGMENTS, items=items))
        .dataset
    )
    assert dataset.merges
    assert set(dataset.merges[0].source_item_ids) == {"item_a", "item_b"}
    assert dataset.merges[0].result_item_id == "item_a"


def test_dataset_validates_its_own_references() -> None:
    dataset = (
        DatasetBuilder()
        .build(
            DatasetInputs(
                video_id="synthetic01",
                title=None,
                segments=SEGMENTS,
                items=[
                    item(
                        "item_concept",
                        "Threat identification",
                        ["segment_002"],
                        item_type=ItemType.CONCEPT,
                    ),
                    item("item_fact", "It identifies potential threats.", ["segment_002"]),
                ],
            )
        )
        .dataset
    )
    dataset.model_validate(dataset.model_dump(mode="json"))
    assert dataset.item("item_fact") is not None


def test_extraction_metadata_is_retained() -> None:
    dataset = (
        DatasetBuilder()
        .build(
            DatasetInputs(
                video_id="synthetic01",
                title=None,
                segments=SEGMENTS,
                items=[
                    item(
                        "item_def",
                        "It identifies potential threats and evaluates their likelihood.",
                        ["segment_002"],
                        pipeline=PipelineId.DEFINITION,
                    )
                ],
            )
        )
        .dataset
    )
    item_record = dataset.items[0]
    assert item_record.provenance
    assert item_record.provenance[0].pipeline_id is PipelineId.DEFINITION
    assert item_record.provenance[0].model.model == "llama3.1:8b"
