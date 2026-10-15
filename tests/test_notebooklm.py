"""NotebookLM handoff (§8 of Phase 8): export packets, import answers, merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_intel.aggregation.dataset import DatasetBuilder, DatasetInputs
from video_intel.notebooklm.handoff import (
    MANUAL_WORKFLOW_NOTICE,
    ImportValidationError,
    build_prompt_document,
    build_source_packet,
    contract_document,
    export_packet,
    import_records,
    load_imported_records,
    parse_import,
    store_import,
)
from video_intel.schemas.common import ItemType, PipelineId, VerificationStatus
from video_intel.schemas.extraction import ExtractedItem
from video_intel.schemas.knowledge import KnowledgeDataset
from video_intel.schemas.transcript import NormalizedTranscript, hash_transcript


@pytest.fixture
def normalized() -> NormalizedTranscript:
    segments = [
        {
            "segment_id": "segment_0001",
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "text": "Risk assessment is the process of identifying and evaluating risk.",
        },
        {
            "segment_id": "segment_0002",
            "start_seconds": 8.0,
            "end_seconds": 16.0,
            "text": "It identifies potential threats and evaluates their likelihood.",
        },
    ]
    return NormalizedTranscript(
        video_id="synthetic01",
        source_transcript_hash=hash_transcript("a b"),
        segments=segments,
    )


@pytest.fixture
def dataset() -> KnowledgeDataset:
    return KnowledgeDataset(
        video_id="synthetic01",
        title="Risk assessment explained",
        items=[],
        topics=[],
    )


def answer(items: list[dict]) -> dict:
    return {"pipeline_id": "concept", "items": items}


def item(statement: str, segments: list[str], item_type: str = "concept") -> dict:
    return {
        "type": item_type,
        "statement": statement,
        "details": [],
        "source_segment_ids": segments,
        "uncertainties": [],
    }


# --------------------------------------------------------------------- export
def test_source_packet_keeps_ids_and_timestamps(dataset, normalized) -> None:
    packet = build_source_packet(dataset, normalized)
    assert "Risk assessment explained" in packet
    assert "[segment_0001 00:00:00.000-00:00:08.000]" in packet
    assert "[segment_0002" in packet
    assert "evaluating risk." in packet


def test_source_packet_can_be_truncated(dataset, normalized) -> None:
    packet = build_source_packet(dataset, normalized, max_chars=80)
    assert "packet truncated" in packet
    assert "segment_0002" not in packet


def test_prompt_document_states_the_contract_and_the_manual_workflow(dataset, normalized) -> None:
    document = build_prompt_document(dataset)
    assert "manual handoff" in document
    assert "does not assume" in document
    assert "source_segment_ids" in document
    assert "Task: concept" in document
    assert "Task: fact" in document


def test_contract_document_is_valid_json(dataset, normalized) -> None:
    schema = json.loads(contract_document())
    assert schema["title"] == "video-intel extraction response"
    assert "items" in schema["properties"]


def test_export_writes_all_four_files(tmp_path: Path, dataset, normalized) -> None:
    packet = export_packet(dataset, normalized, tmp_path)
    assert packet.directory.name == "notebooklm"
    for path in packet.files():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert "video-intel notebooklm import" in packet.readme_path.read_text(encoding="utf-8")


def test_manual_notice_denies_an_assumed_api(dataset) -> None:
    assert "does not assume" in MANUAL_WORKFLOW_NOTICE
    assert "does not automate private endpoints" in MANUAL_WORKFLOW_NOTICE


# --------------------------------------------------------------------- import
def test_import_accepts_a_well_formed_answer(tmp_path: Path, normalized) -> None:
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps(
            answer(
                [
                    item(
                        "It identifies potential threats and evaluates their likelihood.",
                        ["segment_0002"],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    result = import_records(
        path, video_id="synthetic01", known_segment_ids={"segment_0001", "segment_0002"}
    )
    assert result.ok
    assert len(result.items) == 1
    item_record = result.items[0]
    assert item_record.item_id.startswith("item_notebooklm_answer_")
    assert item_record.provenance is not None
    assert item_record.provenance.model.provider_name == "notebooklm_handoff"
    assert item_record.provenance.pipeline_id is PipelineId.CONCEPT


def test_import_rejects_malformed_json(tmp_path: Path, normalized) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ImportValidationError, match="malformed JSON"):
        import_records(path, video_id="synthetic01", known_segment_ids={"segment_0001"})


def test_import_rejects_a_contract_violation(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"items": [{"statement": "no type or citation"}]}), encoding="utf-8")
    result = import_records(path, video_id="synthetic01", known_segment_ids={"segment_0001"})
    assert result.errors
    assert "source_segment_ids" in result.errors[0] or "type" in result.errors[0]


def test_import_reports_unknown_segment_ids(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps(answer([item("A grounded claim.", ["segment_9999"])])), encoding="utf-8"
    )
    result = import_records(path, video_id="synthetic01", known_segment_ids={"segment_0001"})
    assert result.items == []
    assert result.rejected_items == 1
    assert "segment_9999" in result.unknown_segment_ids


def test_import_drops_unknown_ids_but_keeps_valid_ones(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps(
            answer([item("A grounded claim about threats.", ["segment_0001", "segment_4242"])])
        ),
        encoding="utf-8",
    )
    result = import_records(path, video_id="synthetic01", known_segment_ids={"segment_0001"})
    assert len(result.items) == 1
    assert result.items[0].source_segment_ids == ["segment_0001"]
    assert "segment_4242" in result.unknown_segment_ids


def test_parse_import_accepts_a_bare_array() -> None:
    response = parse_import([item("A claim.", ["segment_0001"])], video_id="v")
    assert len(response.items) == 1


def test_parse_import_rejects_non_objects() -> None:
    with pytest.raises(ImportValidationError, match="JSON object or array"):
        parse_import("just text", video_id="v")


def test_import_can_override_the_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(answer([item("A claim.", ["segment_0001"])])), encoding="utf-8")
    result = import_records(
        path,
        video_id="v",
        known_segment_ids={"segment_0001"},
        pipeline_id=PipelineId.FACT,
    )
    assert result.items[0].provenance is not None
    assert result.items[0].provenance.pipeline_id is PipelineId.FACT


# ---------------------------------------------------------- verification path
def test_imported_items_are_verified_like_any_model_output(dataset, normalized) -> None:
    """An import is model output, not fact: it faces the same verification."""
    extracted = [
        ExtractedItem(
            item_id="item_notebooklm_test_001",
            type=ItemType.CONCEPT,
            statement="It identifies potential threats and evaluates their likelihood.",
            source_segment_ids=["segment_0002"],
        ),
        ExtractedItem(
            item_id="item_notebooklm_test_002",
            type=ItemType.CONCEPT,
            statement="NotebookLM adds that incidents fell by ninety percent.",
            source_segment_ids=["segment_0002"],
        ),
    ]
    result = DatasetBuilder().build(
        DatasetInputs(
            video_id="synthetic01",
            title=None,
            segments=normalized.segments,
            items=extracted,
        )
    )
    retained = {entry.item_id for entry in result.dataset.items}
    assert "item_notebooklm_test_001" in retained
    assert "item_notebooklm_test_002" not in retained, "invented claims must not survive"

    dropped = {entry.item_id for entry in result.dropped}
    assert dropped == {"item_notebooklm_test_002"}
    statuses = [entry.verification for entry in result.dropped]
    assert VerificationStatus.UNSUPPORTED in statuses


# ------------------------------------------------------------------- storage
def test_stored_imports_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(answer([item("A claim.", ["segment_0001"])])), encoding="utf-8")
    result = import_records(path, video_id="v", known_segment_ids={"segment_0001"})
    stored = store_import(result, tmp_path)
    assert stored.is_file()

    loaded = load_imported_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].parsed is not None
    assert len(loaded[0].parsed.items) == 1


def test_loading_imports_survives_a_corrupt_file(tmp_path: Path) -> None:
    directory = tmp_path / "notebooklm" / "imported"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("{", encoding="utf-8")
    assert load_imported_records(tmp_path) == []


def test_loading_imports_without_a_directory(tmp_path: Path) -> None:
    assert load_imported_records(tmp_path / "nothing") == []
