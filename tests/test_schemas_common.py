"""Common primitives: timestamps, slugs, id patterns, enum values."""

from __future__ import annotations

import pytest

from video_intel.schemas.common import (
    SCHEMA_VERSION,
    ExecutionMode,
    OutputMode,
    PipelineId,
    ProviderKind,
    StageId,
    VerificationStatus,
    format_timestamp,
    slugify,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00.000"),
        (1.5, "00:00:01.500"),
        (61.25, "00:01:01.250"),
        (3661.125, "01:01:01.125"),
        (359999.999, "99:59:59.999"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_format_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative timestamp"):
        format_timestamp(-1.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Risk Assessment 101", "risk-assessment-101"),
        ("  Spaces   and---punctuation! ", "spaces-and-punctuation"),
        ("", "unknown"),
        ("///", "unknown"),
    ],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected


def test_slugify_truncates_without_trailing_hyphen() -> None:
    slug = slugify("a" * 60, max_length=10)
    assert len(slug) == 10
    assert not slug.endswith("-")


def test_schema_version_is_declared() -> None:
    assert SCHEMA_VERSION.count(".") == 2


def test_verification_states_match_specification() -> None:
    assert {state.value for state in VerificationStatus} == {
        "supported",
        "partially_supported",
        "unsupported",
        "needs_review",
    }


def test_pipeline_ids_match_specification() -> None:
    assert {pipeline.value for pipeline in PipelineId} == {
        "concept",
        "definition",
        "fact",
        "example",
        "framework",
        "process",
        "comparison",
        "coverage_audit",
    }


def test_stage_ids_cover_the_pipeline() -> None:
    values = {stage.value for stage in StageId}
    assert {"raw_transcript", "normalized_transcript", "chunks", "knowledge_dataset"} <= values
    assert {"output_notes", "output_json", "output_ppt", "output_summary"} <= values


def test_execution_modes_and_output_modes() -> None:
    assert {mode.value for mode in ExecutionMode} == {"local_only", "cloud_only", "hybrid"}
    assert {mode.value for mode in OutputMode} == {"deep", "ppt", "quick"}


def test_provider_kinds_include_local_and_cloud() -> None:
    values = {kind.value for kind in ProviderKind}
    assert "ollama" in values
    assert {"deepseek", "groq", "openai_compatible", "gemini"} <= values
    assert "notebooklm_handoff" in values
