"""Transcript and chunk schemas: timestamp fidelity, ordering, determinism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from video_intel.schemas.chunk import ChunkingConfig, chunk_set_hash
from video_intel.schemas.common import TranscriptKind
from video_intel.schemas.transcript import (
    NormalizationStep,
    NormalizedTranscript,
    RawTranscript,
    TranscriptSegment,
    hash_transcript,
)


def test_fixture_transcript_validates(transcript: RawTranscript) -> None:
    assert transcript.video_id == "synthetic001"
    assert len(transcript.segments) == 12
    assert transcript.kind is TranscriptKind.AUTOMATIC


def test_timestamps_survive_roundtrip(transcript: RawTranscript) -> None:
    restored = RawTranscript.model_validate_json(transcript.model_dump_json())
    original = [(s.segment_id, s.start_seconds, s.end_seconds) for s in transcript.segments]
    copied = [(s.segment_id, s.start_seconds, s.end_seconds) for s in restored.segments]
    assert original == copied


def test_segment_ids_are_preserved_in_order(transcript: RawTranscript) -> None:
    ids = [segment.segment_id for segment in transcript.segments]
    assert ids == sorted(ids)
    assert ids[0] == "segment_001"


def test_segment_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="precedes start_seconds"):
        TranscriptSegment(segment_id="segment_001", start_seconds=10.0, end_seconds=4.0, text="x y")


def test_segment_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="text is blank"):
        TranscriptSegment(segment_id="segment_001", start_seconds=0.0, end_seconds=1.0, text="   ")


def test_segment_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate(
            {
                "segment_id": "segment_001",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "text": "hello there",
                "speaker": "narrator",
            }
        )


def test_segment_id_pattern_is_enforced() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment(segment_id="seg-1", start_seconds=0.0, end_seconds=1.0, text="hello you")


def test_duplicate_segment_ids_rejected() -> None:
    segment = TranscriptSegment(
        segment_id="segment_001", start_seconds=0.0, end_seconds=1.0, text="hello there"
    )
    duplicate = segment.model_copy()
    with pytest.raises(ValidationError, match="duplicate segment_id"):
        RawTranscript(video_id="v1", segments=[segment, duplicate])


def test_raw_text_is_kept_alongside_cleaned_text() -> None:
    segment = TranscriptSegment(
        segment_id="segment_001",
        start_seconds=0.0,
        end_seconds=2.0,
        text="cleaned text",
        raw_text="  [Music] cleaned   text  ",
    )
    assert segment.text == "cleaned text"
    assert segment.original_text == "  [Music] cleaned   text  "


def test_normalized_transcript_records_steps_and_source_hash(transcript: RawTranscript) -> None:
    normalized = NormalizedTranscript(
        video_id=transcript.video_id,
        source_transcript_hash=hash_transcript(transcript.text()),
        segments=transcript.segments,
        steps=[
            NormalizationStep(name="whitespace", description="collapse runs", segments_changed=3)
        ],
    )
    assert normalized.steps[0].segments_changed == 3
    assert normalized.source_transcript_hash == hash_transcript(transcript.text())
    assert [s.segment_id for s in normalized.segments] == [
        s.segment_id for s in transcript.segments
    ]


def test_hash_transcript_is_stable_and_content_sensitive() -> None:
    assert hash_transcript("abc") == hash_transcript("abc")
    assert hash_transcript("abc") != hash_transcript("abd")


def test_chunking_config_rejects_overlap_larger_than_window() -> None:
    with pytest.raises(ValidationError, match="overlap_chars must be smaller"):
        ChunkingConfig(max_chars=1000, overlap_chars=1000)


def test_chunking_config_estimates_tokens() -> None:
    assert ChunkingConfig(max_chars=4000, estimated_chars_per_token=4).estimated_tokens == 1000


def test_chunk_set_hash_is_deterministic() -> None:
    config = ChunkingConfig(max_chars=3000, overlap_chars=300)
    assert chunk_set_hash(config, "abc") == chunk_set_hash(config, "abc")
    assert chunk_set_hash(config, "abc") != chunk_set_hash(config, "abd")
    assert chunk_set_hash(config, "abc") != chunk_set_hash(ChunkingConfig(max_chars=4000), "abc")


def test_chunks_cite_segments_and_keep_timestamps(chunks: object) -> None:
    assert [chunk.chunk_id for chunk in chunks.chunks] == ["chunk_0001", "chunk_0002"]  # type: ignore[attr-defined]
    assert chunks.chunks[1].start_seconds == 99.0  # type: ignore[attr-defined]
    assert chunks.chunks[1].overlaps_previous is True  # type: ignore[attr-defined]
    assert "segment_005" in chunks.chunks[0].segment_ids  # type: ignore[attr-defined]
    assert "segment_005" in chunks.chunks[1].segment_ids  # type: ignore[attr-defined]
