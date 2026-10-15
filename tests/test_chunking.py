"""Chunking (§6): boundaries, overlap, coverage, determinism, timestamps."""

from __future__ import annotations

import pytest

from video_intel.pipeline.chunking import (
    ChunkingError,
    build_pieces,
    chunk_transcript,
    split_sentences,
)
from video_intel.schemas.chunk import ChunkingConfig
from video_intel.schemas.transcript import TranscriptSegment

VIDEO_ID = "synthetic01"


def segments(*texts: str, start: float = 0.0, step: float = 10.0) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            segment_id=f"segment_{index + 1:04d}",
            start_seconds=start + step * index,
            end_seconds=start + step * index + step - 1,
            text=text,
        )
        for index, text in enumerate(texts)
    ]


SAMPLE = segments(
    "Risk assessment identifies potential threats. It evaluates their likelihood.",
    "It also evaluates their potential impact. Teams document assets first.",
    "The four treatment options are mitigation. They are transfer, acceptance, avoidance.",
)


def test_sentence_splitting() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("no terminator here") == ["no terminator here"]
    assert split_sentences("   ") == []


def test_pieces_never_exceed_the_budget() -> None:
    long_text = " ".join(f"word{index}" for index in range(400))
    pieces = build_pieces(
        segments(long_text), ChunkingConfig(max_chars=300, overlap_chars=50, min_chars=50)
    )
    assert len(pieces) > 1
    assert all(len(piece.text) <= 300 for piece in pieces)
    assert "".join(piece.text.replace(" ", "") for piece in pieces) == long_text.replace(" ", "")


def test_chunks_respect_the_character_budget() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=200, overlap_chars=40, min_chars=50),
        video_id=VIDEO_ID,
    )
    assert result.chunks
    for chunk in result.chunks:
        assert chunk.char_count <= 200, chunk.chunk_id


def test_every_segment_is_covered_exactly_by_the_union() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=180, overlap_chars=60, min_chars=50),
        video_id=VIDEO_ID,
    )
    covered = {sid for chunk in result.chunks for sid in chunk.segment_ids}
    assert covered == {segment.segment_id for segment in SAMPLE}


def test_every_chunk_cites_real_segments() -> None:
    known = {segment.segment_id for segment in SAMPLE}
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=180, overlap_chars=60, min_chars=50),
        video_id=VIDEO_ID,
    )
    for chunk in result.chunks:
        assert set(chunk.segment_ids) <= known


def test_chunks_are_deterministic() -> None:
    config = ChunkingConfig(max_chars=220, overlap_chars=60, min_chars=40)
    first = chunk_transcript(SAMPLE, config, video_id=VIDEO_ID)
    second = chunk_transcript(SAMPLE, config, video_id=VIDEO_ID)
    assert [c.model_dump() for c in first.chunks] == [c.model_dump() for c in second.chunks]
    assert first.cache_key == second.cache_key


def test_ids_and_indexes_are_sequential() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=200, overlap_chars=50, min_chars=50),
        video_id=VIDEO_ID,
    )
    assert [chunk.chunk_id for chunk in result.chunks] == [
        f"chunk_{index + 1:04d}" for index in range(len(result.chunks))
    ]
    assert [chunk.index for chunk in result.chunks] == list(range(len(result.chunks)))


def test_timestamps_come_from_the_segments() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=200, overlap_chars=40, min_chars=50),
        video_id=VIDEO_ID,
    )
    by_id = {segment.segment_id: segment for segment in SAMPLE}
    for chunk in result.chunks:
        cited = [by_id[sid] for sid in chunk.segment_ids]
        assert chunk.start_seconds == min(s.start_seconds for s in cited)
        assert chunk.end_seconds == max(s.end_seconds for s in cited)


def test_neighbouring_chunks_overlap_after_the_first() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=150, overlap_chars=50, min_chars=40),
        video_id=VIDEO_ID,
    )
    assert len(result.chunks) > 2
    assert result.chunks[0].overlaps_previous is False
    assert all(chunk.overlaps_previous for chunk in result.chunks[1:])


def test_overlap_text_is_repeated_across_the_boundary() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=220, overlap_chars=80, min_chars=40),
        video_id=VIDEO_ID,
    )
    first, second = result.chunks[0], result.chunks[1]
    first_tail = first.text[-60:]
    assert first_tail[:30] in second.text


def test_zero_overlap_is_allowed() -> None:
    result = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=200, overlap_chars=0, min_chars=40),
        video_id=VIDEO_ID,
    )
    assert all(chunk.overlaps_previous is False for chunk in result.chunks[1:])


def test_small_tail_chunk_is_merged_when_it_fits() -> None:
    # A trailing fragment below min_chars is folded back when the budget allows.
    texts = ["A" * 90 + ".", "B" * 90 + ".", "tiny tail"]
    config = ChunkingConfig(max_chars=400, overlap_chars=40, min_chars=30)
    result = chunk_transcript(segments(*texts), config, video_id=VIDEO_ID)
    assert all(chunk.char_count >= 30 for chunk in result.chunks)


def test_small_tail_chunk_is_kept_when_merging_would_overflow() -> None:
    texts = ["A" * 200 + ".", "B" * 200 + ".", "tail"]
    config = ChunkingConfig(max_chars=400, overlap_chars=40, min_chars=300)
    result = chunk_transcript(segments(*texts), config, video_id=VIDEO_ID)
    assert all(chunk.char_count <= 400 for chunk in result.chunks)


def test_single_long_sentence_is_split_at_word_boundaries() -> None:
    single = segments(" ".join(f"token{index}" for index in range(200)))
    config = ChunkingConfig(max_chars=250, overlap_chars=30, min_chars=30)
    result = chunk_transcript(single, config, video_id=VIDEO_ID)
    assert len(result.chunks) > 1
    assert all(chunk.char_count <= 250 for chunk in result.chunks)
    words = result.chunks[0].text.split()
    assert words[0] == "token0"


def test_sentence_boundaries_are_respected_when_they_fit() -> None:
    texts = ["First sentence here. Second sentence here.", "Third sentence only."]
    config = ChunkingConfig(max_chars=120, overlap_chars=20, min_chars=20)
    result = chunk_transcript(segments(*texts), config, video_id=VIDEO_ID)
    for chunk in result.chunks:
        assert "Second sentence here." not in chunk.text or "First sentence here." in chunk.text


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(ChunkingError, match="empty transcript"):
        chunk_transcript([], ChunkingConfig(), video_id=VIDEO_ID)


def test_blank_transcript_is_rejected() -> None:
    blank = TranscriptSegment.model_construct(
        segment_id="segment_0001", start_seconds=0.0, end_seconds=1.0, text="   "
    )
    with pytest.raises(ChunkingError, match="no text to chunk"):
        chunk_transcript([blank], ChunkingConfig(), video_id=VIDEO_ID)


def test_cache_key_changes_with_configuration_and_transcript() -> None:
    base = chunk_transcript(
        SAMPLE,
        ChunkingConfig(max_chars=200, overlap_chars=50, min_chars=50),
        video_id=VIDEO_ID,
    )
    other_config = chunk_transcript(
        SAMPLE, ChunkingConfig(max_chars=260, overlap_chars=50, min_chars=50), video_id=VIDEO_ID
    )
    other_text = chunk_transcript(
        segments("Different content entirely for the cache key."),
        ChunkingConfig(max_chars=200, overlap_chars=50, min_chars=50),
        video_id=VIDEO_ID,
    )
    keys = {base.cache_key, other_config.cache_key, other_text.cache_key}
    assert len(keys) == 3


def test_transcript_hash_is_recorded_and_forwarded() -> None:
    result = chunk_transcript(SAMPLE, ChunkingConfig(), video_id=VIDEO_ID, transcript_hash="abc123")
    assert result.transcript_hash == "abc123"
    assert result.cache_key


def test_unicode_content_is_preserved_in_chunk_text() -> None:
    texts = ["Le café coûte 5 € — c'est cher. Oui, vraiment cher."]
    result = chunk_transcript(
        segments(*texts),
        ChunkingConfig(max_chars=120, overlap_chars=20, min_chars=20),
        video_id=VIDEO_ID,
    )
    assert "café" in result.chunks[0].text
    assert "€" in result.chunks[0].text


def test_chunking_of_a_very_long_transcript_stays_within_budget() -> None:
    many = segments(
        *[f"Segment number {index}. It carries a distinct claim." for index in range(200)]
    )
    config = ChunkingConfig(max_chars=600, overlap_chars=100, min_chars=100)
    result = chunk_transcript(many, config, video_id=VIDEO_ID)
    assert len(result.chunks) > 1
    assert all(chunk.char_count <= 600 for chunk in result.chunks)
    covered = {sid for chunk in result.chunks for sid in chunk.segment_ids}
    assert len(covered) == 200
