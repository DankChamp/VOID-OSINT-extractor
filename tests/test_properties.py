"""Property-style checks: invariants that must hold for any input, not just fixtures.

These are the tests most likely to catch edge cases the hand-written fixtures miss:
odd characters, huge segments, many tiny segments, unicode and duplicate text.
"""

from __future__ import annotations

import random

import pytest

from video_intel.aggregation.consolidate import Consolidator, classify_pair
from video_intel.aggregation.verify import SegmentIndex, verify_items
from video_intel.pipeline.chunking import ChunkingError, chunk_transcript
from video_intel.schemas.chunk import ChunkingConfig
from video_intel.schemas.common import ItemType, VerificationStatus
from video_intel.schemas.extraction import ExtractedItem
from video_intel.schemas.transcript import TranscriptSegment

WORDS = [
    "risk",
    "assessment",
    "likelihood",
    "impact",
    "threat",
    "control",
    "audit",
    "mitigation",
    "transfer",
    "acceptance",
    "avoidance",
    "framework",
    "policy",
]
PUNCTUATION = [".", "!", "?", ";", ":", ",", " -", " ...", ""]
UNICODE = ["café", "naïve", "风险", "ελληνικά", "русский", "🇬🇧", "naïve—dash"]


def random_sentence(rng: random.Random) -> str:
    length = rng.randint(3, 18)
    words = [rng.choice(WORDS) for _ in range(length)]
    text = " ".join(words)
    return text.capitalize() + rng.choice(PUNCTUATION)


def random_segments(rng: random.Random, count: int) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    clock = 0.0
    for index in range(count):
        sentences = " ".join(random_sentence(rng) for _ in range(rng.randint(1, 4)))
        if rng.random() < 0.15:
            sentences = f"{rng.choice(UNICODE)} {sentences}"
        duration = round(rng.uniform(1.0, 30.0), 3)
        segments.append(
            TranscriptSegment(
                segment_id=f"segment_{index + 1:04d}",
                start_seconds=clock,
                end_seconds=clock + duration,
                text=sentences,
            )
        )
        clock += duration
    return segments


CONFIGS = [
    ChunkingConfig(max_chars=200, overlap_chars=40, min_chars=50),
    ChunkingConfig(max_chars=600, overlap_chars=120, min_chars=100),
    ChunkingConfig(max_chars=4000, overlap_chars=600, min_chars=500),
    ChunkingConfig(max_chars=300, overlap_chars=0, min_chars=0),
]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("config_index", range(len(CONFIGS)))
def test_chunking_invariants_hold_for_random_transcripts(seed: int, config_index: int) -> None:
    rng = random.Random(seed * 100 + config_index)
    segments = random_segments(rng, rng.randint(1, 40))
    config = CONFIGS[config_index]

    chunks = chunk_transcript(segments, config, video_id="fuzz").chunks

    assert chunks, "a non-empty transcript must produce chunks"
    known = {segment.segment_id for segment in segments}

    covered: set[str] = set()
    for index, chunk in enumerate(chunks):
        assert chunk.index == index
        assert chunk.chunk_id == f"chunk_{index + 1:04d}"
        assert set(chunk.segment_ids) <= known
        assert chunk.text.strip()
        assert chunk.start_seconds <= chunk.end_seconds
        if chunk.char_count > config.max_chars:
            # Only a single sentence longer than the budget may exceed it.
            longest_sentence = max(len(part) for part in chunk.text.split(". "))
            assert len(chunk.text) - longest_sentence + 1 < config.max_chars
        if index == 0:
            assert chunk.overlaps_previous is False
        covered.update(chunk.segment_ids)

    assert covered == known, "every segment must appear in at least one chunk"


@pytest.mark.parametrize("seed", range(8))
def test_chunking_is_deterministic(seed: int) -> None:
    rng = random.Random(seed)
    segments = random_segments(rng, 25)
    config = CONFIGS[seed % len(CONFIGS)]
    first = chunk_transcript(segments, config, video_id="fuzz")
    second = chunk_transcript(segments, config, video_id="fuzz")
    assert [c.model_dump() for c in first.chunks] == [c.model_dump() for c in second.chunks]
    assert first.cache_key == second.cache_key


def test_chunking_never_loses_or_duplicates_text() -> None:
    rng = random.Random(99)
    segments = random_segments(rng, 30)
    config = CONFIGS[1]
    chunks = chunk_transcript(segments, config, video_id="fuzz").chunks

    joined = " ".join(chunk.text for chunk in chunks)
    for word in {segment.text.split()[0] for segment in segments}:
        assert word in joined


def test_chunking_handles_a_single_huge_segment() -> None:
    text = ". ".join(f"Sentence number {index} carries some content" for index in range(400))
    segment = TranscriptSegment(
        segment_id="segment_0001", start_seconds=0.0, end_seconds=900.0, text=text
    )
    config = ChunkingConfig(max_chars=500, overlap_chars=100, min_chars=100)
    chunks = chunk_transcript([segment], config, video_id="fuzz").chunks
    assert len(chunks) > 1
    assert all(chunk.char_count <= 500 for chunk in chunks)


def test_chunking_handles_many_tiny_segments() -> None:
    segments = [
        TranscriptSegment(
            segment_id=f"segment_{index + 1:04d}",
            start_seconds=float(index),
            end_seconds=float(index) + 0.5,
            text=f"word{index}.",
        )
        for index in range(500)
    ]
    config = ChunkingConfig(max_chars=120, overlap_chars=10, min_chars=10)
    chunks = chunk_transcript(segments, config, video_id="fuzz").chunks
    assert len(chunks) > 10
    covered = {sid for chunk in chunks for sid in chunk.segment_ids}
    assert len(covered) == 500


def test_chunking_rejects_an_empty_transcript() -> None:
    with pytest.raises(ChunkingError):
        chunk_transcript([], CONFIGS[0], video_id="fuzz")


def test_chunking_handles_whitespace_only_and_unicode() -> None:
    segments = [
        TranscriptSegment(
            segment_id="segment_0001",
            start_seconds=0.0,
            end_seconds=1.0,
            text="风险 评估 是 过程。",
        ),
        TranscriptSegment(
            segment_id="segment_0002",
            start_seconds=1.0,
            end_seconds=2.0,
            text="Ελληνικά   με    πολλά    κενά.",
        ),
    ]
    chunks = chunk_transcript(segments, CONFIGS[0], video_id="fuzz").chunks
    combined = " ".join(chunk.text for chunk in chunks)
    assert "风险" in combined
    assert "π" in combined


# ------------------------------------------------------------- aggregation fuzz
def test_consolidation_never_loses_a_unique_statement() -> None:
    rng = random.Random(7)
    items = [
        ExtractedItem(
            item_id=f"item_{index:03d}",
            type=ItemType.FACT,
            statement=f"Unique statement number {index} about topic {index % 7}.",
            source_segment_ids=[f"segment_{index % 5 + 1:04d}"],
        )
        for index in range(25)
    ]
    del rng
    result = Consolidator().run(items)
    kept_statements = {item.statement for item in result.items}
    # Every statement survives somewhere: merged as a detail, or still a statement.
    for item in items:
        assert item.statement in kept_statements or any(
            item.statement in kept.statement or item.statement in kept.details
            for kept in result.items
        )


def test_classification_is_symmetric() -> None:
    left = ExtractedItem(
        item_id="item_a",
        type=ItemType.FACT,
        statement="Risk assessment evaluates likelihood.",
        source_segment_ids=["segment_0001"],
    )
    right = ExtractedItem(
        item_id="item_b",
        type=ItemType.FACT,
        statement="Risk assessment evaluates impact.",
        source_segment_ids=["segment_0002"],
    )
    assert classify_pair(left, right).kind == classify_pair(right, left).kind


def test_identical_items_collapse_to_one() -> None:
    item = ExtractedItem(
        item_id="item_a",
        type=ItemType.FACT,
        statement="Teams recovered forty percent faster.",
        source_segment_ids=["segment_0001"],
    )
    clone = item.model_copy(update={"item_id": "item_b"})
    result = Consolidator().run([item, clone])
    assert len(result.items) == 1
    assert result.items[0].source_segment_ids == ["segment_0001"]


def test_verification_is_deterministic() -> None:
    segments = [
        TranscriptSegment(
            segment_id="segment_0001",
            start_seconds=0.0,
            end_seconds=5.0,
            text="Risk assessment evaluates likelihood and impact.",
        )
    ]
    item = ExtractedItem(
        item_id="item_a",
        type=ItemType.FACT,
        statement="Risk assessment evaluates likelihood.",
        source_segment_ids=["segment_0001"],
    )
    index = SegmentIndex.of(segments)
    first = verify_items([item], index).items[0].verification
    second = verify_items([item], index).items[0].verification
    assert first.status is second.status is VerificationStatus.SUPPORTED
    assert first.evidence[0].quote == second.evidence[0].quote
