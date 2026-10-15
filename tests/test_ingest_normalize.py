"""Normalization fidelity: meaning, timing and repetition are preserved (§3)."""

from __future__ import annotations

import pytest
from tests.fakes import CUE_TEXTS, FakeTranscriptSource

from video_intel.ingest.normalize import DEFAULT_RULES, TranscriptNormalizer
from video_intel.ingest.service import build_segments
from video_intel.ingest.sources import CaptionSnippet
from video_intel.schemas.transcript import RawTranscript, hash_transcript

VIDEO_ID = "synthetic01"

REPEATED = (
    ("Let me define it clearly.", 0.0, 3.0),
    ("Risk assessment is the process of evaluating risk.", 3.0, 7.0),
    ("Let me define it clearly.", 7.0, 10.0),
    ("Risk assessment is the process of evaluating risk.", 10.0, 14.0),
)


def _transcript(snippets: tuple[tuple[str, float, float], ...] = CUE_TEXTS) -> RawTranscript:
    track = FakeTranscriptSource(
        snippets=tuple(
            CaptionSnippet(text=text, start=start, duration=end - start)
            for text, start, end in snippets
        )
    ).fetch(VIDEO_ID, ("en",))
    return RawTranscript(
        video_id=VIDEO_ID,
        segments=build_segments(track),
        source="fake",
    )


def test_normalization_records_every_rule() -> None:
    result = TranscriptNormalizer().normalize(_transcript())
    assert [step.name for step in result.steps] == [rule.name for rule in DEFAULT_RULES]
    assert all(step.description for step in result.steps)


def test_whitespace_and_control_characters_are_cleaned() -> None:
    result = TranscriptNormalizer().normalize(_transcript())
    cleaned = {segment.segment_id: segment.text for segment in result.segments}
    assert cleaned["segment_0002"] == "Today we discuss risk assessment ."


def test_timestamps_are_copied_unchanged() -> None:
    raw = _transcript()
    normalized = TranscriptNormalizer().normalize(raw)
    assert [(s.segment_id, s.start_seconds, s.end_seconds) for s in normalized.segments] == [
        (s.segment_id, s.start_seconds, s.end_seconds) for s in raw.segments
    ]


def test_original_text_is_kept_on_each_segment() -> None:
    raw = _transcript()
    normalized = TranscriptNormalizer().normalize(raw)
    changed = [s for s in normalized.segments if s.text != s.original_text]
    assert changed, "fixture is expected to contain text that normalization touches"
    for segment in changed:
        assert segment.raw_text is not None
        assert segment.raw_text != segment.text


def test_source_hash_refers_to_the_raw_transcript() -> None:
    raw = _transcript()
    normalized = TranscriptNormalizer().normalize(raw)
    assert normalized.source_transcript_hash == hash_transcript(raw.text())


def test_repeated_explanations_are_not_removed() -> None:
    raw = _transcript(REPEATED)
    normalized = TranscriptNormalizer().normalize(raw)
    assert len(normalized.segments) == len(raw.segments) == 4
    texts = [segment.text for segment in normalized.segments]
    assert texts.count("Let me define it clearly.") == 2
    assert texts.count("Risk assessment is the process of evaluating risk.") == 2


def test_blank_cues_are_counted_not_silently_dropped() -> None:
    normalized = TranscriptNormalizer().normalize(_transcript())
    assert all(segment.text.strip() for segment in normalized.segments)
    step = next(s for s in normalized.steps if s.name == "drop_empty_segments")
    assert step.segments_changed == 0  # blank cues are already excluded at ingestion


def test_segment_ids_remain_stable_after_normalization() -> None:
    raw = _transcript()
    normalized = TranscriptNormalizer().normalize(raw)
    assert [s.segment_id for s in normalized.segments] == [
        s.segment_id for s in raw.segments if s.text.strip()
    ]


def test_normalization_is_idempotent() -> None:
    normalizer = TranscriptNormalizer()
    once = normalizer.normalize(_transcript())
    twice = normalizer.normalize(
        RawTranscript(
            video_id=once.video_id,
            segments=[s.model_copy(update={"raw_text": s.text}) for s in once.segments],
            source="fake",
        )
    )
    assert [s.text for s in once.segments] == [s.text for s in twice.segments]


def test_normalization_rejects_empty_transcript() -> None:
    empty = RawTranscript.model_construct(video_id="x", segments=[])
    with pytest.raises(ValueError, match="empty transcript"):
        TranscriptNormalizer().normalize(empty)
