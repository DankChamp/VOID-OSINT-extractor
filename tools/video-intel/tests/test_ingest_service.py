"""Ingestion service: segment construction, track selection, failure reporting."""

from __future__ import annotations

import pytest
from video_intel.ingest.errors import (
    InvalidVideoUrl,
    TranscriptRetrievalFailed,
    TranscriptUnavailable,
    VideoUnavailable,
)
from video_intel.ingest.service import (
    IngestRequest,
    TranscriptIngestor,
    build_segments,
    empty_cue_count,
    segment_id_for,
)
from video_intel.ingest.sources import CaptionSnippet, CaptionTrack, TrackInfo
from video_intel.schemas.common import TranscriptKind
from video_intel.testing.fakes import (
    FakeMetadataSource,
    FakeTranscriptSource,
    retrieval_failed,
    transcript_unavailable,
    video_unavailable,
)

VIDEO_ID = "synthetic01"
URL = "https://www.youtube.com/watch?v=synthetic01"


def _ingestor(**kwargs: object) -> TranscriptIngestor:
    return TranscriptIngestor(
        kwargs.get("transcript_source", FakeTranscriptSource()),  # type: ignore[arg-type]
        kwargs.get("metadata_source", FakeMetadataSource()),  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------ segment build
def test_segment_ids_are_sequential_and_stable() -> None:
    track = FakeTranscriptSource().fetch(VIDEO_ID, ("en",))
    segments = build_segments(track)
    assert [s.segment_id for s in segments] == [
        "segment_0001",
        "segment_0002",
        "segment_0003",
        "segment_0004",
    ]
    assert segment_id_for(1) == "segment_0001"


def test_timestamps_come_from_the_cues() -> None:
    track = FakeTranscriptSource().fetch(VIDEO_ID, ("en",))
    segments = build_segments(track)
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == pytest.approx(3.2)
    assert segments[-1].end_seconds == pytest.approx(12.5)


def test_empty_cues_do_not_become_segments() -> None:
    track = FakeTranscriptSource().fetch(VIDEO_ID, ("en",))
    segments = build_segments(track)
    assert all(segment.text.strip() for segment in segments)
    assert len(segments) == 4  # 5 cues, one blank dropped, "[Music]" retained as content
    assert segments[2].text == "[Music]"


def test_cue_text_is_stripped_not_rewritten() -> None:
    snippets = (CaptionSnippet(text="  padded words  ", start=1.0, duration=2.0),)
    track = CaptionTrack(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=snippets,
        source="fake",
    )
    assert build_segments(track)[0].text == "padded words"


def test_dropped_blank_cues_are_counted_in_provenance() -> None:
    source = FakeTranscriptSource()
    result = _ingestor(transcript_source=source).ingest(IngestRequest(url=URL))
    assert result.transcript.source_detail["cue_count"] == "5"
    assert result.transcript.source_detail["empty_cues_dropped"] == "1"
    assert empty_cue_count(source.fetch(VIDEO_ID, ("en",)), result.transcript.segments) == 1


def test_all_blank_track_is_reported_not_fabricated() -> None:
    snippets = (CaptionSnippet(text="   ", start=0.0, duration=1.0),)
    track = CaptionTrack(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
        snippets=snippets,
        source="fake",
    )
    with pytest.raises(TranscriptUnavailable, match="every caption cue was empty"):
        build_segments(track)


# ----------------------------------------------------------------- track selection
def test_language_priority_wins_over_manual_preference() -> None:
    tracks = (
        TrackInfo(language="English", language_code="en", is_generated=True),
        TrackInfo(language="Deutsch", language_code="de", is_generated=False),
    )
    selected = _ingestor().select_track(VIDEO_ID, tracks, ("de", "en"))
    assert selected.language_code == "de"


def test_manual_track_wins_at_equal_language() -> None:
    tracks = (
        TrackInfo(language="English (auto-generated)", language_code="en", is_generated=True),
        TrackInfo(language="English", language_code="en", is_generated=False),
    )
    selected = _ingestor().select_track(VIDEO_ID, tracks, ("en",))
    assert selected.is_generated is False


def test_language_match_is_case_insensitive() -> None:
    tracks = (TrackInfo(language="English", language_code="en", is_generated=False),)
    assert _ingestor().select_track(VIDEO_ID, tracks, ("EN",)).language_code == "en"


def test_missing_language_lists_what_is_available() -> None:
    tracks = (TrackInfo(language="English", language_code="en", is_generated=True),)
    with pytest.raises(TranscriptUnavailable) as excinfo:
        _ingestor().select_track(VIDEO_ID, tracks, ("fr",))
    assert "available: en" in (excinfo.value.hint or "") + excinfo.value.message


def test_no_tracks_reports_transcript_unavailable() -> None:
    with pytest.raises(TranscriptUnavailable, match="no caption tracks"):
        _ingestor().select_track(VIDEO_ID, (), ("en",))


# ------------------------------------------------------------------------- ingest
def test_ingest_returns_metadata_transcript_and_track() -> None:
    source = FakeTranscriptSource()
    result = _ingestor(transcript_source=source).ingest(IngestRequest(url=URL, languages=("en",)))

    assert result.video.video_id == VIDEO_ID
    assert result.metadata.title.startswith("Synthetic")
    assert result.segment_count == 4
    assert result.transcript.kind is TranscriptKind.AUTOMATIC
    assert result.transcript.detected_language == "en"
    assert result.transcript.source == "fake"
    assert source.calls == [(VIDEO_ID, ("en",))]


def test_ingest_records_available_languages_in_provenance() -> None:
    result = _ingestor().ingest(IngestRequest(url=URL))
    assert result.transcript.source_detail["available_languages"] == "en"
    assert result.transcript.source_detail["is_generated"] == "true"


def test_manual_track_is_reported_as_manual() -> None:
    tracks = (TrackInfo(language="English", language_code="en", is_generated=False),)
    source = FakeTranscriptSource(tracks=tracks)
    result = _ingestor(transcript_source=source).ingest(IngestRequest(url=URL))
    assert result.transcript.kind is TranscriptKind.MANUAL


def test_duration_falls_back_to_transcript_end_when_metadata_lacks_it() -> None:
    result = _ingestor(metadata_source=FakeMetadataSource(duration_seconds=None)).ingest(
        IngestRequest(url=URL)
    )
    assert result.duration_seconds == pytest.approx(12.5)


def test_ingest_uses_prefetched_metadata_without_refetching() -> None:
    metadata = FakeMetadataSource().fetch(VIDEO_ID)
    ingestor = _ingestor()
    ingestor.metadata_source = FakeMetadataSource()  # type: ignore[assignment]
    result = ingestor.ingest(IngestRequest(url=URL), metadata=metadata)
    assert result.metadata.title == metadata.title


def test_missing_captions_surface_as_transcript_unavailable() -> None:
    ingestor = _ingestor(transcript_source=FakeTranscriptSource(failure=transcript_unavailable()))
    with pytest.raises(TranscriptUnavailable) as excinfo:
        ingestor.ingest(IngestRequest(url=URL))
    assert "speech-to-text" in (excinfo.value.hint or "")


def test_unavailable_video_is_reported_with_its_own_error() -> None:
    ingestor = _ingestor(transcript_source=FakeTranscriptSource(failure=video_unavailable()))
    with pytest.raises(VideoUnavailable):
        ingestor.ingest(IngestRequest(url=URL))


def test_blocked_requests_are_reported_as_retrieval_failures() -> None:
    ingestor = _ingestor(transcript_source=FakeTranscriptSource(failure=retrieval_failed()))
    with pytest.raises(TranscriptRetrievalFailed) as excinfo:
        ingestor.ingest(IngestRequest(url=URL))
    assert "retry" in (excinfo.value.hint or "")


def test_ingest_rejects_invalid_url_before_any_fetch() -> None:
    source = FakeTranscriptSource()
    with pytest.raises(InvalidVideoUrl):
        _ingestor(transcript_source=source).ingest(IngestRequest(url="https://example.com/x"))
    assert source.calls == []
