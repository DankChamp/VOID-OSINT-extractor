"""YouTube URL validation (§4)."""

from __future__ import annotations

import pytest
from video_intel.ingest.errors import InvalidVideoUrl
from video_intel.ingest.youtube import extract_video_id, parse_youtube_url

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&list=PL123&index=2",
        f"http://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=42",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        VIDEO_ID,
    ],
)
def test_accepted_forms(url: str) -> None:
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/12345",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch",
        "https://youtu.be/",
        "https://www.youtube.com/channel/UC123",
        f"https://www.youtube.com/watch?v={VIDEO_ID}extra",
    ],
)
def test_rejected_forms(url: str) -> None:
    assert extract_video_id(url) is None


def test_parse_returns_canonical_url() -> None:
    parsed = parse_youtube_url(f"https://youtu.be/{VIDEO_ID}?t=90")
    assert parsed.video_id == VIDEO_ID
    assert parsed.canonical_url == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert str(parsed) == parsed.canonical_url


def test_parse_raises_typed_error_with_hint() -> None:
    with pytest.raises(InvalidVideoUrl) as excinfo:
        parse_youtube_url("https://example.com/video/1")
    error = excinfo.value
    assert error.exit_code == 1
    assert "not a recognisable YouTube video URL" in error.message
    assert "youtu.be" in (error.hint or "")
    assert "hint:" in error.render()
