"""Test doubles shipped with the package.

The synthetic sources live inside ``video_intel`` rather than beside the tests so that
they are importable under a name unique to this tool. In a multi-tool workspace every
tool has a ``tests/`` directory, and a top-level ``tests.fakes`` would collide with the
next tool added.

They are part of the distribution deliberately: anything extending video-intel can build
its own tests against the same doubles without reaching into this repository.
"""

from video_intel.testing.fakes import (
    CUE_TEXTS,
    FakeMetadataSource,
    FakeProvider,
    FakeTranscriptSource,
    extraction_response_json,
)

__all__ = [
    "CUE_TEXTS",
    "FakeMetadataSource",
    "FakeProvider",
    "FakeTranscriptSource",
    "extraction_response_json",
]
