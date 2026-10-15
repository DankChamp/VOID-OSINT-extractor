"""Shared pytest fixtures.

Every test runs against a temporary data directory with ``VIDEO_INTEL_*`` and provider
API-key variables cleared, so the suite never reads real configuration or credentials
and never touches the user's home directory.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from video_intel.cli.main import app
from video_intel.config.models import AppConfig
from video_intel.logging_setup import SECRETS
from video_intel.schemas.chunk import Chunk, ChunkingConfig, ChunkSet
from video_intel.schemas.common import PipelineId, ProviderKind
from video_intel.schemas.extraction import ExtractionResponse
from video_intel.schemas.transcript import RawTranscript
from video_intel.storage.paths import ProjectPaths

FIXTURES = Path(__file__).parent / "fixtures"

API_KEY_VARS = (
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in a clean working directory with no ambient credentials.

    Tests must not pick up the developer's own ``video-intel.yaml`` or ``.env``, so the
    process starts in a temporary directory with an empty home.

    The home directory is redirected through every variable a platform consults. Setting
    only ``HOME`` looks sufficient and is not: ``expanduser`` and ``Path.home`` use
    ``USERPROFILE`` on Windows and ignore ``HOME`` entirely, which would let a test on
    Windows read the real user profile - and this project's config resolution looks in
    exactly that place.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOMEDRIVE", tmp_path.anchor.rstrip("\\/") or "C:")
    monkeypatch.setenv("HOMEPATH", str(home).replace(tmp_path.anchor.rstrip("\\/"), "") or "\\home")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.delenv("VIDEO_INTEL_ENV_FILE", raising=False)
    for name in ("VIDEO_INTEL_DATA_DIR", "VIDEO_INTEL_PROJECT_NAME", "VIDEO_INTEL_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("VIDEO_INTEL_DB_PATH", "VIDEO_INTEL_DB_JOURNAL_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name in API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    SECRETS.clear()
    yield
    SECRETS.clear()


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "127.0.0.0", "0.0.0.0", ""}


class NetworkAccessDenied(RuntimeError):
    """Raised when a test tries to open a real network connection."""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail loudly if a unit test reaches for the network (§20).

    Loopback is allowed so tests can start a local server; everything else is blocked.
    ``@pytest.mark.integration`` opts out entirely: the default suite runs offline.
    """
    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _denied(self: socket.socket, address: object) -> None:
        host = str(address[0] if isinstance(address, tuple) else address)
        if host in LOOPBACK_HOSTS:
            # Loopback is allowed: tests may start a local server (the web UI).
            return original_connect(self, address)  # type: ignore[arg-type]
        raise NetworkAccessDenied(
            f"unit tests must not open network connections (attempted {host}); "
            "inject a fake source or use httpx.MockTransport"
        )

    socket.socket.connect = _denied  # type: ignore[method-assign]
    socket.socket.connect_ex = _denied  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def invoke(runner: CliRunner) -> Any:
    def _invoke(*args: str) -> Any:
        return runner.invoke(app, list(args))

    return _invoke


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def project_paths(data_dir: Path) -> ProjectPaths:
    return ProjectPaths(data_dir, "default").ensure()


@pytest.fixture
def transcript_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "synthetic_transcript.json").read_text(encoding="utf-8"))


@pytest.fixture
def transcript(transcript_payload: dict[str, Any]) -> RawTranscript:
    return RawTranscript.model_validate(transcript_payload)


@pytest.fixture
def extraction_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "synthetic_extraction.json").read_text(encoding="utf-8"))


@pytest.fixture
def extraction_response(extraction_payload: dict[str, Any]) -> ExtractionResponse:
    return ExtractionResponse.model_validate(extraction_payload)


@pytest.fixture
def config(data_dir: Path) -> AppConfig:
    from video_intel.config.loader import load_config

    return load_config(project_dir=Path("/nonexistent"), overrides={"data_dir": str(data_dir)})


@pytest.fixture
def chunks(transcript: RawTranscript) -> ChunkSet:
    """Two synthetic chunks over the fixture transcript, with a shared segment."""
    first_ids = [segment.segment_id for segment in transcript.segments[:6]]
    second_ids = [segment.segment_id for segment in transcript.segments[4:]]
    return ChunkSet(
        video_id=transcript.video_id,
        config=ChunkingConfig(),
        transcript_hash="fixture",
        chunks=[
            Chunk(
                chunk_id="chunk_0001",
                index=0,
                start_seconds=0.0,
                end_seconds=128.75,
                text=" ".join(segment.text for segment in transcript.segments[:6]),
                segment_ids=first_ids,
            ),
            Chunk(
                chunk_id="chunk_0002",
                index=1,
                start_seconds=99.0,
                end_seconds=330.0,
                text=" ".join(segment.text for segment in transcript.segments[4:]),
                segment_ids=second_ids,
                overlaps_previous=True,
            ),
        ],
    )


@pytest.fixture
def ollama_model() -> dict[str, str]:
    return {"provider": str(ProviderKind.OLLAMA), "provider_name": "ollama", "model": "llama3.1:8b"}


@pytest.fixture
def pipeline_id() -> PipelineId:
    return PipelineId.FACT


@pytest.fixture
def clean_registry() -> Iterator[None]:
    yield
    SECRETS.clear()
