"""Integration tests against a real local Ollama server (§9).

These are the only tests that talk to a model. They are skipped unless a local Ollama
server is reachable and a model is present, so the default suite stays offline and fast:

    ollama serve
    ollama pull qwen2.5:0.5b
    VIDEO_INTEL_TEST_MODEL=qwen2.5:0.5b uv run pytest -m integration
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from video_intel.config.loader import load_config
from video_intel.extraction.engine import ExtractionEngine, ExtractionSettings
from video_intel.pipeline.chunking import chunk_transcript
from video_intel.providers.base import ProviderRequest, extract_json
from video_intel.providers.ollama import OllamaProvider
from video_intel.schemas.chunk import ChunkingConfig
from video_intel.schemas.common import PipelineId
from video_intel.schemas.transcript import TranscriptSegment

pytestmark = pytest.mark.integration

MODEL_ENV = "VIDEO_INTEL_TEST_MODEL"
DEFAULT_TEST_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

SEGMENTS = [
    TranscriptSegment(
        segment_id="segment_0001",
        start_seconds=0.0,
        end_seconds=9.0,
        text=(
            "Risk assessment is the process of identifying threats and evaluating their "
            "likelihood of occurring."
        ),
    ),
    TranscriptSegment(
        segment_id="segment_0002",
        start_seconds=9.0,
        end_seconds=18.0,
        text="It also evaluates the potential impact of those threats.",
    ),
]


def _reachable() -> bool:
    import httpx

    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0).raise_for_status()
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason="no local Ollama server is reachable"),
]

MODEL = os.environ.get(MODEL_ENV, DEFAULT_TEST_MODEL)


def _available_models() -> set[str]:
    import httpx

    try:
        payload = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0).json()
    except Exception:
        return set()
    return {str(entry.get("name", "")).split(":")[0] for entry in payload.get("models", [])}


requires_model = pytest.mark.skipif(
    MODEL.split(":")[0] not in _available_models(),
    reason=f"{MODEL} is not pulled; run: ollama pull {MODEL}",
)


def _provider() -> OllamaProvider:
    return OllamaProvider(base_url=OLLAMA_URL, model=MODEL, timeout=180.0, max_retries=0)


def test_server_lists_models() -> None:
    """Whatever is installed, the server must answer /api/tags."""
    names = {name.split(":")[0] for name in asyncio.run(_list_models())}
    assert isinstance(names, set)


@requires_model
def test_model_is_available() -> None:
    """The configured test model must really be present on the server."""
    names = {name.split(":")[0] for name in asyncio.run(_list_models())}
    assert MODEL.split(":")[0] in names, (
        f"{MODEL} is not pulled (available: {sorted(names)}); run: ollama pull {MODEL}"
    )


async def _list_models() -> list[str]:
    provider = _provider()
    try:
        return await provider.list_models()
    finally:
        await provider.aclose()


async def _complete(provider: OllamaProvider, request: ProviderRequest) -> str:
    try:
        response = await provider.complete(request)
        return response.text
    finally:
        await provider.aclose()


@requires_model
def test_provider_returns_json_for_the_contract() -> None:
    request = ProviderRequest(
        prompt=(
            'Answer with JSON only: {"items": [{"type": "fact", "statement": "...", '
            '"source_segment_ids": ["segment_0001"], "details": [], "uncertainties": []}]}'
        ),
        model=MODEL,
    )
    payload = extract_json(asyncio.run(_complete(_provider(), request)))
    assert isinstance(payload, dict)
    assert "items" in payload


@requires_model
def test_extraction_end_to_end_against_a_real_model() -> None:
    chunk_set = chunk_transcript(
        SEGMENTS,
        ChunkingConfig(max_chars=4000, overlap_chars=0, min_chars=100),
        video_id="synthetic01",
    )
    provider = _provider()
    engine = ExtractionEngine(
        provider,
        model=MODEL,
        video_id="synthetic01",
        settings=ExtractionSettings(allowed_types={PipelineId.FACT: ()}, concurrency=1),
    )
    engine.bind_transcript(SEGMENTS)

    async def run_and_close() -> object:
        try:
            return await engine.run_pipeline(PipelineId.DEFINITION, chunk_set)
        finally:
            await provider.aclose()

    outcome = asyncio.run(run_and_close())

    assert outcome.records, "the pipeline must attempt every chunk"
    record = outcome.records[0]
    assert record.raw_response, "raw model output is always stored"

    if record.parsed is None:
        # A small local model may fail the contract; that must be reported, not hidden.
        assert record.error
        return

    for item in record.parsed.items:
        assert item.source_segment_ids, "every grounded item cites a segment"
        assert set(item.source_segment_ids) <= {s.segment_id for s in SEGMENTS}
        if item.provenance is not None:
            assert item.provenance.start_seconds is not None


def test_invalid_provider_does_not_crash_the_engine() -> None:
    provider = OllamaProvider(
        base_url="http://127.0.0.1:1", model=MODEL, timeout=2.0, max_retries=0
    )
    chunk_set = chunk_transcript(
        SEGMENTS,
        ChunkingConfig(max_chars=4000, overlap_chars=0, min_chars=100),
        video_id="synthetic01",
    )
    engine = ExtractionEngine(provider, model=MODEL, video_id="synthetic01")
    engine.bind_transcript(SEGMENTS)

    async def run_and_close() -> object:
        try:
            return await engine.run_pipeline(PipelineId.FACT, chunk_set)
        finally:
            await provider.aclose()

    outcome = asyncio.run(run_and_close())
    assert outcome.ok_records == []
    assert outcome.failures


def test_configured_local_provider_builds() -> None:
    config = load_config(
        project_dir=Path("/nonexistent"), overrides={"execution_mode": "local_only"}
    )
    provider = config.provider("ollama")
    assert provider.is_local() is True
