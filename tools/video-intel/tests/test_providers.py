"""Provider adapters and the registry (§7), exercised without network access."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from video_intel.config.loader import load_config
from video_intel.config.models import AppConfig, ProviderConfig
from video_intel.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderRequest,
    ProviderTimeout,
    RateLimitGate,
    extract_json,
    strip_reasoning,
    with_retries,
)
from video_intel.providers.ollama import OllamaProvider
from video_intel.providers.openai_compatible import OpenAICompatibleProvider
from video_intel.providers.registry import ProviderRegistry
from video_intel.schemas.common import ProviderKind

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def ollama_client(handler: Any) -> OllamaProvider:
    return OllamaProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def chat_client(handler: Any, **kwargs: Any) -> OpenAICompatibleProvider:
    defaults: dict[str, Any] = {
        "name": "deepseek",
        "kind": ProviderKind.DEEPSEEK,
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-test-key-value",
        "model": "deepseek-chat",
        "max_retries": 0,
        "client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    }
    return OpenAICompatibleProvider(**{**defaults, **kwargs})


def chat_body(content: str) -> dict[str, Any]:
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


# ------------------------------------------------------------------ json handling
def test_extract_json_from_plain_json() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block() -> None:
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_extract_json_from_surrounding_prose() -> None:
    assert extract_json('Here you go:\n{"a": 3}\nHope that helps!') == {"a": 3}


def test_extract_json_from_array() -> None:
    assert extract_json('[{"a": 4}]') == [{"a": 4}]


def test_extract_json_strips_reasoning() -> None:
    assert strip_reasoning("<think>hmm</think>answer") == "answer"
    assert extract_json('<think>x</think>{"a": 5}') == {"a": 5}


def test_extract_json_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="no JSON value"):
        extract_json("no json at all")


def test_extract_json_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty model response"):
        extract_json("   ")


# ---------------------------------------------------------------------- retries
async def test_retry_succeeds_after_a_transient_failure() -> None:
    calls = {"n": 0}
    delays: list[float] = []

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("boom", retryable=True)
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    result = await with_retries(flaky, attempts=3, base_delay=0.01, sleep=fake_sleep)
    assert result == "ok"
    assert calls["n"] == 3
    assert len(delays) == 2 and all(delay > 0 for delay in delays)


async def test_retry_stops_on_non_retryable_error() -> None:
    calls = {"n": 0}

    async def fatal() -> str:
        calls["n"] += 1
        raise ProviderError("bad request", retryable=False)

    with pytest.raises(ProviderError, match="bad request"):
        await with_retries(fatal, attempts=5, base_delay=0.01, sleep=_no_sleep)
    assert calls["n"] == 1


async def test_retry_gives_up_after_attempts() -> None:
    calls = {"n": 0}

    async def always() -> str:
        calls["n"] += 1
        raise ProviderError("still failing", retryable=True)

    with pytest.raises(ProviderError, match="still failing"):
        await with_retries(always, attempts=2, base_delay=0.01, sleep=_no_sleep)
    assert calls["n"] == 2


async def test_retry_after_header_wins_over_backoff() -> None:
    delays: list[float] = []

    async def limited() -> str:
        raise ProviderRateLimited("slow down", retry_after=1.5)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(ProviderRateLimited):
        await with_retries(limited, attempts=2, base_delay=10.0, sleep=fake_sleep)
    assert delays == [1.5]


async def _no_sleep(delay: float) -> None:
    return None


# ----------------------------------------------------------------------- ollama
async def test_ollama_chat_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llama3.1:8b",
                "message": {"content": '{"items": []}'},
                "prompt_eval_count": 120,
                "eval_count": 15,
            },
        )

    provider = ollama_client(handler)
    response = await provider.complete(
        ProviderRequest(prompt="chunk text", model="llama3.1:8b", system="be strict")
    )

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["body"]["format"] == "json"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["messages"][1]["content"] == "chunk text"
    assert response.text == '{"items": []}'
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 15
    assert response.provider == "ollama"


async def test_ollama_model_ref_identifies_the_provider() -> None:
    ref = ollama_client(lambda r: httpx.Response(200, json={})).model_ref()
    assert ref.provider is ProviderKind.OLLAMA
    assert ref.provider_name == "ollama"
    assert ref.model == "llama3.1:8b"


async def test_ollama_reports_http_errors() -> None:
    provider = ollama_client(lambda r: httpx.Response(400, text="bad model"))
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(ProviderRequest(prompt="x", model="m"))
    assert excinfo.value.status == 400
    assert "bad model" in excinfo.value.message


async def test_ollama_reports_rate_limit_as_retryable() -> None:
    provider = ollama_client(
        lambda r: httpx.Response(429, text="busy", headers={"retry-after": "2"})
    )
    with pytest.raises(ProviderRateLimited) as excinfo:
        await provider.complete(ProviderRequest(prompt="x", model="m"))
    assert excinfo.value.retry_after == 2.0


async def test_ollama_transport_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = ollama_client(handler)
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(ProviderRequest(prompt="x", model="m"))
    assert excinfo.value.retryable is True


async def test_ollama_timeout_is_reported_as_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    provider = ollama_client(handler)
    with pytest.raises(ProviderTimeout):
        await provider.complete(ProviderRequest(prompt="x", model="m"))


async def test_ollama_lists_local_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]}
        )

    models = await ollama_client(handler).list_models()
    assert models == ["llama3.1:8b", "qwen2.5:7b"]


async def test_ollama_unreachable_server_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="not reachable"):
        await ollama_client(handler).list_models()


async def test_ollama_closes_only_clients_it_created() -> None:
    injected = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    provider = OllamaProvider(client=injected)
    await provider.aclose()
    assert not injected.is_closed  # injected clients belong to the caller


# -------------------------------------------------------- openai-compatible
async def test_chat_request_carries_bearer_key_and_json_mode() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=chat_body('{"items": [1]}'))

    provider = chat_client(handler)
    response = await provider.complete(
        ProviderRequest(prompt="hello", model="deepseek-chat", max_tokens=512, temperature=0.2)
    )

    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-key-value"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["max_tokens"] == 512
    assert seen["body"]["temperature"] == 0.2
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7


async def test_chat_without_api_key_fails_before_the_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be sent without a key")

    provider = chat_client(handler, api_key=None)
    with pytest.raises(ProviderError, match="needs an API key"):
        await provider.complete(ProviderRequest(prompt="x", model="m"))


async def test_chat_error_body_is_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "invalid model", "type": "invalid_request"}}
        )

    with pytest.raises(ProviderError) as excinfo:
        await chat_client(handler).complete(ProviderRequest(prompt="x", model="m"))
    assert "invalid model" in excinfo.value.message
    assert excinfo.value.retryable is False


async def test_chat_server_error_is_retryable() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await chat_client(lambda r: httpx.Response(503, text="down")).complete(
            ProviderRequest(prompt="x", model="m")
        )
    assert excinfo.value.retryable is True


async def test_chat_rate_limit_is_retryable() -> None:
    with pytest.raises(ProviderRateLimited):
        await chat_client(
            lambda r: httpx.Response(429, json={"error": {"message": "slow"}})
        ).complete(ProviderRequest(prompt="x", model="m"))


async def test_chat_rate_limit_reads_the_delay_from_the_body() -> None:
    """Groq states the delay in the message, not in a header."""
    from video_intel.providers.openai_compatible import _retry_hint_from_text

    detail = (
        "Rate limit reached on tokens per minute. Please try again in 12.5s. "
        "Upgrade to Dev Tier at https://console.groq.com"
    )
    assert _retry_hint_from_text(detail) == 12.5
    assert _retry_hint_from_text("no delay mentioned") is None

    with pytest.raises(ProviderRateLimited) as excinfo:
        await chat_client(
            lambda r: httpx.Response(429, json={"error": {"message": detail}})
        ).complete(ProviderRequest(prompt="x", model="m"))
    assert excinfo.value.retry_after == 12.5


async def test_chat_empty_choices_yields_empty_text() -> None:
    response = await chat_client(lambda r: httpx.Response(200, json={"choices": []})).complete(
        ProviderRequest(prompt="x", model="m")
    )
    assert response.text == ""


# -------------------------------------------------------------------- registry
@pytest.fixture
def hybrid_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-registry-test-key")
    return load_config(
        project_dir=tmp_path / "none",
        overrides={"execution_mode": "hybrid", "data_dir": str(tmp_path)},
    )


def test_registry_lists_implemented_providers(config: AppConfig) -> None:
    registry = ProviderRegistry(config)
    assert "ollama" in registry.available()
    assert "gemini" not in registry.available()


def test_registry_builds_ollama(config: AppConfig) -> None:
    provider = ProviderRegistry(config).build("ollama")
    assert isinstance(provider, OllamaProvider)
    assert provider.local is True


def test_registry_refuses_remote_provider_under_local_only(config: AppConfig) -> None:
    monkey = pytest.MonkeyPatch()
    monkey.setenv("OPENAI_API_KEY", "sk-test")
    try:
        with pytest.raises(ProviderNotConfigured, match="local_only"):
            ProviderRegistry(config).build("openai")
    finally:
        monkey.undo()


def test_registry_builds_cloud_provider_when_allowed(hybrid_config: AppConfig) -> None:
    provider = ProviderRegistry(hybrid_config).build("openai")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.local is False
    assert provider.base_url == "https://api.openai.com/v1"


def test_registry_caches_instances(hybrid_config: AppConfig) -> None:
    registry = ProviderRegistry(hybrid_config)
    assert registry.build("ollama") is registry.build("ollama")


def test_registry_reports_unimplemented_adapters(hybrid_config: AppConfig) -> None:
    with pytest.raises(ProviderNotConfigured, match="no adapter"):
        ProviderRegistry(hybrid_config).build("gemini")


def test_registry_reports_notebooklm_as_manual_handoff(hybrid_config: AppConfig) -> None:
    with pytest.raises(ProviderNotConfigured, match="notebooklm_handoff"):
        ProviderRegistry(hybrid_config).build("notebooklm_handoff")


def test_registry_rejects_unknown_provider(config: AppConfig) -> None:
    with pytest.raises(ProviderNotConfigured, match="unknown provider"):
        ProviderRegistry(config).build("nope")


def test_registry_rejects_disabled_provider(config: AppConfig) -> None:
    disabled = ProviderConfig(name="ollama_off", kind=ProviderKind.OLLAMA, enabled=False)
    config.providers["ollama_off"] = disabled
    with pytest.raises(ProviderNotConfigured, match="disabled"):
        ProviderRegistry(config).build("ollama_off")


def test_registry_requires_key_for_cloud_provider(config: AppConfig) -> None:
    config.execution_mode = "hybrid"  # type: ignore[assignment]
    with pytest.raises(ProviderNotConfigured, match="needs an API key"):
        ProviderRegistry(config).build("openai")


def test_registry_model_ref_does_not_need_a_key(config: AppConfig) -> None:
    assert ProviderRegistry(config).model_ref("deepseek") == (
        "deepseek",
        "deepseek",
        "deepseek-chat",
    )


async def test_registry_closes_instances(config: AppConfig) -> None:
    registry = ProviderRegistry(config)
    registry.build("ollama")
    await registry.aclose()
    await registry.aclose()  # idempotent


def test_request_fingerprint_changes_with_content() -> None:
    a = ProviderRequest(prompt="one", model="m")
    b = ProviderRequest(prompt="two", model="m")
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == ProviderRequest(prompt="one", model="m").fingerprint()


# ------------------------------------------------------------------ rate limits
async def test_gate_makes_parallel_callers_wait_after_a_429() -> None:
    """One rate-limited call must pause the whole provider, not just itself."""
    gate = RateLimitGate(minimum_delay=0.05)
    order: list[str] = []

    seen: dict[str, int] = {}

    async def call(name: str, fails_first: bool) -> str:
        await gate.wait()
        seen[name] = seen.get(name, 0) + 1
        order.append(f"{name}-start")
        if fails_first and seen[name] == 1:
            raise ProviderRateLimited("slow down", retry_after=0.05)
        return name

    async def fake_sleep(delay: float) -> None:
        order.append(f"sleep-{delay:.2f}")
        await asyncio.sleep(0)

    results = await asyncio.gather(
        with_retries(lambda: call("a", True), attempts=2, gate=gate, sleep=fake_sleep),
        with_retries(lambda: call("b", False), attempts=1, gate=gate, sleep=fake_sleep),
    )
    assert results == ["a", "b"]
    assert any(entry.startswith("sleep-") for entry in order)
    assert gate.penalties >= 1


async def test_gate_is_inert_without_penalties() -> None:
    gate = RateLimitGate()
    await gate.wait()
    assert gate.penalties == 0


async def test_provider_exposes_a_gate() -> None:
    provider = ollama_client(lambda r: httpx.Response(200, json={}))
    assert isinstance(provider.gate, RateLimitGate)
    chat = chat_client(lambda r: httpx.Response(200, json=chat_body("{}")))
    assert isinstance(chat.gate, RateLimitGate)


async def test_rate_limited_call_opens_a_cooldown() -> None:
    handler_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        handler_calls["n"] += 1
        if handler_calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "try again in 0.05s"}})
        return httpx.Response(200, json=chat_body('{"items": []}'))

    provider = chat_client(handler, max_retries=3)

    async def no_sleep(delay: float) -> None:
        await asyncio.sleep(0)

    response = await with_retries(
        lambda: provider.complete(ProviderRequest(prompt="x", model="m")),
        attempts=3,
        sleep=no_sleep,
        gate=provider.gate,
    )
    assert response.text == '{"items": []}'
    assert provider.gate.penalties == 1
