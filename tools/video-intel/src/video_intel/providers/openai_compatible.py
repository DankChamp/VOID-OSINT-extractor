"""OpenAI-compatible chat-completions adapter (§7).

One implementation serves OpenAI, DeepSeek, Groq, LM Studio, vLLM and any other service
that speaks ``POST {base_url}/chat/completions``. Only the base URL, key environment
variable and default model differ, so those live in configuration rather than in code.

Sending transcript text here sends it to that provider. That only happens when a
pipeline is explicitly bound to it, and ``execution_mode: local_only`` refuses such a
binding at configuration time.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from video_intel.providers.base import (
    ProviderError,
    ProviderRateLimited,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeout,
    RateLimitGate,
    with_retries,
)
from video_intel.schemas.common import ProviderKind
from video_intel.schemas.extraction import ModelRef

DEFAULT_CHAT_PATH = "/chat/completions"


class OpenAICompatibleProvider:
    """Adapter for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        name: str,
        kind: ProviderKind,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        chat_path: str = DEFAULT_CHAT_PATH,
        local: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.kind = kind.value
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat_path = chat_path
        self.local = local
        self._client = client
        self._owns_client = client is None
        self.gate = RateLimitGate()

    def model_ref(self, model: str | None = None) -> ModelRef:
        return ModelRef(
            provider=ProviderKind(self.kind),
            provider_name=self.name,
            model=model or self.model,
        )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                f"provider {self.name!r} needs an API key; set the environment variable "
                "named by api_key_env in the configuration"
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        async def call() -> ProviderResponse:
            body: dict[str, Any] = {
                "model": request.model or self.model,
                "messages": self._messages(request),
                "temperature": request.temperature,
                "stream": False,
            }
            if request.max_tokens:
                body["max_tokens"] = request.max_tokens
            if request.response_format == "json":
                body["response_format"] = {"type": "json_object"}
            if request.stop:
                body["stop"] = list(request.stop)

            started = time.monotonic()
            try:
                response = await self._http().post(
                    f"{self.base_url}{self.chat_path}", json=body, headers=self._headers()
                )
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(f"{self.name} timed out after {self.timeout}s") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.name} request failed: {exc}", retryable=True) from exc

            self._raise_for_status(response)
            payload = response.json()
            return ProviderResponse(
                text=self._text_of(payload),
                model=str(payload.get("model") or body["model"]),
                provider=self.name,
                provider_kind=self.kind,
                prompt_tokens=_usage_field(payload, "prompt_tokens"),
                completion_tokens=_usage_field(payload, "completion_tokens"),
                latency_seconds=time.monotonic() - started,
                raw=payload if isinstance(payload, dict) else {},
            )

        response: ProviderResponse = await with_retries(
            call, attempts=self.max_retries + 1, gate=self.gate
        )
        return response

    @staticmethod
    def _messages(request: ProviderRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    @staticmethod
    def _text_of(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = _error_detail(response)
        if response.status_code == 429:
            # Some providers only state the delay in the message body, not in headers.
            retry_after = _retry_after(response) or _retry_hint_from_text(detail)
            raise ProviderRateLimited(
                f"{self.name} rate limited the request: {detail}",
                retry_after=retry_after,
            )
        raise ProviderError(
            f"{self.name} returned an error: {detail}",
            retryable=response.status_code >= 500 or response.status_code == 408,
            status=response.status_code,
        )


def _usage_field(payload: Any, key: str) -> int:
    if isinstance(payload, dict):
        usage = payload.get("usage") or {}
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        if isinstance(error, str):
            return error[:300]
        return str(payload)[:300]
    return response.text[:300]


RETRY_HINT_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _retry_hint_from_text(detail: str) -> float | None:
    """Pull "please try again in 12.5s" out of an error message."""
    match = RETRY_HINT_RE.search(detail)
    return float(match.group(1)) if match else None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


__all__ = ["DEFAULT_CHAT_PATH", "OpenAICompatibleProvider"]
