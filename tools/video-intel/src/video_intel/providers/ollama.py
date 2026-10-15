"""Ollama adapter (§7).

Uses Ollama's supported HTTP API: ``POST /api/chat`` with ``format: "json"`` when a
structured answer is requested. Local inference means transcript text never leaves the
machine, which is why Ollama is the default provider.
"""

from __future__ import annotations

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

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaProvider:
    """Adapter for a local Ollama server."""

    kind = ProviderKind.OLLAMA.value
    local = True

    def __init__(
        self,
        *,
        name: str = "ollama",
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._owns_client = client is None
        self.gate = RateLimitGate()

    def model_ref(self, model: str | None = None) -> ModelRef:
        return ModelRef(
            provider=ProviderKind.OLLAMA,
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

    async def list_models(self) -> list[str]:
        """Model ids the local server reports; empty when it is not running."""
        try:
            response = await self._http().get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama is not reachable at {self.base_url}: {exc}") from exc
        payload = response.json()
        return [str(entry.get("name", "")) for entry in payload.get("models", [])]

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        async def call() -> ProviderResponse:
            body: dict[str, Any] = {
                "model": request.model or self.model,
                "messages": self._messages(request),
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    **({"num_predict": request.max_tokens} if request.max_tokens else {}),
                },
            }
            if request.response_format == "json":
                body["format"] = "json"
            if request.stop:
                body["options"]["stop"] = list(request.stop)

            started = time.monotonic()
            try:
                response = await self._http().post(f"{self.base_url}/api/chat", json=body)
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(f"ollama timed out after {self.timeout}s") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"ollama request failed: {exc}", retryable=True) from exc

            self._raise_for_status(response)
            payload = response.json()
            return ProviderResponse(
                text=str((payload.get("message") or {}).get("content", "")),
                model=str(payload.get("model") or body["model"]),
                provider=self.name,
                provider_kind=self.kind,
                prompt_tokens=int(payload.get("prompt_eval_count") or 0),
                completion_tokens=int(payload.get("eval_count") or 0),
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
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = response.text[:300]
        if response.status_code == 429:
            raise ProviderRateLimited(
                f"ollama is busy (429): {detail}", retry_after=_retry_after(response)
            )
        raise ProviderError(
            f"ollama returned an error: {detail}",
            retryable=response.status_code >= 500,
            status=response.status_code,
        )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "OllamaProvider"]
