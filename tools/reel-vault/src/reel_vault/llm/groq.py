"""Groq chat completions with retries, JSON mode and cost accounting."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from reel_vault.config.models import GroqConfig
from reel_vault.llm.base import ChatError, ChatMessage, ChatResult, ChatUsage

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_ERROR_BODY = 500


def _usage(payload: dict[str, Any], config: GroqConfig) -> ChatUsage:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return ChatUsage()
    try:
        input_tokens = int(raw.get("prompt_tokens") or 0)
        output_tokens = int(raw.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return ChatUsage()
    usd = (
        input_tokens * config.chat_input_usd_per_million
        + output_tokens * config.chat_output_usd_per_million
    ) / 1_000_000
    return ChatUsage(input_tokens=input_tokens, output_tokens=output_tokens, usd=round(usd, 8))


def _content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatError("Groq returned no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ChatError("Groq returned a malformed choice")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ChatError("Groq returned no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ChatError("Groq returned an empty message")
    return content


class GroqChatClient:
    def __init__(
        self,
        config: GroqConfig,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.api_key = api_key or os.environ.get(config.api_key_env)
        self.transport = transport
        self.sleep = sleep

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _body(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        json_mode: bool,
        temperature: float,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> ChatResult:
        if not self.api_key:
            raise ChatError(
                f"{self.config.api_key_env} is not set; this stage needs a Groq key",
                retryable=False,
            )
        if not messages:
            raise ChatError("cannot send an empty conversation", retryable=False)

        attempts = self.config.max_retries + 1
        last_error = "Groq request failed"
        body = self._body(messages, model=model, json_mode=json_mode, temperature=temperature)

        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(
                    timeout=self.config.timeout_seconds, transport=self.transport
                ) as client:
                    response = client.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "User-Agent": "reel-vault/0.1 (+personal archive)",
                        },
                        json=body,
                    )
            except httpx.RequestError as exc:
                last_error = f"Groq request failed: {exc}"
                if attempt < attempts:
                    self._sleep(attempt)
                    continue
                raise ChatError(last_error) from exc

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ChatError("Groq returned malformed JSON") from exc
                if not isinstance(payload, dict):
                    raise ChatError("Groq returned an unexpected payload")
                return ChatResult(
                    text=_content(payload),
                    model=model,
                    usage=_usage(payload, self.config),
                )

            detail = response.text[:MAX_ERROR_BODY]
            last_error = f"Groq HTTP {response.status_code}: {detail}"
            if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                self._sleep(attempt)
                continue
            raise ChatError(
                last_error,
                retryable=response.status_code in RETRYABLE_STATUS,
                status_code=response.status_code,
            )

        raise ChatError(last_error)

    def _sleep(self, attempt: int) -> None:
        self.sleep(self.config.retry_backoff_seconds * (2.0 ** (attempt - 1)))


__all__ = ["RETRYABLE_STATUS", "GroqChatClient"]
