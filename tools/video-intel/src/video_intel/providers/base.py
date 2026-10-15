"""Provider abstraction (§7, §4 of Phase 4).

A provider turns a :class:`ProviderRequest` into a :class:`ProviderResponse`. Adapters
speak HTTP; the pipeline only knows this protocol, so a model can be replaced without
touching extraction code.

Nothing here retries silently forever or hides failures: :class:`ProviderError` carries
whether a retry is worthwhile, and :func:`with_retries` applies bounded backoff.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from video_intel.schemas.extraction import ModelRef

DEFAULT_USER_AGENT = "video-intel/0.1"

# Text that models wrap around JSON despite being asked for JSON only.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THOUGHT_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ProviderError(RuntimeError):
    """A provider call failed. ``retryable`` drives the retry policy."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status

    def render(self) -> str:
        suffix = f" (status {self.status})" if self.status else ""
        return f"{self.message}{suffix}"


class ProviderNotConfigured(ProviderError):
    """The provider is missing, disabled, or has no implemented adapter."""


class ProviderTimeout(ProviderError):
    """The provider did not answer in time."""


class ProviderRateLimited(ProviderError):
    """The provider asked us to slow down."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, retryable=True, status=429)
        self.retry_after = retry_after


@dataclass(frozen=True)
class ProviderRequest:
    """One completion request."""

    prompt: str
    model: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    response_format: str | None = "json"
    stop: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(
            {
                "prompt": self.prompt,
                "system": self.system,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": self.response_format,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderResponse:
    """A completion result plus enough metadata to record provenance."""

    text: str
    model: str
    provider: str
    provider_kind: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def json_payload(self) -> Any:
        """Best-effort JSON decode of the response text."""
        return extract_json(self.text)


@runtime_checkable
class Provider(Protocol):
    """Minimal contract every adapter implements."""

    name: str
    kind: str
    local: bool

    def model_ref(self, model: str) -> ModelRef:
        """Identify this provider+model in stored records."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Return a completion or raise :class:`ProviderError`."""

    async def aclose(self) -> None:
        """Release network resources."""


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks some models emit before the answer."""
    return _THOUGHT_RE.sub("", text).strip()


def extract_json(text: str) -> Any:
    """Decode JSON from a model response, tolerating fences and stray prose.

    Raises:
        ValueError: when no JSON value can be recovered.
    """
    cleaned = strip_reasoning(text).strip()
    if not cleaned:
        raise ValueError("empty model response")

    for candidate in _json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON value found in model response: {cleaned[:200]!r}")


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fenced = _FENCE_RE.findall(text)
    candidates.extend(block.strip() for block in fenced)

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    return candidates


class RateLimitGate:
    """Shared cooldown for a provider.

    When one call is told to slow down, the whole run waits until the hinted time
    instead of issuing more requests that are guaranteed to fail. Parallel callers all
    honour the same gate, which is what stops a burst from turning into a cascade of
    failures.
    """

    def __init__(self, *, minimum_delay: float = 1.0, maximum_delay: float = 120.0) -> None:
        self.minimum_delay = minimum_delay
        self.maximum_delay = maximum_delay
        self._until = 0.0
        self._lock = asyncio.Lock()
        self.penalties = 0

    def _now(self) -> float:
        import time

        return time.monotonic()

    def penalize(self, delay: float | None) -> float:
        """Register a cooldown and return the delay that was applied."""
        seconds = max(self.minimum_delay, min(self.maximum_delay, float(delay or 0.0)))
        self._until = max(self._until, self._now() + seconds)
        self.penalties += 1
        return seconds

    async def wait(self, sleep: Any = asyncio.sleep) -> None:
        while True:
            async with self._lock:
                remaining = self._until - self._now()
            if remaining <= 0:
                return
            await sleep(min(remaining, self.maximum_delay))


async def with_retries(
    operation: Any,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Any = asyncio.sleep,
    gate: RateLimitGate | None = None,
) -> Any:
    """Run ``operation`` with bounded exponential backoff and jitter.

    When ``gate`` is given, every attempt waits for any active cooldown, and a rate-limit
    error opens a new one for the hinted duration.

    Raises:
        ProviderError: the last error, once attempts are exhausted.
    """
    last: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        if gate is not None:
            await gate.wait(sleep)
        try:
            return await operation()
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt >= max(1, attempts):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, base_delay / 2)
            retry_after = getattr(exc, "retry_after", None)
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                delay = min(max_delay, float(retry_after))
            if gate is not None and isinstance(exc, ProviderRateLimited):
                gate.penalize(retry_after)
            await sleep(delay)
    if last is not None:  # pragma: no cover - loop always returns or raises
        raise last
    raise ProviderError("retry loop finished without a result")  # pragma: no cover


__all__ = [
    "DEFAULT_USER_AGENT",
    "Provider",
    "ProviderError",
    "ProviderNotConfigured",
    "ProviderRateLimited",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderTimeout",
    "RateLimitGate",
    "extract_json",
    "strip_reasoning",
    "with_retries",
]
