"""Chat protocols shared by triage, extraction and classification.

The pipeline depends on ``ChatClient`` rather than on a provider SDK, so tests inject a
fake and never open a network connection. Every result carries token usage and the
estimated cost, because an overnight run has to be able to stop on a budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatError(RuntimeError):
    """A provider call failed.

    ``retryable`` distinguishes a transient outage from a bad key or a malformed
    request: the first is worth another attempt, the second never is.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
        usage: ChatUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.usage = usage or ChatUsage()


@dataclass
class ChatUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ChatResult:
    text: str
    model: str
    usage: ChatUsage = field(default_factory=ChatUsage)


@runtime_checkable
class ChatClient(Protocol):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Return one assistant message for ``messages``."""
        ...


__all__ = ["ChatClient", "ChatError", "ChatMessage", "ChatResult", "ChatUsage"]
