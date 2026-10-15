"""Logging setup with mandatory secret redaction (§17, §19).

Two rules are enforced mechanically, not by convention:

* Known secret values (API keys read from the environment) are replaced with
  ``***REDACTED***`` in every log record, whatever handler is attached.
* Patterns that look like credentials (``sk-...``, ``Bearer ...``, ``api_key=...``)
  are scrubbed even when the value is unknown to this process.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

REDACTION = "***REDACTED***"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
JSON_LOG_FORMAT = (
    '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|token|secret|password)\b\s*[:=]\s*\"?'?([^\s\"',}]+)"
    ),
)


class SecretRegistry:
    """Holds secret values that must never appear in output."""

    def __init__(self) -> None:
        self._values: set[str] = set()

    def add(self, value: str | None) -> None:
        if value and len(value) >= 6:
            self._values.add(value)

    def add_many(self, values: list[str | None]) -> None:
        for value in values:
            self.add(value)

    def scrub(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, REDACTION)
        for pattern in _SECRET_PATTERNS:
            if pattern.groups >= 2:
                text = pattern.sub(lambda m: f"{m.group(1)}={REDACTION}", text)
            else:
                text = pattern.sub(REDACTION, text)
        return text

    def clear(self) -> None:
        self._values.clear()


SECRETS = SecretRegistry()


class RedactionFilter(logging.Filter):
    """Scrubs secrets from messages and arguments before any handler writes them."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = SECRETS.scrub(record.msg)
        if record.args:
            record.args = _scrub_args(record.args)
        for key in ("exc_info", "stack_info"):
            setattr(record, key, None)
        return True


def _scrub_args(
    args: tuple[object, ...] | Mapping[str, object] | None,
) -> tuple[object, ...] | Mapping[str, object] | None:
    """Redact string arguments, leaving other types untouched.

    Converting non-string arguments to strings would break ``%d`` and friends in the
    caller's format string.
    """
    if isinstance(args, dict):
        return {
            key: SECRETS.scrub(value) if isinstance(value, str) else value
            for key, value in args.items()
        }
    if isinstance(args, tuple):
        return tuple(SECRETS.scrub(value) if isinstance(value, str) else value for value in args)
    return args


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Configuration happens once in ``setup_logging``."""
    return logging.getLogger(name)


def setup_logging(
    *,
    level: str = "INFO",
    log_format: str = "text",
    log_file: Path | None = None,
    quiet: bool = False,
    force: bool = True,
) -> logging.Logger:
    """Configure root logging for the CLI. Returns the ``video_intel`` logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(numeric_level)
    redaction = RedactionFilter()

    if not quiet:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(
            logging.Formatter(LOG_FORMAT if log_format != "json" else JSON_LOG_FORMAT)
        )
        stream.addFilter(redaction)
        root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT if log_format != "json" else JSON_LOG_FORMAT)
        )
        file_handler.addFilter(redaction)
        root.addHandler(file_handler)

    if not root.handlers:
        root.addHandler(logging.NullHandler())

    return logging.getLogger("video_intel")


def log_context(logger: logging.Logger, **fields: Any) -> None:
    """Log a stage-scoped line with stable key=value context."""
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(rendered)
