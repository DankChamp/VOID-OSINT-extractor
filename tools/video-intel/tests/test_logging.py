"""Logging must never leak secrets (§17)."""

from __future__ import annotations

import logging
from pathlib import Path

from video_intel.logging_setup import REDACTION, SECRETS, SecretRegistry, setup_logging


def _capture(caplog: object, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(f"test.{id(caplog)}")
    logger.setLevel(level)
    return logger


def test_registered_secret_is_redacted(caplog: object) -> None:
    SECRETS.add("sk-live-abcdef123456")
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("calling api with key sk-live-abcdef123456 now")
    assert "sk-live-abcdef123456" not in caplog.text  # type: ignore[attr-defined]
    assert REDACTION in caplog.text  # type: ignore[attr-defined]


def test_unknown_sk_style_keys_are_redacted(caplog: object) -> None:
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("using sk-proj-AAAABBBBCCCCDDDD for the request")
    assert "sk-proj-AAAABBBBCCCCDDDD" not in caplog.text  # type: ignore[attr-defined]


def test_authorization_headers_are_redacted(caplog: object) -> None:
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("sending Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc")
    assert "eyJhbGciOiJIUzI1NiJ9.abc" not in caplog.text  # type: ignore[attr-defined]


def test_api_key_equals_values_are_redacted(caplog: object) -> None:
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("config api_key=supersecretvalue123")
    assert "supersecretvalue123" not in caplog.text  # type: ignore[attr-defined]
    assert "api_key=" in caplog.text  # type: ignore[attr-defined]


def test_secret_registry_ignores_short_values() -> None:
    registry = SecretRegistry()
    registry.add("abc")
    assert registry.scrub("abc") == "abc"


def test_secret_registry_scrubs_multiple_values() -> None:
    registry = SecretRegistry()
    registry.add_many(["secret-value-one", None, "secret-value-two"])
    scrubbed = registry.scrub("first=secret-value-one second=secret-value-two")
    assert "secret-value" not in scrubbed


def test_redaction_filter_handles_args(caplog: object) -> None:
    SECRETS.add("registered-secret-value")
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("key=%s", "registered-secret-value")
    assert "registered-secret-value" not in caplog.text  # type: ignore[attr-defined]


def test_numeric_arguments_survive_redaction(caplog: object) -> None:
    """Redaction must not stringify args: %d would break otherwise."""
    SECRETS.add("another-secret-value")
    logger = _capture(caplog)
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        logger.info("stage=%s bytes=%d", "chunks", 2048)
    assert "bytes=2048" in caplog.text  # type: ignore[attr-defined]


def test_setup_logging_writes_to_file_without_secrets(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "app.log"
    SECRETS.add("secret-in-file-value")
    logger = setup_logging(level="INFO", log_file=log_file, quiet=True)
    logger.info("token is secret-in-file-value")
    for handler in logging.getLogger().handlers:
        handler.flush()
    contents = log_file.read_text(encoding="utf-8")
    assert "secret-in-file-value" not in contents
    assert REDACTION in contents


def test_setup_logging_quiet_suppresses_console(capsys: object) -> None:
    logger = setup_logging(level="INFO", quiet=True)
    logger.info("should not appear on console")
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "should not appear on console" not in captured.err
