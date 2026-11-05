"""Shared fixtures: isolated environment and a hard network guard."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from reel_vault.logging_setup import SECRETS

FIXTURES = Path(__file__).parent / "fixtures"

API_KEY_VARS = ("GROQ_API_KEY",)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    for name in (
        "REEL_VAULT_DATA_DIR",
        "REEL_VAULT_DB_PATH",
        "REEL_VAULT_MEDIA_DIR",
        "REEL_VAULT_LOG_LEVEL",
        "REEL_VAULT_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    SECRETS.clear()
    yield
    SECRETS.clear()


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "127.0.0.0", "0.0.0.0", ""}


class NetworkAccessDenied(RuntimeError):
    """Raised when a unit test tries to open a real network connection."""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _denied(self: socket.socket, address: object) -> None:
        host = str(address[0] if isinstance(address, tuple) else address)
        if host in LOOPBACK_HOSTS:
            return original_connect(self, address)  # type: ignore[arg-type]
        raise NetworkAccessDenied(
            f"unit tests must not open network connections (attempted {host}); "
            "inject a fake fetcher or use httpx.MockTransport"
        )

    socket.socket.connect = _denied  # type: ignore[method-assign]
    socket.socket.connect_ex = _denied  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
