"""Minimal ``.env`` loader.

Credentials live in the environment, not in configuration files. This loads an optional
``.env`` file so a key does not have to be exported by hand in every shell:

* real environment variables always win;
* missing values are filled in, nothing is overwritten;
* parsing is deliberately dumb - ``KEY=value`` lines, optional ``export`` prefix,
  ``#`` comments, surrounding quotes stripped;
* a broken line is ignored rather than crashing the run;
* the file is never written by this project and is git-ignored.

No third-party dependency is added for this (§18).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"
MAX_LINE_LENGTH = 8192

#: How far up the tree the search for a ``.env`` may walk before giving up.
MAX_UPWARD_SEARCH = 64


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines into a mapping."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate ``.env``: explicit override, then the nearest one up the tree.

    The search walks upwards from the starting directory and stops at the first ``.env``
    it finds, which is the behaviour every other dotenv loader has and the one people
    expect when running a command from a subdirectory. A ``.env`` is not treated as a
    project-root marker: in a monorepo it would anchor the whole tree to whichever
    package happens to contain one.
    """
    explicit = os.environ.get("VIDEO_INTEL_ENV_FILE")
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    try:
        current = (start or Path.cwd()).resolve()
    except (OSError, RuntimeError):  # pragma: no cover - unreadable cwd
        return None

    if current.is_file():
        current = current.parent

    for _ in range(MAX_UPWARD_SEARCH):
        candidate = current / ENV_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_env_file(start: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into the environment and return what was applied.

    Args:
        override: when False (default), variables already set are left untouched.
    """
    path = find_env_file(start)
    if path is None:
        return {}

    try:
        if path.stat().st_size > MAX_LINE_LENGTH * 64:  # pragma: no cover - sanity guard
            return {}
        values = parse_env_text(path.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:  # pragma: no cover - unreadable file
        return {}

    applied: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


__all__ = [
    "ENV_FILENAME",
    "MAX_UPWARD_SEARCH",
    "find_env_file",
    "load_env_file",
    "parse_env_text",
]
