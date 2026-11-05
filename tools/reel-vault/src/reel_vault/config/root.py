"""Project root discovery.

The workspace root, not the tool directory, owns ``data/``, ``.env`` and
``reel-vault.yaml``. Walking upwards for the workspace marker first keeps a checkout
usable from any subdirectory and after it has been moved or copied.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT_MARKER_FILE = ".void-osint-root"

PROJECT_NAMES = frozenset({"reel-vault", "reel_vault"})
WORKSPACE_NAMES = frozenset({"void-osint-extractor", "void_osint_extractor"})

_TOOL_MARKERS: tuple[tuple[str, str], ...] = (
    ("file", "reel-vault.yaml"),
    ("file", "reel-vault.yml"),
    ("file", os.path.join("config", "reel-vault.yaml")),
    ("pyproject", "pyproject.toml"),
    ("dir", ".git"),
)

MAX_WALK_LEVELS = 64
_MAX_PYPROJECT_BYTES = 1_000_000


def _declared_project_name(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_PYPROJECT_BYTES:
            return None
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return None
    name = data.get("project", {}).get("name")
    return name if isinstance(name, str) else None


def _matches(directory: Path, kind: str, name: str, *, allowed: frozenset[str]) -> bool:
    candidate = directory / name
    if kind == "dir":
        return candidate.is_dir()
    if not candidate.is_file():
        return False
    if kind == "pyproject":
        return _declared_project_name(candidate) in allowed
    return True


def _walk_all(
    start: Path, markers: tuple[tuple[str, str], ...], allowed: frozenset[str]
) -> Path | None:
    current = start
    for _ in range(MAX_WALK_LEVELS):
        for kind, name in markers:
            if _matches(current, kind, name, allowed=allowed):
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def find_project_root(start: Path | str | None = None) -> Path:
    """Return the directory holding the workspace data, or ``start`` when none exists."""
    try:
        current = Path(start).expanduser() if start is not None else Path.cwd()
        current = current.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - unreadable cwd
        return Path(start or ".").absolute()

    if current.is_file():
        current = current.parent

    workspace = _walk_all(current, (("file", ROOT_MARKER_FILE),), allowed=frozenset())
    if workspace is None:
        workspace = _walk_all(current, (("pyproject", "pyproject.toml"),), allowed=WORKSPACE_NAMES)
    if workspace is not None:
        return workspace

    found = _walk_all(current, _TOOL_MARKERS, allowed=PROJECT_NAMES)
    if found is not None:
        return found

    return Path(start).expanduser().absolute() if start is not None else Path.cwd().absolute()


def resolve_path(value: Path | str, *, base: Path | str | None = None) -> Path:
    """Resolve ``value`` against ``base`` when it is relative; ``~`` is expanded."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    root = Path(base) if base is not None else find_project_root()
    return (root / candidate).absolute()


__all__ = [
    "MAX_WALK_LEVELS",
    "PROJECT_NAMES",
    "ROOT_MARKER_FILE",
    "WORKSPACE_NAMES",
    "find_project_root",
    "resolve_path",
]
