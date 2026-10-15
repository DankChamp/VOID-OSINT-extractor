"""Project root discovery and portable path resolution.

Everything this application reads or writes is relative to a *project root*, not to the
process working directory. That distinction is what makes the tool usable from any
directory, from any device, and after the checkout has been moved or copied:

* ``data_dir``, ``.env`` and ``video-intel.yaml`` are resolved against the root found by
  walking upwards from the working directory;
* a relative ``data_dir`` in configuration is therefore interpreted identically whether
  the user typed ``video-intel analyze`` from the project root or from a subdirectory;
* on Windows the working directory is not stable across shells and launchers, so
  anchoring to a discovered root removes an entire class of "empty data directory"
  reports.

The walk looks for the first of these markers in each directory, starting at ``start``
and moving towards the filesystem root:

1. ``.video-intel-root``  - an explicit, empty marker file;
2. ``video-intel.yaml`` / ``video-intel.yml`` / ``config/video-intel.yaml``;
3. ``pyproject.toml`` that names this project;
4. ``.git`` - a plain checkout, as a last resort.

When no marker is found the working directory itself is the root, which keeps the
behaviour of an ad-hoc ``cd``-and-run session sensible.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_MARKER_FILE = ".video-intel-root"
PROJECT_NAME_TOKEN = "video-intel"

# Ordered marker candidates. The first entry that exists wins.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("file", ROOT_MARKER_FILE),
    ("file", "video-intel.yaml"),
    ("file", "video-intel.yml"),
    ("file", os.path.join("config", "video-intel.yaml")),
    ("pyproject", "pyproject.toml"),
    ("dir", ".git"),
)

MAX_WALK_LEVELS = 64


def _is_project_pyproject(path: Path) -> bool:
    """A ``pyproject.toml`` counts only when it actually describes this project."""
    try:
        if path.stat().st_size > 1_000_000:
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return PROJECT_NAME_TOKEN in text or "video_intel" in text


def _marker_matches(directory: Path, kind: str, name: str) -> bool:
    candidate = directory / name
    if kind == "dir":
        return candidate.is_dir()
    if not candidate.is_file():
        return False
    if kind == "pyproject":
        return _is_project_pyproject(candidate)
    return True


def find_project_root(start: Path | str | None = None) -> Path:
    """Return the nearest enclosing project root, or ``start`` itself if there is none.

    Args:
        start: directory to start walking from. Defaults to the working directory.

    The search stops at the first directory containing a marker. It is bounded, so a
    pathological mount point cannot spin the lookup, and it never raises: a failure to
    walk upwards simply yields the starting directory.
    """
    try:
        current = Path(start).expanduser() if start is not None else Path.cwd()
        current = current.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - unreadable cwd
        return Path(start or ".").absolute()

    if current.is_file():
        current = current.parent

    for _ in range(MAX_WALK_LEVELS):
        for kind, name in _MARKERS:
            if _marker_matches(current, kind, name):
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path(start).expanduser().absolute() if start is not None else Path.cwd().absolute()


def resolve_path(value: Path | str, *, base: Path | str | None = None) -> Path:
    """Resolve ``value`` against ``base`` when it is relative.

    ``~`` is expanded first, so ``~/video-intel-data`` works identically on every
    platform. Absolute values are returned unchanged apart from normalisation.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    root = Path(base) if base is not None else find_project_root()
    return (root / candidate).absolute()


__all__ = ["ROOT_MARKER_FILE", "find_project_root", "resolve_path"]
