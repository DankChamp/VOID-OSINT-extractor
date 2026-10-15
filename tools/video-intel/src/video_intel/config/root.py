"""Project root discovery and portable path resolution.

Everything this application reads or writes is relative to a *project root*, not to the
process working directory. That distinction is what makes the tool usable from any
directory, from any device, and after the checkout has been moved or copied:

* ``data_dir``, ``.env`` and ``video-intel.yaml`` are resolved against the root found by
  walking upwards from the working directory;
* a relative ``data_dir`` in configuration is therefore interpreted identically whether
  the user typed ``video-intel analyze`` from the workspace root or from a subdirectory;
* on Windows the working directory is not stable across shells and launchers, so
  anchoring to a discovered root removes an entire class of "empty data directory"
  reports.

The walk is done in two passes, because in a multi-tool workspace the directory that
*contains this tool* is not the directory that *holds the user's data*. The workspace
root - with its ``data/``, its ``.env`` and its ``video-intel.yaml`` - is the one that
matters, and it is always further up than ``tools/<name>/``.

1. **Workspace pass.** Looking only for a workspace marker: the empty marker file
   ``.void-osint-root``, or a ``pyproject.toml`` that declares the workspace itself.
   This pass is authoritative and runs first, so ``tools/video-intel/pyproject.toml``
   can never win and strand the data directory inside the tool.
2. **Standalone pass.** Looking for this tool's own markers: ``video-intel.yaml``, a
   ``pyproject.toml`` naming this tool, or a plain ``.git`` checkout. This keeps a
   single tool working when it is copied out of the workspace on its own.

When neither pass finds anything the working directory itself is the root, which keeps
the behaviour of an ad-hoc ``cd``-and-run session sensible.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

#: Explicit, empty marker declaring the root of a checkout.
ROOT_MARKER_FILE = ".void-osint-root"

#: Distribution and package names that identify this tool.
PROJECT_NAMES = frozenset({"video-intel", "video_intel"})

#: Distribution name that identifies the multi-tool workspace.
WORKSPACE_NAMES = frozenset({"void-osint-extractor", "void_osint_extractor"})

_MAX_PYPROJECT_BYTES = 1_000_000

#: Tool-level markers, checked nearest-first during the standalone pass.
_TOOL_MARKERS: tuple[tuple[str, str], ...] = (
    ("file", "video-intel.yaml"),
    ("file", "video-intel.yml"),
    ("file", os.path.join("config", "video-intel.yaml")),
    ("pyproject", "pyproject.toml"),
    ("dir", ".git"),
)

MAX_WALK_LEVELS = 64


def _declared_project_name(path: Path) -> str | None:
    """The ``[project].name`` a pyproject.toml declares, or ``None``.

    Parsed rather than pattern-matched: the workspace root's pyproject mentions every
    tool by name while belonging to none of them, so a substring test would let an
    unrelated workspace capture the root.
    """
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
    """Nearest-first search: at each directory, try every marker in priority order.

    Distance is the primary key and marker type only breaks ties inside one directory,
    which is what makes "the nearest directory containing a marker wins" true rather
    than "the highest-priority marker anywhere above wins".
    """
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
    """Return the directory holding this workspace's data, or ``start`` if there is none.

    Args:
        start: directory to start walking from. Defaults to the working directory.

    Both searches stop at the first directory containing a match and are bounded, so a
    pathological mount point cannot spin the lookup. The function never raises: a failure
    to walk upwards simply yields the starting directory.
    """
    try:
        current = Path(start).expanduser() if start is not None else Path.cwd()
        current = current.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - unreadable cwd
        return Path(start or ".").absolute()

    if current.is_file():
        current = current.parent

    # Pass 1: the workspace root wins outright, so the tool's own directory cannot
    # capture it and strand data/ inside tools/<name>/.
    workspace = _walk_all(
        current, ((("file", ROOT_MARKER_FILE),)), allowed=frozenset()
    ) or _walk_all(current, (("pyproject", "pyproject.toml"),), allowed=WORKSPACE_NAMES)
    if workspace is not None:
        return workspace

    # Pass 2: this tool's own markers, so a standalone checkout still works.
    found = _walk_all(current, _TOOL_MARKERS, allowed=PROJECT_NAMES)
    if found is not None:
        return found

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


__all__ = [
    "MAX_WALK_LEVELS",
    "PROJECT_NAMES",
    "ROOT_MARKER_FILE",
    "WORKSPACE_NAMES",
    "find_project_root",
    "resolve_path",
]
