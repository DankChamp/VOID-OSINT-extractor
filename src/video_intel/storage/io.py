"""Atomic, safe JSON I/O for project artifacts (§15).

Writes go to a temporary file in the destination directory and are then renamed, so
an interrupted run can never leave a half-written ``knowledge.json`` behind.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

#: Bounded wait for a rename that lost a race with a Windows file scanner.
REPLACE_RETRY_SECONDS = 2.0
REPLACE_RETRY_INTERVAL = 0.05


def dumps(payload: Any, *, indent: int = 2) -> str:
    """Canonical JSON: readable for humans, stable key order for hashing."""
    return json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=False, default=str)


def write_bytes_atomic(path: Path, data: bytes) -> int:
    """Atomically write bytes; returns the number of bytes written.

    The temporary file lives in the destination directory so the final rename never
    crosses a filesystem boundary, which is what makes it atomic. The containing
    directory is flushed afterwards, so the rename itself survives a power loss rather
    than only the file contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _apply_default_permissions(tmp_path)
        _replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return len(data)


def _apply_default_permissions(path: Path) -> None:
    """Give an artifact the process umask rather than leaving it owner-only.

    ``mkstemp`` creates files as ``0600``. Artifacts are the portable, shareable output
    of this tool, and a data directory that only its creator can read defeats the point of
    copying it to another account or another machine.
    """
    if os.name == "nt":  # pragma: no cover - no POSIX mode bits
        return
    try:
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(path, 0o666 & ~mask)
    except OSError:  # pragma: no cover - filesystem without POSIX modes
        pass


def _replace(source: Path, target: Path) -> None:
    """``os.replace`` with a retry for transient Windows sharing violations.

    On Windows a rename fails with ``PermissionError`` whenever anything still holds a
    handle on the target - an antivirus scanner, an indexer, or a concurrent run. Those
    are transient, so a short bounded retry turns a spurious stage failure into a wait.
    """
    deadline = time.monotonic() + REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(REPLACE_RETRY_INTERVAL)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a rename is durable. Best effort by design."""
    if os.name == "nt":  # pragma: no cover - directories cannot be opened on Windows
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem without directory fsync
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - not supported everywhere
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> int:
    return write_bytes_atomic(path, text.encode(encoding))


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> int:
    return write_text_atomic(path, dumps(payload, indent=indent) + "\n")


def write_model_atomic(path: Path, model: BaseModel, *, indent: int = 2) -> int:
    return write_json_atomic(path, model.model_dump(mode="json"), indent=indent)


def read_json(path: Path) -> Any:
    """Read JSON from disk, raising a clear error for missing or malformed files.

    Reads with ``utf-8-sig``, which accepts plain UTF-8 and UTF-8 with a byte-order mark.
    Artifacts are frequently opened and re-saved in a Windows editor, and a default
    ``utf-8`` read turns that into an uncaught ``UnicodeDecodeError`` on the next run.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise OSError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


def read_model(path: Path, model: type[ModelT]) -> ModelT:
    """Read and validate a JSON file into a pydantic model."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}, got {type(payload).__name__}")
    validated: ModelT = model.model_validate(payload)
    return validated


ModelT = TypeVar("ModelT", bound=BaseModel)
