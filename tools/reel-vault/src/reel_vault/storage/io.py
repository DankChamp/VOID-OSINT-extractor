"""Atomic JSON/text I/O.

Writes go to a temporary file in the destination directory and are then renamed, so an
interrupted run cannot leave a half-written artifact behind.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

REPLACE_RETRY_SECONDS = 2.0
REPLACE_RETRY_INTERVAL = 0.05


def dumps(payload: Any, *, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=False, default=str)


def write_bytes_atomic(path: Path, data: bytes) -> int:
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
    if os.name == "nt":  # pragma: no cover - no POSIX mode bits
        return
    try:
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(path, 0o666 & ~mask)
    except OSError:  # pragma: no cover - filesystem without POSIX modes
        pass


def _replace(source: Path, target: Path) -> None:
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
    if os.name == "nt":  # pragma: no cover
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
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
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 text: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


ModelT = TypeVar("ModelT", bound=BaseModel)


def read_model(path: Path, model: type[ModelT]) -> ModelT:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}, got {type(payload).__name__}")
    return model.model_validate(payload)


__all__ = [
    "dumps",
    "read_json",
    "read_model",
    "write_bytes_atomic",
    "write_json_atomic",
    "write_model_atomic",
    "write_text_atomic",
]
