"""Minimal ``.env`` reader.

Keys are read at call time and never written to logs or artifacts. Values already
present in the process environment always win.
"""

from __future__ import annotations

import os
from pathlib import Path

from reel_vault.config.root import find_project_root


def env_file_candidates(root: Path | None = None) -> list[Path]:
    base = root if root is not None else find_project_root()
    candidates = [base / ".env"]
    explicit = os.environ.get("REEL_VAULT_ENV_FILE")
    if explicit:
        candidates.insert(0, Path(explicit).expanduser())
    return candidates


def load_env_file(path: Path | None = None) -> Path | None:
    """Load the first existing env file without overwriting real environment values."""
    candidates = [path] if path is not None else env_file_candidates()
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return candidate
    return None


__all__ = ["env_file_candidates", "load_env_file"]
