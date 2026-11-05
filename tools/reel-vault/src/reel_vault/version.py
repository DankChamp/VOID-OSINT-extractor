"""Package version, resolved from installed metadata when available."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reel-vault")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
