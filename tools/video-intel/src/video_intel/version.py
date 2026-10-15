"""Single source of truth for the package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("video-intel")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
