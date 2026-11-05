"""External-drive safety.

Media may live on a USB disk that can be unplugged mid-run. Before touching the media
root the pipeline proves it is present and writable; if it is not, it waits briefly for
the drive to come back and then stops cleanly instead of scattering partial files.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from reel_vault.media.base import DriveUnavailable

PROBE_PREFIX = ".reel-vault-probe-"


def check_reachable(media_dir: Path) -> bool:
    """Return True when ``media_dir`` exists and accepts a temporary file."""
    try:
        if not media_dir.is_dir():
            return False
        with tempfile.NamedTemporaryFile(dir=media_dir, prefix=PROBE_PREFIX, delete=True):
            return True
    except OSError:
        return False


def require_reachable(
    media_dir: Path,
    *,
    wait_seconds: float = 60.0,
    interval_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until the drive is back, or raise ``DriveUnavailable`` after the wait."""
    deadline = clock() + max(0.0, wait_seconds)
    while True:
        if check_reachable(media_dir):
            return
        remaining = deadline - clock()
        if remaining <= 0:
            raise DriveUnavailable(
                f"media directory is not reachable: {media_dir}. "
                "Reconnect the drive and run the same command again."
            )
        sleep(min(interval_seconds, remaining))


__all__ = ["DriveUnavailable", "check_reachable", "require_reachable"]
