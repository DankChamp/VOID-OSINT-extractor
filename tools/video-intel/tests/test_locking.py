"""Job locking: one run per video, stale locks recovered (§15, §19)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from video_intel.storage.locking import LOCK_FILENAME, LockError, lock_owner, video_lock


def test_lock_is_created_and_released(tmp_path: Path) -> None:
    with video_lock(tmp_path) as path:
        assert path.is_file()
        assert path.name == LOCK_FILENAME
        assert lock_owner(path)
    assert not path.exists()


def test_another_live_process_is_refused(tmp_path: Path) -> None:
    """A second run from a different, live process must be told to wait."""
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = tmp_path / LOCK_FILENAME
        path.write_text(f"{helper.pid} {time.time():.0f} analyze\n", encoding="utf-8")
        with pytest.raises(LockError) as excinfo, video_lock(tmp_path):
            pass
        assert "already working" in str(excinfo.value)
        assert str(helper.pid) in str(excinfo.value)
    finally:
        helper.terminate()
        helper.wait(timeout=10)


def test_lock_is_free_again_after_release(tmp_path: Path) -> None:
    with video_lock(tmp_path):
        pass
    with video_lock(tmp_path):
        assert (tmp_path / LOCK_FILENAME).is_file()


def test_lock_of_a_dead_process_is_stale(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILENAME
    path.write_text("999999 0 analyze\n", encoding="utf-8")
    with video_lock(tmp_path):
        assert path.is_file()


def test_lock_with_a_corrupt_owner_is_reclaimed(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILENAME
    path.write_text("not-a-pid", encoding="utf-8")
    with video_lock(tmp_path):
        assert path.is_file()


def test_reentrant_lock_in_the_same_process_is_allowed(tmp_path: Path) -> None:
    """Nesting inside one process must not deadlock: the outer holder owns it."""
    with video_lock(tmp_path):
        with video_lock(tmp_path) as nested:
            assert nested.is_file()
        # The outer holder is still in place after the nested call returns.
        assert (tmp_path / LOCK_FILENAME).is_file()
    assert not (tmp_path / LOCK_FILENAME).exists()


def test_a_foreign_lock_from_a_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILENAME
    path.write_text(f"{os.getpid()} {time.time():.0f} analyze\n", encoding="utf-8")
    # Same pid but not held by us any more: the file is stale and may be replaced.
    with video_lock(tmp_path) as held:
        assert held.is_file()


def test_lock_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "video"
    with video_lock(target):
        assert target.is_dir()


def test_lock_file_names_the_purpose(tmp_path: Path) -> None:
    with video_lock(tmp_path, purpose="notebooklm") as path:
        assert "notebooklm" in path.read_text(encoding="utf-8")


def test_lock_owner_is_none_when_free(tmp_path: Path) -> None:
    assert lock_owner(tmp_path / LOCK_FILENAME) is None
