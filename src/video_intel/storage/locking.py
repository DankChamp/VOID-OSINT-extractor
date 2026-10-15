"""Per-video job lock (§15, §19).

Two concurrent runs on the same video would race on artifact writes and stage records.
The lock is advisory and self-healing: it is created exclusively, carries the owning
process id and host, and is released when the owning process ends, so a crashed run does
not block the next one forever.

The lock has to survive the same ``data/`` directory being shared between machines, which
drives most of the design here:

* a **host identity** is recorded alongside the pid, because a bare pid means nothing on
  another device - pid 4321 on one laptop is usually some unrelated process on the next;
* **liveness probing is platform-correct**; on Windows ``os.kill(pid, 0)`` raises
  ``EINVAL`` for every pid, which previously made every pid look alive;
* when the owner is a *different, live* host, liveness cannot be probed at all, so only a
  generous age threshold is used to reclaim the lock;
* the in-process "already held" set is keyed by the same resolved path used everywhere
  else, so a process cannot delete its own live lock through a differently spelled path.
"""

from __future__ import annotations

import errno
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".lock"
STALE_AFTER_SECONDS = 6 * 60 * 60

#: First token of a lock file written by this release. Earlier releases wrote
#: ``pid started purpose`` with no marker, which would otherwise be misread as a host
#: name; the explicit version tag makes the two layouts unambiguous.
LOCK_FORMAT = "v2"

#: A lock written by a *different* host can only be judged by age, because its pid cannot
#: be probed. The threshold is far longer than a normal run so that a shared folder does
#: not hand the same video to two machines a minute apart.
FOREIGN_STALE_AFTER_SECONDS = 24 * 60 * 60

# Locks held by this process, so the same process may nest safely instead of
# deadlocking against itself.
_HELD: set[str] = set()
_HELD_LOCK = threading.Lock()


class LockError(RuntimeError):
    """Another run already holds the lock for this video."""


def _boot_time() -> float:
    """Unix time at which this machine last booted, or ``0.0`` when unknown."""
    if os.name == "posix":
        try:
            with open("/proc/uptime", encoding="ascii") as handle:
                uptime = float(handle.read().split()[0])
            return time.time() - uptime
        except (OSError, ValueError, IndexError):
            return 0.0
    return 0.0


def host_id() -> str:
    """A short identifier for this machine and boot.

    Stable across the whole session, which is exactly the lifetime a lock file needs to
    be interpreted: a pid recorded before a reboot must not be trusted afterwards.
    """
    return f"{socket.gethostname()}:{int(_boot_time())}"


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process on *this* machine.

    On Windows a signal-0 probe is not available - CPython's ``os.kill`` rejects every
    signal other than the console control events - so the process handle is queried
    directly. Falling back to "alive" on an unknown error is deliberate: it errs towards
    not stealing a lock that might still be held.
    """
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return True
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _read_owner(path: Path) -> tuple[int | None, float, str]:
    """Parse a lock file into ``(pid, started_at, host_id)``.

    Understands both layouts. A file starting with the ``v2`` marker is
    ``v2 pid started host purpose`` and carries a real host id. An unmarked legacy file
    is ``pid started purpose``; because those could only have been written by a run on
    this same machine, its pid is probed locally, exactly as before.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip().split()
    except OSError:
        return None, 0.0, ""

    if raw and raw[0] == LOCK_FORMAT:
        rest = raw[1:]
        legacy = False
    else:
        rest = raw
        legacy = True

    try:
        pid = int(rest[0]) if rest else None
    except ValueError:
        return None, 0.0, ""
    try:
        started = float(rest[1]) if len(rest) > 1 else 0.0
    except ValueError:
        started = 0.0

    if legacy:
        # Same machine, unknown boot: probe the pid the way the original code did.
        return pid, started, host_id() if pid is not None else ""
    return pid, started, rest[2] if len(rest) > 2 else ""


def _lock_key(path: Path) -> str:
    """A single spelling for a lock path, used consistently for the held-lock set."""
    try:
        return str(path.resolve())
    except OSError:  # pragma: no cover - unresolvable path
        return str(path.absolute())


def _is_stale(path: Path) -> bool:
    """Whether an existing lock may be taken over.

    The decisions, in order: an unparseable lock is stale; a lock this process already
    holds is stale only under the resolved spelling, so the inner call cannot delete the
    outer one's file; a lock whose pid is dead on this host is stale immediately; a lock
    from another host can only be reclaimed on age.
    """
    pid, started, host = _read_owner(path)
    if pid is None:
        return True
    if host == host_id():
        if pid == os.getpid():
            return _lock_key(path) not in _HELD
        if not _pid_alive(pid):
            return True
        return bool(started) and (time.time() - started) > STALE_AFTER_SECONDS
    # Unknown or foreign host: pid numbers are not ours to interpret.
    return bool(started) and (time.time() - started) > FOREIGN_STALE_AFTER_SECONDS


@contextmanager
def video_lock(directory: Path, *, purpose: str = "analyze") -> Iterator[Path]:
    """Hold an exclusive lock for one video directory.

    Raises:
        LockError: when another live process holds the lock.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_FILENAME
    key = _lock_key(path)

    if key in _HELD:
        # Re-entrant call from the same process: the outer holder already owns it.
        yield path
        return

    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _is_stale(path):
                path.unlink(missing_ok=True)
                continue
            pid, started, host = _read_owner(path)
            owner_host = host.split(":", 1)[0] if host else "another host"
            raise LockError(
                f"another {purpose} run is already working on this video "
                f"(process {pid or 'unknown'} on {owner_host}"
                + (
                    f", since {time.strftime('%H:%M:%S', time.localtime(started))}"
                    if started
                    else ""
                )
                + f"); wait for it or remove {path}"
            ) from None
        else:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(f"{LOCK_FORMAT} {os.getpid()} {time.time():.0f} {host_id()} {purpose}\n")
            break

    with _HELD_LOCK:
        _HELD.add(key)
    try:
        yield path
    finally:
        with _HELD_LOCK:
            _HELD.discard(key)
        try:
            pid, _started, host = _read_owner(path)
            if pid in (None, os.getpid()) and host in ("", host_id()):
                path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            pass


def lock_owner(path: Path) -> str | None:
    """Human-readable owner of a lock file, or ``None`` when it is free."""
    if not path.exists():
        return None
    pid, started, host = _read_owner(path)
    if pid is None:
        return None
    if host == host_id():
        location = "this device"
    else:
        location = host.split(":", 1)[0] if host else "unknown device"
    when = time.strftime("%H:%M:%S", time.localtime(started)) if started else "unknown time"
    return f"pid {pid} on {location} since {when}"


__all__ = [
    "FOREIGN_STALE_AFTER_SECONDS",
    "LOCK_FILENAME",
    "STALE_AFTER_SECONDS",
    "LockError",
    "host_id",
    "lock_owner",
    "video_lock",
]
