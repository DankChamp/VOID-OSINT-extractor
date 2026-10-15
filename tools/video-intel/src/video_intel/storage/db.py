"""SQLite processing state (§15).

SQLite is the index for projects, videos, cached stages and generated artifacts.
The authoritative content stays in the JSON artifacts on disk; this database makes
them queryable and supports cache invalidation and resume.

Three properties matter for using the same ``data/`` directory on more than one device,
and each is handled explicitly below:

* **The schema is always present.** Any command that touches the database first ensures
  the tables exist and are migrated. A copy of ``state.db`` taken while a run was killed
  can contain zero tables, and previously that surfaced as a bare
  ``OperationalError: no such table: stages`` traceback.
* **WAL is used only where it is safe.** Write-ahead logging needs shared-memory
  support that network filesystems do not reliably provide. The journal mode is
  verified after being set, and falls back to a rollback journal with a longer busy
  timeout when the filesystem cannot support it.
* **The database is self-contained when idle.** Committed WAL frames are checkpointed
  on close, so copying ``state.db`` on its own - to a USB stick, a sync client, or
  another machine - carries the data with it instead of producing an empty file.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from video_intel.schemas.common import StageId, StageState, utc_now
from video_intel.schemas.manifest import StageManifest, StageRecord

SCHEMA_VERSION = 2

#: Seconds SQLite waits for a competing writer before raising "database is locked".
#: Network storage adds tens or hundreds of milliseconds per operation, so the 5 second
#: library default is not a safe floor.
BUSY_TIMEOUT_MS = 30_000

JOURNAL_MODE_ENV = "VIDEO_INTEL_DB_JOURNAL_MODE"
DB_PATH_ENV = "VIDEO_INTEL_DB_PATH"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config_hash TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    project_id       TEXT NOT NULL,
    video_id         TEXT NOT NULL,
    url              TEXT,
    title            TEXT,
    duration_seconds REAL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (project_id, video_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stages (
    project_id       TEXT NOT NULL,
    video_id         TEXT NOT NULL,
    stage_id         TEXT NOT NULL,
    state            TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    input_hash       TEXT,
    output_hash      TEXT,
    config_hash      TEXT,
    started_at       TEXT,
    completed_at     TEXT,
    duration_seconds REAL,
    error            TEXT,
    detail           TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, video_id, stage_id),
    FOREIGN KEY (project_id, video_id) REFERENCES videos(project_id, video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    project_id      TEXT NOT NULL,
    video_id        TEXT NOT NULL,
    stage_id        TEXT NOT NULL,
    filename        TEXT NOT NULL,
    relative_path   TEXT NOT NULL,
    bytes_written   INTEGER NOT NULL DEFAULT 0,
    dataset_hash    TEXT,
    dataset_version INTEGER,
    generated_at    TEXT NOT NULL,
    PRIMARY KEY (project_id, video_id, stage_id, filename),
    FOREIGN KEY (project_id, video_id) REFERENCES videos(project_id, video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_stages_state ON stages(state);
CREATE INDEX IF NOT EXISTS idx_videos_project ON videos(project_id);
"""

# Schemas this code can upgrade in place. Each entry lists the statements that bring that
# version up to the next one; they are applied in order and the stored version is then
# rewritten. Keeping the table explicit means an upgrade path exists for every future
# change, rather than assuming the file was created by the current release.
#
# v1 -> v2 needs no statement: a database written by v1 and copied without its
# write-ahead log can be missing tables entirely, and re-running the idempotent
# CREATE IF NOT EXISTS script above fills them in.
_MIGRATIONS: dict[int, tuple[str, ...]] = {1: (), 2: ()}


class StateStoreError(RuntimeError):
    """The state database cannot be opened or is unusable."""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def resolve_db_path(default: Path) -> Path:
    """Where the state database lives.

    ``VIDEO_INTEL_DB_PATH`` lets the index be kept on local disk while the JSON artifacts
    live on a shared folder. That is the recommended shape when ``data/`` is on a network
    mount or a synced directory, because SQLite wants a local filesystem.
    """
    override = os.environ.get(DB_PATH_ENV)
    return Path(override).expanduser() if override else default


def _requested_journal_mode() -> str | None:
    requested = os.environ.get(JOURNAL_MODE_ENV, "").strip().lower()
    return requested if requested in {"wal", "delete", "truncate", "persist", "memory"} else None


class StateStore:
    """Thin, explicit wrapper over SQLite. No ORM, no hidden behaviour."""

    def __init__(self, path: Path) -> None:
        self.path = resolve_db_path(Path(path))
        self._ensured: set[str] = set()
        self._ensure_lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing
    def _set_journal_mode(self, connection: sqlite3.Connection) -> str:
        """Pick the safest journal mode this filesystem actually supports.

        WAL is requested first because it is much better for the concurrent readers and
        writers a multi-device setup produces. SQLite refuses WAL on filesystems without
        the required shared-memory primitives, and simply keeps the previous mode, so the
        result is read back rather than assumed. ``VIDEO_INTEL_DB_JOURNAL_MODE`` forces a
        mode for troubleshooting.
        """
        requested = _requested_journal_mode()
        for mode in [requested] if requested else ["wal", "delete"]:
            try:
                row = connection.execute(f"PRAGMA journal_mode={mode}").fetchone()
            except sqlite3.DatabaseError:
                continue
            actual = str(row[0]).lower() if row else mode
            if actual == mode:
                return actual
        return "delete"

    def _apply_schema(self, connection: sqlite3.Connection) -> None:
        """Create any missing table, then migrate from the stored version."""
        connection.executescript(SCHEMA_SQL)

        row = connection.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
        try:
            found = int(row[0]) if row else 0
        except (TypeError, ValueError):
            found = 0

        for version in range(found + 1, SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS.get(version, ()):
                connection.execute(statement)

        connection.execute(
            "INSERT INTO schema_info(key, value) VALUES('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )

    def _ensure_ready(self) -> None:
        """Guarantee the schema exists before any query runs.

        Memoised per resolved path: the check costs one indexed lookup per connection,
        and doing it once per process is enough because the schema only ever changes
        through this class.
        """
        key = str(self.path.absolute())
        if key in self._ensured:
            return
        with self._ensure_lock:
            if key in self._ensured:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._apply_pragmas(self.connect())
            try:
                self._apply_schema(connection)
            finally:
                self._checkpoint_and_close(connection)
            self._ensured.add(key)

    def _apply_pragmas(self, connection: sqlite3.Connection) -> sqlite3.Connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._set_journal_mode(connection)
        return connection

    @staticmethod
    def _checkpoint_and_close(connection: sqlite3.Connection) -> None:
        """Fold the WAL back into the main file so a copy of it is complete.

        Without this, a database left behind by an interrupted run keeps its schema and
        recent writes in ``state.db-wal``. Copying only ``state.db`` to another machine
        then yields a valid-looking file containing no tables at all.
        """
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()
            if mode and str(mode[0]).lower() == "wal":
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:  # pragma: no cover - best effort
            pass
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            timeout=BUSY_TIMEOUT_MS / 1000,
        )
        connection.row_factory = sqlite3.Row
        return self._apply_pragmas(connection)

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """Connection context manager with real commit/rollback handling.

        The connection runs in autocommit mode, so an explicit transaction is opened here
        rather than left to the driver; otherwise a failure partway through a multi-statement
        block would leave earlier statements committed.
        """
        self._ensure_ready()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.DatabaseError as exc:  # pragma: no cover - lock timeout
            self._checkpoint_and_close(connection)
            raise StateStoreError(
                f"cannot open a transaction on {self.path}: {exc}. "
                "If this directory is on a network share, point VIDEO_INTEL_DB_PATH at "
                "local disk."
            ) from exc
        try:
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            # The rollback is best effort: a connection killed mid-statement may already
            # be unusable, and the original exception is the one worth reporting.
            with contextlib.suppress(sqlite3.DatabaseError):
                connection.execute("ROLLBACK")
            raise
        finally:
            self._checkpoint_and_close(connection)

    def initialize(self) -> None:
        """Create or upgrade the schema. Safe to call repeatedly."""
        self._ensured.clear()
        self._ensure_ready()

    def checkpoint(self) -> None:
        """Fold the WAL into the main database file.

        Call before copying ``data/`` to another device, or shutting down a machine that
        shares the directory, so the copy is complete even if it contains only
        ``state.db``.
        """
        self._ensure_ready()
        connection = self.connect()
        try:
            self._checkpoint_and_close(connection)
        except sqlite3.DatabaseError:  # pragma: no cover - best effort
            connection.close()

    def schema_version(self) -> int | None:
        """The stored schema version, or ``None`` when the file has never been created."""
        if not self.path.is_file():
            return None
        try:
            connection = self.connect()
        except sqlite3.DatabaseError:
            return None
        try:
            row = connection.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
            return int(row[0]) if row else None
        except (sqlite3.DatabaseError, TypeError, ValueError):
            return None
        finally:
            self._checkpoint_and_close(connection)

    # ----------------------------------------------------------------- projects
    def upsert_project(
        self, project_id: str, name: str, *, config_hash: str | None = None, now: str
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO projects(project_id, name, config_hash, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name=excluded.name,
                    config_hash=excluded.config_hash,
                    updated_at=excluded.updated_at
                """,
                (project_id, name, config_hash, now, now),
            )

    def list_projects(self) -> list[sqlite3.Row]:
        with self.session() as connection:
            return list(connection.execute("SELECT * FROM projects ORDER BY project_id").fetchall())

    # ------------------------------------------------------------------- videos
    def upsert_video(
        self,
        project_id: str,
        video_id: str,
        *,
        url: str | None = None,
        title: str | None = None,
        duration_seconds: float | None = None,
        now: str,
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO videos(project_id, video_id, url, title, duration_seconds,
                                   created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, video_id) DO UPDATE SET
                    url=excluded.url,
                    title=COALESCE(excluded.title, videos.title),
                    duration_seconds=COALESCE(excluded.duration_seconds, videos.duration_seconds),
                    updated_at=excluded.updated_at
                """,
                (project_id, video_id, url, title, duration_seconds, now, now),
            )

    def video_ids(self, project_id: str) -> list[str]:
        with self.session() as connection:
            rows = connection.execute(
                "SELECT video_id FROM videos WHERE project_id=? ORDER BY video_id",
                (project_id,),
            ).fetchall()
        return [row["video_id"] for row in rows]

    # ------------------------------------------------------------------- stages
    def upsert_stage(self, project_id: str, video_id: str, record: StageRecord) -> None:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO stages(project_id, video_id, stage_id, state, attempts,
                                   input_hash, output_hash, config_hash, started_at,
                                   completed_at, duration_seconds, error, detail)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, video_id, stage_id) DO UPDATE SET
                    state=excluded.state,
                    attempts=excluded.attempts,
                    input_hash=excluded.input_hash,
                    output_hash=excluded.output_hash,
                    config_hash=excluded.config_hash,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    duration_seconds=excluded.duration_seconds,
                    error=excluded.error,
                    detail=excluded.detail
                """,
                (
                    project_id,
                    video_id,
                    record.stage_id.value,
                    record.state.value,
                    record.attempts,
                    record.input_hash,
                    record.output_hash,
                    record.config_hash,
                    record.started_at.isoformat() if record.started_at else None,
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.duration_seconds,
                    record.error,
                    json.dumps(record.detail, sort_keys=True),
                ),
            )

    def record_stage(
        self,
        project_id: str,
        video_id: str,
        stage_id: StageId,
        state: StageState,
        *,
        attempts: int = 0,
        input_hash: str | None = None,
        output_hash: str | None = None,
        config_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Convenience wrapper used by the pipeline runner (Phase 5+)."""
        now = utc_now()
        record = StageRecord(
            stage_id=stage_id,
            state=state,
            attempts=attempts,
            input_hash=input_hash,
            output_hash=output_hash,
            config_hash=config_hash,
            started_at=now if state is StageState.RUNNING else None,
            completed_at=now if state in {StageState.COMPLETED, StageState.FAILED} else None,
            error=error,
        )
        self.upsert_stage(project_id, video_id, record)

    def load_manifest(self, project_id: str, video_id: str) -> StageManifest:
        """Rebuild the stage manifest from the database."""
        manifest = StageManifest(video_id=video_id)
        with self.session() as connection:
            rows = connection.execute(
                "SELECT * FROM stages WHERE project_id=? AND video_id=? ORDER BY stage_id",
                (project_id, video_id),
            ).fetchall()
        for row in rows:
            manifest.stages.append(
                StageRecord(
                    stage_id=StageId(row["stage_id"]),
                    state=StageState(row["state"]),
                    attempts=row["attempts"],
                    input_hash=row["input_hash"],
                    output_hash=row["output_hash"],
                    config_hash=row["config_hash"],
                    started_at=_parse_dt(row["started_at"]),
                    completed_at=_parse_dt(row["completed_at"]),
                    duration_seconds=row["duration_seconds"],
                    error=row["error"],
                )
            )
        return manifest

    # ----------------------------------------------------------------- artifacts
    def record_artifact(
        self,
        project_id: str,
        video_id: str,
        stage_id: StageId,
        filename: str,
        relative_path: str,
        *,
        bytes_written: int,
        dataset_hash: str | None = None,
        dataset_version: int | None = None,
        now: str,
    ) -> None:
        if PurePath(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
            raise ValueError(
                f"relative_path must be relative to the video directory, got {relative_path!r}"
            )
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(project_id, video_id, stage_id, filename,
                                      relative_path, bytes_written, dataset_hash,
                                      dataset_version, generated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, video_id, stage_id, filename) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    bytes_written=excluded.bytes_written,
                    dataset_hash=excluded.dataset_hash,
                    dataset_version=excluded.dataset_version,
                    generated_at=excluded.generated_at
                """,
                (
                    project_id,
                    video_id,
                    stage_id.value,
                    filename,
                    relative_path,
                    bytes_written,
                    dataset_hash,
                    dataset_version,
                    now,
                ),
            )

    def artifacts_for(self, project_id: str, video_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM artifacts WHERE project_id=? AND video_id=? "
                    "ORDER BY stage_id, filename",
                    (project_id, video_id),
                ).fetchall()
            )


def db_diagnostics(path: Path) -> dict[str, Any]:
    """Facts about a state database, for ``doctor`` and for troubleshooting.

    Reports the schema version and journal mode without raising, because the point of
    this function is to explain a broken or copied database rather than to fail.
    """
    resolved = resolve_db_path(Path(path))
    report: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else 0,
        "schema_version": None,
        "journal_mode": None,
        "sidecars": sorted(
            suffix
            for suffix in ("-wal", "-shm")
            if resolved.with_name(resolved.name + suffix).is_file()
        ),
        "error": None,
    }
    if not report["exists"]:
        return report
    try:
        connection = sqlite3.connect(resolved, timeout=5)
        try:
            report["journal_mode"] = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
            row = connection.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
            report["schema_version"] = int(row[0]) if row else None
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)
    return report


__all__ = [
    "BUSY_TIMEOUT_MS",
    "DB_PATH_ENV",
    "JOURNAL_MODE_ENV",
    "SCHEMA_VERSION",
    "StateStore",
    "StateStoreError",
    "db_diagnostics",
    "resolve_db_path",
]
