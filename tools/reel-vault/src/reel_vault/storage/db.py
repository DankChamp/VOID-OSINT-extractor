"""SQLite state.

The authoritative content stays in JSON artifacts on disk. SQLite is the index that
makes the queue queryable, supports resume, and records costs and job state.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from reel_vault.importer import ImportReport
from reel_vault.schemas.common import JobState, ReelStatus, iso_now
from reel_vault.schemas.source import SourceReel

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    counts      TEXT NOT NULL DEFAULT '{}',
    warnings    TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reels (
    shortcode     TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    author        TEXT,
    saved_at      TEXT,
    collections   TEXT NOT NULL DEFAULT '[]',
    caption       TEXT,
    hashtags      TEXT NOT NULL DEFAULT '[]',
    source_import TEXT NOT NULL,
    status        TEXT NOT NULL,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    shortcode    TEXT NOT NULL,
    stage        TEXT NOT NULL,
    state        TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (shortcode, stage)
);

CREATE TABLE IF NOT EXISTS metadata (
    shortcode    TEXT PRIMARY KEY,
    payload      TEXT NOT NULL,
    fetch_status TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media (
    shortcode  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    bytes      INTEGER NOT NULL DEFAULT 0,
    verified   INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    shortcode  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triage (
    shortcode  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    shortcode  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classifications (
    shortcode     TEXT PRIMARY KEY,
    primary_label TEXT,
    secondary     TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0,
    method        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    shortcode  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    shortcode  TEXT,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    units      REAL NOT NULL,
    usd        REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    shortcode    TEXT NOT NULL,
    name         TEXT NOT NULL,
    path         TEXT NOT NULL,
    content_hash TEXT,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (shortcode, name)
);

CREATE INDEX IF NOT EXISTS idx_reels_status ON reels(status);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_costs_shortcode ON costs(shortcode);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_info(key, value) VALUES('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- imports and reels

    def import_report(self, report: ImportReport) -> dict[str, int]:
        now = iso_now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO imports(source, imported_at, counts, warnings) VALUES(?, ?, ?, ?)",
                (
                    report.source,
                    report.imported_at,
                    json.dumps(report.counts),
                    json.dumps(report.warnings),
                ),
            )
            for reel in report.reels:
                self._upsert_reel(reel, now)
        return report.counts

    def _upsert_reel(self, reel: SourceReel, now: str) -> None:
        row = self._conn.execute(
            "SELECT collections, caption, author FROM reels WHERE shortcode = ?", (reel.shortcode,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO reels(
                    shortcode, url, media_type, author, saved_at, collections, caption,
                    hashtags, source_import, status, first_seen, last_seen
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reel.shortcode,
                    reel.url,
                    reel.media_type.value,
                    reel.author,
                    reel.saved_at.isoformat() if reel.saved_at else None,
                    json.dumps(sorted(set(reel.collections))),
                    reel.caption,
                    json.dumps(sorted(set(reel.hashtags))),
                    reel.source_import,
                    ReelStatus.NEW.value,
                    now,
                    now,
                ),
            )
            return

        old_collections = set(json.loads(row["collections"] or "[]"))
        old_caption = row["caption"]
        old_author = row["author"]
        self._conn.execute(
            """
            UPDATE reels
               SET url = ?, media_type = ?, author = ?, saved_at = ?, collections = ?,
                   caption = ?, hashtags = ?, source_import = ?, last_seen = ?
             WHERE shortcode = ?
            """,
            (
                reel.url,
                reel.media_type.value,
                old_author or reel.author,
                reel.saved_at.isoformat() if reel.saved_at else None,
                json.dumps(sorted(old_collections | set(reel.collections))),
                old_caption or reel.caption,
                json.dumps(sorted(set(reel.hashtags))),
                reel.source_import,
                now,
                reel.shortcode,
            ),
        )

    def get_reel(self, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM reels WHERE shortcode = ?", (shortcode,)).fetchone()
        return _row_to_reel(row) if row is not None else None

    def list_reels(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        search: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append(
                "shortcode IN (SELECT shortcode FROM classifications WHERE primary_label = ? "
                "OR secondary LIKE ?)"
            )
            params.extend([category, f'%"{category}"%'])
        if search:
            clauses.append("(author LIKE ? OR caption LIKE ? OR shortcode LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern, pattern])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM reels {where} ORDER BY COALESCE(saved_at, first_seen) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_reel(row) for row in rows]

    def set_status(self, shortcode: str, status: ReelStatus | str) -> None:
        value = status.value if isinstance(status, ReelStatus) else status
        with self._conn:
            self._conn.execute(
                "UPDATE reels SET status = ? WHERE shortcode = ?", (value, shortcode)
            )

    # -------------------------------------------------------------------------- jobs

    def set_job(
        self,
        shortcode: str,
        stage: str,
        state: JobState | str,
        *,
        error: str | None = None,
    ) -> None:
        value = state.value if isinstance(state, JobState) else state
        now = iso_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO jobs(shortcode, stage, state, attempts, error, started_at, completed_at)
                VALUES(?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(shortcode, stage) DO UPDATE SET
                    state = excluded.state,
                    attempts = jobs.attempts + 1,
                    error = excluded.error,
                    completed_at = excluded.completed_at
                """,
                (
                    shortcode,
                    stage,
                    value,
                    error,
                    now if value == JobState.RUNNING.value else None,
                    now if value in {JobState.DONE.value, JobState.FAILED.value} else None,
                ),
            )

    def get_job(self, shortcode: str, stage: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE shortcode = ? AND stage = ?", (shortcode, stage)
        ).fetchone()
        return dict(row) if row is not None else None

    # ---------------------------------------------------------------------- payloads

    def save_metadata(self, shortcode: str, payload: dict[str, Any], fetch_status: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO metadata(shortcode, payload, fetch_status, fetched_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(shortcode) DO UPDATE SET
                    payload = excluded.payload,
                    fetch_status = excluded.fetch_status,
                    fetched_at = excluded.fetched_at
                """,
                (shortcode, json.dumps(payload), fetch_status, iso_now()),
            )

    def get_metadata(self, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM metadata WHERE shortcode = ?", (shortcode,)
        ).fetchone()
        return _payload_row(row) if row is not None else None

    def save_media(
        self, shortcode: str, payload: dict[str, Any], *, bytes_written: int, verified: bool
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO media(shortcode, payload, bytes, verified, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(shortcode) DO UPDATE SET
                    payload = excluded.payload,
                    bytes = excluded.bytes,
                    verified = excluded.verified,
                    updated_at = excluded.updated_at
                """,
                (shortcode, json.dumps(payload), bytes_written, int(verified), iso_now()),
            )

    def get_media(self, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM media WHERE shortcode = ?", (shortcode,)).fetchone()
        return _payload_row(row) if row is not None else None

    def save_transcript(self, shortcode: str, payload: dict[str, Any]) -> None:
        self._save_payload("transcripts", shortcode, payload)

    def get_transcript(self, shortcode: str) -> dict[str, Any] | None:
        return self._get_payload("transcripts", shortcode)

    def save_triage(self, shortcode: str, payload: dict[str, Any]) -> None:
        self._save_payload("triage", shortcode, payload)

    def get_triage(self, shortcode: str) -> dict[str, Any] | None:
        return self._get_payload("triage", shortcode)

    def save_extraction(self, shortcode: str, payload: dict[str, Any]) -> None:
        self._save_payload("extractions", shortcode, payload)

    def get_extraction(self, shortcode: str) -> dict[str, Any] | None:
        return self._get_payload("extractions", shortcode)

    def _save_payload(self, table: str, shortcode: str, payload: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                f"""
                INSERT INTO {table}(shortcode, payload, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(shortcode) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (shortcode, json.dumps(payload), iso_now()),
            )

    def _get_payload(self, table: str, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE shortcode = ?", (shortcode,)
        ).fetchone()
        return _payload_row(row) if row is not None else None

    def save_classification(
        self,
        shortcode: str,
        *,
        primary_label: str | None,
        secondary: list[str],
        confidence: float,
        method: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO classifications(
                    shortcode, primary_label, secondary, confidence, method, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(shortcode) DO UPDATE SET
                    primary_label = excluded.primary_label,
                    secondary = excluded.secondary,
                    confidence = excluded.confidence,
                    method = excluded.method,
                    created_at = excluded.created_at
                """,
                (shortcode, primary_label, json.dumps(secondary), confidence, method, iso_now()),
            )

    def get_classification(self, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM classifications WHERE shortcode = ?", (shortcode,)
        ).fetchone()
        if row is None:
            return None
        return {
            "shortcode": row["shortcode"],
            "primary": row["primary_label"],
            "secondary": json.loads(row["secondary"]),
            "confidence": row["confidence"],
            "method": row["method"],
            "created_at": row["created_at"],
        }

    def save_review(self, shortcode: str, payload: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO reviews(shortcode, payload, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(shortcode) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (shortcode, json.dumps(payload), iso_now()),
            )

    def get_review(self, shortcode: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM reviews WHERE shortcode = ?", (shortcode,)
        ).fetchone()
        return _payload_row(row) if row is not None else None

    # ------------------------------------------------------------------------- costs

    def record_cost(
        self, shortcode: str | None, provider: str, model: str, *, units: float, usd: float
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO costs(shortcode, provider, model, units, usd, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (shortcode, provider, model, units, usd, iso_now()),
            )

    def total_cost(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(usd), 0) AS total FROM costs").fetchone()
        return float(row["total"]) if row is not None else 0.0

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) AS n FROM reels").fetchone()["n"]
        by_status = {
            row["status"]: row["n"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM reels GROUP BY status"
            ).fetchall()
        }
        by_category = {
            row["primary_label"]: row["n"]
            for row in self._conn.execute(
                "SELECT primary_label, COUNT(*) AS n FROM classifications "
                "WHERE primary_label IS NOT NULL GROUP BY primary_label ORDER BY n DESC"
            ).fetchall()
        }
        return {
            "reels": total,
            "by_status": by_status,
            "by_category": by_category,
            "cost_usd": self.total_cost(),
        }


def _row_to_reel(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "shortcode": row["shortcode"],
        "url": row["url"],
        "media_type": row["media_type"],
        "author": row["author"],
        "saved_at": row["saved_at"],
        "collections": json.loads(row["collections"]),
        "caption": row["caption"],
        "hashtags": json.loads(row["hashtags"]),
        "source_import": row["source_import"],
        "status": row["status"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
    }


def _payload_row(row: sqlite3.Row) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(row["payload"])
    row_data = dict(row)
    row_data.pop("payload", None)
    payload["_row"] = row_data
    return payload


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["SCHEMA_VERSION", "Store"]
