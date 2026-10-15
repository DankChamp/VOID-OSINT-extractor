"""Cached stage execution (§15, §19).

One place decides whether a stage is reused or recomputed, writes its artifact
atomically, and records the outcome both in SQLite and in the per-video
``state.json`` manifest. Stages are content-addressed: reuse requires a matching
input hash and configuration hash, so changing an output format never invalidates
extraction work and changing the transcript always invalidates what depends on it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from video_intel.logging_setup import get_logger
from video_intel.schemas.common import StageId, StageState, utc_now
from video_intel.schemas.manifest import StageManifest, StageRecord
from video_intel.storage.db import StateStore
from video_intel.storage.io import write_model_atomic
from video_intel.storage.paths import VideoPaths

logger = get_logger(__name__)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_hash(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


@dataclass
class StageContext:
    """Everything a stage needs from its project."""

    project_id: str
    video_id: str
    paths: VideoPaths
    store: StateStore
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StageOutcome:
    """Result of running (or reusing) one stage."""

    stage_id: StageId
    state: StageState
    reused: bool
    path: Path | None
    output_hash: str | None
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.state is StageState.COMPLETED


class StageRunner:
    """Runs stages with cache-aware reuse, atomic artifacts and recorded state."""

    def __init__(self, context: StageContext, *, reuse: bool = True) -> None:
        self.context = context
        self.reuse = reuse

    # ------------------------------------------------------------------ manifest
    def load_manifest(self) -> StageManifest:
        manifest = self.context.store.load_manifest(
            project_id=self.context.project_id, video_id=self.context.video_id
        )
        state_file = self.context.paths.state
        if state_file.is_file():
            from video_intel.storage.io import read_model

            disk = read_model(state_file, StageManifest)
            for record in disk.stages:
                if manifest.get(record.stage_id) is None:
                    manifest.upsert(record)
        return manifest

    def save_manifest(self, manifest: StageManifest) -> None:
        self.context.paths.ensure()
        write_model_atomic(self.context.paths.state, manifest)
        for record in manifest.stages:
            self.context.store.upsert_stage(self.context.project_id, self.context.video_id, record)

    # ---------------------------------------------------------------------- run
    def run(
        self,
        stage_id: StageId,
        compute: Callable[[], BaseModel | str],
        *,
        artifact_path: Path,
        input_hash: str,
        config_hash: str | None = None,
        force: bool = False,
        writer: Callable[[Path, Any], int] | None = None,
    ) -> StageOutcome:
        """Execute one stage, reusing a valid cached artifact when possible.

        Args:
            compute: produces the stage result (a schema object, or text for outputs).
            writer: how to persist it; defaults to atomic JSON for schema objects and
                atomic UTF-8 for strings.
        """
        manifest = self.load_manifest()
        previous = manifest.get(stage_id)

        if (
            self.reuse
            and not force
            and previous is not None
            and artifact_path.is_file()
            and previous.state is StageState.COMPLETED
            and not previous.is_stale(input_hash=input_hash, config_hash=config_hash)
        ):
            logger.info("stage=%s state=reused artifact=%s", stage_id.value, artifact_path.name)
            return StageOutcome(
                stage_id=stage_id,
                state=StageState.COMPLETED,
                reused=True,
                path=artifact_path,
                output_hash=previous.output_hash or file_hash(artifact_path),
                duration_seconds=0.0,
            )

        started = utc_now()
        self._record(manifest, stage_id, StageState.RUNNING, input_hash, config_hash, started)
        self.save_manifest(manifest)

        logger.info("stage=%s state=running", stage_id.value)
        try:
            result = compute()
        except Exception as exc:  # stage failures are recorded, then re-raised
            elapsed = (utc_now() - started).total_seconds()
            self._record(
                manifest,
                stage_id,
                StageState.FAILED,
                input_hash,
                config_hash,
                started,
                duration=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.save_manifest(manifest)
            logger.error("stage=%s state=failed error=%s", stage_id.value, exc)
            raise

        write = writer or _default_writer
        write(artifact_path, result)
        output_hash = file_hash(artifact_path)
        elapsed = (utc_now() - started).total_seconds()
        self._record(
            manifest,
            stage_id,
            StageState.COMPLETED,
            input_hash,
            config_hash,
            started,
            duration=elapsed,
            output_hash=output_hash,
        )
        self.save_manifest(manifest)
        logger.info(
            "stage=%s state=completed artifact=%s bytes=%d",
            stage_id.value,
            artifact_path.name,
            artifact_path.stat().st_size,
        )
        return StageOutcome(
            stage_id=stage_id,
            state=StageState.COMPLETED,
            reused=False,
            path=artifact_path,
            output_hash=output_hash,
            duration_seconds=elapsed,
        )

    def note(
        self,
        stage_id: StageId,
        state: StageState,
        *,
        input_hash: str | None = None,
        output_hash: str | None = None,
        config_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a stage this runner did not execute itself.

        Used by stages that manage their own artifacts in several files (extraction
        writes one record per chunk) but must still appear in the manifest and in
        ``inspect``.
        """
        manifest = self.load_manifest()
        now = utc_now()
        previous = manifest.get(stage_id)
        manifest.upsert(
            StageRecord(
                stage_id=stage_id,
                state=state,
                attempts=(previous.attempts if previous else 0) + 1,
                input_hash=input_hash,
                output_hash=output_hash,
                config_hash=config_hash,
                started_at=now,
                completed_at=now,
                error=error,
            )
        )
        self.save_manifest(manifest)

    # ----------------------------------------------------------------- internals
    def _record(
        self,
        manifest: StageManifest,
        stage_id: StageId,
        state: StageState,
        input_hash: str,
        config_hash: str | None,
        started: datetime,
        *,
        duration: float | None = None,
        output_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        previous = manifest.get(stage_id)
        attempts = (previous.attempts if previous else 0) + (
            0 if state is StageState.RUNNING else 1
        )
        completed_at = utc_now() if state in {StageState.COMPLETED, StageState.FAILED} else None
        manifest.upsert(
            StageRecord(
                stage_id=stage_id,
                state=state,
                attempts=attempts,
                input_hash=input_hash,
                output_hash=output_hash,
                config_hash=config_hash,
                started_at=started,
                completed_at=completed_at,
                duration_seconds=duration,
                error=error,
            )
        )


def _default_writer(path: Path, payload: BaseModel | str) -> int:
    """Schema objects are stored as JSON; plain strings are stored verbatim."""
    from video_intel.storage.io import write_text_atomic

    if isinstance(payload, str):
        return write_text_atomic(path, payload)
    return write_model_atomic(path, payload)


def elapsed(seconds: float) -> timedelta:
    return timedelta(seconds=round(seconds, 3))


__all__ = [
    "StageContext",
    "StageOutcome",
    "StageRunner",
    "elapsed",
    "file_hash",
    "payload_hash",
]
