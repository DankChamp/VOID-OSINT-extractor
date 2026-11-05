"""Pipeline orchestration — runs the full fetch → transcribe → triage → extract →
classify → media pipeline with checkpointing, unattended limits and drive-safety."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from reel_vault.classification.service import ClassificationService
from reel_vault.cli.context import AppContext
from reel_vault.enrich.service import EnrichmentService
from reel_vault.extract.service import ExtractionService
from reel_vault.logging_setup import get_logger
from reel_vault.media.base import DriveUnavailable
from reel_vault.media.service import MediaService
from reel_vault.schemas.common import ReelStatus
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_json_atomic
from reel_vault.storage.paths import WorkspacePaths
from reel_vault.transcribe.service import TranscriptionService
from reel_vault.triage.service import TriageService

logger = get_logger(__name__)

# Terminal statuses: a reel in one of these needs no further pipeline work.
TERMINAL_STATUSES = frozenset(
    {
        ReelStatus.DONE.value,
        ReelStatus.STUB.value,
        ReelStatus.UNAVAILABLE.value,
    }
)

STAGE_ORDER: tuple[str, ...] = ("fetch", "transcribe", "triage", "extract", "classify", "media")


@dataclass(frozen=True)
class RunOptions:
    limit: int | None = None
    dry_run: bool = False
    depth: str = "adaptive"
    unattended: bool = False
    max_hours: float = 0.0
    max_cost_usd: float = 0.0
    stop_at: str | None = None
    resume: bool = False
    force: bool = False
    download_mode: str | None = None
    media_dir: str | None = None


@dataclass
class RunReport:
    started_at: str
    finished_at: str | None = None
    mode: str = "adaptive"
    unattended: bool = False
    limit: int | None = None
    processed: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    bytes_downloaded: int = 0
    audio_minutes: float = 0.0
    cost_usd: float = 0.0
    stage_counts: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    per_category: dict[str, int] = field(default_factory=dict)
    exit_code: int = 0
    resume_command: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "mode": self.mode,
            "unattended": self.unattended,
            "limit": self.limit,
            "processed": self.processed,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "bytes_downloaded": self.bytes_downloaded,
            "audio_minutes": round(self.audio_minutes, 2),
            "cost_usd": round(self.cost_usd, 6),
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "failures": dict(sorted(self.failures.items())),
            "per_category": dict(sorted(self.per_category.items())),
            "exit_code": self.exit_code,
            "resume_command": self.resume_command,
        }

    def to_markdown(self) -> str:
        lines = [
            "# reel-vault run report",
            f"Started: {self.started_at}",
            f"Finished: {self.finished_at or '—'}",
            f"Mode: {self.mode} (unattended: {self.unattended})",
            "",
            "## Summary",
            f"- Processed: {self.processed}",
            f"- Completed: {self.completed}",
            f"- Failed: {self.failed}",
            f"- Skipped: {self.skipped}",
            f"- Bytes downloaded: {self.bytes_downloaded:,}",
            f"- Audio minutes: {self.audio_minutes:.1f}",
            f"- Estimated cost: ${self.cost_usd:.4f}",
            f"- Exit code: {self.exit_code}",
            "",
        ]
        if self.stage_counts:
            lines.append("## Stage counts")
            for stage, count in sorted(self.stage_counts.items()):
                lines.append(f"- {stage}: {count}")
            lines.append("")
        if self.per_category:
            lines.append("## Per-category counts")
            for category, count in sorted(self.per_category.items()):
                lines.append(f"- {category}: {count}")
            lines.append("")
        if self.failures:
            lines.append("## Failures")
            for code, reason in sorted(self.failures.items()):
                lines.append(f"- {code}: {reason}")
            lines.append("")
        if self.resume_command:
            lines.append("## Resume")
            lines.append(f"```bash\n{self.resume_command}\n```")
        return "\n".join(lines)


class PipelineRunner:
    """Runs the full pipeline for all pending reels."""

    def __init__(
        self,
        context: AppContext,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.context = context
        self.paths: WorkspacePaths = context.paths
        self.config = context.config
        self.sleep = sleep
        self.clock = clock
        self.store: Store = context.store()

    # ------------------------------------------------------------------ selection

    def _pending_reels(self, limit: int | None, force: bool) -> list[dict[str, Any]]:
        with self.store as store:
            reels = store.list_reels(limit=10000)
            if not force:
                reels = [r for r in reels if str(r["status"]) not in TERMINAL_STATUSES]
            if limit is not None:
                reels = reels[: max(0, limit)]
        return reels

    # ------------------------------------------------------------------ limits

    def _time_limit_reached(self, started: float, max_hours: float, stop_at: str | None) -> bool:
        if max_hours <= 0 and stop_at is None:
            return False
        now = datetime.now(UTC)
        if stop_at:
            try:
                hour, minute = (int(x) for x in stop_at.split(":"))
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                return bool(now >= target)
            except ValueError:
                logger.warning("invalid --stop-at value: %s", stop_at)
        if max_hours > 0:
            elapsed_hours = (self.clock() - started) / 3600.0
            return bool(elapsed_hours >= max_hours)
        return False

    def _cost_limit_reached(self, max_cost_usd: float) -> bool:
        if max_cost_usd <= 0:
            return False
        with self.store as store:
            return store.total_cost() >= max_cost_usd

    # ------------------------------------------------------------------ stages

    def _run_fetch(self, limit: int | None) -> dict[str, Any]:
        service = EnrichmentService(self.store, self.paths, self.config)
        summary = service.fetch_pending(
            limit=limit,
            force=False,
        )
        return summary.as_dict()

    def _run_transcribe(self, limit: int | None) -> dict[str, Any]:
        service = TranscriptionService(self.store, self.paths, self.config)
        summary = service.transcribe_pending(limit=limit)
        return summary.as_dict()

    def _run_triage(self, limit: int | None, dry_run: bool) -> dict[str, Any]:
        service = TriageService(self.store, self.paths, self.config)
        summary = service.triage_pending(limit=limit, dry_run=dry_run)
        return summary.as_dict()

    def _run_extract(self, limit: int | None) -> dict[str, Any]:
        service = ExtractionService(self.store, self.paths, self.config)
        summary = service.extract_pending(limit=limit)
        return summary.as_dict()

    def _run_classify(self, limit: int | None) -> dict[str, Any]:
        service = ClassificationService(self.store, self.paths, self.config)
        summary = service.classify_pending(limit=limit)
        return summary.as_dict()

    def _run_media(self, mode: str | None, limit: int | None) -> dict[str, Any]:

        selected_mode = mode or self.config.download.mode
        if selected_mode not in {"none", "archive", "all"}:
            selected_mode = "archive"
        service = MediaService(self.store, self.paths, self.config)
        try:
            summary = service.download_pending(
                mode=selected_mode,  # type: ignore[arg-type]
                quality=self.config.download.quality,
                limit=limit,
            )
        except DriveUnavailable as exc:
            raise DriveUnavailable(str(exc)) from exc
        return summary.as_dict()

    # ------------------------------------------------------------------ run report

    def _write_report(self, report: RunReport) -> None:
        runs_dir = self.paths.runs
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        report_json = run_dir / "report.json"
        report_md = run_dir / "report.md"
        write_json_atomic(report_json, report.as_dict())
        report_md.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("run report written to %s", run_dir)

    # ------------------------------------------------------------------ main entry

    def run(self, options: RunOptions) -> int:
        report = RunReport(
            started_at=datetime.now(UTC).isoformat(),
            mode=options.depth,
            unattended=options.unattended,
            limit=options.limit,
        )

        if options.dry_run:
            console_report = self._dry_run(options, report)
            self._write_report(report)
            return console_report

        started = self.clock()
        exit_code = 0

        try:
            # Stage 1: fetch metadata
            if not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ) and not self._cost_limit_reached(options.max_cost_usd):
                fetch_summary = self._run_fetch(options.limit)
                report.stage_counts["fetch"] = fetch_summary.get("ok", 0) + fetch_summary.get(
                    "failed", 0
                )
                report.processed += fetch_summary.get("processed", 0)
                report.failed += fetch_summary.get("failed", 0)

            # Stage 2: transcribe
            if not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ) and not self._cost_limit_reached(options.max_cost_usd):
                transcribe_summary = self._run_transcribe(options.limit)
                report.stage_counts["transcribe"] = transcribe_summary.get("transcribed", 0)
                report.processed += transcribe_summary.get("processed", 0)
                report.failed += transcribe_summary.get("failed", 0)
                report.audio_minutes += transcribe_summary.get("audio_seconds", 0) / 60.0
                report.cost_usd += transcribe_summary.get("cost_usd", 0.0)

            # Stage 3: triage
            if not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ) and not self._cost_limit_reached(options.max_cost_usd):
                triage_summary = self._run_triage(options.limit, dry_run=False)
                report.stage_counts["triage"] = triage_summary.get("processed", 0)
                report.processed += triage_summary.get("processed", 0)
                report.failed += triage_summary.get("failed", 0)
                report.cost_usd += triage_summary.get("cost_usd", 0.0)

            # Stage 4: extract
            if not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ) and not self._cost_limit_reached(options.max_cost_usd):
                extract_summary = self._run_extract(options.limit)
                report.stage_counts["extract"] = extract_summary.get("processed", 0)
                report.processed += extract_summary.get("processed", 0)
                report.failed += extract_summary.get("failed", 0)
                report.cost_usd += extract_summary.get("cost_usd", 0.0)

            # Stage 5: classify
            if not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ) and not self._cost_limit_reached(options.max_cost_usd):
                classify_summary = self._run_classify(options.limit)
                report.stage_counts["classify"] = classify_summary.get("processed", 0)
                report.processed += classify_summary.get("processed", 0)
                report.failed += classify_summary.get("failed", 0)
                report.cost_usd += classify_summary.get("cost_usd", 0.0)

            # Stage 6: media download
            if options.download_mode != "none" and not self._time_limit_reached(
                started, options.max_hours, options.stop_at
            ):
                try:
                    media_summary = self._run_media(options.download_mode, options.limit)
                    report.stage_counts["media"] = media_summary.get("downloaded", 0)
                    report.processed += media_summary.get("processed", 0)
                    report.failed += media_summary.get("failed", 0)
                    report.bytes_downloaded += media_summary.get("bytes", 0)
                except DriveUnavailable as exc:
                    logger.error("media drive unavailable: %s", exc)
                    report.exit_code = 4
                    exit_code = 4
                    report.failures["drive"] = str(exc)

        except DriveUnavailable as exc:
            logger.error("media drive unavailable: %s", exc)
            report.exit_code = 4
            exit_code = 4
            report.failures["drive"] = str(exc)

        # Per-category counts
        with self.store as store:
            stats = store.stats()
        report.per_category = stats.get("by_category", {})
        report.completed = stats.get("by_status", {}).get(ReelStatus.DONE.value, 0)

        report.finished_at = datetime.now(UTC).isoformat()

        # Resume command
        cmd_parts = ["reel-vault run --resume"]
        if options.limit is not None:
            cmd_parts.append(f"--limit {options.limit}")
        if options.max_hours > 0:
            cmd_parts.append(f"--max-hours {options.max_hours}")
        if options.max_cost_usd > 0:
            cmd_parts.append(f"--max-cost-usd {options.max_cost_usd}")
        report.resume_command = " ".join(cmd_parts)

        self._write_report(report)
        return exit_code

    def _dry_run(self, options: RunOptions, report: RunReport) -> int:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        reels = self._pending_reels(options.limit, options.force)
        table = Table(title="pipeline plan (dry run, no model calls)")
        table.add_column("shortcode")
        table.add_column("status")
        table.add_column("next_stage")
        for reel in reels:
            status = str(reel["status"])
            next_stage = self._next_stage(status)
            table.add_row(str(reel["shortcode"]), status, next_stage)
        console.print(table)
        console.print(f"[dim]{len(reels)} reel(s) would be processed[/dim]")
        report.processed = len(reels)
        report.skipped = len(reels)
        return 0

    @staticmethod
    def _next_stage(status: str) -> str:
        stage_map = {
            ReelStatus.NEW.value: "fetch",
            ReelStatus.FETCHED.value: "transcribe",
            ReelStatus.TRANSCRIBED.value: "triage",
            ReelStatus.TRANSCRIPT_FAILED.value: "transcribe",
            ReelStatus.TRIAGED.value: "extract",
            ReelStatus.STUB.value: "done",
            ReelStatus.EXTRACTED.value: "classify",
            ReelStatus.EXTRACTION_FAILED.value: "extract",
            ReelStatus.CLASSIFIED.value: "media",
            ReelStatus.CLASSIFICATION_FAILED.value: "classify",
            ReelStatus.MEDIA_FAILED.value: "media",
            ReelStatus.TRIAGE_FAILED.value: "triage",
        }
        return stage_map.get(status, "done")


def run_pipeline(context: AppContext, options: RunOptions) -> int:
    """Entry point called by the CLI ``run`` command."""
    runner = PipelineRunner(context)
    return runner.run(options)


__all__ = [
    "PipelineRunner",
    "RunOptions",
    "RunReport",
    "run_pipeline",
]
