"""``video-intel`` command line interface (§16).

Phase 1 ships the command surface, configuration inspection and storage layout.
The ``analyze`` command validates input and prints the execution plan; it does not
yet ingest transcripts (Phase 2) and reports that honestly instead of pretending.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from video_intel.cli.builders import build_ingestor
from video_intel.cli.context import AppContext, build_context, fail_config
from video_intel.config.loader import ConfigError
from video_intel.config.models import AppConfig, PipelineConfig
from video_intel.extraction.engine import ExtractionOutcome
from video_intel.ingest.errors import IngestError
from video_intel.ingest.service import IngestRequest
from video_intel.ingest.youtube import parse_youtube_url
from video_intel.logging_setup import get_logger
from video_intel.notebooklm.handoff import (
    MANUAL_WORKFLOW_NOTICE,
    ImportValidationError,
    export_packet,
    import_records,
    load_imported_records,
    store_import,
)
from video_intel.pipeline.analysis_stages import (
    ExtractionStageResult,
    build_dataset,
    dataset_stage,
    load_dataset,
    output_stages,
    run_extractions,
)
from video_intel.pipeline.ingest_stages import (
    IngestionOutcome,
    chunks_after_normalization,
    ingest_stages,
)
from video_intel.pipeline.runner import StageContext, StageOutcome, StageRunner, payload_hash
from video_intel.pipeline.stages import OUTPUT_STAGES as OUTPUT_STAGE_ORDER
from video_intel.pipeline.stages import STAGE_ORDER, stages_up_to
from video_intel.providers.base import Provider, ProviderError
from video_intel.schemas.common import (
    ExecutionMode,
    OutputMode,
    PipelineId,
    ProviderKind,
    StageId,
    StageState,
    format_timestamp,
    utc_now,
)
from video_intel.schemas.extraction import ExtractedItem, RawExtractionRecord
from video_intel.schemas.knowledge import ExtractionRun, KnowledgeDataset
from video_intel.schemas.outputs import OUTPUT_STAGES_BY_MODE
from video_intel.schemas.transcript import NormalizedTranscript
from video_intel.storage.io import read_json, read_model
from video_intel.storage.locking import LockError, video_lock
from video_intel.storage.paths import ProjectPaths, VideoPaths, relative_key

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_NOT_IMPLEMENTED = 3

app = typer.Typer(
    name="video-intel",
    help="Transcript-first, multi-AI knowledge extraction for YouTube videos.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect and validate configuration.")
app.add_typer(config_app, name="config")
notebooklm_app = typer.Typer(
    help="Manual NotebookLM handoff: export source packets, import results."
)
app.add_typer(notebooklm_app, name="notebooklm")

console = Console()
err_console = Console(stderr=True)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("video-intel")
    except PackageNotFoundError:  # pragma: no cover - editable install without metadata
        return "0.0.0+local"


ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to a configuration file.", show_default=False),
]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]
QuietOption = Annotated[bool, typer.Option("--quiet", "-q", help="Suppress console logging.")]


def _load_context(
    config_path: Path | None, verbose: bool, quiet: bool, *, log_to_file: bool = False
) -> AppContext:
    try:
        return build_context(
            config_path=config_path, verbose=verbose, quiet=quiet, log_to_file=log_to_file
        )
    except ConfigError as exc:
        fail_config(exc)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"video-intel {_version()}")
        raise typer.Exit(code=EXIT_OK)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Video intelligence: turn a transcript into a reusable knowledge dataset."""


# ---------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------
@app.command()
def analyze(
    url: Annotated[str, typer.Argument(help="YouTube video URL.")],
    mode: Annotated[
        OutputMode, typer.Option("--mode", "-m", help="Analysis mode.")
    ] = OutputMode.DEEP,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Override the provider used by every pipeline."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--pipeline", help="Named pipeline preset, e.g. 'maximum-coverage'."),
    ] = None,
    execution_mode: Annotated[
        str | None,
        typer.Option("--execution-mode", help="local_only, cloud_only or hybrid."),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Project name (default: 'default').")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Override the storage root.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the execution plan without processing."),
    ] = False,
    until: Annotated[
        StageId | None,
        typer.Option(
            "--until",
            help=(
                "Stop after this stage. Runnable stages: "
                + ", ".join(stage.value for stage in STAGE_ORDER)
            ),
        ),
    ] = None,
    languages: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            "-l",
            help="Caption language priority (repeatable). Overrides configuration.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Ignore cached stage results and recompute."),
    ] = False,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Ingest the transcript for a YouTube URL, then run the stages that exist.

    Implemented today: metadata, raw transcript and normalization (Phase 2). Chunking
    (Phase 3), extraction (Phase 5), verification (Phase 6) and outputs (Phase 7) are
    not implemented, so the command stops with an explicit message instead of
    substituting an ordinary summary for structured extraction.
    """
    try:
        video_id = parse_youtube_url(url).video_id
    except IngestError as exc:
        err_console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc

    overrides: dict[str, Any] = {}
    if execution_mode:
        overrides["execution_mode"] = execution_mode
    if project:
        overrides["project_name"] = project
    if data_dir:
        overrides["data_dir"] = str(data_dir)

    ctx = _load_context(config, verbose, quiet, log_to_file=True)
    config_obj = ctx.config

    if preset:
        try:
            pipelines = config_obj.pipelines_for_preset(preset)
        except KeyError as exc:
            err_console.print(f"[red]{exc.args[0]}[/red]")
            raise typer.Exit(code=EXIT_CONFIG) from exc
    else:
        pipelines = config_obj.pipelines_for_mode(mode)
        if not pipelines:
            pipelines = config_obj.enabled_pipelines()

    if provider:
        if provider not in config_obj.providers:
            err_console.print(
                f"[red]unknown provider '{provider}'[/red]; "
                f"run 'video-intel providers' to list configured providers"
            )
            raise typer.Exit(code=EXIT_CONFIG)
        if (
            config_obj.execution_mode is ExecutionMode.LOCAL_ONLY
            and not config_obj.provider(provider).is_local()
        ):
            err_console.print(
                f"[red]execution_mode is local_only but '{provider}' sends data to a remote "
                "provider[/red]. Pass --execution-mode hybrid or cloud_only to allow it "
                "explicitly."
            )
            raise typer.Exit(code=EXIT_CONFIG)
        for pipeline in pipelines:
            pipeline.provider = provider
        logger.debug("provider_override=%s", provider)

    runnable_stages = STAGE_ORDER
    target_stage = until or runnable_stages[-1]
    if target_stage not in runnable_stages:
        err_console.print(
            f"[red]stage '{target_stage.value}' is not implemented[/red]; runnable stages: "
            + ", ".join(stage.value for stage in runnable_stages)
        )
        raise typer.Exit(code=EXIT_CONFIG)

    paths = ctx.project_paths.video(video_id)
    _print_plan(
        ctx,
        video_id=video_id,
        url=url,
        mode=mode,
        pipelines=pipelines,
        paths=paths,
        until=target_stage,
    )

    if dry_run:
        console.print("[green]dry run: nothing was fetched, processed or written[/green]")
        raise typer.Exit(code=EXIT_OK)

    try:
        exit_code = _run_pipeline(
            ctx,
            video_id=video_id,
            url=url,
            languages=tuple(languages) if languages else config_obj.ingest.language_tuple(),
            until=target_stage,
            force=force,
            mode=mode,
            pipelines=pipelines,
        )
    except LockError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=EXIT_ERROR) from exc
    raise typer.Exit(code=exit_code)


def _run_pipeline(
    ctx: AppContext,
    *,
    video_id: str,
    url: str,
    languages: tuple[str, ...],
    until: StageId,
    force: bool,
    mode: OutputMode,
    pipelines: list[PipelineConfig],
) -> int:
    """Run every implemented stage in order, holding the per-video lock."""
    project_paths = ctx.project_paths.ensure()
    paths = project_paths.video(video_id)
    with video_lock(paths.root, purpose="analyze"):
        return _run_locked(
            ctx,
            project_paths=project_paths,
            paths=paths,
            video_id=video_id,
            url=url,
            languages=languages,
            until=until,
            force=force,
            mode=mode,
            pipelines=pipelines,
        )


def _run_locked(
    ctx: AppContext,
    *,
    project_paths: ProjectPaths,
    paths: VideoPaths,
    video_id: str,
    url: str,
    languages: tuple[str, ...],
    until: StageId,
    force: bool,
    mode: OutputMode,
    pipelines: list[PipelineConfig],
) -> int:
    """The pipeline body, executed while holding the per-video lock."""
    config_obj = ctx.config
    paths = paths.ensure()
    store = ctx.store
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project(
        project_paths.project_id,
        config_obj.project_name,
        config_hash=config_obj.hash(),
        now=now,
    )
    store.upsert_video(project_paths.project_id, video_id, url=url, now=now)

    context = StageContext(
        project_id=project_paths.project_id,
        video_id=video_id,
        paths=paths,
        store=store,
    )
    runner = StageRunner(context, reuse=not force)

    with closing(build_ingestor(config_obj)) as ingestor:
        try:
            ingestion = ingest_stages(
                runner=runner,
                ingestor=ingestor,
                request=IngestRequest(url=url, languages=languages),
                config=config_obj,
                force=force,
                until=until,
            )
            chunk_set, chunk_outcome = chunks_after_normalization(
                runner=runner, config=config_obj, until=until, force=force
            )
        except IngestError as exc:
            err_console.print(f"[red]{exc.message}[/red]")
            if exc.hint:
                err_console.print(f"[dim]hint: {exc.hint}[/dim]")
            return exc.exit_code

    if chunk_outcome is not None:
        ingestion = replace(ingestion, chunks=chunk_set, stages=(*ingestion.stages, chunk_outcome))
    _print_ingestion_summary(ingestion, paths.root, until=until)

    if _stops_before_analysis(until) or chunk_set is None or ingestion.normalized is None:
        err_console.print(
            "\n[yellow]stopped before analysis[/yellow] as requested by --until.\n"
            "Re-run without --until to continue into extraction."
        )
        return EXIT_OK

    exit_code = _run_analysis(
        ctx, context, runner, ingestion, config_obj, pipelines, mode, force, until
    )
    _close_providers(ctx)
    return exit_code


def _requested_output_stages(mode: OutputMode, until: StageId) -> set[StageId]:
    """Outputs for this run: the mode's set, cut at ``--until`` when one is given."""
    requested = set(OUTPUT_STAGES_BY_MODE[mode])
    if until not in STAGE_ORDER:
        return requested
    if until not in OUTPUT_STAGE_ORDER:
        return requested
    cutoff = OUTPUT_STAGE_ORDER.index(until)
    return {stage for stage in requested if stage in OUTPUT_STAGE_ORDER[: cutoff + 1]}


def _stops_before_analysis(until: StageId) -> bool:
    """True when ``--until`` names a stage that precedes extraction."""
    if until not in STAGE_ORDER:
        return True
    return STAGE_ORDER.index(until) < STAGE_ORDER.index(StageId.EXTRACTIONS)


def _run_analysis(
    ctx: AppContext,
    context: StageContext,
    runner: StageRunner,
    ingestion: IngestionOutcome,
    config_obj: AppConfig,
    pipelines: list[PipelineConfig],
    mode: OutputMode,
    force: bool,
    until: StageId,
) -> int:
    """Extraction, verification, dataset assembly and outputs."""
    registry = ctx.registry()
    chunk_set = ingestion.chunks
    normalized = ingestion.normalized
    assert chunk_set is not None and normalized is not None

    by_name: dict[str, Provider] = {}
    for pipeline in pipelines:
        if pipeline.provider not in by_name:
            try:
                by_name[pipeline.provider] = registry.build(pipeline.provider, model=pipeline.model)
            except ProviderError as exc:
                err_console.print(f"[red]{exc.render()}[/red]")
                err_console.print(
                    "[dim]fix the configuration or choose another provider; no extraction was "
                    "claimed.[/dim]"
                )
                return EXIT_CONFIG

    max_passes = config_obj.max_extraction_passes
    started = utc_now()
    all_items: list[ExtractedItem] = []
    runs: list[ExtractionRun] = []
    failures: list[str] = []

    async def extract_all() -> list[ExtractionStageResult]:
        results = []
        used: list[Provider] = []
        try:
            for pipeline in pipelines:
                provider = by_name[pipeline.provider]
                if provider not in used:
                    used.append(provider)
                results.append(
                    await run_extractions(
                        runner=runner,
                        provider=provider,
                        model=pipeline.model,
                        chunk_set=chunk_set,
                        normalized=normalized,
                        pipelines=[pipeline],
                        max_passes=max_passes,
                        force=force,
                        concurrency=config_obj.providers[pipeline.provider].concurrency,
                        on_progress=lambda message: console.print(f"  [dim]{message}[/dim]"),
                    )
                )
            return results
        finally:
            # Always release connections, including on the failure path.
            for provider in used:
                await provider.aclose()

    try:
        results = asyncio.run(extract_all())
    except ProviderError as exc:
        err_console.print(f"[red]{exc.render()}[/red]")
        return EXIT_ERROR

    for result in results:
        all_items.extend(result.outcome.items)
        runs.extend(result.runs)
        failures.extend(result.outcome.failures)

    elapsed = (utc_now() - started).total_seconds()
    console.print(
        f"\n[bold]extraction[/bold]  {len(all_items)} item(s) from "
        f"{len(chunk_set.chunks)} chunk(s) in {elapsed:.1f}s"
    )
    if failures:
        console.print(f"[yellow]  {len(failures)} chunk/pipeline call(s) failed[/yellow]")
        for failure in failures[:5]:
            console.print(f"    [dim]{failure}[/dim]")

    if not all_items and failures:
        # §19: never report success when the provider produced nothing at all.
        err_console.print(
            "\n[red]no provider returned usable output[/red]\n"
            f"{len(failures)} extraction call(s) failed; nothing was verified and no "
            "knowledge dataset was produced."
        )
        return EXIT_ERROR

    dataset_result = build_dataset(
        video_id=context.video_id,
        title=ingestion.metadata.title,
        normalized=normalized,
        extraction=ExtractionOutcome(items=all_items),
        runs=runs,
        transcript_hash=normalized.source_transcript_hash,
        config_hash=config_obj.hash(),
        completed=[stage.stage_id for stage in ingestion.stages]
        + [StageId.EXTRACTIONS]
        + [StageId.VERIFY, StageId.CONFLICTS, StageId.KNOWLEDGE_DATASET]
        + list(OUTPUT_STAGES_BY_MODE[mode]),
        failed=[stage.stage_id for stage in ingestion.stages if stage.state is StageState.FAILED]
        + ([StageId.EXTRACTIONS] if failures else []),
        config=config_obj,
    )
    dataset = dataset_result.dataset
    dataset_stage(
        runner=runner,
        dataset=dataset,
        input_hash=payload_hash(
            f"{normalized.source_transcript_hash}|{sorted(i.item_id for i in all_items)}"
        ),
        config_hash=config_obj.hash(),
        force=force,
    )

    requested = _requested_output_stages(mode, until)
    outcomes = output_stages(runner=runner, dataset=dataset, stages=requested, force=force)

    _print_analysis_summary(dataset, paths=context.paths, outcomes=outcomes)
    return EXIT_OK


def _close_providers(ctx: AppContext) -> None:
    registry = getattr(ctx, "_registry", None)
    if registry is None:
        return
    try:
        asyncio.run(registry.aclose())
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("provider cleanup failed", exc_info=True)


def _print_analysis_summary(
    dataset: KnowledgeDataset, *, paths: VideoPaths, outcomes: dict[StageId, StageOutcome]
) -> None:
    report = dataset.completeness
    console.print("\n[bold green]analysis complete[/bold green]")
    console.print(f"  dataset     {len(dataset.items)} item(s) in {len(dataset.topics)} topic(s)")
    states: dict[str, int] = {}
    for item in dataset.items:
        states[str(item.verification)] = states.get(str(item.verification), 0) + 1
    console.print(
        "  verification "
        + (", ".join(f"{k}={v}" for k, v in sorted(states.items())) or "nothing retained")
    )
    if report.unresolved_conflicts:
        console.print(f"[yellow]  {report.unresolved_conflicts} unresolved conflict(s)[/yellow]")
    console.print(f"  completeness {'complete' if report.complete else 'partial'}")
    for stage_id, outcome in sorted(outcomes.items(), key=lambda item: item[0].value):
        state = "reused" if outcome.reused else "generated"
        console.print(f"  {outcome.path.name if outcome.path else stage_id.value}: {state}")
    console.print(f"  stored in   {paths.root}")


def _print_plan(
    ctx: AppContext,
    *,
    video_id: str,
    url: str,
    mode: OutputMode,
    pipelines: list[Any],
    paths: Any,
    until: StageId | None = None,
) -> None:
    config_obj = ctx.config
    console.print(f"[bold]video[/bold]      {video_id}")
    console.print(f"[bold]url[/bold]        {url}")
    console.print(f"[bold]mode[/bold]       {mode.value}")
    console.print(
        f"[bold]execution[/bold]  {config_obj.execution_mode.value} (data dir: {ctx.data_dir})"
    )
    console.print(f"[bold]workspace[/bold]  {paths.root}")

    table = Table(title="Pipeline plan", header_style="bold")
    table.add_column("pipeline")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("passes", justify="right")
    table.add_column("adapter")
    for pipeline in pipelines:
        provider_cfg = config_obj.providers.get(pipeline.provider)
        adapter = "ready" if provider_cfg and provider_cfg.implemented else "not implemented"
        table.add_row(
            pipeline.pipeline_id.value,
            pipeline.provider,
            pipeline.model,
            str(pipeline.passes),
            adapter,
        )
    console.print(table)

    if until is not None:
        stages = list(stages_up_to(until))
        console.print("[bold]stages[/bold]   " + " -> ".join(stage.value for stage in stages))

    outputs = config_obj.outputs_for_mode(mode)
    if outputs:
        console.print("[bold]outputs[/bold]   " + ", ".join(outputs) + " (not yet generated)")
    console.print(f"[bold]config hash[/bold] {config_obj.hash()[:16]}")


def _print_ingestion_summary(outcome: IngestionOutcome, root: Path, *, until: StageId) -> None:
    """Report what was stored, with real counts and timings."""
    console.print()
    console.print("[bold green]ingestion complete[/bold green]")
    console.print(f"  title       {outcome.metadata.title}")
    console.print(f"  channel     {outcome.metadata.channel or 'unknown'}")
    duration = outcome.metadata.duration_seconds
    console.print(
        f"  duration    {format_timestamp(duration)}"
        if duration is not None
        else "  duration    unknown (not reported by the source)"
    )
    if outcome.transcript is not None:
        console.print(
            f"  transcript  {outcome.transcript.segment_count} segments, "
            f"language={outcome.transcript.detected_language or 'unknown'}, "
            f"kind={outcome.transcript.kind.value}"
        )
    if outcome.normalized is not None:
        steps = "; ".join(f"{s.name}={s.segments_changed}" for s in outcome.normalized.steps)
        console.print(f"  normalize   {steps}")
    if outcome.chunks is not None:
        console.print(
            f"  chunks      {len(outcome.chunks.chunks)} chunks, "
            f"~{outcome.chunks.total_chars // outcome.chunks.config.estimated_chars_per_token} "
            "tokens of text, overlap "
            f"{outcome.chunks.config.overlap_chars} chars"
        )
    if outcome.reused_stages:
        console.print(f"  reused      {', '.join(outcome.reused_stages)}")
    if outcome.executed_stages:
        console.print(f"  executed    {', '.join(outcome.executed_stages)}")
    console.print(f"  stored in   {root}")
    console.print(f"  stopped at  {until.value}")


# ---------------------------------------------------------------------------------
# providers / pipelines
# ---------------------------------------------------------------------------------
@app.command()
def providers(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """List configured providers, key availability and adapter status."""
    ctx = _load_context(config, verbose, quiet)
    configs = [ctx.config.providers[name] for name in sorted(ctx.config.providers)]

    if as_json:
        console.print_json(json.dumps([cfg.redacted() for cfg in configs], indent=2))
        return

    table = Table(title="Providers", header_style="bold")
    for column in ("name", "kind", "data", "adapter", "default model", "api key", "enabled"):
        table.add_column(column)
    for cfg in configs:
        table.add_row(
            cfg.name,
            str(cfg.kind),
            "local" if cfg.is_local() else "cloud",
            "ready" if cfg.implemented else "not implemented",
            cfg.default_model or "-",
            "set" if cfg.has_api_key() else ("env missing" if cfg.api_key_env else "none"),
            "yes" if cfg.enabled else "no",
        )
    console.print(table)

    if ProviderKind.NOTEBOOKLM_HANDOFF in {cfg.kind for cfg in configs}:
        console.print(
            "[dim]notebooklm_handoff is a manual import/export workflow, "
            "not an automated API integration (Phase 8).[/dim]"
        )


@app.command()
def pipelines(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of tables.")] = False,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """List extraction pipelines, modes and presets."""
    ctx = _load_context(config, verbose, quiet)
    config_obj = ctx.config

    if as_json:
        payload = {
            "pipelines": {
                pipeline_id.value: pipeline.model_dump(mode="json")
                for pipeline_id, pipeline in sorted(
                    config_obj.pipelines.items(), key=lambda kv: kv[0].value
                )
            },
            "modes": {
                mode.value: mode_config.model_dump(mode="json")
                for mode, mode_config in sorted(
                    config_obj.modes.items(), key=lambda kv: kv[0].value
                )
            },
            "presets": {
                name: preset.model_dump(mode="json")
                for name, preset in sorted(config_obj.presets.items())
            },
        }
        console.print_json(json.dumps(payload, indent=2))
        return

    table = Table(title="Pipelines", header_style="bold")
    for column in ("pipeline", "provider", "model", "passes", "enabled", "purpose"):
        table.add_column(column)
    for pipeline_id, pipeline in sorted(config_obj.pipelines.items(), key=lambda kv: kv[0].value):
        table.add_row(
            pipeline_id.value,
            pipeline.provider,
            pipeline.model,
            str(pipeline.passes),
            "yes" if pipeline.enabled else "no",
            pipeline.description,
        )
    console.print(table)

    modes_table = Table(title="Modes", header_style="bold")
    for column in ("mode", "pipelines", "outputs"):
        modes_table.add_column(column)
    for mode, mode_config in sorted(config_obj.modes.items(), key=lambda kv: kv[0].value):
        modes_table.add_row(
            mode.value,
            ", ".join(p.value for p in mode_config.pipelines) or "-",
            ", ".join(mode_config.outputs) or "-",
        )
    console.print(modes_table)

    presets_table = Table(title="Presets (--pipeline)", header_style="bold")
    for column in ("preset", "mode", "pipelines"):
        presets_table.add_column(column)
    for name, preset in sorted(config_obj.presets.items()):
        presets_table.add_row(
            name,
            preset.mode.value if preset.mode else "-",
            ", ".join(p.value for p in preset.pipelines),
        )
    console.print(presets_table)


# ---------------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------------
@app.command()
def inspect(
    video_id: Annotated[str, typer.Argument(help="Video id to inspect.")],
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Show stored artifacts and stage state for a video."""
    ctx = _load_context(config, verbose, quiet)
    paths = ctx.project_paths.video(video_id)

    console.print(f"[bold]project[/bold]  {ctx.config.project_name}")
    console.print(f"[bold]video[/bold]    {video_id}")
    console.print(f"[bold]directory[/bold] {paths.root}")

    if not paths.root.exists():
        console.print("[yellow]no stored data for this video yet[/yellow]")
        raise typer.Exit(code=EXIT_OK)

    artifacts = paths.existing_artifacts()
    if not artifacts:
        console.print("[yellow]directory exists but holds no artifacts[/yellow]")

    table = Table(title="Artifacts", header_style="bold")
    table.add_column("stage")
    table.add_column("file")
    table.add_column("bytes", justify="right")
    for stage, path in sorted(artifacts.items()):
        table.add_row(stage, relative_key(paths.root, path), f"{path.stat().st_size:,}")
    console.print(table)

    store = ctx.store
    if store.path.exists():
        manifest = store.load_manifest(project_id=ctx.project_paths.project_id, video_id=video_id)
        if manifest.stages:
            state_table = Table(title="Stage state", header_style="bold")
            for column in ("stage", "state", "attempts", "updated", "error"):
                state_table.add_column(column)
            for record in manifest.stages:
                state_table.add_row(
                    record.stage_id.value,
                    record.state.value,
                    str(record.attempts),
                    record.completed_at.isoformat(timespec="seconds")
                    if record.completed_at
                    else "-",
                    record.error or "-",
                )
            console.print(state_table)
        else:
            console.print("[dim]no stage state recorded[/dim]")
    else:
        console.print(f"[dim]state database not created yet ({store.path})[/dim]")


# ---------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------
@config_app.command("show")
def config_show(
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Print the effective configuration with secrets removed."""
    ctx = _load_context(config, verbose, quiet)
    console.print_json(json.dumps(ctx.config.redaction_safe_dump(), indent=2))
    console.print(f"[dim]config hash: {ctx.config.hash()}[/dim]")
    console.print(
        "[dim]only the names of API-key environment variables are stored; values are "
        "never written to disk or logs.[/dim]"
    )


@config_app.command("validate")
def config_validate(
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Validate configuration files and report the merged source list."""
    ctx = _load_context(config, verbose, quiet)
    console.print("[green]configuration is valid[/green]")
    for source in ctx.config.source_files:
        console.print(f"  source: {source}")
    console.print(f"  data_dir: {ctx.data_dir}")
    console.print(f"  config hash: {ctx.config.hash()}")


@config_app.command("path")
def config_path() -> None:
    """Show where configuration is read from, in precedence order."""
    from video_intel.config.loader import PACKAGED_CONFIG, user_config_path

    console.print("configuration precedence (lowest first):")
    console.print(f"  1. packaged  {PACKAGED_CONFIG}")
    console.print(f"  2. user      {user_config_path()}")
    console.print("  3. project   ./video-intel.yaml (or ./config/video-intel.yaml)")
    console.print("  4. env       VIDEO_INTEL_* variables")
    console.print("  5. cli       explicit command flags")


# ---------------------------------------------------------------------------------
# notebooklm (manual handoff, Phase 8)
# ---------------------------------------------------------------------------------
@notebooklm_app.command("export")
def notebooklm_export(
    video_id: Annotated[str, typer.Argument(help="Video id to build a packet for.")],
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Write a source packet, prompts and the JSON contract for NotebookLM."""
    ctx = _load_context(config, verbose, quiet)
    paths = ctx.project_paths.video(video_id)
    dataset_path = paths.knowledge
    if not dataset_path.is_file():
        err_console.print(
            f"[red]no dataset for {video_id} yet[/red]; run 'video-intel analyze' first."
        )
        raise typer.Exit(code=EXIT_ERROR)

    dataset = read_model(dataset_path, KnowledgeDataset)
    normalized = read_model(paths.normalized_transcript, NormalizedTranscript)
    packet = export_packet(dataset, normalized, paths.outputs)

    console.print("[green]NotebookLM packet written[/green]")
    for path in packet.files():
        console.print(f"  {path.name} ({path.stat().st_size:,} bytes)")
    console.print(f"  directory: {packet.directory}")
    console.print("[dim]" + MANUAL_WORKFLOW_NOTICE.strip().splitlines()[0] + "[/dim]")


@notebooklm_app.command("import")
def notebooklm_import(
    source: Annotated[Path, typer.Argument(help="JSON file produced in NotebookLM.")],
    video_id: Annotated[str, typer.Option("--video", help="Video id to attach the import to.")],
    pipeline: Annotated[
        PipelineId | None,
        typer.Option("--pipeline", help="Override the pipeline the answer belongs to."),
    ] = None,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Rebuild the dataset after importing.")
    ] = True,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Validate a NotebookLM answer and merge it into the video's extractions."""
    ctx = _load_context(config, verbose, quiet)
    paths = ctx.project_paths.video(video_id)
    if not paths.normalized_transcript.is_file():
        err_console.print(f"[red]no transcript stored for {video_id}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    normalized = read_model(paths.normalized_transcript, NormalizedTranscript)
    known = {segment.segment_id for segment in normalized.segments}

    try:
        result = import_records(
            source, video_id=video_id, known_segment_ids=known, pipeline_id=pipeline
        )
    except ImportValidationError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=EXIT_ERROR) from exc

    if result.errors and not result.items:
        err_console.print(f"[red]import rejected:[/red] {'; '.join(result.errors)}")
        raise typer.Exit(code=EXIT_ERROR)

    stored = store_import(result, paths.outputs)
    console.print(f"[green]imported[/green] {result.summary()}")
    console.print(f"  stored in {stored}")

    if rebuild:
        exit_code = _rebuild_dataset(ctx, video_id)
        if exit_code != EXIT_OK:
            raise typer.Exit(code=exit_code)


def _rebuild_dataset(ctx: AppContext, video_id: str) -> int:
    """Re-assemble the dataset from stored records, without calling any provider."""
    project_paths = ctx.project_paths
    paths = project_paths.video(video_id)
    store = ctx.store
    context = StageContext(
        project_id=project_paths.project_id, video_id=video_id, paths=paths, store=store
    )
    runner = StageRunner(context)

    normalized = read_model(paths.normalized_transcript, NormalizedTranscript)
    items: list[ExtractedItem] = []
    runs: list[ExtractionRun] = []

    for record in _stored_records(paths):
        if record.parsed is None:
            continue
        items.extend(record.parsed.items)
        runs.append(
            ExtractionRun(
                pipeline_id=record.pipeline_id,
                model=record.model,
                items_extracted=len(record.parsed.items),
                items_retained=len(record.parsed.items),
            )
        )

    dataset = load_dataset(context)
    merged = build_dataset(
        video_id=video_id,
        title=dataset.title,
        normalized=normalized,
        extraction=ExtractionOutcome(items=items),
        runs=runs,
        transcript_hash=normalized.source_transcript_hash,
        config_hash=ctx.config.hash(),
        completed=[
            StageId.RAW_TRANSCRIPT,
            StageId.NORMALIZED_TRANSCRIPT,
            StageId.CHUNKS,
            StageId.EXTRACTIONS,
            StageId.VERIFY,
            StageId.CONFLICTS,
        ],
        failed=dataset.completeness.failed_stages,
        config=ctx.config,
    ).dataset

    dataset_stage(
        runner=runner,
        dataset=merged,
        input_hash=payload_hash(
            f"{normalized.source_transcript_hash}|{sorted(i.item_id for i in items)}"
        ),
        config_hash=ctx.config.hash(),
        force=True,
    )
    outcomes = output_stages(
        runner=runner,
        dataset=merged,
        stages=set(OUTPUT_STAGES_BY_MODE[OutputMode.DEEP]),
        force=True,
    )
    console.print(
        f"[green]dataset rebuilt[/green] {len(merged.items)} item(s), "
        f"{len(outcomes)} output file(s)"
    )
    return EXIT_OK


def _stored_records(paths: VideoPaths) -> list[RawExtractionRecord]:
    """Every stored extraction record for a video, including NotebookLM imports."""
    records: list[RawExtractionRecord] = []
    if paths.raw_extractions.is_dir():
        for path in sorted(paths.raw_extractions.rglob("*.json")):
            if path.name == "index.json":
                continue
            try:
                payload = read_json(path)
            except (ValueError, FileNotFoundError):
                continue
            entries = payload.get("records") if isinstance(payload, dict) else None
            candidates = entries if isinstance(entries, list) else [payload]
            for entry in candidates:
                try:
                    records.append(RawExtractionRecord.model_validate(entry))
                except ValueError:
                    continue
    records.extend(load_imported_records(paths.outputs))
    return records


# ---------------------------------------------------------------------------------
# web (local frontend over the same CLI)
# ---------------------------------------------------------------------------------
@app.command()
def web(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8765,
    data_dir_web: Annotated[
        Path | None, typer.Option("--data-dir", help="Override the storage root.")
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the page in a browser.")
    ] = False,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Serve the local web frontend (127.0.0.1 only by default)."""
    _load_context(config, verbose, quiet, log_to_file=True)
    from video_intel.web.server import serve

    serve(
        host=host,
        port=port,
        data_dir=str(data_dir_web) if data_dir_web else None,
        config_path=str(config) if config else None,
        open_browser=open_browser,
    )


# ---------------------------------------------------------------------------------
# init (storage bootstrap)
# ---------------------------------------------------------------------------------
@app.command()
def init(
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Create the project directory layout and the state database."""
    ctx = _load_context(config, verbose, quiet, log_to_file=True)
    project_paths = ctx.project_paths.ensure()
    store = ctx.store
    store.initialize()
    store.upsert_project(
        project_paths.project_id,
        ctx.config.project_name,
        config_hash=ctx.config.hash(),
        now=utc_now().isoformat(),
    )

    from video_intel.schemas.manifest import ProjectManifest
    from video_intel.storage.io import write_model_atomic

    manifest = ProjectManifest(
        project_id=project_paths.project_id,
        name=ctx.config.project_name,
        config_hash=ctx.config.hash(),
    )
    write_model_atomic(project_paths.project_file, manifest)

    console.print(f"[green]initialised project[/green] {ctx.config.project_name}")
    console.print(f"  directory: {project_paths.root}")
    console.print(f"  database:  {store.path}")
    console.print(f"  logs:      {project_paths.logs}")


@app.command()
def checkpoint(
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Make the data directory safe to copy to another device.

    SQLite write-ahead logging keeps recent commits - including the schema itself -
    in a ``state.db-wal`` sidecar until they are checkpointed. Copying only
    ``state.db`` before that happens produces a valid-looking file containing
    nothing. This folds the log back into the database so a single file carries
    its own data, which matters when the directory lives on a USB stick, a sync
    client or a network share.
    """
    ctx = _load_context(config, verbose, quiet, log_to_file=False)
    store = ctx.store
    store.checkpoint()

    from video_intel.storage.db import db_diagnostics

    report = db_diagnostics(store.path)
    console.print(f"[green]checkpointed[/green] {store.path}")
    if report["sidecars"]:
        console.print(
            f"[yellow]note:[/yellow] sidecars still present: {', '.join(report['sidecars'])}"
        )
    console.print(f"  size:     {report['size_bytes']:,} bytes")
    console.print(f"  schema:   v{report['schema_version']}")
    console.print(f"  journal:  {report['journal_mode']}")


@app.command()
def doctor(
    config: ConfigOption = None,
    verbose: VerboseOption = False,
    quiet: QuietOption = False,
) -> None:
    """Report what this machine can do: configuration, paths, database and providers.

    Read-only: it creates no directories and writes no state, so it is safe to run on a
    machine that is not set up yet - which is exactly when it is most useful.
    """
    ctx = _load_context(config, verbose, quiet, log_to_file=False)

    from video_intel.config.root import find_project_root
    from video_intel.storage.db import db_diagnostics

    paths = ctx.project_paths
    report = db_diagnostics(ctx.store.path)

    table = Table(title="video-intel", header_style="bold", show_header=False)
    table.add_column("field")
    table.add_column("value")
    table.add_row("project root", str(find_project_root()))
    table.add_row("project", ctx.config.project_name)
    table.add_row("data dir", str(paths.root))
    table.add_row("database", str(report["path"]))
    table.add_row("config files", ", ".join(Path(p).name for p in ctx.config.source_files))
    table.add_row("execution mode", str(ctx.config.execution_mode))
    table.add_row("config hash", ctx.config.hash()[:16])
    console.print(table)

    if not report["exists"]:
        console.print("[yellow]no state database yet[/yellow] — run: video-intel init")
    else:
        console.print(
            f"state db       {report['size_bytes']:,} bytes, "
            f"schema v{report['schema_version']}, journal {report['journal_mode']}"
        )
        if report["sidecars"]:
            console.print(
                f"[yellow]uncheckpointed sidecars:[/yellow] {', '.join(report['sidecars'])}"
                "\n  run `video-intel checkpoint` before copying data/ to another device"
            )
        if report["error"]:
            console.print(f"[yellow]database:[/yellow] {report['error']}")

    videos = len(paths.videos())
    if videos:
        console.print(f"[green]data directory ready[/green] — {videos} stored video(s)")

    providers = Table(title="providers", header_style="bold")
    providers.add_column("provider")
    providers.add_column("kind")
    providers.add_column("model")
    providers.add_column("key")
    for name, provider in sorted(ctx.config.providers.items()):
        providers.add_row(
            name,
            str(provider.kind),
            provider.default_model or "-",
            provider.api_key_env
            if provider.resolve_api_key()
            else f"missing ({provider.api_key_env})",
        )
    console.print(providers)


def cli() -> None:
    """Console-script entry point.

    Named ``cli`` rather than ``main`` so it does not shadow this module inside the
    ``video_intel.cli`` package.
    """
    app()


if __name__ == "__main__":  # pragma: no cover
    cli()
