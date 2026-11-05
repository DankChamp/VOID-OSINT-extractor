"""``reel-vault`` command line interface."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from reel_vault.cli.context import context_from_typer
from reel_vault.importer import parse_export
from reel_vault.storage.io import write_json_atomic
from reel_vault.version import __version__

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_DRIVE_LOST = 4
EXIT_LIMIT = 5

app = typer.Typer(
    name="reel-vault",
    help="Official Instagram export -> classified, archived reel library.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect and validate configuration.")
app.add_typer(config_app, name="config")

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"reel-vault {__version__}")
        raise typer.Exit(code=EXIT_OK)


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a configuration file.", show_default=False),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Override the artifact directory.", show_default=False),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Override the SQLite database.", show_default=False),
    ] = None,
    media_dir: Annotated[
        Path | None,
        typer.Option("--media-dir", help="Override the media directory.", show_default=False),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress console logging.")] = False,
) -> None:
    overrides: dict[str, Any] = {}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if db_path is not None:
        overrides["db_path"] = db_path
    if media_dir is not None:
        overrides["media_dir"] = media_dir
    ctx.obj = {
        "config": config,
        "overrides": overrides,
        "verbose": verbose,
        "quiet": quiet,
    }


# --------------------------------------------------------------------------- config


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the effective configuration with paths resolved."""
    context = context_from_typer(ctx)
    console.print_json(json.dumps(context.config.redaction_safe_dump(), indent=2))


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """Validate the merged configuration."""
    context = context_from_typer(ctx)
    console.print("[green]configuration is valid[/green]")
    for source in context.config.source_files:
        console.print(f"  {source}")


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Show where configuration is read from."""
    context = context_from_typer(ctx)
    if not context.config.source_files:
        console.print("packaged defaults only")
        return
    for source in context.config.source_files:
        console.print(source)


# --------------------------------------------------------------------------- doctor


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Report what this device can do."""
    context = context_from_typer(ctx)
    config = context.config
    paths = context.paths

    table = Table(title="reel-vault doctor", show_header=False)
    table.add_column("check")
    table.add_column("result")

    table.add_row("version", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("data dir", str(config.data_dir))
    table.add_row("database", str(config.db_path))
    table.add_row("media dir", str(config.media_dir))

    for label, binary in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")):
        found = shutil.which(binary)
        table.add_row(label, found or "[red]missing[/red]")

    for label, module in (
        ("yt-dlp", "yt_dlp"),
        ("textual", "textual"),
        ("reportlab", "reportlab"),
        ("rapidocr", "rapidocr"),
        ("beautifulsoup4", "bs4"),
        ("ftfy", "ftfy"),
    ):
        table.add_row(
            label, "available" if _module_available(module) else "[yellow]missing[/yellow]"
        )

    key_env = config.groq.api_key_env
    table.add_row(
        "Groq key",
        f"{key_env} is set" if os.environ.get(key_env) else f"[yellow]{key_env} missing[/yellow]",
    )

    if config.media_dir.exists():
        usage = shutil.disk_usage(config.media_dir)
        table.add_row("media free", f"{_human_bytes(usage.free)} of {_human_bytes(usage.total)}")
        from reel_vault.media.drive import check_reachable

        table.add_row(
            "media writable",
            "yes" if check_reachable(config.media_dir) else "[red]no[/red]",
        )
    else:
        table.add_row("media dir", "[red]not reachable[/red]")

    console.print(table)
    if not paths.database.exists():
        console.print("[dim]database will be created on first import[/dim]")


# --------------------------------------------------------------------------- import


@app.command("import")
def import_export(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Export ZIP or unpacked export directory.")],
) -> None:
    """Import an official Instagram export (ZIP or directory)."""
    context = context_from_typer(ctx)
    try:
        report = parse_export(path)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]import failed:[/red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc

    with context.store() as store:
        counts = store.import_report(report)

    for reel in report.reels:
        reel_paths = context.paths.reel(reel.shortcode).ensure()
        write_json_atomic(reel_paths.source, reel.model_dump(mode="json"))

    console.print(
        f"[green]imported[/green] {counts['reels']} reels, {counts['posts']} posts, "
        f"{counts['collections']} collections"
    )
    if report.warnings:
        console.print(f"[yellow]{len(report.warnings)} warning(s)[/yellow]")
        for warning in report.warnings[:10]:
            console.print(f"  {warning}")


# --------------------------------------------------------------------------- listing


@app.command("list")
def list_reels(
    ctx: typer.Context,
    status: Annotated[str | None, typer.Option("--status", help="Filter by status.")] = None,
    category: Annotated[
        str | None, typer.Option("--category", help="Filter by primary category.")
    ] = None,
    search: Annotated[
        str | None, typer.Option("--search", help="Search author, caption or shortcode.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum rows.")] = 50,
) -> None:
    """List imported reels."""
    context = context_from_typer(ctx)
    with context.store() as store:
        reels = store.list_reels(status=status, category=category, search=search, limit=limit)
        table = Table(title=f"reels ({len(reels)})")
        table.add_column("shortcode")
        table.add_column("type")
        table.add_column("author")
        table.add_column("status")
        table.add_column("category")
        table.add_column("saved")
        table.add_column("caption", overflow="fold")
        for reel in reels:
            classification = store.get_classification(reel["shortcode"])
            table.add_row(
                reel["shortcode"],
                reel["media_type"],
                reel["author"] or "-",
                reel["status"],
                (classification or {}).get("primary") or "-",
                (reel["saved_at"] or "-")[:10],
                (reel["caption"] or "")[:60],
            )
    console.print(table)


@app.command()
def show(
    ctx: typer.Context,
    shortcode: Annotated[str, typer.Argument(help="Instagram shortcode.")],
) -> None:
    """Show everything stored for one reel."""
    context = context_from_typer(ctx)
    with context.store() as store:
        reel = store.get_reel(shortcode)
        if reel is None:
            err_console.print(f"[red]unknown shortcode:[/red] {shortcode}")
            raise typer.Exit(code=EXIT_ERROR)
        payload: dict[str, Any] = {"reel": reel}
        for name, value in (
            ("metadata", store.get_metadata(shortcode)),
            ("transcript", store.get_transcript(shortcode)),
            ("triage", store.get_triage(shortcode)),
            ("extraction", store.get_extraction(shortcode)),
            ("classification", store.get_classification(shortcode)),
            ("review", store.get_review(shortcode)),
        ):
            if value is not None:
                payload[name] = value
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command()
def stats(ctx: typer.Context) -> None:
    """Counts, categories and recorded cost."""
    context = context_from_typer(ctx)
    with context.store() as store:
        data = store.stats()
    console.print_json(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- enrichment


@app.command("fetch")
def fetch_metadata(
    ctx: typer.Context,
    limit: Annotated[int | None, typer.Option("--limit", help="Fetch at most N reels.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Refetch reels that already succeeded.")
    ] = False,
    shortcode: Annotated[
        str | None, typer.Option("--shortcode", help="Fetch one specific reel.")
    ] = None,
    delay: Annotated[
        float | None,
        typer.Option("--delay", help="Override the delay between requests, in seconds."),
    ] = None,
) -> None:
    """Fetch public metadata and thumbnails anonymously."""
    from reel_vault.enrich.service import EnrichmentService

    context = context_from_typer(ctx)
    with context.store() as store:
        if shortcode is not None and store.get_reel(shortcode) is None:
            err_console.print(f"[red]unknown shortcode:[/red] {shortcode}")
            raise typer.Exit(code=EXIT_ERROR)
        service = EnrichmentService(store, context.paths, context.config, request_delay=delay)
        summary = service.fetch_pending(
            limit=limit,
            force=force,
            shortcodes=[shortcode] if shortcode else None,
        )

    console.print(
        f"[green]fetched[/green] {summary.ok} ok, "
        f"{summary.unavailable} unavailable, {summary.failed} failed, "
        f"{summary.skipped} skipped"
    )
    if summary.statuses:
        for status, count in sorted(summary.statuses.items()):
            console.print(f"  {status}: {count}")


# --------------------------------------------------------------------------- media


@app.command("download")
def download_media(
    ctx: typer.Context,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="none | archive | all (default: configuration)."),
    ] = None,
    quality: Annotated[
        str | None,
        typer.Option("--quality", help="best | 1080 | 720 (default: configuration)."),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Download at most N reels.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-download existing media.")] = False,
    shortcode: Annotated[
        str | None, typer.Option("--shortcode", help="Download one specific reel.")
    ] = None,
) -> None:
    """Download and verify media for eligible reels."""
    from reel_vault.config.models import DownloadMode, Quality
    from reel_vault.media.base import DriveUnavailable
    from reel_vault.media.service import MediaService

    context = context_from_typer(ctx)
    selected_mode = mode or context.config.download.mode
    selected_quality = quality or context.config.download.quality
    if selected_mode not in {"none", "archive", "all"}:
        err_console.print(f"[red]invalid --mode:[/red] {selected_mode}")
        raise typer.Exit(code=EXIT_CONFIG)
    if selected_quality not in {"best", "1080", "720"}:
        err_console.print(f"[red]invalid --quality:[/red] {selected_quality}")
        raise typer.Exit(code=EXIT_CONFIG)

    with context.store() as store:
        if shortcode is not None and store.get_reel(shortcode) is None:
            err_console.print(f"[red]unknown shortcode:[/red] {shortcode}")
            raise typer.Exit(code=EXIT_ERROR)
        service = MediaService(store, context.paths, context.config)
        try:
            summary = service.download_pending(
                mode=cast(DownloadMode, selected_mode),
                quality=cast(Quality, selected_quality),
                limit=limit,
                force=force,
                shortcodes=[shortcode] if shortcode else None,
            )
        except DriveUnavailable as exc:
            err_console.print(f"[red]media drive unavailable:[/red] {exc}")
            raise typer.Exit(code=EXIT_DRIVE_LOST) from exc

    console.print(
        f"[green]media[/green] {summary.downloaded} downloaded, "
        f"{summary.skipped} skipped, {summary.failed} failed, "
        f"{_human_bytes(summary.bytes)}"
    )
    for code, reason in sorted(summary.failures.items()):
        console.print(f"  [yellow]{code}[/yellow]: {reason}")
    if selected_mode == "archive" and summary.downloaded == 0 and summary.skipped:
        console.print(
            "[dim]archive mode downloads only posts triaged as archive; "
            "run `reel-vault run` or use --mode all.[/dim]"
        )


# --------------------------------------------------------------------------- pipeline


@app.command()
def run(
    ctx: typer.Context,
    limit: Annotated[int | None, typer.Option("--limit", help="Process at most N reels.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only; call no models.")] = False,
    depth: Annotated[
        str, typer.Option("--depth", help="adaptive | light | standard | full.")
    ] = "adaptive",
) -> None:
    """Fetch, transcribe, triage, extract and classify new reels."""
    from reel_vault.pipeline.runner import RunOptions, run_pipeline

    context = context_from_typer(ctx)
    options = RunOptions(limit=limit, dry_run=dry_run, depth=depth)
    code = run_pipeline(context, options)
    raise typer.Exit(code=code)


@app.command()
def export(
    ctx: typer.Context,
    formats: Annotated[
        str, typer.Option("--formats", help="Comma-separated: md,json,html,pdf,obsidian.")
    ] = "md,json,html",
    out: Annotated[Path | None, typer.Option("--out", help="Output root.")] = None,
) -> None:
    """Generate documents from the current dataset."""
    from reel_vault.outputs.render import export_all

    context = context_from_typer(ctx)
    code = export_all(context, formats=formats, out=out)
    raise typer.Exit(code=code)


@app.command()
def review(ctx: typer.Context) -> None:
    """Open the interactive review queue."""
    from reel_vault.review.tui import run_review

    context = context_from_typer(ctx)
    raise typer.Exit(code=run_review(context))


@app.command()
def classify(
    ctx: typer.Context,
    shortcode: Annotated[str, typer.Argument(help="Instagram shortcode.")],
    primary: Annotated[
        str | None, typer.Option("--primary", help="Set the primary category.")
    ] = None,
    add: Annotated[
        list[str] | None, typer.Option("--add", help="Add a secondary category.")
    ] = None,
) -> None:
    """Manually override a reel's categories."""
    context = context_from_typer(ctx)
    with context.store() as store:
        reel = store.get_reel(shortcode)
        if reel is None:
            err_console.print(f"[red]unknown shortcode:[/red] {shortcode}")
            raise typer.Exit(code=EXIT_ERROR)
        existing = store.get_review(shortcode) or {}
        overrides = dict(existing.get("overrides") or {})
        if primary:
            overrides["primary"] = primary
        if add:
            overrides["add"] = sorted(set(overrides.get("add") or []) | set(add))
        existing["overrides"] = overrides
        store.save_review(shortcode, existing)
    console.print(f"[green]saved[/green] review overrides for {shortcode}")


__all__ = ["app", "main_callback"]


if __name__ == "__main__":  # pragma: no cover
    app()
