"""``OSINT`` — the full interactive TUI for reel-vault.

A single Textual application that covers every stage of the pipeline:

* Import an official Instagram export
* Run fetch → transcribe → triage → extract → classify → media
* Review and correct reels (approve, reclassify, force decision, add note, open URL)
* Export documents (Markdown, JSON, PDF, Obsidian)
* Inspect configuration and run the doctor check

Everything is offline and local-first.
"""

from __future__ import annotations

import datetime
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from reel_vault.cli.context import AppContext, build_context
from reel_vault.importer import parse_export
from reel_vault.media.base import DriveUnavailable
from reel_vault.outputs.render import SUPPORTED_FORMATS, export_all
from reel_vault.pipeline.runner import RunOptions, run_pipeline
from reel_vault.storage.db import Store
from reel_vault.storage.io import write_json_atomic

APP_CSS = """
Screen {
    layout: vertical;
    background: #050a14;
    color: #e2e8f0;
}

#title {
    dock: top;
    height: 4;
    background: linear-gradient(90deg, #0f172a, #1e293b);
    color: #f8fafc;
    text-style: bold;
    text-align: center;
    content-align: center middle;
    padding: 1;
}

#subtitle {
    dock: top;
    height: 1;
    background: #050a14;
    color: #94a3b8;
    text-align: center;
    text-style: italic;
}

#main {
    height: 1fr;
    padding: 1 2;
}

#stats {
    height: 7;
    border: solid #334155;
    margin: 0 1;
    padding: 1;
    background: #0f172a;
}

.stat-row {
    height: 1;
    color: #cbd5e1;
    margin: 0 0;
}

#tabs-container {
    height: 1fr;
}

TabbedContent {
    height: 1fr;
    background: #050a14;
}

.tab-pane {
    height: 1fr;
    padding: 1;
}

#log {
    dock: bottom;
    height: 12;
    border: solid #334155;
    margin: 0 1;
    padding: 1;
    background: #020617;
    color: #94a3b8;
    overflow-y: auto;
}

#action-bar {
    dock: bottom;
    height: 3;
    background: #0f172a;
    color: #cbd5e1;
    padding: 0 2;
    text-align: center;
}

.dialog {
    align: center middle;
}

.dialog #panel {
    width: 70;
    height: auto;
    min-height: 14;
    border: thick #38bdf8;
    padding: 1;
    background: #0f172a;
    layout: vertical;
}

.dialog-title {
    text-style: bold;
    color: #38bdf8;
    text-align: center;
    margin-bottom: 1;
}

.button-row {
    height: 3;
    margin: 1 0;
    align: center middle;
}

.button-row Button {
    margin: 0 1;
}

.card-title {
    text-style: bold;
    color: #f8fafc;
    margin-bottom: 1;
}

.status-ok {
    color: #22c55e;
}

.status-warn {
    color: #eab308;
}

.status-error {
    color: #ef4444;
}

#reel-table, #review-table {
    background: #0f172a;
}

DataTable > .table-header {
    background: #1e293b;
    color: #94a3b8;
    text-style: bold;
}

DataTable > .table-row {
    background: #0f172a;
}

DataTable > .table-row--cursor {
    background: #1e293b;
    color: #38bdf8;
}

TabbedContent .tab-bar {
    height: 3;
    background: #0f172a;
}

TabbedContent .tab-bar .tab--active {
    background: #38bdf8;
    color: #050a14;
    text-style: bold;
}

TabbedContent .tab-bar .tab--inactive {
    background: #050a14;
    color: #64748b;
}

Input {
    background: #020617;
    border: solid #334155;
    color: #f8fafc;
    margin: 0 0;
}

Select {
    background: #020617;
    border: solid #334155;
    color: #f8fafc;
    margin: 0 0;
}

Static {
    margin: 0 0;
}
"""

KEY_HINT_TEXT = (
    "[b]1[/b] Import  [b]2[/b] Run Pipeline  [b]3[/b] Review  [b]4[/b] Export  "
    "[b]5[/b] Config  [b]6[/b] Doctor  [b]q[/b] Quit"
)


class OSINTApp(App[None]):
    """The full reel-vault TUI."""

    CSS = APP_CSS
    TITLE = "reel-vault — OSINT archive"

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.context = context
        self.paths = context.paths
        self.config = context.config
        self.store: Store = context.store()
        self._selected_row_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("reel-vault", id="title")
        yield Label("Official Instagram export → classified, archived reel library", id="subtitle")
        with Vertical(id="main"):
            yield self._stats_panel()
            with TabbedContent(id="tabs-container"):
                yield TabPane("Queue", id="tab-queue")
                yield TabPane("Review", id="tab-review")
                yield TabPane("Export", id="tab-export")
                yield TabPane("Config", id="tab-config")
                yield TabPane("Doctor", id="tab-doctor")
        yield VerticalScroll(id="log")
        yield Static(KEY_HINT_TEXT, id="action-bar")
        yield Footer()

    def _stats_panel(self) -> Vertical:
        panel = Vertical(id="stats")
        panel.mount(Label("Library stats", classes="card-title"))
        self._stats_total = Label("Total reels: 0")
        self._stats_by_status = Label("By status: —")
        self._stats_cost = Label("Estimated cost: $0.0000")
        panel.mount(self._stats_total)
        panel.mount(self._stats_by_status)
        panel.mount(self._stats_cost)
        return panel

    def on_mount(self) -> None:
        self._refresh_stats()
        self._load_queue()
        self._load_review()
        self._load_config()
        self._load_doctor()

    # ------------------------------------------------------------------ stats

    def _refresh_stats(self) -> None:
        with self.store as store:
            stats = store.stats()
        total = stats.get("reels", 0)
        by_status = stats.get("by_status", {})
        status_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) or "—"
        self._stats_total.update(f"Total reels: {total}")
        self._stats_by_status.update(f"By status: {status_str}")
        self._stats_cost.update(f"Estimated cost: ${stats.get('cost_usd', 0.0):.4f}")

    def _write_log(self, message: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static(message))
        log.scroll_end()

    # ------------------------------------------------------------------ queue tab

    def _load_queue(self) -> None:
        try:
            tab = self.query_one("#tab-queue")
        except Exception:
            return
        tab.remove_children()
        tab.mount(Label("Saved reels", classes="card-title"))

        with self.store as store:
            reels = store.list_reels(limit=200)

        table: DataTable[Any] = DataTable(id="reel-table", zebra_stripes=True, cursor_type="row")
        for col in ("shortcode", "status", "decision", "score", "category", "caption"):
            table.add_column(col, key=col)

        for reel in reels:
            shortcode = str(reel["shortcode"])
            classification = store.get_classification(shortcode)
            triage = store.get_triage(shortcode)
            table.add_row(
                shortcode,
                str(reel["status"]),
                str((triage or {}).get("decision") or "—"),
                str((triage or {}).get("value_score") or 0),
                (classification or {}).get("primary") or "—",
                str(reel["caption"] or "")[:60],
                key=shortcode,
            )
        tab.mount(table)

    # ------------------------------------------------------------------ review tab

    def _load_review(self) -> None:
        try:
            tab = self.query_one("#tab-review")
        except Exception:
            return
        tab.remove_children()
        tab.mount(Label("Needs review", classes="card-title"))

        button_row = Vertical(classes="button-row")
        button_row.mount(Button("Approve selected", id="btn-approve", variant="success"))
        button_row.mount(Button("Reclassify selected", id="btn-reclassify", variant="warning"))
        button_row.mount(Button("Force decision", id="btn-force", variant="warning"))
        button_row.mount(Button("Add note", id="btn-note", variant="primary"))
        button_row.mount(Button("Open URL", id="btn-open", variant="default"))
        tab.mount(button_row)

        table: DataTable[Any] = DataTable(id="review-table", zebra_stripes=True, cursor_type="row")
        for col in ("shortcode", "score", "category", "decision", "reason"):
            table.add_column(col, key=col)

        with self.store as store:
            reels = store.list_reels(limit=200)
            for reel in reels:
                shortcode = str(reel["shortcode"])
                triage = store.get_triage(shortcode)
                if not (triage or {}).get("needs_review"):
                    continue
                classification = store.get_classification(shortcode)
                table.add_row(
                    shortcode,
                    str((triage or {}).get("value_score") or 0),
                    (classification or {}).get("primary") or "—",
                    str((triage or {}).get("decision") or "—"),
                    str((triage or {}).get("reason") or "")[:60],
                    key=shortcode,
                )
        tab.mount(table)

    # ------------------------------------------------------------------ export tab

    def _load_export(self) -> None:
        try:
            tab = self.query_one("#tab-export")
        except Exception:
            return
        tab.remove_children()
        tab.mount(Label("Export documents", classes="card-title"))
        formats = ", ".join(SUPPORTED_FORMATS)
        tab.mount(Label(f"Supported formats: {formats}"))
        tab.mount(Input(placeholder="Output root (optional)", id="inp-export-out"))
        tab.mount(Button("Export all formats", id="btn-export", variant="success"))
        tab.mount(Static(id="export-result"))

    # ------------------------------------------------------------------ config tab

    def _load_config(self) -> None:
        try:
            tab = self.query_one("#tab-config")
        except Exception:
            return
        tab.remove_children()
        tab.mount(Label("Configuration", classes="card-title"))
        cfg = self.config.redaction_safe_dump()
        for key, value in cfg.items():
            tab.mount(Static(f"{key}: {value}"))

    # ------------------------------------------------------------------ doctor tab

    def _load_doctor(self) -> None:
        try:
            tab = self.query_one("#tab-doctor")
        except Exception:
            return
        tab.remove_children()
        tab.mount(Label("Doctor check", classes="card-title"))
        tab.mount(Button("Run doctor", id="btn-doctor", variant="primary"))
        tab.mount(Static(id="doctor-result"))

    # ------------------------------------------------------------------ keys

    def on_key(self, event: Key) -> None:
        key = event.key
        if key == "q":
            self.app.exit()
        elif key == "1":
            self._push_import()
        elif key == "2":
            self._run_pipeline()
        elif key == "3":
            tabs = self.query_one("#tabs-container", TabbedContent)
            tabs.active = "tab-review"
        elif key == "4":
            tabs = self.query_one("#tabs-container", TabbedContent)
            tabs.active = "tab-export"
        elif key == "5":
            tabs = self.query_one("#tabs-container", TabbedContent)
            tabs.active = "tab-config"
        elif key == "6":
            tabs = self.query_one("#tabs-container", TabbedContent)
            tabs.active = "tab-doctor"

    # ------------------------------------------------------------------ button actions

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-run":
            self._run_pipeline()
        elif button_id == "btn-import":
            self._push_import()
        elif button_id == "btn-export":
            self._do_export()
        elif button_id == "btn-doctor":
            self._do_doctor()
        elif button_id == "btn-approve":
            self._do_approve()
        elif button_id == "btn-reclassify":
            self._do_reclassify()
        elif button_id == "btn-force":
            self._do_force()
        elif button_id == "btn-note":
            self._do_note()
        elif button_id == "btn-open":
            self._do_open()
        elif button_id in {"btn-save", "btn-cancel"}:
            pass  # handled by sub-screens

    # ------------------------------------------------------------------ pipeline

    def _run_pipeline(self) -> None:
        self._write_log("Starting pipeline run…")
        try:
            context = build_context()
        except Exception as exc:
            self._write_log(f"[red]config error: {exc}[/red]")
            return
        options = RunOptions(limit=None, dry_run=False, depth="adaptive")
        try:
            code = run_pipeline(context, options)
        except DriveUnavailable as exc:
            self._write_log(f"[red]drive unavailable: {exc}[/red]")
            return
        except Exception as exc:
            self._write_log(f"[red]pipeline error: {exc}[/red]")
            return
        self._write_log(f"[green]pipeline finished with exit code {code}[/green]")
        self._refresh_stats()
        self._load_queue()
        self._load_review()

    # ------------------------------------------------------------------ import

    def _push_import(self) -> None:
        self.app.push_screen(ImportScreen(self.context, self._on_import_done))

    def _on_import_done(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self._write_log(f"[green]imported {result.get('reels', 0)} reels[/green]")
        self._refresh_stats()
        self._load_queue()
        self._load_review()

    # ------------------------------------------------------------------ export

    def _do_export(self) -> None:
        try:
            out_input = self.query_one("#inp-export-out", Input)
            out = Path(out_input.value) if out_input.value.strip() else None
        except Exception:
            out = None
        self._write_log("Exporting documents…")
        try:
            context = build_context()
        except Exception as exc:
            self._write_log(f"[red]config error: {exc}[/red]")
            return
        code = export_all(context, formats=",".join(SUPPORTED_FORMATS), out=out)
        target = out or context.paths.outputs
        self._write_log(f"[green]export finished (code {code}) → {target}[/green]")

    # ------------------------------------------------------------------ doctor

    def _do_doctor(self) -> None:
        self._write_log("Running doctor…")
        try:
            import shutil

            checks = {
                "ffmpeg": shutil.which("ffmpeg"),
                "ffprobe": shutil.which("ffprobe"),
                "yt-dlp": "yt_dlp",
                "reportlab": "reportlab",
                "textual": "textual",
                "rapidocr": "rapidocr",
            }
            from importlib.util import find_spec

            results = []
            for name, check in checks.items():
                if isinstance(check, str) and not check.startswith("/"):
                    ok = find_spec(check) is not None
                else:
                    ok = check is not None
                results.append(f"  {name}: {'ok' if ok else 'missing'}")
            self._write_log("\n".join(results))
        except Exception as exc:
            self._write_log(f"[red]doctor error: {exc}[/red]")
            return
        self._write_log("[green]doctor check complete[/green]")

    # ------------------------------------------------------------------ review helpers

    def _selected_shortcode(self, table_id: str) -> str | None:
        try:
            table = self.query_one(table_id, DataTable)
        except Exception:
            return None
        cursor = table.cursor_row
        if cursor < 0:
            return None
        try:
            row = table.get_row_at(cursor)
        except Exception:
            return None
        return str(row[0]) if row else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Track the currently selected row key."""
        self._selected_row_key = str(event.row_key.value)

    def _do_approve(self) -> None:
        shortcode = self._selected_shortcode("#review-table")
        if not shortcode:
            shortcode = getattr(self, "_selected_row_key", None)
        if not shortcode:
            self._write_log("[yellow]no reel selected[/yellow]")
            return
        with self.store as store:
            existing = store.get_review(shortcode) or {}
            existing["approved"] = True
            existing["approved_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            store.save_review(shortcode, existing)
        self._write_log(f"[green]approved {shortcode}[/green]")
        self._load_review()

    def _do_reclassify(self) -> None:
        shortcode = self._selected_shortcode("#review-table")
        if not shortcode:
            self._write_log("[yellow]no reel selected[/yellow]")
            return
        self.app.push_screen(ReclassifyScreen(shortcode, self.store, self._on_reclassified))

    def _on_reclassified(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self._write_log(
            f"[green]reclassified {result.get('shortcode')}: {result.get('primary')}[/green]"
        )
        self._load_review()

    def _do_force(self) -> None:
        shortcode = self._selected_shortcode("#review-table")
        if not shortcode:
            self._write_log("[yellow]no reel selected[/yellow]")
            return
        self.app.push_screen(ForceScreen(shortcode, self.store, self._on_forced))

    def _on_forced(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self._write_log(
            f"[green]forced {result.get('shortcode')} → {result.get('decision')}[/green]"
        )
        self._load_review()

    def _do_note(self) -> None:
        shortcode = self._selected_shortcode("#review-table")
        if not shortcode:
            self._write_log("[yellow]no reel selected[/yellow]")
            return
        self.app.push_screen(NoteScreen(shortcode, self.store, self._on_note))

    def _on_note(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self._write_log(f"[green]note added to {result.get('shortcode')}[/green]")
        self._load_review()

    def _do_open(self) -> None:
        shortcode = self._selected_shortcode("#review-table")
        if not shortcode:
            self._write_log("[yellow]no reel selected[/yellow]")
            return
        with self.store as store:
            reel = store.get_reel(shortcode)
        if reel:
            webbrowser.open(reel["url"])
            self._write_log(f"[green]opened {reel['url']}[/green]")


# --------------------------------------------------------------------------- sub screens


class ImportScreen(Screen[dict[str, Any] | None]):
    """Import an Instagram export ZIP or directory."""

    CSS = """
    Screen {
        align: center middle;
    }
    #panel {
        width: 70;
        height: 20;
        border: thick #38bdf8;
        padding: 1;
        background: #0f172a;
        layout: vertical;
    }
    """

    def __init__(
        self,
        context: AppContext,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.context = context
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        yield Label("Import Instagram export", classes="dialog-title")
        yield Input(placeholder="Path to ZIP or unpacked export directory", id="inp-path")
        with Horizontal(classes="button-row"):
            yield Button("Import", id="btn-import", variant="success")
            yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.pop_screen()
            return
        path = self.query_one("#inp-path", Input).value.strip()
        if not path:
            self.notify("Please enter a path", severity="error")
            return
        try:
            context = build_context()
        except Exception as exc:
            self.notify(f"Config error: {exc}", severity="error")
            return
        try:
            report = parse_export(path)
        except Exception as exc:
            self.notify(f"Import failed: {exc}", severity="error")
            return
        store = Store(context.paths.database)
        with store:
            counts = store.import_report(report)
            for reel in report.reels:
                reel_paths = context.paths.reel(reel.shortcode).ensure()
                write_json_atomic(reel_paths.source, reel.model_dump(mode="json"))
        self.on_done({"reels": counts.get("reels", 0), "posts": counts.get("posts", 0)})
        self.app.pop_screen()


class ReclassifyScreen(Screen[dict[str, Any] | None]):
    """Reclassify a reel's categories."""

    CSS = """
    Screen {
        align: center middle;
    }
    #panel {
        width: 60;
        height: 16;
        border: thick #38bdf8;
        padding: 1;
        background: #0f172a;
        layout: vertical;
    }
    """

    CATEGORIES = (
        "learning & study tricks",
        "news & information",
        "comedy & funny",
        "tech & AI",
        "money & business",
        "health & fitness",
        "food & recipes",
        "travel & places",
        "style & design",
        "life hacks & DIY",
        "tools & resources",
        "inspiration & motivation",
        "psychology & mind",
        "other",
    )

    def __init__(
        self,
        shortcode: str,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.shortcode = shortcode
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        classification = self.store.get_classification(self.shortcode)
        current = (classification or {}).get("primary") or "other"
        yield Label(f"Reclassify {self.shortcode}", classes="dialog-title")
        yield Label("Primary category:")
        yield Select[str](
            options=[(c, c) for c in self.CATEGORIES],
            value=current,
            id="sel-primary",
        )
        with Horizontal(classes="button-row"):
            yield Button("Save", id="btn-save", variant="success")
            yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.pop_screen()
            return
        primary = str(self.query_one("#sel-primary", Select[str]).value)
        with self.store as store:
            existing = store.get_review(self.shortcode) or {}
            existing["primary"] = primary
            store.save_review(self.shortcode, existing)
        self.on_done({"shortcode": self.shortcode, "primary": primary})
        self.app.pop_screen()


class ForceScreen(Screen[dict[str, Any] | None]):
    """Force a triage decision."""

    CSS = """
    Screen {
        align: center middle;
    }
    #panel {
        width: 55;
        height: 14;
        border: thick #38bdf8;
        padding: 1;
        background: #0f172a;
        layout: vertical;
    }
    """

    def __init__(
        self,
        shortcode: str,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.shortcode = shortcode
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        triage = self.store.get_triage(self.shortcode) or {}
        current = str(triage.get("decision") or "extract")
        yield Label(f"Force decision for {self.shortcode}", classes="dialog-title")
        yield Select[str](
            options=[("extract", "extract"), ("archive", "archive"), ("stub", "stub")],
            value=current,
            id="sel-decision",
        )
        with Horizontal(classes="button-row"):
            yield Button("Save", id="btn-save", variant="success")
            yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.pop_screen()
            return
        decision = str(self.query_one("#sel-decision", Select[str]).value)
        triage = self.store.get_triage(self.shortcode) or {}
        triage["decision"] = decision
        triage["force_decision"] = decision
        triage["forced_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        self.store.save_triage(self.shortcode, triage)
        self.on_done({"shortcode": self.shortcode, "decision": decision})
        self.app.pop_screen()


class NoteScreen(Screen[dict[str, Any] | None]):
    """Add a note to a reel."""

    CSS = """
    Screen {
        align: center middle;
    }
    #panel {
        width: 70;
        height: 16;
        border: thick #38bdf8;
        padding: 1;
        background: #0f172a;
        layout: vertical;
    }
    """

    def __init__(
        self,
        shortcode: str,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.shortcode = shortcode
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        yield Label(f"Add note to {self.shortcode}", classes="dialog-title")
        yield Input(placeholder="Your note…", id="inp-note")
        with Horizontal(classes="button-row"):
            yield Button("Save", id="btn-save", variant="success")
            yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.pop_screen()
            return
        note = self.query_one("#inp-note", Input).value.strip()
        import datetime as _dt

        with self.store as store:
            existing = store.get_review(self.shortcode) or {}
            notes = existing.get("notes", [])
            if isinstance(notes, str):
                notes = [notes]
            notes.append(note)
            existing["notes"] = notes
            existing["noted_at"] = _dt.datetime.now(datetime.UTC).isoformat()
            store.save_review(self.shortcode, existing)
        self.on_done({"shortcode": self.shortcode, "note": note})
        self.app.pop_screen()


def run_tui(context: AppContext) -> int:
    """Launch the full OSINT TUI. Returns 0 on clean exit."""
    app = OSINTApp(context)
    app.run()
    return 0


def run_tui_app() -> None:
    """Entry point for the ``OSINT`` console script."""
    context = build_context()
    app = OSINTApp(context)
    app.run()


__all__ = ["OSINTApp", "run_tui", "run_tui_app"]
