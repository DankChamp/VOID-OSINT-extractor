"""Interactive review queue — a fully offline Textual TUI.

Left pane: filtered reel list (shortcode, score, category, status, decision).
Right pane: caption, transcript excerpt, triage reason, extracted items, links.

Keys:
  a  approve        mark the selected reel as reviewed
  r  reclassify     pick primary + secondary categories (multi-select)
  f  force          change the triage decision (archive / extract / stub)
  n  note           add a free-text note
  o  open           open the reel URL in the default browser
  /  search         filter the reel list by shortcode, author or caption
  q  quit           leave the TUI

Every action writes to ``review.json`` so the TUI never needs a model call
to correct a label.
"""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rich.text import Text  # noqa: F401
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Select,
    Static,
)

from reel_vault.cli.context import AppContext
from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths

REVIEW_FILTERS: tuple[str, ...] = ("all", "needs_review", "low_value", "failed", "archive")
QUEUE_COLUMNS = ("shortcode", "score", "category", "status", "decision", "caption")


@dataclass
class ReelSummary:
    shortcode: str
    score: int
    category: str
    status: str
    decision: str
    caption: str
    url: str


class ReviewScreen(Screen[None]):
    """The main review screen."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 40;
        border: solid $primary;
        padding: 1;
        layout: vertical;
    }

    #details {
        width: 1fr;
        border: solid $secondary;
        padding: 1;
        layout: vertical;
    }

    #reel-table {
        height: 1fr;
    }

    #filter-select {
        margin-bottom: 1;
    }

    #search-input {
        margin-bottom: 1;
    }

    #detail-scroll {
        height: 1fr;
        overflow-y: auto;
    }

    .detail-label {
        text-style: bold;
        color: $text-muted;
    }

    .detail-value {
        margin-bottom: 1;
    }

    .key-hint {
        color: $text-muted;
        text-style: italic;
    }

    #status-line {
        dock: bottom;
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }
    """

    BINDINGS = [  # noqa: RUF012 - Textual requires BINDINGS as a class attribute list
        ("a", "approve", "Approve"),
        ("r", "reclassify", "Reclassify"),
        ("f", "force", "Force decision"),
        ("n", "note", "Add note"),
        ("o", "open", "Open URL"),
        ("/", "search_focus", "Search"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.context = context
        self.paths: WorkspacePaths = context.paths
        self.config = context.config
        self.store: Store = context.store()
        self.reels: list[ReelSummary] = []
        self.selected_index: int | None = None
        self.filter_value: str = "all"
        self.search_text: str = ""

    # ------------------------------------------------------------------ compose

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Reel queue")
                yield Select(
                    options=[(f.title(), f) for f in REVIEW_FILTERS],
                    value="all",
                    id="filter-select",
                )
                yield Input(placeholder="Search shortcode, author, caption…", id="search-input")
                yield DataTable(id="reel-table", zebra_stripes=True, cursor_type="row")
                yield Static("", id="hint")
            with Vertical(id="details"):
                yield Static("Select a reel to review", id="detail-header")
                yield VerticalScroll(id="detail-scroll")
        yield Static(
            "a=approve  r=reclassify  f=force  n=note  o=open  /=search  q=quit",
            id="status-line",
        )

    def on_mount(self) -> None:
        table = self.query_one("#reel-table", DataTable)
        for col in QUEUE_COLUMNS:
            table.add_column(col, key=col)
        self._load_reels()
        self._refresh_hint()

    # ------------------------------------------------------------------ data loading

    def _load_reels(self) -> None:
        self.reels = []
        with self.store as store:
            rows = store.list_reels(limit=1000)
            for row in rows:
                classification = store.get_classification(row["shortcode"])
                triage = store.get_triage(row["shortcode"])
                self.reels.append(
                    ReelSummary(
                        shortcode=str(row["shortcode"]),
                        score=int((triage or {}).get("value_score", 0) or 0),
                        category=(classification or {}).get("primary") or "-",
                        status=str(row["status"]),
                        decision=str((triage or {}).get("decision") or "-"),
                        caption=str(row["caption"] or "")[:80],
                        url=str(row["url"]),
                    )
                )
        self._apply_filter_and_search()
        self._populate_table()

    def _apply_filter_and_search(self) -> None:
        text = self.search_text.casefold()
        filtered: list[ReelSummary] = []
        for reel in self.reels:
            if self.filter_value != "all":
                if self.filter_value == "needs_review":
                    triage = self.store.get_triage(reel.shortcode)
                    if not (triage or {}).get("needs_review"):
                        continue
                elif self.filter_value == "low_value":
                    if reel.decision not in {"stub", "-"}:
                        continue
                elif self.filter_value == "failed":
                    if reel.status not in {
                        "failed",
                        "triage_failed",
                        "extraction_failed",
                        "classification_failed",
                        "transcript_failed",
                        "media_failed",
                    }:
                        continue
                elif self.filter_value == "archive":
                    if reel.decision != "archive":
                        continue
            if (
                text
                and text not in reel.shortcode.casefold()
                and text not in reel.caption.casefold()
                and text not in reel.category.casefold()
            ):
                continue
            filtered.append(reel)
        filtered.sort(key=lambda r: (r.decision != "archive", -r.score))
        self.reels = filtered

    def _populate_table(self) -> None:
        table = self.query_one("#reel-table", DataTable)
        table.clear()
        for reel in self.reels:
            table.add_row(
                reel.shortcode,
                str(reel.score),
                reel.category,
                reel.status,
                reel.decision,
                reel.caption,
                key=reel.shortcode,
            )
        if self.reels and self.selected_index is None:
            self.selected_index = 0
            self._show_details(self.reels[0])

    def _refresh_hint(self) -> None:
        hint = self.query_one("#hint", Static)
        hint.update(f"{len(self.reels)} reel(s) — filter: {self.filter_value}")

    # ------------------------------------------------------------------ selection

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_index = getattr(event, "row_index", None)
        if row_index is None:
            return
        self.selected_index = row_index
        self._show_details(self.reels[row_index])

    def _show_details(self, reel: ReelSummary) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        scroll.remove_children()

        header = self.query_one("#detail-header", Static)
        header.update(f"Reel: {reel.shortcode}")

        with self.store as store:
            metadata = store.get_metadata(reel.shortcode)
            transcript = store.get_transcript(reel.shortcode)
            triage = store.get_triage(reel.shortcode)
            extraction = store.get_extraction(reel.shortcode)
            classification = store.get_classification(reel.shortcode)
            review = store.get_review(reel.shortcode)

        sections: list[tuple[str, str]] = []

        caption = (metadata or {}).get("caption") or reel.caption
        sections.append(("Caption", caption or "(none)"))
        sections.append(("URL", reel.url))

        if transcript and transcript.get("text"):
            text = str(transcript["text"])
            sections.append(("Transcript", text[:1200] + ("…" if len(text) > 1200 else "")))

        if triage:
            sections.append(("Triage decision", str(triage.get("decision") or "-")))
            sections.append(("Triage score", str(triage.get("value_score") or "-")))
            sections.append(("Triage reason", str(triage.get("reason") or "-")))
            if triage.get("archive_reason"):
                sections.append(("Archive reason", str(triage["archive_reason"])))
            if triage.get("rule_reason"):
                sections.append(("Rule reason", str(triage["rule_reason"])))
            if triage.get("needs_review"):
                sections.append(("Needs review", "yes"))

        if classification:
            primary = classification.get("primary")
            secondary = classification.get("secondary") or []
            sections.append(("Primary category", primary or "-"))
            if secondary:
                sections.append(("Secondary categories", ", ".join(secondary)))
            sections.append(
                ("Classification confidence", str(classification.get("confidence") or "-"))
            )

        if extraction:
            items = extraction.get("items") or []
            if items:
                item_lines = [
                    f"  • {it.get('field', '')}: {it.get('text', '')}" for it in items[:10]
                ]
                sections.append(("Extracted items", "\n".join(item_lines)))
            links = extraction.get("links") or []
            if links:
                link_lines = [f"  • {lk.get('url', '')}" for lk in links[:10]]
                sections.append(("Links", "\n".join(link_lines)))

        if review:
            notes = review.get("notes")
            if notes:
                sections.append(("Notes", str(notes)))
            overrides = review.get("overrides") or {}
            if overrides:
                sections.append(("Overrides", json.dumps(overrides, indent=2)))

        for label, value in sections:
            scroll.mount(Label(label))
            scroll.mount(Static(str(value)))

    # ------------------------------------------------------------------ actions

    def action_approve(self) -> None:
        if self.selected_index is None:
            return
        reel = self.reels[self.selected_index]
        with self.store as store:
            existing = store.get_review(reel.shortcode) or {}
            existing["approved"] = True
            existing["approved_at"] = datetime.now(UTC).isoformat()
            store.save_review(reel.shortcode, existing)
        self.notify(f"Approved {reel.shortcode}")

    def action_reclassify(self) -> None:
        if self.selected_index is None:
            return
        reel = self.reels[self.selected_index]
        self.app.push_screen(ReclassifyScreen(reel, self.store, self._on_reclassified))

    def _on_reclassified(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self.notify(f"Reclassified: {result.get('primary', '-')}")
        self._load_reels()

    def action_force(self) -> None:
        if self.selected_index is None:
            return
        reel = self.reels[self.selected_index]
        self.app.push_screen(ForceDecisionScreen(reel, self.store, self._on_forced))

    def _on_forced(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self.notify(f"Decision forced: {result.get('decision', '-')}")
        self._load_reels()

    def action_note(self) -> None:
        if self.selected_index is None:
            return
        reel = self.reels[self.selected_index]
        self.app.push_screen(NoteScreen(reel, self.store, self._on_note))

    def _on_note(self, result: dict[str, Any] | None) -> None:
        if not result:
            return
        self.notify("Note saved")
        self._load_reels()

    def action_open(self) -> None:
        if self.selected_index is None:
            return
        reel = self.reels[self.selected_index]
        try:
            webbrowser.open(reel.url)
        except Exception as exc:
            self.notify(f"Cannot open URL: {exc}", severity="error")

    def action_search_focus(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "filter-select":
            self.filter_value = str(event.value)
            self.selected_index = None
            self._apply_filter_and_search()
            self._populate_table()
            self._refresh_hint()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self.search_text = event.value
            self.selected_index = None
            self._apply_filter_and_search()
            self._populate_table()
            self._refresh_hint()

    def action_quit(self) -> None:
        self.app.exit()


class ReclassifyScreen(Screen[None]):
    """Pick primary + secondary categories for one reel."""

    CSS = """
    Screen {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: 30;
        border: solid $primary;
        padding: 1;
        layout: vertical;
    }
    """

    CATEGORIES: tuple[str, ...] = (
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
        reel: ReelSummary,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.reel = reel
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        classification = self.store.get_classification(self.reel.shortcode)
        current_primary = (classification or {}).get("primary") or ""
        current_secondary = (classification or {}).get("secondary") or []

        with Vertical(id="dialog"):
            yield Label(f"Reclassify {self.reel.shortcode}")
            yield Label("Primary category:")
            yield Select[str](
                options=[(c, c) for c in self.CATEGORIES],
                value=current_primary or "other",
                id="primary-select",
            )
            yield Label("Secondary categories (hold Ctrl to multi-select):")
            yield Select[str](
                options=[(c, c) for c in self.CATEGORIES],
                value=current_secondary[0] if current_secondary else "",
                allow_blank=True,
                id="secondary-select",
            )
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            self.app.pop_screen()
            return
        primary_select = self.query_one("#primary-select", Select[str])
        secondary_select = self.query_one("#secondary-select", Select[str])
        primary_value = primary_select.value
        secondary_value = secondary_select.value
        primary: str | None = None if primary_value in (Select.BLANK, "") else str(primary_value)
        secondary: list[str] = []
        if secondary_value not in (Select.BLANK, ""):
            secondary = [str(secondary_value)]
        result = {"primary": primary, "secondary": secondary}
        with self.store as store:
            existing = store.get_review(self.reel.shortcode) or {}
            existing["primary"] = primary
            existing["secondary"] = secondary
            existing["reclassified_at"] = datetime.now(UTC).isoformat()
            store.save_review(self.reel.shortcode, existing)
        self.on_done(result)
        self.app.pop_screen()


class ForceDecisionScreen(Screen[None]):
    """Force a triage decision (archive / extract / stub)."""

    CSS = """
    Screen {
        align: center middle;
    }
    #dialog {
        width: 50;
        height: 20;
        border: solid $primary;
        padding: 1;
        layout: vertical;
    }
    """

    def __init__(
        self,
        reel: ReelSummary,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.reel = reel
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Force decision for {self.reel.shortcode}")
            yield Select[str](
                options=[("archive", "archive"), ("extract", "extract"), ("stub", "stub")],
                value=self.reel.decision
                if self.reel.decision in {"archive", "extract", "stub"}
                else "extract",
                id="decision-select",
            )
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            self.app.pop_screen()
            return
        decision_select = self.query_one("#decision-select", Select[str])
        decision_value = decision_select.value
        if decision_value in (Select.BLANK, ""):
            self.app.pop_screen()
            return
        decision = str(decision_value)
        with self.store as store:
            triage = store.get_triage(self.reel.shortcode) or {}
            triage["decision"] = decision
            triage["force_decision"] = decision
            triage["forced_at"] = datetime.now(UTC).isoformat()
            store.save_triage(self.reel.shortcode, triage)
        self.on_done({"decision": decision})
        self.app.pop_screen()


class NoteScreen(Screen[None]):
    """Add a free-text note to one reel."""

    CSS = """
    Screen {
        align: center middle;
    }
    #dialog {
        width: 70;
        height: 25;
        border: solid $primary;
        padding: 1;
        layout: vertical;
    }
    """

    def __init__(
        self,
        reel: ReelSummary,
        store: Store,
        on_done: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__()
        self.reel = reel
        self.store = store
        self.on_done = on_done

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Add note for {self.reel.shortcode}")
            yield Input(placeholder="Your note…", id="note-input")
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            self.app.pop_screen()
            return
        note = self.query_one("#note-input", Input).value.strip()
        with self.store as store:
            existing = store.get_review(self.reel.shortcode) or {}
            notes = existing.get("notes", [])
            if isinstance(notes, str):
                notes = [notes]
            notes.append(note)
            existing["notes"] = notes
            existing["noted_at"] = datetime.now(UTC).isoformat()
            store.save_review(self.reel.shortcode, existing)
        self.on_done({"note": note})
        self.app.pop_screen()


class ReviewApp(App[None]):
    """Top-level Textual application for the review TUI."""

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.context = context

    def on_mount(self) -> None:
        self.push_screen(ReviewScreen(self.context))


def run_review(context: AppContext) -> int:
    """Launch the interactive review TUI. Returns 0 on clean exit."""
    app = ReviewApp(context)
    app.run()
    return 0


__all__ = ["ReviewApp", "ReviewScreen", "run_review"]
