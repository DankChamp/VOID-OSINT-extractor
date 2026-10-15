"""Output generation (§7 of Phase 7).

Every artifact is rendered from the consolidated dataset — never from the transcript and
never from a separate summarization pass. Each artifact records the dataset version and
hash it came from, so changing an output format cannot trigger AI work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_intel.schemas.common import format_timestamp
from video_intel.schemas.knowledge import KnowledgeDataset, KnowledgeItem, KnowledgeTopic
from video_intel.schemas.outputs import PresentationOutline, SlideOutline

MAX_SLIDES = 24
MIN_TARGET_SLIDES = 10


def dataset_hash(dataset: KnowledgeDataset) -> str:
    """Stable hash of the dataset content, used to bind artifacts to their source."""
    payload = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def timestamp_label(item: KnowledgeItem) -> str:
    if item.start_seconds is None:
        return ""
    end = item.end_seconds if item.end_seconds is not None else item.start_seconds
    return f"{format_timestamp(item.start_seconds)} to {format_timestamp(end)}"


# --------------------------------------------------------------------- markdown
def render_full_notes(dataset: KnowledgeDataset) -> str:
    """Comprehensive Markdown notes (§14 A): coverage over brevity."""
    lines: list[str] = [f"# {dataset.title or dataset.video_id}", ""]
    lines.extend(_dataset_header(dataset))

    if not dataset.items:
        lines.extend(["", "No verified knowledge items were extracted.", ""])
        return "\n".join(lines) + "\n"

    for topic in _ordered_topics(dataset):
        lines.extend(["", f"## {topic.title}", ""])
        if topic.summary:
            lines.extend([topic.summary, ""])
        for item in _items_of(dataset, topic):
            lines.extend(_render_item(item))

    lines.extend(_render_conflicts(dataset))
    lines.extend(_render_uncertainty(dataset))
    return "\n".join(lines).rstrip() + "\n"


def _dataset_header(dataset: KnowledgeDataset) -> list[str]:
    report = dataset.completeness
    lines = [
        f"Video id: `{dataset.video_id}`",
        f"Dataset version: {dataset.dataset_version} (schema {dataset.schema_version})",
        f"Items: {len(dataset.items)} | Topics: {len(dataset.topics)} | "
        f"Conflicts: {len(dataset.conflicts)}",
        f"Completeness: {'complete' if report.complete else 'partial'}",
    ]
    if report.failed_stages:
        lines.append(f"Failed stages: {', '.join(stage.value for stage in report.failed_stages)}")
    if report.skipped_stages:
        lines.append(f"Skipped stages: {', '.join(stage.value for stage in report.skipped_stages)}")
    if report.unverified_items:
        lines.append(f"Items needing review: {report.unverified_items}")
    lines.append("")
    lines.append("All timestamps below come from the video transcript.")
    return lines


def _ordered_topics(dataset: KnowledgeDataset) -> list[KnowledgeTopic]:
    by_parent: dict[str | None, list[KnowledgeTopic]] = {}
    for topic in dataset.topics:
        by_parent.setdefault(topic.parent_id, []).append(topic)
    ordered: list[KnowledgeTopic] = []

    def walk(parent_id: str | None) -> None:
        for topic in by_parent.get(parent_id, []):
            ordered.append(topic)
            walk(topic.topic_id)

    walk(None)
    return ordered or list(dataset.topics)


def _items_of(dataset: KnowledgeDataset, topic: KnowledgeTopic) -> list[KnowledgeItem]:
    return [item for item in dataset.items if item.item_id in set(topic.item_ids)]


def _render_item(item: KnowledgeItem) -> list[str]:
    stamp = timestamp_label(item)
    suffix = f" _{stamp}_" if stamp else ""
    lines = [f"### {item.statement}{suffix}", ""]
    for detail in item.details:
        lines.append(f"- {detail}")
    if item.uncertainties:
        lines.append("")
        for uncertainty in item.uncertainties:
            lines.append(f"- _uncertainty:_ {uncertainty}")
    if item.verification.value != "supported":
        # Anything short of fully supported is labelled, so a weakly supported claim is
        # never presented as something the speaker plainly said.
        lines.append(f"- _verification:_ `{item.verification.value}`")
    segments = ", ".join(f"`{sid}`" for sid in item.source_segment_ids)
    lines.extend(["", f"Source: {segments}", ""])
    return lines


def _render_conflicts(dataset: KnowledgeDataset) -> list[str]:
    if not dataset.conflicts:
        return []
    lines = ["", "## Unresolved conflicts", ""]
    for conflict in dataset.conflicts:
        state = "resolved" if conflict.resolved else "**needs review**"
        lines.append(f"- {conflict.description} ({state})")
        if conflict.resolution:
            lines.append(f"  - resolution: {conflict.resolution}")
    lines.append("")
    return lines


def _render_uncertainty(dataset: KnowledgeDataset) -> list[str]:
    uncertain = [item for item in dataset.items if item.uncertainties]
    if not uncertain:
        return []
    lines = ["", "## Uncertainty", ""]
    for item in uncertain:
        lines.append(f"- {item.statement}: {'; '.join(item.uncertainties)}")
    lines.append("")
    return lines


def render_summary(dataset: KnowledgeDataset, *, max_items: int = 12) -> str:
    """A concise overview built from the dataset (§14 C)."""
    lines = [f"# Summary — {dataset.title or dataset.video_id}", ""]
    lines.extend(_dataset_header(dataset))

    trusted = [item for item in dataset.items if item.is_trusted]
    if not trusted:
        lines.extend(["", "No verified content is available.", ""])
        return "\n".join(lines) + "\n"

    by_topic: dict[str, list[KnowledgeItem]] = {}
    for topic in _ordered_topics(dataset):
        items = [item for item in _items_of(dataset, topic) if item.is_trusted]
        if items:
            by_topic[topic.title] = items

    for title, items in list(by_topic.items())[:8]:
        lines.extend(["", f"## {title}", ""])
        for item in items[:max_items]:
            stamp = timestamp_label(item)
            marker = f" ({stamp})" if stamp else ""
            lines.append(f"- {item.statement}{marker}")
    if len(by_topic) > 8:
        lines.extend(["", f"_… and {len(by_topic) - 8} more topics in full_notes.md_"])

    if dataset.completeness.unresolved_conflicts:
        lines.extend(
            ["", f"_{dataset.completeness.unresolved_conflicts} conflict(s) await review._"]
        )
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------- ppt outline
def build_outline(
    dataset: KnowledgeDataset, *, target_slides: int = MIN_TARGET_SLIDES
) -> PresentationOutline:
    """Presentation outline (§14 B).

    Slide count follows the material. When there is not enough content the outline says
    so instead of padding slides.
    """
    slides: list[SlideOutline] = []

    if not dataset.topics:
        return PresentationOutline(
            title=dataset.title or dataset.video_id,
            slides=[],
            target_slides=target_slides,
            meets_target=False,
            limitation_note="No topics were extracted, so no outline can be produced.",
        )

    title_slide = SlideOutline(
        slide_number=1,
        title=dataset.title or dataset.video_id,
        objective="Overview of what the speaker covered.",
        key_points=[topic.title for topic in _ordered_topics(dataset)[:6]] or ["No topics found"],
        topic_ids=[topic.topic_id for topic in _ordered_topics(dataset)],
    )
    slides.append(title_slide)

    for topic in _ordered_topics(dataset):
        items = [item for item in _items_of(dataset, topic) if item.is_trusted]
        if not items:
            continue
        statements, facts, examples = _split_item_roles(items)
        timestamps = [timestamp_label(item) for item in items if timestamp_label(item)]
        slides.append(
            SlideOutline(
                slide_number=len(slides) + 1,
                title=topic.title,
                objective=topic.summary or statements[0] if statements else None,
                key_points=statements[:6],
                supporting_items=facts[:4],
                examples=examples[:3],
                source_timestamps=timestamps[:6],
                speaker_notes=_speaker_notes(topic, items),
                topic_ids=[topic.topic_id],
                item_ids=[item.item_id for item in items],
            )
        )
        if len(slides) >= MAX_SLIDES:
            break

    conflicts = [c for c in dataset.conflicts if not c.resolved]
    if conflicts:
        slides.append(
            SlideOutline(
                slide_number=len(slides) + 1,
                title="Open questions and conflicts",
                objective="Show what the transcript does not settle.",
                key_points=[conflict.description for conflict in conflicts[:6]],
                source_timestamps=[],
                speaker_notes="These points are unresolved in the transcript.",
            )
        )

    meets_target = len(slides) >= target_slides
    limitation = None
    if not meets_target:
        limitation = (
            f"Only {len(slides)} meaningful slide(s) could be built from this video's "
            f"extracted content; the specification target is {target_slides}. "
            "Nothing was invented to reach the target."
        )
    return PresentationOutline(
        title=dataset.title or dataset.video_id,
        slides=slides,
        target_slides=target_slides,
        meets_target=meets_target,
        limitation_note=limitation,
    )


def _split_item_roles(items: Iterable[KnowledgeItem]) -> tuple[list[str], list[str], list[str]]:
    statements: list[str] = []
    facts: list[str] = []
    examples: list[str] = []
    for item in items:
        if item.type == "example":
            examples.append(item.statement)
        elif item.type in {"fact", "claim"}:
            facts.append(item.statement)
        else:
            statements.append(item.statement)
    return statements, facts, examples


def _speaker_notes(topic: KnowledgeTopic, items: Sequence[KnowledgeItem]) -> str | None:
    details = [detail for item in items for detail in item.details]
    if not details:
        return None
    return "Supporting detail: " + "; ".join(details[:3])


def render_ppt_outline(outline: PresentationOutline) -> str:
    lines = [f"# Presentation outline — {outline.title}", ""]
    if outline.limitation_note:
        lines.extend([f"> **Limitation:** {outline.limitation_note}", ""])
    for slide in outline.slides:
        lines.append(f"## Slide {slide.slide_number}: {slide.title}")
        lines.append("")
        if slide.objective:
            lines.extend([f"**Objective:** {slide.objective}", ""])
        for point in slide.key_points:
            lines.append(f"- {point}")
        for fact in slide.supporting_items:
            lines.append(f"- _fact:_ {fact}")
        for example in slide.examples:
            lines.append(f"- _example:_ {example}")
        if slide.source_timestamps:
            lines.extend(["", "Source: " + ", ".join(slide.source_timestamps)])
        if slide.speaker_notes:
            lines.extend(["", f"_Speaker notes:_ {slide.speaker_notes}"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_knowledge_json(dataset: KnowledgeDataset) -> str:
    """Full structured export including provenance and conflicts (§14 D)."""
    return json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class RenderedOutput:
    """One rendered artifact, ready to be written."""

    filename: str
    content: str

    def write(self, directory: Path) -> Path:
        from video_intel.storage.io import write_text_atomic

        path = directory / self.filename
        write_text_atomic(path, self.content)
        return path


def render_outputs(dataset: KnowledgeDataset) -> dict[str, RenderedOutput]:
    """Every output the specification lists, all from the same dataset."""
    outline = build_outline(dataset)
    return {
        "full_notes.md": RenderedOutput("full_notes.md", render_full_notes(dataset)),
        "summary.md": RenderedOutput("summary.md", render_summary(dataset)),
        "ppt_outline.md": RenderedOutput("ppt_outline.md", render_ppt_outline(outline)),
        "knowledge.json": RenderedOutput("knowledge.json", render_knowledge_json(dataset)),
    }


__all__ = [
    "MAX_SLIDES",
    "MIN_TARGET_SLIDES",
    "RenderedOutput",
    "build_outline",
    "dataset_hash",
    "render_full_notes",
    "render_knowledge_json",
    "render_outputs",
    "render_ppt_outline",
    "render_summary",
    "timestamp_label",
]
