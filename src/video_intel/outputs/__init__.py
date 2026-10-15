"""Output generation from the consolidated dataset (§14)."""

from video_intel.outputs.render import (
    RenderedOutput,
    build_outline,
    dataset_hash,
    render_full_notes,
    render_knowledge_json,
    render_outputs,
    render_ppt_outline,
    render_summary,
)

__all__ = [
    "RenderedOutput",
    "build_outline",
    "dataset_hash",
    "render_full_notes",
    "render_knowledge_json",
    "render_outputs",
    "render_ppt_outline",
    "render_summary",
]
