"""Storage foundation: on-disk layout, atomic JSON I/O and SQLite state."""

from video_intel.storage.db import SCHEMA_VERSION, StateStore
from video_intel.storage.io import (
    read_json,
    read_model,
    write_json_atomic,
    write_model_atomic,
    write_text_atomic,
)
from video_intel.storage.paths import (
    STAGE_FILES,
    ProjectPaths,
    VideoPaths,
    project_id_for,
    video_id_from_url,
)

__all__ = [
    "SCHEMA_VERSION",
    "STAGE_FILES",
    "ProjectPaths",
    "StateStore",
    "VideoPaths",
    "project_id_for",
    "read_json",
    "read_model",
    "video_id_from_url",
    "write_json_atomic",
    "write_model_atomic",
    "write_text_atomic",
]
