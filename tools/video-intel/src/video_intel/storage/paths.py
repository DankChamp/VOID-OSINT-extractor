"""Project path layout (§15).

::

    data/
      projects/
        project_id/
          project.json
          video_id/
            metadata.json
            raw_transcript.json
            normalized_transcript.json
            chunks.json
            raw_extractions/
            verified_items.json
            knowledge.json
            conflicts.json
            outputs/
              full_notes.md
              ppt_outline.md
              summary.md
            state.json
      state.db
      logs/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath

from video_intel.schemas.common import slugify

METADATA_FILE = "metadata.json"
RAW_TRANSCRIPT_FILE = "raw_transcript.json"
NORMALIZED_TRANSCRIPT_FILE = "normalized_transcript.json"
CHUNKS_FILE = "chunks.json"
RAW_EXTRACTIONS_DIR = "raw_extractions"
VERIFIED_ITEMS_FILE = "verified_items.json"
KNOWLEDGE_FILE = "knowledge.json"
CONFLICTS_FILE = "conflicts.json"
STATE_FILE = "state.json"
PROJECT_FILE = "project.json"
OUTPUTS_DIR = "outputs"

STAGE_FILES: dict[str, str] = {
    "metadata": METADATA_FILE,
    "raw_transcript": RAW_TRANSCRIPT_FILE,
    "normalized_transcript": NORMALIZED_TRANSCRIPT_FILE,
    "chunks": CHUNKS_FILE,
    "verified_items": VERIFIED_ITEMS_FILE,
    "knowledge": KNOWLEDGE_FILE,
    "conflicts": CONFLICTS_FILE,
    "state": STATE_FILE,
}


@dataclass(frozen=True)
class VideoPaths:
    """Filesystem locations for one video."""

    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_FILE

    @property
    def raw_transcript(self) -> Path:
        return self.root / RAW_TRANSCRIPT_FILE

    @property
    def normalized_transcript(self) -> Path:
        return self.root / NORMALIZED_TRANSCRIPT_FILE

    @property
    def chunks(self) -> Path:
        return self.root / CHUNKS_FILE

    @property
    def raw_extractions(self) -> Path:
        return self.root / RAW_EXTRACTIONS_DIR

    @property
    def verified_items(self) -> Path:
        return self.root / VERIFIED_ITEMS_FILE

    @property
    def knowledge(self) -> Path:
        return self.root / KNOWLEDGE_FILE

    @property
    def conflicts(self) -> Path:
        return self.root / CONFLICTS_FILE

    @property
    def state(self) -> Path:
        return self.root / STATE_FILE

    @property
    def outputs(self) -> Path:
        return self.root / OUTPUTS_DIR

    def output_file(self, filename: str) -> Path:
        validate_component(filename, what="output filename")
        return self.outputs / filename

    def raw_extraction_file(self, pipeline_id: str, index: int) -> Path:
        return self.raw_extractions / f"{safe_component(pipeline_id)}_{index:04d}.json"

    def ensure(self) -> VideoPaths:
        self.raw_extractions.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        return self

    def existing_artifacts(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for stage, filename in STAGE_FILES.items():
            path = self.root / filename
            if path.is_file():
                found[stage] = path
        if self.outputs.is_dir():
            for output in sorted(self.outputs.iterdir()):
                if output.is_file():
                    found[relative_key(self.root, output)] = output
        return found


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations for one project."""

    data_dir: Path
    project_id: str

    @property
    def root(self) -> Path:
        return self.data_dir / "projects" / self.project_id

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def database(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def logs(self) -> Path:
        return self.data_dir / "logs"

    def video(self, video_id: str) -> VideoPaths:
        validate_component(video_id, what="video_id")
        return VideoPaths(self.root / video_id)

    def videos(self) -> list[VideoPaths]:
        if not self.root.is_dir():
            return []
        return [VideoPaths(entry) for entry in sorted(self.root.iterdir()) if entry.is_dir()]

    def ensure(self) -> ProjectPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        return self


def project_id_for(project_name: str) -> str:
    """Stable, filesystem-safe project identifier."""
    return slugify(project_name)


def safe_component(value: str) -> str:
    """Reduce a value to characters that are safe inside a filename.

    The output is deliberately **ASCII-only**. Non-ASCII letters survive ``str.isalnum()``
    but not identically across filesystems: APFS and HFS+ normalise to NFD while ext4 and
    NTFS store the code points verbatim, so a single accented character can become two
    distinct byte sequences and the same project stops resolving on another device.
    Underscores are preserved so pipeline ids such as ``coverage_audit`` round-trip
    unchanged in artifact names.
    """
    cleaned = "".join(
        char if (char.isascii() and (char.isalnum() or char in "-_.")) else "-"
        for char in value.strip()
    )
    cleaned = cleaned.strip("-.")
    return cleaned or "unknown"


#: Device names Windows reserves. They are pure alphanumeric, so a basic character
#: filter lets them through, and they cannot be used as a directory name on NTFS.
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

#: Characters that separate path components on any supported platform. Both are always
#: rejected, regardless of the host, so a directory tree stays portable.
SEPARATORS = ("/", "\\")


def _is_reserved_component(name: str) -> bool:
    """Whether a single path component cannot exist on a Windows filesystem."""
    if name.rstrip(" .") != name:
        return True  # Windows silently strips trailing dots and spaces
    return name.split(".", 1)[0].lower() in WINDOWS_RESERVED_NAMES


def validate_component(name: str, *, what: str = "name") -> str:
    """Raise unless ``name`` is a single path component valid on every platform."""
    if not name:
        raise ValueError(f"{what} must not be empty")
    if name in {".", ".."}:
        raise ValueError(f"invalid {what}: {name!r}")
    if any(separator in name for separator in SEPARATORS):
        raise ValueError(f"{what} must not contain a path separator: {name!r}")
    if PurePath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise ValueError(f"{what} must be relative: {name!r}")
    if _is_reserved_component(name):
        raise ValueError(f"{what} is reserved on Windows filesystems: {name!r}")
    return name


def relative_key(root: Path, path: Path) -> str:
    """A path relative to ``root``, rendered with ``/`` on every platform.

    Using a fixed separator keeps JSON keys, the ``inspect`` table and the web API
    identical whether the run happened on Linux or Windows.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def video_id_from_url(url: str) -> str | None:
    """Return the YouTube video id for a URL, or ``None``.

    Thin wrapper over :mod:`video_intel.ingest.youtube` so URL parsing has a single
    implementation; ingestion adds the typed error and the canonical URL.
    """
    from video_intel.ingest.youtube import extract_video_id

    return extract_video_id(url)
