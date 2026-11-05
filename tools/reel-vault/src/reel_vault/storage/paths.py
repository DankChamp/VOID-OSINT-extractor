"""Filesystem layout.

::

    <data_dir>/
      state.db
      reels/<shortcode>/
        source.json
        metadata.json
        caption.txt
        transcript.json
        triage.json
        extraction.json
        classification.json
        review.json
        media.json
        keyframes/
        outputs/reel.md
      media/<shortcode>/
      imports/
      outputs/
      runs/
      logs/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath

SOURCE_FILE = "source.json"
METADATA_FILE = "metadata.json"
CAPTION_FILE = "caption.txt"
TRANSCRIPT_FILE = "transcript.json"
TRIAGE_FILE = "triage.json"
EXTRACTION_FILE = "extraction.json"
CLASSIFICATION_FILE = "classification.json"
REVIEW_FILE = "review.json"
MEDIA_FILE = "media.json"
OCR_FILE = "ocr.json"
KEYFRAMES_DIR = "keyframes"
OUTPUTS_DIR = "outputs"
REEL_MARKDOWN = "reel.md"

STAGE_FILES: dict[str, str] = {
    "source": SOURCE_FILE,
    "metadata": METADATA_FILE,
    "transcript": TRANSCRIPT_FILE,
    "triage": TRIAGE_FILE,
    "extraction": EXTRACTION_FILE,
    "classification": CLASSIFICATION_FILE,
    "review": REVIEW_FILE,
    "media": MEDIA_FILE,
    "ocr": OCR_FILE,
}

WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
SEPARATORS = ("/", "\\")


def safe_component(value: str) -> str:
    cleaned = "".join(
        char if (char.isascii() and (char.isalnum() or char in "-_.")) else "-"
        for char in value.strip()
    )
    return cleaned.strip("-.") or "unknown"


def _is_reserved(name: str) -> bool:
    if name.rstrip(" .") != name:
        return True
    return name.split(".", 1)[0].lower() in WINDOWS_RESERVED_NAMES


def validate_component(name: str, *, what: str = "name") -> str:
    if not name:
        raise ValueError(f"{what} must not be empty")
    if name in {".", ".."}:
        raise ValueError(f"invalid {what}: {name!r}")
    if any(separator in name for separator in SEPARATORS):
        raise ValueError(f"{what} must not contain a path separator: {name!r}")
    if PurePath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise ValueError(f"{what} must be relative: {name!r}")
    if _is_reserved(name):
        raise ValueError(f"{what} is reserved on Windows filesystems: {name!r}")
    return name


def relative_key(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


@dataclass(frozen=True)
class ReelPaths:
    data_dir: Path
    shortcode: str

    def __post_init__(self) -> None:
        validate_component(self.shortcode, what="shortcode")

    @property
    def root(self) -> Path:
        return self.data_dir / "reels" / self.shortcode

    @property
    def source(self) -> Path:
        return self.root / SOURCE_FILE

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_FILE

    @property
    def caption(self) -> Path:
        return self.root / CAPTION_FILE

    @property
    def transcript(self) -> Path:
        return self.root / TRANSCRIPT_FILE

    @property
    def triage(self) -> Path:
        return self.root / TRIAGE_FILE

    @property
    def extraction(self) -> Path:
        return self.root / EXTRACTION_FILE

    @property
    def classification(self) -> Path:
        return self.root / CLASSIFICATION_FILE

    @property
    def review(self) -> Path:
        return self.root / REVIEW_FILE

    @property
    def media_manifest(self) -> Path:
        return self.root / MEDIA_FILE

    @property
    def ocr(self) -> Path:
        return self.root / OCR_FILE

    @property
    def keyframes(self) -> Path:
        return self.root / KEYFRAMES_DIR

    @property
    def outputs(self) -> Path:
        return self.root / OUTPUTS_DIR

    @property
    def markdown(self) -> Path:
        return self.outputs / REEL_MARKDOWN

    def ensure(self) -> ReelPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        return self

    def existing_artifacts(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for stage, filename in STAGE_FILES.items():
            path = self.root / filename
            if path.is_file():
                found[stage] = path
        if self.markdown.is_file():
            found["markdown"] = self.markdown
        return found


@dataclass(frozen=True)
class WorkspacePaths:
    data_dir: Path
    db_path: Path
    media_dir: Path

    @property
    def reels_root(self) -> Path:
        return self.data_dir / "reels"

    @property
    def imports(self) -> Path:
        return self.data_dir / "imports"

    @property
    def outputs(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def runs(self) -> Path:
        return self.data_dir / "runs"

    @property
    def logs(self) -> Path:
        return self.data_dir / "logs"

    @property
    def database(self) -> Path:
        return self.db_path

    def reel(self, shortcode: str) -> ReelPaths:
        return ReelPaths(self.data_dir, shortcode)

    def media(self, shortcode: str) -> Path:
        validate_component(shortcode, what="shortcode")
        return self.media_dir / shortcode

    def ensure(self) -> WorkspacePaths:
        for directory in (
            self.data_dir,
            self.reels_root,
            self.imports,
            self.outputs,
            self.runs,
            self.logs,
            self.media_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


__all__ = [
    "STAGE_FILES",
    "ReelPaths",
    "WorkspacePaths",
    "relative_key",
    "safe_component",
    "validate_component",
]
