"""Storage: path layout, URL handling, atomic writes and SQLite state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from video_intel.schemas.common import StageId, StageState, utc_now
from video_intel.schemas.manifest import StageManifest, StageRecord
from video_intel.schemas.transcript import RawTranscript
from video_intel.storage.db import SCHEMA_VERSION, StateStore
from video_intel.storage.io import (
    read_json,
    read_model,
    write_json_atomic,
    write_model_atomic,
    write_text_atomic,
)
from video_intel.storage.paths import (
    ProjectPaths,
    VideoPaths,
    project_id_for,
    video_id_from_url,
)


# ------------------------------------------------------------------- path layout
def test_video_layout_matches_specification(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path / "data", "default").video("abc123")
    root = paths.root
    assert root == tmp_path / "data" / "projects" / "default" / "abc123"
    assert paths.metadata.name == "metadata.json"
    assert paths.raw_transcript.name == "raw_transcript.json"
    assert paths.normalized_transcript.name == "normalized_transcript.json"
    assert paths.chunks.name == "chunks.json"
    assert paths.raw_extractions.name == "raw_extractions"
    assert paths.verified_items.name == "verified_items.json"
    assert paths.knowledge.name == "knowledge.json"
    assert paths.conflicts.name == "conflicts.json"
    assert paths.outputs.name == "outputs"


def test_database_and_logs_live_under_the_data_dir(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path / "data", "default")
    assert paths.database == tmp_path / "data" / "state.db"
    assert paths.logs == tmp_path / "data" / "logs"


def test_ensure_creates_directory_structure(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path / "data", "default").ensure().video("abc123").ensure()
    assert paths.raw_extractions.is_dir()
    assert paths.outputs.is_dir()


def test_video_id_rejects_path_traversal(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path, "default")
    for bad in ("../escape", "a/b", "..", ""):
        with pytest.raises(ValueError):
            paths.video(bad)


def test_output_file_rejects_separators(tmp_path: Path) -> None:
    paths = VideoPaths(tmp_path)
    with pytest.raises(ValueError, match="path separator"):
        paths.output_file("../escape.md")
    assert paths.output_file("full_notes.md") == paths.outputs / "full_notes.md"


def test_raw_extraction_filenames_are_namespaced(tmp_path: Path) -> None:
    path = VideoPaths(tmp_path).raw_extraction_file("coverage_audit", 3)
    assert path.name == "coverage_audit_0003.json"
    assert path.parent.name == "raw_extractions"


def test_existing_artifacts_includes_outputs(tmp_path: Path, transcript: RawTranscript) -> None:
    paths = VideoPaths(tmp_path).ensure()
    write_json_atomic(paths.raw_transcript, transcript.model_dump(mode="json"))
    write_text_atomic(paths.output_file("summary.md"), "# summary")
    artifacts = paths.existing_artifacts()
    assert artifacts["raw_transcript"] == paths.raw_transcript
    assert artifacts["outputs/summary.md"].name == "summary.md"


def test_project_id_is_filesystem_safe() -> None:
    assert project_id_for("Risk Assessment / 2026") == "risk-assessment-2026"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_video_id_extraction(url: str, expected: str) -> None:
    assert video_id_from_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=tooshort",
        "not a url",
        "",
        "https://www.youtube.com/",
    ],
)
def test_non_youtube_urls_are_rejected(url: str) -> None:
    assert video_id_from_url(url) is None


# -------------------------------------------------------------------- atomic I/O
def test_write_json_atomic_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "data.json"
    written = write_json_atomic(path, {"a": 1})
    assert path.is_file()
    assert written == path.stat().st_size
    assert read_json(path) == {"a": 1}


def test_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    workdir = tmp_path / "atomic"
    workdir.mkdir()
    path = workdir / "data.json"
    write_json_atomic(path, {"a": 1})
    write_json_atomic(path, {"a": 2})
    assert [entry.name for entry in workdir.iterdir()] == ["data.json"]


def test_failed_write_does_not_clobber_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "atomic"
    workdir.mkdir()
    path = workdir / "data.json"
    write_json_atomic(path, {"good": True})

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(path, {"bad": True})
    assert read_json(path) == {"good": True}
    assert [entry.name for entry in workdir.iterdir()] == ["data.json"]


def test_read_json_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing file"):
        read_json(tmp_path / "absent.json")


def test_read_json_reports_malformed_content(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        read_json(path)


def test_model_roundtrip_preserves_timestamps(tmp_path: Path, transcript: RawTranscript) -> None:
    path = tmp_path / "raw_transcript.json"
    write_model_atomic(path, transcript)
    restored = read_model(path, RawTranscript)
    assert [(s.start_seconds, s.end_seconds) for s in restored.segments] == [
        (s.start_seconds, s.end_seconds) for s in transcript.segments
    ]


def test_read_model_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_model(path, RawTranscript)


# ------------------------------------------------------------------ SQLite state
def test_initialize_is_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    store.initialize()
    with store.session() as connection:
        version = connection.execute(
            "SELECT value FROM schema_info WHERE key='version'"
        ).fetchone()[0]
    assert version == str(SCHEMA_VERSION)


def test_project_and_video_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project("default", "Default", config_hash="cfg", now=now)
    store.upsert_video(
        "default",
        "abc123",
        url="https://youtu.be/abc12345678",
        title="Demo",
        duration_seconds=330.0,
        now=now,
    )
    assert [row["project_id"] for row in store.list_projects()] == ["default"]
    assert store.video_ids("default") == ["abc123"]


def test_stage_state_roundtrips_through_manifest(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project("default", "Default", now=now)
    store.upsert_video("default", "abc123", now=now)

    record = StageRecord(
        stage_id=StageId.CHUNKS,
        state=StageState.COMPLETED,
        attempts=1,
        input_hash="in",
        output_hash="out",
        config_hash="cfg",
        started_at=utc_now(),
        completed_at=utc_now(),
        duration_seconds=1.5,
        detail={"chunks": "2"},
    )
    store.upsert_stage("default", "abc123", record)

    manifest = store.load_manifest("default", "abc123")
    restored = manifest.get(StageId.CHUNKS)
    assert restored is not None
    assert restored.state is StageState.COMPLETED
    assert restored.reusable is True
    assert restored.input_hash == "in"
    assert restored.duration_seconds == pytest.approx(1.5)


def test_failed_stage_is_recorded_with_error(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project("default", "Default", now=now)
    store.upsert_video("default", "abc123", now=now)
    store.record_stage(
        "default", "abc123", StageId.VERIFY, StageState.FAILED, attempts=3, error="timeout"
    )
    manifest = store.load_manifest("default", "abc123")
    record = manifest.get(StageId.VERIFY)
    assert record is not None
    assert record.state is StageState.FAILED
    assert record.error == "timeout"
    assert manifest.failed_stages() == [StageId.VERIFY]


def test_artifacts_are_indexed(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project("default", "Default", now=now)
    store.upsert_video("default", "abc123", now=now)
    store.record_artifact(
        "default",
        "abc123",
        StageId.OUTPUT_NOTES,
        "full_notes.md",
        "outputs/full_notes.md",
        bytes_written=2048,
        dataset_hash="ds1",
        dataset_version=1,
        now=now,
    )
    rows = store.artifacts_for("default", "abc123")
    assert len(rows) == 1
    assert rows[0]["filename"] == "full_notes.md"
    assert rows[0]["bytes_written"] == 2048


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    with pytest.raises(sqlite3.IntegrityError):
        store.record_stage("default", "missing", StageId.CHUNKS, StageState.RUNNING)


def test_manifest_upsert_matches_store_schema(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    now = utc_now().isoformat()
    store.upsert_project("default", "Default", now=now)
    store.upsert_video("default", "abc123", now=now)
    manifest = StageManifest(video_id="abc123")
    manifest.upsert(
        StageRecord(stage_id=StageId.METADATA, state=StageState.COMPLETED, output_hash="h")
    )
    for record in manifest.stages:
        store.upsert_stage("default", "abc123", record)
    assert store.load_manifest("default", "abc123").is_complete(StageId.METADATA)
