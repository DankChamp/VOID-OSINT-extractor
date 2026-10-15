"""Portability and multi-device guarantees.

Every test here corresponds to a way the project previously broke when moved between
machines, copied to a share, or opened on another operating system. They are grouped by
the failure they prevent rather than by the module under test.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import pytest
from video_intel.config.env_file import find_env_file
from video_intel.config.loader import ConfigError, load_config, user_config_path
from video_intel.config.models import AppConfig
from video_intel.config.root import ROOT_MARKER_FILE, find_project_root, resolve_path
from video_intel.storage.db import (
    SCHEMA_VERSION,
    StateStore,
    StateStoreError,
    db_diagnostics,
    resolve_db_path,
)
from video_intel.storage.io import read_json, write_json_atomic
from video_intel.storage.locking import LOCK_FORMAT, LockError, host_id, video_lock
from video_intel.storage.paths import (
    ProjectPaths,
    safe_component,
    validate_component,
)

# --------------------------------------------------------------- project root discovery


def test_project_root_is_found_from_a_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A marker at the root wins over the working directory."""
    (tmp_path / ROOT_MARKER_FILE).write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert find_project_root() == tmp_path.resolve()


def test_workspace_marker_beats_a_nearer_tool_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit workspace marker anywhere above wins over a nearer tool marker.

    This is deliberate, and it is the reason the walk has two passes. In a monorepo the
    tool's own directory sits between the user and the workspace root, and if it could
    win then every command run from inside tools/<name>/ would silently create and read a
    second, empty data tree there.
    """
    (tmp_path / ROOT_MARKER_FILE).write_text("", encoding="utf-8")
    nested = tmp_path / "tools" / "video-intel"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text(
        '[project]\nname = "video-intel"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    monkeypatch.chdir(nested)
    assert find_project_root() == tmp_path.resolve()


def test_nearest_wins_within_the_same_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a workspace marker, the nearest tool-level marker is used."""
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "video-intel.yaml").write_text("project_name: outer\n", encoding="utf-8")
    inner = outer / "tools" / "video-intel"
    inner.mkdir(parents=True)
    (inner / "pyproject.toml").write_text(
        '[project]\nname = "video-intel"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    monkeypatch.chdir(inner)
    assert find_project_root() == inner.resolve()


def test_marker_priority_within_one_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project config is preferred over a bare VCS directory in the same place."""
    (tmp_path / ROOT_MARKER_FILE).write_text("", encoding="utf-8")
    (tmp_path / "video-intel.yaml").write_text("project_name: x\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == tmp_path.resolve()


def test_project_config_is_found_from_a_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Configuration resolution must not depend on the working directory."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "video-intel.yaml").write_text("project_name: fromroot\n", encoding="utf-8")
    nested = root / "deep" / "nested"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    assert load_config().project_name == "fromroot"

    monkeypatch.chdir(root)
    assert load_config().project_name == "fromroot"


def test_relative_data_dir_resolves_against_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The packaged ``data_dir: "data"`` lands in the project root from any directory."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "video-intel.yaml").write_text("data_dir: ./data\n", encoding="utf-8")
    nested = root / "somewhere"
    nested.mkdir()

    monkeypatch.chdir(nested)
    config = load_config()
    assert config.data_dir == (root / "data").resolve()


def test_env_file_is_found_from_a_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The nearest .env up the tree is used, as every dotenv loader does."""
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("OPENAI_API_KEY=from-root\n", encoding="utf-8")
    nested = root / "deep" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    found = find_env_file()
    assert found is not None
    assert found.parent == root


def test_nearest_env_file_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("OPENAI_API_KEY=outer\n", encoding="utf-8")
    nested = root / "inner"
    nested.mkdir(parents=True)
    (nested / ".env").write_text("OPENAI_API_KEY=inner\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    found = find_env_file()
    assert found == nested / ".env"


def test_explicit_env_file_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "custom.env"
    explicit.write_text("OPENAI_API_KEY=explicit\n", encoding="utf-8")
    monkeypatch.setenv("VIDEO_INTEL_ENV_FILE", str(explicit))
    found = find_env_file()
    assert found == explicit


def test_resolve_path_expands_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert resolve_path("~/library") == tmp_path / "library"


def test_user_config_absent_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    assert user_config_path() is None


def test_user_config_is_found_in_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows puts user configuration in %APPDATA%, not in ~/.config."""
    appdata = tmp_path / "AppData" / "Roaming"
    (appdata / "video-intel").mkdir(parents=True)
    (appdata / "video-intel" / "config.yaml").write_text(
        "project_name: windows\n", encoding="utf-8"
    )
    monkeypatch.setenv("APPDATA", str(appdata))
    found = user_config_path()
    assert found is not None and found.name == "config.yaml"
    assert load_config().project_name == "windows"


# ------------------------------------------------------------- portable configuration hash


def _config_with_sources(sources: list[str], data_dir: Path) -> AppConfig:
    payload = {
        "config_version": "1.0",
        "data_dir": str(data_dir),
        "project_name": "default",
        "source_files": sources,
        "providers": {},
        "pipelines": {},
        "modes": {},
        "presets": {},
    }
    return AppConfig.model_validate(payload)


def test_config_hash_is_independent_of_install_location(tmp_path: Path) -> None:
    """Two identical configurations on two machines must share one cache key.

    This is the regression that made every copied checkout silently invalidate its
    transcript, chunk and dataset caches.
    """
    linux = _config_with_sources(
        ["/home/alice/src/video_intel/config/defaults/config.yaml"], tmp_path / "a" / "data"
    )
    windows = _config_with_sources(
        ["C:\\Users\\bob\\site-packages\\video_intel\\config\\defaults\\config.yaml"],
        tmp_path / "b" / "data",
    )
    assert linux.data_dir != windows.data_dir
    assert linux.source_files[0] != windows.source_files[0]
    assert linux.hash() == windows.hash()


def test_config_hash_still_tracks_behavioural_change(tmp_path: Path) -> None:
    """Dropping the location must not drop real configuration differences."""
    sources = ["/somewhere/config.yaml"]
    base = _config_with_sources(sources, tmp_path / "data")
    changed = base.model_copy(update={"project_name": "other"})
    assert base.hash() != changed.hash()


def test_config_hash_distinguishes_differing_settings(tmp_path: Path) -> None:
    base = _config_with_sources(["/x/config.yaml"], tmp_path / "data")
    tuned = base.model_copy(update={"max_extraction_passes": base.max_extraction_passes + 1})
    assert base.hash() != tuned.hash()


def test_display_dump_keeps_absolute_paths_for_humans(tmp_path: Path) -> None:
    """Stripping paths from the hash must not remove them from `config show`."""
    config = _config_with_sources(["/srv/app/config.yaml"], tmp_path / "data")
    assert config.redaction_safe_dump()["data_dir"] == str(tmp_path / "data")
    assert config.portable_dump()["data_dir"] == "data"


def test_bom_prefixed_config_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Notepad-saved YAML file starts with a BOM and must still load."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "video-intel.yaml").write_text("﻿project_name: frombom\n", encoding="utf-8")
    monkeypatch.chdir(root)
    assert load_config().project_name == "frombom"


def test_malformed_config_reports_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "video-intel.yaml"
    broken.write_text("project_name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(explicit_path=broken)
    assert "invalid YAML" in str(excinfo.value)


# --------------------------------------------------------------------- state database


def test_queries_create_the_schema_without_an_explicit_init(tmp_path: Path) -> None:
    """Any command may touch the database first; it must never find no tables.

    This is the crash that produced a bare `no such table: stages` traceback.
    """
    store = StateStore(tmp_path / "state.db")
    assert store.load_manifest("default", "abc").stages == []


def test_table_less_database_is_repaired(tmp_path: Path) -> None:
    """A state.db copied without its write-ahead log is rebuilt on next use."""
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE unrelated(x)")
    connection.commit()
    connection.close()

    store = StateStore(path)
    store.initialize()
    assert store.schema_version() == SCHEMA_VERSION

    probe = sqlite3.connect(path)
    try:
        tables = {
            row[0] for row in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        probe.close()
    assert {"projects", "videos", "stages", "artifacts", "schema_info"} <= tables


def test_committed_state_survives_copying_only_the_main_file(tmp_path: Path) -> None:
    """Copying state.db alone must carry the schema with it.

    Without a checkpoint on close, a database left by an interrupted run keeps its schema
    in state.db-wal, and the copy is a valid-looking file containing nothing.
    """
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    store.upsert_project("default", "Default", now="2024-01-01T00:00:00+00:00")

    destination = tmp_path / "copy.db"
    destination.write_bytes((tmp_path / "state.db").read_bytes())

    copied = StateStore(destination)
    assert copied.schema_version() == SCHEMA_VERSION
    assert [row["project_id"] for row in copied.list_projects()] == ["default"]


def test_checkpoint_is_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    store.checkpoint()
    store.checkpoint()
    assert store.schema_version() == SCHEMA_VERSION


def test_database_path_can_be_redirected_off_a_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite wants local disk; artifacts can live on a network folder."""
    shared = tmp_path / "nas" / "data"
    shared.mkdir(parents=True)
    monkeypatch.setenv("VIDEO_INTEL_DB_PATH", str(tmp_path / "local" / "state.db"))
    assert resolve_db_path(shared / "state.db") == tmp_path / "local" / "state.db"


def test_journal_mode_override_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WAL is not supported on every filesystem, so it must be overridable."""
    monkeypatch.setenv("VIDEO_INTEL_DB_JOURNAL_MODE", "delete")
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    connection = sqlite3.connect(tmp_path / "state.db")
    try:
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        connection.close()
    assert mode == "delete"


def test_busy_timeout_is_applied(tmp_path: Path) -> None:
    """A 5 second default is not survivable over a network mount."""
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    connection = store.connect()
    try:
        assert int(connection.execute("PRAGMA busy_timeout").fetchone()[0]) >= 30_000
    finally:
        connection.close()


def test_session_rolls_back_a_failed_statement(tmp_path: Path) -> None:
    """A mid-block failure must not leave earlier statements committed."""
    store = StateStore(tmp_path / "state.db")
    store.initialize()
    with pytest.raises(sqlite3.DatabaseError), store.session() as connection:
        connection.execute(
            "INSERT INTO projects(project_id, name, created_at, updated_at) "
            "VALUES('p', 'P', 'now', 'now')"
        )
        connection.execute("INSERT INTO nonexistent_table VALUES (1)")
    assert [row["project_id"] for row in store.list_projects()] == []


def test_transaction_failure_raises_a_readable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock that cannot be acquired must not surface as a raw driver traceback."""

    class RefusesToBegin:
        """A connection whose ``BEGIN`` always fails, as a busy lock would."""

        def __init__(self, wrapped: sqlite3.Connection) -> None:
            self._wrapped = wrapped

        def execute(self, statement: str, *args: object) -> object:
            if statement.startswith("BEGIN"):
                raise sqlite3.OperationalError("database is locked")
            return self._wrapped.execute(statement, *args)

        def close(self) -> None:
            self._wrapped.close()

    store = StateStore(tmp_path / "state.db")
    store.initialize()
    real_connect = store.connect
    monkeypatch.setattr(store, "connect", lambda: RefusesToBegin(real_connect()))  # type: ignore[method-assign,return-value]

    with pytest.raises(StateStoreError) as excinfo, store.session() as connection:
        connection.execute("SELECT 1")
    assert "VIDEO_INTEL_DB_PATH" in str(excinfo.value)


def test_artifact_path_must_be_relative_on_every_platform(tmp_path: Path) -> None:
    from video_intel.schemas.common import StageId

    store = StateStore(tmp_path / "state.db")
    store.initialize()
    store.upsert_project("default", "Default", now="2024-01-01T00:00:00+00:00")
    store.upsert_video("default", "abc", now="2024-01-01T00:00:00+00:00")

    for absolute in ("/abs/knowledge.json", "C:\\data\\knowledge.json", "\\\\nas\\share\\k.json"):
        with pytest.raises(ValueError, match="relative"):
            store.record_artifact(
                "default",
                "abc",
                StageId.KNOWLEDGE_DATASET,
                "knowledge.json",
                absolute,
                bytes_written=1,
                now="2024-01-01T00:00:00+00:00",
            )
    store.record_artifact(
        "default",
        "abc",
        StageId.KNOWLEDGE_DATASET,
        "knowledge.json",
        "outputs/knowledge.json",
        bytes_written=1,
        now="2024-01-01T00:00:00+00:00",
    )
    assert len(store.artifacts_for("default", "abc")) == 1


def test_diagnostics_report_a_table_less_file(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    report = db_diagnostics(path)
    assert report["exists"] is True
    assert report["error"] is not None
    assert db_diagnostics(tmp_path / "missing.db")["exists"] is False


# ---------------------------------------------------------------------------- locking


def test_lock_file_records_the_device(tmp_path: Path) -> None:
    with video_lock(tmp_path) as path:
        tokens = path.read_text(encoding="utf-8").split()
    assert tokens[0] == LOCK_FORMAT
    assert tokens[3].startswith(host_id().split(":")[0])


def test_a_lock_from_another_live_device_is_respected(tmp_path: Path) -> None:
    """A foreign pid must not be probed locally, because it means nothing here."""
    path = tmp_path / ".lock"
    path.write_text(
        f"{LOCK_FORMAT} 4242 {time.time():.0f} other-laptop:99 analyze\n", encoding="utf-8"
    )
    with pytest.raises(LockError) as excinfo, video_lock(tmp_path):
        pass
    assert "other-laptop" in str(excinfo.value)


def test_an_abandoned_lock_from_another_device_is_reclaimed(tmp_path: Path) -> None:
    """Age is the only signal available for a foreign owner, so it must eventually act."""
    stale = time.time() - (3 * 24 * 60 * 60)
    path = tmp_path / ".lock"
    path.write_text(f"{LOCK_FORMAT} 4242 {stale:.0f} other-laptop:99 analyze\n", encoding="utf-8")
    with video_lock(tmp_path) as held:
        assert held.is_file()


def test_legacy_lock_file_is_still_understood(tmp_path: Path) -> None:
    """A lock written by an earlier release must keep its original meaning.

    The old layout had no host field, so its pid was probed locally - which is correct,
    because such a lock could only have been written by this machine.
    """
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = tmp_path / ".lock"
        path.write_text(f"{helper.pid} {time.time():.0f} analyze\n", encoding="utf-8")
        with pytest.raises(LockError), video_lock(tmp_path):
            pass
    finally:
        helper.terminate()
        helper.wait(timeout=10)


def test_held_lock_is_not_stolen_through_a_second_spelling(tmp_path: Path) -> None:
    """A symlinked path to the same directory must not delete our own live lock.

    The held-lock set is keyed by the resolved path; if the staleness check compared an
    unresolved spelling it would conclude the lock was abandoned and take it twice.
    """
    directory = tmp_path / "video"
    directory.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(directory, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")

    with video_lock(directory):
        # The alias resolves to the same lock; the inner call must be re-entrant.
        with video_lock(alias):
            assert (directory / ".lock").is_file()
        assert (directory / ".lock").is_file()


def test_lock_error_names_the_holding_device(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.write_text(f"{LOCK_FORMAT} 4242 {time.time():.0f} desktop-2:5 analyze\n", encoding="utf-8")
    with pytest.raises(LockError) as excinfo, video_lock(tmp_path):
        pass
    message = str(excinfo.value)
    assert "desktop-2" in message and "4242" in message


# ------------------------------------------------------------------------- path safety


@pytest.mark.parametrize(
    "name",
    [
        "../escape.md",
        "..\\escape.md",
        "sub/dir.md",
        "sub\\dir.md",
        "",
        ".",
        "..",
        "con",
        "NUL",
        "com1",
        "LPT9",
        "trailing.",
        "trailing ",
        "/absolute.md",
    ],
)
def test_component_validation_rejects_portability_hazards(name: str) -> None:
    with pytest.raises(ValueError):
        validate_component(name)


@pytest.mark.parametrize("name", ["full_notes.md", "knowledge.json", "v7KtPLhSMkU", "a_b-c.1"])
def test_component_validation_accepts_real_names(name: str) -> None:
    assert validate_component(name) == name


def test_video_id_rejects_a_windows_separator() -> None:
    paths = ProjectPaths(Path("/tmp/data"), "default")
    with pytest.raises(ValueError):
        paths.video("a\\b")


def test_safe_component_is_ascii_only() -> None:
    """Non-ASCII survives isalnum() but not filesystem NFC/NFD normalisation."""
    assert safe_component("café").isascii()
    assert safe_component("项目").isascii()
    assert safe_component("coverage_audit") == "coverage_audit"


def test_relative_key_uses_posix_separators(tmp_path: Path) -> None:
    from video_intel.storage.paths import relative_key

    root = tmp_path / "video"
    target = root / "outputs" / "summary.md"
    target.parent.mkdir(parents=True)
    assert relative_key(root, target) == "outputs/summary.md"


def test_output_file_rejects_a_backslash() -> None:
    paths = ProjectPaths(Path("/tmp/data"), "default").video("abc")
    with pytest.raises(ValueError):
        paths.output_file("..\\escape.md")


# ------------------------------------------------------------------- artifact durability


def test_json_with_a_bom_is_readable(tmp_path: Path) -> None:
    """A dataset round-tripped through a Windows editor must still parse."""
    path = tmp_path / "knowledge.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"items": [1, 2]}).encode("utf-8"))
    assert read_json(path) == {"items": [1, 2]}


def test_artifacts_are_not_owner_private(tmp_path: Path) -> None:
    """mkstemp creates 0600 files; a copied library must stay readable."""
    if os.name == "nt":  # pragma: no cover - no POSIX mode bits
        pytest.skip("POSIX permissions only")
    path = tmp_path / "knowledge.json"
    write_json_atomic(path, {"items": []})
    assert path.stat().st_mode & 0o044, "artifact should be group/other readable"


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts"
    path = directory / "knowledge.json"
    write_json_atomic(path, {"a": 1})
    assert [entry.name for entry in directory.iterdir()] == ["knowledge.json"]


def test_invalid_utf8_artifact_reports_clearly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_json(path)


# ---------------------------------------------------------------------- workspace layout

# <repo>/tools/video-intel/tests/test_portability.py
TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_workspace_root_is_found_from_inside_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tools/<name>/pyproject.toml must not capture the root and strand data/ inside it.

    The workspace holds data/, .env and video-intel.yaml; the tool directory holds code.
    Running from anywhere inside the tool must resolve to the workspace.
    """
    workspace = tmp_path / "workspace"
    tool = workspace / "tools" / "video-intel"
    (tool / "src").mkdir(parents=True)

    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "void-osint-extractor"\n[tool.uv]\npackage = false\n',
        encoding="utf-8",
    )
    (tool / "pyproject.toml").write_text(
        '[project]\nname = "video-intel"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    monkeypatch.chdir(tool / "src")
    assert find_project_root() == workspace.resolve()

    monkeypatch.chdir(tool)
    assert find_project_root() == workspace.resolve()


def test_explicit_workspace_marker_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty marker file declares the root even inside an unrelated checkout."""
    workspace = tmp_path / "workspace"
    inner = workspace / "nested"
    inner.mkdir(parents=True)
    (workspace / ROOT_MARKER_FILE).write_text("", encoding="utf-8")
    monkeypatch.chdir(inner)
    assert find_project_root() == workspace.resolve()


def test_tool_still_works_standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool copied out on its own resolves to its own directory."""
    tool = tmp_path / "video-intel"
    tool.mkdir()
    (tool / "pyproject.toml").write_text(
        '[project]\nname = "video-intel"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tool)
    assert find_project_root() == tool.resolve()


def test_unrelated_pyproject_is_not_a_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Some other project's pyproject.toml must not capture this one."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "something-else"\n', encoding="utf-8"
    )
    (tmp_path / ROOT_MARKER_FILE).write_text("", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert find_project_root() == tmp_path.resolve()


def test_workspace_declares_every_tool_as_a_member() -> None:
    """Adding a tool must not require editing the workspace manifest."""
    import tomllib

    workspace = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    assert workspace["tool"]["uv"]["workspace"]["members"] == ["tools/*"]
    assert workspace["tool"]["uv"]["package"] is False


def test_tool_is_an_installable_distribution() -> None:
    """The tool keeps its own name, scripts and build backend inside the workspace."""
    import tomllib

    tool = tomllib.loads((TOOL_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    assert tool["project"]["name"] == "video-intel"
    assert tool["project"]["scripts"] == {"video-intel": "video_intel.cli.main:cli"}
    assert tool["build-system"]["build-backend"] == "hatchling.build"
    assert tool["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/video_intel"]


def test_shared_venv_holds_every_tool() -> None:
    """One interpreter at the root; tools must not each demand their own."""
    launcher = (REPO_ROOT / "run.py").read_text(encoding="utf-8")
    assert 'VENV_DIR = REPO_ROOT / ".venv"' in launcher
    for tool in sorted((REPO_ROOT / "tools").iterdir()):
        assert not (tool / ".venv").exists(), f"{tool.name} must not carry its own venv"


# ---------------------------------------------------------------------- packaged launcher


def test_launcher_is_importable_without_dependencies() -> None:
    """run.py must work on a machine that has not installed anything yet."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0
    assert "run.py" in result.stdout


def test_launcher_discovers_the_tools() -> None:
    """Tools are found from the filesystem, so adding one needs no code change.

    Deliberately not asserting that this is the *only* tool: the workspace is expected
    to grow, and a test pinned to a single tool would fail on the day the next one lands.
    """
    import runpy

    namespace = runpy.run_path(str(REPO_ROOT / "run.py"))
    tools = namespace["discover_tools"]()
    names = [tool.name for tool in tools]

    assert "video-intel" in names
    assert names == sorted(names), "tools should be discovered in a stable order"

    found = namespace["find_tool"]("video-intel")
    assert found is not None
    assert "video-intel" in found.scripts
    assert found.root == TOOL_ROOT
    assert namespace["find_tool"]("no-such-tool") is None


def test_launcher_runs_a_tool_by_name() -> None:
    """`run.py <tool> ...` must reach that tool and not the workspace."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "video-intel", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0
    assert "video-intel" in result.stdout


def test_launcher_rejects_an_unknown_tool() -> None:
    """A typo must fail with guidance, not print the workspace help."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "no-such-tool", "analyze"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode != 0
    assert "no-such-tool" in result.stderr


def test_shell_wrapper_delegates_to_the_python_launcher() -> None:
    """run.sh must not reimplement anything, or the platforms drift apart."""
    wrapper = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "run.py" in wrapper
    assert "exec" in wrapper
    # Every non-comment line is part of the delegation; no command logic lives here.
    body = [
        line.strip()
        for line in wrapper.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert body[-1].endswith('run.py" "$@"')


def test_declared_dependencies_include_the_transcript_client() -> None:
    """A fresh install must not be missing the module every analyze run imports."""
    import tomllib

    tool = tomllib.loads((TOOL_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    dependencies = " ".join(tool["project"]["dependencies"]).lower()
    assert "youtube-transcript-api" in dependencies


def test_gitattributes_keeps_the_launcher_lf() -> None:
    """A CRLF run.sh is not executable on a Windows checkout."""
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "eol=lf" in attributes
    assert "*.sh" in attributes


def test_artifacts_never_carry_an_absolute_path() -> None:
    """OutputArtifact must reject a Windows or UNC absolute path, not just POSIX."""
    from video_intel.schemas.common import StageState
    from video_intel.schemas.outputs import OutputArtifact

    for bad in ("/home/me/knowledge.json", "C:\\Users\\me\\k.json", "\\\\nas\\share\\k.json"):
        with pytest.raises(ValueError, match="relative"):
            OutputArtifact(
                path=bad,
                dataset_hash="a" * 64,
                stage_state=StageState.COMPLETED,
            )
    assert PureWindowsPath("outputs\\summary.md").is_absolute() is False
