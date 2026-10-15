"""Local web frontend (§16, §18).

A thin, dependency-free bridge over the CLI:

* it starts real ``video-intel analyze`` runs as subprocesses, so the web UI uses
  exactly the same pipeline, cache and locking as the terminal;
* it only reads and writes inside the configured data directory;
* it binds to 127.0.0.1 by default.

The specification lists an elaborate GUI as out of scope for the MVP and says not to
start GUI work before the CLI pipeline is reliable. The pipeline now is, so this exists
as a thin front end over the same commands rather than a second implementation.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from video_intel.config.env_file import load_env_file
from video_intel.config.loader import ConfigError, load_config
from video_intel.config.root import find_project_root
from video_intel.pipeline.stages import STAGE_ORDER
from video_intel.schemas.common import OutputMode
from video_intel.storage.io import read_json
from video_intel.storage.locking import lock_owner
from video_intel.storage.paths import (
    WINDOWS_RESERVED_NAMES,
    ProjectPaths,
    VideoPaths,
    project_id_for,
    relative_key,
)

MAX_BODY_BYTES = 64 * 1024
MAX_LOG_LINES = 400
JOB_HISTORY = 50
SAFE_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")

OUTPUT_FILES = ("full_notes.md", "summary.md", "ppt_outline.md", "knowledge.json")


class WebError(Exception):
    """A request the UI should report to the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def display_command(command: list[str]) -> str:
    """Render a command line the way the host shell would accept it.

    POSIX quoting and Windows quoting are genuinely different languages, so the UI shows
    a line the user can actually paste back into the shell they are looking at.
    """
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        return subprocess.list2cmdline(command)
    return shlex.join(command)


@dataclass
class Job:
    """One running or finished CLI invocation."""

    job_id: str
    command: list[str]
    started_at: float
    status: str = "running"
    exit_code: int | None = None
    log: list[str] = field(default_factory=list)
    process: subprocess.Popen[bytes] | None = None
    video_id: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": display_command(self.command),
            "status": self.status,
            "exit_code": self.exit_code,
            "video_id": self.video_id,
            "started_at": self.started_at,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            "log": list(self.log),
        }


class JobRegistry:
    """In-memory job table; single process, single user."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, command: list[str], video_id: str | None) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            command=command,
            started_at=time.time(),
            video_id=video_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > JOB_HISTORY:
                oldest = sorted(self._jobs.values(), key=lambda item: item.started_at)
                for stale in oldest[: len(self._jobs) - JOB_HISTORY]:
                    if stale.status == "running":
                        continue
                    self._jobs.pop(stale.job_id, None)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise WebError(f"unknown job {job_id}", HTTPStatus.NOT_FOUND)
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.started_at, reverse=True)
        return [job.snapshot() for job in jobs]


class WebApp:
    """Request handling, independent of the HTTP server plumbing."""

    def __init__(
        self,
        *,
        data_dir: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        python: str | None = None,
    ) -> None:
        load_env_file()
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.config = self._load_config(data_dir)
        self.python = python or sys.executable
        self.jobs = JobRegistry()
        self.data_dir = Path(self.config.data_dir).expanduser().resolve()

    # ------------------------------------------------------------------ helpers
    def _load_config(self, data_dir: str | os.PathLike[str] | None) -> Any:
        overrides: dict[str, Any] = {}
        if data_dir:
            overrides["data_dir"] = str(data_dir)
        return load_config(explicit_path=self.config_path, overrides=overrides)

    @property
    def project_paths(self) -> ProjectPaths:
        return ProjectPaths(self.data_dir, project_id_for(self.config.project_name))

    @staticmethod
    def safe_name(value: str, *, what: str = "name") -> str:
        """Validate a caller-supplied name before it is joined to a path.

        Rejects anything that is not a plain component, and additionally the device names
        Windows reserves, which pass a character filter because they are alphanumeric.
        """
        if (
            not value
            or len(value) > 128
            or set(value) - SAFE_NAME
            or value.startswith(".")
            or value.rstrip(" .") != value
            or value.split(".", 1)[0].lower() in WINDOWS_RESERVED_NAMES
        ):
            raise WebError(f"invalid {what}: {value!r}")
        return value

    def video_paths(self, video_id: str) -> VideoPaths:
        return self.project_paths.video(self.safe_name(video_id, what="video id"))

    # ------------------------------------------------------------------ actions
    def status(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "project": self.config.project_name,
            "execution_mode": str(self.config.execution_mode),
            "config_hash": self.config.hash(),
            "stages": [stage.value for stage in STAGE_ORDER],
            "modes": sorted(mode.value for mode in self.config.modes),
            "providers": [provider.redacted() for provider in self.config.providers.values()],
            "videos": len(self.videos()),
            "active_jobs": sum(1 for job in self.jobs.list() if job["status"] == "running"),
        }

    def videos(self) -> list[dict[str, Any]]:
        if not self.project_paths.root.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for video in self.project_paths.videos():
            manifest = _read_manifest(video.state)
            records = manifest.stages if manifest else []
            stages = {record.stage_id: record.state for record in records}
            entries.append(
                {
                    "video_id": video.root.name,
                    "stages": stages,
                    "has_dataset": video.knowledge.is_file(),
                    "has_transcript": video.raw_transcript.is_file(),
                    "has_chunks": video.chunks.is_file(),
                    "outputs": sorted(
                        path.name for path in video.outputs.glob("*") if path.is_file()
                    )
                    if video.outputs.is_dir()
                    else [],
                    "lock": lock_owner(video.root / ".lock"),
                }
            )
        return entries

    def video_detail(self, video_id: str) -> dict[str, Any]:
        paths = self.video_paths(video_id)
        manifest = _read_manifest(paths.state)
        detail: dict[str, Any] = {
            "video_id": video_id,
            "exists": paths.root.is_dir(),
            "stages": [
                {
                    "stage_id": record.stage_id.value,
                    "state": record.state.value,
                    "attempts": record.attempts,
                    "error": record.error,
                    "completed_at": (
                        record.completed_at.isoformat() if record.completed_at else None
                    ),
                }
                for record in (manifest.stages if manifest else [])
            ],
            "artifacts": {
                name: relative_key(paths.root, path)
                for name, path in sorted(paths.existing_artifacts().items())
            },
        }
        if paths.metadata.is_file():
            metadata = read_json(paths.metadata)
            detail["metadata"] = {
                "title": metadata.get("title"),
                "channel": metadata.get("channel"),
                "duration_seconds": metadata.get("duration_seconds"),
            }
        if paths.knowledge.is_file():
            dataset = read_json(paths.knowledge)
            items = dataset.get("items", [])
            detail["dataset"] = {
                "items": len(items),
                "topics": len(dataset.get("topics", [])),
                "conflicts": len(dataset.get("conflicts", [])),
                "unsupported": sum(
                    1 for item in items if item.get("verification") == "unsupported"
                ),
                "complete": bool(dataset.get("completeness", {}).get("complete")),
                "unverified_items": dataset.get("completeness", {}).get("unverified_items", 0),
            }
        return detail

    def read_output(self, video_id: str, filename: str) -> str:
        paths = self.video_paths(video_id)
        name = self.safe_name(filename)
        if name not in OUTPUT_FILES:
            raise WebError(f"{filename} is not an output file", HTTPStatus.NOT_FOUND)
        path = paths.output_file(name)
        if not path.is_file():
            raise WebError(f"{name} has not been generated yet", HTTPStatus.NOT_FOUND)
        return path.read_text(encoding="utf-8", errors="replace")

    # --------------------------------------------------------------------- jobs
    def start_analyze(
        self,
        *,
        url: str,
        mode: str = OutputMode.DEEP.value,
        until: str | None = None,
        force: bool = False,
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        from video_intel.ingest.youtube import extract_video_id

        video_id = extract_video_id(url)
        if video_id is None:
            raise WebError(f"not a recognisable YouTube URL: {url}")

        if until and until not in {stage.value for stage in STAGE_ORDER}:
            raise WebError(f"stage '{until}' is not implemented")

        command = [
            self.python,
            "-m",
            "video_intel",
            "analyze",
            url,
            "--mode",
            mode,
            "--data-dir",
            str(self.data_dir),
            "--project",
            self.config.project_name,
        ]
        if self.config_path:
            command += ["--config", str(self.config_path)]
        if until:
            command += ["--until", until]
        if force:
            command.append("--force")
        for language in languages or []:
            command += ["--language", language]

        job = self.jobs.create(command, video_id=video_id)
        self._spawn(job)
        return job.snapshot()

    def start_notebooklm_export(self, video_id: str) -> dict[str, Any]:
        name = self.safe_name(video_id, what="video id")
        command = [
            self.python,
            "-m",
            "video_intel",
            "notebooklm",
            "export",
            name,
            "--data-dir",
            str(self.data_dir),
            "--project",
            self.config.project_name,
        ]
        if self.config_path:
            command += ["--config", str(self.config_path)]
        job = self.jobs.create(command, video_id=name)
        self._spawn(job)
        return job.snapshot()

    def start_notebooklm_import(self, video_id: str, source: Path) -> dict[str, Any]:
        name = self.safe_name(video_id, what="video id")
        resolved = Path(source).expanduser()
        if not resolved.is_file():
            raise WebError(f"file not found: {resolved}")
        command = [
            self.python,
            "-m",
            "video_intel",
            "notebooklm",
            "import",
            str(resolved),
            "--video",
            name,
            "--data-dir",
            str(self.data_dir),
            "--project",
            self.config.project_name,
        ]
        if self.config_path:
            command += ["--config", str(self.config_path)]
        job = self.jobs.create(command, video_id=name)
        self._spawn(job)
        return job.snapshot()

    def _spawn(self, job: Job) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The child inherits the discovered project root as its working directory, not
        # ``data_dir.parent``. With an absolute data directory the parent is some unrelated
        # directory, and the job would silently fail to find video-intel.yaml and .env.
        root = find_project_root()
        process = subprocess.Popen(
            job.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(root) if root.is_dir() else None,
        )
        job.process = process
        thread = threading.Thread(target=self._pump, args=(job,), daemon=True)
        thread.start()

    def _pump(self, job: Job) -> None:
        process = job.process
        if process is None or process.stdout is None:  # pragma: no cover - defensive
            return
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            job.log.append(line)
            if len(job.log) > MAX_LOG_LINES:
                del job.log[: len(job.log) - MAX_LOG_LINES]
        job.exit_code = process.wait()
        job.status = "finished" if job.exit_code == 0 else "failed"
        job.log.append(f"[exit {job.exit_code}]")


def _read_manifest(path: Path) -> Any:
    from video_intel.schemas.manifest import StageManifest

    if not path.is_file():
        return None
    try:
        return StageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


class Handler(BaseHTTPRequestHandler):
    """Routes HTTP requests onto :class:`WebApp`."""

    server_version = "video-intel-web/0.1"
    app: WebApp

    def log_message(self, format: str, *args: Any) -> None:
        return  # the CLI already logs; keep the console readable

    # ------------------------------------------------------------------ helpers
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise WebError("request body too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WebError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise WebError("request body must be a JSON object")
        return payload

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        app = self.app

        try:
            if method == "GET" and route == "/":
                self._send_text(WEB_INDEX.read_text(encoding="utf-8"), content_type="text/html")
            elif method == "GET" and route == "/api/status":
                self._send_json(app.status())
            elif method == "GET" and route == "/api/videos":
                self._send_json({"videos": app.videos()})
            elif method == "GET" and route.startswith("/api/video/"):
                self._send_json(app.video_detail(route.rsplit("/", 1)[-1]))
            elif method == "GET" and route == "/api/output":
                self._send_text(app.read_output(query.get("video", ""), query.get("name", "")))
            elif method == "GET" and route == "/api/jobs":
                self._send_json({"jobs": app.jobs.list()})
            elif method == "GET" and route.startswith("/api/job/"):
                self._send_json(app.jobs.get(route.rsplit("/", 1)[-1]).snapshot())
            elif method == "POST" and route == "/api/analyze":
                body = self._read_body()
                self._send_json(
                    app.start_analyze(
                        url=str(body.get("url", "")),
                        mode=str(body.get("mode") or OutputMode.DEEP.value),
                        until=str(body["until"]) if body.get("until") else None,
                        force=bool(body.get("force")),
                        languages=[str(item) for item in body.get("languages") or []],
                    ),
                    HTTPStatus.ACCEPTED,
                )
            elif method == "POST" and route == "/api/notebooklm/export":
                body = self._read_body()
                self._send_json(
                    app.start_notebooklm_export(str(body.get("video_id", ""))),
                    HTTPStatus.ACCEPTED,
                )
            elif method == "POST" and route == "/api/notebooklm/import":
                body = self._read_body()
                self._send_json(
                    app.start_notebooklm_import(
                        str(body.get("video_id", "")), Path(str(body.get("path", "")))
                    ),
                    HTTPStatus.ACCEPTED,
                )
            else:
                raise WebError(f"no route for {method} {route}", HTTPStatus.NOT_FOUND)
        except WebError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except ConfigError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        except Exception as exc:  # pragma: no cover - last resort
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")


WEB_INDEX = Path(__file__).with_name("index.html")


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    open_browser: bool = False,
) -> None:
    """Run the web frontend until interrupted."""
    app = WebApp(data_dir=data_dir, config_path=config_path)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"video-intel web UI on {url}")
    print(f"  data dir : {app.data_dir}")
    print(f"  project  : {app.config.project_name}")
    print("  press Ctrl+C to stop")
    if open_browser:  # pragma: no cover - convenience only
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nstopping")
    finally:
        server.server_close()


__all__ = ["Handler", "Job", "JobRegistry", "WebApp", "WebError", "serve"]
