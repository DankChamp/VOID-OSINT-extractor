"""Web frontend: request handling, safety rails and the job bridge (§16, §18)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from video_intel.web.server import (
    OUTPUT_FILES,
    Handler,
    WebApp,
    WebError,
)


@pytest.fixture
def app(tmp_path: Path) -> Iterator[WebApp]:
    application = WebApp(data_dir=tmp_path / "data")
    yield application
    application.project_paths.ensure()


@pytest.fixture
def server(app: WebApp) -> Iterator[tuple[str, int, WebApp]]:
    handler = type("BoundHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield host, port, app
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(host: str, port: int, path: str) -> tuple[int, Any]:
    connection = HTTPConnection(host, port, timeout=10)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    connection.close()
    return response.status, body


def post(host: str, port: int, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    connection = HTTPConnection(host, port, timeout=10)
    connection.request(
        "POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    connection.close()
    return response.status, body


# ------------------------------------------------------------------- app logic
def test_status_reports_project_and_providers_without_secrets(app: WebApp) -> None:
    status = app.status()
    assert status["project"] == "default"
    assert "metadata" in status["stages"] and "output_summary" in status["stages"]
    names = {provider["name"] for provider in status["providers"]}
    assert {"ollama", "groq", "deepseek"} <= names
    serialised = json.dumps(status)
    assert "sk-" not in serialised


def test_stage_list_matches_the_pipeline(app: WebApp) -> None:
    stages = app.status()["stages"]
    assert stages.index("chunks") < stages.index("extractions")
    assert stages.index("knowledge_dataset") < stages.index("output_notes")


def test_videos_are_empty_before_any_run(app: WebApp) -> None:
    assert app.videos() == []


def test_video_detail_reports_nothing_for_unknown_ids(app: WebApp) -> None:
    detail = app.video_detail("doesnotexist")
    assert detail["exists"] is False
    assert detail["stages"] == []


def test_path_traversal_is_refused(app: WebApp) -> None:
    for bad in ("../../etc/passwd", "a/b", ".hidden", ""):
        with pytest.raises(WebError):
            app.safe_name(bad)


def test_output_reading_is_limited_to_known_files(app: WebApp) -> None:
    with pytest.raises(WebError, match="not an output file"):
        app.read_output("video1", "knowledge.json.evil")
    with pytest.raises(WebError, match="not an output file"):
        app.read_output("video1", "state.json")


def test_output_reading_reports_a_missing_file(app: WebApp) -> None:
    paths = app.project_paths.video("video1").ensure()
    paths.output_file("summary.md").write_text("# hi", encoding="utf-8")
    assert app.read_output("video1", "summary.md") == "# hi"
    with pytest.raises(WebError, match="has not been generated"):
        app.read_output("video1", "full_notes.md")


def test_all_output_names_are_served(app: WebApp) -> None:
    assert set(OUTPUT_FILES) == {"full_notes.md", "summary.md", "ppt_outline.md", "knowledge.json"}


def test_video_detail_summarises_a_stored_dataset(app: WebApp) -> None:
    from video_intel.storage.io import write_json_atomic

    paths = app.project_paths.video("video1").ensure()
    write_json_atomic(paths.metadata, {"title": "A video", "duration_seconds": 100.0})
    write_json_atomic(
        paths.knowledge,
        {
            "schema_version": "1.0.0",
            "video_id": "video1",
            "items": [
                {"item_id": "a", "verification": "supported"},
                {"item_id": "b", "verification": "unsupported"},
            ],
            "topics": [{"topic_id": "topic_001_x", "title": "T"}],
            "conflicts": [],
            "completeness": {"complete": True, "unverified_items": 0},
        },
    )
    detail = app.video_detail("video1")
    assert detail["metadata"]["title"] == "A video"
    assert detail["dataset"]["items"] == 2
    assert detail["dataset"]["unsupported"] == 1
    assert detail["dataset"]["complete"] is True


def test_analyze_rejects_a_non_youtube_url(app: WebApp) -> None:
    with pytest.raises(WebError, match="not a recognisable YouTube URL"):
        app.start_analyze(url="https://example.com/x")


def test_analyze_rejects_an_unknown_stage(app: WebApp) -> None:
    with pytest.raises(WebError, match="not implemented"):
        app.start_analyze(url="https://youtu.be/dQw4w9WgXcQ", until="audio_download")


def test_notebooklm_import_requires_an_existing_file(app: WebApp, tmp_path: Path) -> None:
    with pytest.raises(WebError, match="file not found"):
        app.start_notebooklm_import("video1", tmp_path / "nope.json")


# ------------------------------------------------------------------ http layer
def test_index_page_is_served(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/")
    assert status == 200
    assert "<title>video-intel</title>" in body
    assert "YouTube URL" in body


def test_status_endpoint(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/status")
    assert status == 200
    payload = json.loads(body)
    assert payload["project"] == "default"
    assert payload["stages"]


def test_videos_endpoint(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/videos")
    assert status == 200
    assert json.loads(body) == {"videos": []}


def test_unknown_route_returns_404_json(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/nope")
    assert status == HTTPStatus.NOT_FOUND
    assert "no route" in json.loads(body)["error"]


def test_bad_json_body_is_reported(server: tuple[str, int, WebApp]) -> None:
    connection = HTTPConnection(*server[:2], timeout=10)
    connection.request(
        "POST", "/api/analyze", body="{not json", headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    connection.close()
    assert response.status == 400
    assert "invalid JSON" in json.loads(body)["error"]


def test_analyze_endpoint_validates_input(server: tuple[str, int, WebApp]) -> None:
    status, body = post(*server[:2], "/api/analyze", {"url": "not-a-url"})
    assert status == 400
    assert "not a recognisable YouTube URL" in json.loads(body)["error"]


def test_output_endpoint_requires_video(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/output?video=&name=full_notes.md")
    assert status == 400
    assert "invalid video id" in json.loads(body)["error"]


def test_jobs_endpoint_lists_jobs(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/jobs")
    assert status == 200
    assert json.loads(body) == {"jobs": []}


def test_unknown_job_returns_404(server: tuple[str, int, WebApp]) -> None:
    status, body = get(*server[:2], "/api/job/deadbeef")
    assert status == 404
    assert "unknown job" in json.loads(body)["error"]
