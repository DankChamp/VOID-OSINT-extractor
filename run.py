#!/usr/bin/env python3
"""video-intel launcher: one entry point for every platform.

    python run.py                 set up, then show status
    python run.py doctor          environment report for this device
    python run.py setup           create the virtual environment and install dependencies
    python run.py test [args]     run the test suite (offline by default)
    python run.py lint            ruff + mypy
    python run.py init            create the project layout and state database
    python run.py analyze URL     run the CLI
    python run.py web [--port N]  start the local web frontend
    python run.py notebooklm ...  export or import a NotebookLM packet
    python run.py <anything else> forwarded to the video-intel CLI
    python run.py help            this list

Why a Python launcher rather than more shell
-------------------------------------------

``run.sh`` covered Linux and macOS and nothing else: it needed bash, assumed
``.venv/bin/python``, used ``python3`` and ``curl`` by name, and wrote to ``/tmp``. This
file does the same job with the standard library only, so the identical commands work on
Windows, macOS and Linux, and it picks the correct virtual-environment layout for the
host. ``run.sh`` is now a thin wrapper around this file, so the two cannot drift apart.

The launcher invokes the CLI as ``python -m video_intel`` rather than through the
generated ``video-intel`` console script: console scripts are per-platform, are not
regenerated when dependencies change, and on Windows add another layer that can be stale.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PORT = 8765
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

IS_TTY = sys.stdout.isatty()


def _colour(code: str, text: str) -> str:
    if not IS_TTY or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(message: str) -> None:
    print(_colour("32", "✓") + f" {message}")


def warn(message: str) -> None:
    print(_colour("33", "!") + f" {message}")


def fail(message: str) -> None:
    print(_colour("31", "✗") + f" {message}", file=sys.stderr)
    raise SystemExit(1)


def info(message: str) -> None:
    print(_colour("2", message))


def venv_python() -> Path:
    """The interpreter inside the project virtual environment.

    ``Scripts`` on Windows, ``bin`` everywhere else. Probed rather than assumed so the
    same source tree works after being copied from a Mac to a PC.
    """
    if os.name == "nt":
        candidate = VENV_DIR / "Scripts" / "python.exe"
    else:
        candidate = VENV_DIR / "bin" / "python"
    return candidate


def base_python() -> str:
    """An interpreter suitable for building the virtual environment.

    The running interpreter is preferred, so ``python run.py`` never silently installs a
    different Python version than the one it was launched with.
    """
    return sys.executable or shutil.which("python3") or shutil.which("python") or "python"


def ensure_installed() -> Path:
    """Return the virtual environment interpreter, creating it when necessary."""
    python = venv_python()
    if not python.exists():
        setup(create_env=True)
    return python


def _importable() -> None:
    """Put the source tree on ``sys.path`` so the launcher works before installation.

    ``doctor`` and ``checkpoint`` need the package, and the point of ``doctor`` is to
    report on a machine that may not be installed correctly yet - so it must not depend
    on the installation it is diagnosing.
    """
    source = str(SRC_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)


def setup(*, create_env: bool = True) -> Path:
    """Create the virtual environment if needed and install the project."""
    python = venv_python()
    if not python.exists():
        if not create_env:
            fail(f"no virtual environment at {VENV_DIR}; run: python run.py setup")
        info("creating the virtual environment…")
        subprocess.run([base_python(), "-m", "venv", str(VENV_DIR)], check=True)
        python = venv_python()

    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "pip", "install", "--python", str(python), "-e", ".[dev]"], check=True)
    else:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-e", ".[dev]"], check=True)

    ok(f"environment ready: {python}")
    return python


def run_cli(python: Path, args: list[str], *, capture: bool = False) -> int:
    """Invoke the CLI as a module, with the source tree importable as a fallback."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    result = subprocess.run(
        [str(python), "-m", "video_intel", *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
        capture_output=capture,
        text=True,
    )
    if capture:
        sys.stdout.write(result.stdout or "")
        sys.stderr.write(result.stderr or "")
    return result.returncode


def run_tool(python: Path, tool: str, args: list[str]) -> int:
    """Run a development tool from the virtual environment.

    Invoked as ``python -m <tool>`` on every platform. The console-script layout differs
    between ``bin`` and ``Scripts``, and a stale entry point is a confusing failure mode,
    so the module form is the only one used here.
    """
    return subprocess.run(
        [str(python), "-m", tool, *args], cwd=str(PROJECT_ROOT), check=False
    ).returncode


def ollama_models() -> list[str] | None:
    """Model names from a reachable local Ollama, or ``None`` when it is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return [str(entry.get("name", "")) for entry in payload.get("models", [])]


def doctor(python: Path) -> None:
    """Report what this device can actually do."""
    print("video-intel doctor")
    print("─" * 46)

    print(f"platform       {sys.platform}")
    print(f"python         {sys.version.split()[0]}  ({sys.executable})")
    print(f"project root   {PROJECT_ROOT}")
    print(f"venv           {python} " + ("" if venv_python().exists() else "(not created yet)"))
    print(f"uv             {shutil.which('uv') or 'not installed (optional)'}")

    _importable()
    from video_intel.config.loader import load_config
    from video_intel.storage.db import db_diagnostics
    from video_intel.storage.paths import ProjectPaths, project_id_for

    config = load_config()
    data_dir = config.data_dir
    paths = ProjectPaths(data_dir, project_id_for(config.project_name))

    print(f"project        {config.project_name}")
    print(f"data dir       {data_dir}")
    print(f"config files   {', '.join(Path(s).name for s in config.source_files)}")

    state = db_diagnostics(paths.database)
    if not state["exists"]:
        info("no state database yet: python run.py init")
    else:
        print(
            f"state db       {state['size_bytes']:,} bytes, "
            f"schema v{state['schema_version']}, journal {state['journal_mode']}"
        )
        if state["sidecars"]:
            warn(
                f"uncheckpointed WAL sidecars present ({', '.join(state['sidecars'])}); "
                "copy data/ only when no run is active, or run 'checkpoint' first"
            )
        if state["error"]:
            warn(f"state database: {state['error']}")

    videos = len(paths.videos())
    if videos:
        ok(f"data directory ready — {videos} stored video(s)")

    if (PROJECT_ROOT / ".env").is_file():
        ok(".env present (keys are read from it; values are never printed)")
    else:
        info("no .env — copy .env.example to .env to use a cloud provider")

    ollama_binary = shutil.which("ollama")
    models = ollama_models()
    if models is None:
        warn(f"ollama server not reachable at {OLLAMA_URL} (local inference unavailable)")
    else:
        shown = ", ".join(models[:6]) or "no models pulled"
        ok(f"ollama server reachable — models: {shown}")
        if not ollama_binary:
            info("ollama CLI not on PATH; the server is still usable over HTTP")
        elif not any(name.split(":")[0] == DEFAULT_MODEL.split(":")[0] for name in models):
            warn(f"model '{DEFAULT_MODEL}' is not pulled yet: ollama pull {DEFAULT_MODEL}")

    for variable in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(variable):
            ok(f"{variable} is set in the environment")

    print("─" * 46)


def checkpoint(python: Path) -> int:
    """Fold the write-ahead log into ``state.db`` so the file can be copied alone."""
    _importable()
    from video_intel.config.loader import load_config
    from video_intel.storage.db import StateStore
    from video_intel.storage.paths import ProjectPaths, project_id_for

    config = load_config()
    store = StateStore(ProjectPaths(config.data_dir, project_id_for(config.project_name)).database)
    store.checkpoint()
    ok(f"checkpointed {store.path}")
    return 0


def start_ollama() -> int:
    """Start a local Ollama server in the background, if one is installed."""
    if ollama_models() is not None:
        ok("ollama already running")
        return 0
    binary = shutil.which("ollama")
    if not binary:
        warn("ollama is not installed; skipping")
        return 0
    log = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or PROJECT_ROOT) / (
        "video-intel-ollama.log"
    )
    info(f"starting ollama serve in the background (log: {log})…")
    creationflags = 0
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    try:
        with log.open("ab") as handle:
            subprocess.Popen(
                [binary, "serve"],
                stdout=handle,
                stderr=handle,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
    except OSError as exc:  # pragma: no cover - platform dependent
        warn(f"could not start ollama: {exc}")
        return 1
    import time

    for _ in range(20):
        time.sleep(1)
        if ollama_models() is not None:
            ok(f"ollama is up (log: {log})")
            return 0
    warn(f"ollama did not answer within 20s; check {log}")
    return 1


def first_run(python: Path) -> int:
    setup()
    print()
    doctor(python)
    print()
    if run_cli(python, ["init"]) != 0:
        return 1
    print()
    print(f"""{_colour("32", "Everything is set up.")}

  start the web UI      python run.py web            → http://127.0.0.1:{DEFAULT_PORT}/
  analyze a video       python run.py analyze "https://www.youtube.com/watch?v=VIDEO_ID"
  quick run             python run.py analyze "URL" -m quick
  local model           ollama serve && ollama pull {DEFAULT_MODEL}
  use a cloud provider   put the key in .env, set execution_mode: hybrid in video-intel.yaml

  this directory is now device-independent: copy it anywhere, or point
  VIDEO_INTEL_DATA_DIR at a shared folder to use one library from several machines.
""")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return first_run(ensure_installed())

    command, rest = raw[0], raw[1:]
    known = {
        "setup",
        "doctor",
        "test",
        "integration",
        "lint",
        "init",
        "web",
        "ollama",
        "pull",
        "checkpoint",
        "shell",
        "help",
        "-h",
        "--help",
    }

    if command in {"help", "-h", "--help"}:
        print(__doc__)
        return 0

    python = ensure_installed()

    if command == "setup":
        setup()
        return 0
    if command == "doctor":
        doctor(python)
        return 0
    if command == "test":
        return run_tool(python, "pytest", rest)
    if command == "integration":
        return run_tool(python, "pytest", ["-m", "integration", *rest])
    if command == "lint":
        code = run_tool(python, "ruff", ["check", "."])
        code = run_tool(python, "ruff", ["format", "--check", "."]) or code
        return run_tool(python, "mypy", rest) or code
    if command == "ollama":
        return start_ollama()
    if command == "pull":
        binary = shutil.which("ollama")
        if not binary:
            warn("ollama is not installed")
            return 1
        return subprocess.run([binary, "pull", *rest]).returncode
    if command == "checkpoint":
        return checkpoint(python)
    if command == "shell":
        shell = os.environ.get("SHELL") or os.environ.get("COMSPEC")
        if shell and Path(shell).is_file():
            if os.name == "nt":  # pragma: no cover - exercised on Windows only
                subprocess.run([shell, "/K", "prompt"])
            else:
                os.execv(shell, [shell])
        return run_cli(python, ["--help"])

    if command == "web":
        port = DEFAULT_PORT
        args: list[str] = []
        index = 0
        while index < len(rest):
            if rest[index] == "--port" and index + 1 < len(rest):
                port = int(rest[index + 1])
                index += 2
                continue
            args.append(rest[index])
            index += 1
        print(f"{_colour('32', 'video-intel web UI')}: http://127.0.0.1:{port}/")
        info("runs the same CLI under the hood; press Ctrl+C to stop")
        return run_cli(python, ["web", "--port", str(port), *args])

    # Everything else is the CLI itself.
    if command not in known:
        rest = raw
    return run_cli(python, rest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print()
        raise SystemExit(130) from None
