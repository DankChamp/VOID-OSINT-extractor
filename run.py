#!/usr/bin/env python3
"""VOID OSINT extractor: one entry point for every tool, on every platform.

    python run.py                     set up, then list the tools
    python run.py setup               create the shared venv and install every tool
    python run.py doctor              per-tool environment report
    python run.py test [args]         run every tool's test suite
    python run.py lint                ruff + mypy over every tool
    python run.py <tool> <command>    run a tool's CLI, e.g.
                                      python run.py video-intel analyze "URL" -m quick
    python run.py web [--port 8765]   start a tool's web frontend
    python run.py help                this list

Why this is Python rather than more shell
-----------------------------------------

A shell script covers Linux and macOS and nothing else: it needs bash, assumes
``.venv/bin/python``, and depends on ``python3`` and ``curl`` by name. This file uses the
standard library only, so the identical commands work on Windows, macOS and Linux and
the correct virtual-environment layout is picked for the host. ``run.sh`` is a thin
wrapper around this file, so the two cannot drift apart.

How tools are found
--------------------

Every directory under ``tools/`` containing a ``pyproject.toml`` is a tool. The launcher
reads each one to learn its name and console scripts, so adding a tool needs no change
here at all. All tools share the single virtual environment in ``.venv`` at the root, so
they cannot pin conflicting versions of the same library.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = REPO_ROOT / "tools"
VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_PORT = 8765
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

IS_TTY = sys.stdout.isatty()


def _colour(code: str, text: str) -> str:
    if not IS_TTY or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def _emit(message: str) -> None:
    """Print a line, flushed.

    Subprocesses write straight to the file descriptor, so a buffered print would
    reorder output around their lines and make the launcher's progress look jumbled.
    """
    print(message, flush=True)


def ok(message: str) -> None:
    _emit(_colour("32", "✓") + f" {message}")


def warn(message: str) -> None:
    _emit(_colour("33", "!") + f" {message}")


def fail(message: str) -> None:
    print(_colour("31", "✗") + f" {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def info(message: str) -> None:
    _emit(_colour("2", message))


# ------------------------------------------------------------------ tool discovery


@dataclass
class Tool:
    """One installable tool under ``tools/``."""

    name: str
    root: Path
    scripts: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def source_dir(self) -> Path:
        return self.root / "src"

    def script_path(self, script: str) -> Path:
        """Absolute path to one of this tool's console scripts in the shared venv.

        ``Scripts`` on Windows, ``bin`` elsewhere, with a ``.exe`` suffix there. Probed
        rather than assumed so a checkout copied between platforms still resolves.
        """
        if os.name == "nt":
            return VENV_DIR / "Scripts" / f"{script}.exe"
        return VENV_DIR / "bin" / script

    @property
    def primary_script(self) -> str | None:
        """The console script that best represents this tool, if it declares one."""
        return self.scripts[0] if self.scripts else None

    def __str__(self) -> str:
        return self.name


def _parse_pyproject(path: Path) -> dict[str, Any] | None:
    try:
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, Exception):
        return None


def discover_tools() -> list[Tool]:
    """Every tool in ``tools/``, in directory order."""
    if not TOOLS_DIR.is_dir():
        return []
    found: list[Tool] = []
    for entry in sorted(TOOLS_DIR.iterdir()):
        pyproject = entry / "pyproject.toml"
        if not entry.is_dir() or not pyproject.is_file():
            continue
        data = _parse_pyproject(pyproject)
        project = (data or {}).get("project", {})
        name = project.get("name") or entry.name
        scripts = sorted(project.get("scripts", {}) or {})
        found.append(
            Tool(
                name=name,
                root=entry,
                scripts=scripts,
                description=str(project.get("description", "")),
            )
        )
    return found


def find_tool(name: str) -> Tool | None:
    for tool in discover_tools():
        if tool.name == name or tool.root.name == name:
            return tool
    return None


# ------------------------------------------------------------------ virtual environment


def venv_python() -> Path:
    """The shared interpreter: ``Scripts`` on Windows, ``bin`` elsewhere.

    Probed rather than assumed, so the same checkout works after being copied from a Mac
    to a PC.
    """
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def base_python() -> str:
    """An interpreter suitable for building the virtual environment.

    The running interpreter is preferred, so ``python run.py`` never silently installs a
    different Python version than the one it was launched with.
    """
    return sys.executable or shutil.which("python3") or shutil.which("python") or "python"


def ensure_installed() -> Path:
    """Return the shared interpreter, setting the workspace up when necessary."""
    python = venv_python()
    if not python.exists():
        setup(create_env=True)
    return python


def _pip_install(python: Path, tools: list[Tool]) -> None:
    """Install every tool and the shared dev tooling with plain pip."""
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    for tool in tools:
        subprocess.run([str(python), "-m", "pip", "install", "-e", str(tool.root)], check=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "anyio>=4.0",
            "pytest-cov>=5.0",
            "mypy>=1.10",
            "ruff>=0.5",
        ],
        check=True,
    )


def setup(*, create_env: bool = True) -> Path:
    """Create the shared virtual environment and install every tool, editable.

    ``uv sync`` is preferred because it understands the workspace and resolves every
    member in one pass. It is not required: a missing, broken or too-old ``uv`` falls
    back to pip rather than failing the install.
    """
    python = venv_python()
    tools = discover_tools()
    if not python.exists():
        if not create_env:
            fail(f"no virtual environment at {VENV_DIR}; run: python run.py setup")
        info("creating the shared virtual environment…")
        subprocess.run([base_python(), "-m", "venv", str(VENV_DIR)], check=True)
        python = venv_python()

    uv = shutil.which("uv")
    installed_with_uv = False
    if uv:
        try:
            # --all-packages installs every workspace member (that is, every tool);
            # without it uv only syncs the root, which ships no code of its own.
            # --all-groups brings in the shared dev tooling.
            subprocess.run(
                [uv, "sync", "--all-packages", "--all-groups"],
                cwd=str(REPO_ROOT),
                check=True,
            )
            installed_with_uv = True
        except (subprocess.CalledProcessError, OSError) as exc:
            warn(f"uv could not set up the workspace ({exc}); falling back to pip")

    if not installed_with_uv:
        _pip_install(python, tools)

    ok(f"environment ready: {python}")
    for tool in tools:
        ok(f"installed {tool.name}")
    return python


def run_tool(python: Path, tool: str, args: list[str]) -> int:
    """Run a development tool from the shared environment.

    Invoked as ``python -m <tool>`` on every platform: the console-script layout differs
    between ``bin`` and ``Scripts``, and a stale entry point is a confusing failure mode.
    """
    return subprocess.run(
        [str(python), "-m", tool, *args], cwd=str(REPO_ROOT), check=False
    ).returncode


def run_cli(python: Path, tool: Tool, args: list[str]) -> int:
    """Run a tool's command line entry point.

    The console script the tool declares in ``[project.scripts]`` is preferred, because
    that is genuinely its CLI and it always exists once installed. ``python -m <package>``
    is only a fallback: it needs a ``__main__`` module, which plenty of tools do not ship.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    sources = (
        [str(entry / "src") for entry in sorted(TOOLS_DIR.iterdir()) if (entry / "src").is_dir()]
        if TOOLS_DIR.is_dir()
        else []
    )
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join([*sources, existing]) if existing else os.pathsep.join(sources)
    )

    script = tool.primary_script
    if script:
        path = tool.script_path(script)
        if path.exists():
            return subprocess.run(
                [str(path), *args], cwd=str(REPO_ROOT), env=env, check=False
            ).returncode

    package = tool.name.replace("-", "_")
    if (tool.source_dir / package / "__main__.py").is_file():
        return subprocess.run(
            [str(python), "-m", package, *args], cwd=str(REPO_ROOT), env=env, check=False
        ).returncode

    fail(
        f"cannot launch {tool.name}: it declares no console script and has no "
        f"{package}/__main__.py. Add a [project.scripts] entry to "
        f"{tool.root.name}/pyproject.toml, then run: python run.py setup"
    )


# -------------------------------------------------------------------------- reporting


def ollama_models() -> list[str] | None:
    """Model names from a reachable local Ollama, or ``None`` when it is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [str(entry.get("name", "")) for entry in payload.get("models", [])]


def workspace_report() -> None:
    """Everything about this checkout that a user might need to know."""
    print("VOID OSINT extractor")
    print("─" * 46)
    print(f"platform       {sys.platform}")
    print(f"python         {sys.version.split()[0]}")
    print(f"workspace root {REPO_ROOT}")
    print(
        f"shared venv    {venv_python()} " + ("" if venv_python().exists() else "(not created yet)")
    )
    print(f"uv             {shutil.which('uv') or 'not installed (optional)'}")
    print()

    tools = discover_tools()
    if not tools:
        warn("no tools found under tools/")
    for tool in tools:
        commands = ", ".join(tool.scripts) or "none"
        print(f"  {tool.name:<16} {commands}")
        if tool.description:
            print(_colour("2", f"  {'':<16} {tool.description[:66]}"))
    print()

    for path in (REPO_ROOT / ".env",):
        if path.is_file():
            ok(".env present (keys are read from it; values are never printed)")
        else:
            info("no .env — copy .env.example to .env to use a cloud provider")

    models = ollama_models()
    if models is None:
        warn(f"ollama server not reachable at {OLLAMA_URL} (local inference unavailable)")
    else:
        shown = ", ".join(models[:6]) or "no models pulled"
        ok(f"ollama server reachable — models: {shown}")
        if not any(name.split(":")[0] == DEFAULT_MODEL.split(":")[0] for name in models):
            warn(f"model '{DEFAULT_MODEL}' is not pulled yet: ollama pull {DEFAULT_MODEL}")

    for variable in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(variable):
            ok(f"{variable} is set in the environment")
    print("─" * 46)


def doctor(python: Path) -> int:
    """Report the workspace, then ask each tool what it can do."""
    workspace_report()
    status = 0
    for tool in discover_tools():
        print()
        info(f"── {tool.name} " + "─" * max(1, 40 - len(tool.name)))
        status = run_cli(python, tool, ["doctor"]) or status
    return status


def test_all(python: Path, args: list[str]) -> int:
    """Run every tool's suite in one pytest process."""
    return run_tool(python, "pytest", args)


def lint_all(python: Path, args: list[str]) -> int:
    """Ruff over the workspace, then mypy over every installed package."""
    code = run_tool(python, "ruff", ["check", "."])
    code = run_tool(python, "ruff", ["format", "--check", "."]) or code

    packages: list[str] = []
    for tool in discover_tools():
        if not tool.source_dir.is_dir():
            continue
        # Paths, not package names: a tool's package is only importable once its src/
        # is installed, and mypy resolves these relative to the working directory.
        packages.extend(
            str(entry.relative_to(REPO_ROOT))
            for entry in sorted(tool.source_dir.iterdir())
            if entry.is_dir() and (entry / "__init__.py").is_file()
        )
    if packages:
        code = run_tool(python, "mypy", [*packages, *args]) or code
    else:
        warn("no importable packages found; skipping mypy")
    return code


def start_ollama() -> int:
    """Start a local Ollama server in the background, if one is installed."""
    if ollama_models() is not None:
        ok("ollama already running")
        return 0
    binary = shutil.which("ollama")
    if not binary:
        warn("ollama is not installed; skipping")
        return 0
    log = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or REPO_ROOT) / (
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
    workspace_report()
    print()
    print(f"""{_colour("32", "Everything is set up.")}

  see the tools        python run.py doctor
  run a tool           python run.py <tool> --help
  check everything     python run.py test && python run.py lint

  this checkout is device-independent: copy it anywhere and run `python run.py`.
  each tool reads VIDEO_INTEL_* settings and the workspace data directory, so a
  shared library can be pointed at from several machines with VIDEO_INTEL_DATA_DIR.
""")
    return 0


USAGE = __doc__


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return first_run(ensure_installed())

    command, rest = raw[0], raw[1:]

    if command in {"help", "-h", "--help"}:
        print(USAGE)
        return 0

    # Commands the launcher implements itself. Anything else is a tool name, or a tool's
    # own flags when no tool matches, in which case it is forwarded verbatim.
    launcher_commands = {
        "setup",
        "doctor",
        "status",
        "test",
        "lint",
        "ollama",
        "pull",
        "shell",
    }

    if command == "setup":
        setup()
        return 0
    if command in {"doctor", "status"}:
        return doctor(ensure_installed())
    if command == "ollama":
        return start_ollama()
    if command == "pull":
        binary = shutil.which("ollama")
        if not binary:
            warn("ollama is not installed")
            return 1
        return subprocess.run([binary, "pull", *rest]).returncode

    python = ensure_installed()

    if command == "test":
        return test_all(python, rest)
    if command == "lint":
        return lint_all(python, rest)
    if command == "shell":
        shell = os.environ.get("SHELL") or os.environ.get("COMSPEC")
        if shell and Path(shell).is_file():
            if os.name == "nt":  # pragma: no cover - exercised on Windows only
                subprocess.run([shell, "/K", "prompt"])
            else:
                os.execv(shell, [shell])
        print(USAGE)
        return 0

    tool = find_tool(command) if command not in launcher_commands else None
    if tool is not None:
        arguments = rest
        if arguments and arguments[0] == "web":
            arguments = _web_arguments(arguments)
        return run_cli(python, tool, arguments)

    if command in launcher_commands:
        print(USAGE)
        return 2

    fail(f"unknown command or tool: {command!r} (try: python run.py help)")


def _web_arguments(arguments: list[str]) -> list[str]:
    """Print the URL for a web command, then pass the flags through unchanged."""
    port = DEFAULT_PORT
    index = 1
    while index < len(arguments):
        if arguments[index] == "--port" and index + 1 < len(arguments):
            port = int(arguments[index + 1])
            index += 2
            continue
        index += 1
    print(f"{_colour('32', 'web UI')}: http://127.0.0.1:{port}/")
    info("runs the same CLI under the hood; press Ctrl+C to stop")
    return arguments


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print()
        raise SystemExit(130) from None
