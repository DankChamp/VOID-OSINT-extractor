# VOID OSINT extractor

A workspace of OSINT extraction tools. One shared environment, one launcher, one data
directory — and one tool added at a time without disturbing the others.

| Tool | What it does |
| --- | --- |
| [`video-intel`](tools/video-intel/) | Transcript-first, multi-AI knowledge extraction for YouTube videos |

The transcript is the source of truth in every tool that touches one: models are
processing tools, never the authority on what a source actually said.

## Requirements

- Python 3.11 or newer
- Nothing else. The launcher creates the environment and installs every tool.

## Setup

```bash
python run.py            # create the shared environment, install every tool, list them
```

`python3 run.py` or `./run.sh` do the same thing on macOS and Linux. Windows, macOS and
Linux all take the identical path from here; `run.sh` is only a wrapper around `run.py`.

<details>
<summary>Manual setup</summary>

```bash
uv sync --all-groups                                    # recommended
# or, without uv:
python3 -m venv .venv && .venv/bin/pip install -e tools/video-intel
```

On Windows the environment lives in `.venv\Scripts\` rather than `.venv/bin/`.

</details>

## Commands

```bash
python run.py                       set up, then list the tools
python run.py setup                 create the shared venv and install every tool
python run.py doctor                what this device can do, per tool
python run.py test [args]           run every tool's test suite
python run.py lint                  ruff + mypy over every tool
python run.py <tool> <command>      run a tool's CLI
python run.py help                  this list

python run.py video-intel analyze "https://youtu.be/VIDEO_ID" -m quick
python run.py video-intel web
python run.py video-intel --help
```

## Adding a tool

Create `tools/<name>/pyproject.toml`. The launcher discovers tools from the filesystem
and the workspace globs `tools/*`, so nothing else needs editing:

```text
tools/<name>/
  pyproject.toml     # [project] name, dependencies, [project.scripts]
  src/<package>/
  tests/
```

```bash
python run.py setup        # installs the new tool into the shared environment
python run.py test         # and its suite runs with everything else
```

Two deliberate constraints:

- **One shared environment.** Tools cannot pin conflicting versions of the same library.
  If two tools ever need incompatible versions, split them into separate repositories
  rather than into separate venvs inside this one.
- **The workspace root is not a tool.** It carries no importable code, only shared
  tooling configuration, so a root-level `pip install .` installs nothing.

## Configuration and data

The workspace root holds the things a user owns: `data/`, `.env`, and each tool's
`video-intel.yaml`-style configuration. Tools locate them by walking up to the root, so
commands behave identically from any subdirectory, on any device, after the checkout has
been moved.

Environment variables override everything:

```bash
VIDEO_INTEL_DATA_DIR=~/data/osint          # where artifacts are written
VIDEO_INTEL_EXECUTION_MODE=local_only     # refuse to send a transcript off the machine
VIDEO_INTEL_LOG_LEVEL=DEBUG
```

## Running on more than one device

The checkout is device-independent. Copy it to a laptop, a desktop or a USB stick, run
`python run.py`, and it behaves identically — the configuration resolves the same way
and cached work is reused instead of repeated.

To share one library between machines, point each at a common folder with
`VIDEO_INTEL_DATA_DIR`. Two notes when that folder is a network share:

- Keep the state database on local disk with `VIDEO_INTEL_DB_PATH`; SQLite wants a local
  filesystem, and the launcher verifies the journal mode it actually got.
- Run `video-intel checkpoint` before copying, so a copied `state.db` carries its own
  schema rather than depending on a sidecar. Without it, copying `state.db` alone after an
  interrupted run yields a valid-looking file containing no tables at all.

Full detail, including the per-device locking model, is in
[`tools/video-intel/README.md`](tools/video-intel/README.md).

## Development

```bash
python run.py test        # no network, no API keys, no model
python run.py lint        # ruff check, ruff format --check, mypy --strict
```

Each tool's suite installs a socket guard, so a test that accidentally reaches the
network fails instead of silently depending on it.

## Status

See [`tools/video-intel/docs/`](tools/video-intel/docs/) for per-phase implementation
status and [`project.md`](project.md) for the original specification.