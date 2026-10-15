# video-intel

Transcript-first, multi-AI knowledge extraction for YouTube videos.

A video URL in; a versioned, source-grounded knowledge dataset out. The transcript is
the source of truth — models are processing tools, never the authority on what the
speaker said.

Full specification: [`project.md`](project.md).
Implementation status: [phase 1](docs/phase-1-status.md), [phase 2](docs/phase-2-status.md),
[phases 3-9](docs/phases-3-9-status.md).

## Status

Phases 1-9 of the specification are implemented: the pipeline runs end to end from a
YouTube URL to a versioned knowledge dataset and its outputs.

```text
URL -> metadata -> transcript -> normalization -> chunks -> extraction (8 pipelines)
    -> consolidation -> verification -> conflicts -> dataset -> notes / summary / ppt / json
```

What is real today, and what is not:

- Real: transcript ingestion, chunking, provider calls, schema validation, deterministic
  verification, deduplication, conflict detection, dataset assembly, all four outputs,
  caching, resume.
- Manual: NotebookLM is a file exchange (export a packet, import your JSON answer).
- Not implemented: speech-to-text fallback, Gemini-specific adapter (the
  OpenAI-compatible adapter covers it).

## Requirements

- Python 3.11 or newer
- Nothing else. The launcher creates the virtual environment and installs dependencies.

## Setup

The portable way — one command on every platform:

```bash
python run.py            # create the environment, check it, initialize storage
```

`python3 run.py` or `./run.sh` do the same thing on macOS and Linux. Windows, macOS and
Linux all take the identical path from here; `run.sh` is only a wrapper around `run.py`.

<details>
<summary>Manual setup</summary>

```bash
uv venv && uv pip install -e ".[dev]"      # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env                      # optional; only the providers you intend to use
```

On Windows the virtual environment lives in `.venv\Scripts\` rather than `.venv/bin/`.
The console script (`video-intel`) is installed on every platform, and
`python -m video_intel` always works.

</details>

Secrets are read from environment variables at call time. Nothing sensitive is stored
in config files, project data or logs. `.env` is git-ignored and is never written by
this project.

## Commands

```bash
python run.py doctor                     # what this device can do: python, paths, db, ollama, keys
python run.py test                       # unit tests: no network, no API keys, no model
python run.py lint                       # ruff + mypy, strict

video-intel --version
video-intel init                         # create project layout + state database
video-intel providers                    # configured providers, key availability, adapter status
video-intel pipelines                    # pipelines, modes and presets
video-intel config show                  # effective configuration (secrets omitted)
video-intel config validate              # validate merged configuration
video-intel config path                  # where configuration is read from
video-intel inspect VIDEO_ID             # stored artifacts and stage state

# Analysis
video-intel analyze "YOUTUBE_URL" --dry-run           # print the plan, fetch nothing
video-intel analyze "YOUTUBE_URL"                     # the whole pipeline
video-intel analyze "YOUTUBE_URL" -m quick           # fewer pipelines and outputs
video-intel analyze "YOUTUBE_URL" --pipeline maximum-coverage
video-intel analyze "YOUTUBE_URL" --provider ollama --execution-mode hybrid
video-intel analyze "YOUTUBE_URL" -l en -l es        # caption language priority
video-intel analyze "YOUTUBE_URL" --until chunks     # stop after a named stage
video-intel analyze "YOUTUBE_URL" --force             # ignore cached results

# NotebookLM handoff (manual)
video-intel notebooklm export VIDEO_ID                # source packet + prompts + schema
video-intel notebooklm import answer.json --video VIDEO_ID
```

Stage exit codes: `0` done, `1` ingestion or total provider failure, `2` bad request or
configuration, `3` no longer used by `analyze` (kept for compatibility).

Stages run in order and are cached by content hash. A second `analyze` on the same video
reuses metadata, transcript, chunks and every extraction record; changing an output
format re-renders Markdown without calling a model. Switching models invalidates the
extraction records automatically.

### Providers

```bash
ollama serve                      # local, default; transcripts never leave the machine
video-intel providers             # what is configured, and whether an adapter exists
```

Implemented adapters: Ollama (`/api/chat`, JSON mode) and one OpenAI-compatible client
used for OpenAI, DeepSeek and Groq. `execution_mode: local_only` refuses any remote
provider at configuration time and again when a pipeline would use one.

## Configuration

Precedence, lowest to highest:

1. `src/video_intel/config/defaults/config.yaml` (packaged)
2. the user config — `%APPDATA%\video-intel\config.yaml`, `$XDG_CONFIG_HOME/video-intel/config.yaml`,
   `~/.config/video-intel/config.yaml`, or `~/Library/Application Support/…` on macOS
3. `./video-intel.yaml` (or `./config/video-intel.yaml`), found from any subdirectory
4. `VIDEO_INTEL_*` environment variables
5. explicit CLI flags

```bash
VIDEO_INTEL_DATA_DIR=~/data/video-intel
VIDEO_INTEL_EXECUTION_MODE=local_only
VIDEO_INTEL_LOG_LEVEL=DEBUG
```

`execution_mode: local_only` refuses any pipeline bound to a remote provider, so a
transcript cannot leave the machine by accident.

## Running on more than one device

The project directory is device-independent. Copy it to a laptop, a desktop or a USB
stick, run `python run.py`, and it behaves identically — the configuration resolves the
same way, and cached work is reused instead of repeated.

**Where things are found.** Commands work from the project root or from any
subdirectory. `data/`, `video-intel.yaml` and `.env` are located by walking up to the
project root (recognised by `.video-intel-root`, `video-intel.yaml`, this project's
`pyproject.toml`, or `.git`) rather than relative to the shell's working directory. A
relative `data_dir` is interpreted against that root. Drop an empty `.video-intel-root`
file in a directory to declare it the root explicitly.

**The cache survives a move.** The configuration hash that drives cache invalidation is
location-independent: the same settings produce the same hash on Linux, macOS and
Windows, whatever the install path or the user account. Copying a project to another
machine therefore reuses its transcript, chunk and dataset caches instead of silently
discarding them.

**Sharing one library between machines.** Point every device at a common folder:

```bash
VIDEO_INTEL_DATA_DIR=/mnt/library/video-intel
```

Two things to know when the folder is a network share:

- SQLite wants a local filesystem. Keep the state database local and let the JSON
  artifacts live on the share: `VIDEO_INTEL_DB_PATH=~/.local/state/video-intel.db`.
  Write-ahead logging is used automatically only where it is supported, and the busy
  timeout is raised for slow links.
- Run `python run.py checkpoint` before copying `data/`. It folds the write-ahead log
  into `state.db`, so the single file carries its own schema instead of depending on a
  sidecar.

**Running the same video twice at once.** Each video directory holds a lock carrying
its process id *and the device it was taken on*, so a pid from another machine is never
mistaken for a live local process. A lock left by a dead local process is reclaimed
immediately; one left by another machine is reclaimed only after a day, since its owner
cannot be probed.

**File names travel.** Artifact names are ASCII-only, separators in artifact paths are
always `/`, and names that Windows reserves (`con`, `nul`, `com1`, …) are rejected.
JSON and Markdown artifacts are read with a byte-order mark tolerated, so a file saved by
a Windows editor still loads.

## Storage layout

```text
data/
  state.db
  logs/video-intel.log
  projects/
    default/
      project.json
      VIDEO_ID/
        metadata.json
        raw_transcript.json
        normalized_transcript.json
        chunks.json
        raw_extractions/
        verified_items.json
        knowledge.json
        conflicts.json
        state.json
        outputs/
```

Writes are atomic (temp file + rename, then a directory flush) and keyed by content
hashes, so changing an output format re-generates outputs without re-running extraction.
The schema is created and migrated on first use, so a `state.db` copied without its
write-ahead log repairs itself on the next command.

## Development

```bash
python run.py test        # unit tests: no network, no API keys, no model
python run.py lint        # ruff check, ruff format --check, mypy --strict
```

Integration tests need a local Ollama server and a pulled model; they skip themselves
otherwise:

```bash
ollama serve && ollama pull qwen2.5:0.5b
VIDEO_INTEL_TEST_MODEL=qwen2.5:0.5b python run.py integration
```

The unit suite installs a socket guard, so a test that accidentally reaches the network
fails instead of silently depending on it. `tests/test_portability.py` covers the
cross-device and cross-platform guarantees described above.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Structure, schemas, config, logging, CLI, tests | done |
| 2 | Transcript ingestion and normalization | done |
| 3 | Deterministic chunking with overlap | done |
| 4 | Provider interface, Ollama + one cloud adapter, retries | done |
| 5 | Extraction pipelines (concept → coverage audit) | done |
| 6 | Normalization, dedup, verification, conflicts, dataset | done |
| 7 | Notes, JSON, PPT outline, summary | done |
| 8 | NotebookLM handoff workflow (manual, not an assumed API) | done |
| 9 | Resume, caching, integration tests, docs | done |