# Phase 1 status — Foundation

Date: 2026-10-14
Scope: Phase 1 of `project.md` §21 only (repository inspection, project structure,
schemas and data models, configuration, logging, CLI, unit-test infrastructure).

No transcript has been fetched, no model has been called and no dataset has been
produced by this phase. Everything below is scaffolding that later phases build on,
plus schema-level guarantees that are enforced by tests today.

## 1. Repository state before starting

| Item | Finding |
| --- | --- |
| Repository contents | Only `project.md` (the specification). No source, no tests, no packaging. |
| Version control | Git repository rooted at `/home/void` with **no commits** on branch `lauda`. The project directory is untracked. Nothing has been committed by this phase. |
| Python | 3.14.6 (`/usr/bin/python3`), `uv` 0.12.3 available. |
| Pre-installed relevant packages | pydantic 2.13.4, pydantic-settings 2.15, PyYAML 6.0.3, httpx 0.28.1, rich 15, pytest 9.1.1, mypy 2.3. |
| Not available before setup | typer, ruff, pytest-cov (installed into a project-local `.venv`). |
| Hardware | 8 CPU cores, 15 GiB RAM, **no GPU** (`nvidia-smi` absent). |
| `ollama` binary | Present at `/usr/local/bin/ollama`, but no adapter is written yet (Phase 4). |
| Network | Reachable (PyPI). No test in the suite requires it. |
| Sibling projects | `/home/void/projects/{ntxt,modeldoctor}` are Node/Firebase projects; no Python conventions to reuse. |

## 2. What now exists

```text
VOID OSINT extractor/
├── pyproject.toml            # hatchling build, deps, pytest/ruff/mypy config
├── README.md                 # usage, configuration precedence, roadmap table
├── .env.example              # placeholders only
├── .gitignore
├── docs/phase-1-status.md
├── src/video_intel/
│   ├── version.py
│   ├── logging_setup.py      # console + rotating file logging, secret redaction
│   ├── cli/
│   │   ├── context.py        # config load + logging + paths + store, per invocation
│   │   └── main.py           # analyze, providers, pipelines, inspect, init, config
│   ├── config/
│   │   ├── models.py         # AppConfig, ProviderConfig, PipelineConfig, ModeConfig,
│   │   │                     # PresetConfig, VerificationSettings, TranscriptionSettings
│   │   ├── loader.py         # layered YAML + env loading, deep merge, error reporting
│   │   └── defaults/config.yaml   # packaged defaults (providers, pipelines, modes, presets)
│   ├── schemas/
│   │   ├── common.py         # SCHEMA_VERSION, id patterns, enums, timestamp helpers
│   │   ├── transcript.py     # VideoMetadata, TranscriptSegment, Raw/NormalizedTranscript
│   │   ├── chunk.py          # ChunkingConfig, Chunk, ChunkSet, chunk cache key
│   │   ├── extraction.py     # ModelRef, ItemProvenance, ExtractedItem, ExtractionResponse,
│   │   │                     # RawExtractionRecord
│   │   ├── verification.py   # Evidence, VerificationResult, VerifiedItem, VerificationSummary
│   │   ├── conflict.py       # ConflictRecord, ConflictSet
│   │   ├── knowledge.py      # KnowledgeItem/Topic/Dataset, MergeRecord, ExtractionRun,
│   │   │                     # CompletenessReport
│   │   ├── outputs.py        # OutputRequest, OutputArtifact, PresentationOutline, SlideOutline
│   │   └── manifest.py       # StageRecord, StageManifest, ProjectManifest
│   └── storage/
│       ├── paths.py          # §15 directory layout, URL→video_id, path-traversal guards
│       ├── io.py             # atomic JSON/text writes, typed reads
│       └── db.py             # SQLite schema: projects, videos, stages, artifacts
└── tests/                    # 167 tests, 96% statement coverage
    ├── conftest.py           # env isolation, synthetic fixtures
    ├── fixtures/             # synthetic_transcript.json, synthetic_extraction.json
    └── test_*.py             # common/transcript/extraction/outputs/config/storage/logging/cli
```

## 3. Requirements mapped to implementation

| Spec | Status | Where |
| --- | --- | --- |
| §5 Provenance: stable ids, segment citation, start/end, pipeline, model, verification state, uncertainty flags | done (schema level) | `schemas/extraction.py`, `schemas/knowledge.py` |
| §3 Separate representations: raw / normalized / chunks / raw extractions / verified items / dataset / outputs | done (types + paths) | `schemas/*`, `storage/paths.py` |
| §9 Strict output contract, raw responses stored separately from parsed records | done (schema level) | `ExtractionResponse` (`extra="forbid"`), `RawExtractionRecord` |
| §10 Verification states incl. `partially_supported`, never claim full verification on partial data | done (schema level) | `VerificationStatus`, `VerificationSummary.complete`, `CompletenessReport` |
| §11 Explainable, reversible consolidation | done (schema level) | `MergeRecord(result_item_id, source_item_ids, evidence)` |
| §12 Conflicts recorded, resolved only from evidence | done (schema level) | `ConflictRecord` defaults to `needs_review`, `resolution=None` |
| §13 Versioned, hierarchical dataset with completeness flags | done (schema level) | `SCHEMA_VERSION = 1.0.0` on every persisted model |
| §15 Storage layout, stable ids, atomic writes, stage caching | done | `storage/paths.py`, `storage/io.py`, `storage/db.py`, `StageRecord.is_stale` |
| §16 CLI surface: `analyze`, `providers`, `pipelines`, `inspect` (+ `init`, `config`) | done | `cli/main.py` |
| §17 Secrets in env only, no hard-coded keys, no secret logging, explicit provider selection | done | `ProviderConfig.api_key_env`, `logging_setup.RedactionFilter`, `config show` |
| §18 Python, pydantic, httpx (declared), SQLite, YAML, typer, pytest | done | `pyproject.toml` |
| §19 Error handling for bad config/URL; partial work never labelled complete | done for Phase 1 scope | `ConfigError`, exit codes 1/2/3, `CompletenessReport` |
| §20 Test suite with synthetic fixtures, no network or API keys | done | `tests/` |

### CLI behaviour verified by hand

```text
video-intel --version                                       -> 0
video-intel providers [--json]                              -> 0
video-intel pipelines [--json]                              -> 0
video-intel config show | validate | path                   -> 0
video-intel init                                            -> 0  (creates layout + state.db)
video-intel inspect <video_id>                              -> 0  (reports absence honestly)
video-intel analyze <url> --dry-run                         -> 0  (prints plan, touches nothing)
video-intel analyze <url>                                   -> 3  (NOT IMPLEMENTED, explicit message)
video-intel analyze "https://example.com/x"                 -> 1  (invalid URL)
video-intel analyze <url> --provider openai                 -> 2  (refused under local_only)
video-intel config validate --config <bad file>             -> 2  (configuration error)
```

## 4. Quality gates

```text
.venv/bin/python -m pytest -q     -> 167 passed
.venv/bin/ruff check .            -> All checks passed
.venv/bin/ruff format --check .   -> 35 files already formatted
.venv/bin/mypy                    -> Success: no issues found in 25 source files (strict)
uv build --wheel                   -> config.yaml and py.typed included in the wheel
```

Tests require no network, no API keys and no model. Fixtures are synthetic and
labelled as such.

## 5. Deliberate decisions

1. **The model output contract forbids unknown fields.** Unexpected keys fail
   validation and enter the repair/retry path (§9) instead of being absorbed silently.
2. **The extraction contract requires `source_segment_ids`.** An item with no
   transcript citation cannot be constructed, so unsourced model output cannot enter
   the dataset at all.
3. **`execution_mode: local_only` is enforced in configuration validation**, not just
   documented: any pipeline bound to a remote provider makes the config invalid. The
   CLI repeats the check for `--provider` overrides.
4. **Secrets are referenced by variable name only.** `config show`, `providers --json`
   and every log record go through redaction; a test asserts key values never appear.
5. **Provider adapters are declared `implemented: false`.** The CLI prints that status
   so nothing implies a working integration that does not exist yet.
6. **NotebookLM is configured as `notebooklm_handoff`, not as an API provider**, and
   the CLI states that it is a manual workflow until a supported interface is verified.
7. **Read-only commands create no directories.** `providers`, `pipelines` and
   `config show` do not write a log file into the working directory (tested).
8. **`analyze` without `--dry-run` exits 3** rather than returning a summary. The
   specification forbids substituting ordinary summarization for extraction.
9. **`data_dir` defaults to `./data`** (matching §15) and is resolved against the
   working directory; `VIDEO_INTEL_DATA_DIR` or `--data-dir` override it.

## 6. Known limitations of this phase

- No transcript acquisition, normalization, chunking, extraction, verification,
  conflict resolution, dataset assembly or output rendering exists yet. All such
  models are types only.
- The `analyze` command validates and plans; it does not execute stages.
- Stage caching is defined and stored but nothing populates it yet; resume behaviour
  is Phase 9.
- `ChunkSet`/`Chunk` are not produced by a chunker yet (Phase 3).
- No prompts, no provider HTTP client, no retries, no concurrency (Phase 4).
- `pipelines` shows `passes` per pipeline, but `maximum-coverage` currently differs
  from `default` only by intent in the config comment; a second pass is not
  implemented and must be wired in Phase 5.
- Integration tests and CI are not configured (Phase 9).

## 7. Next concrete step — Phase 2 (transcript ingestion)

1. Add `video_intel/ingest/` with `youtube.py` (metadata + captions via
   `youtube-transcript-api`), `normalize.py`, and typed errors
   (`TranscriptUnavailable`, `VideoUnavailable`, `RetrievalFailed`).
2. Add `youtube-transcript-api` as a dependency; keep the fetch behind an injectable
   client so unit tests use a synthetic fetcher and never touch the network.
3. Implement `transcript_id → segment_001..N` assignment, preserving real cue
   timings, and persist `metadata.json` + `raw_transcript.json` through
   `storage/io.write_model_atomic`.
4. Implement normalization as a recorded list of `NormalizationStep`s that never
   deletes repeated passages and never changes meaning.
5. Wire `StageId.METADATA` / `RAW_TRANSCRIPT` / `NORMALIZED_TRANSCRIPT` into
   `StateStore` so stage state, hashes and resume already work.
6. Extend tests: ingestion against a synthetic fetcher, timestamp preservation,
   missing-caption and unavailable-video paths, and the offline speech-to-text opt-in
   path reported but not fabricated.

Nothing has been committed; the working tree is left for review.
