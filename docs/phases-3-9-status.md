# Phases 3–9 status — Chunking to reliability

Date: 2026-10-14
Scope: Phases 3, 4, 5, 6, 7, 8 and 9 of `project.md` §21, plus a debugging pass.

Phases 1 and 2 are documented in [`phase-1-status.md`](phase-1-status.md) and
[`phase-2-status.md`](phase-2-status.md). The status documents for those phases also
list the bugs that were found and fixed; the ones fixed in this round are in §8 below.

## 1. Repository state when this round started

Phase 2 complete: 281 tests passing, ingestion verified live against a real video.
Two structural gaps were known and are now closed: nothing ran past the transcript, and
several commands implied work that did not exist.

## 2. What was built

```text
src/video_intel/
├── pipeline/
│   ├── stages.py          canonical stage order; what `--until` may name
│   ├── chunking.py        deterministic sentence-aware chunking with overlap (§6)
│   ├── ingest_stages.py   metadata → transcript → normalization → chunks
│   ├── analysis_stages.py extraction, dataset assembly, outputs
│   └── runner.py          cache-aware runner: reuse, atomic writes, stage state
├── providers/
│   ├── base.py            Provider protocol, typed errors, retries, JSON extraction
│   ├── ollama.py          local adapter (Ollama /api/chat, format=json)
│   ├── openai_compatible.py  one adapter for OpenAI/DeepSeek/Groq/vLLM/LM Studio
│   └── registry.py        configuration-driven construction, execution-mode guard
├── extraction/
│   ├── prompts.py         versioned per-pipeline prompts + the JSON contract
│   └── engine.py          per (pipeline, pass, chunk) extraction, repair, provenance
├── aggregation/
│   ├── consolidate.py     dedup + complementary merge + contradiction detection
│   ├── verify.py          deterministic evidence verification
│   ├── conflicts.py       conflict records and evidence-based resolution
│   └── dataset.py         hierarchical, evidence-derived dataset assembly
├── outputs/render.py      full notes, summary, PPT outline, knowledge.json
├── notebooklm/handoff.py  manual export/import packets, validation, merge
└── cli/main.py            the whole pipeline behind `analyze`, `inspect`, `notebooklm`
```

## 3. Requirements mapped to implementation

| Spec | Status | Where / evidence |
| --- | --- | --- |
| §3 Separate representations preserved | done | raw / normalized / chunks / `raw_extractions/*/pass_N/*.json` / `verified_items.json` (written by the verification stage) / `knowledge.json` / `outputs/` |
| §4 Acquisition without fabrication | done | typed errors only; no fallback source exists |
| §5 Provenance on every item | done | `ItemProvenance` per item; timestamps copied from cited segments, never from the model |
| §6 Chunking: sentences, limits, overlap, determinism | done | `chunking.py`; 65 property tests over random transcripts |
| §7 Provider abstraction, Ollama + cloud, no fake NotebookLM API | done | `providers/`; NotebookLM is a manual handoff module |
| §8 Eight pipelines, configurable, user-selectable | done | `extraction/prompts.py`, one profile per pipeline including the coverage audit |
| §9 Strict contract, repair, raw storage, no corruption | done | `extra="forbid"`; one repair attempt; raw text always stored |
| §10 Verification states, no majority voting | done | `verify.py` + `conflicts.py`; conflicts resolved only from transcript evidence |
| §11 Dedup categories, reversible, unique items kept | done | `consolidate.py` + `MergeRecord` |
| §12 Conflict detection and honest status | done | `build_conflicts` resolves only when the transcript states one of the disputed figures |
| §13 Hierarchical dataset, versioned schema | done | `DatasetBuilder`; topics derived from cited segments, never from a fixed outline |
| §14 Outputs generated from the dataset | done | `outputs/render.py`; PPT target shortfall reported, never padded |
| §15 Storage, caching, selective invalidation | done | `StageRunner`, `StageRecord.is_stale`, per-record extraction cache |
| §16 CLI surface, progress, output locations | done | `analyze --until/-l/--force/--dry-run`, `inspect`, `notebooklm export/import` |
| §17 Secrets, execution modes | done | unchanged from Phase 1; enforced in config validation and the registry |
| §19 Partial failure handling | done | per-record failures, failed stage state, exit codes |
| §20 Tests | done | 516 tests; §20's six proof obligations are explicit tests (see §5) |

## 4. CLI behaviour

```text
video-intel analyze URL                            -> 0  full run: ingest, chunk, extract,
                                                          verify, assemble, render
video-intel analyze URL --dry-run                  -> 0  plan only
video-intel analyze URL --until chunks             -> 0  ingestion only
video-intel analyze URL --until extractions        -> 0  stop after extraction
video-intel analyze URL --force                    -> 0  recompute everything
video-intel analyze URL -m quick                   -> 0  fewer pipelines and outputs
video-intel analyze URL (provider down)            -> 1  "no provider returned usable output"
video-intel analyze URL (some provider calls fail) -> 0  partial results kept, marked incomplete
video-intel notebooklm export VIDEO_ID             -> 0  packet + prompts + schema + how-to
video-intel notebooklm import answer.json --video … -> 0  validate, merge, rebuild dataset
video-intel inspect VIDEO_ID                       -> 0  artifacts + stage states
```

## 5. The six proof obligations from §20

| Obligation | Test |
| --- | --- |
| A unique source-supported fact is not removed because only one model extracted it | `test_aggregation.py::test_consolidator_keeps_a_unique_single_model_item` |
| Unsupported model-generated claims are flagged | `test_aggregation.py::test_unsupported_claim_is_flagged_and_excluded_from_outputs`, `test_cli_pipeline.py::test_unsupported_claims_are_excluded_and_reported` |
| Repeated explanations are consolidated without losing distinct details | `test_aggregation.py::test_consolidator_preserves_repeated_details`, `test_ingest_normalize.py::test_repeated_explanations_are_not_removed` |
| Multiple source timestamps survive consolidation | `test_aggregation.py::test_multiple_timestamps_survive_consolidation` |
| Contradictory claims remain unresolved when the transcript cannot resolve them | `test_aggregation.py::test_unresolvable_contradiction_stays_open` |
| Changing an output format does not trigger unnecessary AI calls | `test_cli_pipeline.py::test_changing_output_format_does_not_call_the_model`, `test_reliability.py::test_outputs_are_regenerated_when_only_the_dataset_changes` |

## 6. Quality gates

```text
.venv/bin/python -m pytest -q     -> 516 passed, 3 skipped (92% statement coverage)
.venv/bin/ruff check .            -> All checks passed
.venv/bin/ruff format --check .   -> 85 files already formatted
.venv/bin/mypy                    -> Success: no issues found in 56 source files (strict)
```

Skipped tests are the integration ones that need a specific pulled model.

Integration tests (`-m integration`) were run against a real local Ollama server with
`qwen2.5-coder:7b-opencode`: JSON-contract compliance, extraction end to end with real
timings and citation checks, and correct behaviour against a dead endpoint — 5 passed.

## 7. Deliberate decisions

1. **Deterministic verification, not a second model by default.** Token coverage against
   the cited passages is reproducible and cannot be talked into approving a fabricated
   claim; a verifier model can be configured later and is recorded separately.
2. **Partially supported items are labelled everywhere.** Any item that is not fully
   supported shows its status in the notes, so a weakly supported claim is never
   presented as something the speaker plainly said.
3. **Unknown extraction keys are stripped and counted; ill-typed fields are a contract
   error.** Burning a repair call on a harmless extra key would be wasteful, but a
   malformed `details` field means the model did not follow the contract.
4. **Items citing nothing real are dropped.** An item with no valid citation cannot enter
   the dataset, which makes "unsupported claims are flagged" a structural property rather
   than a post-hoc filter.
5. **Topics are evidence-derived.** A concept item becomes a topic; other items join it
   when most of their cited segments fall inside it; everything else is grouped by type.
   No outline from the specification is hard-coded.
6. **Conflict resolution needs the transcript to state exactly one disputed figure.**
   Anything ambiguous stays `needs_review`, and majority agreement is never used.
7. **Extraction records are keyed to their model.** A record is reused only when the
   pipeline, pass, model, provider and prompt version all match; otherwise it is deleted
   and re-extracted.
8. **Outputs are hashed against the dataset.** Changing an output format re-renders
   Markdown without touching the model.
9. **NotebookLM is a file exchange.** Export writes a timestamped packet, prompts and the
   schema; import validates, grounds and re-verifies whatever comes back.

## 8. Bugs found and fixed in this round

| Found by | Bug | Fix |
| --- | --- | --- |
| Integration tests (ResourceWarning as error) | The Ollama/OpenAI adapters leaked sockets whenever a caller forgot `aclose()`; the CLI's cleanup only ran on the success path | `try/finally` around every provider use in the CLI; integration tests close explicitly |
| Review of the extraction cache | Switching from one model to another silently reused the previous model's extraction records | `_cache_mismatch` compares pipeline, pass, model, provider and prompt version, and deletes stale records |
| Test asserting chunk budgets | The overlap seed could push a chunk past `max_chars` | overlap is capped so the first new piece still fits; budget wins over overlap |
| Test with `overlap_chars: 0` | Zero overlap still repeated one sentence | `overlap_chars <= 0` means no overlap |
| Test of contradictory figures | `forty` vs `sixty` was not detected: number extraction only saw digits | number words are canonicalised to digits for comparison |
| Test of consolidation | Merging kept the timestamp span of only the surviving item | merged provenance widens to cover every cited passage |
| Review of topic building | Requiring two shared segments left items unattached | items attach when most of their evidence sits inside the topic |
| `--until output_summary` run | `selected_stages` rejected any stage past ingestion | one canonical `STAGE_ORDER`; `--until` accepts any runnable stage |
| `--until metadata` run | The condition was inverted, so analysis ran anyway | `_stops_before_analysis` compares against the canonical order |
| Review of the CLI | `--until output_notes` still wrote every output | output selection is cut at the named stage |
| `RawExtractionRecord` | No pass index, so a second pass could reuse first-pass records | `pass_index` added and validated during cache reuse |
| Review | `video_id_from_url` duplicated the ingest URL parser | storage now delegates to `ingest.youtube` |
| Test suite | A unit test opened a real network connection (caught by `filterwarnings = error`) | `block_network` fixture fails any unit test that tries to connect |

## 9. Known limitations

- **No model-quality tuning.** Prompts are versioned and tested structurally, but no
  benchmark measures extraction quality against a labelled transcript. That is the next
  thing worth building: a fixture corpus with expected items.
- **Deterministic verification is lexical.** A paraphrase that shares few words with the
  transcript is marked partially supported even when a human would accept it. The
  threshold is a configuration-free constant today and should become configurable when
  a labelled set exists to tune it against.
- **Consolidation is pairwise and greedy.** It is explainable and reversible, but the
  order of items can influence the outcome. Clusters are not built explicitly.
- **Topic naming uses the concept statement verbatim**, truncated at 120 characters; no
  title synthesis.
- **Gemini has no adapter.** It is declared in configuration and reported as not
  implemented; OpenAI-compatible providers cover Gemini's OpenAI-compatible endpoint.
- **Speech-to-text fallback is still unimplemented** (Phase 2 limitation, unchanged).
- **Resumption is per stage and per extraction record.** A crash mid-pipeline restarts the
  current stage.
- **Concurrency is per video, across devices.** Each video directory holds a lock
  recording the owning process *and the device it was taken on*, so two runs of the same
  video cannot overlap even when the data directory is a shared folder. The lock is
  advisory: the SQLite state database additionally uses WAL where the filesystem supports
  it, a 30 second busy timeout otherwise, and an explicit transaction per stage update.
- **NotebookLM import assumes one JSON file per answer**; a directory of files is not
  scanned automatically.

## 10. Next concrete steps

1. **Extraction quality benchmark.** Build a labelled fixture (transcript + expected
   items) and measure precision/recall per pipeline against the local model. This turns
   prompt work and verification thresholds into measurements instead of opinions.
2. **Tune verification thresholds** against that benchmark, and expose them in
   configuration.
3. **Add a job lock** (`data/projects/<id>/.lock`) so two runs cannot corrupt one video's
   artifacts.
4. **Split long prompts further**: the live run shows that a 4.6k-token prompt on a CPU
   model is slow; a time-budget-aware chunk size would help.
5. **Deeper NotebookLM ergonomics**: scan a directory of answers, and support the
   structured import NotebookLM users actually paste.
