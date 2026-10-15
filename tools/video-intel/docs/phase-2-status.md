# Phase 2 status — Transcript ingestion

Date: 2026-10-14
Scope: Phase 2 of `project.md` §21 (YouTube metadata, transcript acquisition, timestamp
preservation, transcript storage, cleaning and normalization).

Phase 1 (`docs/phase-1-status.md`) is unchanged apart from the fixes listed in §6 below.

**No AI model has been called and no knowledge dataset exists yet.** `analyze` now runs
real ingestion and then stops with an explicit message.

## 1. What was built

```text
src/video_intel/
├── ingest/
│   ├── errors.py       typed failures: InvalidVideoUrl, VideoUnavailable,
│   │                   TranscriptUnavailable, TranscriptRetrievalFailed,
│   │                   MetadataRetrievalFailed, TranscriptionDisabled
│   ├── youtube.py      URL validation (watch/youtu.be/shorts/embed/live/bare id)
│   ├── sources.py      protocols + DTOs: TranscriptSource, MetadataSource,
│   │                   CaptionSnippet, TrackInfo, CaptionTrack
│   ├── clients.py      YouTubeCaptionSource (youtube-transcript-api 1.x),
│   │                   HttpMetadataSource (oEmbed + watch-page player response)
│   ├── normalize.py    named, recorded normalization rules
│   └── service.py      TranscriptIngestor: track selection, segment building,
│                       provenance, typed error propagation
├── pipeline/
│   ├── runner.py       StageRunner: cache reuse, atomic writes, stage state
│   └── ingest_stages.py the three implemented stages + `until` selection
└── cli/
    ├── builders.py     composition root: config -> concrete sources
    └── main.py         analyze --until / -l / --force wired to the stages
```

Configuration gained `ingest:` (languages, manual preference, watch-page metadata,
proxy, timeout) and `VideoMetadata` gained `source_detail`.

Dependency added: `youtube-transcript-api>=1.0` (installed 1.2.4). The 1.x interface was
inspected before use: `YouTubeTranscriptApi().list(video_id)` returns a `TranscriptList`
whose `_manually_created_transcripts` / `_generated_transcripts` are **dicts keyed by
language code**, and `.fetch(video_id, languages=[...])` returns a `FetchedTranscript`
of snippets with `text` / `start` / `duration` plus `language`, `language_code`,
`is_generated`. A first live run exposed exactly that dict-vs-list mismatch and the
adapter was corrected; the bug would not have been caught without the live check.

## 2. Requirements mapped to implementation

| Spec | Status | Evidence |
| --- | --- | --- |
| §4 Accept and validate a YouTube URL | done | `ingest/youtube.py`; 13 accepted forms, 11 rejected forms tested |
| §4 Retrieve metadata incl. title and duration | done | oEmbed (official, keyless) + watch-page `ytInitialPlayerResponse` for `lengthSeconds`, `uploadDate`, caption track list; live run returned title, channel and `1120.0`s duration |
| §4 Obtain captions, preserve segment timestamps | done | 286 real segments ingested; timings copied from cue `start`/`duration`, never recomputed |
| §4 Detect transcript language | done | `language`/`language_code` recorded; language priority from `--language` or config |
| §4 Report manual vs automatic | done | `TranscriptKind.MANUAL` / `AUTOMATIC` from `is_generated`; live run reported `kind=manual` |
| §4 Save the transcript locally | done | `raw_transcript.json` + `normalized_transcript.json`, atomic writes |
| §4 Handle missing captions / unavailable videos / failures | done | five typed errors, each with an actionable hint; tested for all paths |
| §4 Offer an optional speech-to-text path | reported, not built | the error names `transcription.enabled` and states the requirement for authorisation; the backend is Phase 9 work and stays disabled |
| §4 Do not fabricate or substitute | done | no track → `TranscriptUnavailable`; an all-blank track → error; no fallback source exists anywhere in the code |
| §3 Cleaning must not change meaning | done | rules touch only unicode form, control characters, whitespace and wordless cues; `raw_text` keeps the original per segment |
| §3 Never delete repeated passages | done | a fixture with two identical repeated explanations keeps all four segments after normalization; blank cues are counted in provenance (`empty_cues_dropped`), never silently dropped |
| §5 Timestamps come from source, not models | done | segment ids and times are assigned at ingestion only; no later stage can alter them |
| §15 Cache stages, invalidate only what changed | done | `StageRecord.is_stale` on input hash + config hash; changing the language list refetches the transcript but reuses metadata; changing config re-runs only normalization |
| §19 Record failures, keep partial results | done | a failed stage writes `state=failed` with the error, leaves no artifact, and a later run recovers |
| §20 Tests for ingestion and normalization | done | 96 new tests; suite of 281 passes with no network or API keys |

## 3. CLI behaviour (verified live)

```text
video-intel analyze URL --dry-run                     -> 0  plan only, nothing fetched or written
video-intel analyze URL                               -> 3  ingests, then stops: Phase 3/5/6/7 pending
video-intel analyze URL --until metadata              -> 3  metadata only
video-intel analyze URL --until chunks                -> 2  stage not implemented (named, not guessed)
video-intel analyze URL -l de -l en                   -> 3  selects the German track
video-intel analyze URL --force                       -> 3  recomputes, ignoring the cache
video-intel analyze URL                               -> 3  second run: all three stages reused, no network
video-intel analyze "https://example.com/x"           -> 1  invalid URL
video-intel analyze URL  (captions disabled)          -> 1  "no usable transcript" + speech-to-text hint
video-intel inspect VIDEO_ID                          -> 0  artifacts and stage states
```

Live run against `youtube.com/watch?v=aircAruvnKk` (3Blue1Brown, "But what is a neural
network?"): title, channel, `1120.0` s duration, 286 segments, `language=en`,
`kind=manual`, normalization reported `collapse_whitespace=2`, stages recorded with input
and output hashes; the second run reused all three stages with no HTTP requests.

## 4. Quality gates

```text
.venv/bin/python -m pytest -q     -> 281 passed (95% statement coverage)
.venv/bin/ruff check .            -> All checks passed
.venv/bin/ruff format --check .   -> 55 files already formatted
.venv/bin/mypy                    -> Success: no issues found in 36 source files (strict)
```

No test performs a network request; `httpx.MockTransport` covers the metadata client and
a stub covers the caption library.

## 5. Design decisions

1. **Sources are protocols.** `TranscriptIngestor` depends on "something that provides
   captions", not on a scraping library, so the whole pipeline runs offline in tests.
2. **Typed errors instead of exception pass-through.** Library exceptions are translated
   once (translation is idempotent), so the CLI reports causes in application language.
3. **Empty cues are counted, not dropped silently.** A segment must contain text, so
   wordless cues cannot be segments; the count lands in `source_detail` as
   `empty_cues_dropped` next to `cue_count`.
4. **`raw_text` is stamped during normalization**, so every cleaned segment still shows
   exactly what the source said.
5. **Manual captions beat automatic ones at equal language; language priority beats
   both.** Configured in `IngestSettings.prefer_manual_tracks`.
6. **Metadata degrades, it does not guess.** If the watch page cannot be parsed, title and
   channel from oEmbed are kept and duration/upload date stay `None` with an explanation
   in `source_detail`; a live run showed `duration (not reported by the source)` in this
   state rather than an invented number.
7. **The stage runner owns caching for every future phase.** Output stages in Phase 7 will
   reuse the same mechanism, which is what keeps "changing an output format does not
   trigger AI calls" true by construction.
8. **`--until` can only name implemented stages**, and the error lists what exists, so
   the CLI cannot imply a stage it does not have.
9. **The CLI entry point was renamed `main` → `cli`.** `video_intel.cli.main` the module
   was being shadowed by `main` the function, which broke test-time injection at the
   composition root.

## 6. Bugs found and fixed during this phase

| Found by | Bug | Fix |
| --- | --- | --- |
| Live run | `TranscriptList` groups are dicts, so track iteration produced empty language codes and every fetch reported "no track available" | iterate `.values()` |
| Test suite | `RedactionFilter` stringified every logging argument, breaking `%d` in stage logging | only strings are scrubbed; regression test added |
| Test suite | `--until` was accepted by the CLI but ignored by the stages | `ingest_stages(until=...)` runs the implemented prefix |
| Test suite | an early manual run of the metadata stage used a different input hash and was wrongly considered stale | test now uses the same `payload_hash(video_id)` identity |
| Test suite | the normalizer cleaned text without recording the original on the segment | `raw_text` stamped before rules run |
| CLI review | `analyze` let an `InvalidVideoUrl` escape as a traceback | caught and rendered like every other ingest error |

## 7. Known limitations

- Caption retrieval depends on YouTube's internals via `youtube-transcript-api`; if YouTube
  changes its player response, transcripts will fail with `TranscriptRetrievalFailed` and a
  retry hint. No private endpoint is automated and no workaround is faked.
- No speech-to-text fallback exists. The configuration keys and the error message are in
  place; the backend is not written.
- Caption retrieval is synchronous (the library is synchronous). Extraction in Phase 5
  will be async over httpx; ingestion will be called from a worker when concurrency exists.
- Metadata for age-restricted or region-blocked videos fails with `VideoUnavailable`;
  there is no authenticated path (no OAuth support).
- The watch-page parser is a best-effort HTML scrape for duration and upload date. When it
  fails, those fields stay `None`.
- `--until` currently stops within Phase 2 only; later phases append their stages to the
  same ordered list.
- Resume exists per stage, not per pipeline run: an interrupted extraction phase will
  restart that stage (Phase 9 adds finer granularity).

## 8. Next concrete step — Phase 3 (chunking)

1. Add `video_intel/pipeline/chunking.py` with `chunk_transcript(segments, config)`:
   sentence-boundary-aware packing to `max_chars`, `overlap_chars` carried across
   neighbours by re-including trailing segment ids, never splitting a segment.
2. Every chunk keeps `segment_ids`, `start_seconds`, `end_seconds`, `char_count` and a
   deterministic `chunk_0001…` id derived from position only.
3. Add the `chunks` stage to the runner using `ChunkingConfig` plus the normalized
   transcript hash as its input hash, so changing chunk size re-chunks but never re-fetches.
4. Assert determinism in tests: same transcript + same config produces byte-identical
   chunks, and every segment in the transcript appears in at least one chunk.
5. Assert overlap explicitly: consecutive chunks share the configured overlap of text and
   segment ids, and no chunk exceeds `max_chars`.
6. Wire `StageId.CHUNKS` into `IMPLEMENTED_STAGES` so `--until chunks` becomes valid, and
   print a chunk count plus estimated tokens in the run summary.

Nothing has been committed; the working tree is left for review.
