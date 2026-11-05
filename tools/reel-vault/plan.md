# reel-vault — implementation plan

Status: planned, not implemented
Date: 2026-11-05
Workspace: VOID OSINT extractor
Tool: `reel-vault` (package `reel_vault`)

A tool that turns an **official Instagram data export** into a classified, searchable,
permanently archived library of saved reels and posts. No login, no password, no
account-ban risk.

---

## 1. Goal

Given an Instagram export ZIP:

1. Read every saved post and reel, plus collection names.
2. Fetch each post's public metadata, caption, audio and media (anonymous, no login).
3. Transcribe audio, read on-screen text, and let an AI decide what is worth keeping.
4. Sort everything into an editable, multi-label category taxonomy.
5. Let the user review and correct decisions in a terminal UI.
6. Produce Markdown, JSON, an HTML dashboard, PDFs and an Obsidian vault.
7. Archive media to an external drive, unattended, overnight, resumable.

## 2. Non-goals

- No Instagram login, session cookies, password storage, or private API.
- No likes, comments, follows, DMs, or any write action on Instagram.
- No republishing or redistribution of saved content.
- No DOCX or CSV output.
- No web UI. Review happens in a terminal UI.

---

## 3. Locked decisions

| Decision | Choice |
|---|---|
| Access | Official "Download your information" export only (ZIP) |
| Export formats | Both HTML and JSON (HTML has captions, JSON has structure/collections) |
| Scope | All saved posts: reels, images, carousels |
| Enrichment | Anonymous public fetch via `yt-dlp` (no cookies, no login) |
| Transcription | Groq Whisper API |
| Triage | AI reads each post, scores knowledge density, chooses what to extract |
| Rule guardrails | Emotional/personal content → archive whole post; tutorials/news → force key fields |
| OCR | Local RapidOCR (ONNX), nothing uploaded |
| Classification | Rules + Groq, editable YAML taxonomy, multi-label + confidence |
| Low-value posts | Stub + reason (metadata kept, no document, no media) |
| Documents | PDF only (plus Markdown, JSON, HTML dashboard, Obsidian) |
| Interface | CLI + interactive review TUI (Textual) |
| Library size | 200–1,000 saves → resumable queue, rate limits, cost logging |
| Storage | External drive supported; no storage cap by default |
| Long runs | Unattended overnight mode, checkpointed and resumable |

---

## 4. Repository layout

```text
tools/reel-vault/
  pyproject.toml
  plan.md                     this file
  README.md
  src/reel_vault/
    config/         settings, taxonomy loader, archive rules, env
    importer/       ZIP, saved_posts.json, saved_posts.html, collections, dedupe
    enrich/         yt-dlp metadata, thumbnails, media download, link resolution
    transcribe/     ffmpeg audio extraction, Groq Whisper, timestamps
    triage/         rubric, rule guardrails, AI decision, fallback
    extract/        field extraction, OCR, keyframes, grounding
    classify/       rules + LLM, confidence, manual overrides
    storage/        SQLite state, paths, atomic writes, locks, cache
    review/         Textual TUI
    outputs/        Markdown, JSON, HTML, PDF, Obsidian
    runs/           unattended runner, reports, resume
    cli/            typer commands
  tests/
    fixtures/       synthetic export ZIP, fake metadata, fake provider replies
```

The tool is independent of `video-intel`. Both share `.venv` and the `data/` tree, but
never write into each other's directories.

---

## 5. Pipeline

```text
export ZIP
  │
  ├─ 1. import      parse + merge HTML/JSON, dedupe by shortcode,
  │                 map collections, record source import
  │
  ├─ 2. fetch       yt-dlp anonymous: caption, author, duration, thumbnail,
  │                 media URLs, carousel items. Per-reel status.
  │
  ├─ 3. download    media to the external drive (mode-dependent)
  │
  ├─ 4. verify      ffprobe/file checks, non-zero size, checksum
  │
  ├─ 5. transcribe  ffmpeg → audio → Groq Whisper (timed segments)
  │                 split if >25 MB; skip if no audio
  │
  ├─ 6. triage      LLM reads caption + transcript + hashtags + collections
  │                 → decision, value score, content type, extraction plan,
  │                   category hints, confidence
  │
  ├─ 7. execute     only planned actions:
  │                 links, steps/tricks, tools, claims, quotes, recipes,
  │                 people/books/places, code, keyframes, OCR, screenshots
  │
  ├─ 8. ground      every item cites a caption or transcript timestamp;
  │                 uncited items are dropped, never invented
  │
  ├─ 9. classify    rules + LLM, multi-label, confidence, needs_review
  │
  ├─ 10. review     TUI: approve, reclassify, force archive/extract, notes
  │
  └─ 11. export     Markdown, category indexes, knowledge.json,
                    dashboard.html, PDFs, Obsidian vault
```

Everything after import is resumable. A crash mid-reel restarts only that reel.

---

## 6. Import

### Accepted input

- The export ZIP itself, or an unpacked directory.
- Files searched by name, not by fixed path, because Meta moves folders between
  export versions:
  - `saved_posts.json` (wrapper `saved_saved_media`)
  - `saved_posts.html`
  - `saved_collections.json` (wrapper `saved_saved_collections`)
- JSON is used for structure, timestamps and collections.
- HTML is parsed for captions, hashtags and any entry the JSON lacks.
- Entries are merged by shortcode extracted from `/p/`, `/reel/` and `/tv/` URLs.
- Mojibake in account names is repaired with `ftfy`.
- BOMs, CRLF, nested folders and duplicate URLs are tolerated.

### Inventory record

```json
{
  "shortcode": "ABC123xyz",
  "url": "https://www.instagram.com/reel/ABC123xyz/",
  "media_type": "reel | post | carousel | image | unknown",
  "author": "creator_username",
  "saved_at": "2026-09-10T12:00:00Z",
  "collections": ["Psychology"],
  "caption": "text from HTML export, if present",
  "hashtags": ["#example"],
  "source_import": "2026-11-05T21-00-00",
  "status": "new"
}
```

### Re-import

- Merge by shortcode; never duplicate.
- New entries are added as `new`.
- Existing entries keep their transcript, triage, extraction, review and media.
- Removed saves are **not** deleted from the library; they are marked
  `missing_from_export` so the archive survives.
- A parse report records counts, unknown shapes and skipped entries.

---

## 7. Enrichment

`yt-dlp` is used anonymously. No cookies, no account, no login.

- Fetch metadata without downloading first.
- Record `caption`, `author`, `duration`, `upload_date`, `thumbnail`, media URLs,
  carousel entries, view/like counts when available.
- Respect a configurable request delay (default 3 s) and exponential backoff.
- Per-reel statuses: `ok`, `private`, `deleted`, `blocked`, `rate_limited`,
  `unsupported`, `error`.
- A failure marks one reel; it never aborts the run.
- Link resolution: fetch final URL, title and HTTP status with a short timeout.
  No page scraping beyond meta tags.

---

## 8. Media download and external storage

Storage is not capped. Media can go to a 500 GB external HDD.

### Two roots

```text
data_dir    JSON artifacts, outputs, run reports, media by default
db_path     SQLite state database (kept local by default)
media_dir   large media files (defaults to <data_dir>/media)
```

Recommended for a USB drive:

```yaml
data_dir: /media/void/500GB/reel-vault
db_path: ./data/reels/state.db
media_dir: /media/void/500GB/reel-vault/media
```

Keeping `state.db` on local disk means an unplugged drive cannot corrupt the queue.
If the user wants everything on the drive, `db_path` can be pointed there too.

### Layout on the external drive

```text
<data_dir>/
  reels/<shortcode>/          JSON artifacts per reel
  media/<shortcode>/
    video.mp4
    audio.m4a
    thumb.jpg
    keyframes/*.jpg
  outputs/
    categories/*.md
    stubs.md
    knowledge.json
    dashboard.html
    pdf/*.pdf
    obsidian/
  runs/<timestamp>/
    report.md
    report.json
  logs/
```

### Download modes

| Mode | What is downloaded |
|---|---|
| `none` | No media; metadata, transcript and documents only |
| `archive` (default) | Full media only for rule-matched archive posts |
| `all` | Full media for every reel/post that can be fetched |

Flags: `--download none|archive|all`, `--media-dir PATH`, `--data-dir PATH`,
`--db-path PATH`, `--quality best|1080|720`.

### Verification

After every download:

- file exists and is non-zero;
- `ffprobe` can read video/audio;
- carousel image count matches the expected count;
- optional SHA-256 stored in the reel's `media.json`.

Failed verification retries once, then marks the reel `media_failed` and continues.

### Drive safety

- `doctor` checks that `media_dir` exists, is writable, and reports free space.
- Before each reel, the runner checks that the media root is still reachable.
- If the drive disappears: pause, retry for 60 s, then stop cleanly with exit code 4
  and a resume instruction. No partial file is left in place.
- All writes are temp-file + rename.

---

## 9. Transcription

- Extract mono 16 kHz audio with `ffmpeg`.
- Split into chunks under 25 MB (Groq limit).
- Call Groq `whisper-large-v3-turbo` with `verbose_json` for timestamps.
- Store:

```json
{
  "language": "en",
  "model": "whisper-large-v3-turbo",
  "segments": [{"start": 0.0, "end": 3.2, "text": "..."}],
  "text": "full transcript",
  "cost_usd": 0.001
}
```

- Retry on 429/5xx with exponential backoff.
- Skip silently if the media has no audio track.
- Log seconds of audio and estimated cost per reel.
- If Groq is unavailable, the reel continues with caption + OCR only and is marked
  `transcript_failed` for a later retry.

---

## 10. Triage

The model returns validated JSON:

```json
{
  "decision": "extract | archive | stub",
  "value_score": 78,
  "content_type": "tutorial | news | comedy | story | opinion | recipe | ad | other",
  "reason": "One or two sentences, shown to the user.",
  "actions": ["transcript", "ocr", "keyframes", "links", "download_media"],
  "fields": ["steps", "tools", "links", "claims"],
  "categories": [{"name": "learning & study tricks", "confidence": 0.91}],
  "archive_reason": null
}
```

### Decision order

1. **Unavailable** — deleted/private/blocked → `unavailable` stub.
2. **Rule guardrail: emotional/personal** — if the caption/transcript matches the
   archive rule set *and* the LLM confirms the theme → `archive`. Full media, all
   keyframes, caption, transcript, OCR; minimal knowledge extraction.
3. **Rule guardrail: tutorial/news** — force `extract` and force the fields that
   matter (steps for tutorials, links/claims for news).
4. **Otherwise AI score**:
   - `>= extract_threshold` (default 40) → `extract`
   - `< extract_threshold` → `stub`
5. **Manual override** always wins.

### Why rules run alongside the AI

Naive keywords produce false positives ("I love this trick" is not an emotional post).
Therefore:

- keywords are only a **trigger**, never the final decision;
- the LLM must confirm the theme;
- the keyword list is editable in `archive_rules.yaml`;
- every rule-forced decision records `archive_reason` for auditing;
- a rule can be disabled or narrowed without code changes.

### Score rubric

| Score | Meaning |
|---|---|
| 0–20 | pure entertainment, no reusable content |
| 21–40 | light opinion/story |
| 41–60 | one useful idea or link |
| 61–80 | several actionable items |
| 81–100 | dense reference: tutorial, framework, list, recipe, research |

### Fallback

Invalid JSON → one repair attempt → deterministic **Standard** plan
(transcript + links + 3 keyframes) and `needs_review`.

### Dry run

`reel-vault run --dry-run` prints the triage plan for each reel without downloading
media or calling extraction models.

---

## 11. Extraction and grounding

Only the fields and actions chosen by triage are executed.

Supported fields:

- `links` — URLs in caption, bio mentions and transcript; resolved title/status.
- `steps` / `tricks` — ordered actions from a tutorial.
- `tools` — apps, products, software, equipment.
- `claims` — factual assertions, with source links when present.
- `quotes` — short, attributed lines worth keeping.
- `recipes` — ingredients + steps.
- `people`, `books`, `places`, `products`.
- `code` — snippets shown on screen (OCR) or read aloud.
- `notes` — the model's short summary of why the reel matters.

Rules:

- Every item must cite a transcript segment ID or the caption.
- Uncited items are dropped before they reach the dataset.
- Extraction JSON uses `extra="forbid"`; one repair attempt; raw model output stored.
- Keyframes are selected by scene change with a fixed cap; OCR text is deduplicated
  across frames.
- RapidOCR runs locally; no image leaves the machine.

---

## 12. Classification

Editable file: `reel-vault-taxonomy.yaml`.

Starter categories:

`psychology & mind`, `learning & study tricks`, `news & information`,
`comedy & funny`, `tech & AI`, `money & business`, `health & fitness`,
`food & recipes`, `travel & places`, `style & design`, `life hacks & DIY`,
`tools & resources`, `inspiration & motivation`, `other`.

Each category has:

- `description`
- `keywords`
- `example_captions`
- optional `collections` mapping (for example `Psych: psychology & mind`)

Behaviour:

- one **primary** label plus any number of secondary labels;
- confidence per label; below `needs_review_threshold` (default 0.60) → `needs_review`;
- rules prefilter obvious cases, the LLM handles the rest;
- manual labels in `review.json` always override future runs;
- renaming/adding categories never requires code changes.

---

## 13. Review TUI

Textual app, fully offline.

- Queue filters: `needs_review`, `low_value`, `failed`, `archive`, `all`.
- Left pane: reel list with thumbnail, score, primary category, status.
- Right pane: caption, transcript, triage reason, extracted items, links.
- Keys:
  - `a` approve
  - `r` reclassify (multi-select)
  - `f` force archive/extract
  - `n` add note
  - `o` open URL in browser
  - `/` search
  - `q` quit
- Every action writes to `review.json`; no model call is required to correct a label.
- Bulk actions: approve all above a score, reclassify a filtered set.

---

## 14. CLI

```bash
reel-vault import ~/Downloads/instagram-export.zip
reel-vault run                                   # process everything new, adaptive
reel-vault run --limit 50 --dry-run              # show the plan, call no models
reel-vault run --download all --media-dir /media/void/500GB/reel-vault/media
reel-vault run --unattended --max-hours 9        # overnight mode
reel-vault run --resume                          # continue where it stopped
reel-vault run --depth full --force-extract
reel-vault list --category psychology --decision extract
reel-vault list --status needs_review --search "sleep"
reel-vault show <shortcode>
reel-vault review                                # TUI
reel-vault classify <shortcode> --primary "psychology & mind" --add "health & fitness"
reel-vault export --formats md,json,html,pdf,obsidian
reel-vault export --out /media/void/500GB/reel-vault
reel-vault config show|validate|path
reel-vault stats                                 # counts, costs, failures, disk use
reel-vault doctor                                # ffmpeg, Groq, OCR, yt-dlp, drive
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | import or configuration error |
| 2 | bad request |
| 3 | provider unavailable, partial work kept |
| 4 | external drive disappeared; resume later |
| 5 | cost or time limit reached; resume later |

---

## 15. Unattended overnight runs

```bash
reel-vault run --unattended \
  --data-dir /media/void/500GB/reel-vault \
  --media-dir /media/void/500GB/reel-vault/media \
  --download all \
  --max-hours 9
```

Behaviour:

- no prompts and no TUI;
- processes the queue until empty, time limit, or cost limit;
- checkpoint after every reel (SQLite + atomic files);
- retries transient failures with backoff, records permanent failures, continues;
- detects drive loss and stops with exit code 4;
- writes `runs/<timestamp>/report.md` and `report.json`:
  - processed / skipped / failed counts;
  - bytes downloaded;
  - audio minutes and estimated cost;
  - per-category counts;
  - failures with reasons and resume command;
- writes a single progress line to the log, so an overnight terminal stays readable;
- `--stop-at 07:00` alternative to `--max-hours`;
- `--nice` lowers process priority;
- `doctor` warns if the machine is on battery or if sleep is likely to interrupt the run.

Resume is automatic: `run` picks up every reel not in a terminal state. `--resume` is an
explicit alias. Reels marked `rate_limited` are retried after a configurable cooldown.

---

## 16. Outputs

| Output | Detail |
|---|---|
| Per-reel Markdown | Frontmatter (url, author, saved date, collections, categories, score, decision, tags), summary, key items, links, transcript excerpt, screenshots |
| Category index | Grouped by primary category, sorted by score/date, table of contents, stub appendix |
| `stubs.md` | Every low-value, unavailable or failed reel with a one-line reason |
| `knowledge.json` | Versioned schema with provenance for every item |
| `dashboard.html` | Self-contained, searchable/filterable, thumbnails, chips, status, links — no server |
| PDF | Cover + TOC + one section per reel, thumbnails, extracted items, links, screenshots. Per-category by default, `--per-reel` optional |
| Obsidian vault | Folder per category, YAML frontmatter, wikilinks for people/tools/tags, attachments |

No DOCX, no CSV.

PDF is generated with ReportLab. Images are downscaled to keep PDFs reasonable.

---

## 17. Configuration

`reel-vault.yaml` at the workspace root. Precedence:
packaged defaults < user config < workspace `reel-vault.yaml` < `REEL_VAULT_*` env < CLI flags.

```yaml
data_dir: ./data/reels
db_path: ./data/reels/state.db
media_dir: ./data/reels/media

download:
  mode: archive          # none | archive | all
  quality: best
  keep_audio: true
  keep_keyframes: true
  verify: true

run:
  unattended: false
  max_hours: 0
  max_cost_usd: 5
  concurrency: 1
  request_delay_seconds: 3
  rate_limit_cooldown_minutes: 30

groq:
  api_key_env: GROQ_API_KEY
  triage_model: openai/gpt-oss-120b
  extraction_model: openai/gpt-oss-120b
  whisper_model: whisper-large-v3-turbo

triage:
  extract_threshold: 40
  needs_review_threshold: 60
  fallback: standard

transcription:
  enabled: true
  language: null          # null = auto-detect

ocr:
  enabled: true
  max_keyframes: 6

taxonomy_file: ./reel-vault-taxonomy.yaml
archive_rules_file: ./reel-vault-archive-rules.yaml
```

`reel-vault-archive-rules.yaml`:

```yaml
confirm_with_llm: true
archive:
  keywords:
    - sad
    - emo
    - depressed
    - depression
    - cuddle
    - cuddling
    - love
    - heartbreak
    - loneliness
    - comfort
    - nostalgia
  categories:
    - psychology & mind
extract:
  tutorial_keywords: [how to, step, trick, hack, tutorial, guide, recipe]
  news_keywords: [breaking, report, study, research, source]
```

Environment variables: `REEL_VAULT_DATA_DIR`, `REEL_VAULT_DB_PATH`,
`REEL_VAULT_MEDIA_DIR`, `REEL_VAULT_GROQ_MODEL`, `REEL_VAULT_LOG_LEVEL`,
`GROQ_API_KEY`.

---

## 18. Data model

SQLite tables:

| Table | Purpose |
|---|---|
| `reels` | shortcode, url, type, author, saved_at, collections, status |
| `imports` | timestamp, source path, counts, parse report |
| `jobs` | stage, state, attempts, error, started/finished |
| `metadata` | caption, hashtags, duration, thumbnail, fetch status |
| `media` | paths, bytes, checksum, verified |
| `transcripts` | language, model, segments, cost |
| `triage` | decision, score, content type, reason, actions, fields, model |
| `extractions` | grounded items + citations |
| `classifications` | primary, secondary, confidence, method |
| `reviews` | manual overrides, notes, approved/ignored |
| `costs` | provider, model, tokens/seconds, usd |
| `artifacts` | output paths and content hashes |

Caching:

- key = `shortcode + stage + content_hash + model + prompt_version`;
- changing an output format re-renders without any model call;
- changing a model invalidates only that stage.

---

## 19. Dependencies and system requirements

Python:

- `pydantic`, `pydantic-settings`, `PyYAML`, `httpx`, `typer`, `rich`
- `beautifulsoup4`, `ftfy`
- `yt-dlp`
- `reportlab`
- `rapidocr-onnxruntime` (or its successor `rapidocr`)
- `textual`

System:

- `ffmpeg` and `ffprobe` (audio extraction, keyframes, verification)
- `doctor` checks them and prints the platform-specific install command

Cloud:

- `GROQ_API_KEY` (already present in `.env`)
- triage/extraction: `openai/gpt-oss-120b` (configurable)
- audio: `whisper-large-v3-turbo` (configurable)

Costs are logged per reel and shown by `stats`. `--max-cost` stops the run cleanly.

---

## 20. Build phases

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0 | Export parser spike | Parses a real/synthetic ZIP; HTML captions + JSON links + collections produce one inventory; shapes documented |
| 1 | Skeleton | `reel-vault doctor` and `config validate` work; storage, logging, tests |
| 2 | Import | Idempotent re-import, dedupe, collections, merge report |
| 3 | Fetch | yt-dlp metadata + thumbnails; per-reel statuses; rate limiting |
| 4 | Media + external drive | `--media-dir`, `--download all/archive/none`, verification, drive-loss handling |
| 5 | Transcribe | ffmpeg + Groq Whisper, timed segments, chunking, retries, cost log |
| 6 | Triage | Schema, rubric, rule guardrails, fallback, `--dry-run` |
| 7 | Extract | Grounded fields, link resolution, keyframes, RapidOCR, archive mode |
| 8 | Classify | YAML taxonomy, rules + LLM, confidence, overrides |
| 9 | Review TUI | All queue filters and actions work offline |
| 10 | Overnight runner | Unattended mode, checkpoints, resume, run report, exit codes |
| 11 | Outputs | All six outputs generated from one dataset; PDF and Obsidian validated |
| 12 | Hardening | Unit + integration tests, README, `run.py test`, `run.py lint` clean |

Each phase ships with tests and does not start the next until its acceptance criteria pass.

---

## 21. Testing

- Synthetic export ZIP fixture with JSON + HTML + collections.
- Fake yt-dlp metadata and fake Groq replies; no network in unit tests.
- Socket guard, same pattern as `video-intel`.
- Property tests for shortcode extraction, dedupe, merge, grounding, taxonomy.
- Integration tests (marked `integration`) requiring `ffmpeg` and a Groq key.
- Resume test: kill mid-run, restart, assert no duplicate work.
- Drive-loss test: unmount/rename media dir, assert exit code 4 and clean state.
- PDF and Obsidian output validated by parsing the generated files.

---

## 22. Risks and mitigations

| Risk | Mitigation |
|---|---|
| JSON export has no captions | Import both HTML + JSON and merge; documented in README |
| HTML export shape changes | Fixture-tested parser; clear parse report; never silent data loss |
| Instagram blocks anonymous fetches | Backoff, per-reel `unavailable`, retry later; run never aborts |
| yt-dlp breaks over time | Pin a version, `doctor` checks it, clear upgrade path |
| Groq rate limits / cost | Concurrency 1–2, retries, per-run cap, `stats` |
| AI triage is wrong | Score + reason in TUI, one-key override, rule audit trail |
| "Love" false positives | Rules only trigger; LLM must confirm the theme |
| USB drive unplugged overnight | Reachability check, atomic writes, exit 4, resume |
| Power loss / sleep | Checkpoint per reel, resume from queue, `doctor` warning |
| OCR dependency is heavy | Optional at runtime; `doctor` reports it; pipeline degrades to transcript |
| Copyright | Personal archive only; no republishing or redistribution tooling |

---

## 23. Open items to confirm before Phase 0

1. **Sample export** — a redacted ZIP, or just `saved_posts.html`, `saved_posts.json` and
   `saved_collections.json`, so the parser is built against the real shape.
2. **Category list** — confirm the 14 starter categories, or rename/add/remove.
   "Learning & study tricks" is the reading of "leaning tricks".
3. **Archive keywords** — confirm the starter list (sad, emo, depressed, cuddle,
   love, heartbreak, loneliness, comfort, nostalgia) and whether the rule should apply
   to all such posts or only clearly personal ones (the LLM confirmation step covers
   the difference).
4. **Drive path** — the actual mount point for the 500 GB drive, so `doctor` and the
   default config can point at it.
5. **Groq model availability** — confirm `openai/gpt-oss-120b` and
   `whisper-large-v3-turbo` are enabled for the key in `.env`.
