# reel-vault

Official Instagram export → classified, archived reel library.

`reel-vault` imports an official Instagram "Download your information" export and then
automatically fetches, transcribes, triages, extracts, classifies and archives your saved
reels and posts. Everything after import runs offline on your machine — no login, no
password, no session cookies, no private API.

## Quick start

```bash
# 1. Install
cd tools/reel-vault
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Import your Instagram export (ZIP or unpacked directory)
reel-vault import ~/Downloads/instagram-export.zip

# 3. Run the full pipeline
reel-vault run

# 4. Review and correct in the interactive TUI
reel-vault review

# 5. Export documents
reel-vault export --formats md,json,html,pdf,obsidian
```

## Commands

| Command | Purpose |
|---------|---------|
| `reel-vault import <path>` | Import an official Instagram export (ZIP or directory) |
| `reel-vault run` | Run the full pipeline: fetch → transcribe → triage → extract → classify → media |
| `reel-vault run --unattended --max-hours 9` | Overnight mode with time/cost limits and drive-safety |
| `reel-vault run --dry-run` | Show the plan without calling any model |
| `reel-vault fetch` | Fetch public metadata and thumbnails anonymously via yt-dlp |
| `reel-vault download` | Download and verify media (respects `--mode archive|all|none`) |
| `reel-vault transcribe` | Extract audio and transcribe with Groq Whisper |
| `reel-vault triage` | Decide what is worth extracting, archiving or stubbing |
| `reel-vault triage --dry-run` | Heuristic plan without model calls |
| `reel-vault extract` | Grounded field extraction, link resolution, keyframes, OCR |
| `reel-vault classify` | Taxonomy classification with rules + LLM |
| `reel-vault classify <sc> --primary "X" --add "Y"` | Manual category override |
| `reel-vault review` | Interactive review TUI (approve, reclassify, force, note, open) |
| `reel-vault list` | List reels with filters |
| `reel-vault show <sc>` | Show everything stored for one reel |
| `reel-vault stats` | Counts, categories and recorded cost |
| `reel-vault export` | Generate Markdown, JSON, HTML, PDF and Obsidian outputs |
| `reel-vault config show\|validate\|path` | Inspect configuration |
| `reel-vault doctor` | Check ffmpeg, Groq, OCR, yt-dlp, drive |

## Configuration

Configuration lives in `reel-vault.yaml` at the workspace root. Precedence (lowest to
highest): packaged defaults → user config → workspace `reel-vault.yaml` →
`REEL_VAULT_*` environment variables → CLI flags.

Key environment variables:

- `REEL_VAULT_DATA_DIR` — data directory (default: `./data/reels`)
- `REEL_VAULT_DB_PATH` — SQLite database path (default: `./data/reels/state.db`)
- `REEL_VAULT_MEDIA_DIR` — media archive directory (can point to an external HDD)
- `REEL_VAULT_GROQ_MODEL` — chat model for triage/extraction/classification
- `REEL_VAULT_LOG_LEVEL` — logging level
- `GROQ_API_KEY` — Groq API key (required for triage, extraction, classification, transcription)

## Outputs

| Output | Description |
|--------|-------------|
| `reels/<sc>/outputs/reel.md` | Per-reel Markdown with frontmatter, summary, items, links, transcript |
| `outputs/category-index.md` | Reels grouped by primary category |
| `outputs/stubs.md` | Low-value / unavailable / failed reels with reasons |
| `outputs/knowledge.json` | Versioned knowledge base with provenance for every extracted item |
| `outputs/dashboard.html` | Self-contained, client-side searchable/filterable HTML dashboard |
| `outputs/reel-vault.pdf` | Cover + TOC + one section per reel (ReportLab) |
| `outputs/obsidian/` | Obsidian vault: folder per category, YAML frontmatter, wikilinks |

No DOCX, no CSV.

## Pipeline status flow

```
new → fetched → transcribed → triaged → extracted → classified → done
              ↳ stub (low value)  ↳ unavailable (deleted/private)
```

Triage decisions:
- **extract** — reusable knowledge worth structured notes
- **archive** — emotional/personal post worth keeping whole
- **stub** — low value, metadata only

## Overnight unattended mode

```bash
reel-vault run --unattended \
  --data-dir /media/void/500GB/reel-vault \
  --media-dir /media/void/500GB/reel-vault/media \
  --download all \
  --max-hours 9
```

- No prompts and no TUI
- Checkpoint after every reel (SQLite + atomic files)
- Detects drive loss and stops with exit code 4
- Writes `runs/<timestamp>/report.md` and `report.json`
- Resume is automatic: `reel-vault run --resume` picks up where it stopped

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | done |
| 1 | import or configuration error |
| 2 | bad request |
| 3 | provider unavailable, partial work kept |
| 4 | external drive disappeared; resume later |
| 5 | cost or time limit reached; resume later |

## Development

```bash
# Run all tests
python run.py test

# Run lint (ruff + mypy)
python run.py lint

# Run just reel-vault tests
cd tools/reel-vault
.venv/bin/python -m pytest tests
```

## Architecture

- **Import**: HTML + JSON export parsing, ZIP/directory input, collections merge
- **Enrich**: Anonymous yt-dlp metadata + thumbnail fetching, per-reel statuses
- **Media**: yt-dlp download, temp-dir staging, ffprobe/checksum verification, atomic move
- **Transcribe**: ffmpeg mono 16 kHz audio, Groq Whisper `verbose_json`, chunking, retries
- **Triage**: Rule guardrails (emotional/archive, tutorial/news), LLM scoring, fallback
- **Extract**: Grounded field extraction with citations, link resolution, keyframes, RapidOCR
- **Classify**: YAML taxonomy, rules prefilter + LLM labeling, confidence thresholds
- **Review**: Textual TUI for manual correction, writes to `review.json`
- **Outputs**: Markdown, JSON knowledge base, HTML dashboard, PDF, Obsidian vault

## Privacy

All processing runs locally on your machine. Media and transcripts never leave your
device. The only external calls are to Groq's API (for chat and transcription) and
anonymous yt-dlp requests to Instagram's public pages (no authentication).