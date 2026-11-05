# reel-vault — OSINT Archive

Official Instagram export → classified, archived reel library. Everything runs
offline on your machine; no login, password, or private API.

## Install from local files

```bash
# From the workspace root:
./install.sh

# Activate the venv, then run:
source .venv/bin/activate
OSINT
```

The `OSINT` command launches the full-screen TUI. After installation you can also
run it directly: `.venv/bin/OSINT`.

## TUI — the ``OSINT`` app

A single Textual interface covers every pipeline stage:

| Key | Screen | What it does |
|-----|--------|--------------|
| `1` | Import | Import an Instagram export ZIP or directory |
| `2` | Run | Run fetch → transcribe → triage → extract → classify → media |
| `3` | Review | Approve, reclassify, force decisions, add notes, open URLs |
| `4` | Export | Generate Markdown, JSON, PDF, Obsidian outputs |
| `5` | Config | Show configuration values |
| `6` | Doctor | Check ffmpeg, Groq, yt-dlp, OCR availability |
| `q` | Quit | Exit the TUI |

## Outputs

| Output | Path |
|--------|------|
| Per-reel Markdown | `data/reels/<sc>/outputs/reel.md` |
| Category index | `data/reels/outputs/category-index.md` |
| Stubs list | `data/reels/outputs/stubs.md` |
| Knowledge base (JSON) | `data/reels/outputs/knowledge.json` |
| PDF document | `data/reels/outputs/reel-vault.pdf` |
| Obsidian vault | `data/reels/outputs/obsidian/` |

No DOCX, no CSV.

## CLI

The `reel-vault` command is also available for scripting:

```bash
reel-vault import ~/Downloads/instagram-export.zip
reel-vault run --unattended --max-hours 9
reel-vault run --dry-run
reel-vault export --formats md,json,pdf,obsidian
reel-vault doctor
reel-vault stats
```

Exit codes: 0=done, 1=error, 2=bad request, 3=provider unavailable, 4=drive lost,
5=limit reached.

## Requirements

Python 3.11+, ffmpeg, ffprobe, Groq API key (`GROQ_API_KEY` env var).
