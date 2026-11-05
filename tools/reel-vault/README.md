# reel-vault

Official Instagram export → a classified, archived, searchable reel library.

Part of the [VOID OSINT extractor](../../README.md) workspace.
Full specification: [`plan.md`](plan.md).

## Status

Under active construction. The tool is built phase by phase; see `plan.md` §20.

## What it does

1. Imports `saved_posts.json`, `saved_posts.html` and `saved_collections.json`
   from an official Instagram data export.
2. Fetches public metadata for each saved post (no login, no account risk).
3. Transcribes audio, reads on-screen text, and lets an AI decide what is worth
   extracting.
4. Sorts every post into an editable, multi-label category taxonomy.
5. Supports a terminal review queue and exports Markdown, JSON, HTML, PDF and an
   Obsidian vault.
6. Archives media to an external drive, unattended and resumable.

## Commands

```bash
python run.py reel-vault --help
```

```bash
reel-vault import ~/Downloads/instagram-export.zip
reel-vault run
reel-vault review
reel-vault export
reel-vault doctor
```

## Development

```bash
python run.py test
python run.py lint
```
