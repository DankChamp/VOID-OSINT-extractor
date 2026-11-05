"""Parse the official Instagram "Download your information" export.

The export is a ZIP (or an unpacked copy) containing ``saved_posts.json``,
``saved_posts.html`` and ``saved_collections.json``. Meta moves folders between export
versions, so files are located by basename rather than by a fixed path.

* JSON supplies the permalink, the author and the save timestamp.
* HTML supplies captions and hashtags for most entries; the JSON variant omits them.
* Collections are merged into every reel they contain.

Nothing is guessed silently: entries without a usable Instagram permalink are reported
as warnings, not invented.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import ftfy
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from reel_vault.schemas.common import MediaType, iso_now
from reel_vault.schemas.source import SourceReel

SAVED_POSTS_JSON = "saved_posts.json"
SAVED_POSTS_HTML = "saved_posts.html"
SAVED_COLLECTIONS_JSON = "saved_collections.json"

MAX_MEMBER_BYTES = 50 * 1024 * 1024

_SHORTCODE_RE = re.compile(r"instagram\.com/(?:[^/]+/)?(p|reel|tv)/([A-Za-z0-9_-]+)", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
_POST_BY_RE = re.compile(r"(?i)\bpost by\s+@?([A-Za-z0-9._]+)")
_TIMESTAMP_KEYS = ("saved on", "saved_on", "saved")


class ImportReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    imported_at: str = Field(default_factory=iso_now)
    reels: list[SourceReel] = Field(default_factory=list)
    collections: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "reels": sum(1 for reel in self.reels if reel.media_type == MediaType.REEL),
            "posts": sum(1 for reel in self.reels if reel.media_type != MediaType.REEL),
            "collections": len(self.collections),
            "warnings": len(self.warnings),
        }


def shortcode_from_url(url: str) -> str | None:
    match = _SHORTCODE_RE.search(url)
    return match.group(2) if match else None


def media_type_from_url(url: str) -> MediaType:
    match = _SHORTCODE_RE.search(url)
    if not match:
        return MediaType.UNKNOWN
    kind = match.group(1).lower()
    if kind == "reel":
        return MediaType.REEL
    return MediaType.POST


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    repaired = ftfy.fix_text(value).strip()
    return repaired or None


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _read_export_files(path: Path) -> tuple[dict[str, bytes], list[str]]:
    wanted = {SAVED_POSTS_JSON, SAVED_POSTS_HTML, SAVED_COLLECTIONS_JSON}
    found: dict[str, bytes] = {}
    warnings: list[str] = []

    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.name in wanted and candidate.name not in found:
                try:
                    if candidate.stat().st_size <= MAX_MEMBER_BYTES:
                        found[candidate.name] = candidate.read_bytes()
                except OSError as exc:
                    warnings.append(f"cannot read {candidate}: {exc}")
        return found, warnings

    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename).name
                if name not in wanted or name in found:
                    continue
                if member.file_size > MAX_MEMBER_BYTES:
                    warnings.append(f"skipped oversized export member {member.filename}")
                    continue
                found[name] = archive.read(member)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot read export {path}: {exc}") from exc
    return found, warnings


def _json_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("saved_saved_media", "saved_media", "saved_posts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


def _entry_reference(entry: dict[str, Any]) -> tuple[str | None, Any]:
    """Return ``(href, timestamp)`` from any known export shape."""
    map_data = entry.get("string_map_data")
    if isinstance(map_data, dict):
        for key, value in map_data.items():
            if key.lower() in _TIMESTAMP_KEYS and isinstance(value, dict):
                return _as_str(value.get("href")), value.get("timestamp")
        for value in map_data.values():
            if isinstance(value, dict) and value.get("href"):
                return _as_str(value.get("href")), value.get("timestamp")

    list_data = entry.get("string_list_data")
    if isinstance(list_data, list):
        for value in list_data:
            if isinstance(value, dict) and value.get("href"):
                return _as_str(value.get("href")), value.get("timestamp")

    for key in ("href", "url", "link"):
        if entry.get(key):
            return _as_str(entry.get(key)), entry.get("timestamp")
    return None, None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _parse_json(text: str, report: ImportReport, *, source_import: str) -> dict[str, SourceReel]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        report.warnings.append(f"{SAVED_POSTS_JSON} is not valid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        report.warnings.append(f"{SAVED_POSTS_JSON} does not contain a JSON object")
        return {}

    reels: dict[str, SourceReel] = {}
    for entry in _json_entries(payload):
        href, raw_timestamp = _entry_reference(entry)
        if not href:
            report.warnings.append("saved_posts.json entry has no permalink")
            continue
        shortcode = shortcode_from_url(href)
        if not shortcode:
            report.warnings.append(f"unrecognised permalink in saved_posts.json: {href}")
            continue
        author = _clean_text(_as_str(entry.get("title")))
        reels[shortcode] = SourceReel(
            shortcode=shortcode,
            url=href,
            media_type=media_type_from_url(href),
            author=author,
            saved_at=_timestamp(raw_timestamp),
            source_import=source_import,
        )
    return reels


def _html_caption(container_text: str, href: str, author: str | None) -> str | None:
    lines: list[str] = []
    for raw_line in container_text.splitlines():
        line = raw_line.strip()
        if not line or href in line:
            continue
        if _POST_BY_RE.fullmatch(line) or line.lower().startswith("post by"):
            continue
        if author and line.lower() in {author.lower(), f"@{author.lower()}"}:
            continue
        lines.append(line)
    if not lines:
        return None
    return max(lines, key=len)


def _parse_html(text: str, report: ImportReport, *, source_import: str) -> dict[str, SourceReel]:
    soup = BeautifulSoup(text, "html.parser")
    reels: dict[str, SourceReel] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        shortcode = shortcode_from_url(href)
        if not shortcode:
            continue
        container = anchor.find_parent(["div", "li", "tr", "article"])
        container_text = (
            container.get_text("\n", strip=True)
            if container is not None
            else anchor.get_text("\n", strip=True)
        )
        author_match = _POST_BY_RE.search(container_text)
        author = _clean_text(author_match.group(1)) if author_match else None
        caption = _clean_text(_html_caption(container_text, href, author))
        hashtags = [f"#{match}" for match in _HASHTAG_RE.findall(caption or "")]
        existing = reels.get(shortcode)
        if existing is not None:
            existing.caption = existing.caption or caption
            existing.author = existing.author or author
            existing.hashtags = sorted(set(existing.hashtags) | set(hashtags))
            continue
        reels[shortcode] = SourceReel(
            shortcode=shortcode,
            url=href,
            media_type=media_type_from_url(href),
            author=author,
            caption=caption,
            hashtags=hashtags,
            source_import=source_import,
        )

    if not reels:
        report.warnings.append(f"{SAVED_POSTS_HTML} contained no recognisable Instagram permalinks")
    return reels


def _parse_collections(text: str, report: ImportReport) -> dict[str, list[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        report.warnings.append(f"{SAVED_COLLECTIONS_JSON} is not valid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        report.warnings.append(f"{SAVED_COLLECTIONS_JSON} does not contain a JSON object")
        return {}

    raw_collections = payload.get("saved_saved_collections")
    if not isinstance(raw_collections, list):
        for value in payload.values():
            if isinstance(value, list):
                raw_collections = value
                break
    if not isinstance(raw_collections, list):
        return {}

    collections: dict[str, list[str]] = {}
    for entry in raw_collections:
        if not isinstance(entry, dict):
            continue
        name = _clean_text(
            _as_str(entry.get("title"))
            or _as_str(entry.get("name"))
            or _as_str(entry.get("collection_name"))
        )
        if not name:
            continue
        codes: list[str] = []
        for key in ("string_list_data", "string_map_data"):
            values = entry.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        code = shortcode_from_url(_as_str(value.get("href")) or "")
                        if code:
                            codes.append(code)
            elif isinstance(values, dict):
                for value in values.values():
                    if isinstance(value, dict):
                        code = shortcode_from_url(_as_str(value.get("href")) or "")
                        if code:
                            codes.append(code)
        if not codes:
            href = _as_str(entry.get("href"))
            code = shortcode_from_url(href or "")
            if code:
                codes.append(code)
        collections[name] = sorted(set(collections.get(name, [])) | set(codes))
    return collections


def parse_export(path: Path | str) -> ImportReport:
    """Parse a ZIP or an unpacked export into one merged inventory."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"export not found: {source}")

    report = ImportReport(source=str(source))
    files, warnings = _read_export_files(source)
    report.warnings.extend(warnings)
    report.files = sorted(files)

    if not files:
        raise ValueError(
            f"no saved_posts.json, saved_posts.html or saved_collections.json found in {source}"
        )

    source_import = report.imported_at
    merged: dict[str, SourceReel] = {}

    if SAVED_POSTS_JSON in files:
        merged.update(
            _parse_json(
                files[SAVED_POSTS_JSON].decode("utf-8-sig", errors="replace"),
                report,
                source_import=source_import,
            )
        )

    if SAVED_POSTS_HTML in files:
        html_reels = _parse_html(
            files[SAVED_POSTS_HTML].decode("utf-8-sig", errors="replace"),
            report,
            source_import=source_import,
        )
        for shortcode, reel in html_reels.items():
            existing = merged.get(shortcode)
            if existing is None:
                merged[shortcode] = reel
                continue
            existing.caption = existing.caption or reel.caption
            existing.author = existing.author or reel.author
            existing.hashtags = sorted(set(existing.hashtags) | set(reel.hashtags))

    if SAVED_COLLECTIONS_JSON in files:
        report.collections = _parse_collections(
            files[SAVED_COLLECTIONS_JSON].decode("utf-8-sig", errors="replace"), report
        )
        for name, codes in report.collections.items():
            for code in codes:
                if code in merged:
                    merged[code].collections = sorted(set(merged[code].collections) | {name})

    report.reels = sorted(
        merged.values(),
        key=lambda reel: (reel.saved_at or datetime.min.replace(tzinfo=UTC), reel.shortcode),
        reverse=True,
    )
    return report


__all__ = [
    "SAVED_COLLECTIONS_JSON",
    "SAVED_POSTS_HTML",
    "SAVED_POSTS_JSON",
    "ImportReport",
    "media_type_from_url",
    "parse_export",
    "shortcode_from_url",
]
