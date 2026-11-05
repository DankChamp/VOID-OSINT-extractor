"""Phase 0: the official export parser."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from reel_vault.importer import parse_export
from reel_vault.importer.parser import media_type_from_url, shortcode_from_url
from reel_vault.schemas.common import MediaType


def test_shortcode_and_media_type() -> None:
    assert shortcode_from_url("https://www.instagram.com/reel/AbC_123/") == "AbC_123"
    assert shortcode_from_url("https://www.instagram.com/p/XyZ-9/") == "XyZ-9"
    assert shortcode_from_url("https://www.instagram.com/tv/OLD1/") == "OLD1"
    assert shortcode_from_url("https://example.com/reel/nope/") is None
    assert media_type_from_url("https://www.instagram.com/reel/AbC/") is MediaType.REEL
    assert media_type_from_url("https://www.instagram.com/p/AbC/") is MediaType.POST


def test_directory_export_merges_json_html_and_collections(fixtures_dir: Path) -> None:
    report = parse_export(fixtures_dir)
    by_code = {reel.shortcode: reel for reel in report.reels}

    assert set(by_code) == {"AAA111", "BBB222", "CCC333", "DDD444"}
    assert report.collections == {"Psychology": ["AAA111"], "Recipes": ["BBB222"]}

    psychology = by_code["AAA111"]
    assert psychology.author == "jane_doe"
    assert psychology.media_type is MediaType.REEL
    assert psychology.caption is not None and "psychology tricks" in psychology.caption
    assert psychology.hashtags == ["#mindset", "#psychology"]
    assert psychology.collections == ["Psychology"]
    assert psychology.saved_at is not None

    recipe = by_code["BBB222"]
    assert recipe.media_type is MediaType.POST
    assert recipe.collections == ["Recipes"]
    assert recipe.caption is not None and "pasta recipe" in recipe.caption

    html_only = by_code["DDD444"]
    assert html_only.author == "html_only"
    assert html_only.caption is not None and "HTML export" in html_only.caption


def test_zip_export_is_read_without_unpacking(fixtures_dir: Path, tmp_path: Path) -> None:
    archive_path = tmp_path / "export.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in ("saved_posts.json", "saved_posts.html", "saved_collections.json"):
            archive.write(fixtures_dir / name, arcname=f"your_instagram_activity/saved/{name}")

    report = parse_export(archive_path)
    assert len(report.reels) == 4
    assert report.collections["Psychology"] == ["AAA111"]


def test_json_only_export_has_no_captions(fixtures_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "json_only"
    target.mkdir()
    (target / "saved_posts.json").write_text(
        (fixtures_dir / "saved_posts.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    report = parse_export(target)
    assert len(report.reels) == 3
    assert all(reel.caption is None for reel in report.reels)


def test_html_only_export(fixtures_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "html_only"
    target.mkdir()
    (target / "saved_posts.html").write_text(
        (fixtures_dir / "saved_posts.html").read_text(encoding="utf-8"), encoding="utf-8"
    )

    report = parse_export(target)
    assert len(report.reels) == 4
    assert report.reels[0].shortcode == "DDD444"


def test_missing_export_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_export(tmp_path / "nope.zip")


def test_export_without_known_files_raises(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    (target / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_export(target)


def test_malformed_json_is_reported_not_fatal(tmp_path: Path) -> None:
    target = tmp_path / "broken"
    target.mkdir()
    (target / "saved_posts.json").write_text("{not json", encoding="utf-8")
    report = parse_export(target)
    assert report.reels == []
    assert any("not valid JSON" in warning for warning in report.warnings)


def test_alternate_entry_shapes_are_tolerated(tmp_path: Path) -> None:
    target = tmp_path / "alternate"
    target.mkdir()
    payload = {
        "saved_saved_media": [
            {
                "title": "one",
                "string_list_data": [{"href": "https://www.instagram.com/reel/ONE1/"}],
            },
            {
                "title": "two",
                "href": "https://www.instagram.com/p/TWO2/",
                "timestamp": "2026-01-02T03:04:05Z",
            },
        ]
    }
    (target / "saved_posts.json").write_text(json.dumps(payload), encoding="utf-8")
    report = parse_export(target)
    by_code = {reel.shortcode: reel for reel in report.reels}
    assert set(by_code) == {"ONE1", "TWO2"}
    assert by_code["TWO2"].saved_at is not None
    assert by_code["TWO2"].saved_at.year == 2026
