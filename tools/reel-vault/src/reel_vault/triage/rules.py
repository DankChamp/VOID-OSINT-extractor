"""Editable rule guardrails.

Keywords are a *trigger*, never the final decision. The rule set is loaded from
``archive_rules_file`` when configured, otherwise sensible defaults apply. Matching is
word-boundary aware so "I love this trick" does not fire the emotional archive rule.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_ARCHIVE_KEYWORDS: tuple[str, ...] = (
    "sad",
    "emo",
    "depressed",
    "depression",
    "cuddle",
    "cuddling",
    "love",
    "heartbreak",
    "loneliness",
    "comfort",
    "nostalgia",
)
DEFAULT_ARCHIVE_CATEGORIES: tuple[str, ...] = ("psychology & mind",)
DEFAULT_TUTORIAL_KEYWORDS: tuple[str, ...] = (
    "how to",
    "step",
    "trick",
    "hack",
    "tutorial",
    "guide",
    "recipe",
)
DEFAULT_NEWS_KEYWORDS: tuple[str, ...] = ("breaking", "report", "study", "research", "source")


class RuleError(RuntimeError):
    """The archive rules file is missing or malformed."""


class _ArchiveSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] | None = None
    categories: list[str] | None = None


class _ExtractSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tutorial_keywords: list[str] | None = None
    news_keywords: list[str] | None = None


class _RulesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_with_llm: bool = True
    archive: _ArchiveSection = Field(default_factory=_ArchiveSection)
    extract: _ExtractSection = Field(default_factory=_ExtractSection)


def _clean(values: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for value in values:
        text = " ".join(str(value).split())
        if text:
            seen.setdefault(text.casefold(), text)
    return tuple(seen.values())


@dataclass(frozen=True)
class ArchiveRules:
    confirm_with_llm: bool = True
    archive_keywords: tuple[str, ...] = DEFAULT_ARCHIVE_KEYWORDS
    archive_categories: tuple[str, ...] = DEFAULT_ARCHIVE_CATEGORIES
    tutorial_keywords: tuple[str, ...] = DEFAULT_TUTORIAL_KEYWORDS
    news_keywords: tuple[str, ...] = DEFAULT_NEWS_KEYWORDS
    source: str | None = None

    def detect(self, text: str, collections: Sequence[str] = ()) -> RuleMatches:
        folded = text.casefold()
        archive = _find(folded, self.archive_keywords)
        categories = _match_categories(collections, self.archive_categories)
        tutorial = _find(folded, self.tutorial_keywords)
        news = _find(folded, self.news_keywords)
        return RuleMatches(
            archive_keywords=archive,
            archive_categories=categories,
            tutorial_keywords=tutorial,
            news_keywords=news,
        )


@dataclass(frozen=True)
class RuleMatches:
    archive_keywords: tuple[str, ...] = ()
    archive_categories: tuple[str, ...] = ()
    tutorial_keywords: tuple[str, ...] = ()
    news_keywords: tuple[str, ...] = ()

    @property
    def archive_triggered(self) -> bool:
        return bool(self.archive_keywords or self.archive_categories)

    @property
    def tutorial_triggered(self) -> bool:
        return bool(self.tutorial_keywords)

    @property
    def news_triggered(self) -> bool:
        return bool(self.news_keywords)

    @property
    def forced_extract(self) -> bool:
        return self.tutorial_triggered or self.news_triggered

    @property
    def triggered(self) -> bool:
        return self.archive_triggered or self.forced_extract

    def hints(self) -> list[str]:
        """Human-readable hints handed to the model, never decisions."""
        hints: list[str] = []
        if self.archive_keywords:
            hints.append("emotional/personal keywords matched: " + ", ".join(self.archive_keywords))
        if self.archive_categories:
            hints.append(
                "emotional/personal collection matched: " + ", ".join(self.archive_categories)
            )
        if self.tutorial_keywords:
            hints.append("tutorial keywords matched: " + ", ".join(self.tutorial_keywords))
        if self.news_keywords:
            hints.append("news/research keywords matched: " + ", ".join(self.news_keywords))
        return hints


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)")


def _find(text: str, keywords: Iterable[str]) -> tuple[str, ...]:
    found: list[str] = []
    for keyword in keywords:
        if keyword and _keyword_pattern(keyword).search(text):
            found.append(keyword)
    return tuple(found)


def _match_categories(collections: Sequence[str], categories: Iterable[str]) -> tuple[str, ...]:
    normalized = [" ".join(value.casefold().split()) for value in collections if value.strip()]
    found: list[str] = []
    for category in categories:
        needle = " ".join(category.casefold().split())
        if not needle:
            continue
        if any(needle == value or needle in value or value in needle for value in normalized):
            found.append(category)
    return tuple(found)


def load_rules(path: Path | None = None) -> ArchiveRules:
    """Load rules from ``path``, or return the packaged defaults."""
    if path is None:
        return ArchiveRules()
    if not path.is_file():
        raise RuleError(f"archive rules file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise RuleError(f"cannot read archive rules file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuleError(f"invalid YAML in archive rules file {path}: {exc}") from exc
    if raw is None:
        return ArchiveRules(source=str(path))
    if not isinstance(raw, dict):
        raise RuleError(f"archive rules file {path} must contain a mapping at the top level")

    try:
        parsed = _RulesFile.model_validate(raw)
    except ValidationError as exc:
        raise RuleError(f"invalid archive rules file {path}:\n{exc}") from exc

    archive_keywords = (
        DEFAULT_ARCHIVE_KEYWORDS
        if parsed.archive.keywords is None
        else _clean(parsed.archive.keywords)
    )
    archive_categories = (
        DEFAULT_ARCHIVE_CATEGORIES
        if parsed.archive.categories is None
        else _clean(parsed.archive.categories)
    )
    tutorial_keywords = (
        DEFAULT_TUTORIAL_KEYWORDS
        if parsed.extract.tutorial_keywords is None
        else _clean(parsed.extract.tutorial_keywords)
    )
    news_keywords = (
        DEFAULT_NEWS_KEYWORDS
        if parsed.extract.news_keywords is None
        else _clean(parsed.extract.news_keywords)
    )

    return ArchiveRules(
        confirm_with_llm=parsed.confirm_with_llm,
        archive_keywords=archive_keywords,
        archive_categories=archive_categories,
        tutorial_keywords=tutorial_keywords,
        news_keywords=news_keywords,
        source=str(path),
    )


__all__ = [
    "DEFAULT_ARCHIVE_CATEGORIES",
    "DEFAULT_ARCHIVE_KEYWORDS",
    "DEFAULT_NEWS_KEYWORDS",
    "DEFAULT_TUTORIAL_KEYWORDS",
    "ArchiveRules",
    "RuleError",
    "RuleMatches",
    "load_rules",
]
