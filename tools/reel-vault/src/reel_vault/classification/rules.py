"""Rule-based prefilter for classification.

Matches caption, transcript, hashtags and saved collections against each taxonomy
category's keywords and collections using word-boundary aware matching. Produces
candidate labels with confidences that the service can accept outright or pass to the
LLM for confirmation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from reel_vault.classification.taxonomy import Taxonomy


class CategoryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\\w){re.escape(keyword.casefold())}(?!\\w)")


def _keyword_matches(text: str, keywords: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for keyword in keywords:
        if keyword and _keyword_pattern(keyword).search(text):
            found.append(keyword)
    return tuple(found)


def _collection_matches(
    collections: Sequence[str], category_collections: Sequence[str]
) -> tuple[str, ...]:
    normalized = [" ".join(value.casefold().split()) for value in collections if value.strip()]
    found: list[str] = []
    for category_collection in category_collections:
        needle = " ".join(category_collection.casefold().split())
        if not needle:
            continue
        if any(needle == value or needle in value or value in needle for value in normalized):
            found.append(category_collection)
    return tuple(found)


def prefilter(
    taxonomy: Taxonomy,
    *,
    caption: str | None = None,
    transcript: str | None = None,
    hashtags: Sequence[str] = (),
    collections: Sequence[str] = (),
) -> list[CategoryMatch]:
    """Return candidate category matches sorted by confidence descending."""
    text_parts: list[str] = []
    if caption:
        text_parts.append(caption)
    if transcript:
        text_parts.append(transcript)
    if hashtags:
        text_parts.append(" ".join(hashtags))
    text = " ".join(text_parts).casefold()

    normalized_collections = [
        " ".join(value.casefold().split()) for value in collections if value.strip()
    ]

    matches: list[CategoryMatch] = []
    for category in taxonomy.categories:
        keywords_matched = _keyword_matches(text, category.keywords)
        collections_matched = _collection_matches(normalized_collections, category.collections)

        confidence = 0.0
        reasons: list[str] = []
        if keywords_matched:
            confidence = max(confidence, 0.8)
            reasons.append("keywords: " + ", ".join(keywords_matched))
        if collections_matched:
            confidence = max(confidence, 0.9)
            reasons.append("collections: " + ", ".join(collections_matched))

        if confidence > 0:
            matches.append(
                CategoryMatch(
                    category=category.name,
                    confidence=min(confidence, 0.95),
                    reason="; ".join(reasons),
                )
            )

    matches.sort(key=lambda match: match.confidence, reverse=True)
    return matches


__all__ = ["CategoryMatch", "prefilter"]
