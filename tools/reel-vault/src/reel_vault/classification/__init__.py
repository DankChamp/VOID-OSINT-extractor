"""Classification: taxonomy rules + LLM labeling with confidence."""

from __future__ import annotations

from reel_vault.classification.rules import CategoryMatch, prefilter
from reel_vault.classification.service import ClassificationService, ClassificationSummary
from reel_vault.classification.taxonomy import (
    DEFAULT_CATEGORIES,
    DEFAULT_TAXONOMY_YAML,
    Category,
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
)

__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_TAXONOMY_YAML",
    "Category",
    "CategoryMatch",
    "ClassificationService",
    "ClassificationSummary",
    "Taxonomy",
    "TaxonomyError",
    "load_taxonomy",
    "prefilter",
]
