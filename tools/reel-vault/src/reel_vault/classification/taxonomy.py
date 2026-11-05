"""Editable taxonomy for reel classification.

Loaded from ``reel-vault-taxonomy.yaml`` (or ``config.taxonomy_file``). Each category
has a name, description, keywords, example captions and optional collection mappings.
Rules prefilter obvious cases; the LLM handles the rest. Renaming or adding categories
never requires code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "psychology & mind",
    "learning & study tricks",
    "news & information",
    "comedy & funny",
    "tech & AI",
    "money & business",
    "health & fitness",
    "food & recipes",
    "travel & places",
    "style & design",
    "life hacks & DIY",
    "tools & resources",
    "inspiration & motivation",
    "other",
)

DEFAULT_TAXONOMY_YAML = """\
categories:
  - name: psychology & mind
    description: Emotional, reflective, relationship and comfort posts.
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
    example_captions:
      - "feeling sad and lonely tonight"
      - "this breakup destroyed me"
    collections:
      - psychology & mind
      - Psych

  - name: learning & study tricks
    description: Tutorials, study methods, frameworks and actionable how-to advice.
    keywords:
      - how to
      - step
      - trick
      - hack
      - tutorial
      - guide
      - study
      - learn
    example_captions:
      - "how to focus while studying"
      - "3 tricks to remember anything"
    collections:
      - Learning
      - learning & study tricks

  - name: news & information
    description: Breaking news, reports, research and sourced claims.
    keywords:
      - breaking
      - report
      - study
      - research
      - source
      - study shows
    example_captions:
      - "breaking: a new study reports"
      - "researchers found that"
    collections:
      - News

  - name: comedy & funny
    description: Humor, memes, sketches and entertainment.
    keywords:
      - funny
      - meme
      - hilarious
      - comedy
      - laugh
    example_captions:
      - "this had me dead"
      - "why is this so funny"
    collections:
      - Funny

  - name: tech & AI
    description: Technology, AI, software, gadgets and coding.
    keywords:
      - ai
      - python
      - code
      - software
      - gadget
      - phone
      - app
    example_captions:
      - "this AI tool is wild"
      - "python trick you need"
    collections:
      - Tech

  - name: money & business
    description: Finance, investing, side hustles and career advice.
    keywords:
      - money
      - invest
      - side hustle
      - business
      - salary
      - budget
    example_captions:
      - "how I make $5k/month"
      - "investing for beginners"
    collections:
      - Money

  - name: health & fitness
    description: Workouts, nutrition, wellness and medical info.
    keywords:
      - workout
      - gym
      - fitness
      - health
      - nutrition
      - diet
      - sleep
    example_captions:
      - "morning workout routine"
      - "best foods for energy"
    collections:
      - Health

  - name: food & recipes
    description: Cooking, recipes, ingredients and food tips.
    keywords:
      - recipe
      - cook
      - ingredient
      - food
      - bake
    example_captions:
      - "5 ingredient pasta"
      - "easy dinner idea"
    collections:
      - Food

  - name: travel & places
    description: Trips, destinations, hotels and travel guides.
    keywords:
      - travel
      - trip
      - destination
      - hotel
      - flight
      - vacation
    example_captions:
      - "things to do in Tokyo"
      - "best hotels in Paris"
    collections:
      - Travel

  - name: style & design
    description: Fashion, interiors, aesthetics and design.
    keywords:
      - fashion
      - outfit
      - style
      - design
      - aesthetic
      - decor
    example_captions:
      - "outfit of the day"
      - "bedroom makeover"
    collections:
      - Style

  - name: life hacks & DIY
    description: Home hacks, DIY projects and everyday tips.
    keywords:
      - diy
      - hack
      - home
      - tip
      - trick
      - project
    example_captions:
      - "kitchen hack that works"
      - "diy shelf tutorial"
    collections:
      - DIY

  - name: tools & resources
    description: Websites, apps, products and resources worth bookmarking.
    keywords:
      - tool
      - website
      - resource
      - app
      - free
      - link
    example_captions:
      - "useful website for students"
      - "best free tool"
    collections:
      - Tools

  - name: inspiration & motivation
    description: Motivational quotes, mindset and inspirational content.
    keywords:
      - motivation
      - inspire
      - mindset
      - quote
      - success
    example_captions:
      - "stay motivated"
      - "mindset shift"
    collections:
      - Inspiration

  - name: other
    description: Anything that does not fit the categories above.
    keywords: []
    example_captions: []
    collections: []
"""


class TaxonomyError(RuntimeError):
    """The taxonomy file is missing or malformed."""


class _CategoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    example_captions: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)


class _TaxonomyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    categories: list[_CategoryFile] = Field(default_factory=list)


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    keywords: tuple[str, ...]
    example_captions: tuple[str, ...]
    collections: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    categories: tuple[Category, ...]
    source: str | None = None

    def by_name(self, name: str) -> Category | None:
        folded = name.casefold()
        for category in self.categories:
            if category.name.casefold() == folded:
                return category
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(category.name for category in self.categories)


def _clean(values: Any) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for value in values or []:
        text = " ".join(str(value).split())
        if text:
            seen.setdefault(text.casefold(), text)
    return tuple(seen.values())


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load taxonomy from ``path``, or return the packaged defaults."""
    if path is None:
        parsed = _TaxonomyFile.model_validate(yaml.safe_load(DEFAULT_TAXONOMY_YAML))
        categories = tuple(
            Category(
                name=category.name,
                description=category.description,
                keywords=_clean(category.keywords),
                example_captions=_clean(category.example_captions),
                collections=_clean(category.collections),
            )
            for category in parsed.categories
        )
        return Taxonomy(categories=categories)
    if not path.is_file():
        raise TaxonomyError(f"taxonomy file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise TaxonomyError(f"cannot read taxonomy file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"invalid YAML in taxonomy file {path}: {exc}") from exc
    if raw is None:
        return Taxonomy(categories=(), source=str(path))
    if not isinstance(raw, dict):
        raise TaxonomyError(f"taxonomy file {path} must contain a mapping at the top level")

    try:
        parsed = _TaxonomyFile.model_validate(raw)
    except ValidationError as exc:
        raise TaxonomyError(f"invalid taxonomy file {path}:\\n{exc}") from exc

    categories = tuple(
        Category(
            name=category.name,
            description=category.description,
            keywords=_clean(category.keywords),
            example_captions=_clean(category.example_captions),
            collections=_clean(category.collections),
        )
        for category in parsed.categories
    )
    return Taxonomy(categories=categories, source=str(path))


__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_TAXONOMY_YAML",
    "Category",
    "Taxonomy",
    "TaxonomyError",
    "load_taxonomy",
]
