"""Link resolution: titles and reachability for URLs found in captions and transcripts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedLink:
    url: str
    final_url: str | None = None
    title: str | None = None
    status: str = "ok"
    source: str | None = None
    error: str | None = None


_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def find_urls(text: str) -> list[str]:
    """Extract candidate URLs from free text, stripping trailing punctuation."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,:;!?)]}'\"")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def resolve_links(
    urls: list[str],
    *,
    max_links: int = 12,
    timeout_seconds: float = 10.0,
    fetch: Callable[[str, float], tuple[str, str | None, str, str | None]] | None = None,
) -> list[ResolvedLink]:
    """Resolve a list of URLs to (final_url, title, status).

    ``fetch`` is injected in tests. By default a real HTTP HEAD/GET is performed.
    """
    resolved: list[ResolvedLink] = []
    seen: set[str] = set()
    for url in urls:
        if len(resolved) >= max_links:
            break
        if url in seen:
            continue
        seen.add(url)
        try:
            if fetch is not None:
                final_url, title, status, error = fetch(url, timeout_seconds)
            else:
                final_url, title, status, error = _default_fetch(url, timeout_seconds)
        except Exception as exc:  # pragma: no cover - network dependent
            final_url, title, status, error = url, None, "error", str(exc)
        resolved.append(
            ResolvedLink(
                url=url,
                final_url=final_url or url,
                title=title,
                status=status,
                error=error,
            )
        )
    return resolved


def _default_fetch(url: str, timeout_seconds: float) -> tuple[str, str | None, str, str | None]:
    import urllib.request

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "reel-vault/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            final_url = str(response.geturl() or url)
            status = "ok" if response.status < 400 else "unreachable"
            return final_url, None, status, None
    except Exception as exc:
        return url, None, "error", str(exc)


__all__ = ["ResolvedLink", "find_urls", "resolve_links"]
