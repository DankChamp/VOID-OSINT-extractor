"""Deterministic transcript chunking (§6).

Guarantees, all covered by tests:

* Segment boundaries are respected wherever possible: a definition, list or explanation
  is not split just because a character budget ran out.
* A chunk never exceeds ``max_chars`` unless a *single* sentence is longer than the
  budget, in which case that sentence is split at a word boundary.
* Neighbouring chunks overlap by roughly ``overlap_chars`` of text, expressed by
  re-including the trailing pieces, so a statement that straddles the boundary is fully
  visible in at least one chunk. The budget always wins over the requested overlap.
* Every segment appears in at least one chunk, and the union of chunk segment ids equals
  the transcript's segment ids.
* The same transcript and configuration always produce identical chunks.

Timestamps come from the segments themselves; no chunk invents timing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from video_intel.schemas.chunk import Chunk, ChunkingConfig, ChunkSet, chunk_set_hash
from video_intel.schemas.transcript import TranscriptSegment, hash_transcript

SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")
JOINER = " "
MIN_OVERLAP_PIECES = 1


class ChunkingError(ValueError):
    """Raised when a transcript cannot be chunked under the given configuration."""


@dataclass(frozen=True)
class _Piece:
    """One indivisible unit of text plus the segment it came from."""

    text: str
    segment_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class _Group:
    """One chunk: optional repeated overlap plus the new pieces it introduces."""

    overlap: tuple[_Piece, ...]
    body: tuple[_Piece, ...]

    @property
    def pieces(self) -> tuple[_Piece, ...]:
        return (*self.overlap, *self.body)

    def render(self) -> str:
        return JOINER.join(piece.text for piece in self.pieces)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; returns ``[text]`` for text without sentence ends."""
    parts = [part.strip() for part in SENTENCE_END_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split a sentence that alone exceeds the budget, at word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def build_pieces(segments: list[TranscriptSegment], config: ChunkingConfig) -> list[_Piece]:
    """Break segments into sentence-level pieces that never exceed the budget."""
    pieces: list[_Piece] = []
    for segment in segments:
        sentences = split_sentences(segment.text) if config.respect_sentences else [segment.text]
        for sentence in sentences:
            for fragment in _split_oversized_sentence(sentence, config.max_chars):
                pieces.append(
                    _Piece(
                        text=fragment,
                        segment_id=segment.segment_id,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                    )
                )
    return pieces


def _render_length(pieces: tuple[_Piece, ...] | list[_Piece]) -> int:
    return len(JOINER.join(piece.text for piece in pieces))


def _overlap_for(pieces: list[_Piece], start: int, config: ChunkingConfig) -> tuple[_Piece, ...]:
    """Trailing text before ``start`` repeated at the head of the next chunk.

    Capped so the first new piece still fits within ``max_chars``: if the budget cannot
    hold both the requested overlap and that piece, the overlap shrinks and may be empty.
    """
    if start == 0 or start >= len(pieces) or config.overlap_chars <= 0:
        return ()

    first_new = pieces[start]
    seed: list[_Piece] = []
    total = 0
    for piece in reversed(pieces[:start]):
        addition = len(piece.text) + (len(JOINER) if seed else 0)
        if total + addition + len(JOINER) + len(first_new.text) > config.max_chars:
            break
        seed.insert(0, piece)
        total += addition
        if total >= config.overlap_chars and len(seed) >= MIN_OVERLAP_PIECES:
            break
    return tuple(seed)


def _merge_tail(groups: list[_Group], config: ChunkingConfig) -> list[_Group]:
    """Fold a trailing chunk that is too small into its predecessor when it fits.

    A chunk below ``min_chars`` usually means the budget cut a definition or a list in
    half, which §6 asks us to avoid. Merging is skipped when it would exceed
    ``max_chars``; the size limit always wins.
    """
    while len(groups) > 1:
        tail, head = groups[-1], groups[-2]
        if _render_length(tail.body) >= config.min_chars:
            break
        combined = (*head.pieces, *tail.body)
        if len(JOINER.join(piece.text for piece in combined)) > config.max_chars:
            break
        groups[-2:] = [_Group(overlap=head.overlap, body=combined)]
    return groups


def _build_groups(pieces: list[_Piece], config: ChunkingConfig) -> list[_Group]:
    """Stream pieces into chunks, prepending overlap and enforcing the size limit."""
    groups: list[_Group] = []
    index = 0
    while index < len(pieces):
        overlap = _overlap_for(pieces, index, config)
        current: list[_Piece] = list(overlap)
        consumed = index
        while consumed < len(pieces):
            addition = len(pieces[consumed].text) + (len(JOINER) if current else 0)
            if current and _render_length(current) + addition > config.max_chars:
                break
            current.append(pieces[consumed])
            consumed += 1
        if consumed == index:  # only reachable if one piece alone exceeds the budget
            current.append(pieces[index])
            consumed += 1
        groups.append(_Group(overlap=overlap, body=tuple(current[len(overlap) :])))
        index = consumed
    return _merge_tail(groups, config)


def _ordered_segment_ids(pieces: tuple[_Piece, ...]) -> list[str]:
    """Segment ids in first-appearance order, without duplicates."""
    seen: dict[str, None] = {}
    for piece in pieces:
        seen.setdefault(piece.segment_id, None)
    return list(seen)


def chunk_transcript(
    segments: list[TranscriptSegment],
    config: ChunkingConfig,
    *,
    video_id: str,
    transcript_hash: str | None = None,
) -> ChunkSet:
    """Split a transcript into overlapping, timestamped chunks.

    Raises:
        ChunkingError: when there is nothing to chunk.
    """
    if not segments:
        raise ChunkingError("cannot chunk an empty transcript")

    pieces = build_pieces(segments, config)
    if not pieces:
        raise ChunkingError("transcript contains no text to chunk")

    chunks: list[Chunk] = []
    for index, group in enumerate(_build_groups(pieces, config)):
        pieces_in_chunk = group.pieces
        segment_ids = _ordered_segment_ids(pieces_in_chunk)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{index + 1:04d}",
                index=index,
                start_seconds=min(piece.start_seconds for piece in pieces_in_chunk),
                end_seconds=max(piece.end_seconds for piece in pieces_in_chunk),
                text=group.render(),
                segment_ids=segment_ids,
                overlaps_previous=bool(group.overlap) and index > 0,
            )
        )

    digest = transcript_hash or hash_transcript(JOINER.join(s.text for s in segments))
    return ChunkSet(
        video_id=video_id,
        config=config,
        transcript_hash=digest,
        chunks=chunks,
        cache_key=chunk_set_hash(config, digest),
    )
