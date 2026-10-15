"""Transcript normalization (§3, §4).

Normalization is conservative and fully auditable:

* Every rule is named, documented and recorded with the number of segments it changed.
* Original text is always kept on each segment (``raw_text``); the raw transcript file
  is never modified.
* Timestamps and segment ids are copied, never recomputed.
* Repeated passages are never removed — the speaker may repeat a point to add context,
  and consolidation happens later, in the knowledge representation only.
* Only clearly non-semantic noise is touched: unicode form, control characters,
  redundant whitespace, and cues that contain no text at all.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from video_intel.schemas.transcript import (
    NormalizationStep,
    NormalizedTranscript,
    RawTranscript,
    TranscriptSegment,
    hash_transcript,
)

_Rule = Callable[[TranscriptSegment], TranscriptSegment | None]


@dataclass(frozen=True)
class NormalizationRule:
    """One named normalization rule."""

    name: str
    description: str
    apply: _Rule


def _nfkc(segment: TranscriptSegment) -> TranscriptSegment:
    text = unicodedata.normalize("NFKC", segment.text)
    return segment if text == segment.text else segment.model_copy(update={"text": text})


def _strip_control_characters(segment: TranscriptSegment) -> TranscriptSegment:
    text = "".join(
        char for char in segment.text if char in "\n\t" or unicodedata.category(char)[0] != "C"
    )
    return segment if text == segment.text else segment.model_copy(update={"text": text})


def _collapse_whitespace(segment: TranscriptSegment) -> TranscriptSegment:
    collapsed = " ".join(segment.text.split())
    return segment if collapsed == segment.text else segment.model_copy(update={"text": collapsed})


def _drop_empty(segment: TranscriptSegment) -> TranscriptSegment | None:
    """Return ``None`` for a cue with no words at all."""
    return None if not segment.text.strip() else segment


DEFAULT_RULES: tuple[NormalizationRule, ...] = (
    NormalizationRule("unicode_nfkc", "normalise unicode to NFKC form", _nfkc),
    NormalizationRule(
        "strip_control_characters",
        "remove control characters other than newline and tab",
        _strip_control_characters,
    ),
    NormalizationRule(
        "collapse_whitespace",
        "collapse whitespace runs and strip leading/trailing space",
        _collapse_whitespace,
    ),
    NormalizationRule("drop_empty_segments", "drop cues that contain no words", _drop_empty),
)


class TranscriptNormalizer:
    """Applies an ordered rule set and records what each rule changed."""

    def __init__(self, rules: tuple[NormalizationRule, ...] = DEFAULT_RULES) -> None:
        self.rules = rules

    def normalize(self, transcript: RawTranscript) -> NormalizedTranscript:
        if not transcript.segments:
            raise ValueError("cannot normalize an empty transcript")

        segments: list[TranscriptSegment] = [
            segment.model_copy(update={"raw_text": segment.text})
            if segment.raw_text is None
            else segment
            for segment in transcript.segments
        ]
        steps: list[NormalizationStep] = []
        for rule in self.rules:
            changed = 0
            result: list[TranscriptSegment] = []
            for segment in segments:
                updated = rule.apply(segment)
                if updated is None:
                    changed += 1
                    continue
                if updated.text != segment.text:
                    changed += 1
                result.append(updated)
            segments = result
            steps.append(
                NormalizationStep(
                    name=rule.name,
                    description=rule.description,
                    segments_changed=changed,
                )
            )

        if not segments:
            raise ValueError(
                f"normalization removed every segment for video {transcript.video_id}; "
                "the raw transcript is preserved"
            )

        return NormalizedTranscript(
            video_id=transcript.video_id,
            source_transcript_hash=hash_transcript(transcript.text()),
            segments=segments,
            steps=steps,
        )
