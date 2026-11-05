"""OCR protocols and errors.

OCR is optional: ``rapidocr>=3.0`` is an optional extra because its ONNX runtime does
not publish wheels for every Python version. When it is missing, extraction degrades to
caption + transcript only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


class OcrError(RuntimeError):
    """OCR failed for one frame; the frame is skipped."""


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OcrResult:
    text: str
    lines: list[OcrLine]
    engine: str
    available: bool
    error: str | None = None


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, image: Path) -> OcrResult:
        """Run OCR on one image file and return detected lines."""


__all__ = ["OcrEngine", "OcrError", "OcrLine", "OcrResult"]
