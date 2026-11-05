"""RapidOCR engine backed by ``rapidocr>=3.0``.

The engine never sends images anywhere: inference runs entirely on this machine. If
RapidOCR is not importable, ``available`` is ``False`` and every read returns an error
so the caller can fall back to transcript-only extraction.
"""

from __future__ import annotations

from pathlib import Path

from reel_vault.ocr.base import OcrError, OcrLine, OcrResult


class RapidOcrEngine:
    def __init__(self) -> None:
        self._reader = None
        self._available = False
        self._import_error: str | None = None
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]

            self._reader = RapidOCR()
            self._available = True
        except ImportError as exc:  # pragma: no cover - environment dependent
            self._import_error = str(exc)

    @property
    def available(self) -> bool:
        return self._available

    def read(self, image: Path) -> OcrResult:
        if not self._available:
            return OcrResult(
                text="",
                lines=[],
                engine="rapidocr",
                available=False,
                error=f"rapidocr is not installed ({self._import_error})",
            )
        reader = self._reader
        if reader is None:  # pragma: no cover - defensive
            return OcrResult(
                text="",
                lines=[],
                engine="rapidocr",
                available=False,
                error="rapidocr reader is not initialized",
            )
        if not image.is_file():
            return OcrResult(
                text="",
                lines=[],
                engine="rapidocr",
                available=False,
                error=f"image not found: {image}",
            )

        try:
            result = reader(str(image))
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise OcrError(f"RapidOCR failed for {image}: {exc}") from exc

        lines: list[OcrLine] = []
        texts: list[str] = []
        if result:
            # rapidocr returns a list of (box, text, confidence) triples or
            # (result, elapsed) pairs depending on the version.
            entries = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            for entry in entries or []:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    text = str(entry[1])
                    confidence = (
                        float(entry[2]) if len(entry) > 2 and entry[2] is not None else None
                    )
                elif isinstance(entry, dict):
                    text = str(entry.get("text", ""))
                    confidence = entry.get("confidence")
                else:
                    continue
                if text:
                    lines.append(OcrLine(text=text, confidence=confidence))
                    texts.append(text)

        return OcrResult(
            text="\n".join(texts),
            lines=lines,
            engine="rapidocr",
            available=self._available,
        )


__all__ = ["RapidOcrEngine"]
