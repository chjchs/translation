from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator


FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate a single string while preserving the original if translation fails."""
    if not text or not text.strip():
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception:
        return text


def _get_original_font_size(page: fitz.Page, rect: fitz.Rect) -> float:
    """Get the largest font size used by the original text in this block."""
    best_size = 12.0
    try:
        data = page.get_text("dict", clip=rect)
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0))
                    if size > best_size:
                        best_size = size
    except Exception:
        pass
    return max(4.0, best_size)


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str, original_font_size: float) -> bool:
    """Insert translated text using Noto Sans KR, starting at the original size."""
    fontsize = max(4.0, original_font_size)

    # Korean text often occupies more horizontal space than the English text.
    # Shrink only when necessary, rather than forcing every block to 12pt.
    while fontsize >= 4:
        result = page.insert_textbox(
            rect,
            text,
            fontsize=fontsize,
            fontname=FONT_NAME,
            fontfile=str(FONT_PATH),
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )

        if result >= 0:
            return True

        fontsize -= 0.5

    return False


def translate_pdf_file(
    input_pdf_path: str,
    output_pdf_path: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
) -> int:
    """Translate PDF text while preserving the original page layout and images.

    Original text is redacted and replaced in the same text-block rectangle.
    The original font size is used as the starting point for the translation.
    Images and vector graphics underneath/near text are explicitly preserved.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            "Put NotoSansKR-Regular.ttf in the fonts folder."
        )

    doc = fitz.open(input_pdf_path)
    translated_count = 0

    try:
        for page in doc:
            blocks = page.get_text("blocks")
            replacements: list[tuple[fitz.Rect, str, float]] = []

            for block in blocks:
                text = (block[4] or "").strip()

                if not text or len(text) < 2:
                    continue

                translated = translate_text_blocks(text, source_lang, target_lang)

                if translated == text:
                    continue

                rect = fitz.Rect(block[0], block[1], block[2], block[3])
                original_font_size = _get_original_font_size(page, rect)
                replacements.append((rect, translated, original_font_size))

                print("원문:", text[:50])
                print("번역:", translated[:50])
                print("원본 폰트 크기:", original_font_size)

            # Remove only the original text. PyMuPDF's default redaction can
            # also remove images/graphics intersecting the rectangle, so make
            # those operations explicit: preserve images and vector graphics.
            for rect, _, _ in replacements:
                page.add_redact_annot(rect, fill=(1, 1, 1))

            if replacements:
                page.apply_redactions(images=0, graphics=0, text=0)

            for rect, translated, original_font_size in replacements:
                inserted = _insert_translation(
                    page,
                    rect,
                    translated,
                    original_font_size,
                )
                print("삽입 결과:", inserted)

                if inserted:
                    translated_count += 1

        doc.save(output_pdf_path, garbage=4, deflate=True)
    finally:
        doc.close()

    return translated_count


def iter_pdf_text_blocks(pdf_path: str) -> Iterable[tuple[float, float, float, float, str]]:
    """Return PDF text blocks in the original page order."""
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for block in page.get_text("blocks"):
                yield block[0], block[1], block[2], block[3], (block[4] or "").strip()
    finally:
        doc.close()
