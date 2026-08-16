from __future__ import annotations

from typing import Iterable

import fitz
from deep_translator import GoogleTranslator


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate a single string while preserving the original if translation fails."""
    if not text or not text.strip():
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception:
        return text


def translate_pdf_file(
    input_pdf_path: str,
    output_pdf_path: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
) -> int:
    """Translate text in a PDF while leaving non-text elements intact.

    The function overlays translated text over the original text blocks on each page,
    preserving all other page content such as images, vector drawings, and layout.
    """
    doc = fitz.open(input_pdf_path)
    translated_count = 0

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = (block[4] or "").strip()
            if not text or len(text) < 2:
                continue

            translated = translate_text_blocks(text, source_lang, target_lang)
            if translated == text:
                continue

            rect = fitz.Rect(block[0], block[1], block[2], block[3])
            fontsize = max(8.0, min(18.0, (block[3] - block[1]) * 0.75))
            page.insert_textbox(
                rect,
                translated,
                fontsize=fontsize,
                fontname="helv",
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT,
            )
            translated_count += 1

    doc.save(output_pdf_path)
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
