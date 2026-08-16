from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator


FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansCJKkr-Regular.ttf"


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate a single string while preserving the original if translation fails."""
    if not text or not text.strip():
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception:
        return text


def _get_font_size(rect: fitz.Rect, text: str) -> float:
    """Choose a font size that fits the translated text into the original block."""
    # Start close to the original block height and shrink until the text fits.
    size = max(4.0, min(12.0, rect.height * 0.8))
    return size


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str) -> bool:
    """Insert translated text into rect using Noto Sans CJK KR, shrinking if needed."""
    fontsize = _get_font_size(rect, text)

    while fontsize >= 4:
        result = page.insert_textbox(
            rect,
            text,
            fontsize=fontsize,
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
    """Translate PDF text blocks while preserving the existing page layout.

    The original text in each translated text block is redacted first, then the
    translated text is inserted into the same rectangle using Noto Sans CJK KR.
    Images, drawings, and the rest of the page are left intact.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans CJK KR font not found: {FONT_PATH}\n"
            "Put NotoSansCJKkr-Regular.ttf in the fonts folder."
        )

    doc = fitz.open(input_pdf_path)
    translated_count = 0

    try:
        for page in doc:
            blocks = page.get_text("blocks")
            replacements: list[tuple[fitz.Rect, str]] = []

            for block in blocks:
                text = (block[4] or "").strip()

                if not text or len(text) < 2:
                    continue

                translated = translate_text_blocks(text, source_lang, target_lang)

                if translated == text:
                    continue

                rect = fitz.Rect(block[0], block[1], block[2], block[3])
                replacements.append((rect, translated))

                print("원문:", text[:50])
                print("번역:", translated[:50])

            # Redact the original text only after collecting all blocks.
            # Using redaction removes the old text from the page rather than
            # merely drawing the translation on top of it.
            for rect, _ in replacements:
                page.add_redact_annot(rect, fill=(1, 1, 1))

            if replacements:
                page.apply_redactions()

            for rect, translated in replacements:
                inserted = _insert_translation(page, rect, translated)
                print("삽입 결과:", inserted)

                if inserted:
                    translated_count += 1

        doc.save(output_pdf_path)
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
