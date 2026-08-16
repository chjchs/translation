from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator


FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"
BULLET_CHARS = "•●○◦▪▫■□◆◇"


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate a single string while preserving the original if translation fails."""
    if not text or not text.strip():
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception:
        return text


def _get_span_info(page: fitz.Page) -> list[dict]:
    """Return text lines with their spans, positions, and font sizes."""
    result = []
    data = page.get_text("dict")

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            bbox = fitz.Rect(line["bbox"])
            result.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "size": max(float(s.get("size", 0)) for s in spans),
                    "spans": spans,
                }
            )
    return result


def _is_bullet_only(text: str) -> bool:
    """Return True when a line contains only a bullet/list marker."""
    return text.strip() in BULLET_CHARS


def _bullet_prefix(text: str) -> str | None:
    """Return the bullet prefix when the text begins with a bullet marker."""
    stripped = text.lstrip()
    if stripped and stripped[0] in BULLET_CHARS:
        return stripped[0]
    return None


def _group_lines(lines: list[dict]) -> list[dict]:
    """Group bullet-only lines with the following text line when appropriate.

    PDF extraction can split a visual bullet such as ``● 교육`` into two lines
    even though the original slide treats them as one list item. This function
    joins a bullet-only line with the nearest following text line when they are
    vertically adjacent and the following line starts at roughly the same
    horizontal position as the bullet's visual text area.
    """
    groups: list[dict] = []
    i = 0

    while i < len(lines):
        current = lines[i]
        text = current["text"]

        if _is_bullet_only(text) and i + 1 < len(lines):
            nxt = lines[i + 1]
            vertical_gap = nxt["bbox"].y0 - current["bbox"].y1
            height = max(current["bbox"].height, nxt["bbox"].height, 1.0)

            # A bullet and its text are normally very close vertically.
            # Avoid joining unrelated lines farther away.
            if -0.5 * height <= vertical_gap <= 1.5 * height:
                combined_rect = fitz.Rect(
                    min(current["bbox"].x0, nxt["bbox"].x0),
                    min(current["bbox"].y0, nxt["bbox"].y0),
                    max(current["bbox"].x1, nxt["bbox"].x1),
                    max(current["bbox"].y1, nxt["bbox"].y1),
                )
                groups.append(
                    {
                        "text": f"{text} {nxt['text']}",
                        "bbox": combined_rect,
                        "size": max(current["size"], nxt["size"]),
                    }
                )
                i += 2
                continue

        groups.append(current)
        i += 1

    return groups


def _get_original_font_size(line: dict) -> float:
    return max(4.0, float(line.get("size", 12.0)))


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str, original_font_size: float) -> bool:
    """Insert translated text at the original size, shrinking only if required."""
    fontsize = max(4.0, original_font_size)

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
    """Translate PDF text while preserving bullet layout, images, and graphics."""
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            "Put NotoSansKR-Regular.ttf in the fonts folder."
        )

    doc = fitz.open(input_pdf_path)
    translated_count = 0

    try:
        for page in doc:
            lines = _get_span_info(page)
            groups = _group_lines(lines)
            replacements: list[tuple[fitz.Rect, str, float]] = []

            for item in groups:
                text = item["text"].strip()
                if not text or len(text) < 2:
                    continue

                # Keep list markers unchanged. The marker is part of the visual
                # layout and should not be translated or independently placed.
                translated = translate_text_blocks(text, source_lang, target_lang)
                if translated == text:
                    continue

                rect = fitz.Rect(item["bbox"])
                original_font_size = _get_original_font_size(item)
                replacements.append((rect, translated, original_font_size))

                print("원문:", text[:80])
                print("번역:", translated[:80])
                print("원본 폰트 크기:", original_font_size)

            # Do not paint over the redacted area. In particular, a white fill
            # can hide an image/background underneath a text block.
            for rect, _, _ in replacements:
                page.add_redact_annot(rect, fill=False)

            if replacements:
                # Preserve images and graphics. Only the original text is removed.
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