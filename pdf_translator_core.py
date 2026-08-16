from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator

from grouping_engine import group_page

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"
IMAGE_OVERLAP_TOLERANCE = 0.5


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate a single logical group while preserving the original if translation fails."""
    if not text or not text.strip():
        return text
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception as exc:
        print(f"번역 실패, 원문 유지: {exc}")
        return text


def _get_span_info(page: fitz.Page) -> list[dict]:
    """Extract PDF lines while retaining every original span and layout metadata."""
    result = []
    data = page.get_text("dict")
    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            result.append(
                {
                    "text": text,
                    "bbox": fitz.Rect(line["bbox"]),
                    "size": max(float(s.get("size", 0) or 0) for s in spans),
                    "spans": spans,
                    "block_index": block_index,
                    "line_index": line_index,
                }
            )
    return result


def _get_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Return visible image rectangles, including repeated image occurrences."""
    rects: list[fitz.Rect] = []
    try:
        for info in page.get_image_info(hashes=False, xrefs=False):
            if info.get("bbox"):
                rects.append(fitz.Rect(info["bbox"]))
    except Exception:
        pass
    try:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") == 1 and block.get("bbox"):
                rects.append(fitz.Rect(block["bbox"]))
    except Exception:
        pass
    return rects


def _overlaps_image(rect: fitz.Rect, image_rects: list[fitz.Rect]) -> bool:
    expanded = fitz.Rect(
        rect.x0 - IMAGE_OVERLAP_TOLERANCE,
        rect.y0 - IMAGE_OVERLAP_TOLERANCE,
        rect.x1 + IMAGE_OVERLAP_TOLERANCE,
        rect.y1 + IMAGE_OVERLAP_TOLERANCE,
    )
    return any(expanded.intersects(image_rect) for image_rect in image_rects)


def _get_underline_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Find thin horizontal vector lines that are likely text underlines."""
    result: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return result

    for drawing in drawings:
        rect = fitz.Rect(drawing.get("rect", (0, 0, 0, 0)))
        if (
            rect.width >= 4
            and rect.height <= 2.5
            and rect.width <= page.rect.width * 0.9
        ):
            result.append(rect)
    return result


def _underlines_for_replacement(
    rect: fitz.Rect, underline_rects: list[fitz.Rect]
) -> list[fitz.Rect]:
    """Return underline-like drawings belonging to a translated text box."""
    result = []
    for line in underline_rects:
        horizontal_overlap = max(0.0, min(rect.x1, line.x1) - max(rect.x0, line.x0))
        if horizontal_overlap < min(line.width * 0.5, rect.width * 0.4):
            continue
        # Underlines normally sit directly on or just below the text bbox.
        if line.y0 < rect.y0 - 2 or line.y0 > rect.y1 + max(4.0, rect.height * 0.35):
            continue
        result.append(line)
    return result


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str, original_font_size: float) -> bool:
    """Insert translated text, shrinking only when the original box is too small."""
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
    debug_grouping: bool = False,
) -> int:
    """Translate PDF using deterministic layout grouping, without AI grouping."""
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            "Put NotoSansKR-Regular.ttf in the fonts folder."
        )

    doc = fitz.open(input_pdf_path)
    translated_count = 0
    skipped_image_count = 0
    try:
        for page_number, page in enumerate(doc, start=1):
            lines = _get_span_info(page)
            groups = group_page(page, lines, debug=debug_grouping)
            image_rects = _get_image_rects(page)
            underline_rects = _get_underline_rects(page)
            replacements: list[tuple[fitz.Rect, str, float]] = []

            print(f"페이지 {page_number}: {len(lines)} lines -> {len(groups)} logical groups")

            for group_index, item in enumerate(groups):
                text = item["text"].strip()
                if not text or len(text) < 2:
                    continue

                rect = fitz.Rect(item["bbox"])
                if _overlaps_image(rect, image_rects):
                    skipped_image_count += 1
                    print(f"이미지 겹침으로 건너뜀 (페이지 {page_number}, group {group_index}): {text[:80]}")
                    continue

                translated = translate_text_blocks(text, source_lang, target_lang)
                if translated == text:
                    continue

                original_font_size = max(4.0, float(item.get("size", 12.0)))
                replacements.append((rect, translated, original_font_size))

                print("원문:", text[:120])
                print("번역:", translated[:120])
                print("group type:", item.get("group_type"))
                print("group bbox:", rect)

            # Remove the original text. If a thin vector line is attached to a
            # replaced text box (typically an underline from the source PDF),
            # redact that line too. Other graphics remain untouched.
            for rect, _, _ in replacements:
                page.add_redact_annot(rect, fill=False, cross_out=False)
                for underline in _underlines_for_replacement(rect, underline_rects):
                    page.add_redact_annot(underline, fill=False, cross_out=False)

            if replacements:
                page.apply_redactions(images=0, graphics=2, text=0)

            for rect, translated, original_font_size in replacements:
                inserted = _insert_translation(page, rect, translated, original_font_size)
                print("삽입 결과:", inserted)
                if inserted:
                    translated_count += 1

        doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        doc.close()

    print("이미지와 겹쳐 건너뛴 텍스트:", skipped_image_count)
    return translated_count


def iter_pdf_text_blocks(pdf_path: str) -> Iterable[tuple[float, float, float, float, str]]:
    """Return PDF text blocks in original page order."""
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for block in page.get_text("blocks"):
                yield block[0], block[1], block[2], block[3], (block[4] or "").strip()
    finally:
        doc.close()
