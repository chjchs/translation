from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator

from grouping_engine import group_page

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"
IMAGE_OVERLAP_TOLERANCE = 0.5
IMAGE_NEAR_DISTANCE_RATIO = 1.5
IMAGE_NEAR_DISTANCE_MIN = 4.0
IMAGE_NEAR_DISTANCE_MAX = 18.0


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


def _distance_to_rect(text_rect: fitz.Rect, image_rect: fitz.Rect) -> float:
    """Return the shortest edge-to-edge distance between two rectangles."""
    dx = max(image_rect.x0 - text_rect.x1, text_rect.x0 - image_rect.x1, 0.0)
    dy = max(image_rect.y0 - text_rect.y1, text_rect.y0 - image_rect.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _nearest_image_distance(rect: fitz.Rect, image_rects: list[fitz.Rect]) -> tuple[float, fitz.Rect | None]:
    if not image_rects:
        return float("inf"), None
    distances = [(_distance_to_rect(rect, image_rect), image_rect) for image_rect in image_rects]
    return min(distances, key=lambda item: item[0])


def _is_near_image(rect: fitz.Rect, image_rects: list[fitz.Rect]) -> tuple[bool, float, fitz.Rect | None]:
    """Classify non-overlapping text as near an image using text size as the scale."""
    distance, image_rect = _nearest_image_distance(rect, image_rects)
    if image_rect is None:
        return False, distance, None

    text_height = max(1.0, rect.height)
    threshold = min(
        IMAGE_NEAR_DISTANCE_MAX,
        max(IMAGE_NEAR_DISTANCE_MIN, text_height * IMAGE_NEAR_DISTANCE_RATIO),
    )
    return distance <= threshold, distance, image_rect


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str, original_font_size: float) -> bool:
    """Insert translated text onto a newly-created blank page."""
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
    """Create completely new blank pages and place translations directly on them.

    The source PDF is used only for analysis. Its pages are never rendered, copied,
    redacted, or used as visual backgrounds. Each output page starts blank and the
    translated logical groups are inserted at their original coordinates.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            f"Put NotoSansKR-Regular.ttf in the fonts folder."
        )

    source_doc = fitz.open(input_pdf_path)
    output_doc = fitz.open()
    translated_count = 0
    skipped_image_count = 0
    translated_near_image_count = 0

    try:
        for page_number, source_page in enumerate(source_doc, start=1):
            lines = _get_span_info(source_page)
            groups = group_page(source_page, lines, debug=debug_grouping)
            image_rects = _get_image_rects(source_page)

            # IMPORTANT: this is a genuinely empty page. Nothing from source_page
            # is copied into it. The source page is only queried for coordinates and
            # grouping information.
            output_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )

            print(f"페이지 {page_number}: {len(lines)} lines -> {len(groups)} logical groups")

            for group_index, item in enumerate(groups):
                text = item["text"].strip()
                if not text or len(text) < 2:
                    continue

                rect = fitz.Rect(item["bbox"])
                if _overlaps_image(rect, image_rects):
                    skipped_image_count += 1
                    print(
                        f"이미지 겹침으로 건너뜀 (페이지 {page_number}, group {group_index}): "
                        f"{text[:80]}"
                    )
                    continue

                near_image, image_distance, nearest_image = _is_near_image(rect, image_rects)
                if near_image:
                    translated_near_image_count += 1
                    print(
                        f"이미지 근처 텍스트 번역 (페이지 {page_number}, group {group_index}, "
                        f"거리 {image_distance:.1f}): {text[:80]}"
                    )

                translated = translate_text_blocks(text, source_lang, target_lang)
                if translated == text:
                    continue

                original_font_size = max(4.0, float(item.get("size", 12.0)))
                inserted = _insert_translation(
                    output_page,
                    rect,
                    translated,
                    original_font_size,
                )

                print("원문:", text[:120])
                print("번역:", translated[:120])
                print("삽입 결과:", inserted)
                print("group type:", item.get("group_type"))
                print("group bbox:", rect)
                if near_image and nearest_image is not None:
                    print("nearest image bbox:", nearest_image)

                if inserted:
                    translated_count += 1

        output_doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        output_doc.close()
        source_doc.close()

    print("이미지와 겹쳐 건너뛴 텍스트:", skipped_image_count)
    print("이미지와 가깝지만 바깥에 있어 번역한 텍스트:", translated_near_image_count)
    print("빈 페이지를 생성하고 번역문만 직접 삽입했습니다.")
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
