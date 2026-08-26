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
    """Insert translated text without modifying the source page."""
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


def _render_translation_background(
    source_page: fitz.Page,
    replacements: list[tuple[fitz.Rect, str, float]],
    dpi: int = 150,
) -> tuple[bytes, float]:
    """Render the original page to an image and mask only text that will be translated.

    The source PDF page is never redacted or modified. The returned image becomes the
    visual background of a newly-created output page, after which translations are
    inserted as real PDF text.
    """
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = source_page.get_pixmap(matrix=matrix, alpha=False)

    # Work on a temporary document so the raster background can be edited without
    # touching the original page or its PDF objects.
    image_doc = fitz.open()
    try:
        image_page = image_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
        image_page.insert_image(source_page.rect, pixmap=pix)

        # Mask only groups that actually have a different translation. Text inside
        # images is never included because image-overlapping groups are skipped.
        for rect, _, _ in replacements:
            image_page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)

        rendered = image_page.get_pixmap(matrix=matrix, alpha=False)
        return rendered.tobytes("png"), scale
    finally:
        image_doc.close()


def translate_pdf_file(
    input_pdf_path: str,
    output_pdf_path: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
    debug_grouping: bool = False,
) -> int:
    """Translate each source page onto a newly-created page.

    Unlike the old implementation, the original page is never redacted, recolored,
    or otherwise modified. For every source page we create a fresh page, use a
    rasterized copy of the source as its visual background, mask only the text groups
    being replaced, and insert the Korean translations as a new text layer.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            "Put NotoSansKR-Regular.ttf in the fonts folder."
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
            replacements: list[tuple[fitz.Rect, str, float]] = []

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
                replacements.append((rect, translated, original_font_size))

                print("원문:", text[:120])
                print("번역:", translated[:120])
                print("group type:", item.get("group_type"))
                print("group bbox:", rect)
                if near_image and nearest_image is not None:
                    print("nearest image bbox:", nearest_image)

            # Every output page is created independently. The source page remains
            # untouched in source_doc for the entire operation.
            output_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )

            if replacements:
                background_png, _ = _render_translation_background(source_page, replacements)
                output_page.insert_image(output_page.rect, stream=background_png)
            else:
                # Pages without translations are copied visually as-is.
                pix = source_page.get_pixmap(matrix=fitz.Matrix(150 / 72.0, 150 / 72.0), alpha=False)
                output_page.insert_image(output_page.rect, pixmap=pix)

            for rect, translated, original_font_size in replacements:
                inserted = _insert_translation(output_page, rect, translated, original_font_size)
                print("삽입 결과:", inserted)
                if inserted:
                    translated_count += 1

        output_doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        output_doc.close()
        source_doc.close()

    print("이미지와 겹쳐 건너뛴 텍스트:", skipped_image_count)
    print("이미지와 가깝지만 바깥에 있어 번역한 텍스트:", translated_near_image_count)
    print("원본 페이지는 수정하지 않고 새 페이지에 번역 결과를 생성했습니다.")
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
