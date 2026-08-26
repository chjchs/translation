from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"

# Translation requests can occasionally receive Google's HTML error page instead
# of a translated string. Keep retries conservative so a temporary failure does
# not abort the whole PDF translation.
TRANSLATION_RETRIES = 3
TRANSLATION_RETRY_DELAY = 2.0
GOOGLE_ERROR_MARKERS = (
    "Error 500",
    "Server Error",
    "That's an error",
    "There was an error",
    "Please try again later",
    "That’s an error",
)


def _looks_like_google_error(text: str) -> bool:
    """Return True when a response looks like Google's error page rather than a translation."""
    if not text:
        return True
    normalized = text.strip().lower()
    return any(marker.lower() in normalized for marker in GOOGLE_ERROR_MARKERS)


def translate_text_blocks(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
) -> str:
    """Translate text with retries and protection against Google's error-page response."""
    if not text or not text.strip():
        return text

    last_error: Exception | None = None
    for attempt in range(1, TRANSLATION_RETRIES + 1):
        try:
            translated = GoogleTranslator(
                source=source_lang,
                target=target_lang,
            ).translate(text)

            if translated and not _looks_like_google_error(translated):
                return translated

            last_error = RuntimeError(
                "Google Translate returned an empty response or an error-page response"
            )
            print(
                f"번역 응답 이상 (시도 {attempt}/{TRANSLATION_RETRIES}): "
                f"{str(translated)[:120]!r}"
            )
        except Exception as exc:
            last_error = exc
            print(
                f"번역 요청 실패 (시도 {attempt}/{TRANSLATION_RETRIES}): {exc}"
            )

        if attempt < TRANSLATION_RETRIES:
            time.sleep(TRANSLATION_RETRY_DELAY * attempt)

    print(f"번역 최종 실패, 원문 유지: {last_error}")
    return text


def _get_span_info(page: fitz.Page) -> list[dict]:
    """Extract PDF lines while retaining original span and layout metadata."""
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


def _get_image_occurrences(page: fitz.Page) -> list[dict]:
    """Return every visible image occurrence and its original rectangle."""
    result: list[dict] = []
    try:
        for info in page.get_image_info(hashes=False, xrefs=True):
            bbox = info.get("bbox")
            xref = info.get("xref")
            if bbox and xref:
                result.append({"bbox": fitz.Rect(bbox), "xref": int(xref)})
    except Exception as exc:
        print(f"이미지 정보 추출 실패: {exc}")
    return result


def _copy_images_to_page(
    source_doc: fitz.Document,
    output_page: fitz.Page,
    image_occurrences: list[dict],
) -> int:
    """Copy source images onto the blank output page at their original positions."""
    copied = 0
    for occurrence in image_occurrences:
        rect = occurrence["bbox"]
        xref = occurrence["xref"]
        try:
            image = source_doc.extract_image(xref)
            image_bytes = image.get("image")
            if not image_bytes:
                continue

            # Images are placed first. Text is inserted afterwards, so text that
            # originally sat over an image remains visible on top of that image.
            output_page.insert_image(rect, stream=image_bytes, overlay=False)
            copied += 1
        except Exception as exc:
            print(f"이미지 복사 실패 (xref={xref}, bbox={rect}): {exc}")
    return copied


def _insert_translation(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    original_font_size: float,
) -> bool:
    """Insert translated text onto the new page, with a transparent background."""
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
            overlay=True,
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
    """Create blank output pages, copy only images, then place translated text.

    The source PDF is used only for extracting text coordinates and image objects.
    No source page is rendered, redacted, or copied as a whole. Each output page is
    blank, source images are placed first, and translated text is placed transparently
    on top. Image-overlap and image-nearby text is therefore never skipped.
    """
    del debug_grouping  # grouping is intentionally no longer used on this branch

    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Noto Sans KR font not found: {FONT_PATH}\n"
            "Put NotoSansKR-Regular.ttf in the fonts folder."
        )

    source_doc = fitz.open(input_pdf_path)
    output_doc = fitz.open()
    translated_count = 0
    image_count = 0

    try:
        for page_number, source_page in enumerate(source_doc, start=1):
            lines = _get_span_info(source_page)
            image_occurrences = _get_image_occurrences(source_page)

            output_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )

            # 1. Start from a genuinely blank page.
            # 2. Copy only the original images.
            # 3. Add translated text afterwards, so text can safely overlap images.
            copied_images = _copy_images_to_page(
                source_doc,
                output_page,
                image_occurrences,
            )
            image_count += copied_images

            print(
                f"페이지 {page_number}: {len(lines)} lines, "
                f"{copied_images}/{len(image_occurrences)} images"
            )

            for line_index, item in enumerate(lines):
                text = item["text"].strip()
                if not text or len(text) < 2:
                    continue

                rect = fitz.Rect(item["bbox"])
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

                print(f"라인 {line_index}")
                print("원문:", text[:120])
                print("번역:", translated[:120])
                print("삽입 결과:", inserted)
                print("bbox:", rect)

                if inserted:
                    translated_count += 1

        output_doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        output_doc.close()
        source_doc.close()

    print("복사한 이미지:", image_count)
    print("번역하여 삽입한 텍스트:", translated_count)
    print("이미지 위/근처 텍스트도 스킵하지 않고 번역했습니다.")
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
