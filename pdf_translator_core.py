from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"

# Keep the number of requests low and give Google time to recover from transient
# server/rate-limit responses.
TRANSLATION_BATCH_SIZE = 10
TRANSLATION_RETRIES = 3
TRANSLATION_RETRY_DELAYS = (10.0, 30.0, 60.0)
BATCH_DELAY = 1.0
GOOGLE_ERROR_MARKERS = (
    "Error 500",
    "Server Error",
    "That's an error",
    "There was an error",
    "Please try again later",
    "That’s an error",
)


def _looks_like_google_error(text: str) -> bool:
    if not text:
        return True
    normalized = str(text).strip().lower()
    return any(marker.lower() in normalized for marker in GOOGLE_ERROR_MARKERS)


def _translate_batch_once(
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[str]:
    """Translate a batch while preserving one result per input item."""
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    results = translator.translate_batch(texts)
    if not isinstance(results, list) or len(results) != len(texts):
        raise RuntimeError(
            f"Google Translate returned an unexpected batch response: "
            f"expected {len(texts)}, got {len(results) if isinstance(results, list) else type(results).__name__}"
        )

    cleaned: list[str] = []
    for original, translated in zip(texts, results):
        if not translated or _looks_like_google_error(str(translated)):
            raise RuntimeError("Google Translate returned an empty response or an error-page response")
        cleaned.append(str(translated))
    return cleaned


def _translate_batch_with_retry(
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[str] | None:
    """Retry a whole batch with exponential backoff; return None after final failure."""
    for attempt in range(1, TRANSLATION_RETRIES + 1):
        try:
            return _translate_batch_once(texts, source_lang, target_lang)
        except Exception as exc:
            print(
                f"번역 batch 실패 (시도 {attempt}/{TRANSLATION_RETRIES}, "
                f"{len(texts)}개): {exc}"
            )
            if attempt < TRANSLATION_RETRIES:
                delay = TRANSLATION_RETRY_DELAYS[attempt - 1]
                print(f"{delay:.0f}초 후 batch 재시도합니다.")
                time.sleep(delay)
    return None


def translate_text_blocks(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
) -> str:
    """Backward-compatible single-text translation helper."""
    if not text or not text.strip():
        return text
    result = _translate_batch_with_retry([text], source_lang, target_lang)
    return result[0] if result else text


def _translate_lines(
    lines: list[dict],
    source_lang: str,
    target_lang: str,
) -> list[str]:
    """Translate independent PDF lines in batches without grouping their layout."""
    translations = [item["text"] for item in lines]

    for start in range(0, len(lines), TRANSLATION_BATCH_SIZE):
        end = min(start + TRANSLATION_BATCH_SIZE, len(lines))
        batch = [lines[i]["text"] for i in range(start, end)]
        result = _translate_batch_with_retry(batch, source_lang, target_lang)

        if result is None:
            # If a whole batch fails, split it recursively. This isolates a bad
            # request and prevents one transient failure from losing the page.
            if len(batch) > 1:
                midpoint = len(batch) // 2
                left = _translate_batch_with_retry(batch[:midpoint], source_lang, target_lang)
                right = _translate_batch_with_retry(batch[midpoint:], source_lang, target_lang)
                if left is not None:
                    translations[start:start + midpoint] = left
                if right is not None:
                    translations[start + midpoint:end] = right
            continue

        translations[start:end] = result
        if end < len(lines):
            time.sleep(BATCH_DELAY)

    return translations


def _get_span_info(page: fitz.Page) -> list[dict]:
    """Extract PDF lines while retaining original line coordinates and font metadata."""
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
    """Copy source images onto the blank translated page at original positions."""
    copied = 0
    for occurrence in image_occurrences:
        rect = occurrence["bbox"]
        xref = occurrence["xref"]
        try:
            image = source_doc.extract_image(xref)
            image_bytes = image.get("image")
            if not image_bytes:
                continue
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
    """Insert translated text with a transparent background."""
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
    """Write [translated page, original page] for every source page.

    The translated page starts completely blank. Images are copied first and
    translated text is then placed transparently on top. The original source
    page is retained immediately after it for reference.
    """
    del debug_grouping  # grouping is intentionally not used on this branch

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

            # Page A: genuinely blank page containing copied images + translations.
            translated_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            copied_images = _copy_images_to_page(
                source_doc,
                translated_page,
                image_occurrences,
            )
            image_count += copied_images

            translations = _translate_lines(lines, source_lang, target_lang)

            print(
                f"페이지 {page_number}: {len(lines)} lines, "
                f"{copied_images}/{len(image_occurrences)} images"
            )

            for line_index, (item, translated) in enumerate(zip(lines, translations)):
                text = item["text"].strip()
                if not text or len(text) < 2:
                    continue

                # If translation failed, retain no text on the translated page;
                # the original page immediately following it remains available.
                if translated == text:
                    continue

                rect = fitz.Rect(item["bbox"])
                original_font_size = max(4.0, float(item.get("size", 12.0)))
                inserted = _insert_translation(
                    translated_page,
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

            # Page B: retain the complete original page for direct comparison.
            # show_pdf_page copies the original page without rasterizing it.
            original_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            original_page.show_pdf_page(original_page.rect, source_doc, page_number - 1)

            print(f"페이지 {page_number}: 번역 페이지 + 원본 페이지 추가 완료")

        output_doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        output_doc.close()
        source_doc.close()

    print("복사한 이미지:", image_count)
    print("번역하여 삽입한 텍스트:", translated_count)
    print("페이지 순서: [번역 페이지, 원본 페이지] × 전체 페이지")
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
