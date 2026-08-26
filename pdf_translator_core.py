from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import deepl
import fitz
from grouping_engine import group_page

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_PATH = FONT_DIR / "NotoSansKR-Regular.ttf"
FONT_BOLD_PATH = FONT_DIR / "NotoSansKR-Bold.ttf"
FONT_NAME = "NotoSansKR"
FONT_BOLD_NAME = "NotoSansKRBold"
RETRIES = 3
TRANSLATION_DELAY = 1
RETRY_BACKOFF = 3


def _deepl_target_language(target_lang: str) -> str:
    mapping = {"ko": "KO", "en": "EN-US", "ja": "JA", "zh": "ZH", "zh-cn": "ZH", "fr": "FR", "de": "DE", "es": "ES", "it": "IT", "pt": "PT-PT", "pt-br": "PT-BR", "ru": "RU"}
    return mapping.get(target_lang.lower(), target_lang.upper())


def _deepl_source_language(source_lang: str) -> str | None:
    if not source_lang or source_lang.lower() == "auto":
        return None
    mapping = {"ko": "KO", "en": "EN", "ja": "JA", "zh": "ZH", "zh-cn": "ZH", "fr": "FR", "de": "DE", "es": "ES", "it": "IT", "pt": "PT", "ru": "RU"}
    return mapping.get(source_lang.lower(), source_lang.upper())


def _escape_xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _span_style_flags(span: dict[str, Any]) -> tuple[bool, bool]:
    flags = int(span.get("flags", 0) or 0)
    font = str(span.get("font", "")).lower()
    return bool(flags & 16) or "bold" in font, bool(flags & 2) or "italic" in font or "oblique" in font


def _build_tagged_text(group: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    span_meta: dict[str, dict[str, Any]] = {}
    chunks: list[str] = []
    token = uuid.uuid4().hex[:10]
    span_index = 0

    for line_index, line in enumerate(group.get("lines", [])):
        if line_index:
            chunks.append("\n")
        for span in line.get("spans", []):
            raw = str(span.get("text", ""))
            if not raw:
                continue
            key = f"s{span_index}"
            span_meta[key] = span
            bold, italic = _span_style_flags(span)
            opening = ("<bold>" if bold else "") + ("<italic>" if italic else "")
            closing = ("</italic>" if italic else "") + ("</bold>" if bold else "")
            marker = f"[[MAP_{token}_{span_index}]]"
            chunks.append(opening + _escape_xml_text(raw) + closing + marker)
            span_index += 1

    return "".join(chunks), span_meta


def _strip_format_tags(text: str) -> str:
    return re.sub(r"</?(?:bold|italic)(?:\s+[^>]*)?>", "", text)


def _strip_mapping_markers(text: str) -> str:
    return re.sub(r"\[\[MAP_[A-Za-z0-9]+_\d+\]\]", "", text)


def _normalize_boundary_whitespace(text: str) -> str:
    """Normalize whitespace around formatting tags without variable-length lookbehind."""
    text = re.sub(r"\s+(?=<\s*/?(?:bold|italic)\b)", "", text)
    text = re.sub(r"(</(?:bold|italic)>)\s+", r"\1 ", text)
    text = re.sub(r"(<(?:bold|italic)>)[ \t\u00a0]+", r"\1", text)
    text = re.sub(r"[ \t\u00a0]+(</(?:bold|italic)>)", r"\1", text)
    return text


def _clean_piece(text: str) -> str:
    text = _normalize_boundary_whitespace(text)
    text = _strip_format_tags(text)
    text = _strip_mapping_markers(text)
    return text


def _parse_tagged_translation(value: str, span_meta: dict[str, dict[str, Any]]) -> dict[str, str]:
    matches = list(re.finditer(r"\[\[MAP_([A-Za-z0-9]+)_(\d+)\]\]", value))
    if not matches:
        raise ValueError("DeepL removed all mapping markers")

    result: dict[str, str] = {}
    expected_token = matches[0].group(1)
    previous_end = 0
    for match in matches:
        if match.group(1) != expected_token:
            continue
        index = int(match.group(2))
        key = f"s{index}"
        if key not in span_meta:
            continue
        piece = value[previous_end:match.start()]
        piece = _clean_piece(piece)
        if index == 0:
            piece = piece.lstrip("\n")
        result[key] = piece
        previous_end = match.end()

    if len(result) != len(span_meta):
        missing = [k for k in span_meta if k not in result]
        raise ValueError(f"DeepL did not preserve all mapping markers: {missing}")
    return result


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko", tagged: bool = False) -> tuple[str, bool]:
    if not text.strip():
        return text, False
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY environment variable is not set")
    translator = deepl.DeepLClient(api_key)
    kwargs: dict[str, Any] = {"source_lang": _deepl_source_language(source_lang), "target_lang": _deepl_target_language(target_lang), "preserve_formatting": True}
    if tagged:
        kwargs["tag_handling"] = "xml"
        kwargs["outline_detection"] = False
    for attempt in range(RETRIES):
        try:
            result = translator.translate_text(text, **kwargs)
            translated = str(result.text).strip()
            if translated:
                return translated, True
            raise RuntimeError("DeepL returned an empty response")
        except Exception as exc:
            print(f"번역 group 실패 (시도 {attempt + 1}/{RETRIES}): {exc}")
            if attempt < RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    return text, False


def _get_span_info(page: fitz.Page) -> list[dict[str, Any]]:
    result = []
    data = page.get_text("dict")
    for bi, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for li, line in enumerate(block.get("lines", [])):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(str(s.get("text", "")) for s in spans).strip()
            if not text:
                continue
            dominant = max(spans, key=lambda s: float(s.get("size", 12) or 12))
            result.append({"text": text, "bbox": fitz.Rect(line["bbox"]), "size": float(dominant.get("size", 12) or 12), "spans": spans, "block_index": bi, "line_index": li, "dir": tuple(line.get("dir", (1.0, 0.0)))})
    return result


def _images(page: fitz.Page) -> list[dict[str, Any]]:
    result = []
    try:
        for info in page.get_image_info(hashes=False, xrefs=True):
            if info.get("bbox") and info.get("xref"):
                result.append({"bbox": fitz.Rect(info["bbox"]), "xref": int(info["xref"])})
    except Exception as exc:
        print(f"이미지 정보 추출 실패: {exc}")
    return result


def _copy_images(doc: fitz.Document, page: fitz.Page, images: list[dict[str, Any]]) -> int:
    count = 0
    for item in images:
        try:
            data = doc.extract_image(item["xref"]).get("image")
            if data:
                page.insert_image(item["bbox"], stream=data, overlay=False)
                count += 1
        except Exception as exc:
            print(f"이미지 복사 실패: {exc}")
    return count


def _span_style(span: dict[str, Any]) -> tuple[str, tuple[float, float, float], bool, bool]:
    flags = int(span.get("flags", 0) or 0)
    font = str(span.get("font", "")).lower()
    bold = bool(flags & 16) or "bold" in font
    italic = bool(flags & 2) or "italic" in font or "oblique" in font
    color_value = int(span.get("color", 0) or 0)
    try:
        rgb = fitz.sRGB_to_rgb(color_value)
        color = tuple(v / 255 for v in rgb)
    except Exception:
        color = (0.0, 0.0, 0.0)
    return (FONT_BOLD_NAME if bold else FONT_NAME, color, bold, italic)


def _style(group: dict[str, Any]) -> dict[str, Any]:
    spans = group.get("spans", [])
    if not spans:
        return {"size": 12.0}
    s = max(spans, key=lambda x: float(x.get("size", 12) or 12))
    return {"size": float(s.get("size", 12) or 12)}


def _align(group: dict[str, Any], page_rect: fitz.Rect) -> int:
    if group.get("direction") == "rtl":
        return fitz.TEXT_ALIGN_RIGHT
    r = fitz.Rect(group["bbox"])
    left, right = r.x0 - page_rect.x0, page_rect.x1 - r.x1
    if abs(left - right) <= max(6.0, r.height * .8):
        return fitz.TEXT_ALIGN_CENTER
    return fitz.TEXT_ALIGN_LEFT


def _insert_span(page: fitz.Page, span: dict[str, Any], text: str) -> bool:
    if not text:
        return True
    rect = fitz.Rect(span["bbox"])
    size = max(4.0, float(span.get("size", 12) or 12))
    font_name, color, _, _ = _span_style(span)
    while size >= 4:
        result = page.insert_textbox(rect, text, fontsize=size, fontname=font_name, fontfile=str(FONT_BOLD_PATH if font_name == FONT_BOLD_NAME else FONT_PATH), color=color, align=fitz.TEXT_ALIGN_LEFT, overlay=True)
        if result >= 0:
            return True
        size -= .5
    return False


def _insert_group(page: fitz.Page, group: dict[str, Any], translated: str, tagged_translation: bool = False) -> bool:
    if tagged_translation:
        try:
            _, span_meta = _build_tagged_text(group)
            translated_by_span = _parse_tagged_translation(translated, span_meta)
            inserted_any = False
            for key, span in span_meta.items():
                inserted_any = _insert_span(page, span, translated_by_span[key]) or inserted_any
            return inserted_any
        except Exception as exc:
            print(f"서식 매핑 실패, 마커 제거 후 그룹 단위 삽입으로 전환: {exc}")
            clean_text = _clean_piece(translated)
    else:
        clean_text = _clean_piece(translated)

    style = _style(group)
    rect = fitz.Rect(group["bbox"])
    size = max(4.0, style["size"])
    while size >= 4:
        result = page.insert_textbox(rect, clean_text, fontsize=size, fontname=FONT_NAME, fontfile=str(FONT_PATH), color=(0, 0, 0), align=_align(group, page.rect), overlay=True)
        if result >= 0:
            return True
        size -= .5
    return False


def translate_pdf_file(input_pdf_path: str, output_pdf_path: str, source_lang: str = "auto", target_lang: str = "ko", debug_grouping: bool = False) -> int:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Noto Sans KR font not found: {FONT_PATH}")
    source_doc = fitz.open(input_pdf_path)
    output_doc = fitz.open()
    translated_count = total_groups = image_count = 0
    try:
        for page_number, source_page in enumerate(source_doc, start=1):
            lines = _get_span_info(source_page)
            groups = group_page(source_page, lines, debug=debug_grouping)
            total_groups += len(groups)
            translated_page = output_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
            image_count += _copy_images(source_doc, translated_page, _images(source_page))
            print(f"페이지 {page_number}: {len(lines)} lines -> {len(groups)} groups")
            request_made = False
            for group_index, group in enumerate(groups):
                text = str(group.get("text", "")).strip()
                if len(text) < 2:
                    continue
                if request_made:
                    time.sleep(TRANSLATION_DELAY)
                tagged_text, _ = _build_tagged_text(group)
                translated, success = translate_text_blocks(tagged_text, source_lang, target_lang, tagged=True)
                request_made = True
                if success:
                    inserted = _insert_group(translated_page, group, translated, tagged_translation=True)
                    display_translation = _clean_piece(translated)[:200]
                else:
                    print("번역 실패, 원문 그대로 삽입:", text[:200])
                    clean_original = _clean_piece(tagged_text)
                    inserted = _insert_group(translated_page, group, clean_original, tagged_translation=False)
                    display_translation = text[:200]
                print(f"그룹 {group_index} ({group.get('group_type')}): {len(group.get('lines', []))} lines")
                print("원문:", text[:200])
                print("번역:", display_translation)
                print("삽입 결과:", inserted, "direction:", group.get("direction"), "bbox:", group.get("bbox"))
                if inserted:
                    translated_count += 1
            original_page = output_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
            original_page.show_pdf_page(original_page.rect, source_doc, source_page.number)
        output_doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        output_doc.close()
        source_doc.close()
    print("총 그룹:", total_groups)
    print("번역하여 삽입한 그룹:", translated_count)
    print("복사한 이미지:", image_count)
    print("페이지 순서: [번역 페이지, 원본 페이지]")
    return translated_count


def iter_pdf_text_blocks(pdf_path: str) -> Iterable[tuple[float, float, float, float, str]]:
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for block in page.get_text("blocks"):
                yield block[0], block[1], block[2], block[3], (block[4] or "").strip()
    finally:
        doc.close()
