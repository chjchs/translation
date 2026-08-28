from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

import deepl
import fitz
from grouping_engine import group_page

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_PATH = FONT_DIR / "NotoSansKR-Regular.ttf"
FONT_BOLD_PATH = FONT_DIR / "NotoSansKR-Bold.ttf"
FONT_NAME = "NotoSansKR"

KOREAN = r"가-힣"
# Match inline English(Korean) hints robustly across normal spaces, NBSPs,
# zero-width formatting characters and spaces around the parentheses.
HINT_PATTERN = re.compile(
    rf"(?P<source>[A-Za-z][A-Za-z0-9+./'’\-]*(?:[\s\u00a0\u200b]+[A-Za-z][A-Za-z0-9+./'’\-]*){{0,15}})"
    rf"[\s\u00a0\u200b]*\([\s\u00a0\u200b]*"
    rf"(?P<target>[{KOREAN}][{KOREAN}0-9·\-]*(?:[\s\u00a0\u200b]+[{KOREAN}][{KOREAN}0-9·\-]*){{0,12}})"
    rf"[\s\u00a0\u200b]*[,)"
)


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s\u00a0\u200b]+", " ", text).strip()


def _target_lang(lang: str) -> str:
    mapping = {"ko": "KO", "en": "EN-US", "ja": "JA", "zh": "ZH", "zh-cn": "ZH", "fr": "FR", "de": "DE", "es": "ES", "it": "IT", "pt": "PT-PT", "pt-br": "PT-BR", "ru": "RU"}
    return mapping.get(lang.lower(), lang.upper())


def _source_lang(lang: str) -> str | None:
    if not lang or lang.lower() == "auto":
        return None
    mapping = {"ko": "KO", "en": "EN", "ja": "JA", "zh": "ZH", "zh-cn": "ZH", "fr": "FR", "de": "DE", "es": "ES", "it": "IT", "pt": "PT", "ru": "RU"}
    return mapping.get(lang.lower(), lang.upper())


def _span_flags(span: dict[str, Any]) -> tuple[bool, bool, bool]:
    flags = int(span.get("flags", 0) or 0)
    char_flags = int(span.get("char_flags", 0) or 0)
    font = str(span.get("font", "")).lower()
    return bool(flags & 16) or "bold" in font, bool(flags & 2) or "italic" in font or "oblique" in font, bool(char_flags & 2)


def _span_color(span: dict[str, Any]) -> str:
    try:
        rgb = fitz.sRGB_to_rgb(int(span.get("color", 0) or 0))
        return "#%02x%02x%02x" % rgb
    except Exception:
        return "#000000"


def _has_special_style(span: dict[str, Any]) -> bool:
    bold, italic, underline = _span_flags(span)
    color = _span_color(span)
    return bold or italic or underline or color.lower() != "#000000"


def _span_to_html(span: dict[str, Any]) -> str:
    text = html.escape(str(span.get("text", "")), quote=False)
    if not _has_special_style(span):
        return text
    bold, italic, underline = _span_flags(span)
    styles: list[str] = []
    color = _span_color(span)
    if color.lower() != "#000000":
        styles.append(f"color:{color}")
    if underline:
        styles.append("text-decoration:underline")
    result = text
    if styles:
        result = f'<span style="{";".join(styles)}">{result}</span>'
    if italic:
        result = f"<i>{result}</i>"
    if bold:
        result = f"<b>{result}</b>"
    return result


def _group_to_html(group: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in group.get("lines", []):
        parts = [_span_to_html(span) for span in line.get("spans", []) if span.get("text", "")]
        lines.append("".join(parts))
    return "<br>".join(lines)


def _plain_group_text(group: dict[str, Any]) -> str:
    return "\n".join("".join(str(span.get("text", "")) for span in line.get("spans", [])) for line in group.get("lines", [])).strip()


def _extract_local_translation_hints(text: str) -> list[tuple[str, str]]:
    """Extract English(Korean) hints from this group only."""
    normalized = _normalize_text(text)
    hints: list[tuple[str, str]] = []
    for match in HINT_PATTERN.finditer(normalized):
        source = _normalize_text(match.group("source"))
        target = _normalize_text(match.group("target"))
        if not source or not target or len(source) > 80 or len(target) > 60:
            continue
        pair = (source, target)
        if pair not in hints:
            hints.append(pair)
    return hints


def _build_local_context(text: str) -> str | None:
    hints = _extract_local_translation_hints(text)
    if not hints:
        return None
    return "Local translation hints from this text: " + "; ".join(f"{source} = {target}" for source, target in hints)


def _translate_html(html_text: str, source_lang: str, target_lang: str, context: str | None = None) -> str:
    key = os.getenv("DEEPL_API_KEY")
    if not key:
        raise RuntimeError("DEEPL_API_KEY environment variable is not set")
    translator = deepl.DeepLClient(key)
    kwargs: dict[str, Any] = {"target_lang": _target_lang(target_lang), "tag_handling": "html", "tag_handling_version": "v2", "preserve_formatting": True}
    source = _source_lang(source_lang)
    if source:
        kwargs["source_lang"] = source
    if context:
        kwargs["context"] = context
    return str(translator.translate_text(html_text, **kwargs).text).strip()


def _get_span_info(page: fitz.Page) -> list[dict[str, Any]]:
    collect_styles = getattr(fitz, "TEXT_COLLECT_STYLES", 32768)
    flags = getattr(fitz, "TEXTFLAGS_DICT", 0) | collect_styles
    data = page.get_text("dict", flags=flags) if flags else page.get_text("dict")
    result: list[dict[str, Any]] = []
    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = [s for s in line.get("spans", []) if str(s.get("text", "")).strip()]
            if not spans:
                continue
            text = "".join(str(s.get("text", "")) for s in spans).strip()
            if not text:
                continue
            dominant = max(spans, key=lambda s: float(s.get("size", 12) or 12))
            result.append({"text": text, "bbox": fitz.Rect(line["bbox"]), "size": float(dominant.get("size", 12) or 12), "spans": spans, "block_index": block_index, "line_index": line_index, "dir": tuple(line.get("dir", (1.0, 0.0)))})
    return result


def _images(page: fitz.Page) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
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
            image = doc.extract_image(item["xref"]).get("image")
            if image:
                page.insert_image(item["bbox"], stream=image, overlay=False)
                count += 1
        except Exception as exc:
            print(f"이미지 복사 실패: {exc}")
    return count


def _insert_html_group(page: fitz.Page, group: dict[str, Any], translated_html: str, archive: fitz.Archive) -> bool:
    rect = fitz.Rect(group["bbox"])
    dominant_size = max((float(s.get("size", 12) or 12) for s in group.get("spans", [])), default=12.0)
    css = f"""
    @font-face {{ font-family: {FONT_NAME}; src: url(NotoSansKR-Regular.ttf); }}
    @font-face {{ font-family: {FONT_NAME}; src: url(NotoSansKR-Bold.ttf); font-weight: bold; }}
    * {{ font-family: {FONT_NAME}; font-size: {dominant_size:g}pt; margin: 0; padding: 0; }}
    """
    size = dominant_size
    while size >= 4:
        adjusted_css = css.replace(f"font-size: {dominant_size:g}pt", f"font-size: {size:g}pt")
        try:
            result = page.insert_htmlbox(rect, translated_html, css=adjusted_css, archive=archive, scale_low=0.55, overlay=True)
            if result[0] >= 0:
                return True
        except Exception as exc:
            print(f"HTML 삽입 실패 (font={size:g}): {exc}")
        size -= 0.5
    return False


def translate_pdf_file(input_pdf_path: str, output_pdf_path: str, source_lang: str = "auto", target_lang: str = "ko", debug_grouping: bool = False) -> int:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Noto Sans KR font not found: {FONT_PATH}")
    if not FONT_BOLD_PATH.exists():
        raise FileNotFoundError(f"Noto Sans KR bold font not found: {FONT_BOLD_PATH}")
    source_doc = fitz.open(input_pdf_path)
    output_doc = fitz.open()
    archive = fitz.Archive(str(FONT_DIR))
    translated_count = 0
    total_groups = 0
    image_count = 0
    try:
        for page_number, source_page in enumerate(source_doc, start=1):
            lines = _get_span_info(source_page)
            groups = group_page(source_page, lines, debug=debug_grouping)
            total_groups += len(groups)
            translated_page = output_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
            image_count += _copy_images(source_doc, translated_page, _images(source_page))
            print(f"페이지 {page_number}: {len(lines)} lines -> {len(groups)} groups")
            for group_index, group in enumerate(groups):
                text = _plain_group_text(group)
                if len(text) < 2:
                    continue
                source_html = _group_to_html(group)
                local_context = _build_local_context(text)
                try:
                    translated_html = _translate_html(source_html, source_lang, target_lang, local_context)
                except Exception as exc:
                    print(f"번역 실패: {exc}")
                    translated_html = ""
                inserted = False
                if translated_html:
                    inserted = _insert_html_group(translated_page, group, translated_html, archive)
                    print("원문 HTML:", source_html[:300])
                    if local_context:
                        print("로컬 번역 힌트:", local_context)
                    print("번역 HTML:", translated_html[:300])
                if not inserted:
                    fallback = html.escape(text).replace("\n", "<br>")
                    inserted = _insert_html_group(translated_page, group, fallback, archive)
                    print("HTML 번역 삽입 실패 -> 원문 fallback")
                print(f"그룹 {group_index} ({group.get('group_type')}): {len(group.get('lines', []))} lines, inserted={inserted}")
                print("원문:", text[:200])
                print("삽입 위치:", group.get("bbox"))
                if inserted:
                    translated_count += 1
            original_page = output_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
            original_page.show_pdf_page(original_page.rect, source_doc, source_page.number)
        output_doc.subset_fonts(verbose=False)
        output_doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
    finally:
        output_doc.close()
        source_doc.close()
    print("총 그룹:", total_groups)
    print("번역하여 삽입한 그룹:", translated_count)
    print("복사한 이미지:", image_count)
    print("페이지 순서: [번역 페이지, 원본 페이지]")
    return translated_count
