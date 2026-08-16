from __future__ import annotations

import html
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import fitz

from ai_grouping import group_page

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"
BULLET_CHARS = "•●○◦▪▫■□◆◇"
IMAGE_OVERLAP_TOLERANCE = 0.5
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_TRANSLATION_MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini")


def _openai_translation(text: str, source_lang: str, target_lang: str) -> str | None:
    """Translate one logical group with OpenAI. Never falls back to Google."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not text.strip():
        return None

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        "This is one logical PDF text group. Preserve paragraph meaning and line "
        "structure where useful. Return only the translation.\n\n" + text
    )
    body = {
        "model": OPENAI_TRANSLATION_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a precise PDF translation assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        translated = result["choices"][0]["message"]["content"].strip()
        return translated or None
    except Exception as exc:
        print(f"AI translation failed; keeping original text: {exc}")
        return None


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    """Translate one logical text group with OpenAI only."""
    if not text or not text.strip():
        return text
    translated = _openai_translation(text, source_lang, target_lang)
    return translated if translated is not None else text


def _get_span_info(page: fitz.Page) -> list[dict]:
    """Extract lines while retaining every original span and PDF block identity."""
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
            result.append({
                "text": text,
                "bbox": fitz.Rect(line["bbox"]),
                "size": max(float(s.get("size", 0)) for s in spans),
                "spans": spans,
                "block_index": block_index,
                "line_index": line_index,
            })
    return result


def _get_image_rects(page: fitz.Page) -> list[fitz.Rect]:
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


def _is_bullet_only(text: str) -> bool:
    return text.strip() in BULLET_CHARS


def _local_group_lines(lines: list[dict]) -> list[dict]:
    """Safe non-AI grouping: keep PDF lines separate except bullet + following line."""
    groups: list[dict] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        if _is_bullet_only(current["text"]) and i + 1 < len(lines):
            nxt = lines[i + 1]
            vertical_gap = nxt["bbox"].y0 - current["bbox"].y1
            height = max(current["bbox"].height, nxt["bbox"].height, 1.0)
            if -0.5 * height <= vertical_gap <= 1.5 * height:
                groups.append(_make_group([current, nxt], source="local"))
                i += 2
                continue
        groups.append(_make_group([current], source="local"))
        i += 1
    return groups


def _make_group(lines: list[dict], source: str) -> dict:
    bbox = fitz.Rect(lines[0]["bbox"])
    spans: list[dict] = []
    for line in lines:
        bbox |= line["bbox"]
        spans.extend(line.get("spans", []))
    return {
        "text": "\n".join(line["text"] for line in lines),
        "bbox": bbox,
        "size": max(float(line.get("size", 12)) for line in lines),
        "spans": spans,
        "lines": lines,
        "group_source": source,
    }


def _groups_from_ai(lines: list[dict], ids: list[list[int]]) -> list[dict]:
    return [_make_group([lines[i] for i in group], source="ai") for group in ids]


def _group_lines(page: fitz.Page, lines: list[dict], use_ai: bool) -> list[dict]:
    if use_ai:
        ai_ids = group_page(page, lines)
        if ai_ids:
            return _groups_from_ai(lines, ai_ids)
    return _local_group_lines(lines)


def _span_style(span: dict) -> tuple[str, bool, float]:
    color = int(span.get("color", 0) or 0)
    r = (color >> 16) & 255
    g = (color >> 8) & 255
    b = color & 255
    flags = int(span.get("flags", 0) or 0)
    bold = bool(flags & 16) or "bold" in str(span.get("font", "")).lower()
    size = max(4.0, float(span.get("size", 12) or 12))
    return f"#{r:02x}{g:02x}{b:02x}", bold, size


def _translate_group_with_markers(group: dict, source_lang: str, target_lang: str) -> list[tuple[str, dict]] | None:
    """Translate one logical group once and preserve original span markers."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    spans = group.get("spans", [])
    if not api_key or not spans:
        return None

    pieces = [f"<s{i}>{span.get('text', '')}</s{i}>" for i, span in enumerate(spans)]
    source = "".join(pieces)
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        "This is ONE translation unit. Preserve every <sN>...</sN> tag exactly "
        "once and in the same order. Do not add, remove, merge, or rename tags. "
        "Return only the translated tagged text.\n\n" + source
    )
    body = {
        "model": OPENAI_TRANSLATION_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You translate PDF text while preserving inline style markers."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        translated = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"AI styled translation failed; keeping original group: {exc}")
        return None

    matches = re.findall(r"<s(\d+)>(.*?)</s\1>", translated, flags=re.DOTALL)
    if len(matches) != len(spans):
        print("AI translation markers invalid; keeping original group")
        return None
    return [(text, spans[int(index)]) for index, text in matches]


def _insert_styled_html(page: fitz.Page, rect: fitz.Rect, segments: list[tuple[str, dict]]) -> bool:
    archive = fitz.Archive()
    try:
        archive.add(str(FONT_PATH), path="fonts/NotoSansKR-Regular.ttf")
        body = []
        for text, original_span in segments:
            color, bold, size = _span_style(original_span)
            weight = "700" if bold else "400"
            body.append(
                f'<span style="color:{color};font-size:{size}pt;font-weight:{weight};">'
                f"{html.escape(text).replace(chr(10), '<br>')}"
                "</span>"
            )
        css = (
            "@font-face { font-family: NotoSansKR; src: url('fonts/NotoSansKR-Regular.ttf'); }"
            "body { margin: 0; padding: 0; font-family: NotoSansKR; line-height: 1.05; }"
        )
        result, scale = page.insert_htmlbox(rect, "<body>" + "".join(body) + "</body>", css=css, archive=archive)
        print("삽입 결과:", result, "scale:", scale)
        return result >= 0
    finally:
        archive = None


def _insert_translation(page: fitz.Page, rect: fitz.Rect, text: str, original_font_size: float) -> bool:
    fontsize = max(4.0, original_font_size)
    while fontsize >= 4:
        result = page.insert_textbox(
            rect, text, fontsize=fontsize, fontname=FONT_NAME,
            fontfile=str(FONT_PATH), color=(0, 0, 0), align=fitz.TEXT_ALIGN_LEFT,
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
    use_ai_grouping: bool = True,
) -> int:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Noto Sans KR font not found: {FONT_PATH}")

    doc = fitz.open(input_pdf_path)
    translated_count = 0
    skipped_image_count = 0
    try:
        for page_number, page in enumerate(doc, start=1):
            lines = _get_span_info(page)
            groups = _group_lines(page, lines, use_ai_grouping)
            image_rects = _get_image_rects(page)
            replacements: list[tuple[fitz.Rect, str, float, list[tuple[str, dict]] | None]] = []

            for group in groups:
                text = group["text"].strip()
                if not text or len(text) < 2:
                    continue
                rect = fitz.Rect(group["bbox"])
                if _overlaps_image(rect, image_rects):
                    skipped_image_count += 1
                    print(f"이미지 겹침으로 건너뜀 (페이지 {page_number}): {text[:80]}")
                    continue

                styled = _translate_group_with_markers(group, source_lang, target_lang) if use_ai_grouping else None
                if styled:
                    translated = "".join(segment for segment, _ in styled)
                else:
                    translated = _openai_translation(text, source_lang, target_lang) if use_ai_grouping else text
                    if translated is None:
                        # No translation fallback is used. Preserve the original text.
                        continue

                if translated == text:
                    continue
                replacements.append((rect, translated, group["size"], styled))
                print("원문:", text[:120])
                print("번역:", translated[:120])
                print("group source:", group.get("group_source"))

            for rect, _, _, _ in replacements:
                page.add_redact_annot(rect, fill=False, cross_out=False)
            if replacements:
                page.apply_redactions(images=0, graphics=0, text=0)

            for rect, translated, original_size, styled in replacements:
                inserted = _insert_styled_html(page, rect, styled) if styled else _insert_translation(page, rect, translated, original_size)
                if inserted:
                    translated_count += 1

        doc.save(output_pdf_path, garbage=2, deflate=True)
    finally:
        doc.close()
    print("이미지와 겹쳐 건너뛴 텍스트:", skipped_image_count)
    return translated_count


def iter_pdf_text_blocks(pdf_path: str) -> Iterable[tuple[float, float, float, float, str]]:
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for block in page.get_text("blocks"):
                yield block[0], block[1], block[2], block[3], (block[4] or "").strip()
    finally:
        doc.close()