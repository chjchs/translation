from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
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
MAX_TRANSLATION_RETRIES = 4


def _openai_request(messages: list[dict], timeout: int = 120) -> dict | None:
    """Make one OpenAI request with exponential backoff for HTTP 429."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set")
        return None

    body = {
        "model": OPENAI_TRANSLATION_MODEL,
        "temperature": 0,
        "messages": messages,
    }

    for attempt in range(MAX_TRANSLATION_RETRIES + 1):
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
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code != 429 or attempt >= MAX_TRANSLATION_RETRIES:
                print(f"OpenAI request failed ({exc.code}): {detail[:500]}")
                return None

            retry_after = exc.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                wait = 2 ** attempt
            wait = min(max(wait, 1.0), 30.0)
            print(f"OpenAI rate limit (429). Retrying in {wait:.1f}s ({attempt + 1}/{MAX_TRANSLATION_RETRIES})")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= MAX_TRANSLATION_RETRIES:
                print(f"OpenAI request failed: {exc}")
                return None
            wait = min(2 ** attempt, 16)
            print(f"OpenAI network error. Retrying in {wait}s ({attempt + 1}/{MAX_TRANSLATION_RETRIES})")
            time.sleep(wait)
        except Exception as exc:
            print(f"OpenAI request failed: {exc}")
            return None
    return None


def _openai_translation(text: str, source_lang: str, target_lang: str) -> str | None:
    """Single-group translation helper kept for compatibility; no Google fallback."""
    if not text.strip():
        return text
    result = _openai_request([
        {"role": "system", "content": "You are a precise PDF translation assistant."},
        {"role": "user", "content": (
            f"Translate the following text from {source_lang} to {target_lang}. "
            "Preserve paragraph meaning and line structure where useful. "
            "Return only the translation.\n\n" + text
        )},
    ])
    if not result:
        return None
    try:
        return result["choices"][0]["message"]["content"].strip() or None
    except (KeyError, TypeError):
        return None


def translate_text_blocks(text: str, source_lang: str = "auto", target_lang: str = "ko") -> str:
    if not text or not text.strip():
        return text
    translated = _openai_translation(text, source_lang, target_lang)
    return translated if translated is not None else text


def _get_span_info(page: fitz.Page) -> list[dict]:
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


def _build_batch_payload(groups: list[dict]) -> tuple[str, dict[int, list[dict]]]:
    """Build one request for a page and retain the original span mapping locally."""
    group_payload = []
    marker_map: dict[int, list[dict]] = {}
    for group_id, group in enumerate(groups):
        spans = group.get("spans", [])
        marker_map[group_id] = spans
        tagged = "".join(
            f"<s{i}>{html.escape(str(span.get('text', '')))}</s{i}>"
            for i, span in enumerate(spans)
        )
        group_payload.append({"id": group_id, "text": tagged})

    return json.dumps(group_payload, ensure_ascii=False), marker_map


def _translate_groups_batch(
    groups: list[dict], source_lang: str, target_lang: str
) -> dict[int, list[tuple[str, dict]]]:
    """Translate all groups on one page with one OpenAI request."""
    if not groups or not os.getenv("OPENAI_API_KEY", "").strip():
        return {}

    payload, marker_map = _build_batch_payload(groups)
    system = (
        "You translate PDF text groups. Return valid JSON only in this exact shape: "
        '{"translations":[{"id":0,"text":"<s0>...</s0>..."}]}. '
        "Translate each group independently but use the supplied context. "
        "Preserve every <sN> tag exactly once and in the same order within its group. "
        "Do not add, remove, merge, or rename tags. Return every supplied group id."
    )
    user = (
        f"Translate these logical PDF text groups from {source_lang} to {target_lang}. "
        "Do not translate the XML-like marker names themselves.\n\n" + payload
    )
    result = _openai_request([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    if not result:
        return {}

    try:
        content = result["choices"][0]["message"]["content"]
        data = json.loads(content)
        items = data["translations"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        print("Batch AI translation returned invalid JSON; page will be skipped")
        return {}

    output: dict[int, list[tuple[str, dict]]] = {}
    for item in items:
        try:
            group_id = int(item["id"])
            translated = str(item["text"])
        except (KeyError, TypeError, ValueError):
            continue
        if group_id not in marker_map:
            continue
        spans = marker_map[group_id]
        matches = re.findall(r"<s(\d+)>(.*?)</s\1>", translated, flags=re.DOTALL)
        expected = list(range(len(spans)))
        actual = [int(index) for index, _ in matches]
        if actual != expected:
            print(f"Invalid style markers for group {group_id}; group kept unchanged")
            continue
        output[group_id] = [(text, spans[int(index)]) for index, text in matches]
    return output


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
    if use_ai_grouping and not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required when AI grouping/translation is enabled")

    doc = fitz.open(input_pdf_path)
    translated_count = 0
    skipped_image_count = 0
    try:
        for page_number, page in enumerate(doc, start=1):
            lines = _get_span_info(page)
            groups = _group_lines(page, lines, use_ai_grouping)
            image_rects = _get_image_rects(page)
            eligible_groups: list[tuple[int, dict]] = []

            for group_id, group in enumerate(groups):
                text = group["text"].strip()
                if not text or len(text) < 2:
                    continue
                rect = fitz.Rect(group["bbox"])
                if _overlaps_image(rect, image_rects):
                    skipped_image_count += 1
                    print(f"이미지 겹침으로 건너뜀 (페이지 {page_number}): {text[:80]}")
                    continue
                eligible_groups.append((group_id, group))

            # One translation request per page instead of one request per group.
            styled_by_id = _translate_groups_batch(
                [group for _, group in eligible_groups], source_lang, target_lang
            ) if use_ai_grouping else {}

            replacements: list[tuple[fitz.Rect, str, float, list[tuple[str, dict]] | None]] = []
            for local_id, (original_group_id, group) in enumerate(eligible_groups):
                text = group["text"].strip()
                styled = styled_by_id.get(local_id)
                if not styled:
                    print(f"페이지 {page_number}, group {original_group_id}: translation unavailable; keeping original")
                    continue
                translated = "".join(segment for segment, _ in styled)
                if not translated or translated == text:
                    continue
                replacements.append((fitz.Rect(group["bbox"]), translated, group["size"], styled))
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
