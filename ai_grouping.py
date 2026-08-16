from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

import fitz

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
MODEL = os.getenv("OPENAI_GROUPING_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
You analyze a PDF page for translation preprocessing.
Your job is ONLY to determine logical text groups. Do not translate.

A logical group is text that a human would regard as one translation unit:
- a heading
- one paragraph, including wrapped lines
- one bullet item, including its bullet
- one caption or label

Keep separate objects separate even when they are physically close:
- heading vs body
- unrelated labels
- figure/table text vs surrounding prose
- separate columns
- footer/page number

Use both the annotated page image and the supplied line metadata. The IDs in
red boxes on the image correspond to metadata IDs.

Return JSON only:
{"groups":[[0,1],[2],[3,4,5]]}
Every input ID must occur exactly once. Preserve page reading order.
""".strip()


def _annotated_page(page: fitz.Page, lines: list[dict[str, Any]]) -> str:
    """Render the page and draw stable IDs over every extracted line."""
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    image = fitz.Pixmap(fitz.csRGB, pix)
    # Draw on a separate pixmap through a temporary page because fitz Pixmap
    # drawing APIs are intentionally limited across versions.
    tmp = fitz.open()
    try:
        p = tmp.new_page(width=page.rect.width, height=page.rect.height)
        p.show_pdf_page(p.rect, page.parent, page.number)
        for line in lines:
            r = fitz.Rect(line["bbox"])
            p.draw_rect(r, color=(1, 0, 0), width=0.7)
            p.insert_text((r.x0, max(r.y0 - 2, 7)), str(line["_ai_id"]), fontsize=7, color=(1, 0, 0))
        annotated = p.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        return base64.b64encode(annotated.tobytes("png")).decode("ascii")
    finally:
        tmp.close()


def _line_metadata(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for line in lines:
        result.append(
            {
                "id": line["_ai_id"],
                "text": line["text"],
                "bbox": [round(x, 1) for x in line["bbox"]],
                "block_index": line.get("block_index"),
                "line_index": line.get("line_index"),
                "font_sizes": sorted({round(float(s.get("size", 0)), 1) for s in line.get("spans", [])}),
                "fonts": sorted({str(s.get("font", "")) for s in line.get("spans", [])}),
                "colors": sorted({int(s.get("color", 0) or 0) for s in line.get("spans", [])}),
                "bold": any(
                    bool(int(s.get("flags", 0) or 0) & 16)
                    or "bold" in str(s.get("font", "")).lower()
                    for s in line.get("spans", [])
                ),
            }
        )
    return result


def _request(image_b64: str, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps({"lines": metadata}, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                ],
            },
        ],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI grouping HTTP {exc.code}: {detail[:500]}") from exc
    content = result["choices"][0]["message"]["content"]
    return json.loads(content)


def _validate(raw: dict[str, Any], ids: set[int]) -> list[list[int]]:
    groups = raw.get("groups")
    if not isinstance(groups, list):
        raise ValueError("AI response has no groups list")
    normalized: list[list[int]] = []
    seen: set[int] = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Invalid AI group")
        current = [int(x) for x in group]
        if any(x not in ids or x in seen for x in current):
            raise ValueError("AI grouping contains missing or duplicate IDs")
        seen.update(current)
        normalized.append(current)
    if seen != ids:
        raise ValueError("AI grouping did not cover every line")
    return normalized


def group_page(page: fitz.Page, lines: list[dict[str, Any]]) -> list[list[int]] | None:
    """Ask vision AI for page-level logical groups; return None on failure."""
    if not lines or not os.getenv("OPENAI_API_KEY", "").strip():
        return None
    for i, line in enumerate(lines):
        line["_ai_id"] = i
    try:
        image = _annotated_page(page, lines)
        raw = _request(image, _line_metadata(lines))
        groups = _validate(raw, set(range(len(lines))))
        print(f"AI grouping: {len(lines)} lines -> {len(groups)} logical groups")
        return groups
    except Exception as exc:
        print(f"AI grouping failed; using local fallback: {exc}")
        return None
    finally:
        for line in lines:
            line.pop("_ai_id", None)
