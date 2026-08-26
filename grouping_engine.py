from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz

BULLET_CHARS = "•●○◦▪▫■□◆◇"
NUMBERED_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])\s+")
CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?|table|scheme|chart|source)\b", re.IGNORECASE)
HEADING_WORD_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s:–—-]{2,}$")

@dataclass
class LineFeatures:
    index: int
    x0: float
    y0: float
    x1: float
    y1: float
    height: float
    font_size: float
    font_names: tuple[str, ...]
    colors: tuple[int, ...]
    bold: bool
    block_index: int
    text: str
    indentation: float
    is_bullet: bool
    is_numbered: bool
    is_caption: bool
    is_heading_candidate: bool


def _line_font_size(line: dict[str, Any]) -> float:
    return max((float(s.get("size", 0) or 0) for s in line.get("spans", [])), default=12.0)


def _line_fonts(line: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(s.get("font", "")) for s in line.get("spans", [])}))


def _line_colors(line: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted({int(s.get("color", 0) or 0) for s in line.get("spans", [] )}))


def _is_bold(line: dict[str, Any]) -> bool:
    return any(bool(int(s.get("flags", 0) or 0) & 16) or "bold" in str(s.get("font", "")).lower() for s in line.get("spans", []))


def _looks_like_heading(text: str, font_size: float, median_size: float, bold: bool) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 140 or CAPTION_RE.match(stripped):
        return False
    uppercase = bool(HEADING_WORD_RE.match(stripped))
    size_signal = font_size >= median_size * 1.18 if median_size else False
    return (bold and (size_signal or uppercase)) or (size_signal and len(stripped.split()) <= 14) or (uppercase and len(stripped.split()) <= 12)


def _features(lines: list[dict[str, Any]]) -> list[LineFeatures]:
    if not lines:
        return []
    sizes = sorted(_line_font_size(line) for line in lines)
    median_size = sizes[len(sizes) // 2] if sizes else 12.0
    page_left = min(float(line["bbox"].x0) for line in lines)
    result = []
    for index, line in enumerate(lines):
        rect = fitz.Rect(line["bbox"])
        text = str(line.get("text", "")).strip()
        size = _line_font_size(line)
        result.append(LineFeatures(index, rect.x0, rect.y0, rect.x1, rect.y1, max(rect.height, 1.0), size, _line_fonts(line), _line_colors(line), _is_bold(line), int(line.get("block_index", -1)), text, rect.x0 - page_left, text[:1] in BULLET_CHARS, bool(NUMBERED_ITEM_RE.match(text)), bool(CAPTION_RE.match(text)), _looks_like_heading(text, size, median_size, _is_bold(line))))
    return result


def _vertical_gap(a: LineFeatures, b: LineFeatures) -> float:
    return b.y0 - a.y1


def _connection_score(a: LineFeatures, b: LineFeatures) -> float:
    gap = _vertical_gap(a, b)
    avg_height = max((a.height + b.height) / 2.0, 1.0)
    score = 0.0
    normalized_gap = gap / avg_height
    if -0.25 <= normalized_gap <= 0.45:
        score += 5
    elif 0.45 < normalized_gap <= 1.0:
        score += 2
    elif normalized_gap > 1.6:
        score -= 5
    elif normalized_gap < -0.8:
        score -= 3

    x_delta = abs(a.x0 - b.x0)
    if x_delta <= max(2.0, avg_height * 0.15):
        score += 4
    elif x_delta <= avg_height * 0.45:
        score += 2
    elif x_delta > avg_height * 1.25:
        score -= 4

    score += 5 if a.block_index == b.block_index else -1
    if abs(a.font_size - b.font_size) <= max(0.5, avg_height * 0.08):
        score += 2
    elif abs(a.font_size - b.font_size) > avg_height * 0.35:
        score -= 3
    if a.font_names and b.font_names and set(a.font_names) & set(b.font_names):
        score += 1
    if a.bold == b.bold:
        score += 1
    if a.colors and b.colors and set(a.colors) & set(b.colors):
        score += 1

    if a.is_heading_candidate or b.is_heading_candidate:
        score -= 6
    if a.is_caption or b.is_caption:
        score -= 6
    if a.is_bullet != b.is_bullet:
        score -= 3
    if a.is_numbered != b.is_numbered:
        score -= 3
    return score


def _same_group(a: LineFeatures, b: LineFeatures) -> bool:
    score = _connection_score(a, b)
    if b.is_caption or b.is_heading_candidate or a.is_caption or a.is_heading_candidate:
        return False
    if a.is_bullet or b.is_bullet or a.is_numbered or b.is_numbered:
        return score >= 7 and a.is_bullet == b.is_bullet and a.is_numbered == b.is_numbered
    return score >= 7


def _make_group(lines: list[dict[str, Any]], source: str, group_type: str) -> dict[str, Any]:
    bbox = fitz.Rect(lines[0]["bbox"])
    spans: list[dict] = []
    for line in lines:
        bbox |= fitz.Rect(line["bbox"])
        spans.extend(line.get("spans", []))
    # Preserve direction/alignment/style metadata from the dominant first line.
    directions = [line.get("dir", (1.0, 0.0)) for line in lines]
    rtl_count = sum(1 for d in directions if d and d[0] < 0 and abs(d[0]) >= abs(d[1]))
    vertical_count = sum(1 for d in directions if d and abs(d[1]) > abs(d[0]))
    direction = "rtl" if rtl_count > len(lines) / 2 else ("vertical" if vertical_count > len(lines) / 2 else "ltr")
    return {
        "text": "\n".join(str(line["text"]) for line in lines),
        "bbox": bbox,
        "size": max(float(line.get("size", 12) or 12) for line in lines),
        "spans": spans,
        "lines": lines,
        "block_indices": sorted({line.get("block_index", -1) for line in lines}),
        "group_source": source,
        "group_type": group_type,
        "direction": direction,
        "line_directions": directions,
    }


def group_lines(lines: list[dict[str, Any]], debug: bool = False) -> list[dict[str, Any]]:
    if not lines:
        return []
    features = _features(lines)
    groups = []
    current_indices = [0]
    for i in range(1, len(features)):
        previous, current = features[i - 1], features[i]
        connected = _same_group(previous, current)
        if debug:
            print(f"GROUPING line {previous.index} -> {current.index}: score={_connection_score(previous, current):.1f}, same_group={connected}, text={current.text[:70]!r}")
        if connected:
            current_indices.append(i)
        else:
            data = [lines[index] for index in current_indices]
            first = features[current_indices[0]]
            group_type = "caption" if first.is_caption else "heading" if first.is_heading_candidate else "list_item" if (first.is_bullet or first.is_numbered) else "paragraph"
            groups.append(_make_group(data, "rule", group_type))
            current_indices = [i]
    data = [lines[index] for index in current_indices]
    first = features[current_indices[0]]
    group_type = "caption" if first.is_caption else "heading" if first.is_heading_candidate else "list_item" if (first.is_bullet or first.is_numbered) else "paragraph"
    groups.append(_make_group(data, "rule", group_type))
    return groups


def group_page(page: fitz.Page, lines: list[dict[str, Any]], debug: bool = False) -> list[dict[str, Any]]:
    del page
    return group_lines(lines, debug=debug)
