from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from deep_translator import GoogleTranslator


FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_PATH = FONT_DIR / "NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"
BULLET_CHARS = "•●○◦▪▫■□◆◇"
IMAGE_OVERLAP_TOLERANCE = 0.5


def _find_font(*names: str) -> Path:
    """Find a font file in fonts/; fall back to regular Korean font."""
    for name