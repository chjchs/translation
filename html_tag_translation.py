from __future__ import annotations

import os
from typing import Any

import deepl


def deepl_translate_html(
    html_text: str,
    source_lang: str = "auto",
    target_lang: str = "ko",
) -> str:
    """Translate HTML while asking DeepL to preserve the markup."""
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY environment variable is not set")

    translator = deepl.DeepLClient(api_key)
    kwargs: dict[str, Any] = {
        "target_lang": target_lang.upper(),
        "preserve_formatting": True,
        "tag_handling": "html",
    }
    if source_lang and source_lang.lower() != "auto":
        kwargs["source_lang"] = source_lang.upper()

    result = translator.translate_text(html_text, **kwargs)
    return str(result.text)


def main() -> None:
    source = (
        'Activation of <b>smooth muscle</b> by '
        '<span style="color:red">Ca2+ influx</span>.'
    )
    translated = deepl_translate_html(source, "en", "ko")
    print("SOURCE:")
    print(source)
    print("\nTRANSLATED:")
    print(translated)


if __name__ == "__main__":
    main()
