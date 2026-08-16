from unittest.mock import patch

from pdf_translator_core import translate_text_blocks


@patch("pdf_translator_core.GoogleTranslator")
def test_translate_text_blocks(mock_translator_cls):
    mock_translator = mock_translator_cls.return_value
    mock_translator.translate.side_effect = lambda text: {
        "Hello world": "안녕하세요 세계",
        "Sample text": "샘플 텍스트",
    }.get(text, text)

    result = translate_text_blocks("Hello world", "en", "ko")
    assert result == "안녕하세요 세계"

    result = translate_text_blocks("Sample text", "en", "ko")
    assert result == "샘플 텍스트"
