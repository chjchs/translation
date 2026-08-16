from __future__ import annotations

import argparse

from pdf_translator_core import translate_pdf_file


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 파일을 번역해 새 PDF로 저장합니다.")
    parser.add_argument("input_pdf", help="읽을 원본 PDF 경로")
    parser.add_argument("output_pdf", help="저장할 번역된 PDF 경로")
    parser.add_argument("--source-lang", default="auto", help="예: auto, en, ja, zh, ko")
    parser.add_argument("--target-lang", default="ko", help="예: ko, en, ja, zh")
    parser.add_argument("--no-ai-grouping", action="store_true", help="AI page grouping을 끄고 기존 local grouping을 사용")
    args = parser.parse_args()

    count = translate_pdf_file(
        args.input_pdf,
        args.output_pdf,
        args.source_lang,
        args.target_lang,
        use_ai_grouping=not args.no_ai_grouping,
    )
    print(f"완료: {count}개의 logical text group을 번역했습니다. 결과: {args.output_pdf}")


if __name__ == "__main__":
    main()
