from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from html_pdf_translator import translate_pdf_file


class PdfTranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDF Translator")
        self.root.geometry("540x260")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.source_lang = tk.StringVar(value="auto")
        self.target_lang = tk.StringVar(value="ko")

        tk.Label(root, text="입력 PDF", font=("Malgun Gothic", 11)).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        tk.Entry(root, textvariable=self.input_path, width=48).grid(row=0, column=1, padx=12, pady=(12, 4))
        tk.Button(root, text="찾기", command=self.choose_input_file).grid(row=0, column=2, padx=(0, 12), pady=(12, 4))

        tk.Label(root, text="출력 PDF", font=("Malgun Gothic", 11)).grid(row=1, column=0, sticky="w", padx=12, pady=4)
        tk.Entry(root, textvariable=self.output_path, width=48).grid(row=1, column=1, padx=12, pady=4)
        tk.Button(root, text="저장 위치", command=self.choose_output_file).grid(row=1, column=2, padx=(0, 12), pady=4)

        tk.Label(root, text="원본 언어", font=("Malgun Gothic", 11)).grid(row=2, column=0, sticky="w", padx=12, pady=8)
        tk.Entry(root, textvariable=self.source_lang, width=12).grid(row=2, column=1, sticky="w", padx=12, pady=8)

        tk.Label(root, text="번역 언어", font=("Malgun Gothic", 11)).grid(row=2, column=2, sticky="w", padx=12, pady=8)
        tk.Entry(root, textvariable=self.target_lang, width=12).grid(row=2, column=3, sticky="w", padx=(0, 12), pady=8)

        tk.Button(root, text="PDF 번역 시작", command=self.translate_pdf, width=20, height=2, bg="#D0EBFF").grid(row=3, column=1, pady=18)

    def choose_input_file(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if selected:
            self.input_path.set(selected)
            default_output = os.path.splitext(selected)[0] + "_translated.pdf"
            self.output_path.set(default_output)

    def choose_output_file(self) -> None:
        selected = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if selected:
            self.output_path.set(selected)

    def translate_pdf(self) -> None:
        input_file = self.input_path.get().strip()
        output_file = self.output_path.get().strip()
        if not input_file or not output_file:
            messagebox.showwarning("입력 확인", "입력 PDF와 출력 경로를 모두 선택해 주세요.")
            return

        try:
            count = translate_pdf_file(input_file, output_file, self.source_lang.get(), self.target_lang.get())
            messagebox.showinfo("완료", f"번역이 끝났습니다.\n변환된 텍스트 블록 수: {count}\n결과 파일: {output_file}")
        except Exception as exc:
            messagebox.showerror("오류", f"PDF 번역 중 문제가 발생했습니다.\n{exc}")


def main() -> None:
    root = tk.Tk()
    PdfTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
