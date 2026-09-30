from pathlib import Path
import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page": index + 1,
                    "text": text,
                })
    return pages