from pathlib import Path
from hashlib import sha256

import fitz  # PyMuPDF


def calculate_file_fingerprint(file_bytes: bytes) -> str:
    """计算文件 SHA-256 指纹，用于阶段 21 的文档去重"""
    return sha256(file_bytes).hexdigest()


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