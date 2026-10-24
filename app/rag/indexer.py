from pathlib import Path

from app.rag.vector_store import upsert_chunks
from app.services.chunk_service import chunk_pages
from app.services.embedding_service import embed_texts
from app.services.pdf_service import extract_pdf_pages


async def index_document(
    pdf_path: Path,
    source_name: str,
    document_id: str,
    fingerprint: str,
    user_id: str = "local-demo",
) -> dict:
    pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise ValueError("PDF 没有可提取的文本页；扫描件需先增加 OCR。")

    chunks = chunk_pages(pages, source=source_name)
    if not chunks:
        raise ValueError("PDF 解析成功但未生成 Chunk。")

    vectors: list[list[float]] = []
    batch_size = 20
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_vectors = await embed_texts([item.page_content for item in batch])
        vectors.extend(batch_vectors)

    upsert_chunks(
        chunks=chunks,
        vectors=vectors,
        document_id=document_id,
        fingerprint=fingerprint,
        user_id=user_id,
    )
    return {"pages": len(pages), "chunks": len(chunks), "document_id": document_id}