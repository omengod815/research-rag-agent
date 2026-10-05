from pathlib import Path
from app.services.pdf_service import extract_pdf_pages
from app.services.chunk_service import chunk_pages
from app.services.embedding_service import embed_texts
from app.rag.vector_store import upsert_chunks

async def index_document(pdf_path: Path, source_name: str) -> dict:
    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, source=source_name)

    vectors: list[list[float]] = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_vectors = await embed_texts([c.page_content for c in batch])
        vectors.extend(batch_vectors)

    upsert_chunks(chunks, vectors)
    return {"pages": len(pages), "chunks": len(chunks)}