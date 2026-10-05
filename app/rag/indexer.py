from pathlib import Path

from app.services.pdf_service import extract_pdf_pages
from app.services.chunk_service import chunk_pages
from app.services.embedding_service import embed_texts
from app.rag.vector_store import upsert_chunks


async def index_document(
    pdf_path: Path,
    source_name: str,
) -> dict:
    # 1. PDF 解析
    pages = extract_pdf_pages(pdf_path)

    # 2. Chunk
    chunks = chunk_pages(
        pages,
        source=source_name,
    )

    # Qwen Embedding 单次最多 20 条
    batch_size = 20

    indexed_count = 0

    # 3. 分批 Embedding + 分批写入 Qdrant
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        batch_vectors = await embed_texts(
            [chunk.page_content for chunk in batch]
        )

        upsert_chunks(
            batch,
            batch_vectors,
        )

        indexed_count += len(batch)

        print(
            f"Indexing: {indexed_count}/{len(chunks)} chunks"
        )

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "indexed_chunks": indexed_count,
    }