from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.rag.indexer import index_document
from app.rag.vector_store import (
    delete_document_points,
    find_document_by_fingerprint,
)
from app.services.pdf_service import calculate_file_fingerprint

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Query(default="local-demo"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 25 MiB limit")

    # 1. 计算 SHA-256 指纹并查重
    fingerprint = calculate_file_fingerprint(file_bytes)
    existing = find_document_by_fingerprint(fingerprint, user_id=user_id)
    if existing:
        return {
            "duplicate": True,
            "document_id": existing["document_id"],
            "filename": existing.get("source", file.filename),
            "message": "相同文档已入库，跳过重复索引",
        }

    # 2. 未重复才落盘并索引
    document_id = uuid4().hex
    saved_path = UPLOAD_DIR / f"{document_id}.pdf"
    saved_path.write_bytes(file_bytes)

    result = await index_document(
        saved_path,
        source_name=file.filename,
        document_id=document_id,
        fingerprint=fingerprint,
        user_id=user_id,
    )
    return {"duplicate": False, **result}


@router.delete("/{document_id}")
async def remove_document(
    document_id: str,
    user_id: str = Query(default="local-demo"),
):
    deleted = delete_document_points(document_id, user_id=user_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found or already deleted")
    return {"document_id": document_id, "deleted_chunks": deleted}