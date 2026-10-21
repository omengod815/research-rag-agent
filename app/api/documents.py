from hashlib import sha256
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.rag.indexer import index_document
from app.rag.vector_store import delete_document_points, find_document_by_fingerprint

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(default="local-demo"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF 超过 25 MiB 限制。")

    fingerprint = sha256(content).hexdigest()
    existing_id = find_document_by_fingerprint(fingerprint, user_id=user_id)
    if existing_id:
        return {
            "duplicate": True,
            "indexed": False,
            "document_id": existing_id,
            "fingerprint": fingerprint,
            "message": "该用户已经索引过内容相同的 PDF，跳过重复 Embedding。",
        }

    document_id = fingerprint[:24]
    saved_path = UPLOAD_DIR / f"{fingerprint}.pdf"
    saved_path.write_bytes(content)

    try:
        result = await index_document(
            pdf_path=saved_path,
            source_name=Path(file.filename).name,
            document_id=document_id,
            fingerprint=fingerprint,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 索引失败: {type(exc).__name__}: {exc}") from exc

    return {
        "duplicate": False,
        "indexed": True,
        "filename": Path(file.filename).name,
        "document_id": document_id,
        "fingerprint": fingerprint,
        **result,
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Query(default="local-demo", min_length=1, max_length=100),
):
    delete_document_points(document_id=document_id, user_id=user_id)
    return {"deleted": True, "document_id": document_id, "user_id": user_id}