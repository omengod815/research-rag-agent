from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.rag.indexer import index_document

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    saved_path = UPLOAD_DIR / f"{uuid4().hex}.pdf"
    saved_path.write_bytes(await file.read())

    indexed = await index_document(saved_path, source_name=file.filename)
    return {
        "document_id": saved_path.stem,
        "filename": file.filename,
        **indexed,
        "indexed": True,
    }