from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.pdf_service import extract_pdf_pages

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    saved_name = f"{uuid4().hex}.pdf"
    saved_path = UPLOAD_DIR / saved_name
    saved_path.write_bytes(await file.read())

    pages = extract_pdf_pages(saved_path)
    return {
        "document_id": saved_path.stem,
        "filename": file.filename,
        "page_count_with_text": len(pages),
        "preview": pages[:2],
    }