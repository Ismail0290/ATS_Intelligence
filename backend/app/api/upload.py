from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.ml.resume_parser import extract_resume_text
from app.models.user import User

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/resume")
async def upload_resume(
    file: UploadFile = File(...),
    save_to_profile: bool = True,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PDF, DOCX, or TXT.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 5 MB.")

    try:
        text = extract_resume_text(content, file.filename or "resume.pdf")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the file.")

    if save_to_profile:
        result = await db.execute(select(User).where(User.id == current_user["sub"]))
        user = result.scalar_one_or_none()
        if user:
            user.resume_text = text
            await db.flush()

    return {
        "filename": file.filename,
        "extracted_text": text,
        "char_count": len(text),
        "saved_to_profile": save_to_profile,
    }
