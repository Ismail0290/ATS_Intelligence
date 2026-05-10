import io
import csv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.evaluation import Evaluation
from app.models.user import User
from app.models.job import Job
from app.schemas.schemas import EvaluationOut, EvaluationWithJob

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


@router.get("", response_model=list[EvaluationWithJob])
async def list_candidates(
    decision: str | None = None,
    job_id: str | None = None,
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user["role"] not in ("recruiter", "admin"):
        raise HTTPException(status_code=403, detail="Access denied")

    q = select(Evaluation, Job.title.label("job_title"), Job.company.label("job_company"))\
        .join(Job, Evaluation.job_id == Job.id, isouter=True)

    if decision:
        q = q.where(func.upper(Evaluation.final_decision) == decision.upper())
    if job_id:
        q = q.where(Evaluation.job_id == job_id)

    q = q.order_by(Evaluation.created_at.desc()).offset(skip).limit(limit)
    rows = await db.execute(q)

    results = []
    for ev, job_title, job_company in rows:
        data = EvaluationOut.model_validate(ev).model_dump()
        data["job_title"] = job_title
        data["job_company"] = job_company
        results.append(EvaluationWithJob(**data))
    return results


@router.get("/my", response_model=list[EvaluationWithJob])
async def my_evaluations(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(Evaluation, Job.title.label("job_title"), Job.company.label("job_company"))\
        .join(Job, Evaluation.job_id == Job.id, isouter=True)\
        .where(Evaluation.candidate_id == current_user["sub"])\
        .order_by(Evaluation.created_at.desc())
    rows = await db.execute(q)

    results = []
    for ev, job_title, job_company in rows:
        data = EvaluationOut.model_validate(ev).model_dump()
        data["job_title"] = job_title
        data["job_company"] = job_company
        results.append(EvaluationWithJob(**data))
    return results


@router.get("/export/csv")
async def export_csv(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user["role"] not in ("recruiter", "admin"):
        raise HTTPException(status_code=403, detail="Access denied")

    q = select(Evaluation, User.email, User.full_name, Job.title)\
        .join(User, Evaluation.candidate_id == User.id, isouter=True)\
        .join(Job, Evaluation.job_id == Job.id, isouter=True)\
        .order_by(Evaluation.created_at.desc())
    rows = await db.execute(q)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "candidate_email", "candidate_name", "job_title",
        "match_score", "skill_match_ratio", "num_matched_skills", "num_missing_skills",
        "comm_score", "conf_score", "tech_score",
        "ml_decision", "ml_confidence", "final_decision", "model_used",
        "explanation", "matched_skills", "missing_skills", "created_at"
    ])
    for ev, email, full_name, job_title in rows:
        writer.writerow([
            email, full_name, job_title,
            round(ev.match_score or 0, 2),
            round(ev.skill_match_ratio or 0, 3),
            ev.num_matched_skills, ev.num_missing_skills,
            round(ev.comm_score or 0, 3),
            round(ev.conf_score or 0, 3),
            round(ev.tech_score or 0, 3),
            ev.ml_decision, round(ev.ml_confidence or 0, 1), ev.final_decision, ev.model_used,
            ev.explanation,
            ",".join(ev.matched_skills or []),
            ",".join(ev.missing_skills or []),
            ev.created_at.isoformat() if ev.created_at else "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ats_candidates.csv"},
    )
