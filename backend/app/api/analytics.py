from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, text, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.evaluation import Evaluation
from app.models.job import Job
from app.models.user import User
from app.schemas.schemas import AnalyticsOverview

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _require_admin_or_recruiter(current_user: dict):
    if current_user["role"] not in ("recruiter", "admin"):
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/overview", response_model=AnalyticsOverview)
async def analytics_overview(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin_or_recruiter(current_user)

    total_ev = await db.scalar(select(func.count(Evaluation.id)))
    total_jobs = await db.scalar(select(func.count(Job.id)).where(Job.is_active == True))
    total_candidates = await db.scalar(
        select(func.count(func.distinct(Evaluation.candidate_id)))
    )
    total_selected = await db.scalar(
        select(func.count(Evaluation.id)).where(
            func.upper(Evaluation.final_decision) == "SELECT"
        )
    )
    total_consider = await db.scalar(
        select(func.count(Evaluation.id)).where(
            func.upper(Evaluation.final_decision) == "CONSIDER"
        )
    )
    total_rejected = await db.scalar(
        select(func.count(Evaluation.id)).where(
            func.upper(Evaluation.final_decision) == "REJECT"
        )
    )
    avg_match = await db.scalar(select(func.avg(Evaluation.match_score)))

    return AnalyticsOverview(
        total_candidates=total_candidates or 0,
        total_selected=total_selected or 0,
        total_consider=total_consider or 0,
        total_rejected=total_rejected or 0,
        avg_match_score=round(avg_match or 0.0, 2),
        total_jobs=total_jobs or 0,
        total_evaluations=total_ev or 0,
    )


@router.get("/scores")
async def score_distribution(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin_or_recruiter(current_user)

    result = await db.execute(
        select(
            Evaluation.match_score,
            Evaluation.skill_match_ratio,
            Evaluation.comm_score,
            Evaluation.tech_score,
            Evaluation.conf_score,
            Evaluation.final_decision,
            Evaluation.ml_confidence,
        ).limit(1000)
    )
    rows = result.fetchall()
    return [
        {
            "match_score": r.match_score,
            "skill_match_ratio": r.skill_match_ratio,
            "comm_score": r.comm_score,
            "tech_score": r.tech_score,
            "conf_score": r.conf_score,
            "final_decision": r.final_decision,
            "ml_confidence": r.ml_confidence,
        }
        for r in rows
    ]


@router.get("/skills")
async def skill_analytics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin_or_recruiter(current_user)

    # Get matched/missing skill arrays from all evaluations
    result = await db.execute(
        select(Evaluation.matched_skills, Evaluation.missing_skills).limit(500)
    )
    rows = result.fetchall()

    matched_counter: dict[str, int] = {}
    missing_counter: dict[str, int] = {}
    for row in rows:
        for s in (row.matched_skills or []):
            matched_counter[s] = matched_counter.get(s, 0) + 1
        for s in (row.missing_skills or []):
            missing_counter[s] = missing_counter.get(s, 0) + 1

    all_skills = set(matched_counter.keys()) | set(missing_counter.keys())
    data = [
        {
            "skill": s,
            "matched_count": matched_counter.get(s, 0),
            "missing_count": missing_counter.get(s, 0),
        }
        for s in all_skills
    ]
    data.sort(key=lambda x: x["matched_count"] + x["missing_count"], reverse=True)
    return data[:30]


@router.get("/model-performance")
async def model_performance(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return pre-computed model metrics from metadata.json."""
    _require_admin_or_recruiter(current_user)
    import json, os
    meta_path = os.path.join(os.path.dirname(__file__), "../../../metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


@router.get("/jobs-breakdown")
async def jobs_breakdown(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin_or_recruiter(current_user)

    result = await db.execute(
        select(
            Job.title,
            func.count(Evaluation.id).label("total"),
            func.avg(Evaluation.match_score).label("avg_match"),
            func.sum(
                func.cast(func.upper(Evaluation.final_decision) == "SELECT", Integer)
            ).label("selected"),
        )
        .join(Evaluation, Evaluation.job_id == Job.id, isouter=True)
        .group_by(Job.title)
        .order_by(func.count(Evaluation.id).desc())
    )
    rows = result.fetchall()
    return [
        {
            "job_title": r.title,
            "total": r.total or 0,
            "avg_match": round(r.avg_match or 0, 2),
            "selected": r.selected or 0,
        }
        for r in rows
    ]
