from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.ml import model_loader, pipeline
from app.models.evaluation import Evaluation
from app.models.job import Job
from app.models.user import User
from app.schemas.schemas import EvaluateRequest, EvaluationOut

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


@router.post("", response_model=EvaluationOut, status_code=status.HTTP_201_CREATED)
async def evaluate_candidate(
    payload: EvaluateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch job
    result = await db.execute(select(Job).where(Job.id == payload.job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 2. Compute features using exact ML pipeline logic
    sbert = model_loader.get_sbert()
    try:
        features = pipeline.compute_features_single(
            resume=payload.resume_text,
            transcript=payload.transcript,
            job_description=job.description,
            sbert=sbert,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    # 3. Run ML prediction
    ml_decision, ml_confidence = model_loader.run_prediction(features, payload.model_name)

    # 4. Rule-based fallback
    rule_dec = pipeline.rule_based_decision(
        features["match_score"],
        features["skill_match_ratio"],
        payload.thresh_select,
        payload.thresh_consider,
    )

    # 5. Final decision
    if ml_decision is not None:
        final = pipeline.determine_final_decision(
            ml_decision, ml_confidence, features["match_score"], features["skill_match_ratio"]
        )
    else:
        final = rule_dec
        ml_decision = rule_dec.lower()
        ml_confidence = features["match_score"]

    # 6. Generate explanation
    explanation = pipeline.generate_explanation(features)

    # 7. Persist evaluation
    evaluation = Evaluation(
        candidate_id=current_user["sub"],
        job_id=payload.job_id,
        transcript=payload.transcript,
        match_score=features["match_score"],
        skill_match_ratio=features["skill_match_ratio"],
        num_matched_skills=features["num_matched_skills"],
        num_missing_skills=features["num_missing_skills"],
        num_candidate_skills=features["num_candidate_skills"],
        comm_score=features["comm_score"],
        conf_score=features["conf_score"],
        subj_score=features["subj_score"],
        tech_score=features["tech_score"],
        response_len=features["response_len"],
        sentence_cmplx=features["sentence_cmplx"],
        ml_decision=ml_decision,
        ml_confidence=ml_confidence,
        rule_decision=rule_dec,
        final_decision=final,
        explanation=explanation,
        model_used=payload.model_name,
        matched_skills=features["matched_skills"],
        missing_skills=features["missing_skills"],
    )
    db.add(evaluation)
    await db.flush()
    await db.refresh(evaluation)

    return EvaluationOut.model_validate(evaluation)


@router.get("/{evaluation_id}", response_model=EvaluationOut)
async def get_evaluation(
    evaluation_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Evaluation).where(Evaluation.id == evaluation_id))
    ev = result.scalar_one_or_none()
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    # Candidates can only see their own
    if current_user["role"] == "candidate" and str(ev.candidate_id) != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return EvaluationOut.model_validate(ev)
