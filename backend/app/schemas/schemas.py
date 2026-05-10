from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, field_validator


# ─── Auth ────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = "candidate"  # candidate | recruiter | admin

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("candidate", "recruiter", "admin"):
            raise ValueError("role must be candidate, recruiter, or admin")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


# ─── User ────────────────────────────────────────────────────────────────────
class UserOut(BaseModel):
    id: UUID
    email: str
    full_name: Optional[str]
    role: str
    resume_text: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    resume_text: Optional[str] = None


# ─── Job ─────────────────────────────────────────────────────────────────────
class JobCreate(BaseModel):
    title: str
    company: Optional[str] = None
    description: str
    required_skills: Optional[list[str]] = []
    location: Optional[str] = None
    employment_type: Optional[str] = None
    salary_range: Optional[str] = None


class JobUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    description: Optional[str] = None
    required_skills: Optional[list[str]] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    salary_range: Optional[str] = None
    is_active: Optional[bool] = None


class JobOut(BaseModel):
    id: UUID
    title: str
    company: Optional[str]
    description: str
    required_skills: Optional[list[str]]
    location: Optional[str]
    employment_type: Optional[str]
    salary_range: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Evaluation ───────────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    job_id: UUID
    resume_text: str  # pre-extracted text or pasted text
    transcript: str = ""
    model_name: str = "mlp_neural_net"
    thresh_select: float = 75.0
    thresh_consider: float = 55.0

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v):
        allowed = (
            "logistic_regression", "random_forest", "xgboost", "mlp_neural_net",
            "dnn", "lstm", "gru", "transformer"
        )
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v


class EvaluationOut(BaseModel):
    id: UUID
    candidate_id: Optional[UUID]
    job_id: Optional[UUID]

    # Features
    match_score: Optional[float]
    skill_match_ratio: Optional[float]
    num_matched_skills: Optional[int]
    num_missing_skills: Optional[int]
    num_candidate_skills: Optional[int]
    comm_score: Optional[float]
    conf_score: Optional[float]
    subj_score: Optional[float]
    tech_score: Optional[float]
    response_len: Optional[float]
    sentence_cmplx: Optional[float]

    # Decisions
    ml_decision: Optional[str]
    ml_confidence: Optional[float]
    rule_decision: Optional[str]
    final_decision: Optional[str]
    explanation: Optional[str]
    model_used: Optional[str]

    # Skills
    matched_skills: Optional[list[str]]
    missing_skills: Optional[list[str]]

    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationWithJob(EvaluationOut):
    job_title: Optional[str] = None
    job_company: Optional[str] = None


# ─── Analytics ───────────────────────────────────────────────────────────────
class AnalyticsOverview(BaseModel):
    total_candidates: int
    total_selected: int
    total_consider: int
    total_rejected: int
    avg_match_score: float
    total_jobs: int
    total_evaluations: int


class ScoreDistribution(BaseModel):
    bucket: str
    count: int


class SkillFrequency(BaseModel):
    skill: str
    matched_count: int
    missing_count: int
