import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    transcript: Mapped[str | None] = mapped_column(Text)

    # ML features
    match_score: Mapped[float | None] = mapped_column(Float)
    skill_match_ratio: Mapped[float | None] = mapped_column(Float)
    num_matched_skills: Mapped[int | None] = mapped_column(Integer)
    num_missing_skills: Mapped[int | None] = mapped_column(Integer)
    num_candidate_skills: Mapped[int | None] = mapped_column(Integer)
    comm_score: Mapped[float | None] = mapped_column(Float)
    conf_score: Mapped[float | None] = mapped_column(Float)
    subj_score: Mapped[float | None] = mapped_column(Float)
    tech_score: Mapped[float | None] = mapped_column(Float)
    response_len: Mapped[float | None] = mapped_column(Float)
    sentence_cmplx: Mapped[float | None] = mapped_column(Float)

    # Results
    ml_decision: Mapped[str | None] = mapped_column(String(20))      # select | reject
    ml_confidence: Mapped[float | None] = mapped_column(Float)
    rule_decision: Mapped[str | None] = mapped_column(String(20))    # SELECT | CONSIDER | REJECT
    final_decision: Mapped[str | None] = mapped_column(String(20))   # SELECT | CONSIDER | REJECT
    explanation: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(50))

    # Skills
    matched_skills: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    missing_skills: Mapped[list[str] | None] = mapped_column(ARRAY(String))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
