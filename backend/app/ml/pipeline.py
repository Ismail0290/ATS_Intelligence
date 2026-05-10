"""
ML Pipeline — verbatim reuse of all logic from the original Streamlit app.py
Adapted for single-record API calls (no pandas batch, no Streamlit spinners).
"""

import re
import numpy as np
from typing import Any

# ─── Constants (exact copy from app.py) ──────────────────────────────────────
SKILLS = [
    "python", "java", "javascript", "typescript", "scala", "r", "sql", "c++", "golang",
    "machine learning", "deep learning", "neural network", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit learn", "huggingface",
    "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt",
    "data analysis", "data engineering", "etl", "feature engineering",
    "aws", "gcp", "azure", "docker", "kubernetes", "ci cd", "terraform",
    "microservices", "api", "rest", "graphql",
    "system design", "cloud", "agile", "leadership", "communication",
    "product management", "project management", "data visualization",
    "tableau", "power bi", "excel",
    "ecommerce", "seo", "google analytics", "shopify", "digital marketing",
]

FEATURE_COLS = [
    'match_score', 'skill_match_ratio', 'num_matched_skills',
    'num_missing_skills', 'num_candidate_skills', 'comm_score',
    'conf_score', 'subj_score', 'tech_score', 'response_len', 'sentence_cmplx',
]


# ─── Text Utilities (verbatim from app.py) ────────────────────────────────────
def clean_text(text: str, stop_words: set) -> str:
    """Verbatim copy from original app.py."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d{3}[-.\\s]\d{3}[-.\\s]\d{4}', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def extract_skills(text: str) -> list[str]:
    """Verbatim copy from original app.py."""
    return [s for s in SKILLS if s in text]


def skill_gap(cand_skills: list, jd_skills: list) -> tuple[list, list]:
    """Verbatim copy from original app.py."""
    missing = list(set(jd_skills) - set(cand_skills))
    matched = list(set(jd_skills) & set(cand_skills))
    return missing, matched


# ─── Feature Computation (single-record, adapted from compute_features) ───────
def compute_features_single(
    resume: str,
    transcript: str,
    job_description: str,
    sbert,  # SentenceTransformer instance
) -> dict[str, Any]:
    """
    Single-record version of compute_features() from app.py.
    Returns a dict with all feature values + raw skill lists.
    """
    from nltk.corpus import stopwords
    from textblob import TextBlob
    from sklearn.metrics.pairwise import cosine_similarity

    stop_words = set(stopwords.words('english'))

    c_resume = clean_text(resume, stop_words)
    c_trans = clean_text(transcript, stop_words)
    c_jd = clean_text(job_description, stop_words)

    # Candidate text: resume weighted x2 + transcript (exact from app.py line 222)
    candidate_text = c_resume + " " + c_resume + " " + c_trans

    # SBERT embeddings + cosine similarity
    cand_emb = sbert.encode([candidate_text], convert_to_numpy=True)
    jd_emb = sbert.encode([c_jd], convert_to_numpy=True)
    match_score = float(cosine_similarity(cand_emb, jd_emb)[0][0] * 100)

    # Skills
    cand_skills = extract_skills(candidate_text)
    jd_skills = extract_skills(c_jd)
    missing, matched = skill_gap(cand_skills, jd_skills)
    num_jd_skills = len(jd_skills)
    skill_match_ratio = len(matched) / num_jd_skills if num_jd_skills > 0 else 0.0

    # Communication features (verbatim from app.py)
    words = c_trans.split()
    comm_score = len(set(words)) / len(words) if len(words) >= 5 else 0.0
    conf_score = float(TextBlob(c_trans).sentiment.polarity) if c_trans.strip() else 0.0
    subj_score = float(TextBlob(c_trans).sentiment.subjectivity) if c_trans.strip() else 0.0
    tech_score = sum(1 for s in SKILLS if s in candidate_text) / len(SKILLS) if candidate_text.strip() else 0.0
    response_len = float(np.log1p(len(c_trans.split())))

    # Sentence complexity
    sents = [s for s in re.split(r'[.!?]', c_trans) if len(s.split()) > 2]
    sentence_cmplx = float(np.mean([len(s.split()) for s in sents])) if sents else 0.0

    return {
        "match_score": match_score,
        "skill_match_ratio": skill_match_ratio,
        "num_matched_skills": len(matched),
        "num_missing_skills": len(missing),
        "num_candidate_skills": len(cand_skills),
        "comm_score": comm_score,
        "conf_score": conf_score,
        "subj_score": subj_score,
        "tech_score": tech_score,
        "response_len": response_len,
        "sentence_cmplx": sentence_cmplx,
        # Extra (not ML features, used for display)
        "matched_skills": matched,
        "missing_skills": missing,
        "jd_skills": jd_skills,
        "cand_skills": cand_skills,
    }


# ─── Explanation Generator (verbatim from app.py) ────────────────────────────
def generate_explanation(features: dict) -> str:
    """Verbatim copy from original app.py, adapted to use a dict instead of pd.Series."""
    parts = []
    match_score = features['match_score']
    if match_score >= 75:
        parts.append(f"Strong overall alignment with JD (match score: {match_score:.1f}%).")
    elif match_score >= 55:
        parts.append(f"Moderate alignment with JD (match score: {match_score:.1f}%).")
    else:
        parts.append(f"Low alignment with JD (match score: {match_score:.1f}%).")

    missing_skills = features.get('missing_skills', [])
    if len(missing_skills) == 0:
        parts.append("All required JD skills are present.")
    elif len(missing_skills) <= 2:
        parts.append(f"Minor skill gap — missing: {', '.join(missing_skills)}.")
    else:
        parts.append(f"Significant skill gaps — missing: {', '.join(missing_skills[:5])}.")

    comm_score = features['comm_score']
    if comm_score >= 0.6:
        parts.append("Strong communication vocabulary diversity.")
    elif comm_score < 0.35:
        parts.append("Limited vocabulary diversity in interview responses.")

    conf_score = features['conf_score']
    if conf_score > 0.2:
        parts.append("Positive and confident tone in interview.")
    elif conf_score < -0.1:
        parts.append("Negative or uncertain tone detected in interview.")

    tech_score = features['tech_score']
    if tech_score > 0.4:
        parts.append("High technical keyword density.")
    elif tech_score < 0.15:
        parts.append("Low technical depth in responses.")

    return " ".join(parts)


# ─── Rule-Based Decision (verbatim from app.py) ───────────────────────────────
def rule_based_decision(
    match_score: float,
    skill_match_ratio: float,
    thresh_select: float = 75.0,
    thresh_consider: float = 55.0,
) -> str:
    """Verbatim logic from app.py lines 298-304."""
    if match_score > thresh_select and skill_match_ratio > 0.5:
        return "SELECT"
    elif match_score > thresh_consider and skill_match_ratio > 0.3:
        return "CONSIDER"
    else:
        return "REJECT"


# ─── Final Decision (combines ML + rule-based) ────────────────────────────────
def determine_final_decision(
    ml_decision: str,
    ml_confidence: float,
    match_score: float,
    skill_match_ratio: float,
) -> str:
    """
    ML models output binary (select/reject).
    CONSIDER is derived when ML says select with low confidence or
    rule-based says CONSIDER.
    """
    rule = rule_based_decision(match_score, skill_match_ratio)
    if ml_decision.lower() == "select" and ml_confidence >= 70:
        return "SELECT"
    elif rule == "CONSIDER" or (ml_decision.lower() == "select" and ml_confidence < 70):
        return "CONSIDER"
    else:
        return "REJECT"
