-- ============================================================
-- ATS Intelligence Database Schema
-- Run on Supabase SQL Editor or any PostgreSQL instance
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'candidate'
        CHECK (role IN ('candidate', 'recruiter', 'admin')),
    resume_text TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- Jobs
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    company TEXT,
    description TEXT NOT NULL,
    required_skills TEXT[],
    location TEXT,
    employment_type TEXT,
    salary_range TEXT,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs(is_active);
CREATE INDEX IF NOT EXISTS idx_jobs_created_by ON jobs(created_by);

-- Evaluations
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES users(id) ON DELETE SET NULL,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    transcript TEXT,
    -- ML Features
    match_score FLOAT,
    skill_match_ratio FLOAT,
    num_matched_skills INT,
    num_missing_skills INT,
    num_candidate_skills INT,
    comm_score FLOAT,
    conf_score FLOAT,
    subj_score FLOAT,
    tech_score FLOAT,
    response_len FLOAT,
    sentence_cmplx FLOAT,
    -- Results
    ml_decision TEXT,
    ml_confidence FLOAT,
    rule_decision TEXT,
    final_decision TEXT,
    explanation TEXT,
    model_used TEXT,
    -- Skills
    matched_skills TEXT[],
    missing_skills TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_candidate ON evaluations(candidate_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_job ON evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_decision ON evaluations(final_decision);
