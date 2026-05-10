# ATS Intelligence Platform

This is a modern, scalable, full-stack Next.js 15 and FastAPI application for AI-powered candidate screening and resume evaluation using SBERT embeddings and ensemble machine learning models.

## Infrastructure

The application uses:
- **Frontend**: Next.js 15 (React), Tailwind CSS, shadcn/ui
- **Backend**: FastAPI (Python), SQLAlchemy Async, Pydantic, scikit-learn, sentence-transformers
- **Database**: Supabase PostgreSQL (Remote connection pooler with `sslmode=require`)

## Local Setup & Development

We rely on **Supabase** for our database rather than a local Docker container. This simplifies development and ensures parity with production.

### 1. Supabase Database Initialization
1. Create a project on [Supabase](https://supabase.com).
2. Go to **Database** -> **SQL Editor**.
3. Copy the contents of `backend/schema.sql` and run it to initialize your tables.
4. Copy the contents of `backend/seed.sql` and run it to populate your mock job data.

### 2. Environment Setup
Create a `.env` file in the root of the project (copy from `.env.example`):
```bash
cp .env.example .env
```
Update the `DATABASE_URL` with your Supabase Transaction connection string. It should look like:
`postgresql://postgres:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require`

### 3. Backend Setup
The backend requires Python 3.10+ and various ML libraries.
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the backend development server:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 4. Frontend Setup
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:3000`.

## Docker (Optional)
If you prefer to run both the frontend and backend in Docker, you can still use Docker Compose. The database is still external.
```bash
docker-compose up --build
```
