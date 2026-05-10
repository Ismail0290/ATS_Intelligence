"""
FastAPI application factory with lifespan for ML model loading.
"""

import logging
import os
from contextlib import asynccontextmanager

# Fix OpenMP deadlock on macOS ARM during XGBoost load
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.ml import model_loader
from app.api import auth, jobs, evaluate, candidates, analytics, upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup."""
    logger.info("⚡ ATS Intelligence API starting up...")

    # Download NLTK data
    model_loader.download_nltk()

    # Load SBERT
    model_loader.load_sbert(settings.SBERT_MODEL)

    # Load ML models (.pkl files)
    model_loader.load_ml_models(settings.MODEL_DIR)

    # Create DB tables (for dev; use Alembic migrations in prod)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ All models loaded. API ready.")
    yield
    logger.info("🛑 ATS Intelligence API shutting down.")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Production-grade ATS Intelligence API with SBERT + ensemble ML",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Routers
    app.include_router(auth.router)
    app.include_router(jobs.router)
    app.include_router(evaluate.router)
    app.include_router(candidates.router)
    app.include_router(analytics.router)
    app.include_router(upload.router)

    @app.get("/health", tags=["health"])
    async def health():
        models = model_loader.get_ml_models()
        return {
            "status": "ok",
            "version": settings.VERSION,
            "models_loaded": list(models.keys()),
            "models_info": {k: v["framework"] for k, v in models.items()}
        }

    return app


app = create_app()
