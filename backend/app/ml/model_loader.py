"""
Singleton loader for all trained ML models and preprocessing artifacts.
Loaded once at FastAPI startup via lifespan context manager.
"""

import os
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.pipeline import FEATURE_COLS

logger = logging.getLogger(__name__)

# Global singletons
_sbert = None
_ml_models: dict[str, Any] = {}
_scaler = None
_label_encoder = None


def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    global _sbert
    if _sbert is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT model: {model_name}")
        _sbert = SentenceTransformer(model_name)
        logger.info("SBERT loaded successfully")
    return _sbert


def load_ml_models(model_dir: str = "./models"):
    global _ml_models, _scaler, _label_encoder
    model_dir = Path(model_dir)

    names = ["logistic_regression", "random_forest", "xgboost", "mlp_neural_net"]
    for name in names:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            _ml_models[name] = joblib.load(path)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {path}")

    scaler_path = model_dir / "scaler.pkl"
    le_path = model_dir / "label_encoder.pkl"
    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
    if le_path.exists():
        _label_encoder = joblib.load(le_path)

    logger.info(f"ML artifacts loaded: {list(_ml_models.keys())}, scaler={_scaler is not None}, le={_label_encoder is not None}")


def download_nltk():
    import nltk
    import socket
    
    # Set a short timeout so API startup doesn't hang indefinitely
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3.0)
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download failed or timed out: {e}")
    finally:
        socket.setdefaulttimeout(old_timeout)


def get_sbert():
    if _sbert is None:
        raise RuntimeError("SBERT not loaded. Call load_sbert() at startup.")
    return _sbert


def get_ml_models() -> dict[str, Any]:
    return _ml_models


def get_scaler():
    return _scaler


def get_label_encoder():
    return _label_encoder


def run_prediction(features: dict, model_name: str = "mlp_neural_net") -> tuple[str, float]:
    """
    Run ML prediction on feature dict.
    Returns (decision_label, confidence_percent).
    """
    model = _ml_models.get(model_name)
    if model is None or _scaler is None or _label_encoder is None:
        # Fallback to rule-based if models not available
        return None, 0.0

    # Build feature vector in exact column order
    X = np.array([[features.get(col, 0.0) for col in FEATURE_COLS]], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = _scaler.transform(X)

    pred_enc = model.predict(X_scaled)
    pred_label = _label_encoder.inverse_transform(pred_enc)[0]
    confidence = float(model.predict_proba(X_scaled).max(axis=1)[0] * 100)

    return pred_label, confidence
