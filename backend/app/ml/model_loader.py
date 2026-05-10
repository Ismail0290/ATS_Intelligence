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
_ml_models: dict[str, dict[str, Any]] = {}
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


def load_dl_model(folder_path: Path) -> tuple[Any, str, dict]:
    """
    Auto-detects and loads a DL model from a folder.
    Returns (model, framework, artifacts)
    """
    model = None
    framework = "unknown"
    artifacts = {}
    
    # Load artifacts like tokenizer, scaler, label_encoder inside the folder
    for pkl_file in folder_path.glob("*.pkl"):
        name_lower = pkl_file.name.lower()
        if "tokenizer" in name_lower:
            artifacts["tokenizer"] = joblib.load(pkl_file)
        elif "scaler" in name_lower:
            artifacts["scaler"] = joblib.load(pkl_file)
        elif "label_encoder" in name_lower or "encoder" in name_lower:
            artifacts["label_encoder"] = joblib.load(pkl_file)

    keras_files = list(folder_path.glob("*.keras")) + list(folder_path.glob("*.h5"))
    torch_files = list(folder_path.glob("*.pt")) + list(folder_path.glob("*.pth"))
    pkl_files = list(folder_path.glob("*.pkl")) + list(folder_path.glob("*.joblib"))
    
    # Filter out known artifact names from pkl_files
    model_pkls = [p for p in pkl_files if not any(x in p.name.lower() for x in ["tokenizer", "scaler", "encoder"])]

    if keras_files:
        from tensorflow.keras.models import load_model
        model = load_model(keras_files[0])
        framework = "tensorflow"
    elif torch_files:
        import torch
        model = torch.load(torch_files[0], map_location=torch.device('cpu'))
        if hasattr(model, 'eval'):
            model.eval()
        framework = "torch"
    elif model_pkls:
        model = joblib.load(model_pkls[0])
        framework = "sklearn"
        
    return model, framework, artifacts


def load_ml_models(model_dir: str = "./models"):
    global _ml_models, _scaler, _label_encoder
    model_dir = Path(model_dir)

    # 1. Load original sklearn models
    names = ["logistic_regression", "random_forest", "xgboost", "mlp_neural_net"]
    for name in names:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            try:
                model = joblib.load(path)
                _ml_models[name] = {"model": model, "framework": "sklearn", "artifacts": {}}
                logger.info(f"Loaded sklearn model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load sklearn model {name}: {e}")
        else:
            logger.warning(f"Model not found: {path}")

    # load global scaler/encoder
    scaler_path = model_dir / "scaler.pkl"
    le_path = model_dir / "label_encoder.pkl"
    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
    if le_path.exists():
        _label_encoder = joblib.load(le_path)

    # 2. Dynamically discover and load DL models
    project_root = Path(os.getcwd())
    if project_root.name == "backend":
        project_root = project_root.parent
        
    dl_folders = ["dnn_classifier", "lstm_classifier", "gru_classifier", "transformer_classifier"]
    
    for folder_name in dl_folders:
        folder_path = project_root / folder_name
        
        if folder_path.exists() and folder_path.is_dir():
            try:
                model, framework, artifacts = load_dl_model(folder_path)
                if model is not None:
                    # Registry key is "dnn" from "dnn_classifier"
                    reg_name = folder_name.replace("_classifier", "")
                    _ml_models[reg_name] = {"model": model, "framework": framework, "artifacts": artifacts}
                    logger.info(f"Loaded DL model: {reg_name} from {folder_path} (Framework: {framework})")
                else:
                    logger.warning(f"No valid model file found in {folder_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {folder_path}: {e}")
        else:
            logger.warning(f"Model folder not found: {folder_path}")

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
    registry_entry = _ml_models.get(model_name)
    if registry_entry is None or _scaler is None or _label_encoder is None:
        logger.warning(f"Prediction rejected: Model '{model_name}' not found or artifacts missing.")
        return None, 0.0

    model = registry_entry["model"]
    framework = registry_entry["framework"]
    artifacts = registry_entry.get("artifacts", {})
    
    scaler = artifacts.get("scaler", _scaler)
    le = artifacts.get("label_encoder", _label_encoder)
    tokenizer = artifacts.get("tokenizer")

    # 1. Base Tabular Extraction
    X_tabular = np.array([[features.get(col, 0.0) for col in FEATURE_COLS]], dtype=np.float64)
    X_tabular = np.nan_to_num(X_tabular, nan=0.0)
    X_scaled = scaler.transform(X_tabular) if scaler else X_tabular

    # 2. Input Routing & Preprocessing
    X_input = None
    input_type = "unknown"
    path_name = "unknown"

    try:
        if model_name in ["logistic_regression", "random_forest", "xgboost", "mlp_neural_net"]:
            X_input = X_scaled
            input_type = str(X_input.dtype)
            path_name = "Tabular Features"

        elif model_name == "dnn":
            cand_emb = features.get("cand_emb")
            if cand_emb is None:
                cand_emb = np.zeros((1, 384), dtype=np.float32)
            else:
                if len(cand_emb.shape) == 1:
                    cand_emb = cand_emb.reshape(1, -1)
            
            X_input = np.concatenate([X_scaled, cand_emb], axis=1).astype(np.float32)
            input_type = str(X_input.dtype)
            path_name = "Combined Embedding Features"

        elif model_name in ["lstm", "gru"]:
            if tokenizer is None:
                logger.error(f"Prediction rejected: Missing Keras tokenizer for sequence model '{model_name}'.")
                return None, 0.0
                
            text = features.get("candidate_text", "")
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([text])
            X_input = pad_sequences(seq, maxlen=500).astype(np.int64)
            input_type = str(X_input.dtype)
            path_name = "Sequence Tokens"

        elif model_name == "transformer":
            from transformers import AutoTokenizer
            import torch
            
            # Assume tokenizer is saved in transformer_classifier/ as huggingface format
            tokenizer_path = Path(os.getcwd()) / "transformer_classifier"
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            except Exception as e:
                logger.warning(f"Could not load HuggingFace tokenizer from {tokenizer_path}, falling back to bert-base-uncased. Error: {e}")
                hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                
            text = features.get("candidate_text", "")
            inputs = hf_tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            
            X_input = inputs
            input_type = "dict[str, torch.Tensor]"
            path_name = "Transformer Contextual Inputs"

        else:
            X_input = X_scaled
            input_type = str(X_input.dtype)
            path_name = "Default Tabular"

        # Log Preprocessing
        shape_str = str({k: str(v.shape) for k, v in X_input.items()}) if isinstance(X_input, dict) else str(X_input.shape)
        logger.info(
            f"Inference -> Model: {model_name} | Path: {path_name} | Shape: {shape_str} | "
            f"Dtype: {input_type} | Framework: {framework}"
        )

        # 3. Inference Execution
        if framework == "sklearn":
            pred_enc = model.predict(X_input)
            pred_label = le.inverse_transform(pred_enc)[0]
            confidence = float(model.predict_proba(X_input).max(axis=1)[0] * 100)
            
        elif framework == "tensorflow":
            preds = model.predict(X_input, verbose=0)
            if preds.shape[1] == 1:
                prob = preds[0][0]
                pred_class = 1 if prob > 0.5 else 0
                confidence = float(prob if pred_class == 1 else 1.0 - prob) * 100
                pred_label = le.inverse_transform([pred_class])[0]
            else:
                pred_class = np.argmax(preds, axis=1)[0]
                confidence = float(np.max(preds, axis=1)[0] * 100)
                pred_label = le.inverse_transform([pred_class])[0]
                
        elif framework == "torch":
            import torch
            with torch.no_grad():
                if isinstance(X_input, dict):
                    outputs = model(**X_input)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                else:
                    X_tensor = torch.tensor(X_input)
                    if X_tensor.dtype != torch.int64 and X_tensor.dtype != torch.float32:
                        X_tensor = X_tensor.float()
                    logits = model(X_tensor)
                
                if len(logits.shape) > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)
                    pred_label = le.inverse_transform([pred_class.item()])[0]
                    confidence = float(confidence.item() * 100)
                else:
                    prob = torch.sigmoid(logits).item()
                    pred_class = 1 if prob > 0.5 else 0
                    confidence = float(prob if pred_class == 1 else 1.0 - prob) * 100
                    pred_label = le.inverse_transform([pred_class])[0]
        else:
            logger.error(f"Unknown framework '{framework}' for model '{model_name}'.")
            return None, 0.0

        logger.info(f"Prediction Success -> Model: {model_name} | Label: {pred_label} | Prob: {confidence:.2f}%")
        return pred_label, confidence
            
    except Exception as e:
        logger.error(f"Prediction error for model {model_name}: {e}")
        return None, 0.0
