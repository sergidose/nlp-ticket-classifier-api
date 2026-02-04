"""
FastAPI app:
- GET /health
- GET /model-info
- POST /predict

Carga el modelo al arrancar si existe (startup).
Si no existe, /predict devuelve 503 con mensaje para entrenar.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import METRICS_PATH, MODEL_PATH, load_metrics, load_model, predict_one

app = FastAPI(title="NLP Ticket Classifier API", version="0.1.0")

# Modelo global (se carga en startup)
_model = None


class PredictRequest(BaseModel):
    # validación sencilla: mínimo 3 chars
    text: str = Field(..., min_length=3, description="Ticket text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.on_event("startup")
def _startup() -> None:
    """
    Hook de arranque: si existe el modelo, lo cargamos.
    Si no existe, la API sigue viva pero /predict dará 503.
    """
    global _model
    if Path(MODEL_PATH).exists():
        _model = load_model(Path(MODEL_PATH))


@app.get("/health")
def health():
    """Healthcheck simple para Docker/K8s y tests."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Info útil para README y para depurar si está cargado."""
    return {
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "metrics": load_metrics(Path(METRICS_PATH)),
        "loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Devuelve label + confidence.
    Si el modelo no está cargado, 503 con instrucción.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m src.train")

    label, conf = predict_one(_model, req.text)
    return PredictResponse(label=label, confidence=conf)
