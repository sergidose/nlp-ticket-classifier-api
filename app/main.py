"""
FastAPI app:
- GET /health
- GET /model-info
- POST /predict

Carga el modelo al arrancar si existe (startup).
Si no existe, /predict devuelve 503 con mensaje para entrenar.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.inference import load_metrics, load_model, predict_one

# Rutas configurables por env vars (útil para tests/CI/Docker)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ticket_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "models/metrics.json"))


class PredictIn(BaseModel):
    """Input del endpoint /predict."""

    text: str = Field(..., min_length=1, description="Texto del ticket a clasificar")


class PredictOut(BaseModel):
    """Output del endpoint /predict."""

    label: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler (reemplaza a @app.on_event("startup")).

    Carga modelo y métricas al arrancar la app, y los deja en app.state
    para que los endpoints puedan acceder.
    """
    # Carga modelo (si no existe, la API responde 503 en /predict)
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        model = None

    # Carga métricas (si no existen, devolvemos {})
    try:
        metrics = load_metrics(METRICS_PATH)
    except FileNotFoundError:
        metrics = {}

    app.state.model = model
    app.state.metrics = metrics
    app.state.model_path = str(MODEL_PATH)
    app.state.metrics_path = str(METRICS_PATH)

    yield

    # Cleanup opcional
    app.state.model = None


app = FastAPI(
    title="NLP Ticket Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    return {
        "loaded": request.app.state.model is not None,
        "model_path": request.app.state.model_path,
        "metrics_path": request.app.state.metrics_path,
        "metrics": request.app.state.metrics,
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn, request: Request) -> PredictOut:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    label, confidence = predict_one(model, payload.text)
    return PredictOut(label=label, confidence=float(confidence))
