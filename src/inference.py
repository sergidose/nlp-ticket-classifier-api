"""
Funciones de inferencia:
- load_model: carga el pipeline entrenado
- load_metrics: lee metrics.json si existe
- predict_one: devuelve (label, confidence)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "ticket_model.joblib")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(ROOT / "models" / "metrics.json")))


def load_model(model_path: Path = MODEL_PATH):
    """Carga el pipeline (TF-IDF + modelo) guardado en joblib."""
    return joblib.load(str(model_path))


def load_metrics(metrics_path: Path = METRICS_PATH) -> Dict[str, Any]:
    """Devuelve dict con métricas. Si no existe, devuelve {}."""
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def predict_one(model, text: str) -> Tuple[str, float]:
    """
    Predice una categoría. Si el modelo soporta predict_proba, devuelve también confidence.
    """
    label = model.predict([text])[0]

    # algunos modelos podrían no tener predict_proba
    if not hasattr(model, "predict_proba"):
        return str(label), 0.0

    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = classes.index(label)
    return str(label), float(probs[idx])
