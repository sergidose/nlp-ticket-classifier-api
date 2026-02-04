"""
Entrenamiento de un clasificador de tickets con:
TF-IDF (1-2 grams) + LogisticRegression.

Guarda:
- models/ticket_model.joblib (pipeline completo: vectorizador + modelo)
- models/metrics.json (accuracy y F1 macro + info de train/test)

Permite override por variables de entorno:
DATA_PATH, MODEL_PATH, METRICS_PATH
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Raíz del repo
ROOT = Path(__file__).resolve().parents[1]

# Paths por defecto (override con env vars)
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT / "data" / "raw" / "tickets.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "ticket_model.joblib")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(ROOT / "models" / "metrics.json")))


@dataclass
class Metrics:
    """Métricas básicas para mostrar en /model-info y en README."""

    accuracy: float
    f1_macro: float
    n_train: int
    n_test: int
    trained_at_utc: str


def train_and_save(data_path: Path, model_path: Path, metrics_path: Path) -> Metrics:
    """
    Entrena el pipeline y guarda modelo + métricas.
    Se usa tanto en CLI como en tests (en carpeta temporal).
    """
    df = pd.read_csv(data_path)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # stratify=y para mantener proporción de clases en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: vectorización + modelo (se guarda todo junto)
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    m = Metrics(
        accuracy=float(accuracy_score(y_test, pred)),
        f1_macro=float(f1_score(y_test, pred, average="macro")),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        trained_at_utc=datetime.now(UTC).isoformat(),
    )

    # crea carpetas si no existen
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # guarda modelo (pipeline) y métricas
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(asdict(m), ensure_ascii=False), encoding="utf-8")

    return m


def main() -> None:
    # si no hay dataset, obligamos al usuario a generarlo
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run: python scripts/generate_data.py"
        )

    m = train_and_save(DATA_PATH, MODEL_PATH, METRICS_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Metrics saved to {METRICS_PATH}")
    print(m)


if __name__ == "__main__":
    main()
