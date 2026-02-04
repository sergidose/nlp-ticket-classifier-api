"""
Arranque “smart” para Docker:
- Si no existe dataset -> lo genera
- Si no existe modelo/metrics -> entrena
- Luego levanta Uvicorn

IMPORTANTE:
Cuando ejecutas "python scripts/start_api.py", Python puede NO incluir /app en sys.path.
Por eso forzamos sys.path y chdir a la raíz del repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # ✅ Asegura imports tipo "app.main" y "src.train"
    sys.path.insert(0, str(root))
    os.chdir(root)

    data_path = Path(os.getenv("DATA_PATH", str(root / "data" / "raw" / "tickets.csv")))
    model_path = Path(os.getenv("MODEL_PATH", str(root / "models" / "ticket_model.joblib")))
    metrics_path = Path(os.getenv("METRICS_PATH", str(root / "models" / "metrics.json")))
    port = int(os.getenv("PORT", "8001"))

    # 1) dataset
    if not data_path.exists():
        from scripts.generate_data import main as gen_main

        gen_main()

    # 2) modelo
    if not model_path.exists() or not metrics_path.exists():
        from src.train import train_and_save

        train_and_save(data_path, model_path, metrics_path)

    # 3) api
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
