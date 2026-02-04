![CI](https://github.com/sergidose/nlp-ticket-classifier-api/actions/workflows/ci.yml/badge.svg)

# NLP Ticket Classifier API

Clasificador de tickets de soporte **end-to-end**: genera dataset sintético reproducible, entrena un modelo NLP clásico (**TF-IDF + Logistic Regression**) y sirve predicciones con **FastAPI**.

> 🎯 Objetivo de portfolio: mostrar un proyecto completo “data → train → model → API → tests → Docker”.

---

## Features

- ✅ Dataset sintético reproducible (`scripts/generate_data.py`)
- ✅ Training pipeline NLP (TF-IDF 1–2 grams + LogisticRegression) (`src/train.py`)
- ✅ Artefactos: `models/ticket_model.joblib` + `models/metrics.json`
- ✅ FastAPI endpoints: `/health`, `/model-info`, `/predict`
- ✅ Tests con `pytest` (incluye test end-to-end con entrenamiento en carpeta temporal)
- ✅ Code quality: `ruff` + `pre-commit`
- ✅ Docker + Docker Compose (arranque “smart”: si falta dataset/modelo, se generan/entrenan)

---

## Quickstart (Docker) ✅ recomendado

Requisitos: Docker + Docker Compose.

```bash
docker compose up --build
```

Abre:

- http://127.0.0.1:8001/docs

- http://127.0.0.1:8001/model-info

El contenedor arranca con un script que:

1. "genera dataset si no existe,"
2. "entrena si no existe el modelo/métricas,"
3. "levanta la API."

---

## Endpoints

- GET /health

    Healthcheck para comprobar que la API está viva.

    Response:
    ```json
    { "status": "ok" }
    ```

- GET /model-info

    Información del modelo (ruta, métricas y si está cargado).

    Campos típicos:

        - loaded: true/false
        - metrics: accuracy, f1_macro, etc.

## POST /predict

Clasifica un texto de ticket y devuelve etiqueta + confianza.

Request:
```json
{ "text": "No puedo iniciar sesión, no me llega el email de verificación" }
```

Response:
```json
{ "label": "account_access", "confidence": 0.93 }
```

Etiquetas esperadas:

    - billing
    - technical_support
    - account_access
    - cancellation
    - shipping_delivery
    - general_inquiry

## Example requests

- Opción A) Desde Swagger UI (fácil)

    http://127.0.0.1:8001/docs

    POST /predict → “Try it out”

- Opción B) cURL (Linux / WSL / Mac)

    ```bash
    curl -X POST "http://127.0.0.1:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{"text":"Me han cobrado dos veces este mes y quiero revisar mi factura"}'
    ```

    ```bash
    curl -X POST "http://127.0.0.1:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{"text":"Mi pedido no ha llegado y el tracking no se actualiza"}'
    ```

- Opción C) cURL (Windows CMD)

    ```bash
    curl -X POST "http://127.0.0.1:8001/predict" ^
        -H "Content-Type: application/json" ^
        -d "{\"text\":\"No puedo iniciar sesión, no me llega el email de verificación\"}"
    ```

## Local setup (Windows)

1) Crear venv + instalar dependencias

    ```bat
    py -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
    ```

2) Generar dataset

    ```bat
    .venv\Scripts\python scripts\generate_data.py
    ```

    Salida esperada:

    - data/raw/tickets.csv

3) Entrenar

    ```bat
    .venv\Scripts\python -m src.train
    ```

    Salida esperada:

    - models/ticket_model.joblib

    - models/metrics.json

4) Ejecutar API

    ```bat
    .venv\Scripts\python -m uvicorn app.main:app --reload --port 8001
    ```

    Abrir:

    - http://127.0.0.1:8001/docs

## Tests

```bat
.venv\Scripts\python -m pytest -q
```

## Lint / format (Ruff) + pre-commit

- Instalar hooks:

    ```bat
    .venv\Scripts\pre-commit install
    ```

- Ejecutar sobre todo el repo:

    ```bat
    .venv\Scripts\pre-commit run --all-files
    ```

## Configuration (env vars)

Puedes cambiar rutas/puerto usando variables de entorno:

- PORT (default: 8001)
- DATA_PATH (default: data/raw/tickets.csv)
- MODEL_PATH (default: models/ticket_model.joblib)
- METRICS_PATH (default: models/metrics.json)

Ejemplo (Windows PowerShell):

```powershell
$env:PORT="8001"
$env:MODEL_PATH="E:\nlp-ticket-classifier-api\models\ticket_model.joblib"
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8001
```

## Project structure

- app/
    FastAPI app y endpoints.

- src/
    Lógica de ML/NLP (train + inferencia).

- scripts/
    Scripts auxiliares:
    - generate_data.py crea dataset sintético
    - start_api.py (Docker) genera/entrena si hace falta y arranca la API

- data/raw/
    Dataset (CSV).

- models/
    Artefactos del modelo y métricas.

- tests/
    Tests end-to-end y sanity.

## Notes

- El dataset es sintético por diseño (portfolio): evita dependencias externas y hace el demo   reproducible.

- Si quieres usar datos reales:
    - reemplaza scripts/generate_data.py por un download_data.py o un import desde tu fuente,
    - mantén el resto del pipeline igual.
