FROM python:3.12-slim

WORKDIR /app

# Copiamos requirements primero para cachear capas
COPY requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copiamos el código
COPY . .

# Por defecto: levantar API (pero antes entrenar si falta modelo)
CMD ["python", "scripts/start_api.py"]
