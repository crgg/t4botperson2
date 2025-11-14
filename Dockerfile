FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema (psycopg2, curl para healthcheck, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Primero requirements para aprovechar cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

EXPOSE 8000

# ⚠️ Si tu app está en main.py cambia "t4ever_api:app" por "main:app"
CMD ["uvicorn", "t4ever_api:app", "--host", "0.0.0.0", "--port", "8000"]
