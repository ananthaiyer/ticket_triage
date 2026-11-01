FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source and pre-trained baseline models
COPY src/ src/
COPY models/ models/

# Make 'triage' importable without a package install step
ENV PYTHONPATH=/app/src

# Render injects PORT at runtime; 8000 is the local dev default
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn triage.api:app --host 0.0.0.0 --port $PORT"]
