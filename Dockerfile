# Multi-stage Dockerfile for Customer Respond Agent
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts in .local are on PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/policies/ ./data/policies/
COPY .env.example .env.example
COPY pyproject.toml .

# Create data directories
RUN mkdir -p data/chroma

# Initialize database and ingest policies at build time
RUN python -c "from app.db import init_db; init_db()" && \
    python -c "from app.rag.ingest import ingest_policies; ingest_policies()"

# Expose ports
EXPOSE 8000 8501

# Default command runs API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]