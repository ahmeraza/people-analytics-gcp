# api/Dockerfile
# Multi-stage build for a lean Cloud Run container

# ── Stage 1: Dependencies ───────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy app code
COPY --chown=appuser:appuser main.py .

# Cloud Run injects PORT; default to 8080
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

EXPOSE 8080

# Uvicorn: single worker (Cloud Run scales via instances, not workers)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
