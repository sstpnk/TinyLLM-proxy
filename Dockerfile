# =============================================================================
# Dockerfile — TinyLLM
# =============================================================================
# Multi-stage build for minimal final image size.
# =============================================================================

# -- builder -----------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -- runtime -----------------------------------------------------------------
FROM python:3.11-slim

# Create a non-root user
RUN addgroup --system --gid 1001 tinyllm \
    && adduser --system --uid 1001 --gid 1001 tinyllm

WORKDIR /app

# Copy installed site-packages from builder (system-wide)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy application code
COPY tinyllm/ tinyllm/
COPY config.yaml .

# Environment
ENV PYTHONUNBUFFERED=1

EXPOSE 4000

USER tinyllm

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import socket, urllib.request; s=socket.socket(); s.settimeout(8); s.connect(('127.0.0.1',4000)); s.close(); r=urllib.request.urlopen('http://127.0.0.1:4000/health/liveliness', timeout=8); assert r.status==200" || exit 1

ENTRYPOINT ["python", "-m", "tinyllm"]
