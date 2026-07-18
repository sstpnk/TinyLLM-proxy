# =============================================================================
# Dockerfile — TinyLLM
# =============================================================================
# Multi-stage build for minimal final image size.
# =============================================================================

# -- builder -----------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# -- runtime -----------------------------------------------------------------
FROM python:3.11-slim

# Create a non-root user
RUN addgroup --system --gid 1001 tinyllm \
    && adduser --system --uid 1001 --gid 1001 tinyllm

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY tinyllm/ tinyllm/
COPY config.yaml .

# Environment
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TINYLLM_API_KEYS="" \
    OPENCODE_ZEN_API_KEY="" \
    OPENROUTER_API_KEY="" \
    ZAI_API_KEY=""

EXPOSE 4000

USER tinyllm

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness')" || exit 1

ENTRYPOINT ["python", "-m", "tinyllm"]
