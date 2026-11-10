# syntax=docker/dockerfile:1.7
#
# ImpactLens: review UI + inference API in one image, serving on port 7860.
# Built and run on Hugging Face Docker Spaces or any container host.
#
#   docker build -t impactlens .
#   docker run --rm -p 7860:7860 --env-file .env impactlens
#
# No secret is needed at build time and none is baked in. API_KEY and HF_TOKEN
# are supplied at run time by the host's secret store.

# Base images pinned by digest (python 3.12.16 / node 22.23.4, both built 2026-11-05).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ARG NODE_IMAGE=node:22-bookworm-slim@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d

# ---------------------------------------------------------------- frontend
FROM ${NODE_IMAGE} AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
# Optional build secret `extra_ca`: a CA bundle for builds behind a TLS-intercepting
# proxy. Mounted only for this step, never written to a layer, ignored when absent.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi; \
    npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------- python dependencies
FROM ${PYTHON_IMAGE} AS python-deps
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install uv==0.11.32
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
# Exactly the locked versions (CPU PyTorch wheels), without dev tools.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-editable

# ----------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
# Hugging Face Spaces runs containers as UID 1000; use the same everywhere.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin user
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    HOST=0.0.0.0 \
    PORT=7860 \
    FRONTEND_DIST=/app/frontend/dist \
    DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2
COPY --from=python-deps /opt/venv /opt/venv
COPY --from=frontend --chown=user:user /build/dist /app/frontend/dist
# Optional: a checkpoint placed in models/ before building can be served with
# MODEL_PATH=/app/models/<file>.pt. The recommended path is a pinned Hub revision.
COPY --chown=user:user models/ /app/models/
COPY --chown=user:user schemas/ /app/schemas/
WORKDIR /app
USER user
EXPOSE 7860
# /health answers 200 whenever the process serves; /ready is 503 until a model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '7860')}/health\", timeout=4)"]
CMD ["python", "-m", "impactlens.service"]
