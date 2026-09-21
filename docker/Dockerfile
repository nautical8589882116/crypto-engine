# Crypto Mission Control brain — Railway / Docker.
# Pure-Python (no Rust kernel needed for crypto), so this builds on any Python 3.12.

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/wheels

# ---- Runtime stage ----
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md ./
COPY scripts ./scripts
COPY dashboard ./dashboard
# The trained ONNX model the engine worker loads (ENGINE_MODEL). Small (~100KB);
# committed so the image is self-contained.
COPY models ./models
COPY --from=builder /build/wheels /build/wheels
RUN pip install --no-cache-dir /build/wheels/*.whl \
    && pip install --no-cache-dir websockets python-binance pydantic pydantic-settings onnxruntime httpx numpy requests

# Persistent tape/corpus store. No VOLUME (Railway rejects it); mount a Volume
# at /data in Railway (Service > Settings > Volumes, mount path /data).
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1 \
    RECORD_DIR=/data/tapes \
    ENGINE_MODEL=/app/models/mamba_tbar.onnx

EXPOSE 8000
CMD ["python", "-m", "quant_crypto.entrypoint"]
