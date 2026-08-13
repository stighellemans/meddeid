# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/cache/meddeid/huggingface \
    MEDDEID_MODEL=stighellemans/meddeid-dutch-synth \
    MEDDEID_BACKEND=torch \
    MEDDEID_DEVICE=cpu \
    MEDDEID_WINDOW_BATCH_SIZE=32 \
    MEDDEID_WORKERS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PORT=8000

COPY pyproject.toml README.md LICENSE NOTICE /opt/meddeid/
COPY src /opt/meddeid/src

RUN python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "torch>=2.2" \
    && python -m pip install --no-cache-dir "/opt/meddeid[server]" \
    && useradd --create-home --uid 10001 meddeid \
    && mkdir -p /var/cache/meddeid \
    && chown -R meddeid:meddeid /var/cache/meddeid

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4))"

CMD ["sh", "-c", "exec uvicorn meddeid.server:create_app --factory --host 0.0.0.0 --port ${PORT} --workers ${MEDDEID_WORKERS}"]
