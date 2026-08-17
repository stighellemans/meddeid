# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

FROM ${PYTHON_IMAGE} AS builder

ARG MEDDEID_CORE_COMMIT=9b51c5b93aadfd9f59014e136a3f72ff38f7ad55
ARG MEDDEID_LANGUAGE_NL_COMMIT=886d102dcf36cec8d86173e8eb4d3471cde20f45
ARG MEDDEID_MODEL_ID=stighellemans/meddeid-dutch-synth
ARG MEDDEID_MODEL_REVISION=cbe68a93e808c919de97052dc6ef031d2dce4a61
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    GIT_TERMINAL_PROMPT=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY constraints/container.txt /tmp/container-constraints.txt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
      pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 packaging==26.3 \
    && python -m pip install \
      --constraint /tmp/container-constraints.txt \
      --index-url "${TORCH_INDEX_URL}" \
      "torch>=2.2"

RUN python -m pip install \
    --constraint /tmp/container-constraints.txt \
    --retries 5 \
    --timeout 60 \
    "meddeid-core @ git+https://github.com/stighellemans/meddeid-core.git@${MEDDEID_CORE_COMMIT}" \
    "meddeid-language-nl @ git+https://github.com/stighellemans/meddeid-language-nl.git@${MEDDEID_LANGUAGE_NL_COMMIT}"

RUN python -m pip install \
      --constraint /tmp/container-constraints.txt huggingface-hub \
    && hf download "${MEDDEID_MODEL_ID}" \
      --revision "${MEDDEID_MODEL_REVISION}" \
      --local-dir /opt/meddeid-model

WORKDIR /build/meddeid
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN python -m pip install --constraint /tmp/container-constraints.txt '.[server]' \
    && python -m pip check \
    && python -c "from meddeid.bundle import load_model_bundle; load_model_bundle('/opt/meddeid-model/bundle.json', validate_package=True)" \
    && python -m pip uninstall --yes pip setuptools wheel


FROM ${PYTHON_IMAGE} AS runtime

ARG MEDDEID_CORE_COMMIT=9b51c5b93aadfd9f59014e136a3f72ff38f7ad55
ARG MEDDEID_LANGUAGE_NL_COMMIT=886d102dcf36cec8d86173e8eb4d3471cde20f45
ARG MEDDEID_MODEL_ID=stighellemans/meddeid-dutch-synth
ARG MEDDEID_MODEL_REVISION=cbe68a93e808c919de97052dc6ef031d2dce4a61
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="MedDeID API" \
      org.opencontainers.image.description="Local Dutch clinical-text de-identification API" \
      org.opencontainers.image.source="https://github.com/stighellemans/meddeid" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.meddeid.core-revision="${MEDDEID_CORE_COMMIT}" \
      io.meddeid.language-nl-revision="${MEDDEID_LANGUAGE_NL_COMMIT}" \
      io.meddeid.model-id="${MEDDEID_MODEL_ID}" \
      io.meddeid.model-revision="${MEDDEID_MODEL_REVISION}"

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/meddeid \
    HF_HOME=/var/cache/meddeid/huggingface \
    MEDDEID_MODEL=/opt/meddeid-model \
    MEDDEID_REVISION=${MEDDEID_MODEL_REVISION} \
    MEDDEID_OFFLINE=true \
    MEDDEID_BACKEND=torch \
    MEDDEID_DEVICE=cpu \
    MEDDEID_WINDOW_BATCH_SIZE=32 \
    MEDDEID_WORKERS=1 \
    MEDDEID_DOCS_ENABLED=false \
    MEDDEID_UI_ENABLED=false \
    MEDDEID_REQUIRE_API_KEY=false \
    MEDDEID_MAX_INPUT_CHARS=20000 \
    MEDDEID_MAX_BATCH_DOCUMENTS=32 \
    MEDDEID_MAX_BATCH_CHARS=200000 \
    MEDDEID_MAX_REQUEST_BYTES=2000000 \
    MEDDEID_MAX_CONCURRENT_REQUESTS=1 \
    MEDDEID_QUEUE_TIMEOUT_SECONDS=30 \
    MEDDEID_ACCESS_LOG=true \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PORT=8000

RUN apt-get update \
    && apt-get upgrade --yes --no-install-recommends \
    && /usr/local/bin/python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/meddeid-model /opt/meddeid-model
COPY LICENSE NOTICE /licenses/meddeid/

RUN groupadd --gid 10001 meddeid \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/meddeid meddeid \
    && mkdir -p /var/cache/meddeid \
    && chown -R 10001:10001 /var/cache/meddeid /home/meddeid \
    && chmod -R a-w /opt/venv /opt/meddeid-model /licenses

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4))"

CMD ["meddeid-server"]
