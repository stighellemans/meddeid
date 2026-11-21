# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

FROM scratch AS component_wheels

FROM ${PYTHON_IMAGE} AS builder
ARG MEDDEID_COMPONENT_SOURCE=git

ARG MEDDEID_CORE_COMMIT=fa8f437c836cc75f02fb0e7f0d1039e0fa5a03fb
ARG MEDDEID_LANGUAGE_EN_COMMIT=c2c1df0bca705263bf0d208f5760a1fc56e5f094
ARG MEDDEID_LANGUAGE_NL_COMMIT=8debae6a6c385b3571c0aa11b3046ae856dde8ea
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MEDDEID_ACCELERATOR=cpu
ARG MEDDEID_CUDA_VERSION=none
ARG MEDDEID_TORCH_PRECISION=fp32
ARG MEDDEID_TORCH_COMPILE_MODE=off

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
      "torch==${TORCH_VERSION}" \
    && python -c "import torch; expected='${MEDDEID_CUDA_VERSION}'; actual=torch.version.cuda or 'none'; assert actual == expected, f'expected CUDA {expected}, found {actual}'" \
    && test "${MEDDEID_ACCELERATOR}" = cpu -o "${MEDDEID_ACCELERATOR}" = cuda

COPY deploy/install-rendering-components.sh /tmp/install-rendering-components.sh
RUN --mount=type=bind,from=component_wheels,target=/component-wheels sh /tmp/install-rendering-components.sh

WORKDIR /build/meddeid
COPY pyproject.toml README.md LICENSE NOTICE MEDDEID-PRIVATE-FINE-TUNING-EXCEPTION-1.0.txt ./
COPY src ./src

RUN python -m pip install --constraint /tmp/container-constraints.txt '.[server]' \
    && python -m pip check \
    && python -m pip uninstall --yes hf-xet pip setuptools wheel \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf \
      /opt/venv/lib/python*/site-packages/torch/include \
      /opt/venv/lib/python*/site-packages/torch/test \
    && find /opt/venv/lib/python*/site-packages/torch/bin \
      -depth -mindepth 1 ! -name torch_shm_manager -delete \
    && if [ "${MEDDEID_TORCH_COMPILE_MODE}" = off ]; then \
        find /opt/venv/lib/python*/site-packages/torch/lib -maxdepth 1 -type f \
          \( -name '*test*.so' -o -name libaoti_custom_ops.so \
          -o -name libbackend_with_compiler.so \) -delete; \
        if [ "${MEDDEID_ACCELERATOR}" = cuda ]; then \
          rm -rf /opt/venv/lib/python*/site-packages/triton; \
          find /opt/venv/lib/python*/site-packages/nvidia -type f -name '*.a' -delete; \
        fi; \
      fi


FROM ${PYTHON_IMAGE} AS runtime

ARG MEDDEID_CORE_COMMIT=fa8f437c836cc75f02fb0e7f0d1039e0fa5a03fb
ARG MEDDEID_LANGUAGE_EN_COMMIT=c2c1df0bca705263bf0d208f5760a1fc56e5f094
ARG MEDDEID_LANGUAGE_NL_COMMIT=8debae6a6c385b3571c0aa11b3046ae856dde8ea
ARG TORCH_VERSION=2.13.0
ARG MEDDEID_ACCELERATOR=cpu
ARG MEDDEID_CUDA_VERSION=none
ARG MEDDEID_DEVICE=cpu
ARG MEDDEID_TORCH_PRECISION=fp32
ARG MEDDEID_TORCH_COMPILE_MODE=off

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/meddeid \
    HF_HOME=/var/cache/meddeid/huggingface \
    MEDDEID_OFFLINE=false \
    MEDDEID_BACKEND=torch \
    MEDDEID_DEVICE=${MEDDEID_DEVICE} \
    MEDDEID_TORCH_PRECISION=${MEDDEID_TORCH_PRECISION} \
    MEDDEID_TORCH_COMPILE_MODE=${MEDDEID_TORCH_COMPILE_MODE} \
    MEDDEID_TORCH_COMPILE_DYNAMIC=true \
    MEDDEID_SERVING_PROFILE=latency \
    MEDDEID_MICROBATCH_ENABLED=auto \
    MEDDEID_WINDOW_BATCH_SIZE=32 \
    MEDDEID_WORKERS=1 \
    MEDDEID_DOCS_ENABLED=false \
    MEDDEID_UI_ENABLED=false \
    MEDDEID_REQUIRE_API_KEY=false \
    MEDDEID_MAX_INPUT_CHARS=20000 \
    MEDDEID_MAX_BATCH_DOCUMENTS=32 \
    MEDDEID_MAX_BATCH_CHARS=200000 \
    MEDDEID_MAX_REQUEST_BYTES=2000000 \
    MEDDEID_MAX_CONCURRENT_REQUESTS=auto \
    MEDDEID_QUEUE_TIMEOUT_SECONDS=30 \
    MEDDEID_ACCESS_LOG=true \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PORT=8000

RUN apt-get update \
    && apt-get upgrade --yes --no-install-recommends \
    && if [ "${MEDDEID_TORCH_COMPILE_MODE}" != off ]; then \
      apt-get install --yes --no-install-recommends gcc g++ libc6-dev; \
    fi \
    && /usr/local/bin/python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY LICENSE NOTICE MEDDEID-PRIVATE-FINE-TUNING-EXCEPTION-1.0.txt /licenses/meddeid/

RUN python -c "import torch; expected='${MEDDEID_CUDA_VERSION}'; actual=torch.version.cuda or 'none'; assert actual == expected, f'expected CUDA {expected}, found {actual}'"

RUN groupadd --gid 10001 meddeid \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/meddeid meddeid \
    && mkdir -p /var/cache/meddeid/huggingface /models/meddeid /model-source \
    && touch /models/meddeid/.mountpoint /model-source/.mountpoint \
    && chown -R 10001:10001 \
      /var/cache/meddeid /home/meddeid /models /model-source

# Keep release-only metadata after filesystem construction so changing a tag,
# source revision, or build date does not invalidate multi-gigabyte layers.
ARG MEDDEID_VERSION=0.4.3rc5
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="MedDeID API" \
      org.opencontainers.image.description="Model-independent local clinical-text de-identification API" \
      org.opencontainers.image.source="https://github.com/stighellemans/meddeid" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${MEDDEID_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.meddeid.core-revision="${MEDDEID_CORE_COMMIT}" \
      io.meddeid.language-en-revision="${MEDDEID_LANGUAGE_EN_COMMIT}" \
      io.meddeid.language-nl-revision="${MEDDEID_LANGUAGE_NL_COMMIT}" \
      io.meddeid.accelerator="${MEDDEID_ACCELERATOR}" \
      io.meddeid.torch-version="${TORCH_VERSION}" \
      io.meddeid.cuda-version="${MEDDEID_CUDA_VERSION}" \
      io.meddeid.torch-precision="${MEDDEID_TORCH_PRECISION}" \
      io.meddeid.torch-compile-mode="${MEDDEID_TORCH_COMPILE_MODE}"

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4))"

CMD ["meddeid-server"]
