# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

FROM ${PYTHON_IMAGE} AS builder

ARG MEDDEID_CORE_COMMIT=abcc04b92f684d161a2cc66a1d01cd92968008bc
ARG MEDDEID_LANGUAGE_EN_COMMIT=cf624a922c83bcd0a53bc7ca284d191ded226282
ARG MEDDEID_LANGUAGE_NL_COMMIT=399db287c798e06f38d45557bd78748dd9d68b55
ARG MEDDEID_MODEL_ID=stighellemans/meddeid-dutch-synth
ARG MEDDEID_MODEL_REVISION=1f20655454dcbd042647cacdfff6b6802a970959
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

RUN python -m pip install \
    --constraint /tmp/container-constraints.txt \
    --retries 5 \
    --timeout 60 \
    "meddeid-core @ git+https://github.com/stighellemans/meddeid-core.git@${MEDDEID_CORE_COMMIT}" \
    "meddeid-language-en @ git+https://github.com/stighellemans/meddeid-language-en.git@${MEDDEID_LANGUAGE_EN_COMMIT}" \
    "meddeid-language-nl @ git+https://github.com/stighellemans/meddeid-language-nl.git@${MEDDEID_LANGUAGE_NL_COMMIT}"

RUN python -m pip install \
      --constraint /tmp/container-constraints.txt huggingface-hub \
    && hf download "${MEDDEID_MODEL_ID}" \
      --revision "${MEDDEID_MODEL_REVISION}" \
      --local-dir /opt/meddeid-model \
    && find /opt/meddeid-model/.cache -depth -delete

WORKDIR /build/meddeid
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN python -m pip install --constraint /tmp/container-constraints.txt '.[server]' \
    && python -m pip check \
    && python -c "from meddeid.bundle import load_model_bundle; load_model_bundle('/opt/meddeid-model/bundle.json', validate_package=True)" \
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

ARG MEDDEID_CORE_COMMIT=abcc04b92f684d161a2cc66a1d01cd92968008bc
ARG MEDDEID_LANGUAGE_EN_COMMIT=cf624a922c83bcd0a53bc7ca284d191ded226282
ARG MEDDEID_LANGUAGE_NL_COMMIT=399db287c798e06f38d45557bd78748dd9d68b55
ARG MEDDEID_MODEL_ID=stighellemans/meddeid-dutch-synth
ARG MEDDEID_MODEL_REVISION=1f20655454dcbd042647cacdfff6b6802a970959
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
    MEDDEID_MODEL=/opt/meddeid-model \
    MEDDEID_REVISION=${MEDDEID_MODEL_REVISION} \
    MEDDEID_OFFLINE=true \
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
COPY --from=builder /opt/meddeid-model /opt/meddeid-model
COPY LICENSE NOTICE /licenses/meddeid/

RUN python -c "import torch; expected='${MEDDEID_CUDA_VERSION}'; actual=torch.version.cuda or 'none'; assert actual == expected, f'expected CUDA {expected}, found {actual}'"

RUN groupadd --gid 10001 meddeid \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/meddeid meddeid \
    && mkdir -p /var/cache/meddeid/huggingface \
    && chown -R 10001:10001 /var/cache/meddeid /home/meddeid \
    && test "$(stat -c %U /opt/meddeid-model)" = root

# Keep release-only metadata after filesystem construction so changing a tag,
# source revision, or build date does not invalidate multi-gigabyte layers.
ARG MEDDEID_VERSION=0.3.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="MedDeID API" \
      org.opencontainers.image.description="Local Dutch clinical-text de-identification API" \
      org.opencontainers.image.source="https://github.com/stighellemans/meddeid" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${MEDDEID_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.meddeid.core-revision="${MEDDEID_CORE_COMMIT}" \
      io.meddeid.language-en-revision="${MEDDEID_LANGUAGE_EN_COMMIT}" \
      io.meddeid.language-nl-revision="${MEDDEID_LANGUAGE_NL_COMMIT}" \
      io.meddeid.model-id="${MEDDEID_MODEL_ID}" \
      io.meddeid.model-revision="${MEDDEID_MODEL_REVISION}" \
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
