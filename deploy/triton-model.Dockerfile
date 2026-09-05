# syntax=docker/dockerfile:1.7

ARG TRITON_BASE_IMAGE=meddeid-triton-runtime:26.07-tensorrt-only
FROM ${TRITON_BASE_IMAGE}

ENV HOME=/tmp \
    CUDA_CACHE_PATH=/tmp/cuda-cache

ARG MODEL_REVISION
ARG BUNDLE_SHA256
ARG GPU_TARGET
ARG TRITON_STACK
ARG TENSORRT_VERSION
ARG TRITON_FULL_IMAGE
ARG TRITON_MIN_IMAGE
ARG TRITON_COMPOSE_REVISION
ARG TRT_PRECISION
ARG TRT_OUTPUT_PRECISION
ARG BUILD_MANIFEST_SHA256

COPY --from=model_repository . /models
COPY LICENSE NOTICE /licenses/meddeid/

RUN test -s /models/build-manifest.json \
    && test "$(sha256sum /models/build-manifest.json | cut -d' ' -f1)" = "${BUILD_MANIFEST_SHA256}" \
    && test "$(find /models -name model.plan -type f | wc -l)" -eq 1 \
    && test "$(stat -c %U /models)" = root

# Keep release-only metadata after the plan copy and validation layers.
ARG MEDDEID_VERSION=0.3.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="MedDeID TensorRT model server" \
      org.opencontainers.image.description="Target-specific MedDeID TensorRT plan served by NVIDIA Triton" \
      org.opencontainers.image.source="https://github.com/stighellemans/meddeid" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND LicenseRef-NVIDIA-Deep-Learning-Container" \
      org.opencontainers.image.version="${MEDDEID_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.meddeid.source-revision="${VCS_REF}" \
      io.meddeid.model-revision="${MODEL_REVISION}" \
      io.meddeid.bundle-sha256="${BUNDLE_SHA256}" \
      io.meddeid.gpu-target="${GPU_TARGET}" \
      io.meddeid.triton-stack="${TRITON_STACK}" \
      io.meddeid.tensorrt-version="${TENSORRT_VERSION}" \
      io.meddeid.triton-full-image="${TRITON_FULL_IMAGE}" \
      io.meddeid.triton-min-image="${TRITON_MIN_IMAGE}" \
      io.meddeid.triton-compose-revision="${TRITON_COMPOSE_REVISION}" \
      io.meddeid.tensorrt-precision="${TRT_PRECISION}" \
      io.meddeid.tensorrt-output-precision="${TRT_OUTPUT_PRECISION}" \
      io.meddeid.build-manifest-sha256="${BUILD_MANIFEST_SHA256}"

USER 10000:10000
EXPOSE 8000 8001 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=6 \
  CMD curl --fail --silent http://127.0.0.1:8000/v2/health/ready >/dev/null || exit 1

CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true", "--disable-auto-complete-config"]
