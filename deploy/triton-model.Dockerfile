ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.02-py3
FROM ${TRITON_BASE_IMAGE}

ARG MODEL_REVISION
ARG BUNDLE_SHA256
ARG GPU_TARGET
LABEL org.opencontainers.image.title="MedDeID TensorRT model server" \
      org.opencontainers.image.revision="${MODEL_REVISION}" \
      io.meddeid.bundle-sha256="${BUNDLE_SHA256}" \
      io.meddeid.gpu-target="${GPU_TARGET}"

COPY deploy/triton/model_repository /models
CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]
