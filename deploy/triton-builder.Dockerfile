# syntax=docker/dockerfile:1.7

# This image is an ephemeral local tool. It deliberately contains the exporter,
# PyTorch, ONNX, TensorRT engine builder, and compiler dependencies that are
# excluded from the deployed gateway and Triton runtime images.
ARG TENSORRT_BUILDER_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:715352ed6840166888d79b135dd2a8d3f23ffe3169b8cacb6ca4df270388a8df
FROM ${TENSORRT_BUILDER_IMAGE}

ENV VIRTUAL_ENV=/opt/meddeid-builder \
    PATH=/opt/meddeid-builder/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY constraints/container.txt constraints/container.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src src
COPY deploy deploy

RUN python -m pip install \
      --constraint constraints/container.txt \
      '.[dev]' distro requests onnx onnxscript \
    && chmod +x deploy/build_triton_local.sh \
    && command -v hf \
    && command -v trtexec \
    && python -c "import meddeid, onnx, torch"

ENTRYPOINT ["/workspace/deploy/build_triton_local.sh"]
