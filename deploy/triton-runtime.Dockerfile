# syntax=docker/dockerfile:1.7

# NVIDIA's official compose.py output is the source of the Triton and
# TensorRT runtime files. The final stage deliberately starts from the
# matching CUDA base so compiler toolchains, profilers, HPC-X, unused CUDA
# math libraries, and TensorRT engine-builder resources are not shipped.
ARG TRITON_COMPOSED_IMAGE
ARG CUDA_BASE_IMAGE
FROM ${TRITON_COMPOSED_IMAGE} AS composed
FROM ${CUDA_BASE_IMAGE}

ARG TRITON_STACK
ARG TRITON_SERVER_VERSION
ARG TENSORRT_VERSION
ARG TRITON_FULL_IMAGE
ARG TRITON_MIN_IMAGE
ARG TRITON_COMPOSE_REVISION
ARG CUDA_BASE_IMAGE

LABEL org.opencontainers.image.title="MedDeID TensorRT-only Triton runtime" \
      org.opencontainers.image.description="Minimal NVIDIA Triton runtime projected from the official TensorRT-only composition" \
      org.opencontainers.image.source="https://github.com/stighellemans/meddeid" \
      org.opencontainers.image.licenses="LicenseRef-NVIDIA-Deep-Learning-Container" \
      io.meddeid.triton-stack="${TRITON_STACK}" \
      io.meddeid.triton-server-version="${TRITON_SERVER_VERSION}" \
      io.meddeid.tensorrt-version="${TENSORRT_VERSION}" \
      io.meddeid.triton-full-image="${TRITON_FULL_IMAGE}" \
      io.meddeid.triton-min-image="${TRITON_MIN_IMAGE}" \
      io.meddeid.triton-compose-revision="${TRITON_COMPOSE_REVISION}" \
      io.meddeid.cuda-base-image="${CUDA_BASE_IMAGE}" \
      io.meddeid.runtime-closure="triton,tensorrt,nccl,dcgm,cupti"

ENV PATH=/opt/tritonserver/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/tritonserver/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates \
      curl \
      libb64-0d \
      libcurl4t64 \
      libnuma1 \
      libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=composed /opt/tritonserver/LICENSE /opt/tritonserver/LICENSE
COPY --from=composed /opt/tritonserver/TRITON_VERSION /opt/tritonserver/TRITON_VERSION
COPY --from=composed /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf
COPY --from=composed /opt/tritonserver/bin /opt/tritonserver/bin
COPY --from=composed /opt/tritonserver/lib /opt/tritonserver/lib
COPY --from=composed /opt/tritonserver/backends/tensorrt /opt/tritonserver/backends/tensorrt

# Required direct runtime dependencies reported by the Triton executable and
# TensorRT backend. Builder resources and non-SM75 engine builders are not
# part of this closure because the final image only loads a prebuilt plan.
COPY --from=composed /usr/local/cuda-13.3/targets/x86_64-linux/lib/libcupti.so.2026.2.1 /usr/local/cuda-13.3/targets/x86_64-linux/lib/
COPY --from=composed /usr/lib/x86_64-linux-gnu/libdcgm.so.4.5.3 /usr/lib/x86_64-linux-gnu/
COPY --from=composed /usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7 /usr/lib/x86_64-linux-gnu/
COPY --from=composed /usr/lib/x86_64-linux-gnu/libnvinfer.so.11.1.0 /usr/lib/x86_64-linux-gnu/
COPY --from=composed /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.11.1.0 /usr/lib/x86_64-linux-gnu/

RUN ln -s libcupti.so.2026.2.1 /usr/local/cuda-13.3/targets/x86_64-linux/lib/libcupti.so.13 \
    && ln -s libcupti.so.13 /usr/local/cuda-13.3/targets/x86_64-linux/lib/libcupti.so \
    && ln -s libdcgm.so.4.5.3 /usr/lib/x86_64-linux-gnu/libdcgm.so.4 \
    && ln -s libdcgm.so.4 /usr/lib/x86_64-linux-gnu/libdcgm.so \
    && ln -s libnccl.so.2.30.7 /usr/lib/x86_64-linux-gnu/libnccl.so.2 \
    && ln -s libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so \
    && ln -s libnvinfer.so.11.1.0 /usr/lib/x86_64-linux-gnu/libnvinfer.so.11 \
    && ln -s libnvinfer.so.11 /usr/lib/x86_64-linux-gnu/libnvinfer.so \
    && ln -s libnvinfer_plugin.so.11.1.0 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.11 \
    && ln -s libnvinfer_plugin.so.11 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so \
    && ldconfig \
    && test -x /opt/tritonserver/bin/tritonserver \
    && test -s /opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so \
    && ! find /usr/lib/x86_64-linux-gnu -name 'libnvinfer_builder_resource*' -print -quit | grep -q . \
    && ! find /usr/local/cuda-13.3 -type f \( -name nvcc -o -name ptxas -o -name nsys \) -print -quit | grep -q . \
    && ldd /opt/tritonserver/bin/tritonserver | tee /tmp/tritonserver.ldd \
    && ldd /opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so | tee /tmp/tensorrt-backend.ldd \
    && ! grep -q 'not found' /tmp/tritonserver.ldd \
    && ! grep -q 'not found' /tmp/tensorrt-backend.ldd \
    && rm /tmp/tritonserver.ldd /tmp/tensorrt-backend.ldd

WORKDIR /opt/tritonserver
EXPOSE 8000 8001 8002
CMD ["tritonserver"]
