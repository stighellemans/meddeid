# syntax=docker/dockerfile:1.7

# Ephemeral helper for reproducing the slim runtime image with the host Docker
# daemon. It is never part of the deployed service.
FROM docker:29.1.3-cli-alpine3.23@sha256:4fa0ee1f3a7e4354c4ea34558b6d4ee32859baf4973d4c8ccc8e7fe3dd730c04

RUN apk add --no-cache bash git py3-pip python3 \
    && python3 -m venv /opt/runtime-builder \
    && /opt/runtime-builder/bin/python -m pip install distro requests

ENV PATH=/opt/runtime-builder/bin:${PATH}
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/workspace/deploy/build_slim_triton_runtime.sh"]
