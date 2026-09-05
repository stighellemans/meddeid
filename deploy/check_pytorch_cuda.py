#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the installed PyTorch build executes on CUDA"
    )
    parser.add_argument("--expected-cuda", required=True)
    args = parser.parse_args()

    if torch.version.cuda != args.expected_cuda:
        raise SystemExit(
            f"expected a CUDA {args.expected_cuda} PyTorch build, "
            f"found {torch.version.cuda!r}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access an NVIDIA GPU")
    unnecessary = [
        name
        for name in ("tensorrt", "tritonclient", "onnxruntime")
        if importlib.util.find_spec(name) is not None
    ]
    if unnecessary:
        raise SystemExit(
            "CUDA image contains unrelated inference frameworks: "
            + ", ".join(unnecessary)
        )
    if importlib.util.find_spec("triton") is not None:
        raise SystemExit(
            "CUDA eager image contains the optional PyTorch compiler runtime"
        )

    device = torch.device("cuda:0")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = left @ right
    torch.cuda.synchronize(device)
    if result.cpu().tolist() != [[19.0, 22.0], [43.0, 50.0]]:
        raise SystemExit("CUDA matrix multiplication returned an unexpected result")

    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "schema": "meddeid.pytorch-cuda-check.v1",
                "status": "ok",
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_device": {
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": properties.total_memory,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
