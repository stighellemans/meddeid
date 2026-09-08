"""Hardware-family contracts shared by plan builders and selectors."""

import argparse
import re

from .gpu_identity import gpu_key

FAMILY_MODES = {"turing": "sameComputeCapability", "ampere-plus": "ampere+"}


def capability(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+", value):
        raise ValueError(f"Invalid NVIDIA compute capability: {value!r}")
    return tuple(int(part) for part in value.split("."))


def family_for_gpu(compute_capability: str) -> str:
    cc = capability(compute_capability)
    if cc == (7, 5):
        return "turing"
    if cc >= (8, 0):
        return "ampere-plus"
    raise ValueError(
        f"No MedDeID TensorRT family for compute capability {compute_capability}. "
        "Use a supported NVIDIA GPU or the CPU deployment."
    )


def validate_family(target):
    family = target.get("family")
    if family is None:
        if target.get("id") in FAMILY_MODES:
            raise ValueError("A family plan must explicitly record its compatibility mode")
        return
    if family not in FAMILY_MODES:
        raise ValueError(f"Unknown TensorRT plan family: {family!r}")
    if target.get("id") != family or target.get("compatibility_mode") != FAMILY_MODES[family]:
        raise ValueError("Plan family and TensorRT compatibility mode do not match")
    if family_for_gpu(str(target.get("compute_capability", ""))) != family:
        raise ValueError("Plan family does not match its build GPU capability")


def plan_supports_gpu(target, name: str, compute_capability: str) -> bool:
    validate_family(target)
    if target.get("family"):
        try:
            return target["family"] == family_for_gpu(compute_capability)
        except ValueError:
            return False
    # Legacy exact-GPU plans must never gain family coverage implicitly.
    return (
        gpu_key(str(target.get("gpu_name", ""))) == gpu_key(name)
        and str(target.get("compute_capability")) == compute_capability
    )


def main():
    parser = argparse.ArgumentParser(description="Select a TensorRT build family")
    parser.add_argument("compute_capability")
    parser.add_argument("--mode", action="store_true")
    args = parser.parse_args()
    try:
        family = family_for_gpu(args.compute_capability)
        print(FAMILY_MODES[family] if args.mode else family)
    except ValueError as exc:
        parser.exit(2, f"meddeid-triton: error: {exc}\n")


if __name__ == "__main__":
    main()
