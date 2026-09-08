"""Start the gateway with the exact model revision used by its selected plan."""

import json
import os
import re
from pathlib import Path


def pin_gateway_revision(environ, manifest):
    model = manifest["model"]
    if environ.get("MEDDEID_LOCAL_BUNDLE") == "true":
        from .bundle import load_model_bundle

        bundle = load_model_bundle(
            Path(environ["MEDDEID_MODEL"]) / "bundle.json", validate_package=True
        )
        if bundle.contract_hash() != model["bundle_sha256"]:
            raise ValueError("Local model bundle does not match the compiled plan")
        environ["MEDDEID_REVISION"] = ""
        return
    if environ.get("MEDDEID_MODEL") != model["id"]:
        raise ValueError("Gateway model does not match the compiled plan")
    resolved = model["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Compiled Hub plan does not record an immutable revision")
    requested = environ.get("MEDDEID_REVISION", "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", requested) and requested != resolved:
        raise ValueError("Gateway revision does not match the compiled plan")
    environ["MEDDEID_REVISION"] = resolved


def main():
    try:
        manifest = json.loads(Path("/models/build-manifest.json").read_text())
        pin_gateway_revision(os.environ, manifest)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"meddeid-triton-gateway: error: {exc}") from None
    # Every worker inherits the pinned revision; no second main lookup.
    from .server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
