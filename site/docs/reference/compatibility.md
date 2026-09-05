# Compatibility

Use this page when you combine MedDeID packages, reopen an older project, or
pin a deployment. If you install only `meddeid` in a clean environment, `pip`
normally selects the compatible supporting packages for you.

## What you need to pin

| Situation | Recommended approach |
|---|---|
| Trying MedDeID locally | Install the package normally and let `pip` resolve its dependencies |
| Reproducing a study or saved result | Record exact package versions, model revision, language profile, and artifact manifests |
| Running a container in production | Pin the image digest after validating that exact image |
| Passing data between tools | Keep the generated manifest with the file so the next tool can verify its contract and checksum |

## Current compatible versions

| Component | Version | Distribution | Compatible MedDeID dependencies |
|---|---:|---|---|
| [`meddeid-core`](https://github.com/stighellemans/meddeid-core) | 0.2.0 | PyPI | None |
| [`meddeid-language-nl`](https://github.com/stighellemans/meddeid-language-nl) | 0.2.0 / npm 0.1.0 | PyPI and npm | `meddeid-core >=0.2,<0.3` |
| [`meddeid-language-en`](https://github.com/stighellemans/meddeid-language-en) | 0.2.0 | PyPI | `meddeid-core >=0.2,<0.3` |
| [`meddeid`](https://github.com/stighellemans/meddeid) | 0.3.0 | PyPI and CPU/CUDA/TensorRT deployment artifacts | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-data`](https://github.com/stighellemans/meddeid-data) | 0.3.0 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-eval`](https://github.com/stighellemans/meddeid-eval) | 0.3.0 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-training`](https://github.com/stighellemans/meddeid-training) | 0.2.0 | PyPI | `meddeid-core >=0.2,<0.3`; `meddeid-eval >=0.3,<0.4` |
| Browser applications | 0.1.0 | Public GHCR images and source | Generated taxonomy contract version 1 |

All Python packages require Python 3.10 or later. You need Node.js 20 or later
only when running a browser application from source; the published container
images do not require a local Node.js installation.

When installing several Python components together, respect the dependency
ranges in the final column. Do not force an incompatible version past `pip`;
use a clean environment or upgrade the related components together.

## Choose a container tag or digest

The CPU API and three browser-application images support `linux/amd64` and
`linux/arm64`. The initial GPU release candidates are `linux/amd64`: PyTorch
CUDA is runtime-specific, while every TensorRT image is also GPU-target-specific.
A native Apple MPS installation is available through the Python package on
Apple silicon; ordinary Linux Docker containers cannot access the host Metal
device.
A version tag is convenient for evaluation. A digest identifies the exact
published bytes and is required for a validated deployment.

If you are choosing a runtime rather than looking up a version, start with the
[production deployment guide](../workflows/production-deployment.md#choose-a-deployment).

| Image | Version tag | Pinning |
|---|---|---|
| `ghcr.io/stighellemans/meddeid-api` | `0.3.0` | Resolve and pin with `docker buildx imagetools inspect` |
| `ghcr.io/stighellemans/meddeid-api` (PyTorch CUDA) | `0.3.0-cuda12.9` | Resolve and pin the GPU tag independently from the CPU image |
| `ghcr.io/stighellemans/meddeid-triton-gateway` | `0.3.0` | Weight-free API gateway paired with the matching target-specific model server |
| `ghcr.io/stighellemans/meddeid-triton-t4-sm75` | `0.3.0-trt26.07-fp16` | FP16 plan compute with FP32 binary HTTP outputs; use only on the published target/stack and pin its digest |
| `ghcr.io/stighellemans/meddeid-annotate` | `0.1.0` | `sha256:72f3e0fa0935da41e635e668573ec9c434cc3e8e1ef97bc793917bdfe6a7b78d` |
| `ghcr.io/stighellemans/meddeid-curate` | `0.1.0` | `sha256:8b3dde675cadc81f42a7fc34917d7b472c1556d14bc3acd1babf5bee8699875b` |
| `ghcr.io/stighellemans/meddeid-subannotate` | `0.1.0` | `sha256:d7da6967cb29b6cf8377458959dca84626a9c0e157320b42fe8815f49e880c87` |

The T4 target is the only ready optimized TensorRT plan. If you need an
optimized build for A10G, L4, or another NVIDIA GPU, email
[stig.hellemans@uantwerpen.be](mailto:stig.hellemans@uantwerpen.be). Each target
requires separate validation before it can be offered. Use the portable
PyTorch CUDA image when the GPU model is not fixed, and do not send sensitive
data by email.

Keep the digest you validated in the deployment configuration and its release
record. A source-code update does not change an existing image; it must be
built and published as a new image version first.

## Check contracts when exchanging files

Current components use the contracts below. Most readers encounter them only
inside generated manifests. They become important when you integrate another
system or try to combine files created by different release lines.

| Contract | Current value | Authority |
|---|---|---|
| Record schema | `meddeid.schema.v1` | `meddeid-core` |
| Offset unit | `unicode_codepoints` | `meddeid-core` |
| Taxonomy contract | version 1 | `meddeid-core/contracts/taxonomy.json` |
| Taxonomy | `ProductionLabels-v1.1` | Core taxonomy plus published annotation guidelines |
| Public language-profile resources | `nl-BE`, `nl-NL`, `en-GB`, `en-US` | Language packages; each model bundle declares its supported subset |
| CLI/server model selection | Explicit `--model` / `MEDDEID_MODEL` | No silent language or use-case default |

If a manifest declares a different schema, taxonomy, offset unit, or profile
contract, do not assume that the file is interchangeable. Use the migration
guidance from the component that owns the changed contract.

??? info "Detailed subannotation contracts"
    `meddeid-subannotate` uses
    `meddeid.subannotation-profile.v1` for profiles and
    `meddeid.subannotation-profile-selection.v1` for saved selections. Its
    built-in profile is `neutral` / `core-pii-neutral`; the Dutch language
    package provides `nl-BE` / `core-pii-nl-be`.
