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

<span id="rendering-release-candidate"></span>

## MedDeID 0.4.3

Install the stable release from PyPI. A job that calls the gateway can reuse
stored detections and render views on CPU with core and the language package
for its notes. For Dutch, run

```bash
python -m pip install 'meddeid-core==0.2.4' 'meddeid-language-nl==0.2.4'
```

On a machine that runs the model and serves HTTP, install the inference
package with its server extra. To reproduce this release exactly, pin both
profiles and core.

```bash
python -m pip install 'meddeid[server]==0.4.3' 'meddeid-core==0.2.4' \
  'meddeid-language-en==0.2.4' 'meddeid-language-nl==0.2.4'
```

Both HTTP endpoints reject `metadata.date_shift_days` with status 422 whenever
the field is present, including null. Migrate older callers by removing it
and choosing a patient scope or an explicit override in `render`.
[How to shift event dates](../workflows/production-deployment.md#shift-event-dates-with-a-patient-scope)

## Published suite 0.3.1 { #current-compatible-versions }

This historical suite pins MedDeID 0.4.2 with the components below. The image table in the next section lists the current inference release.
See the [tagged suite manifest](https://github.com/stighellemans/meddeid-suite/blob/v0.3.1/suite-lock.yaml).

| Component | Version | Distribution | Compatible MedDeID dependencies |
|---|---:|---|---|
| [`meddeid-core`](https://github.com/stighellemans/meddeid-core) | 0.2.1 | PyPI | None |
| [`meddeid-language-nl`](https://github.com/stighellemans/meddeid-language-nl) | 0.2.1 / npm 0.2.1 | PyPI and npm | `meddeid-core >=0.2,<0.3` |
| [`meddeid-language-en`](https://github.com/stighellemans/meddeid-language-en) | 0.2.1 / npm 0.2.1 | PyPI and npm | `meddeid-core >=0.2,<0.3` |
| [`meddeid`](https://github.com/stighellemans/meddeid) | 0.4.2 | PyPI and CPU/CUDA deployment artifacts; compatible TensorRT artifacts remain at 0.4.0 | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-data`](https://github.com/stighellemans/meddeid-data) | 0.4.2 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-eval`](https://github.com/stighellemans/meddeid-eval) | 0.5.0 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| [`meddeid-training`](https://github.com/stighellemans/meddeid-training) | 0.3.1 | PyPI | `meddeid-core >=0.2,<0.3`; `meddeid-eval >=0.5,<0.6` |
| Browser applications | 0.3.1 | Public GHCR images and source | Generated taxonomy contract version 1 |

All Python packages require Python 3.10 or later. You need Node.js 20 or later
only when running a browser application from source; the published container
images do not require a local Node.js installation.

When installing several Python components together, respect the dependency
ranges in the final column. Do not force an incompatible version past `pip`;
use a clean environment or upgrade the related components together.

## Choose a container tag or digest

The CPU API and three browser-application images support `linux/amd64` and
`linux/arm64`. The GPU images are `linux/amd64`. PyTorch CUDA uses one portable
NVIDIA runtime; TensorRT combines a weight-free runtime with a separately
published plan for each model and GPU target.
A native Apple MPS installation is available through the Python package on
Apple silicon; ordinary Linux Docker containers cannot access the host Metal
device.
A version tag is convenient for evaluation. A digest identifies the exact
published bytes and is required for a validated deployment.

If you are choosing a runtime rather than looking up a version, start with the
[production deployment guide](../workflows/production-deployment.md#choose-a-deployment).

| Image | Version tag | Pinning |
|---|---|---|
| `ghcr.io/stighellemans/meddeid-api` | `0.4.3` | Resolve and pin with `docker buildx imagetools inspect` |
| `ghcr.io/stighellemans/meddeid-api` (PyTorch CUDA) | `0.4.3-cuda12.9` | Resolve and pin the GPU tag independently from the CPU image |
| `ghcr.io/stighellemans/meddeid-triton-gateway` | `0.4.3` | Weight-free API gateway; Compose connects it to Triton |
| `ghcr.io/stighellemans/meddeid-triton-runtime` | `0.4.0-trt26.07` | Weight-free Triton runtime for the matching release stack |
| `ghcr.io/stighellemans/meddeid-triton-plan-t4-sm75` | release-, model-, and stack-specific | Compiled model plan; Compose selects and verifies the matching Dutch or English artifact |
| `ghcr.io/stighellemans/meddeid-triton-plan-ampere-plus` | release-, model-, and stack-specific | Hardware-compatible plan for Ampere and newer GPUs, available for both public models |
| `ghcr.io/stighellemans/meddeid-annotate` | `0.3.1` | Resolve and pin with `docker buildx imagetools inspect` |
| `ghcr.io/stighellemans/meddeid-curate` | `0.3.1` | Resolve and pin with `docker buildx imagetools inspect` |
| `ghcr.io/stighellemans/meddeid-subannotate` | `0.3.1` | Resolve and pin with `docker buildx imagetools inspect` |

Compose selects a plan using the GPU, model revision and language profile.
The published catalog contains exact T4 plans and hardware-compatible
Ampere+ plans. The published performance measurements are from a T4.
[Choose or build a TensorRT plan](../workflows/production-deployment.md#deploy-the-optimized-nvidia-service)

Record the image and plan digests you tested. The 0.4.3 gateway uses the
0.4.0 runtime and four compiled plans. The plans retain their original tags
and manifest versions so Compose can verify their model and runtime identity.

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
