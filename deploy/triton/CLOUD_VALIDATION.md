# Azure and Verda: prepared GPU acceptance tests

Status: acceptance checklist. On 2026-09-08 the owner authorized temporary
Azure T4 and Verda A100 hosts for public-model family builds. Provisioning or
starting a build is not a passed acceptance test; record each result separately.
Hospital models and artifact publication are outside this run's scope.
The [user walkthrough](USER_EXPERIENCE.md) describes the experience to verify.

## Test matrix

| Host | Model source | Plan source | What this proves |
|---|---|---|---|
| Azure, one T4 | Pinned public Dutch model | Existing release artifact | An external user can download, verify, start and reuse a published plan |
| Same Azure T4 | Public Dutch, revision empty | Resolved default revision | Default resolution works automatically; API and plan use one commit |
| Verda, one A100 | Same pinned public Dutch | Locally compiled | Automatic Ampere-family build |
| Same Verda GPU | English bundle in a local directory | Separately compiled | Switching to a different model, local mount, no Hub token or contribution prompt |
| Optional follow-up | Newly trained, synthetic-only compatible bundle | Separately compiled | A genuinely new checkpoint, not merely a different existing public model |

Using the English checkpoint locally tests the hospital-style directory route
without exposing a hospital model. It is a different model, not a newly trained
checkpoint. The optional final row needs an explicitly selected safe bundle.
Do not transfer patient-trained weights or patient notes merely for this test.

## Before testing a published plan

Read-only registry inspection on 2026-09-08 found:

- `ghcr.io/stighellemans/meddeid-triton-gateway:0.3.0` accessible, index digest
  `sha256:cb5a982a2011800d706507cff475522a45b8e3e6650746fa2c3754497c9eddda`.
- Anonymous lookup of `meddeid-triton-runtime:0.3.0-trt26.07` and
  `meddeid-triton-plan-t4-sm75:0.3.0-trt26.07-fp16-dutch-synthetic` returned
  `403 Forbidden`. That does not distinguish private from missing publication.

Resolve this publication/access issue before testing that route. Separate
candidate builds can proceed from source on an authorized host, but do not
prove that published-plan setup works. The external-user test must pass
without maintainer registry credentials. Do not substitute a freshly built
plan and report the existing-plan route as passed. Confirm immutable artifact
digests, matching runtime versions, and the release catalog before testing.
These checks inspect registry metadata; they do not download model weights:

```bash
docker buildx imagetools inspect ghcr.io/stighellemans/meddeid-triton-gateway:0.3.0
docker buildx imagetools inspect ghcr.io/stighellemans/meddeid-triton-runtime:0.3.0-trt26.07
docker buildx imagetools inspect ghcr.io/stighellemans/meddeid-triton-plan-t4-sm75:0.3.0-trt26.07-fp16-dutch-synthetic
```

Also freeze a reviewed candidate source revision including these changes.
Local working-tree edits are not present in a fresh `git clone` or an already
published image. Test a candidate gateway built from that source first; repeat
the plain public-image journey after release. Never overwrite an existing
release tag to hide this distinction.

## Host selection and access

Azure's `Standard_NC4as_T4_v3` provides one T4. Confirm subscription quota,
regional availability and price; no capacity or price is assumed here.
See [Microsoft's NCasT4_v3 specification](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series).

For Verda, select a full, single NVIDIA GPU supported by the pinned TensorRT
stack but absent from the release's T4 plans. Do not select an older GPU merely
because it is inexpensive. Check current options using the
[Verda CLI](https://docs.verda.com/cli/instances/):

```bash
verda instance-types --gpu
verda locations
verda availability --location <selected-location>
verda images
```

After approval, create a dedicated VM through the provider console (or Verda's
`verda vm create` wizard). Use Linux x86-64, SSH keys, sufficient RAM and a
planning allowance of at least 128 GB disk for transient builder layers.
Recheck actual space after pulling images. Prefer a non-interruptible instance
for this acceptance run. Allow SSH only from the operator's approved address;
leave ports 8000/8001 closed publicly. No GitHub Actions runner is needed for
these direct end-user tests. Do not run the destructive runner-cleanup scripts
from `deploy/azure` on an existing personal or hospital server.

Set a provider-side shutdown deadline where available and record who owns
cleanup. Confirm the NVIDIA driver against the pinned stack, Docker Engine,
Compose and NVIDIA Container Toolkit. On each VM:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
docker compose version
docker version
df -h .
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Pass: one expected GPU is visible both on the host and in Docker. A CUDA-labelled
VM image alone is not proof that the pinned runtime supports its driver/GPU.

Access the API from the workstation through SSH, not a public API port:

```bash
ssh -N -L 18000:127.0.0.1:8000 <user>@<approved-vm-address>
```

## Candidate checkout and credentials

On the VM, check out the approved candidate revision. Do not copy the whole
developer workspace or its `.env` files to the cloud. Create a fresh test key:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
git checkout <approved-candidate-commit>
cp .env.triton.example .env.triton
chmod 600 .env.triton
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Save the generated key in .env.triton; keep localhost binding and auth enabled.
mkdir -p evidence
```

Build the changed gateway so these tests exercise the candidate, not old code:

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build gateway-image
```

Add `MEDDEID_API_IMAGE=meddeid-triton-gateway:local` to the test `.env.triton`.
Do not use the local-images overlay for the existing-plan test: that overlay
also changes where the plan is obtained. The runtime and plan must remain the
release artifacts. After publication, repeat without the gateway override.

## A. Azure: existing public plan

Use the template's pinned Dutch model. Once artifact access is fixed:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml up --detach
docker compose --env-file .env.triton -f compose.triton.yaml logs plan
docker compose --env-file .env.triton -f compose.triton.yaml ps --all
curl --fail --silent http://127.0.0.1:8000/health
```

Pass: plan step exits 0, both servers become healthy, and
`deploy/triton/model_repository/build-manifest.json` matches the model, revision,
GPU, suite version and runtime. Record the downloaded artifact digest. Verify
that there is no compiler/PyTorch in the gateway and no compiler in the runtime
using the release's existing image-validation checks; image names alone are
not evidence of weight-free/minimal contents.

Recreate while preserving files/cache and confirm plan reuse:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml down
docker compose --env-file .env.triton -f compose.triton.yaml up --detach
docker compose --env-file .env.triton -f compose.triton.yaml logs plan
```

Then repeat with revision empty. If `main` has advanced beyond published plans,
the correct outcome is specific local-build guidance, not a silent fallback.
For successful runs, inspect the gateway's selected plan commit without secrets:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml exec meddeid \
  python -c "import json; print(json.load(open('/models/build-manifest.json'))['model']['revision'])"
```

This shows the selected plan commit. Also check response model provenance
against it; a file by itself does not prove the API loaded the intended model.
Restore the pinned commit before comparisons.

## B. Verda: build a public plan for the detected GPU

Use the pinned Dutch settings. First run normal startup and retain the expected
missing-plan explanation. Then follow walkthrough section 5 to build only the
plan and start it with `compose.triton.local-plan.yaml`.

The builder must complete without reading stdin or preparing an issue. Preserve
its plan and manifest. Test the same family artifact on a second compatible GPU
before declaring cross-GPU support. Never relabel the old T4 plan as family-wide.

## C. Verda: a different model from a local directory

Prepare a safe English bundle on the VM (this initial download is explicit;
the later directory build itself does not contact the Hub for the model):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub
hf download stighellemans/meddeid-english-synth \
  --revision e519d454250ba67bef62c0063dbbf242c329b750 \
  --local-dir ./test-models/english
hf cache verify stighellemans/meddeid-english-synth \
  --revision e519d454250ba67bef62c0063dbbf242c329b750 \
  --local-dir ./test-models/english
realpath ./test-models/english
```

Stop the Dutch local-plan deployment using the same Compose files used to
start it. Archive `deploy/triton/local_model_repository` to a distinct directory
such as `deploy/triton/dutch-plan-evidence` (only if that destination is absent).
Do not overwrite another run. Set `.env.triton`:

```dotenv
MEDDEID_MODEL=/input-model
MEDDEID_REVISION=
MEDDEID_LANGUAGE_PROFILE=en-GB
MEDDEID_LOCAL_MODEL_DIRECTORY=<absolute-path-printed-by-realpath>
```

Follow walkthrough section 6. Pass: the builder uses the directory, requests no
Hub token, produces a separate English plan on the detected GPU, and never
offers publication. The gateway uses the same directory's metadata/tokenizer.
Run this build without HF_TOKEN; keep the test model and source bundle unchanged.

The optional genuinely new-checkpoint test repeats these commands with a
selected synthetic-only compatible bundle. Do not count the English test as
proof of arbitrary architecture or freshly trained model support.

## Compare output and measure performance for every successful case

Build a CPU reference from the same candidate source, with the same model and
revision. It serves localhost port 8001 and avoids GPU contention:

```bash
MEDDEID_API_IMAGE=meddeid-api:ux-reference \
  docker compose --env-file .env.triton -f compose.yaml build meddeid
MEDDEID_API_IMAGE=meddeid-api:ux-reference MEDDEID_PORT=8001 \
  MEDDEID_BACKEND=torch MEDDEID_DEVICE=cpu MEDDEID_WORKERS=1 \
  docker compose -p meddeid-reference --env-file .env.triton -f compose.yaml up --detach
```

For case C, add `-f deploy/triton/compose.test-local-model.yaml` to the reference
startup and teardown commands. Keep the same language profile. For unpinned
Hub tests, set the CPU reference revision to the candidate plan's resolved
commit before comparing; do not compare two independent `main` lookups.

Read the same test API key into the shell without echoing or putting it in
command history (Bash):

```bash
read -r -s -p 'Test API key: ' MEDDEID_API_KEY
export MEDDEID_API_KEY
python3 deploy/validate_triton_parity.py deploy/triton/synthetic-smoke.jsonl \
  --reference-url http://127.0.0.1:8001 --candidate-url http://127.0.0.1:8000 \
  --output evidence/parity-smoke.json
```

Expected summary: `{"passed": true, "documents": 4, "differences": 0}`.
The four invented notes are plumbing checks, not accuracy validation. Repeat
parity with the pinned public benchmark recorded in the release catalog for
each language before declaring release readiness. A mismatch must be explained,
not hidden by comparing only HTTP success codes or relaxing checks.

For bounded performance checks with explicitly controlled document lengths:

```bash
python3 deploy/make_benchmark_lengths.py deploy/triton/synthetic-smoke.jsonl \
  evidence/performance-1000.jsonl --length 1000
python3 deploy/make_benchmark_lengths.py deploy/triton/synthetic-smoke.jsonl \
  evidence/performance-long.jsonl --length 8000 --length 12000 --length 18000
python3 deploy/benchmark_http.py evidence/performance-1000.jsonl \
  --endpoint-mode single --batch-size 1 --concurrency 1 --requests 40 \
  --output evidence/online.json
python3 deploy/benchmark_http.py evidence/performance-1000.jsonl \
  --batch-size 16 --concurrency 4 --requests 40 --repeat-fixture 64 \
  --output evidence/batch.json
python3 deploy/benchmark_http.py evidence/performance-long.jsonl \
  --batch-size 1 --concurrency 1 --requests 12 \
  --output evidence/long-notes.json
unset MEDDEID_API_KEY
```

Keep a separate evidence directory per case before repeating these commands.
Record p50/p95 latency, documents/second, errors, GPU memory, warmup, batch size,
concurrency and document lengths. Do not present those measurements as clinical
accuracy or universal hardware performance. Stop the CPU reference while
benchmarking to avoid shared CPU contention. Profiles stay at their default
for acceptance; any optional profile comparison is a separate measured experiment.

## Negative cases and acceptance record

Exercise on disposable test copies, not by corrupting the only successful plan:

- Incorrect language, model or pinned revision: clear failure, no fallback.
- Wrong GPU/stack manifest or altered plan bytes: rejected before readiness.
- Nonempty build directory: refused without overwriting the retained plan.
- Empty revision with unavailable Hub: sanitized explanation, no leaked token.
- Local bundle: no contribution prompt or source upload, including unattended runs.
- Unsupported GPU/driver/stack: useful preflight error; newer-stack work is a
  separate candidate, not an automatic production upgrade.
- API authentication required for inference; no text in access logs; only
  localhost ports exposed. Synthetic tests only.

Retain a per-case checklist with source commit, image and artifact digests,
manifest, GPU/driver, parity report, timing summaries, and sanitized transcripts.
Capture actual build and failure wording for comparison with the review
walkthrough. Mark each case passed/failed/blocked/not-run. GPU launch success
alone is not acceptance. Official contribution/publication remains a separate
maintainer-reviewed step.

## Finish and stop billing

Use the exact Compose files and project name used to start each service with
`down` (without `--volumes`); retain model and evidence files until copied to an
approved location. Never export `.env.triton`, credentials, or an unreviewed
Docker configuration as public evidence. After retaining results, resolve and
inspect the exact temporary VM IDs, then remove the approved test VMs through
their provider consoles. Check remaining disks, snapshots and public IPs for
continuing charges; stopping Compose does not stop cloud billing. Verify cleanup
in both accounts. Do not delete unrelated resources or a shared resource group.
