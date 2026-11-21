# Production deployment

<span class="source-label">Owner: meddeid</span>

Use this guide when MedDeID will run as a shared service inside your
organization. The service processes notes locally, but its output is not
automatically anonymous. Validate the model on representative local data before
using it operationally.

```text
approved client -> your application -> MedDeID API -> local model
                         |
                         +-> governed data store
```

The examples use the **0.4.3** rendering API. This gateway rejects the
retired metadata field on single and batch requests. Use the images tagged for this release.
[Check gateway versions](../reference/compatibility.md#meddeid-043) · [Prepare a hospital deployment](hospital/deploy-service.md#choose-the-release)

## Choose how the calling job authenticates

**The built-in MedDeID API key is optional.** If an HTTPS proxy is the only
route to the model service, it can authenticate each calling job and check
its permissions. IT can then revoke one job’s access without rotating a
shared credential. In that setup, leave `MEDDEID_API_KEY` unset and
`MEDDEID_REQUIRE_API_KEY=false` on the private backend.
[Proxy configuration in the hospital cookbook](hospital/deploy-service.md#start-the-https-service)

When MedDeID itself authenticates inference requests, set `MEDDEID_API_KEY`
to a secret shared with approved clients. They send it in `X-API-Key` over
HTTPS. A configured key is always checked. Set `MEDDEID_REQUIRE_API_KEY=true`
as well so a missing key stops the service at startup. The published CUDA
and TensorRT Compose files require a key. The hospital cookbook supplies a
separate Compose file for proxy authentication.

The API key controls **who can call the service**. IT manages a separate
**mapping key** for deriving patient-specific date shifts and name aliases.
Rotating an access credential therefore leaves the mappings reproducible as
long as the mapping key and render settings stay the same.

<span id="choose-and-retain-date-shifts-outside-meddeid"></span>

## Shift event dates with a patient scope

Suppose two visits for the same patient are seven days apart. The study needs
that interval, but the actual visit dates are identifying. Using the same
patient ID in both notes gives the renderer a consistent scope. Adding IT’s
secret mapping key lets it derive a repeatable shift that someone with the
patient ID alone cannot calculate. IT can reproduce the offset later.
[Recover the offset after a run](#recover-the-offset-after-a-run)

**For study datasets, render inside a job controlled by hospital IT.** Let
the gateway detect spans. IT grants the rendering job access to a specific
version of the secret in the hospital’s secret store. This lets a research
team use the resulting text without needing the key. In the job, build the
scope with the stable patient ID and the secret. Here `mapping_key` is the
value read from that version.

```python
from meddeid_core.rendering import PseudonymizationScope

scope = PseudonymizationScope(patient_id="example-007", secret=mapping_key)
```

Pass this scope to `apply_spans` with `date_replacement=True`. Only the
trusted job needs the mapping key. Researchers receive their approved text
view. Keep paired original and shifted dates inside the same restricted
boundary, since comparing the pair reveals that patient’s offset.
[Render from stored detections](hospital/databricks-job.md#render-the-study-text)

**For an application that needs the shifted text in one HTTP call, let the
gateway add the secret.** IT mounts a pinned key file on the gateway and sets
`MEDDEID_PSEUDONYMIZATION_KEY_FILE` to its path. The file contains at least
32 random bytes encoded as hexadecimal. Authorize only callers entitled to
render with that key. Send the patient ID in the JSON below to
`POST /deidentify`. The gateway combines it with the mounted key and rejects
a client-supplied `secret` in the HTTP scope with status 422.
[Configure keyed rendering on the service](hospital/databricks-job.md#when-the-caller-needs-one-text-immediately)

```json
{
  "text": "Controle op 15/01/2025.",
  "metadata": {"lang": "nl-BE"},
  "render": {
    "pseudonymization_scope": {"patient_id": "example-007"},
    "date_replacement": true
  }
}
```

For the second visit, send the new note with the same scope. With the same
key version, algorithm and shift range, the two shifted dates in `deid_text`
will still be seven days apart. Other detected identifiers receive bracketed
labels in this example. Enable `name_replacement` for scoped name aliases or
`age_replacement` to render clinical ages separately.

### Recover the offset after a run

An authorized job can call `derive_date_offset` on CPU at any time. It needs
the complete original scope, access to the original key version, and the same
derivation algorithm and shift range. The calculation works without the note,
its detected spans or a running gateway.

For the request above, mount the **key version used for that run** in the job
and point `MEDDEID_PSEUDONYMIZATION_KEY_FILE` to it. The request used the
default shift range.

```python
import os
from pathlib import Path
from meddeid_core.rendering import PseudonymizationScope, derive_date_offset

key_file = Path(os.environ["MEDDEID_PSEUDONYMIZATION_KEY_FILE"])
scope = PseudonymizationScope(
    patient_id="example-007",
    secret=key_file.read_text(encoding="ascii").strip(),
)
offset_days = derive_date_offset(scope)
```

`offset_days` is a signed integer. For example, a value of `-400` means the
renderer moved recognized event dates 400 days earlier. If the original scope
also included a study ID, include it again. If the request set
`render.date_shift_range`, pass the same range as `date_shift_range` to
`derive_date_offset`. After key rotation, use the recorded key version.

If the request specified `render.date_offset_override`, that signed value was
the configured offset. Keep it with the render policy so you can look it up
later.

### Specify a fixed shift for one request

If a study already has an assigned offset, set `render.date_offset_override`
to that signed number of days and enable `date_replacement`. For example,
replace the `render` object above with the following object to move
15 January 2025 back 731 days to 15 January 2023.

```json
{
  "date_replacement": true,
  "date_offset_override": -731
}
```

The override takes precedence over scope-based date derivation for that
document. This date-only request works without a scope or mapping key. If the
same request also needs scoped name aliases, keep its scope and enable
`name_replacement`.

!!! note "Remove the legacy metadata field before upgrading"
    The distributed 0.4.3rc4 wheel accepts `metadata.date_shift_days`.
    The 0.4.3 gateway returns HTTP 422 (`retired_date_shift_days`)
    whenever the field is present, even if its value is null or the request
    omits `render`. One such document rejects an entire batch before inference.
    Remove the field from serialized requests. Both versions support the
    patient scope and fixed override shown above.

## Before you start

<span id="decide-what-the-surrounding-application-must-keep"></span>

### What should you keep after processing a note? { #what-the-hospital-stores-for-each-document-revision }

For a one-off request, returning the approved text may be enough. Save
detections when you expect to revisit a note, for example when a second study
needs different masking rules. The cookbook follows one note from source to
study release using Databricks and Delta tables as a worked example.
[Hospital integration cookbook](hospital/index.md)

**Which version of the note did we process?** Keep a reference to the exact
source revision in the hospital’s document system. If you save detections,
store their positions and labels with that reference and a hash of the input
text. A clinician can correct a note without changing its document ID.
Inserting even one word can make an old position point to the wrong name.
[How to verify stored detections](../concepts/data-contract.md#store-and-reuse-detections)

**What needs to run again?** Record which model bundle, language profile and
clinical inputs produced the detections. A corrected patient name can change
which names MedDeID finds even if the note is untouched. Run detection again
if the text, model or detection inputs change. If a study changes only how
names or dates should appear, reuse the verified positions and render the text
on CPU. In the restricted run record, keep the render settings, scope
reference and versions of the renderer and mapping key. IT can then reproduce
the view using the key in its secret store.

**Which text can we release?** A timeline study may need the interval
between visits, while another study can work with dates masked. Generate the
view each study is approved to receive. Before release, check that every
eligible note has a result for the intended revision and policy, and exclude
withdrawn notes.
Keep source notes and detections within the hospital’s restricted environment.

### Optional study feedback { #plan-how-users-report-mistakes }

A production interface can let users mark a missed identifier, reject an
unnecessary redaction, correct a span boundary, or change a label. Store the
original prediction and the human correction separately, together with the
document reference and model provenance. Do not silently replace the original
inference record.

Feedback does not update the running model automatically. Review and curate it,
remove unsuitable records, and evaluate a newly trained model on independent
test data before deployment. See [Prepare and annotate data](prepare-and-annotate.md)
and [Train and evaluate](train-and-evaluate.md) when feedback will be used for
model improvement.

### Establish the operating boundary

Run MedDeID inside an approved data boundary. Place network-accessible
deployments behind your organization's authenticated TLS reverse proxy or
private service mesh. The surrounding platform remains responsible for user
access, authorization, network policy, monitoring, secrets, database security,
backups, retention, and incident response.

For each deployment:

1. use an immutable container digest rather than a moving tag;
2. authorize callers at the private proxy or configure MedDeID’s own API key;
3. keep the service on a private interface or network;
4. avoid logging notes, results, span contents, metadata, or API keys; and
5. validate de-identification quality on representative local notes.

## Choose a deployment

Choose the backend for the hardware your organization will operate. The
gateway version determines whether the service accepts `render`. The
image commands below use the 0.4.3 checkout. For a private hospital service,
use the [hospital deployment instructions](hospital/deploy-service.md#choose-the-release).

| Deployment                                       | Requirements                                     | Online response\* | Batch speed\* | Choose when                                            |
| ------------------------------------------------ | ------------------------------------------------ | ----------------: | ------------: | ------------------------------------------------------ |
| [CPU](#deploy-with-cpu)                          | Docker                                           |            110 ms |    1.85 doc/s | You want the simplest setup or have no GPU             |
| [CUDA](#deploy-with-an-nvidia-gpu)<sup>**</sup>               | Docker, NVIDIA driver, NVIDIA toolkit            |           13.6 ms |    65.0 doc/s | You want support across compatible NVIDIA GPUs         |
| [TensorRT](#deploy-the-optimized-nvidia-service)<sup>**</sup> | Docker, supported NVIDIA GPU, NVIDIA driver, NVIDIA toolkit |            8.7 ms |   168.8 doc/s | A released TensorRT plan matches your GPU and model |
| [MPS](#run-on-apple-silicon)                     | Apple silicon, Python                            |            9.4 ms |    42.1 doc/s | You want to run natively on a Mac                      |

<small>\* Online response is the median for one short note; short notes averaged
314 characters. Batch speed uses 16-note requests at concurrency 8; those notes
averaged about 1,400 characters. Results are medians of three runs using the
recommended configuration for each workload. CPU, CUDA, and TensorRT were
tested on an Azure T4 host; MPS was tested on an M4 Pro. Treat the figures as
comparisons, not capacity guarantees.</small>

<small>** CUDA and the published TensorRT plans use FP16 neural computation by
default to increase performance. Validation has found only minimal output
changes relative to FP32, so MedDeID treats these paths as quasi-consistent
rather than identical. For the checked full-consistency CUDA configuration, set
`MEDDEID_TORCH_PRECISION=fp32` and restart the service. A published TensorRT
plan is precompiled for FP16 and cannot be changed dynamically to FP32; using
FP32 requires building and validating a separate TensorRT plan.</small>

MedDeID currently detects the
installed GPU during startup. If no published plan matches it, startup explains
how to build a local plan on that server. To request an officially validated
target for another NVIDIA GPU, email
[stig.hellemans@uantwerpen.be](mailto:stig.hellemans@uantwerpen.be). A new
target must be built and validated for that GPU before it can be offered.
Do not send patient data or other sensitive information by email.

You download only the runtime you choose. The CPU and CUDA images do not
contain a model; the selected model is downloaded once at startup and reused
from the local cache.

## Deploy with CPU

The CPU image is the easiest starting point and supports both AMD64 and ARM64.
Install Docker, clone the repository, and create a private environment file:

```bash
git clone --branch v0.4.3 https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.example .env.cpu
chmod 600 .env.cpu
```

The supplied CPU environment template selects the Dutch public model,
binds to localhost and enables MedDeID’s own API key. For this setup, pin
the approved image digest and replace the example key.

```dotenv
MEDDEID_API_IMAGE=ghcr.io/stighellemans/meddeid-api@sha256:<approved-digest>
MEDDEID_API_KEY=<secret>
```

Start the service with ordinary Compose commands:

```bash
docker compose --env-file .env.cpu pull meddeid
docker compose --env-file .env.cpu up --detach meddeid
docker compose --env-file .env.cpu ps
```

To serve another compatible MedDeID model, change `MEDDEID_MODEL`,
`MEDDEID_REVISION`, and `MEDDEID_LANGUAGE_PROFILE` together, then restart the
service. The image itself stays the same.

## Deploy with an NVIDIA GPU

Use the CUDA image when you have a compatible NVIDIA GPU but do not want the
deployment tied to one specific GPU model. The host needs an NVIDIA driver,
Docker Engine, and NVIDIA Container Toolkit.

```bash
git clone --branch v0.4.3 https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.cuda.example .env.cuda
chmod 600 .env.cuda
# This CUDA Compose file requires an API key. Set it and pin the image digest in .env.cuda.

docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

The CUDA template has the same prefilled model and security choices as the CPU
template. It also contains the one workload choice:

```dotenv
# latency suits individual requests; throughput enables bounded CUDA batching.
MEDDEID_SERVING_PROFILE=latency
```

Keep `latency` for an online service. Use `throughput` when you want to maximize
batch processing. The CUDA deployment requires a working GPU and fails instead of
silently falling back to CPU.

## Deploy the optimized NVIDIA service

Choose this option when you want the fastest tested deployment. Install Docker,
the NVIDIA driver, and NVIDIA Container Toolkit, then create the configuration:

```bash
git clone --branch v0.4.3 https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.triton.example .env.triton
chmod 600 .env.triton
```

MedDeID detects the GPU from the name reported by NVIDIA and selects the
matching plan. This release provides one exact plan for NVIDIA T4 and one
hardware-compatible plan for Ampere and newer GPUs. The displayed performance
figures were measured on T4 and should not be treated as measurements of other
GPUs.
The published TensorRT Compose file requires an API key. Replace the example
value before starting it. The template selects the Dutch public model by
default. Change the following three settings
together when using another validated model:

```dotenv
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth
MEDDEID_REVISION=1f20655454dcbd042647cacdfff6b6802a970959
MEDDEID_LANGUAGE_PROFILE=nl-BE
```

For the English public model, use its model ID and revision from
[`meddeid models`](inference.md#select-a-model-and-language), then choose `en-GB` or
`en-US`. Start the service with ordinary Compose commands:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml pull

docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  up --detach

docker compose --env-file .env.triton -f compose.triton.yaml ps
```

If startup reports a failed dependency, read the explanation with:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml logs plan
```

On first start, Compose obtains the weight-free API gateway, the weight-free
Triton runtime, and the validated plan matching the hardware, model, revision,
and MedDeID release. It verifies the plan and keeps it in the local deployment
folder for later starts. The Dutch and English models have separate plans, but
you do not need to locate or mount those plans yourself.

The compiled plan contains the model parameters, but the container images do
not. To change model, update the three model settings and restart. If this
release has no validated plan for the detected GPU and model, it will direct you to the local build below.

### Advanced: build a plan locally

Use this when no plan is available for your GPU or model. Keep the same
`.env.triton` and run the build on the NVIDIA server that will host MedDeID.

Usually only the plan needs building. The release gateway and runtime can be
reused on GPUs supported by their CUDA/TensorRT stack. A GPU requiring a newer
stack needs a matching builder and runtime; recompiling with an unsupported
stack will not add support.

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  build plan-builder

docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  run --rm plan-builder
```

The builder downloads the selected model and compiles it into
`deploy/triton/local_model_repository`. The result stays on this device;
nothing is published to the internet. Start it with the release containers:

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.local-plan.yaml \
  up --detach
```

#### Use an institution-specific model

For a **0.4.3** build, start from the tagged source on the target host,
select its private plan catalog and build the matching gateway. Use the local
image, local model and local plan overrides together when starting that stack.
[Prepare the private model catalog](hospital/deploy-service.md#if-you-use-tensorrt)

Keep your hospital's model in a local directory containing its MedDeID bundle.
No Hugging Face account, upload, or token is needed. Set these values in `.env.triton`:

```dotenv
# The host directory below is mounted at /input-model inside the containers.
MEDDEID_MODEL=/input-model
MEDDEID_REVISION=
MEDDEID_LANGUAGE_PROFILE=nl-BE
MEDDEID_LOCAL_MODEL_DIRECTORY=/approved/models/my-meddeid-model
```

Build the builder once, then compile from that directory. The model and resulting
plan stay on this server; neither is uploaded to the internet:

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build plan-builder

docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  -f compose.triton.build.local-model.yaml \
  run --rm plan-builder
```

The gateway also needs that bundle's tokenizer and metadata. Start with both
the local plan and local model mounts:

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.local-plan.yaml \
  -f compose.triton.local-model.yaml \
  up --detach
```

Container images and build dependencies still need to be available locally or
downloaded. For a fully disconnected hospital, prepare those separately before
transferring them into the approved environment.

#### Build the containers locally too

Use this additional step when policy requires building the gateway and runtime
inside the hospital. A different GPU by itself does not require rebuilding the
gateway. Keep the builder and runtime on the same supported stack.

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build
docker compose --env-file .env.triton -f compose.triton.build.yaml run --rm runtime-builder
```

After building your plan as above, start the locally built containers:

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.local-images.yaml \
  up --detach
```

The large builder contains the compiler, PyTorch, ONNX, and TensorRT build
tools. It runs only during the build; none of those tools or the source model
is copied into the deployed gateway or runtime image. The runtime-builder uses
the host Docker socket, which grants broad control over Docker. Run it only
from a reviewed checkout on a dedicated build host. The
[advanced TensorRT guide](https://github.com/stighellemans/meddeid/blob/main/deploy/triton/README.md)
explains validation, artifact publication, and custom targets.

When using a mounted model directory, also add `-f compose.triton.local-model.yaml`
to this startup command. Validate the resulting service on representative data
before using it for clinical work.

## Run on Apple silicon

Apple's Metal GPU is available through a native macOS installation. Docker
Desktop cannot pass it into the Linux MedDeID container, so install and run the
server directly:

```bash
git clone --branch v0.4.3 https://github.com/stighellemans/meddeid.git
cd meddeid
python -m pip install '.[server]'
cp server.env.example meddeid-server.env
chmod 600 meddeid-server.env
```

The generic server template selects the Dutch public model, the latency
profile and built-in API-key authentication. For this setup, inject the key
from the institution’s secret manager. If the hospital proxy authenticates
callers, configure the private backend as described under
[caller authentication](#choose-how-the-calling-job-authenticates).

Start the server with that configuration:

```bash
meddeid-server --env-file meddeid-server.env
```

The file is the persistent configuration for this service, so the model and
other settings do not need to be repeated at every start. Existing process
environment variables take precedence, allowing a process supervisor or secret
manager to inject the API key without storing it in the file. Keep the file out
of source control.

MedDeID automatically chooses MPS when it is available and CUDA is not. An
explicit device setting is therefore unnecessary for the normal Apple path.
MedDeID deliberately does not write a machine-wide default model: another
service or command on the same host may require a different validated model.

Use an organization-managed process supervisor when this becomes an unattended
service and configure it to run the same `--env-file` command after restarts.
The same security and validation requirements apply as for a container
deployment.

## Process individual requests or batches

For one note, use `POST /deidentify` with the
[patient scope example](#shift-event-dates-with-a-patient-scope). For
`POST /deidentify-batch`, place each note in the `documents` array with its
own `document_id`, `text`, `metadata` and `render`. Notes for the same patient
can share a scope while retaining distinct document IDs.

With an explicit `render` object, read `deid_text` for the selected view,
`spans` for detections in the original text and `provenance` for model
identity. Batch results also echo each `document_id`.
[Single and batch HTTP examples](rendering.md#request-one-text-from-the-gateway)

If interactive requests and large batch jobs must run at the same time, operate
separate service instances so a large job cannot delay an interactive request.
Tune concurrency and batching only after measuring the real workload on the
target hardware.

### Optional: optimize sustained traffic

Most deployments can keep the supplied default. If a CUDA or MPS service will
continuously process thousands or millions of notes, first use
`POST /deidentify-batch`. After measuring the real workload, you can also set
`MEDDEID_SERVING_PROFILE=throughput` to favor overall processing speed over the
response time of an individual request. The CPU and TensorRT deployments did
not show a separate benefit from this setting on at T4 but it can differ in your setting.

The [technical production reference](https://github.com/stighellemans/meddeid/blob/main/docs/production.md#latency-and-throughput-serving-profiles)
contains the detailed measurements and tuning controls.

## Edit only what you need

The environment templates deliberately show only the choices most operators
need. Their starting values select the Dutch public model, localhost binding,
API-key authentication, the latency profile, and an API-only service.

Review a setting only when the corresponding decision applies:

| Decision                                                         | Setting                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| Have MedDeID authenticate inference requests                     | Set `MEDDEID_API_KEY` and use `MEDDEID_REQUIRE_API_KEY=true` to catch a missing key at startup |
| Derive patient-scoped dates or names on the gateway    | `MEDDEID_PSEUDONYMIZATION_KEY_FILE` (path to the mounted mapping key) |
| Production container: pin the approved image digest              | `MEDDEID_API_IMAGE` or `MEDDEID_PYTORCH_CUDA_IMAGE`                 |
| Use another model                                                | `MEDDEID_MODEL`, `MEDDEID_REVISION`, and `MEDDEID_LANGUAGE_PROFILE` |
| Sustained CUDA or MPS batch processing                           | `MEDDEID_SERVING_PROFILE=throughput`                                |
| Expose through a trusted private network or reverse proxy        | `MEDDEID_BIND_ADDRESS`                                              |
| Temporarily provide the browser or interactive API documentation | `MEDDEID_UI_ENABLED` or `MEDDEID_DOCS_ENABLED`                      |

The server loads one `MEDDEID_MODEL` when it starts. When this is a Hugging
Face ID, MedDeID downloads the selected revision on first startup and reuses
the local cache afterwards. A local path loads an existing model bundle
directly. Change the setting and restart the service to load another model.
The selected repository must follow the MedDeID bundle contract; an arbitrary
Hugging Face token-classification model is not enough.

Inference requests cannot name a model, switch the running model, or ask the
server to download another one. Run separate service instances when clients
need access to different models at the same time.

A loaded model declares the language profiles it supports.
`MEDDEID_LANGUAGE_PROFILE` selects the default. A caller may instead send
`metadata.lang`, but the requested profile must be supported by the model.
Otherwise, the request is rejected. Selecting a profile does not download it;
the profile must already be supported by the model.

Set `MEDDEID_OFFLINE=true` only when the selected model is already cached or
supplied through a mounted local path. CPU and CUDA download the model weights;
the TensorRT gateway downloads only the matching tokenizer and bundle metadata,
because inference weights are held in the separately compiled TensorRT plan.

Advanced settings do not belong in the initial environment file. Leave their
defaults unchanged until representative measurements justify an override. The
[environment variable
reference](https://github.com/stighellemans/meddeid/blob/main/docs/production.md#environment-variable-reference)
lists the advanced settings and defaults; the rest of the technical production
guide contains the measured CPU, CUDA, MPS, and T4 behavior.

### Browser interface and API documentation

Keep the browser interface disabled for an API-only service. Interactive API
documentation can be useful during integration and acceptance testing; enable
it with `MEDDEID_DOCS_ENABLED=true` when approved users need it.

The documentation routes are not protected by MedDeID's inference API key. If
they remain enabled in production, restrict `/docs`, `/redoc`, and
`/openapi.json` separately at the reverse proxy. Disabling these routes does
not disable the inference API.

## Check and operate the service

Use the health endpoints from inside the trusted network:

- `GET /live` returns HTTP 200 while the API process can respond. Use it as the
  liveness probe.
- `GET /health` returns HTTP 200 when the model and runtime are ready and HTTP
  503 when they are unavailable. Use it as the readiness and startup probe.

```bash
curl --fail http://127.0.0.1:8000/live
curl --fail http://127.0.0.1:8000/health
```

Keep the probes separate. A model may still be loading while the process is
healthy; using `/health` as a liveness probe can cause repeated restarts before
startup completes. Both routes are intentionally unauthenticated for
orchestration and return no clinical text.

Use the response `X-Request-ID` to correlate failures without recording patient
text. Treat HTTP 503 with `Retry-After` as an overload signal.

Before an upgrade, record the image digest, model revision, package versions,
and language profile. Compare the old and new versions on the same local
validation set, deploy the new version gradually, and retain the previous
digest for rollback.

Stop a Compose deployment with the same environment and Compose files used to
start it. For example, the CPU deployment uses:

```bash
docker compose --env-file .env.cpu down
```

## Technical details

- [Complete production configuration and benchmark evidence](https://github.com/stighellemans/meddeid/blob/main/docs/production.md)
- [CUDA image requirements](https://github.com/stighellemans/meddeid/blob/main/deploy/pytorch-cuda/README.md)
- [TensorRT target and validation details](https://github.com/stighellemans/meddeid/blob/main/deploy/triton/README.md)
- [Released versions and container tags](../reference/compatibility.md)
