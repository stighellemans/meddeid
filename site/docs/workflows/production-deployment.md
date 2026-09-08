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

## Before you start

MedDeID is the inference part of a larger application. It accepts a note and
returns de-identified text, the detected spans, processing information,
warnings, and model provenance. It does not provide a clinical database,
choose pseudonymization values, manage researcher access, or retain corrections
for future training. Plan those responsibilities before choosing a deployment.

### Decide what the surrounding application must keep

A SQL database is a practical default when MedDeID is part of a production
service. It can keep the source, result, and review history linked without
putting that state inside the inference container. MedDeID does not connect to
that database or prescribe its schema. If the original note already has an
authoritative home, store a stable reference instead of an unnecessary second
copy.

For each processed note, consider retaining:

| Information                        | Why retain it                                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Original note + ID                 | Span offsets are only interpretable using the original note                                                           |
| De-identified note                 | The final output for downstream users                                                                                 |
| Each returned span                 | Preserves its `begin` and `end` character positions, label, matched text, replacement, and confidence                 |
| Date-shift value and its scope     | Keeps dates consistent and allows an authorized system to reverse the shift when policy permits                       |
| Processing warnings and provenance | Records fallbacks and identifies the exact model, profile, and software that produced the result                      |
| Review status and corrections      | Lets users flag missed or unnecessary redactions and creates material that can later be curated for model improvement |

The span positions refer to the original input text. Keep that text unchanged:
editing it, normalizing line endings, or changing Unicode characters can make
stored positions point to the wrong characters. Retaining individual spans
also lets an authorized project to only redact certain categories or redo post-processing without having to rerun the entire batch on a GPU.

Spans contain identifying source text, and a stored date shift is part of the
re-identification link. Treat both as sensitive as the original note! A
researcher who only needs de-identified text should not automatically receive
all spans, original values, or shift. But it can be useful to set an automated system in place to selectively revert pseudonymization by the researcher when he/she has certain reasons for it. This can be logged and automatically approved then.

### Choose and retain date shifts outside MedDeID

The calling application—not MedDeID—chooses the date shift according to local
policy or pseudonymization requirements. It sends that integer with the note as
`metadata.date_shift_days`. Use an approved secure random process and choose
the scope deliberately. A consistent per-subject shift usually preserves the
intervals between that subject's notes; a new shift for every note does not.

MedDeID uses placeholders when the value is omitted or zero. A non-zero value
shifts recognized dates and is reported back under `processing`. The default
configuration warns when the absolute shift is below 366 days, because of higher re-identification risks, but the
institution remains responsible for selecting an acceptable range and storing
the chosen value securely.

Reversal is not an API feature. An authorized application can subtract the
stored shift from a successfully shifted date, or use the stored source span to
restore an approved original value. Some outputs are deliberately not
reversible from the replacement alone: an unparsed date may become a
placeholder, and a birth date may become a generalized age or shifted year.
Keep the original note or source spans when controlled restoration is a real
requirement.

### Plan how users report mistakes

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
2. store the API key in the organization's secret manager;
3. keep the service on a private interface or network;
4. avoid logging notes, results, span contents, metadata, or API keys; and
5. validate de-identification quality on representative local notes.

## Choose a deployment

All options expose the same API and produce the same de-identification result.
Choose primarily by the hardware your organization will operate.

| Deployment                                       | Requirements                                     | Online response\* | Batch speed\* | Choose when                                            |
| ------------------------------------------------ | ------------------------------------------------ | ----------------: | ------------: | ------------------------------------------------------ |
| [CPU](#deploy-with-cpu)                          | Docker                                           |            110 ms |    1.85 doc/s | You want the simplest setup or have no GPU             |
| [CUDA](#deploy-with-an-nvidia-gpu)               | Docker, NVIDIA driver, NVIDIA toolkit            |           13.6 ms |    65.0 doc/s | You want support across compatible NVIDIA GPUs         |
| [TensorRT](#deploy-the-optimized-nvidia-service) | Docker, NVIDIA T4, NVIDIA driver, NVIDIA toolkit |            8.7 ms |   168.8 doc/s | You have a fixed T4 and want the fastest tested option |
| [MPS](#run-on-apple-silicon)                     | Apple silicon, Python                            |            9.4 ms |    42.1 doc/s | You want to run natively on a Mac                      |

<small>\* Online response is the median for one short note; short notes averaged
314 characters. Batch speed uses 16-note requests at concurrency 8; those notes
averaged about 1,400 characters. Results are medians of three runs using the
recommended configuration for each workload. CPU, CUDA, and TensorRT were
tested on an Azure T4 host; MPS was tested on an M4 Pro. Treat the figures as
comparisons, not capacity guarantees.</small>

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
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.example .env.cpu
chmod 600 .env.cpu
```

The template already selects the Dutch public model, binds to localhost,
requires authentication, and keeps the UI and API documentation disabled. For
a first deployment, replace the image tag with its approved digest and set the
API key:

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
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.cuda.example .env.cuda
chmod 600 .env.cuda
# Replace the API key and pin the image digest in .env.cuda.

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
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.triton.example .env.triton
chmod 600 .env.triton
```

MedDeID detects the GPU from the name reported by NVIDIA and selects the
matching plan. It currently supports a plan for Turing-family GPUs (T4/RTX 20-series), and a plan for Ampere and newer GPUs. GPUs other than T4 still need performance validation.
The template selects the Dutch public model by default. For a first deployment, replace only the API key. Change the following three settings
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
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
python -m pip install '.[server]'
cp server.env.example meddeid-server.env
chmod 600 meddeid-server.env
```

The generic server template already selects the Dutch public model and the
latency profile. Replace its API key. Change the prefilled model, revision, and
language profile only when you have validated another model for this service.
Provide the API key through the institution's secret-management process where
possible.

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

Use `POST /deidentify` when clients submit one note at a time. Use
`POST /deidentify-batch` for a planned collection of notes. The supplied
deployment templates contain sensible starting settings for both cases.

Before sending a note, the calling application should retrieve or create the
date shift for the appropriate subject or study and include it in the request:

```json
{
  "text": "Controle op 15/01/2025.",
  "metadata": {
    "lang": "nl-BE",
    "date_shift_days": -731
  }
}
```

The value above is illustrative. Generate the real value from institutional
policy rather than hard-coding it. After the response, store the returned
`deid_text`, `spans`, `processing`, `warnings`, and `provenance` with the
application's document record before adding any later human corrections.

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
| Always: replace the example secret                               | `MEDDEID_API_KEY`                                                   |
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

- `GET /live` confirms that the server process is running.
- `GET /health` confirms that the service and model are ready.

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
