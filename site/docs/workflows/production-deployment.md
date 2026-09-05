# Production deployment

<span class="source-label">Owner: meddeid</span>

Use this guide when MedDeID will run as a shared service inside your
organization. The service processes notes locally, but its output is not
automatically anonymous. Validate the model on representative local data before
using it operationally.

```text
approved client -> your TLS and access controls -> MedDeID API -> local model
```

## Choose a deployment

All deployment options expose the same API and use the same MedDeID processing
rules. Choose primarily by the hardware your organization will operate.

| Your situation | Recommended deployment |
|---|---|
| You want the simplest setup or do not have a GPU | [CPU container](#deploy-with-cpu) |
| You have an NVIDIA GPU and want portability across compatible GPU models | [CUDA container](#deploy-with-an-nvidia-gpu) |
| You have a fixed NVIDIA T4 and want the optimized option | [TensorRT for T4](#deploy-the-optimized-t4-service) |
| You want to run the service on an Apple-silicon Mac | [Native MPS](#run-on-apple-silicon) |

The T4 image is the currently available optimized TensorRT target. If your
organization needs an optimized build for another NVIDIA GPU, email
[stig.hellemans@uantwerpen.be](mailto:stig.hellemans@uantwerpen.be). A new
target must be built and validated for that GPU before it can be offered.
Do not send patient data or other sensitive information by email.

You download only the runtime you choose; selecting a GPU option does not also
download the CPU or other GPU runtimes.

## Before you start

Run MedDeID inside an approved data boundary. Place network-accessible
deployments behind your organization's authenticated TLS reverse proxy or
private service mesh. The surrounding platform remains responsible for user
access, authorization, network policy, monitoring, secrets, backups, and
incident response.

For each deployment:

1. use an immutable container digest rather than a moving tag;
2. store the API key in the organization's secret manager;
3. keep the service on a private interface or network;
4. avoid logging notes, results, metadata, or API keys; and
5. validate de-identification quality on representative local notes.

## Deploy with CPU

The CPU image is the easiest starting point and supports both AMD64 and ARM64.
Install Docker, clone the repository, and create a private environment file:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.example meddeid-production.env
chmod 600 meddeid-production.env
```

In `meddeid-production.env`, replace the image tag with the digest your
organization validated. Generate a strong API key and require it:

```dotenv
MEDDEID_API_IMAGE=ghcr.io/stighellemans/meddeid-api@sha256:<approved-digest>
MEDDEID_API_KEY=<secret>
MEDDEID_REQUIRE_API_KEY=true
```

Start the service with ordinary Compose commands:

```bash
docker compose --env-file meddeid-production.env pull meddeid
docker compose --env-file meddeid-production.env up --detach meddeid
docker compose --env-file meddeid-production.env ps
```

## Deploy with an NVIDIA GPU

Use the CUDA image when you have a compatible NVIDIA GPU but do not want the
deployment tied to one specific GPU model. The host needs an NVIDIA driver,
Docker Engine, and NVIDIA Container Toolkit.

```bash
cp .env.cuda.example .env.cuda
chmod 600 .env.cuda
# Replace the example API key and pin the validated image digest in .env.cuda.

docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

The CUDA configuration requires a working GPU and fails instead of silently
falling back to CPU.

## Deploy the optimized T4 service

Choose this option only for an NVIDIA T4 deployment. Its optimized model cannot
be assumed to work on another GPU model.

```bash
cp .env.triton.example .env.triton
chmod 600 .env.triton
# Replace the example API key and pin both validated image digests in .env.triton.

docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  up --detach
```

The deployment consists of a MedDeID API gateway and a separate T4 model
server. Compose connects them on an internal network; clients continue to use
the ordinary MedDeID API.

## Run on Apple silicon

Apple's Metal GPU is available through a native macOS installation. Docker
Desktop cannot pass it into the Linux MedDeID container, so install and run the
server directly:

```bash
python -m pip install 'meddeid[server]'
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth \
meddeid-server
```

MedDeID automatically chooses MPS when it is available and CUDA is not. Use an
explicit device setting only to diagnose or override that choice.

Use an organization-managed process supervisor when this becomes an unattended
service. The same security and validation requirements apply as for a container
deployment.

## Process individual requests or batches

Use `POST /deidentify` when clients submit one note at a time. Use
`POST /deidentify-batch` for a planned collection of notes. The supplied
deployment templates contain sensible starting settings for both cases.

If interactive requests and large batch jobs must run at the same time, operate
separate service instances so a large job cannot delay an interactive request.
Tune concurrency and batching only after measuring the real workload on the
target hardware. The [technical production
reference](https://github.com/stighellemans/meddeid/blob/main/docs/production.md#latency-and-throughput-serving-profiles)
explains the optional performance tuning for sustained traffic.

## Configure the service

Each deployment includes an environment template with suitable starting
values. Operators initially decide only:

| Setting | What to decide |
|---|---|
| Container image | Pin the exact digest that was tested and approved |
| `MEDDEID_API_KEY` | Supply a strong secret through the deployment platform |
| `MEDDEID_BIND_ADDRESS` | Keep `127.0.0.1` unless a private network or trusted proxy requires another address |

The selected image and template already contain the appropriate runtime
settings. Leave the remaining values unchanged until representative
measurements justify a different configuration. The [complete production
reference](https://github.com/stighellemans/meddeid/blob/main/docs/production.md)
explains every setting and the measured CPU, CUDA, MPS, and T4 behavior.

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
docker compose --env-file meddeid-production.env down
```

## Technical details

- [Complete production configuration and benchmark evidence](https://github.com/stighellemans/meddeid/blob/main/docs/production.md)
- [CUDA image requirements](https://github.com/stighellemans/meddeid/blob/main/deploy/pytorch-cuda/README.md)
- [TensorRT target and validation details](https://github.com/stighellemans/meddeid/blob/main/deploy/triton/README.md)
- [Released versions and container tags](../reference/compatibility.md)
