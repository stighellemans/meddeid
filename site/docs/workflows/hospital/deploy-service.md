# Set up the private inference service

The Databricks job will send original patient notes to this service. It needs
an address it can reach inside the hospital, a way to prove who is calling,
and a way to check that the expected model answered. The steps below set up
that connection using the PyTorch CUDA image and a single T4 GPU.

## Choose the release

A healthy service can still be running the wrong model. Give the data team the
approved **model name and bundle contract hash**. Their first request will
compare both with the values the service reports. Separately, IT needs the
bundle's **file checksums** to verify the files it installs on the host.

As release owner, record the source commits for MedDeID, core and the language
packages. Build the serving image from those revisions, push it to the
hospital's registry and record its digest. The digest lets IT pull the exact
image you tested. Supply the approved Nginx image digest, the private model
bundle, matching Python wheels and a synthetic test case. The compatibility
guide lists the packages for the server and the smaller set the Databricks job
needs. [MedDeID 0.4.3rc5 versions and install commands](../../reference/compatibility.md#rendering-release-candidate)

The Compose file in this recipe runs PyTorch. Hospitals using TensorRT have
an additional build step for the private model. [Prepare a private TensorRT plan](#if-you-use-tensorrt)

## Prepare a private GPU host

The VM runs the model. It needs a GPU that Docker can use, persistent storage
for the bundle and permission to pull the two container images. Databricks
needs to reach the VM over HTTPS from the job network.

- **The hospital already manages Azure with Terraform.** Add the host to the
  existing project so it uses the same subnet, registry and deployment
  process. [Add the first GPU host](add-gpu-host.md)
- **This installation has its own Terraform state.** Use the standalone
  configuration. Supply your Azure Storage backend configuration,
  `subscription_id`, `resource_group`, `subnet_id`, `registry_id`,
  `host_image_id` and `ssh_public_key`. For a registry using ABAC repository
  permissions, set `registry_pull_role` to `Container Registry Repository Reader`.
  [Standalone Terraform configuration](infra/main.tf)

For the standalone configuration, run these commands from the MedDeID checkout.
Review the saved plan before applying it.

```bash
cd site/docs/workflows/hospital/infra
terraform init -backend-config="$TF_BACKEND_CONFIG"
terraform plan -out=host.tfplan
terraform apply host.tfplan
```

Point the private DNS name at the VM's private IP and allow TCP 443 from the
job network. Use a host image with Docker Compose, the NVIDIA driver, NVIDIA
Container Toolkit and Azure CLI. On the VM, check `nvidia-smi`,
`docker compose version` and GPU access from an approved CUDA container.
Verify the model files against their checksums and place the bundle in a
persistent directory the container user can read and traverse. Compose mounts
it read-only.

## Start the HTTPS service

Databricks connects to Nginx on private port 443. Nginx sends the caller's
token to the hospital's authorization service. That service checks the issuer,
the intended API (the *audience*) and the caller's permission. Only after it
allows the call does Nginx forward the note to MedDeID. The model container is
reachable only on the internal container network. In this setup the job’s
identity controls access, so the service runs without a built-in MedDeID API
key. IT can revoke that job’s access at the proxy without changing the secret
used to derive patient aliases and date shifts.

Copy the two configuration files into one release directory on the VM. Create
a hospital-owned `.env` beside them with these settings. [Compose file](compose.yaml) · [Nginx configuration](nginx.conf)

```dotenv
MODEL_IMAGE=registry.example.org/clinical/meddeid-cuda@sha256:<approved-digest>
NGINX_IMAGE=registry.example.org/platform/nginx@sha256:<approved-digest>
MODEL_DIR=/srv/clinical-model/bundle
LANGUAGE_PROFILE=nl-BE
PRIVATE_IP=<private-vm-ip>
TLS_DIR=/etc/clinical-inference/https
```

Use an Nginx image with SSL and `auth_request` support. In `TLS_DIR`, place
the certificate chain issued for the private DNS name (`fullchain.pem`), its
private key (`privkey.pem`) and the CA chain Nginx uses to verify the
authorization service (`authorization-ca.pem`). Give the Nginx process
(group 101) read access to the files and traversal access to the directory.

In that directory, create `authorize.conf` with the hospital's HTTPS
access-decision URL. The endpoint must return 2xx for an allowed caller and
401 or 403 for a denied call.

```nginx
proxy_pass https://access.example.org/clinical-inference/authorize;
```

From the release directory, sign in to the registry with the VM's managed
identity and start the containers. Set `REGISTRY_NAME` to the Azure Container
Registry name.

```bash
az login --identity --output none
az acr login --name "$REGISTRY_NAME"
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 300
docker compose ps
```

## Check what the job can reach

From the job network, call `GET /health` through the private HTTPS address
with TLS verification enabled. Try both cases.

| Call | Expected result | What it tells you |
| --- | --- | --- |
| Without a token | HTTP 401 or 403 | The gateway enforces the hospital's access decision. |
| With the authorized job token | HTTP 200 with `ready: true` | The job can reach the gateway and the model is ready to accept a note. |

`GET /live` only tells you the API process is responding. Give the data team
the private HTTPS base URL, API audience and issuing CA chain so they can
test an actual prediction. [Try one fictional note from Databricks](databricks-job.md#try-one-fictional-note)

When the certificate rotates, replace the mounted files, check the configuration
with `docker compose exec nginx nginx -t` and reload it with
`docker compose exec nginx nginx -s reload`.

## If you use TensorRT

TensorRT runs a compiled version of the model called a *plan*. The published
catalog describes plans for the public synthetic models. For a private model,
rebuild the gateway from the chosen source, compile the T4 plan and pair it
with a compatible Triton runtime. Adding model files to an older gateway image
leaves its original serving code in place.

From the MedDeID source directory, prepare a catalog for the selected version
before building on the target GPU.

```bash
export MEDDEID_VERSION=$(python deploy/read_project_version.py)
export MEDDEID_TRITON_RELEASE_CATALOG="$PWD/hospital-triton-release.json"
jq --arg version "$MEDDEID_VERSION" '.meddeid_version = $version | .plans = [] | del(.images)' \
  deploy/triton/release.json > "$MEDDEID_TRITON_RELEASE_CATALOG"
```

Use the local-model, local-plan and local-images Compose overrides for this
build. Validate the model on the target GPU and keep the catalog with the
build manifest, bundle and image digests. [Build and serve an institution-specific model](../production-deployment.md#use-an-institution-specific-model)
