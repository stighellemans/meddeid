# Hospital integration cookbook

This recipe deploys a private inference service and connects a Databricks job to it. The example uses Azure, a single NVIDIA T4, the PyTorch CUDA serving image and a Dutch model. The gateway accepts JSON over HTTPS. The job can render additional text views from the returned character offsets while the GPU is offline.

| Owner | Supplies |
| --- | --- |
| Release owner | Approved CUDA and Nginx image digests, model bundle and checksum, expected model name and bundle contract hash, matching Python wheel versions and a synthetic acceptance case |
| Hospital IT | Private subnet, DNS, registry, VM image, certificate chain, authorization endpoint and job identity |
| Data platform team | Authoritative document revisions, clinical context, a versioned mapping key, restricted Delta storage and the study's export policy |

## 1. Prepare a release

Record the source commits for MedDeID, core and each language package with the build. Build the serving image from these revisions, push it to the hospital's approved registry and record the resulting digest. The rendering candidate in this checkout uses MedDeID 0.4.3rc4 with core 0.2.4rc4 and language wheels 0.2.4rc1. Install candidate wheels from the approved release feed. The private model bundle is a separate artifact. Verify its file checksums at installation and use its *contract hash* when checking predictions.

For a TensorRT deployment, rebuild the gateway from the chosen source and pair the T4 plan with its compatible Triton runtime. Validate the model on the target GPU. Packaging a model on top of a base image retains the code in that base. The Compose example below serves PyTorch CUDA. The repository also provides a Triton serving stack and plan verification. [Production deployment](../production-deployment.md)

For a locally compiled private TensorRT plan, select the build version and catalog together. From the MedDeID source directory, prepare the catalog before building on the target GPU:

```bash
export MEDDEID_VERSION=$(python deploy/read_project_version.py)
export MEDDEID_TRITON_RELEASE_CATALOG="$PWD/hospital-triton-release.json"
jq --arg version "$MEDDEID_VERSION" '.meddeid_version = $version | .plans = [] | del(.images)' \
  deploy/triton/release.json > "$MEDDEID_TRITON_RELEASE_CATALOG"
```

Use the local model and local images Compose overrides with the resulting plan. Keep this catalog with the hospital's build manifest, bundle and image digests. The default catalog describes the published synthetic plans. [Build the TensorRT model repository](../../../../deploy/triton/README.md)

## 2. Provision the host with Terraform

**Adding MedDeID to the hospital's existing Terraform?** Follow [Add the first MedDeID GPU host](add-gpu-host.md). The recipe gives the HCL to add, names the VM, NIC and disk by workload and environment, and connects the host to the existing private network and registry.

For a separate deployment, use the [standalone configuration](infra/main.tf). Supply the Azure Storage backend configuration, `subscription_id`, `resource_group`, `subnet_id`, `registry_id`, `host_image_id` and `ssh_public_key`. The host image must include Docker Compose, the GPU driver, NVIDIA Container Toolkit and Azure CLI. Set `registry_pull_role` to the repository-reader role for an ABAC registry. Use the output IP for private DNS and allow HTTPS from the job network.

```bash
cd site/docs/workflows/hospital/infra
terraform init -backend-config="$TF_BACKEND_CONFIG"
terraform plan -out=host.tfplan
terraform apply host.tfplan
```

On the VM, check `nvidia-smi`, `docker compose version` and GPU access from an approved CUDA container. Sign in to ACR as the VM identity. Stage the verified model bundle in a persistent, read-only directory that the model container user can traverse.

## 3. Start the service behind Nginx

Copy [compose.yaml](compose.yaml) and [nginx.conf](nginx.conf) into the same release directory. Compose starts a model container on an internal network and Nginx on the VM's private port 443. Set these values in a site-owned `.env` beside the files.

```dotenv
MODEL_IMAGE=registry.example.org/clinical/meddeid-cuda@sha256:<approved-digest>
NGINX_IMAGE=registry.example.org/platform/nginx@sha256:<approved-digest>
MODEL_DIR=/srv/clinical-model/bundle
LANGUAGE_PROFILE=nl-BE
PRIVATE_IP=<private-vm-ip>
TLS_DIR=/etc/clinical-inference/https
```

Under `TLS_DIR`, install `fullchain.pem` for the private DNS name, `privkey.pem` and the authorization service's `authorization-ca.pem`. Allow Nginx group 101 to read the files and traverse the directory. The Nginx image must support SSL and `auth_request`. Add an `authorize.conf` containing the hospital's HTTPS access-decision URL, for example:

```nginx
proxy_pass https://access.example.org/clinical-inference/authorize;
```

That service validates token issuer, intended audience and caller authorization and returns 2xx for an allowed identity or 401/403 otherwise. Nginx sends the bearer token to it and forwards approved requests to MedDeID. The model container has no published host port. Give the Databricks job the HTTPS base URL, the API audience and the issuing CA chain.

```bash
az login --identity --output none
az acr login --name "$REGISTRY_NAME"
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 300
docker compose ps
```

From the job network, check that an unauthenticated request to `GET /health` receives 401 or 403 and an authorized one receives HTTP 200 with `ready: true`. Keep TLS verification enabled. When rotating the certificate, replace the mounted files, run `docker compose exec nginx nginx -t`, then reload with `docker compose exec nginx nginx -s reload`.

## 4. Run one synthetic note from Databricks

Install the approved `meddeid-core` and language wheels and `azure-keyvault-secrets` in the job environment. Configure a Databricks service credential authorized to call the gateway and read the agreed Key Vault secret version. Store a random mapping key of at least 32 bytes as hexadecimal in that version. Keep the key on the trusted driver.

Place [databricks_example.py](databricks_example.py) on the job's Python import path. It sends one synthetic batch, verifies the echoed document ID, model, bundle and language, checks for an expected name detection and renders the note with the same CPU functions used for structured data.

```python
from databricks_example import check_integration

synthetic_text = check_integration(
    credential=dbutils.credentials.getServiceCredentialsProvider("clinical-job"),
    gateway_url="https://clinical-model.example.org",
    audience="api://clinical-inference",
    model_name="<approved-model-name>",
    bundle_sha256="<approved-bundle-contract-sha256>",
    vault_url="https://<vault-name>.vault.azure.net",
    secret_name="clinical-mapping",
    secret_version="<immutable-version>",
    verify="/etc/ssl/certs/hospital-ca-bundle.pem",
)
```

`GET /live` checks the process. `GET /health` checks inference readiness. Use `POST /deidentify` for one note or `POST /deidentify-batch` with a unique ID per submitted document. Returned spans refer to the **exact original string** and can contain identifying text. Keep requests, responses, credentials and mapping offsets out of job logs.

### Request one rendered text or reuse detections locally

Both HTTP endpoints accept an optional `render` object per document. A single call returns the selected variant in `deid_text` and the original detections in `spans`. Omitting `render` preserves the default server output. With a custom render, spans describe the input. The service omits default-policy replacement strings and date-processing statistics. The effective render date defaults to `metadata.document_creation_date`. Set an explicit offset with `render.date_offset_override` when requesting a custom date shift.

```python
from meddeid_core.client import GatewayClient
from meddeid_core.rendering import apply_spans

client = GatewayClient(gateway_url, bearer_token=get_access_token)
result = client.deidentify(text, metadata={"lang": "nl-BE"}, render={"keep_original": ["Date"]})
text_for_review = result["deid_text"]

# Reuse the original detections for another column, including dates this time.
labels = apply_spans(text, result["spans"], profile_id="nl-BE")

# Each batch document may select its own render settings.
[record] = client.deidentify_batch([{
    "document_id": "example-1", "text": text, "metadata": {"lang": "nl-BE"},
    "render": {"name_replacement": True, "pseudonymization_scope": {"patient_id": "example"}},
}])
```

For keyed rendering inside the service, mount the hospital’s pinned Key Vault secret version as a read-only file on the inference container and set `MEDDEID_PSEUDONYMIZATION_KEY_FILE` to its container path. The file contains the hexadecimal key. Requests carry scope identifiers. For equivalent local results, select the same secret version, language, options and document date on the driver. A job that handles all pseudonymization locally can deploy the service without the key.

For Docker Compose, the optional override below mounts the selected version from a file provisioned by IT. Set `MAPPING_KEY_FILE` to its absolute host path and grant the container user read access. Recreate the service when selecting a new version.

```yaml
services:
  meddeid:
    environment:
      MEDDEID_PSEUDONYMIZATION_KEY_FILE: /run/secrets/meddeid-mapping
    secrets: [meddeid-mapping]
secrets:
  meddeid-mapping:
    file: ${MAPPING_KEY_FILE:?select the pinned mapping key file}
```

## 5. Connect the job to versioned documents

Schedule enrichment after the authoritative document publication succeeds. For each revision, supply a document ID, the exact converted text, a stable patient identifier for local mapping, and available clinical context. Preserve leading zeros in identifiers. Use the original document date to calculate age. Keep the model identity and a revision for the code that assembles detection metadata with each set of spans.

Use `meddeid_core.pipeline.create_annotation_record()` to save batch detections for a later job. Supply a nonempty JSON object identifying the selected source revision. A batch result requires its submitted document ID. For a single-document response omit that argument. Pass the optional expected model name and bundle hash to enforce a release pin. Keep the record in restricted storage. Before rendering, reconstruct the expected source from the row you just selected and use its exact text.

```python
from meddeid_core.pipeline import create_annotation_record, read_spans

source = {"document_id": doc_id, "revision": revision}
annotations = create_annotation_record(
    text, record, document_id=doc_id, source=source,
    expected_model={"name": model_name, "bundle_sha256": bundle_sha256},
)
# On a later run, obtain these values from the newly selected warehouse row.
spans = read_spans(row.text, annotations, expected_source={
    "document_id": row.document_id, "revision": row.revision,
})
```

1. Read a fixed snapshot and select the current eligible revision of each document. Reuse stored spans only when source text, relevant context, model and detection revision still match.
2. Detect missing annotations in bounded batches. Validate the batch IDs, source slices, language and model before accepting results. Treat a source or model mismatch as an error. A timeout or transient 503 can be retried against the same input.
3. Use the CPU renderer for each authorized view. The same patient scope, key version, algorithm and language resources produce the same aliases and date offsets in structured fields and free text. A new rendering policy or key version can reuse validated spans. With the Python inference library, `result.render(...)` uses the language reported by that prediction.
4. Commit annotations, derived texts and the rendering configuration together using a conditional Delta merge on the document revision. Retain completed blocks on retry. Record the successful input and output Delta versions. Release the approved text column to the intended cohort only after checking completeness against those versions.

```python
from meddeid_core.rendering import (
    PseudonymizationScope, apply_spans, derive_date_offset, pseudonymize_name,
)

scope = PseudonymizationScope(patient_id=stable_patient_id, secret=versioned_key)
labels = apply_spans(text, spans)
clinical = apply_spans(
    text, spans, pseudonymization_scope=scope, name_replacement=True,
    age_replacement=True, document_creation_date=clinical_date,
    keep_original={"Date", "Organization"},
)
alias = pseudonymize_name(patient_name, pseudonymization_scope=scope, label="Name:Patient")
offset_days = derive_date_offset(scope)
```

For an append-only Delta result table, the job can persist each validated block
as below. Here `block` is a DataFrame containing exactly one row per key. The
table has the same columns, including annotations, authorized text views and
provenance. `detection_revision` identifies model, bundle, metadata rules and
relevant context; `render_revision` identifies policy, package versions and the
key version. Use a single writer per selection and keep the identifiers non-null.

```python
from delta.tables import DeltaTable

fields = ["document_id", "source_revision", "detection_revision", "render_revision"]
match = " AND ".join(f"target.{name} = incoming.{name}" for name in fields)
target = DeltaTable.forName(spark, result_table)
(target.alias("target").merge(block.alias("incoming"), match)
       .whenNotMatchedInsertAll().execute())
```

A site that enriches the authoritative table in place can use a conditional update
on the same source revision. In either design, recheck eligibility and the exact
source revision at dataset release. A historical result may belong to a withdrawn
note.

Keep the adapter from local source tables to this input contract inside the hospital. Table names, source-system joins, job identities, weights, certificates and release receipts belong to the local deployment. Other hospitals can reuse the interface with their own data model.

## 6. Accept and operate the installation

Repeat the synthetic job after a VM deallocation and restart. Simulate an interruption after a committed block and confirm the next run finishes the pending revisions. Verify that re-rendering a completed selection succeeds with the GPU offline. Before a clinical backfill, validate detection quality, maximum note size and throughput on the hospital's own note types.

For a dedicated GPU, let one scheduler start it on pending detection, wait for readiness, run the job and deallocate it after completion or failure. Include cancellation recovery. To roll back, restore the previous approved image digests, matching bundle, wheels and policy. Keep previously published dataset versions and their key-version references reproducible.
