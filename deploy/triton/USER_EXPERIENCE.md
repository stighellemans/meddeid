# Deployment walkthrough for review

These are illustrative command-and-response journeys, not captured GPU test
results. `<commit>`, `<digest>`, and timing placeholders stand for values from
the actual run. Docker's progress wording varies. No cloud server was started
to prepare this document. See [the test plan](CLOUD_VALIDATION.md) for execution.

## 1. Choose how to run

| Need | Route | User chooses |
|---|---|---|
| One-off notes or files | CLI or Python | Model, language |
| A service without NVIDIA hardware | CPU API | Model, language, API key |
| A straightforward NVIDIA service | CUDA API | Model, language, API key |
| Optimized NVIDIA serving | TensorRT | Model, language, API key; GPU detected automatically |

The revision is optional. Keep the example's commit to reproduce the released
model, or leave it empty for the default Hub revision. Serving profiles are
optional tuning, not an extra deployment choice. The gateway is managed by
Compose; users do not configure a second model in it.

For CPU:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.example .env.cpu
chmod 600 .env.cpu
# Edit model, language and API key in .env.cpu.
docker compose --env-file .env.cpu -f compose.yaml up --detach
docker compose --env-file .env.cpu -f compose.yaml ps
```

Expected: one healthy API container, listening on localhost. First use downloads
the Hub model; subsequent starts reuse the model cache.

For CUDA:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.cuda.example .env.cuda
chmod 600 .env.cuda
# Edit model, language and API key in .env.cuda.
docker compose --env-file .env.cuda -f compose.yaml -f compose.cuda.yaml up --detach
docker compose --env-file .env.cuda -f compose.yaml -f compose.cuda.yaml ps
```

Expected: one healthy GPU API container. No TensorRT plan or local compilation
is required. The remainder of this walkthrough concerns TensorRT.

## 2. Public model with an existing plan

On a Linux NVIDIA server with Docker Compose and NVIDIA Container Toolkit:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.triton.example .env.triton
chmod 600 .env.triton
# Generate an API key, then save it in .env.triton.
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

The relevant settings are already filled in:

```dotenv
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth
MEDDEID_REVISION=1f20655454dcbd042647cacdfff6b6802a970959
MEDDEID_LANGUAGE_PROFILE=nl-BE
MEDDEID_API_KEY=<generated-secret>
```

```bash
docker compose --env-file .env.triton -f compose.triton.yaml pull
docker compose --env-file .env.triton -f compose.triton.yaml up --detach
# If startup reports a failed dependency, this gives the useful explanation.
docker compose --env-file .env.triton -f compose.triton.yaml logs plan
docker compose --env-file .env.triton -f compose.triton.yaml ps --all
```

Illustrative result, assuming accessible release artifacts for the detected GPU:

```text
plan     | Triton plan ready at /models (sha256:<digest>)
plan     | exited with code 0
triton   | healthy
meddeid  | healthy
```

`plan` exiting successfully is normal: it is a preparation step, not a server.
The host stores the compiled model in `deploy/triton/model_repository` and
model downloads in a Docker cache volume. No compiler is installed in the
deployed gateway or runtime image. The plan itself contains model parameters.

Repeating `up --detach` reuses the deployment. Recreating the preparation step
verifies and reuses a matching cached plan. No contribution prompt appears.

## 3. No revision specified

Change only:

```dotenv
MEDDEID_REVISION=
```

Apply a model configuration change by recreating the service:

```bash
docker compose --env-file .env.triton -f compose.triton.yaml down
docker compose --env-file .env.triton -f compose.triton.yaml up --detach
docker compose --env-file .env.triton -f compose.triton.yaml logs plan
```

MedDeID resolves `main` to its current commit and continues automatically.
There is no "copy this revision and retry" prompt. The gateway uses the same
resolved commit as the plan; it does not independently download a newer `main`.

If the commit has a released plan, the result is the same as above. Otherwise
the error identifies the missing model/revision/GPU combination and directs
the user to build it locally. It never silently selects older or different
weights. Empty revision requires Hub connectivity when preparation runs;
pin the commit for repeatable deployments. A local directory needs no Hub lookup.

## 4. English model or another model

For the public English model:

```dotenv
MEDDEID_MODEL=stighellemans/meddeid-english-synth
MEDDEID_REVISION=e519d454250ba67bef62c0063dbbf242c329b750
MEDDEID_LANGUAGE_PROFILE=en-GB
# Use en-US for US formats.
```

Use the same `down`, `up`, and `logs plan` commands as above. Expected: a
different verified plan is selected; no image rebuild is needed. The service
still serves one model. Model/revision changes require recreation; clients
cannot switch models per request. A new checkpoint needs its own compiled
plan, even if its architecture is unchanged.

For a Hub model not covered by this release, local compilation is required.
For an institution's model, use the directory route in section 6.

## 5. Build a missing family plan

Run the normal startup commands first. An illustrative error is:

```text
detected NVIDIA A10G (compute capability 8.6), but ... has no published
TensorRT plan ...
Build a local plan on this server: <local-build documentation link>
The build stays on this device; nothing is published.
```

Detection keeps the official NVIDIA name. Internal filenames may use
`nvidia-a10g`; no user-maintained hardware list is needed for local detection.
Detection does not guarantee that the installed NVIDIA stack supports the GPU.

Build on a compatible NVIDIA host:

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build plan-builder
docker compose --env-file .env.triton -f compose.triton.build.yaml run --rm plan-builder
```

Illustrative progress:

```text
[Model download and verification]
[GPU and driver preflight]
[ONNX export and TensorRT compilation]
Verified Triton plan repository: /output/model_repository
Local Triton plan is ready for <model>@<commit> on <NVIDIA name> (<filename key>).
The plan stays on this device. Nothing has been published to the internet.
```

The builder selects Turing for compute capability 7.5 and Ampere-and-newer
for supported GPUs with capability 8.0 or higher. It finishes without questions,
contribution drafts, account setup or uploads. These are candidate family
builds until validated across the intended hardware.

Start the local plan using the standard images:

```bash
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-plan.yaml up --detach
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-plan.yaml logs plan
```

Expected: `Triton plan ready at /models (local)`, followed by healthy services.
The overlay selects `deploy/triton/local_model_repository`; users do not need
to publish it or enter a registry URL in `.env.triton`.

## 6. Hospital model in a local directory

The directory must contain a compatible MedDeID bundle (`bundle.json`, model
checkpoint, tokenizer and associated files), not just an arbitrary weight file.
Keep it in the hospital's approved filesystem:

```dotenv
MEDDEID_MODEL=/input-model
MEDDEID_REVISION=
MEDDEID_LANGUAGE_PROFILE=nl-BE
MEDDEID_LOCAL_MODEL_DIRECTORY=/approved/models/my-meddeid-model
```

`/approved/models/my-meddeid-model` is the host directory. `/input-model` is
where Compose mounts that same directory inside the builder and gateway.
Neither is a Hugging Face repository ID. No Hub account or token is required.

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build plan-builder
docker compose --env-file .env.triton \
  -f compose.triton.build.yaml -f compose.triton.build.local-model.yaml \
  run --rm plan-builder
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-plan.yaml \
  -f compose.triton.local-model.yaml up --detach
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-plan.yaml \
  -f compose.triton.local-model.yaml logs plan
```

Expected: local bundle validation, GPU detection, compilation, and healthy
serving. No source-model download, Hub lookup, or contribution prompt. The
plan and model stay local. Container images and build dependencies may still
be downloaded; disconnected installations must stage those beforehand.

Keep the source bundle unchanged alongside its plan. To rebuild, stop the
service and archive the exact `local_model_repository` directory to a new name
first. The builder refuses to overwrite a nonempty output directory. Do not
change model files underneath a running service.

## 7. A GPU needs a newer NVIDIA stack

There is no automatic supported-stack upgrade. This is an advanced maintainer
change: select mutually compatible pinned NVIDIA images/versions in
`deploy/triton/versions.env` and the corresponding builder image default in
`compose.triton.build.yaml`. After building, update `deploy/triton/release.json`
runtime expectations to match the candidate stack. Keep these as a reviewed candidate
checkout; do not describe it as a validated release yet.

```bash
docker compose --env-file .env.triton -f compose.triton.build.yaml build
docker compose --env-file .env.triton -f compose.triton.build.yaml run --rm runtime-builder
docker compose --env-file .env.triton -f compose.triton.build.yaml run --rm plan-builder
# Now update the deployment catalog's runtime fields to the tested candidate stack.
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-images.yaml up --detach
docker compose --env-file .env.triton \
  -f compose.triton.yaml -f compose.triton.local-images.yaml logs plan
```

The gateway usually needs no GPU-specific change. Both compiler and runtime
must use the compatible stack; building only a plan cannot repair an unsupported
driver/GPU/stack combination. This full-build route also serves hospitals that
require locally built images. Add both local-model overlays from section 6
when the source is a local directory. Builder tools never become runtime layers.

There is no contribution prompt, regardless of model source, GPU or stack.
Maintainers may use independently reported hardware results to prioritize
trusted release builds.

## 8. Errors the review must cover

| Situation | Expected experience |
|---|---|
| No model configured | Compose asks for `MEDDEID_MODEL`; no implicit Dutch fallback |
| No GPU/toolkit access | GPU detection error; fix host setup before building |
| Unavailable registry artifact | Download/authentication error; do not claim the plan exists locally |
| Hub unavailable with empty revision | Revision-resolution error; use connectivity or a pinned approved commit |
| No plan for resolved model/GPU | Local-build guidance; no different-model fallback |
| Unsupported language | Available/required language information; no silent language substitution |
| Wrong local plan GPU/model/stack or damaged files | Validation error; user-owned plan is not overwritten |
| Nonempty local build output | Stop and ask operator to preserve/move the previous build |

GPU tests must confirm these outcomes and replace illustrative responses with
actual transcripts before this walkthrough is treated as release evidence.
