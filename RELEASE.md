# Release runbook

This checkout prepares the `meddeid==0.4.0` release candidate in the coordinated
MedDeID suite 0.3.0 line. That line reuses `meddeid-core==0.2.1`,
`meddeid-language-en==0.2.1`, `meddeid-language-nl==0.2.1`,
and the legacy-site redirect snapshot at `v0.2.0`; publishes
`meddeid-data==0.4.1`, `meddeid-eval==0.5.0`, and
`meddeid-training==0.3.0`; publishes the three browser applications at 0.3.1;
and publishes the coordinator at `v0.3.0`.

Source defaults that mention `0.4.0` are candidate identities, not evidence
that an artifact is already public. Do not update the suite's released lock or
announce the release until the post-publication checks below have captured
immutable PyPI hashes and GHCR digests.

## Publisher setup

For each Python package repository:

1. create the matching PyPI project/trusted-publisher configuration;
2. authorize its GitHub repository, `publish-python.yml` workflow, and `pypi`
   environment;
3. protect the `pypi` environment with required reviewers;
4. protect version tags and the `main` branch; and
5. enable dependency, secret, and code scanning.

Keep the GHCR package public and linked to the source repository. Recheck the
trusted-publisher subjects whenever a repository or workflow is renamed.

## Release order

Release exactly in dependency order. Do not create all tags simultaneously.

1. `meddeid-data`
2. `meddeid-eval`
3. `meddeid-training`
4. `meddeid-annotate`
5. `meddeid-curate`
6. `meddeid-subannotate`
7. `meddeid`
8. `meddeid-suite`

For each repository:

1. confirm a clean working tree and green CI;
2. update and review the version, changelog/release notes, licences, and notice;
3. build locally with `python -m build` and run `python -m twine check dist/*`;
4. tag the reviewed commit as `v<project.version>` and push the tag;
5. approve the protected PyPI environment;
6. verify the wheel and sdist from a clean environment; and
7. record the tag, commit, file hashes, and PyPI URL in the suite release
   manifest before moving to the dependent package.

The `meddeid` tag triggers the Python, CPU-image, and CUDA-image publication
workflows. Before pushing it, ensure that one clean T4 runner with labels
`linux`, `x64`, `nvidia`, and `t4-sm75` is online for the CUDA job. The Dutch
and English T4 TensorRT plans are promoted from their retained successful
candidate runs; they are not recompiled at the tag. Before tagging, run the
Dutch and English Ampere+ plan gates sequentially on one clean A100 runner and
retain that runner until its exact runtime and gateway images have been
promoted. TensorRT plan and image publication are then dispatched manually from
the signed tag with the four reviewed candidate run IDs.

The CPU workflow builds and smoke-tests the hardened offline image, rejects
fixable high or critical vulnerabilities, then publishes `linux/amd64` and
`linux/arm64` manifests with SBOM and provenance attestations. The GPU workflow
details and publication guard are described below.

## Required post-publication checks

```bash
python -m venv /tmp/meddeid-release-check
/tmp/meddeid-release-check/bin/pip install 'meddeid[server]==0.4.0'
/tmp/meddeid-release-check/bin/pip install \
  'meddeid-data==0.4.1' \
  'meddeid-eval==0.5.0' \
  'meddeid-training==0.3.0'
/tmp/meddeid-release-check/bin/pip check
/tmp/meddeid-release-check/bin/meddeid model-info \
  --model stighellemans/meddeid-dutch-synth

docker pull ghcr.io/stighellemans/meddeid-api:0.4.0
docker image inspect ghcr.io/stighellemans/meddeid-api:0.4.0
docker pull ghcr.io/stighellemans/meddeid-api:0.4.0-cuda12.9
docker image inspect ghcr.io/stighellemans/meddeid-api:0.4.0-cuda12.9
docker pull ghcr.io/stighellemans/meddeid-triton-gateway:0.4.0
docker pull ghcr.io/stighellemans/meddeid-triton-runtime:0.4.0-trt26.07
oras pull ghcr.io/stighellemans/meddeid-triton-plan-t4-sm75:0.4.0-trt26.07-fp16-dutch-synthetic
oras pull ghcr.io/stighellemans/meddeid-triton-plan-t4-sm75:0.4.0-trt26.07-fp16-english-synthetic
oras pull ghcr.io/stighellemans/meddeid-triton-plan-ampere-plus:0.4.0-trt26.07-fp16-dutch-synthetic
oras pull ghcr.io/stighellemans/meddeid-triton-plan-ampere-plus:0.4.0-trt26.07-fp16-english-synthetic
```

GPU images have separate hardware gates. The PyTorch CUDA workflow builds
`meddeid-api:<version>-cuda12.9`, requires real CUDA and authenticated API
inference on its T4 runner, then publishes an AMD64 image with SBOM and
provenance on a matching release tag. The target-driven TensorRT workflow
builds each public model plan on an eligible runner and records its complete
semantic comparison with PyTorch. For this release, run the Dutch and English
Ampere+ candidates sequentially on one clean A100 runner. After all technical
gates pass, the publication workflows promote the exact retained T4 and
Ampere+ plan repositories, plus the exact validated weight-free runtime and
gateway images; they do not recompile them. Named A10G and L4 targets remain
available for optional specialized builds and are not additional release
plans. A CPU image passing does not authorize either GPU artifact; a TensorRT
runtime without matching model-plan artifacts is an incomplete release.

Run `scripts/container_smoke.py` against each pulled API/gateway image with an
internal Docker network and required API key. Confirm that `/docs` is disabled,
the containers are non-root/read-only/capability-free, and the images report
the expected software, model, and source revisions. Run pulled-artifact smoke
and semantic checks for the T4 and Ampere+ plan families on the already
provisioned matching hosts, not only against local candidates. Record every
immutable digest and attestation URL in the suite release candidate before
finalizing its released lock.

Do not announce the release until the public PyPI install, pulled-image smoke
test, rendered documentation, and rollback-by-digest exercise all pass. The
retained-plan and image publication workflows are manual and must be dispatched
from the signed release tag with the reviewed candidate run IDs.

## Failure diagnosis before another paid candidate run

FP16 remains the preferred CUDA release target for throughput. Per the user's
release decision, semantic differences, including reduced masking relative to
the CPU reference, are **report-only**. They do not block publication and do
not trigger numerical repair/rebuild cycles. CUDA and TensorRT release
comparisons use `--semantic-policy report-only` and must finish the entire
fixture, write detailed JSON plus Markdown reports, and retain their counts.

A report's `passed` field means execution and the selected acceptance policy
passed; `strict_passed` separately records exact semantic agreement. Never
rewrite nonzero differences as zero or describe report-only success as exact
parity. Reduced masking counts are relative to the CPU reference, not annotated
ground truth. Missing/duplicate outputs, model identity mismatches, unhealthy
services, HTTP errors and all other technical/security/publication gates remain
blocking. The comparator's standalone default remains strict for callers that
do not explicitly select report-only mode.

A failed gate is a stop condition, not an automatic request to rebuild all
images. Preserve the small `triton-<target>-<model>-evidence-<run-id>` artifact,
including parity JSON, startup reports and logs. The compiled model repository
is a separate artifact; download it only when engine inspection is necessary.
Deallocate idle dedicated Azure VMs after saving diagnostics. Retained disks
and networking can still incur charges; delete the release resource group
when it is no longer needed.

Before retrying, reproduce the exact failing contract using existing images
on a diagnostic host. Do not register a new JIT runner for diagnosis: that
registration intentionally destroys the preceding job's images and cache.
Diagnostic reuse is not release evidence. Final release candidates must still
pass every gate at the exact reviewed commit. The Ampere+ Dutch and English
candidates intentionally run sequentially on the same dedicated A100 host so
the validated runtime and gateway images can be promoted without rebuilding.

For a parity failure, keep the selected model revision, bundle, language profile,
fixture revision and original request batch fixed. The comparison helper builds
nothing; it expects the CPU reference on port 8001 and the already-built CUDA
image plus staged model/fixture variables from the validation workflow:

```bash
./deploy/validate_cuda_comparison.sh parity baseline --document-id targeted-difficult-0023
./deploy/validate_cuda_comparison.sh parity fp16-latency --document-id targeted-difficult-0023
./deploy/validate_cuda_comparison.sh parity fp32-throughput --document-id targeted-difficult-0023
./deploy/validate_cuda_comparison.sh parity fp16-throughput --document-id targeted-difficult-0023
```

`--document-id` replays the entire original batch containing that document, in
its original order. It is diagnostic only, never sufficient release evidence.
Repeat any suspected scheduling-dependent failure and then run the full fixture
without `--document-id` for both public models. Record every semantic difference;
do not remove a differing document, suppress a changed span, or change the
precision only inside the test. Any runtime/default change needs matching public settings,
new performance evidence, and the complete release gates.

The workflow now checks merged gateway mounts and local bundle identity before
CUDA/TensorRT compilation, then full CPU/CUDA parity before TensorRT compilation
or benchmarks. It checkpoints parity reports after each batch and writes changed
document IDs and fields to the live log. A stopped/restarting container fails
readiness immediately when its container ID is supplied. A benchmark or full
release run must not be used as the first test of a mount/configuration repair.

Freeze the release scope before final gates, including user-facing wording,
signing configuration and dependency order. Avoid additional cosmetic releases
while expensive candidate gates are running. Inspect actual job steps/logs;
`in_progress` or CPU usage alone proves activity, not correctness. Use bounded
state-change monitoring and report only meaningful changes.
