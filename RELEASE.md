# Release runbook

This checkout prepares the `meddeid==0.3.0` release candidate in the coordinated
MedDeID suite 0.2.0 line. That line publishes `meddeid-core==0.2.1`,
`meddeid-language-en==0.2.1`, `meddeid-language-nl==0.2.1`,
`meddeid-data==0.4.0`, `meddeid-eval==0.4.0`, and
`meddeid-training==0.2.1`. It also publishes the three browser applications at
0.2.0, the legacy-site redirect snapshot at `v0.2.0`, and the coordinator at
`v0.2.0`.

Source defaults that mention `0.3.0` are candidate identities, not evidence
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

1. `meddeid-core`
2. `meddeid-language-en`
3. `meddeid-language-nl`
4. `meddeid`
5. `meddeid-data`
6. `meddeid-eval`
7. `meddeid-training`
8. `meddeid-annotate`
9. `meddeid-curate`
10. `meddeid-subannotate`
11. `meddeid.github.io`
12. `meddeid-suite`

For each repository:

1. confirm a clean working tree and green CI;
2. update and review the version, changelog/release notes, licences, and notice;
3. build locally with `python -m build` and run `python -m twine check dist/*`;
4. tag the reviewed commit as `v<project.version>` and push the tag;
5. approve the protected PyPI environment;
6. verify the wheel and sdist from a clean environment; and
7. record the tag, commit, file hashes, and PyPI URL in the suite release
   manifest before moving to the dependent package.

The `meddeid` tag also triggers the CPU, CUDA, and TensorRT container workflows.
Before pushing it, ensure that a clean, ephemeral T4 host is registered as a
self-hosted GitHub Actions runner with labels `linux`, `x64`, `nvidia`, and
`t4-sm75`. Run both GPU workflows manually from the exact candidate commit with
publication disabled; retain their parity, benchmark, image-size, GPU-memory,
scan, and image-inspection evidence. A tag must not be pushed while the T4
runner is offline because the release would be only partially published.

The CPU workflow builds and smoke-tests the hardened offline image, rejects
fixable high or critical vulnerabilities, then publishes `linux/amd64` and
`linux/arm64` manifests with SBOM and provenance attestations. The GPU workflow
details and publication guard are described below.

## Required post-publication checks

```bash
python -m venv /tmp/meddeid-release-check
/tmp/meddeid-release-check/bin/pip install 'meddeid[server]==0.3.0'
/tmp/meddeid-release-check/bin/pip install \
  'meddeid-data==0.4.0' \
  'meddeid-eval==0.4.0' \
  'meddeid-training==0.2.1'
/tmp/meddeid-release-check/bin/pip check
/tmp/meddeid-release-check/bin/meddeid model-info \
  --model stighellemans/meddeid-dutch-synth

docker pull ghcr.io/stighellemans/meddeid-api:0.3.0
docker image inspect ghcr.io/stighellemans/meddeid-api:0.3.0
docker pull ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9
docker image inspect ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9
docker pull ghcr.io/stighellemans/meddeid-triton-gateway:0.3.0
docker pull \
  ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.3.0-trt26.07-fp16
```

GPU images have separate hardware gates. The PyTorch CUDA workflow builds
`meddeid-api:<version>-cuda12.9`, requires real CUDA and authenticated API
inference on its T4 runner, then publishes an AMD64 image with SBOM and
provenance on a matching release tag. The target-driven TensorRT workflow
builds the catalog target on its matching runner, requires semantic parity
with PyTorch, records benchmark evidence, and publishes only when that target
is marked `ready`. A release-tag push selects the sole ready default,
`t4-sm75`, and publishes
`meddeid-triton-t4-sm75:<version>-trt26.07-fp16`. Manual runs may validate the
`a10g-sm86` and `l4-sm89` build-on-request candidates, but the workflow refuses
to publish them until their evidence is reviewed and their catalog status is
promoted. A CPU image passing does not authorize either GPU artifact, and no
target-independent TensorRT tag is used.

Run `scripts/container_smoke.py` against each pulled API/gateway image with an
internal Docker network and required API key. Confirm that `/docs` is disabled,
the containers are non-root/read-only/capability-free, and the images report
the expected software, model, and source revisions. Re-run the T4 semantic
parity check against the pulled TensorRT/gateway pair, not only the local
candidate. Record every immutable digest and attestation URL in the suite
release candidate before finalizing its released lock.

Do not announce the release until the public PyPI install, pulled-image smoke
test, rendered documentation, and rollback-by-digest exercise all pass.
