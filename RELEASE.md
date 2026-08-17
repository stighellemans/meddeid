# Release runbook

The current public PyPI release line is `meddeid-core==0.1.1`,
`meddeid-language-nl==0.1.1`, `meddeid==0.1.1`, `meddeid-data==0.2.1`,
`meddeid-eval==0.2.1`, and `meddeid-training==0.1.1`. The CPU API image is published at
`ghcr.io/stighellemans/meddeid-api:0.1.1` with multi-platform digest
`sha256:14fab911f369162b3ccb465b1793693c132720dc638996b64c711a2bb4b8e3b1`.

The steps below are the ongoing maintainer procedure for later releases.

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
2. `meddeid-language-nl`
3. `meddeid`
4. `meddeid-data`
5. `meddeid-eval`
6. `meddeid-training`

For each repository:

1. confirm a clean working tree and green CI;
2. update and review the version, changelog/release notes, licences, and notice;
3. build locally with `python -m build` and run `python -m twine check dist/*`;
4. tag the reviewed commit as `v<project.version>` and push the tag;
5. approve the protected PyPI environment;
6. verify the wheel and sdist from a clean environment; and
7. record the tag, commit, file hashes, and PyPI URL in the suite release
   manifest before moving to the dependent package.

The `meddeid` tag also triggers the container workflow. It first builds and
smoke-tests a hardened offline image, rejects fixable high or critical
vulnerabilities, then publishes `linux/amd64` and `linux/arm64` manifests with
SBOM and provenance attestations.

## Required post-publication checks

```bash
python -m venv /tmp/meddeid-release-check
/tmp/meddeid-release-check/bin/pip install 'meddeid[server]==0.1.1'
/tmp/meddeid-release-check/bin/pip install \
  'meddeid-data==0.2.1' \
  'meddeid-eval==0.2.1' \
  'meddeid-training==0.1.1'
/tmp/meddeid-release-check/bin/pip check
/tmp/meddeid-release-check/bin/meddeid model-info

docker pull ghcr.io/stighellemans/meddeid-api:0.1.1
docker image inspect ghcr.io/stighellemans/meddeid-api:0.1.1
```

Run `scripts/container_smoke.py` against the pulled image with an internal
Docker network and required API key. Confirm that `/docs` is disabled, the
container is non-root/read-only/capability-free, and the image reports the
expected model and source revisions. Record the multi-architecture manifest
digest and attestation URLs.

Do not announce the release until the public PyPI install, pulled-image smoke
test, rendered documentation, and rollback-by-digest exercise all pass.
