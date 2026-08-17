# Production deployment

This guide is for an operator deploying MedDeID inside an approved clinical
data boundary. Local processing reduces data movement; it does not make model
output anonymous or remove the need for institutional validation.

## Supported production shape

The supported first release is the bundled CPU API image behind an
organization-managed reverse proxy or private service mesh:

```text
approved client -> TLS/auth/rate limits -> MedDeID API -> local CPU model
```

The image contains the model and sets `MEDDEID_OFFLINE=true`; no note or model
request needs to leave the deployment boundary. The container itself provides
API-key authentication and workload limits. The surrounding platform must
provide TLS, client identity where required, network policy, rate limits,
central secret storage, monitoring, backup policy, and incident response.

TensorRT/Triton is not a supported release target until a GPU-specific plan and
parity evidence are published.

Release `0.1.0` is available for AMD64 and ARM64. Pin this immutable
multi-platform digest in production:

```text
ghcr.io/stighellemans/meddeid-api@sha256:8cee6d10f68adb432802e5da1e31651d215804e42549987645281a0a0d2ab5f6
```

## Minimum secure configuration

1. Pull an immutable image digest, not a moving tag.
2. Store a random `MEDDEID_API_KEY` in the platform secret manager.
3. Set `MEDDEID_REQUIRE_API_KEY=true`.
4. Keep `MEDDEID_DOCS_ENABLED=false` unless interactive documentation is
   explicitly needed.
5. Keep `MEDDEID_UI_ENABLED=false` unless the single-note browser interface is
   explicitly needed.
6. Bind MedDeID only to a private interface. The supplied Compose file defaults
   to `127.0.0.1`.
7. Terminate TLS at a maintained reverse proxy or service mesh and enforce its
   request-body limit at or below `MEDDEID_MAX_REQUEST_BYTES`.
8. Do not log request bodies, response bodies, headers containing API keys, or
   metadata. Treat all input, output, manifests, caches, and traces as
   sensitive.
9. Validate recall and unnecessary redaction on representative local notes
   before operational use.

Generate a key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Send it as a bearer token:

```bash
curl --fail-with-body https://meddeid.example.org/deidentify \
  -H "Authorization: Bearer ${MEDDEID_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Patiënt Jan Peeters kwam op controle."}'
```

## Capacity and availability

Start with one API worker, one admitted inference request, four Torch threads,
4 vCPU, and 8 GiB RAM. Prefer `/deidentify-batch` for throughput. Every extra
worker loads another model copy and requires independent memory measurement.

Use:

- `GET /live` for process liveness;
- `GET /health` for model/runtime readiness;
- `X-Request-ID` for request correlation without recording patient text; and
- HTTP 503 plus `Retry-After` as the back-pressure signal.

Measure p50/p95/p99 latency, documents and characters per second, memory, and
restart time on the real note-length distribution. Configure client timeouts
and retries only for idempotent requests, using bounded exponential backoff.

## Upgrades and rollback

Record the image digest, model revision, bundle hash, package versions, and
language-profile version returned by `/health`. Before an upgrade:

1. run the same local validation set against old and new digests;
2. compare span and rendered-text changes;
3. confirm resource and memory limits;
4. deploy a canary without logging clinical payloads; and
5. retain the previous digest for rollback.

Never replace the model directory or TensorRT plan in place. Deploy a new
identified artifact and roll back by digest.

## Operational acceptance checklist

- Image signature/provenance, SBOM, digest, and vulnerability review accepted.
- API key stored outside Compose and source control.
- TLS, firewall/network policy, request limit, and rate limit verified.
- Container runs non-root, read-only, capability-free, and without runtime Hub
  access.
- Health alerts and restart policy tested.
- Local recall and unnecessary-redaction acceptance thresholds signed off.
- Human review and incident handling paths documented.
- Data retention and deletion behavior verified for logs and outputs.
