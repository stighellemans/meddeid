# Production deployment

<span class="source-label">Owner: meddeid</span>

The supported first deployment is the bundled CPU API container behind your
organization's TLS reverse proxy or private service mesh. It is designed for
local processing inside an approved data boundary; it is not a public cloud
service and it does not make model output automatically anonymous.

```text
approved client -> TLS, identity and rate limits -> MedDeID API -> bundled CPU model
```

## Secure starting point

For a local evaluation, install Docker, clone the `meddeid` repository, and
run:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
./scripts/start-local.sh
```

The script creates a private API key, pulls the fixed model image, starts the
hardened Compose service on `127.0.0.1`, and prints a simple browser interface
for single notes. It does not require Python or a Hugging Face account. Stop it with
`./scripts/stop-local.sh`.

For an organizational deployment:

1. use an immutable image digest rather than a moving tag;
2. store `MEDDEID_API_KEY` in a secret manager and require it;
3. keep the container on a private network and terminate TLS at a maintained
   reverse proxy or service mesh;
4. enforce the request-size limit at the proxy as well as in MedDeID;
5. never log notes, outputs, metadata, API keys, or request bodies; and
6. validate recall and unnecessary redaction on representative local notes
   before operational use.

The supplied Compose defaults run as a non-root user with a read-only root
filesystem, no Linux capabilities, no-new-privileges, bounded processes,
rotating logs, health checks, and API-key authentication. The bundled model is
fixed and runtime Hub access is disabled.

## Operations

Use `GET /live` for process liveness and `GET /health` for model readiness and
identity. Use the response `X-Request-ID` to correlate failures without
recording patient text. HTTP 503 with `Retry-After` is the overload signal.

Start with one worker and one admitted inference request. Every worker loads a
separate model copy. Measure memory, restart time, throughput, and p50/p95/p99
latency against the real note-length distribution before increasing
concurrency.

For upgrades, record the container digest, model revision, bundle identity,
package versions, and language profile. Compare old and new outputs on the
same local validation set, canary the new digest, and retain the previous
digest for rollback.

TensorRT/Triton remains an advanced prototype until a hardware-specific build,
parity evidence, and an operational GPU release are published.

The full environment-variable and release procedure is maintained in the
[`meddeid` repository](https://github.com/stighellemans/meddeid/blob/main/docs/production.md).
