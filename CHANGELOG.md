# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

## [Unreleased]

## [0.3.0] - 2026-09-05

- Added separate production GPU artifacts: a CUDA 12.9 PyTorch API image with
  the same offline/hardened contract as the CPU image, plus target-specific
  TensorRT/Triton images. NVIDIA T4 release gates now require real GPU
  execution, authenticated API inference, PyTorch/TensorRT output parity,
  benchmark evidence, vulnerability review, SBOM, and provenance before
  publishing explicitly versioned GPU tags.
- Added a checked TensorRT target catalog and one target-driven GPU workflow.
  The optimized T4 plan is the sole ready target; A10G and L4 are declared as
  build-on-request candidates that can be validated through the same parity,
  benchmark, size, security, and provenance gates but cannot be published
  until their reviewed target status becomes ready.
- Reduced the TensorRT deployment to a projected TensorRT-only runtime plus a
  gateway image that omits PyTorch and checkpoint weights. Binary tensor
  transport, 64-window chunks, and concurrent gateway calls improve batching.
  CPU/TensorRT artifacts no longer duplicate model files in metadata-only
  layers, while the CUDA image defaults to measured FP16 eager inference and
  omits compiler-only Triton code, headers, and static archives.
- Added process-wide `latency` and `throughput` serving profiles. CUDA
  throughput deployments use bounded cross-request window microbatching;
  the measured TensorRT default uses request-local 64-window batches across
  four weight-free gateway workers without a second queue. Admission limits
  adapt to the selected profile, and health output reports the active
  scheduler policy.
- Added checked compressed-transfer and unpacked-size budgets for the CPU,
  portable CUDA, TensorRT server, and weight-free gateway artifacts. Release
  CI rejects size regressions as well as accidental cross-framework payloads.
- Made the `research` and `contributor` extras resolve against the current
  suite without selecting the older self-referential inference/training
  extras; their required training dependencies are declared directly.
- Aligned CPU, CUDA, and gateway image inputs to the exact released 0.2.0
  core and language-package tag commits used by clean PyPI installations.
- Reduced each published GPU environment template to image identity, API key,
  bind address, and one performance choice: `latency` or `throughput`. The
  images now carry the benchmarked batching, worker, precision, transport, and
  admission defaults; detailed controls remain advanced overrides.
- Automatic PyTorch device selection now prefers CUDA, then Apple MPS, and
  falls back to CPU; an explicit `--device` or Python `device=` still wins.
- Validated native Apple MPS against CPU on an M4 Pro across interactive,
  batched ETL, and long-note HTTP workloads. MPS preserved semantics over the
  300-document public fixture and was 1.6--2.0 times faster than CPU. The
  throughput profile now enables bounded PyTorch microbatching on MPS as well
  as CUDA; eager execution remains the default because MPS compilation
  regressed the measured workloads.
- `meddeid model-info` now inspects model identity, contracts, profiles, files,
  and software without constructing an inference runtime. Administrators can
  add `--verify-runtime` to load the weights and verify backend/device readiness;
  `runtime.checked` makes the distinction explicit in saved output.
- The local evaluation launcher requires every tester to select a Hub model and
  language profile explicitly; `--build` additionally tests unpublished source.
- The launcher opens the local UI when supported and always prints an API-key
  recovery command, so opening a non-clickable URL cannot lose the copied key.
- Local browser launches prefill the API key through a cleared URL fragment;
  the UI shows the active model/profile, while local builds can select a Hub
  model, revision, and default language profile explicitly.
- `deidentify` and `batch` now accept either a positional input path or an
  explicit `--input` option, and reject missing, unreadable, or non-file inputs
  before model resolution and loading.
- Make `meddeid models` actionable with one concise handoff explaining that a
  listed ID becomes the `--model` value for inference commands.
- CLI and source-server launches now require an explicit model selection instead
  of silently choosing the Dutch synthetic baseline; `meddeid models` lists the
  public model families, regional profiles, and validation scope.
- Model loading status is flushed immediately, distinguishes a real first
  download from a Hugging Face cache hit, reports elapsed time and download
  size, suppresses misleading zero-byte cache progress bars, and disables the
  lingering progress monitor that could redraw completed transfers while the
  Python interpreter was exiting.
- Inference results now expose required per-result provenance across Python,
  CLI JSON, canonical JSONL, and HTTP. Results remain flat in the logical order
  `deid_text`, `spans`, `processing`, `warnings`, `provenance`; the selected
  language profile now lives inside concise consumer provenance beside software
  and model identity. Runtime, dependency, cache, and path details remain in
  the local administrator `model-info` view and batch manifests.
- `model-info` reports unpinned Hub requests as `requested_revision: "latest"`
  beside the immutable resolved commit and includes the actual cache/local root
  plus model-file inventory. Unauthenticated health output is now minimal.
- Servers can optionally restrict exact model sources and request-selectable
  language profiles with `MEDDEID_ALLOWED_MODELS` and
  `MEDDEID_ALLOWED_LANGUAGE_PROFILES`.
- HTTP OpenAPI now describes request metadata, inference spans, and provenance
  with explicit fields instead of arbitrary `additionalProp` dictionaries;
  unknown request fields fail validation. Direct server deployments can load a
  validated, reproducible `KEY=VALUE` configuration with
  `meddeid-server --env-file` while injected environment secrets retain
  precedence.

## [0.2.0] - 2026-08-27

- Language-package scaffolding no longer emits a separate ruleset version;
  rulesets have stable identities and rely on package and Git provenance.
- Added typed workflow-template validation, an onboarding action-adapter
  registry, and semantic artifact validation for language, corpus, checkpoint,
  bundle, and interface gates.
- Guided workflows now use unversioned locale identities and reject selectors
  such as `en-GB@1`; independent artifact versions remain in their manifests.
- Expanded language scaffolding and conformance to cover Python/JavaScript
  registration, per-profile manifests, source locks, resource commands, and
  wheel/sdist/npm package checks.
- Added suite-wide declarative age granularity, deterministic date shifting,
  safe placeholder defaults, structured warnings/provenance, trusted birth-date
  recovery, and consistent replacements across Python, CLI, batch, and HTTP.
- HTTP and single-file CLI JSON no longer echo the complete original note;
  span source fragments and canonical research JSONL remain unchanged for
  validation and evaluation.
- Added the `meddeid guide`, `meddeid doctor`, and `meddeid workflow` front
  door with explicit scientific branches, just-in-time operational choices,
  checksummed resumable manifests, safe invalidation/archiving, detached jobs,
  component-command dry runs, and ten prevalent suite templates.
- Added the simplified `meddeid start`, `meddeid status`, and `meddeid next`
  experience, grouping those templates under six common goals with nested
  choices, concise progress, numbered answers, and `?` explanations.
- Added `meddeid[research]` and `meddeid[contributor]` installation extras.
- Added installed discovery support for separate `en-GB` and `en-US`
  language profiles and removed the Dutch-only source-tree fallback.
- Unified language-profile selection and post-processing provenance across
  Python, single and batch CLI, HTTP, browser, and guided deployment paths;
  multi-profile services now support a locale fallback while bundles remain
  authoritative for locale behavior.

## [0.1.1] - 2026-08-17

- Published the first externally supported MedDeID inference release.
- Added public installation, compatibility, licensing, and verification
  metadata.
- Established independent CI and immutable release artifacts.

For earlier migration history, consult the repository's Git history.
