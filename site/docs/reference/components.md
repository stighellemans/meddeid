# Components

Each component is independently packaged, versioned, tested, and licensed. The
grouped suite workspace is a staging aid, not a runtime dependency. Distribution
status differs by component and is stated below.

<div class="component-grid" markdown>

<div class="component-card" markdown>
### `meddeid`

Python API, CLI, batch inference, decoder, local model loading, and optional HTTP service.

**PyPI:** `pip install meddeid`
</div>

<div class="component-card" markdown>
### `meddeid-core`

Language-neutral schema, taxonomy, ordered model labels, normalization, and offset validation.

**PyPI:** `pip install meddeid-core`
</div>

<div class="component-card" markdown>
### `meddeid-language-nl`

Dutch parsing, rendering, and post-processing for `nl-BE` and `nl-NL`, plus
versioned resources shared by those capabilities.

**PyPI:** `pip install meddeid-language-nl`

**npm:** `npm install @meddeid/language-nl@0.1.0`
</div>

<div class="component-card" markdown>
### `meddeid-language-en`

English parsing, rendering, and locale-specific post-processing for `en-GB`
and `en-US`, provided through the same language-profile contract.

**PyPI:** `pip install meddeid-language-en`
</div>

<div class="component-card" markdown>
### `meddeid-data`

Hospital project import, stable identifiers, deterministic splits, synthetic generation, Synthea integration, and data validation.

**PyPI:** `pip install 'meddeid-data[parquet]'`
</div>

<div class="component-card" markdown>
### `meddeid-training`

Ordinary fitting, publication-grade epoch selection/refit, readable training
diagnostics, and self-contained bundle export.

**PyPI:** `pip install 'meddeid-training[train,plots]'`
</div>

<div class="component-card" markdown>
### `meddeid-eval`

Exact-span and character metrics, core-PII recall, non-PII redaction, detailed
comparison figures, and note-clustered stability analysis.

**PyPI:** `pip install 'meddeid-eval[plots]'`
</div>

<div class="component-card" markdown>
### `meddeid-annotate`

Local, single-assignment primary-span annotation application. Reads and writes canonical JSONL.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-annotate:0.1.0`

The public source repository remains available for development; no npm release
is required to run the application image.
</div>

<div class="component-card" markdown>
### `meddeid-curate`

Optional multi-annotator comparison, explicit reconciliation decisions, and audited current primary gold.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-curate:0.1.0`

The public source repository remains available for development; no npm release
is required to run the application image.
</div>

<div class="component-card" markdown>
### `meddeid-subannotate`

Gold-only core-PII character subannotation and checksummed benchmark-bundle
export. It is language-neutral by default and discovers optional semantic
profiles registered by installed `meddeid-language-*` packages. A workspace
persists its selection and requires a backed-up migration to change it after
review begins.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-subannotate:0.1.0`

The public source repository remains available for development; no npm release
is required to run the neutral-profile application image.
</div>

</div>

## Supporting surfaces

| Surface | Status | Responsibility |
|---|---|---|
| `stighellemans.github.io/meddeid` | Public | Canonical MedDeID landing page and suite documentation |
| `meddeid-demo` | Public | Hosted non-clinical demonstration for synthetic or otherwise non-sensitive text |
| Hugging Face collection | Public artifacts | Current model, synthetic corpus, synthetic benchmark |
| Suite `publication/` | Maintainer staging | Cards, checksums, guidelines, release manifests |
| Suite `internal/` | Private maintainer records | Migration provenance and pre-publication lock |
| Suite `comparators/` | External | Independently licensed benchmark systems |

## Documentation ownership

This site explains which component to use and how cross-component workflows fit together. A component repository is authoritative for its API, CLI flags, configuration keys, tests, and release notes. Artifact cards are authoritative for model or dataset contents, limitations, licences, and immutable revisions.
