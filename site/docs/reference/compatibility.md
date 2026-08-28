# Compatibility

Compatibility is defined by declared package ranges and machine-readable contracts, not by sibling checkout proximity.

## Current compatibility line

| Component | Version | Distribution | Direct MedDeID contract dependencies |
|---|---:|---|---|
| `meddeid-core` | 0.2.0 | PyPI | None |
| `meddeid-language-nl` | 0.2.0 / npm 0.1.0 | PyPI and npm | `meddeid-core >=0.2,<0.3` |
| `meddeid-language-en` | 0.2.0 | PyPI | `meddeid-core >=0.2,<0.3` |
| `meddeid` | 0.2.0 | PyPI and GHCR API image | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| `meddeid-data` | 0.3.0 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| `meddeid-eval` | 0.3.0 | PyPI | `meddeid-core >=0.2,<0.3`; Dutch and English profiles `>=0.2,<0.3` |
| `meddeid-training` | 0.2.0 | PyPI | `meddeid-core >=0.2,<0.3`; `meddeid-eval >=0.3,<0.4` |
| Browser applications | 0.1.0 | Public GHCR images and source | Generated taxonomy contract version 1 |

All Python packages require Python 3.10 or later. The annotation applications require Node.js 20 or later for source-based local use.

!!! note "Published components"
    All current Python packages, the optional `@meddeid/language-nl` JavaScript
    capability, and all four container images are public releases.
    Browser-application versions identify both the reviewed source line and its
    GHCR image; the npm release is the independently versioned Dutch-language
    capability. Published package metadata and immutable artifact manifests
    take precedence. Update this page as part of every coordinated release.

## Published container manifests

All four images are multi-architecture manifests for `linux/amd64` and
`linux/arm64`. Use the version tag for ordinary installation or the digest when
an operational deployment must be byte-for-byte pinned.

| Image | Version tag | Pinning |
|---|---|---|
| `ghcr.io/stighellemans/meddeid-api` | `0.2.0` | Resolve and pin with `docker buildx imagetools inspect` |
| `ghcr.io/stighellemans/meddeid-annotate` | `0.1.0` | `sha256:72f3e0fa0935da41e635e668573ec9c434cc3e8e1ef97bc793917bdfe6a7b78d` |
| `ghcr.io/stighellemans/meddeid-curate` | `0.1.0` | `sha256:8b3dde675cadc81f42a7fc34917d7b472c1556d14bc3acd1babf5bee8699875b` |
| `ghcr.io/stighellemans/meddeid-subannotate` | `0.1.0` | `sha256:d7da6967cb29b6cf8377458959dca84626a9c0e157320b42fe8815f49e880c87` |

## Shared contracts

| Contract | Current value | Authority |
|---|---|---|
| Record schema | `meddeid.schema.v1` | `meddeid-core` |
| Offset unit | `unicode_codepoints` | `meddeid-core` |
| Taxonomy contract | version 1 | `meddeid-core/contracts/taxonomy.json` |
| Taxonomy | `ProductionLabels-v1.1` | Core taxonomy plus published annotation guidelines |
| Public language profiles | `nl-BE`, `nl-NL`, `en-GB`, `en-US` | Model bundles and language packages |
| Subannotation profile contract | `meddeid.subannotation-profile.v1` | `meddeid-subannotate` |
| Persisted profile selection | `meddeid.subannotation-profile-selection.v1` | `meddeid-subannotate` |
| Built-in subannotation profile | `neutral@1` / `core-pii-neutral@1` | `meddeid-subannotate` |
| Belgian-Dutch subannotation profile | `nl-BE@1` / `core-pii-nl-be@1` | `meddeid-language-nl` |
| Default model | `stighellemans/meddeid-dutch-synth` | Published model bundle |

## Release rule

A coordinated release should pin:

1. the immutable Git revision of every component;
2. built wheel or application checksums;
3. schema, taxonomy, and language-profile identities;
4. immutable model and dataset revisions;
5. documentation revision;
6. end-to-end verification results.

Patch releases may clarify documentation or fix behavior without changing a contract. Any incompatible record, taxonomy, profile, or bundle change requires an explicit new contract/version and migration guidance.
