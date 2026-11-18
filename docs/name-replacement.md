# Stable name replacement

Source version 0.4.3 adds one choice: `metadata.name_replacement`, a strict
Boolean that defaults to `false`. Published 0.4.2 servers do not accept it.
It applies to `Name:Patient`, `Name:Caregiver` and `Name:Other` in Python,
JSONL batch and both HTTP endpoints. All replacements stay inside `[]`.

| Input label | `false` or omitted | `true` |
|---|---|---|
| `Name:Patient` | `[Name:Patient]` | A stable name alias |
| `Name:Caregiver` | `[Name:Caregiver]` | A stable name alias |
| `Name:Other` | `[Name:Other]` | A stable name alias |
| Other categories | Their original labels | Their original labels; dates/ages still use their existing settings |

The switch changes rendering after detection. It does not change which names
the model or trusted metadata detect. There is no patient-only exception.
Unsupported name shapes or custom profiles without a name provider retain the
original label and return `name_parse_fallback` or `name_replacement_unsupported`.

## Configure once per study

Before starting the engine, provision `MEDDEID_NAME_KEY` (hex encoding of at
least 32 random bytes) and `MEDDEID_NAME_SCOPE` (a nonempty stable study scope)
from protected deployment configuration. Python callers can instead pass
`name_key=key_bytes, name_scope=scope` to `Deidentifier.from_pretrained()`.
Settings are captured when constructing the engine; restart to change them.
Do not send keys or scopes in document metadata: the runtime rejects those fields.
A true request without configured settings fails before inference (HTTP 422).

Use the same key, scope, algorithm and pinned locale resources in every worker
and retry. Do not generate a key per document or process. A separate scope
provides separate derivation for another study. There is no shared person
mapping table, database call or dependency on task execution order. Each Python
worker loads its language resources and caches the filtered catalog once per
profile. Warm calls normalize the name, compute one HMAC and select its alias.
The catalogs do not grow with the number of patients or documents.

## Use the Boolean

```python
result = deid(
    "Mark Peters kwam op controle.",
    metadata={
        "name_replacement": True,
        "document_creation_date": "2026-05-14",
        "date_shift_days": 400,
    },
)
```

The HTTP request uses the same metadata (`true` in JSON). JSONL batch files
carry the Boolean in each record's `metadata`. For a single text file:

```bash
meddeid deidentify note.txt --model /models/reviewed-bundle --name-replacement
# --no-name-replacement, or omission, keeps the original name labels.
```

`processing.name_replacement` reports `enabled`, `algorithm`, `detected_spans`,
`replaced_spans` and `placeholder_spans`. It contains no key, scope or name map.
Normal inference spans still contain source text; only `deid_text` is the
rendered note. Keep ordinary result records inside the trusted data boundary.

## Reproducible examples

These synthetic examples use `bytes(range(32))`, scope `study-a`, algorithm
`meddeid.name-spelling.v1` and language package 0.2.2. This key is a test fixture.

| Source | nl-BE | en-GB | en-US |
|---|---|---|---|
| Mark Peters | `[Sue-Ann Colen]` | `[Henning Pugh]` | `[Pheobe Means]` |
| M. Peters | `[B. Altuntas]` | `[L. Haigh]` | `[J. Loo]` |
| Mark P. | `[Maïssara H.]` | `[Nikolas C.]` | `[Henric E.]` |
| Dr. Peters | `[Dr. Skyla]` | `[Dr. Hawkins]` | `[Dr. Brown]` |

Every occurrence of a normalized spelling gets the same alias, including across
name subtypes. Titles are removed before derivation and restored in presentation.
Unicode, whitespace, case and optional initial periods normalize; case and
punctuation are retained in the display. Full names and abbreviations remain
separate. This is repeatable rendering of source spans, not an entity-resolution
system and not a promise that re-detecting already replaced text is idempotent.

Finite aliases can collide; they are not patient IDs. Stable aliases preserve
linkability within a scope and do not make a document anonymous. No gender,
ethnicity or clinical attribute is inferred from the selected replacement.

## Names alongside dates and ages

The name Boolean is independent of `date_shift_days`. Omit the latter for
`[Date]` / `[Age_Birthdate]`; zero also yields labels and a warning. A nonzero
shift enables the existing date/age rendering. Always supply the **original**
`document_creation_date` when available, before any date shifting.

With the default age policy, Dutch profile and `date_shift_days=400`:

| Detected span | Original document date | Output |
|---|---|---|
| `Date`: `15/01/2025` | `2026-05-14` | `[19/02/2026]` |
| `Age_Birthdate`: `14/09/2022` | `2026-05-14` | `[3 jaar, 7 maanden oud]` |
| `Age_Birthdate`: `14/05/2013` | omitted | `[2014]` plus `birthdate_year_fallback` |

Explicit age expressions can be generalized without a document date. A full
birthdate needs that reference to compute age; without a usable reference it
falls back to the shifted year. The engine never substitutes today's date.
See [the age policy](age-granularity.md) for calendar precision by age band.

## Build and release order

Source changes span core 0.2.2, both language packages 0.2.2, and runtime 0.4.3.
Docker/Compose and CI pin the corresponding immutable dependency commits.
A source build therefore includes the new code before PyPI publication.
For ordinary pip installs, release core first, then both language packages,
then MedDeID. A push alone does not publish these packages.

Rebuild the Python application image (or Triton gateway) to serve the Boolean.
The model weights, tokenizer and TensorRT engine are unchanged by rendering.
A local installed-package job needs the new package version as well; updating
an HTTP gateway does not update Python packages inside a separate job image.
The existing UZA spelling renderer remains compatible with the nl-BE fixtures.

GitHub's `Container` workflow builds and verifies CPU images on main or manual
runs, but publishes only for a matching `v0.4.3` tag. That tag also starts the
Python publishing workflow. CUDA uses the separate candidate-build, validation
and promotion workflows. Treat an image as available only after its release
workflow succeeds; this source change does not publish or deploy an image.
