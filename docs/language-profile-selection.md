# Language profile selection

A model bundle may support one or several regional language profiles. Profile
selection is per document and deterministic; MedDeID never guesses between GB
and US English.

The resolution order is:

1. `metadata.lang` on the document, when present;
2. the default explicitly selected while loading the model;
3. the bundle profile, but only when the bundle contains exactly one profile;
4. otherwise, an actionable ambiguity error listing supported profiles.

In pseudocode:

```text
if document.metadata.lang is present:
    use the unique supported profile matching it
else if caller selected a default while loading:
    use that default
else if bundle has exactly one profile:
    use that profile
else:
    fail and list the supported profile IDs
```

This lets Python or a service set a deployment default once:

```python
deidentifier = Deidentifier.from_pretrained(
    "path/to/english-bundle",
    language_profile="en-GB",
)

# Uses the deployment default; no repeated locale argument is needed.
deidentifier(note)

# Metadata wins for an individual US document in the same call/service.
deidentifier(us_note, metadata={"lang": "en-US"})
```

For a mixed batch, the same selection and post-processing happen independently
for every document:

```python
results = deidentifier.deidentify_many([
    (gb_note, {"lang": "en-GB"}),
    (us_note, {"lang": "en-US"}),
])
```

For the single-file CLI, supply a locale only when the model has several:

```bash
meddeid deidentify note.txt \
  --model path/to/english-bundle \
  --language-profile en-GB
```

For canonical JSONL batch inference, put `metadata.lang` on each document. This
supports mixed GB/US files. `--language-profile` remains an optional fallback
for rows without `metadata.lang`.

The standard service can set the same fallback without custom Python code:

```bash
MEDDEID_MODEL=path/to/english-bundle \
MEDDEID_LANGUAGE_PROFILE=en-GB \
meddeid-server
```

The browser UI reads the supported profiles from `/health`; it hides the
language control for a single-profile model and shows locale names—not rule
versions—for a multi-profile model.

`en_GB` and `en_US` may be normalized to their hyphenated forms. Bare `en`
always fails because it cannot determine which regional post-processing rules,
date order, address formats, or identifiers apply. An unsupported or conflicting
metadata value also fails; it is never silently replaced by the default.

## Post-processing and provenance

The selected language profile post-processes decoded model spans before text is
redacted. This occurs in the shared Python engine, so Python, both CLI modes,
the HTTP service, and Torch or Triton backends use the same rules. Results expose
the selected `profile_id`. Batch manifests
also count documents by selected profile, making mixed-locale runs auditable.
