# Data contract

<span class="source-label">Authority: meddeid-core</span>

Canonical MedDeID records are newline-delimited JSON objects. The same record shape travels through import, inference, annotation, training, and evaluation.

## Minimal document

```json
{
  "document_id": "doc_01J...",
  "text": "Jan Peeters kwam op controle.",
  "spans": []
}
```

## Annotated document

```json
{
  "document_id": "doc_01J...",
  "text": "Jan Peeters kwam op controle.",
  "spans": [
    {
      "begin": 0,
      "end": 11,
      "text": "Jan Peeters",
      "label": "Name:Patient",
      "category": "Name",
      "subtype": "Patient"
    }
  ],
  "annotated": true,
  "metadata": {
    "document_type": "consultation"
  }
}
```

## Required rules

- `document_id` is stable within one dataset revision.
- `text` is the exact source string to which offsets refer.
- `spans` is the only canonical top-level primary-span container.
- `begin` is inclusive and `end` is exclusive: `[begin, end)`.
- offsets count Unicode code points, not UTF-8 bytes or UTF-16 code units.
- `span.text` must equal `text[begin:end]` under that offset convention.
- labels must exist in the versioned taxonomy contract.
- a reviewed document with no PII has `spans: []` and an explicit completed state.

Alternatives such as `doc_id`, `annotations`, `entities`, `items`, `start`, `Category`, or `Subtype` are not separate supported dialects. The core normalizer can standardize selected fields inside an otherwise canonical record; it does not guess arbitrary schemas.

## Nested benchmark subannotations

Detailed evaluation can partition a primary gold span into absolute character segments:

```json
{
  "begin": 0,
  "end": 11,
  "text": "Jan Peeters",
  "label": "Name:Patient",
  "subannotations": [
    {"begin": 0, "end": 3, "text": "Jan", "category": "given"},
    {"begin": 3, "end": 4, "text": " ", "category": "formatting"},
    {"begin": 4, "end": 11, "text": "Peeters", "category": "family"}
  ]
}
```

A non-empty list must be a complete, contiguous partition of its parent span. All offsets remain absolute document offsets.

## Metadata

`metadata` is an optional object. Import tools preserve source columns there, and inference may use explicitly trusted known values during post-processing. Metadata is not part of the neural model input.

Never place a reversible source identifier or project secret into public canonical metadata. Keep those values in the project's protected private mapping.

## Validate and normalize

```bash
meddeid-normalize-jsonl input.jsonl normalized.jsonl
```

Applications validate records at every handoff. Do not work around validation by editing offsets or hashes manually; correct the producing step and regenerate the derived artifact.
