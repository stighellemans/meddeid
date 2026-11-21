# Data contract

<span class="source-label">Authority: meddeid-core</span>

MedDeID tools exchange newline-delimited JSON (`.jsonl`): each line contains
one document. The shared format lets a document move from import to inference,
human review, training, and evaluation without being converted at every step.

You normally do not need to create these records by hand. MedDeID's import and
inference tools write them for you, but understanding the main fields helps
when integrating another data source or system.

## What a document contains

```json
{
  "document_id": "doc_01J...",
  "text": "Jan Peeters kwam op controle.",
  "spans": []
}
```

| Field | Meaning |
|---|---|
| `document_id` | A stable identifier for the document within this dataset version |
| `text` | The exact source text being processed |
| `spans` | The identifiers found in the text, or an empty list |
| `metadata` | Optional information carried with the document |

Once spans have been created, do not change `text`: every span points to exact
character positions in that string. A reviewed document containing no
identifiers still has `spans: []` and an explicit completed state such as
`"annotated": true`, distinguishing it from a document that has not yet been
reviewed.

## How a span points to text

An annotated document adds one entry to `spans` for every detected or reviewed
identifier:

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
  "annotated": true
}
```

Here, `begin` includes the first character and `end` points immediately after
the final character. The interval `[0, 11)` therefore selects `Jan Peeters`.

For every span:

- `span.text` must exactly match `text[begin:end]`;
- positions count Unicode characters, not encoded bytes; and
- `label`, `category`, and `subtype` must agree with the MedDeID taxonomy.

Use the canonical field names shown above. The normalizer described below can
translate a limited set of common alternatives in existing JSONL files.

## Store and reuse detections

With **core 0.2.4rc4**, a pipeline can save the detections from an HTTP
response as a `meddeid.annotations.v1` record. Use this format when the exact
source text already has a home and another job needs to render it later.
The record stores the source identity you supply, the UTF-8 text hash,
code-point length and offset unit, model provenance, and normalized spans.
Each stored span contains offsets, a label and any available score or origin.
The record omits the note, matched strings and replacements.

```python
from meddeid_core.pipeline import create_annotation_record, read_spans

def save_detections(text, response, document_id, revision):
    source = {"document_id": document_id, "revision": revision}
    return create_annotation_record(text, response, source=source, document_id=document_id)

def load_detections(text, record, document_id, revision):
    source = {"document_id": document_id, "revision": revision}
    return read_spans(text, record, expected_source=source)
```

The save example takes one batch response item and the ID submitted with it.
For a single-document response, omit the `document_id` argument to
`create_annotation_record` and keep the document identity in `source`.
Pass `expected_model={"name": ..., "bundle_sha256": ...}` when the job pins a
particular model bundle.

At read time, obtain the revision and text from the selected source document.
`read_spans` raises `ValueError` if the source identity or text has changed.
Rerun detection for the new revision before rendering. Changes to the render
policy alone can reuse verified spans. Keep records and source identifiers
inside the same access boundary as the notes. [Render the verified spans on CPU](../workflows/rendering.md#make-a-second-view-on-cpu)

## What metadata is for

`metadata` is an optional object. Import tools use it to preserve source
columns, and inference can use explicitly trusted values—such as a known
patient or caregiver name—during local post-processing. Metadata is not added
to the neural model input.

Do not place reversible source identifiers, project secrets, or other values
that should remain private in a dataset intended for sharing. Keep them in the
project's protected private mapping.

??? info "Advanced: detailed benchmark subannotations"
    Detailed evaluation can divide a reviewed identifier into smaller
    character segments:

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

    These are absolute positions in the complete document. Together, the
    segments must cover the parent span without gaps or overlaps. Ordinary
    inference, annotation, and training do not require subannotations.

## Standardize supported field names

Use the normalizer when an existing JSONL file already contains one document
per line and keeps identifiers under `spans`, but uses different names for
some fields. Common conversions include:

| Existing field | MedDeID field |
|---|---|
| `doc_id`, `note_id`, or `record_id` | `document_id` |
| `raw_text` or `plain_text` | `text` |
| `start` or `start_char` inside a span | `begin` |
| `surface` or `span_text` inside a span | `text` |
| `tag`, `type`, or `entity_type` inside a span | `label` |
| `confidence` inside a span | `score` |
| `language` inside metadata | `lang` |

Run it once to write a new, standardized file:

```bash
meddeid-normalize-jsonl input.jsonl normalized.jsonl
```

The command leaves the input file unchanged, preserves unrelated top-level
fields, and can derive `category` and `subtype` from a valid `label`.

It is not a general-purpose importer: it does not convert containers such as
`annotations` or `entities` into `spans`, repair incorrect offsets, or decide
which label an identifier should receive. To import CSV, TSV, Parquet, or a
directory of text files, follow [Create a local
project](../workflows/prepare-and-annotate.md#1-import-the-source-notes). If
validation later fails, correct the source or producing step and regenerate the
derived file.

For the exact machine-readable definitions, see the
[`meddeid-core` repository](https://github.com/stighellemans/meddeid-core).
