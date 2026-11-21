# Prepare study notes in Databricks

Start with one fictional note. Once the job can send it to the private service
and use the answer, connect the job to the table where the hospital publishes
its documents. The job will keep the original text and detections inside the
hospital and release the text column approved for each study.

## Try one fictional note

A successful health check tells you the model is ready. Sending a note answers
two further questions. Is this the model the hospital approved? Can the job
use its detections to produce text?

Install the `meddeid-core==0.2.4` package, the Dutch language wheel and
`azure-keyvault-secrets` in the job environment. Put the example Python file
on the job's import path. [Release installation commands](../../reference/compatibility.md#meddeid-043) · [Databricks example](databricks_example.py)

The example makes a repeatable alias for a patient name. For that it needs a
secret **mapping key**. Store at least 32 random bytes as hexadecimal in a
versioned Key Vault secret. Give the Databricks service credential permission
to call the gateway and read that specific secret version. Keep the key on
the trusted driver. Use the private address, audience and CA chain from IT,
and the expected model name and contract hash from the release owner.

```python
from databricks_example import check_integration

credential = dbutils.credentials.getServiceCredentialsProvider("clinical-job")
synthetic_text = check_integration(
    credential=credential,
    gateway_url="https://clinical-model.example.org",
    audience="api://clinical-inference",
    model_name="<approved-model-name>",
    bundle_sha256="<approved-bundle-contract-sha256>",
    vault_url="https://<vault-name>.vault.azure.net",
    secret_name="clinical-mapping",
    secret_version="<immutable-version>",
    verify="/etc/ssl/certs/hospital-ca-bundle.pem",
)
```

The example sends `Patiënt Alex Peters kwam op 03/03/2026.` in a one-note
batch. It checks the returned document ID, the text at each detected position,
the model and bundle, the language profile and the presence of a patient-name
detection. On success, `synthetic_text` holds the text rendered on the driver
using the selected key version. A model or language mismatch is a reason to
check the deployment before connecting the source table.

## Read the revision that can be used for the study

Start the job after the hospital has finished publishing its documents. Read a
fixed Delta snapshot and select the current eligible revision of each note.
Otherwise a clinician's edit during a long run could change which text belongs
to a result.

For each selected note, the hospital's code supplies four things.

| Input | Why the job needs it |
| --- | --- |
| Document ID and revision | A correction can keep the same document ID. The revision identifies which version the job read. |
| Exact converted text | Positions count characters in this string. Even changing a line break can move a detection. |
| Clinical context used for detection | A known patient name can help MedDeID recognize it in the note. Record the context and the version of the code that prepares it, so the job can tell when either has changed. |
| Stable patient ID and original document date | The ID lets names and dates map consistently for a patient. Preserve leading zeros. The date lets the renderer calculate age at the time the note was written. |

The joins that supply these fields belong in the hospital's own code. Other
hospitals can use the same service with their own source tables.

## Save detections while the GPU is running

Send only revisions that need detection. For one note use `POST /deidentify`.
For a batch use `POST /deidentify-batch` with a unique document ID per item.
Keep batches bounded. Check the returned IDs, matched text, language and model
before saving the results. A timeout or transient HTTP 503 can be retried on
the same input. A source or model mismatch needs investigation.

Use `create_annotation_record` to bind each result to the document the job
actually sent. Here `record` is one item from the batch response, and `doc_id`,
`revision` and `text` come from its input row. For a single-note response,
omit the separate `document_id` argument.

```python
from meddeid_core.pipeline import create_annotation_record

annotations = create_annotation_record(
    text,
    record,
    source={"document_id": doc_id, "revision": revision},
    document_id=doc_id,
    expected_model={"name": model_name, "bundle_sha256": bundle_sha256},
)
```

The record keeps the positions and labels. The matched names travel in the
HTTP response and can be recovered from the source note using those
positions. Save the record in restricted Delta storage with the model, bundle,
language and version of the detection inputs and rules. Keep requests,
responses, credentials and date offsets out of job logs.

## Render the study text

Start each later job from the source row selected for the study. Retrieve its
saved annotations only if the model, language, relevant clinical context and
detection rules still match. Then verify the revision and exact text against
that record. This prevents an old name position from masking the wrong word
in a corrected note.

```python
from meddeid_core.pipeline import read_spans
from meddeid_core.rendering import apply_spans

spans = read_spans(
    row.text,
    annotations,
    expected_source={"document_id": row.document_id, "revision": row.revision},
)
profile = annotations["provenance"]["language_profile"]["profile_id"]
masked_text = apply_spans(row.text, spans, profile_id=profile)
timeline_text = apply_spans(row.text, spans, profile_id=profile, keep_original={"Date"})
```

These are the two views in the cookbook example. Both run on CPU. The study
that is allowed to see visit dates can use `timeline_text`.

### Keep names and dates consistent across notes and table columns

A study might need to follow events for the same patient while hiding the
actual dates. Shifting all that patient's dates by the same number of days
preserves the intervals between visits. Using the same alias for a name in a
note and in a structured name field also makes the two sources agree.

On the trusted driver, use the same patient scope, mapping-key version,
language resources and render settings for both. Here `versioned_key` is the
hexadecimal secret read from Key Vault. `offset_days` is the shift the job
can apply to structured event dates. Keep it in restricted storage.

```python
from meddeid_core.rendering import PseudonymizationScope, derive_date_offset, pseudonymize_name

scope = PseudonymizationScope(patient_id=stable_patient_id, secret=versioned_key)
study_text = apply_spans(
    row.text, spans, profile_id=profile, pseudonymization_scope=scope,
    name_replacement=True, date_replacement=True, age_replacement=True,
    document_creation_date=clinical_date,
)
alias = pseudonymize_name(
    patient_name, profile_id=profile, pseudonymization_scope=scope, label="Name:Patient",
)
offset_days = derive_date_offset(scope)
```

A different study policy or key version means rendering again. It can reuse
the detections verified above.

### When the caller needs one text immediately

Both HTTP endpoints also accept a `render` object on each document. For
example, `{"keep_original": ["Date"]}` asks the service to return the timeline
view in `deid_text`. It also returns the original detections. Omitting
`render` selects the server's default policy. The rendering guide covers
both request methods, date overrides, response fields and `prediction.render()`
for Python inference. [Render detected identifiers](../rendering.md)

??? details "Configure the service to return keyed aliases or date shifts"
    When all keyed rendering happens in Databricks, only the driver needs
    the key. If clients need the service to generate the same aliases or date
    shifts, IT must also give the model container the same key version.

    Set `MAPPING_KEY_FILE` to the absolute host path of a file holding the
    pinned hexadecimal key. Give the container user read access. Save this
    alongside the hospital's `compose.yaml` as `compose.override.yaml` so
    Docker Compose loads it with the service configuration.

    ```yaml
    services:
      meddeid:
        environment:
          MEDDEID_PSEUDONYMIZATION_KEY_FILE: /run/secrets/meddeid-mapping
        secrets: [meddeid-mapping]
    secrets:
      meddeid-mapping:
        file: ${MAPPING_KEY_FILE:?select the pinned mapping key file}
    ```

    Requests carry scope identifiers. The key stays in the mounted file.
    Match the scope, language, options and document date on the driver for
    equivalent results. Recreate the service when selecting a new key version.

## Save completed work and release the approved column

A long run may stop after writing half its notes. Write each completed block
with its annotations, derived texts and rendering configuration in one Delta
merge. On retry, the job can recognize completed work from three facts.

| What might change? | What identifies it? |
| --- | --- |
| The source note | `document_id` and `source_revision` |
| What the model found | `detection_revision` covers the model, bundle, language, relevant context and rules used to prepare the model input. |
| What the study can read | `render_revision` covers the rendering policy, algorithm and package versions, and mapping-key version. |

For an append-only result table, use a single writer per selection.
Here `block` is a DataFrame with the target table's columns and one row per
distinct key. All four key fields must have values.

```python
from delta.tables import DeltaTable

fields = ["document_id", "source_revision", "detection_revision", "render_revision"]
match = " AND ".join(f"target.{name} = incoming.{name}" for name in fields)
target = DeltaTable.forName(spark, result_table)
(target.alias("target").merge(block.alias("incoming"), match)
       .whenNotMatchedInsertAll().execute())
```

If the hospital writes results into its source table, make the update
conditional on the source revision still matching. In either design, record
the successful input and output Delta versions and check that every eligible
revision has the expected result. Recheck eligibility at release time. A
withdrawn note still has a historical result. Release only the approved text
column for the eligible documents to the study's authorized researchers.

??? example "One Delta schema for studies with different access permissions"
    One hospital design writes five fields together on the matching document
    revision. The job verifies the matched text in the gateway response before
    saving the positions and labels. It can then generate these study views on
    CPU from the same detections.

    | Field | What the job stores |
    | --- | --- |
    | `meddeid_annotations` | Source revision and text hash, detected positions and labels, model and language provenance, and signatures of the detection inputs and render policy used to check whether results can be reused. |
    | `text_deid_labels` | Labels for all detected identifiers. |
    | `text_deid_pseudonyms` | Scoped name aliases, shifted event dates and labels for organizations. Birth dates can become clinical ages. |
    | `text_clinical_labels` | Original event dates and organizations, with labels for the other detected identifiers. |
    | `text_clinical_pseudonyms` | Original event dates and organizations, with scoped name aliases. Birth dates can become clinical ages. |

## Before the first clinical batch

- **Does it work for our notes?** Review detections on representative local
  note types for missed identifiers and unnecessary redactions. Measure the
  maximum note size the service can handle and its throughput before planning
  a backfill.
- **Can the job finish after an interruption?** Stop a test run after a block
  has committed. Restart it and check that the remaining revisions finish and
  the committed rows appear only once.
- **When can we turn the GPU off?** Re-render a completed selection with the
  GPU offline. For a dedicated GPU, have one scheduler start it when detection
  is pending, wait for readiness and deallocate it after success, failure or
  cancellation. Repeat the fictional-note test after deallocation and restart.

For rollback, restore the previous approved image digests, bundle, wheels
and rendering policy. Keep the dataset versions and mapping-key references
needed to reproduce earlier releases.
