# Render detected identifiers

In **MedDeID 0.4.3rc5**, one detection run can produce several text variants.
For example, one view can mask every detected identifier while another keeps
event dates for an authorized timeline review. Both use offsets in the same
original note. [Install the candidate packages](../reference/compatibility.md#rendering-release-candidate)

## Request one text from the gateway

Set `MEDDEID_URL` to the gateway’s base URL. For a direct connection,
set `MEDDEID_API_KEY` only if the service uses MedDeID’s built-in key.
Behind the hospital’s HTTPS proxy, initialize the client with the job’s
bearer-token provider. [Hospital client example](hospital/databricks_example.py)

This direct-connection example keeps detected event dates and masks the
other detected identifiers.

```python
import os
from meddeid_core.client import GatewayClient

client = GatewayClient(os.environ["MEDDEID_URL"], api_key=os.environ.get("MEDDEID_API_KEY"))
text = "Patiënt Alex Peters kwam op 03/03/2026."
metadata = {"lang": "nl-BE", "document_creation_date": "2026-03-03"}
result = client.deidentify(text, metadata=metadata, render={"keep_original": ["Date"]})
timeline_text = result["deid_text"]
```

On `POST /deidentify`, `render` sits beside `text` and `metadata`. On
`POST /deidentify-batch`, each item in `documents` has its own `render`
and `document_id`. The client checks the returned IDs and source slices.

```python
[masked_note] = client.deidentify_batch([{
    "document_id": "example-1", "text": text, "metadata": metadata, "render": {}
}])
```

| Request | Text and response fields |
|---|---|
| Omit `render` | The gateway applies its default policy and masks detected dates with labels. The response includes the default replacements, warnings and available processing details. |
| Send `"render": {}` | Each detected span becomes its bracketed label, such as `[Name:Patient]`. |
| Send render options | `deid_text` contains the requested view. As with an empty render object, `spans` contains the original detections and `provenance` identifies the detection run. The gateway omits the default policy's per-span `replacement`, `processing` and `warnings` fields. |

Remove `metadata.date_shift_days` from every request when migrating a client.
The distributed 0.4.3rc4 wheel accepts it. The 0.4.3rc5 gateway
rejects it with HTTP 422. Both support the scope and override examples.
[Patient scope and fixed offset examples](production-deployment.md#shift-event-dates-with-a-patient-scope)

## Make a second view on CPU

Reuse the original string and the spans from the first request. This code
masks every detected identifier, including the dates kept in the timeline
view. It works after the gateway has stopped.

```python
from meddeid_core.rendering import apply_spans

profile = result["provenance"]["language_profile"]["profile_id"]
masked_text = apply_spans(text, result["spans"], profile_id=profile)
```

With the Dutch synthetic baseline used in the quickstart, the two views of
this note are

| View | Output |
|---|---|
| Timeline (`timeline_text`) | `Patiënt [Name:Patient] kwam op 03/03/2026.` |
| All detected identifiers masked (`masked_text`) | `Patiënt [Name:Patient] kwam op [Date].` |

When running inference in Python, the `DeidentificationResult` returned by
`Deidentifier` already holds the source, spans and profile. Use
`prediction.render(keep_original=["Date"])` on that object. For a later job,
save detections with their source revision and verify them when reading them
back. [Store and reuse detections](../concepts/data-contract.md#store-and-reuse-detections)

## Choose a replacement policy

| Option | Effect |
|---|---|
| `categories` | Replace only the listed categories or labels. Other source text stays visible. For example, `["Name:Patient"]` selects patient names only. |
| `keep_original` | Keep the listed categories or labels visible and replace all other detected identifiers. Use one of the two selectors per render. |
| `name_replacement=True` | Generate repeatable name aliases with the language package, mapping key and `pseudonymization_scope`. Use the same settings with `pseudonymize_name` for a structured name field. |
| `date_replacement=True` | Shift recognized event dates. Specify a signed `date_offset_override` in days, or derive the offset from the key and scope. `date_shift_range` sets the minimum and maximum magnitude for derived shifts. |
| `age_replacement=True` | Render recognized `Age_Birthdate` spans independently of event-date replacement. Supply `document_creation_date` as `YYYY-MM-DD` to calculate age at the time of the note. |

Unrecognized name or date values within selected spans fall back to their
bracketed label. The gateway defaults to the detected language profile and the
document date in `metadata`. When rendering locally, pass the document date
explicitly if your policy uses it.

For keyed rendering over HTTP, send the scope identifiers in
`render.pseudonymization_scope`. The operator supplies the hexadecimal key
through `MEDDEID_PSEUDONYMIZATION_KEY_FILE`. For equivalent CPU results, use
the same key version, scope, language resources and render settings. Treat
the selected view according to the identifiers it retains.
[Map structured fields](hospital/databricks-job.md#keep-names-and-dates-consistent-across-notes-and-table-columns) · [Configure keyed rendering on the service](hospital/databricks-job.md#when-the-caller-needs-one-text-immediately)
