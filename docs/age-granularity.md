# Age-granularity policy

MedDeID loads one language-neutral age policy for the complete inference
engine. Language profiles parse and localize ages, but they do not choose their
own privacy boundaries.

The packaged `meddeid-default` policy is equivalent to:

```json
{
  "schema_version": "meddeid.age-granularity.v1",
  "policy_id": "meddeid-default",
  "policy_version": "1",
  "bands": [
    {"until": {"value": 28, "unit": "day", "inclusive": true}, "output": ["day"]},
    {"until": {"value": 90, "unit": "day", "inclusive": true}, "output": ["week", "day"]},
    {"until": {"value": 6, "unit": "month", "inclusive": false}, "output": ["month", "week"]},
    {"until": {"value": 24, "unit": "month", "inclusive": false}, "output": ["month"]},
    {"until": {"value": 12, "unit": "year", "inclusive": false}, "output": ["year", "month"]},
    {"output": ["year"]}
  ]
}
```

Bands are evaluated in order and the final band must be an unbounded catch-all.
Boundary units are `day`, `month`, or `year`; output units may additionally use
`week`. Output units must be unique and ordered from coarse to fine. Years and
months use calendar arithmetic, weeks contain seven days, and any remainder
below the smallest output unit is discarded.

The loader rejects unknown fields, unsupported units, non-positive or
out-of-order boundaries, missing catch-all bands, and invalid JSON before the
model runtime starts. The canonical policy ID, version, and SHA-256 appear in
Python results, batch manifests, `model-info`, and `/health`.

Custom policy files contain configuration only. MedDeID never accepts Python
code, formulas, or a policy supplied by an individual inference request.
