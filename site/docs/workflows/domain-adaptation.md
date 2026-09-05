# Adapt a model to your setting

Domain adaptation continues training with reviewed notes from the setting where
the model will be used—for example, another hospital, speciality, document
type, or writing style.

The aim is not simply to produce a new model. It is to make a fair comparison:
does the adapted model perform better than the unchanged starting model on the
same independent local test set?

## Decide whether adaptation is warranted

Evaluate the existing model on representative local notes first. Adaptation is
worth considering when that evaluation reveals a meaningful and consistent
gap, and when enough governed data and review capacity are available to test an
improvement credibly.

A small sample can reveal integration problems and obvious failure patterns,
but it cannot establish that adaptation is effective across the intended
clinical setting.

## Give every document one fixed role

Split the target-setting notes before annotation, model suggestions, or
training. Every document belongs to either development or test:

| Data | What it is used for | What it must not influence |
|---|---|---|
| Development data | Annotation decisions, training, hyperparameters, and training duration | The final reported test result |
| Test data | One final comparison of the baseline and adapted model | Training, epoch selection, or exclusion decisions |

Within development data, a temporary validation subset can help select the
training duration. After that choice, the stricter refit route recombines all
development data and restarts training from the original model.

## Follow the comparison in five stages

### 1. Fix the protocol

Record the intended setting, document types, annotation rules, split method,
primary metrics, and baseline model before looking at comparative results. Pin
the baseline model to an immutable revision.

Use [Prepare and annotate data](prepare-and-annotate.md) to import the notes,
create the project split, and produce independent reviewed assignments.

### 2. Save the unchanged baseline

Run the pinned baseline model on the test documents and retain the predictions
and manifest unchanged. These results represent performance before adaptation.

Do not show baseline predictions to test annotators unless model-assisted test
review is an explicit part of the protocol. That choice can influence the gold
annotations and must be reported.

### 3. Review development and test data

Development annotations provide the examples used for adaptation. Test
annotations provide the answers used only for the final comparison.

Multiple reviewers and curation are optional study-design choices. Detailed
character-level subannotations are needed only when the planned evaluation
includes core-PII or other character-level metrics; add them to the test gold,
not to the training data.

### 4. Train the adapted model

Use [Train and evaluate](train-and-evaluate.md) to prepare the training views
and choose one of two routes:

- an ordinary fit for exploratory work; or
- epoch selection followed by a clean full-development refit for a study or
  release result.

Both the baseline and adapted run must start from the same pinned model
revision. The refit stage starts again from that model; it does not continue
from the epoch-selection checkpoint.

### 5. Compare on the same test set

Run the exported adapted bundle on the exact test documents used for the
baseline. Score both prediction files with the same gold file, metric
configuration, and `meddeid-eval` version.

Report both missed identifiers and unnecessary removal of clinical text.
Exact-span F1 alone does not describe privacy risk or the clinical usefulness
of the remaining text.

## Record these decisions before starting

| Decision | Why it matters |
|---|---|
| Intended clinical setting and document types | Defines where the conclusion is meant to apply |
| Development/test split method and seed | Shows that the comparison groups were fixed in advance |
| Annotation labels and reviewer protocol | Explains how the gold data was created |
| Whether reviewers see model suggestions | Identifies a possible source of review bias |
| Baseline model and immutable revision | Makes the before-adaptation result reproducible |
| Training route and stopping rule | Prevents the test result from choosing the training duration |
| Primary and secondary metrics | Prevents selecting only favorable results afterwards |
| Exclusion rules | Makes removed documents or spans visible |
| Governance and storage boundary | Defines how source text and derived artifacts are protected |

Keep the role assignment, reviewed-data manifests, baseline predictions,
training configuration, adapted model bundle, final predictions, and score
artifacts together. See [Artifact lineage](../concepts/artifact-lineage.md) for
the complete handoff record.
