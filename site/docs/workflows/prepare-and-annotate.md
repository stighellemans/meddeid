# Prepare and annotate data

This workflow turns exported notes into a reviewed MedDeID dataset. Import and
model inference run locally; reviewers use a local browser application to
inspect every document and correct its identifier spans.

```text
source notes → imported JSONL → optional model suggestions → human review
             → optional curation → optional detailed evaluation labels
```

## Before you begin

Choose the installation that matches how far you plan to go:

- For an end-to-end research project, install the complete Python toolset once:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install 'meddeid[research]'
  ```

- If you only need to import and review data, install `meddeid-data`. Add
  `meddeid` only when reviewers should start from model suggestions.

The review applications run in Docker and are not installed by `pip`. Docker
is not needed to import data or generate model suggestions.

## 1. Import the source notes

<span class="source-label">Owner: meddeid-data</span>

Install the importer if you did not install the complete research toolset:

```bash
python -m pip install meddeid-data
```

Create a project and import a CSV file:

```bash
meddeid-data project create my-project notes.csv \
  --namespace hospital-study \
  --language-profile nl-BE \
  --id-column note_id \
  --text-column note_text
```

The same command accepts TSV files and directories of UTF-8 `.txt` files.
For Parquet, install `meddeid-data[parquet]` first.

The command writes the annotation-ready data to
`my-project/artifacts/annotations.jsonl`. In a table, columns other than the
selected ID and text columns are kept as metadata by default.

MedDeID replaces source IDs with stable project IDs. The project key and the
mapping back to source records are stored under `my-project/private/`.

!!! danger "Protect the private directory"
    The private mapping can reconnect project IDs to source records. Keep it
    inside the approved data boundary, restrict access, and back it up with the
    project. Do not include it in a dataset release.

??? info "If the column mapping is wrong"
    A failed first import keeps the empty project and its private key. Correct
    the column names and repeat `project create` with the same namespace and
    language profile, or use the recovery command printed by MedDeID:

    ```bash
    meddeid-data project import my-project notes.csv \
      --id-column note_id \
      --text-column note_text
    ```

    Once a project contains imported or reviewed data, `project create` will
    not overwrite it. Use `project import` only when you deliberately intend
    to replace the project's imported dataset.

For more complex column mappings, see the
[`meddeid-data` import documentation](https://github.com/stighellemans/meddeid-data#create-an-annotation-ready-dataset).

## 2. Create the split before review when training

Skip this step when you only need one reviewed dataset.

If the data will be used for training or domain adaptation, assign documents to
development and test roles before reviewers or models can influence that
choice:

```bash
meddeid-data project split my-project \
  --seed 42 \
  --train 0.8 \
  --validation 0.1
```

This creates `train.jsonl`, `validation.jsonl`, and `test.jsonl` under
`my-project/splits/`. The remaining 10% becomes the test set. Keep those
roles fixed: training decisions may use the train and validation files, but
must not use the test answers.

In the following steps, process each split as a separate assignment. The
examples use the complete imported file to keep the basic annotation path
short; replace that input with a split file when preparing training data.

## 3. Choose the reviewer's starting point

A reviewer can begin with model suggestions or with an empty assignment. Both
routes lead to the same annotation interface and require full human review.

### Start with model suggestions

Install inference if needed:

```bash
python -m pip install 'meddeid>=0.2,<0.3'
```

Generate a writable reviewer assignment:

```bash
meddeid batch my-project/artifacts/annotations.jsonl \
  --output my-project/assignments/reviewer-a.jsonl \
  --model stighellemans/meddeid-dutch-synth
```

The example model is a Dutch synthetic baseline. Select a model and language
profile that match the notes, and prefer an institution-validated bundle when
one is available. Model spans are only a starting point: the reviewer must
keep, correct, remove, or add identifiers after reading the complete document.

### Start without model suggestions

Copy the empty imported records to a writable assignment:

```bash
cp my-project/artifacts/annotations.jsonl \
  my-project/assignments/reviewer-a.jsonl
```

Do not open the canonical file under `artifacts/` as the writable assignment.
Keeping a separate copy preserves the imported source state.

For a split project, repeat either route with `splits/train.jsonl`,
`splits/validation.jsonl`, and `splits/test.jsonl`, using a distinct output
name for each assignment.

## 4. Review every document

<span class="source-label">Owner: meddeid-annotate</span>

Start the local annotation application:

```bash
docker run --rm -p 127.0.0.1:8787:8787 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e MEDDEID_ANNOTATIONS_PATH=/input/reviewer-a.jsonl \
  -v "$PWD/my-project/assignments/reviewer-a.jsonl:/input/reviewer-a.jsonl" \
  ghcr.io/stighellemans/meddeid-annotate:0.1.0
```

Open `http://127.0.0.1:8787`. The application saves directly to
`reviewer-a.jsonl`.

Read the entire document, not only the highlighted text. Correct the spans and
save every document, including documents that contain no identifiers. Saving
marks the document as reviewed.

Give every reviewer a separate assignment file. If two reviewers need the same
starting point, make both copies before either reviewer begins; never mount the
same writable file in two application instances.

## 5. Package a completed assignment

<span class="source-label">Owner: meddeid-data</span>

When the assignment will be curated or handed off as a reproducible result,
validate it and create its manifest:

```bash
meddeid-data project package-annotation my-project \
  my-project/assignments/reviewer-a.jsonl \
  --annotation-set-id hospital-study-round-1 \
  --annotator-id reviewer-7
```

The command checks that every document was reviewed and writes a manifest next
to the JSONL file. The manifest identifies the annotation set and records the
exact file checksum and data contracts. Use a pseudonymous annotator ID rather
than a name or email address.

## 6. Reconcile multiple reviewers only when required

<span class="source-label">Owner: meddeid-curate</span>

Skip curation when one completed reviewer is authoritative for the project.
When two or more people reviewed independent copies, use `meddeid-curate` to
compare their spans and record the curator's decisions.

```bash
mkdir -p my-project/curation
docker run --rm -p 127.0.0.1:8793:8793 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD/my-project/curation:/app/data" \
  ghcr.io/stighellemans/meddeid-curate:0.1.0
```

Open `http://127.0.0.1:8793`, select each completed reviewer JSONL together
with its manifest, and resolve the differences. Publishing the result writes
`annotations.jsonl`, `decisions.jsonl`, and `manifest.json` under
`my-project/curation/exports/`.

## 7. Add detailed labels only for detailed evaluation

<span class="source-label">Owner: meddeid-subannotate</span>

Ordinary training and span-level evaluation do not require this step.
`meddeid-subannotate` is for benchmarks that need to measure which individual
characters inside an identifier were successfully removed.

Choose the completed reviewer file, or the curator-approved
`curation/exports/annotations.jsonl` when curation was used:

```bash
ANNOTATIONS_PATH="$PWD/my-project/assignments/reviewer-a.jsonl"

mkdir -p my-project/subannotation
docker run --rm -p 127.0.0.1:8791:8787 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e MEDDEID_ANNOTATIONS_PATH=/input/annotations.jsonl \
  -v "$ANNOTATIONS_PATH:/input/annotations.jsonl:ro" \
  -v "$PWD/my-project/subannotation:/app/data" \
  ghcr.io/stighellemans/meddeid-subannotate:0.1.0
```

Open `http://127.0.0.1:8791`. The public image uses a language-neutral
profile. Language-specific suggestion profiles are optional; their setup is
documented in the
[`meddeid-subannotate` repository](https://github.com/stighellemans/meddeid-subannotate).

## What to use next

| Your next goal | Input to keep |
|---|---|
| Train a model | Completed development assignments, plus the fixed project split |
| Evaluate a model | Completed test gold, optionally with detailed subannotations |
| Curate reviewers | Every completed reviewer JSONL together with its manifest |
| Preserve or hand off the dataset | The authoritative JSONL and its manifest; keep private mappings separate |

Continue with [Train and evaluate](train-and-evaluate.md) when the development
and test assignments are complete. See [Artifact lineage](../concepts/artifact-lineage.md)
for how manifests connect the workflow outputs.
