# Prepare and annotate data

This workflow turns hospital exports into files that reviewers can inspect and then into a reviewed dataset.

## 1. Create a local project

<span class="source-label">Owner: meddeid-data</span>

```bash
python -m pip install 'meddeid-data[parquet]'

meddeid-data project create my-project notes.parquet \
  --namespace hospital-study \
  --language-profile nl-BE \
  --id-column note_id \
  --text-column note_text
```

The same command accepts CSV, TSV, Parquet, or a directory of UTF-8 `.txt` files. Table columns other than the ID and text columns become metadata by default.

MedDeID replaces source IDs with stable project IDs. The protected `private/` directory stores the project key and the mapping back to the source records.

!!! danger "Keep the private directory private"
    This directory is needed to recognize the same records when data is imported again, and it can link project IDs back to the originals. Back it up and protect it according to institutional policy.

## 2. Create starting annotations with a model

```bash
meddeid batch my-project/artifacts/annotations.jsonl \
  --output my-project/assignments/primary.jsonl
```

Model predictions appear as starting annotations. A reviewer keeps, corrects, removes, or adds them and then marks the document reviewed.

To review without model suggestions, begin with the empty imported records instead.

## 3. Review identifiers

<span class="source-label">Owner: meddeid-annotate</span>

```bash
docker run --rm -p 127.0.0.1:8787:8787 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e MEDDEID_ANNOTATIONS_PATH=/input/primary.jsonl \
  -v "$PWD/my-project/assignments/primary.jsonl:/input/primary.jsonl" \
  ghcr.io/stighellemans/meddeid-annotate:0.1.0
```

Open `http://127.0.0.1:8787`. The versioned image is public and does not
require a source checkout, Node.js, or a registry login.

The application saves changes directly to the assigned file. For every document, inspect the complete text—not just the highlighted identifiers—and save it even when no identifiers are present.

Never point two reviewers at the same writable assignment. Give each reviewer an isolated copy.

## 4. Prepare completed work for the next step

```bash
meddeid-data project package-annotation my-project \
  my-project/assignments/reviewer-a.jsonl \
  --annotation-set-id hospital-study-round-1 \
  --annotator-id reviewer-7
```

This creates a small record of who reviewed the assignment and which exact file version was completed. That record helps later steps use the intended data.

## 5. Curate only when required

<span class="source-label">Owner: meddeid-curate</span>

Use `meddeid-curate` when two or more reviewers worked independently. It keeps their agreements, brings differences to a curator, and records the final decisions.

```text
completed reviewer A + completed reviewer B
  → meddeid-curate
  → annotations.jsonl + decisions.jsonl + manifest.json
```

One completed reviewer can skip this step. Curation is a study-design decision, not a technical requirement imposed by MedDeID.

Start the public curation image and open `http://127.0.0.1:8793`:

```bash
mkdir -p my-project/curation
docker run --rm -p 127.0.0.1:8793:8793 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD/my-project/curation:/app/data" \
  ghcr.io/stighellemans/meddeid-curate:0.1.0
```

## 6. Add detailed labels for evaluation only

<span class="source-label">Owner: meddeid-subannotate</span>

`meddeid-subannotate` marks which characters inside an identifier count as sensitive. These detailed labels help measure whether a model removed the important parts of each identifier.

The public image starts with the language-neutral profile. Point
`ANNOTATIONS_PATH` at a completed reviewer file, or at the curator-approved
export when curation was used:

```bash
ANNOTATIONS_PATH="$PWD/my-project/assignments/reviewer-a.jsonl"
mkdir -p my-project/subannotation
docker run --rm -p 127.0.0.1:8787:8787 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e MEDDEID_ANNOTATIONS_PATH=/input/annotations.jsonl \
  -v "$ANNOTATIONS_PATH:/input/annotations.jsonl:ro" \
  -v "$PWD/my-project/subannotation:/app/data" \
  ghcr.io/stighellemans/meddeid-subannotate:0.1.0
```

??? info "Advanced: choose a language profile"
    The default `neutral@1` profile makes structural suggestions without assuming a language or country. Choose a language profile once per workspace; later commands reuse that selection.

    ```bash
    cd repos/meddeid-subannotate
    npm install --no-save @meddeid/language-nl@0.1.0
    npm run profile -- set nl-BE@1
    npm run dev
    ```

    These commands add the optional public npm capability to a source checkout;
    the GHCR application image remains language-neutral by default.

    The selected profile must support each document's `lang` value. It is saved with the project and evaluation output. To change the profile after work has started, use a separate workspace or run `npm run profile -- migrate <profile>@<version>`; migration archives the previous work and resets review status.

Do not use this step for training data or ordinary prediction review. It is only for detailed evaluation.

## Outputs and owners

| Output | Produced by | Used by |
|---|---|---|
| Imported project data | `meddeid-data` | `meddeid`, `meddeid-annotate` |
| Reviewed annotations | `meddeid-annotate` | `meddeid-data`, optional `meddeid-curate`, training |
| Curator-approved annotations | `meddeid-curate` | `meddeid-subannotate`, evaluation |
| Detailed evaluation data | `meddeid-subannotate` | `meddeid-eval` |

For the technical file-identification and integrity checks used between tools, see [artifact lineage](../concepts/artifact-lineage.md).
