# Components

Most users only need `meddeid` to de-identify text. Add another component
when you need to prepare data, review annotations, train a model, or evaluate a
system.

Click a component name to open its GitHub repository. That repository contains
the detailed installation instructions, configuration, release notes, and
development documentation for the component.

## Run de-identification

<div class="component-grid" markdown>

<div class="component-card" markdown>
### [`meddeid`](https://github.com/stighellemans/meddeid)

Use this for ordinary de-identification: process one note or a batch through the
CLI or Python API, or run the optional HTTP service.

**PyPI:** `pip install meddeid`

</div>

</div>

## Prepare, review, train, and evaluate

These components correspond to optional workflow steps. Use only the ones your
project needs.

<div class="component-grid" markdown>

<div class="component-card" markdown>
### [`meddeid-data`](https://github.com/stighellemans/meddeid-data)

Use this to import source files into a MedDeID project, create stable document
identifiers and splits, validate data, or generate synthetic data.

**PyPI:** `pip install 'meddeid-data[parquet]'`

</div>

<div class="component-card" markdown>
### [`meddeid-annotate`](https://github.com/stighellemans/meddeid-annotate)

Use this local browser application to review identifiers in an assigned JSONL
file and save completed primary annotations.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-annotate:0.1.0`

</div>

<div class="component-card" markdown>
### [`meddeid-curate`](https://github.com/stighellemans/meddeid-curate)

Use this only when multiple reviewers annotated the same documents and a
curator needs to resolve their differences.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-curate:0.1.0`

</div>

<div class="component-card" markdown>
### [`meddeid-subannotate`](https://github.com/stighellemans/meddeid-subannotate)

Use this only when a benchmark needs detailed character-level labels inside
reviewed identifiers. Ordinary annotation and training do not require it.

**GHCR:** `docker pull ghcr.io/stighellemans/meddeid-subannotate:0.1.0`

</div>

<div class="component-card" markdown>
### [`meddeid-training`](https://github.com/stighellemans/meddeid-training)

Use this to train or adapt a model from reviewed data, select and refit a
checkpoint, and export a model bundle for inference.

**PyPI:** `pip install 'meddeid-training[train,plots]'`

</div>

<div class="component-card" markdown>
### [`meddeid-eval`](https://github.com/stighellemans/meddeid-eval)

Use this to score predictions against reviewed test data, compare systems, and
create performance and stability analyses.

**PyPI:** `pip install 'meddeid-eval[plots]'`

</div>

</div>

## Shared foundations

You normally do not choose these packages directly: `meddeid` and the workflow
tools install the capabilities they need. They are most relevant when you are
integrating with the data contract, developing language behavior, or
contributing to the suite.

<div class="component-grid" markdown>

<div class="component-card" markdown>
### [`meddeid-core`](https://github.com/stighellemans/meddeid-core)

Defines the shared record schema, label taxonomy, normalization, stable
identifiers, and offset validation used by all components.

**PyPI:** `pip install meddeid-core`

</div>

<div class="component-card" markdown>
### [`meddeid-language-nl`](https://github.com/stighellemans/meddeid-language-nl)

Provides Dutch parsing, rendering, resources, and post-processing for `nl-BE`
and `nl-NL`.

**PyPI:** `pip install meddeid-language-nl`

**npm:** `npm install @meddeid/language-nl@0.1.0`

</div>

<div class="component-card" markdown>
### [`meddeid-language-en`](https://github.com/stighellemans/meddeid-language-en)

Provides English parsing, rendering, resources, and locale-specific
post-processing for `en-GB` and `en-US`.

**PyPI:** `pip install meddeid-language-en`

</div>

</div>

## Other useful destinations

These are not additional components to install. They are places you may need
while trying MedDeID, selecting an artifact, or contributing to a coordinated
release.

| Destination                                                                         | Use it when you want to…                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [MedDeID documentation](https://stighellemans.github.io/meddeid/)                   | Choose a workflow and understand how components work together             |
| [Non-clinical demo](https://huggingface.co/spaces/stighellemans/meddeid-demo)       | Try the interface with synthetic or otherwise non-sensitive text          |
| [Hugging Face collection](https://huggingface.co/collections/stighellemans/meddeid) | Find published models, datasets, benchmark data, and their artifact cards |
| [`meddeid-suite`](https://github.com/stighellemans/meddeid-suite)                   | Coordinate versions and verify a release across all components            |

The suite's `publication/` and `internal/` directories are maintainer records,
not reader-facing products. External comparison systems are also not MedDeID
components; run them independently and convert their predictions before using
`meddeid-eval`.

## Documentation ownership

This site helps you choose a component and follow workflows that cross
repository boundaries. Component repositories are authoritative for their API,
CLI options, configuration, tests, and release notes. Hugging Face artifact
cards are authoritative for model and dataset contents, limitations, licences,
and immutable revisions.
