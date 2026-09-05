# Contributing

Thank you for helping improve MedDeID.

## Before opening a change

- Use an issue for substantial behavior or contract changes.
- Do not include patient text, credentials, private infrastructure, or
  restricted datasets.
- Preserve the canonical MedDeID schema, Unicode code-point offsets, immutable
  artifact revisions, and component dependency boundaries.
- Keep changes focused and update the authoritative documentation and
  changelog when user-visible behavior changes.

## Development workflow

Create a branch, install the development dependencies described in
[README.md](README.md), run the repository's complete test/build commands, and
open a pull request. CI must pass before merge. Security reports belong in the
private Security reporting flow described in [SECURITY.md](SECURITY.md), not in
a public issue.

The suite-wide architecture, data contract, privacy boundary, and compatibility
matrix are documented at
<https://stighellemans.github.io/meddeid/>.

## Documentation changes

Improve information at its authoritative home, then link to it elsewhere.

| Change | Authoritative location |
|---|---|
| Choosing components or a cross-suite workflow | `site/docs/` in this repository |
| Package API, CLI flag, configuration, or behavior | The relevant component repository |
| Canonical record or taxonomy rule | `meddeid-core` |
| Language-profile or locale-resource behavior | The relevant `meddeid-language-*` repository |
| Model or dataset contents, limitations, hashes, or licence | Its artifact card |

When writing documentation:

- Begin with what the reader wants to do.
- Tell them which MedDeID tool to use and what to do next.
- Use short sentences and familiar words.
- Write commands and data-field names exactly as they appear in the software.
- Put technical background in Concepts or Reference, then link to it.
- Place privacy warnings beside the step they apply to.
- Link to the source of truth instead of repeating the same information.
- Use only fictional or synthetic examples. Small tests do not prove performance.

Preview the documentation locally:

```bash
cd site
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-docs.txt
python scripts/check_docs.py
mkdocs serve
```

Before opening a pull request, stop the preview server and run the strict site
build from the same `site/` directory:

```bash
python scripts/check_docs.py
mkdocs build --strict
```

CI runs the same checks. A documentation change is complete only when its local
links resolve and the information remains in its authoritative location.
