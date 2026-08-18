# Contributing

Thank you for helping improve MedDeID inference.

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
<https://stighellemans.github.io/meddeid.github.io/>.
