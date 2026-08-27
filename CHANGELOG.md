# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

## [0.2.0] - 2026-08-27

- Language-package scaffolding no longer emits a separate ruleset version;
  rulesets have stable identities and rely on package and Git provenance.
- Added typed workflow-template validation, an onboarding action-adapter
  registry, and semantic artifact validation for language, corpus, checkpoint,
  bundle, and interface gates.
- Guided workflows now use unversioned locale identities and reject selectors
  such as `en-GB@1`; independent artifact versions remain in their manifests.
- Expanded language scaffolding and conformance to cover Python/JavaScript
  registration, per-profile manifests, source locks, resource commands, and
  wheel/sdist/npm package checks.
- Added suite-wide declarative age granularity, deterministic date shifting,
  safe placeholder defaults, structured warnings/provenance, trusted birth-date
  recovery, and consistent replacements across Python, CLI, batch, and HTTP.
- HTTP and single-file CLI JSON no longer echo the complete original note;
  span source fragments and canonical research JSONL remain unchanged for
  validation and evaluation.
- Added the `meddeid guide`, `meddeid doctor`, and `meddeid workflow` front
  door with explicit scientific branches, just-in-time operational choices,
  checksummed resumable manifests, safe invalidation/archiving, detached jobs,
  component-command dry runs, and ten prevalent suite templates.
- Added the simplified `meddeid start`, `meddeid status`, and `meddeid next`
  experience, grouping those templates under six common goals with nested
  choices, concise progress, numbered answers, and `?` explanations.
- Added `meddeid[research]` and `meddeid[contributor]` installation extras.
- Added installed discovery support for separate `en-GB` and `en-US`
  language profiles and removed the Dutch-only source-tree fallback.
- Unified language-profile selection and post-processing provenance across
  Python, single and batch CLI, HTTP, browser, and guided deployment paths;
  multi-profile services now support a locale fallback while bundles remain
  authoritative for locale behavior.

## [0.1.1] - 2026-08-17

- Published the first externally supported MedDeID inference release.
- Added public installation, compatibility, licensing, and verification
  metadata.
- Established independent CI and immutable release artifacts.

For earlier migration history, consult the repository's Git history.
