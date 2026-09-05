# Contribute and collaborate

MedDeID is an open-source framework, and broader language support requires collaboration with people who understand local clinical practice, language, data governance, and model development.

## Who we want to work with

| Partner | How you can help |
|---|---|
| Hospitals and care organizations | Define local requirements, validate MedDeID securely on representative text, and identify important failure modes |
| Clinical and NLP researchers | Develop annotation guidelines, evaluation protocols, datasets, language resources, and reproducible studies |
| Language experts and clinical annotators | Develop terminology resources, post-processing rules, and locale-specific handling of names, dates, addresses, and identifiers |
| Open-source and ML engineers | Build language-profile packages, train model bundles, improve tooling, tests, documentation, and deployment support |

Collaboration does not require sharing patient text publicly. Hospitals can run validation locally and share approved aggregate findings, test designs, software improvements, or synthetic examples.

## What adding a language involves

A credible language addition combines language-specific resources, a compatible
model, representative evaluation, and long-term stewardship:

1. **A defined clinical setting.** Agree on the language and region, participating institutions, document types, identifier categories, and intended use.
2. **Adapted annotation guidelines.** Start from the published [Dutch or English annotation guidelines](../artifacts/index.md#annotation-guidelines), adapt the examples and difficult cases, and test them with clinical annotators.
3. **Representative, governed data.** Prepare training and validation data plus a separate final test set. Include different note types, source systems, writing styles, and uncommon identifiers.
4. **A strong, compact base encoder.** Select an encoder with good coverage of the language and clinical vocabulary. It should have a suitable licence and run with acceptable memory use and speed on local institutional hardware. Compare promising encoders before committing to model training.
5. **Language-specific resources.** Package local names, addresses, dates, identifiers, and processing rules in a separate `meddeid-language-*` profile so they can improve without changing the shared tools.
6. **Model training and independent evaluation.** Train the language model and measure both missed identifiers and unnecessary removal of clinical text. Validate it across institutions and document types before making broad performance claims.
7. **Long-term maintainers.** Identify people who can review future changes to the profile, model, guidelines, and evaluation data.

The shared data structure, annotation applications, training workflow, and evaluation tools should remain language-neutral.

??? info "For language-profile developers"
    If the language needs detailed annotation suggestions, its npm package should export a `meddeid.subannotation-profile.v1` module and register each selection in `package.json#meddeid.subannotationProfiles`. The application can then discover it without a language-specific code change.

    Python and JavaScript tools should use the same packaged lookup resources where practical, with a shared record of their origin and version.

## Get in touch

If your hospital, research group, or open-source team wants to evaluate MedDeID or help add a language, contact [stig.hellemans@uantwerpen.be](mailto:stig.hellemans@uantwerpen.be) with a short description of your setting, language, and proposed contribution.

!!! warning "Do not send sensitive data by email"
    Do not attach patient text, identifiers, credentials, or protected project files. Data access and transfer require an agreed governance and security process first.

## Contribute code or documentation

For development setup, documentation guidance, tests, and the pull-request
process, see the [contributor guide](https://github.com/stighellemans/meddeid/blob/main/CONTRIBUTING.md).

Use the [GitHub issue tracker](https://github.com/stighellemans/meddeid/issues)
to propose a change or report a problem. Do not include patient information,
credentials, or restricted project material.

### Keep documentation at its authoritative source

This site explains how components and artifacts fit into suite workflows. Keep
complete API and configuration details in the component repository, and keep
exact files, limitations, licences, provenance, and revisions in the model or
dataset card. Summarize and link from this site instead of copying the complete
text, which can become stale after a component or artifact is updated.

### Coordinate a release

A coordinated suite release records:

1. the immutable Git revision of every component;
2. built wheel and application checksums;
3. schema, taxonomy, and language-profile identities;
4. immutable model and dataset revisions;
5. the documentation revision; and
6. end-to-end verification results.

A patch release may clarify documentation or fix behavior without changing a
contract. An incompatible record, taxonomy, profile, or model-bundle change
requires a new contract or version together with migration guidance. See the
[`meddeid-suite` coordinator](https://github.com/stighellemans/meddeid-suite)
for the release checklist and pinned compatibility record.
