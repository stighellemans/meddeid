# Contribute and collaborate

MedDeID is an open-source framework, and broader language support requires collaboration with people who understand local clinical practice, language, data governance, and model development.

## Who we want to work with

| Partner | How you can help |
|---|---|
| Hospitals and care organizations | Define local requirements, validate MedDeID securely on representative text, and identify important failure modes |
| Clinical and NLP researchers | Develop annotation guidelines, evaluation protocols, datasets, language resources, and reproducible studies |
| Language experts and clinical annotators | Adapt terminology, labels, post-processing rules, names, dates, addresses, and other locale-specific behavior |
| Open-source and ML engineers | Build language-profile packages, train model bundles, improve tooling, tests, documentation, and deployment support |

Collaboration does not require sharing patient text publicly. Hospitals can run validation locally and share approved aggregate findings, test designs, software improvements, or synthetic examples.

## What adding a language involves

Adding a language is more than translating labels. A credible language addition needs:

1. **A defined clinical setting.** Agree on the language and region, participating institutions, document types, identifier categories, and intended use.
2. **Adapted annotation guidelines.** Start from the published [MedDeID annotation guidelines](https://doi.org/10.5281/zenodo.21992866), adapt the examples and difficult cases, and test them with clinical annotators.
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

## Documentation contributions

The documentation follows a single-source ownership model. Improve information at its authoritative home, then link to it elsewhere.

| Change | Authoritative location |
|---|---|
| Choosing components or a cross-suite workflow | This documentation site |
| Package API, CLI flag, configuration, or internal behavior | That component repository |
| Canonical record or taxonomy rule | `meddeid-core` |
| Language-profile or locale-resource behavior | The relevant `meddeid-language-*` repository |
| Model or dataset limitations, contents, hashes, licence | Its artifact card |
| Release staging, locks, migration provenance | Suite coordinator workspace |
| Historical UX evidence | Suite pilot report |

### Writing principles

- Start from the user outcome.
- Name the owning component early.
- Use plain language in Start and Workflows. Keep file formats, hashes, offsets, and contract details in Concepts or Reference, and link to them.
- Use canonical field and command names exactly.
- State privacy boundaries next to the action that crosses them.
- Link to authoritative detail instead of copying it.
- Label pilot-scale examples as interface tests, not performance evidence.
- Never include real patient or caregiver information.

### Build the documentation locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-docs.txt
python scripts/check_docs.py
mkdocs serve
```

The continuous-integration build runs the same structural check and a strict MkDocs build. A documentation change is complete only when its local links resolve and its owning source remains clear.
