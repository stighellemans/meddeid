# Privacy and security

MedDeID helps detect and remove identifiers. It does not itself establish that an output is anonymous, compliant, or safe to disclose.

## Data boundary

Normal CLI and Python inference run locally. Hugging Face is contacted to download a model unless a local snapshot is supplied; document text is not part of that download request. A public Space or a service deployed outside your institution has a different boundary and must not receive patient data without explicit approval.

## Protect the complete workflow

Sensitive information can appear in more than the source note:

- imported data files;
- annotation assignments and autosaves;
- curation decision logs;
- model predictions and redacted text;
- project metadata, private source-ID maps, and project keys;
- application logs, crash reports, and shell history;
- caches, backups, run records, and evaluation examples.

Apply access control, encryption, retention, backup, and deletion policy to every derived artifact according to its actual content—not its filename.

## Local browser applications

`meddeid-annotate`, `meddeid-curate`, and `meddeid-subannotate` are designed for local operation and do not provide authentication. Bind them to localhost. If an institution deliberately makes one available over a network, place it behind authenticated TLS and apply authorization, audit, session, timeout, and upload controls suitable for clinical data.

## Model risk

Evaluate at least:

- missed identifier characters and spans;
- unnecessary redaction of clinical content;
- performance by document type and source system;
- uncommon names, addresses, identifiers, and formatting;
- effects of OCR, copied text, templates, and encoding;
- changes after model, language-rule, or text-processing updates.

High aggregate performance can conceal a weak subgroup or rare identifier type. Define the risk model before choosing thresholds and review policy.

## Human review

Use human review when a missed identifier could produce material harm, when the target domain differs from evaluation data, or when disclosure decisions require contextual judgment. Reviewers should inspect the complete text rather than only model-highlighted spans.

## Incident readiness

Before production use, define how to:

1. identify the exact model and configuration used for an output;
2. contain a suspected disclosure;
3. trace affected inputs and derived artifacts without broadening access;
4. suspend or roll back a model revision;
5. document corrective validation before resuming use.

Consult institutional privacy, security, ethics, and legal teams for the intended deployment. This documentation is technical guidance, not legal advice.
