"""Declarative, versioned workflow templates for the guided MedDeID CLI.

The templates deliberately describe *why* a stage exists separately from the
command that implements it.  Scientific branches use two predicates:

``applicable_when``
    Whether the stage belongs to the selected study design at all.
``enabled_when``
    Whether the user selected an optional stage that is applicable.

This distinction is what lets status report ``not_applicable`` rather than
silently treating every omitted stage as a user choice.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .workflow_contracts import validate_template

WORKFLOW_CONTRACT = "meddeid.workflow.v1"
TEMPLATE_VERSION = "1"


def _decision(
    key: str,
    prompt: str,
    *,
    kind: str = "string",
    choices: tuple[str, ...] = (),
    choice_labels: dict[str, str] | None = None,
    scope: str = "protocol",
    default: Any = None,
    required: bool = True,
    minimum: int | None = None,
    path_role: str = "input",
    why: str,
    ask_when: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "prompt": prompt,
        "kind": kind,
        "choices": list(choices),
        "choice_labels": dict(choice_labels or {}),
        "scope": scope,
        "default": default,
        "required": required,
        "minimum": minimum,
        "path_role": path_role,
        "why": why,
        "ask_when": ask_when,
    }


def _stage(
    stage_id: str,
    title: str,
    why: str,
    *,
    requires: tuple[str, ...] = (),
    decisions: tuple[str, ...] = (),
    applicable_when: dict[str, Any] | None = None,
    enabled_when: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    outputs: tuple[str, ...] = (),
    expensive: bool = False,
    external: bool = False,
    human: bool = False,
    allows_detach: bool = False,
    simple_label: str | None = None,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "title": title,
        "why": why,
        "requires": list(requires),
        "decisions": list(decisions),
        "applicable_when": applicable_when,
        "enabled_when": enabled_when,
        "action": action or {"kind": "internal", "name": "record-stage"},
        "outputs": list(outputs),
        "expensive": expensive,
        "external": external,
        "human": human,
        "allows_detach": allows_detach,
        "simple_label": simple_label,
    }


def _command(
    *argv: str,
    tools: tuple[str, ...] = (),
    options: tuple[dict[str, Any], ...] = (),
    env: dict[str, str] | None = None,
    env_options: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    """Describe a component command plus decision-controlled CLI options."""

    return {
        "kind": "command",
        "argv": list(argv),
        "tools": list(tools),
        "options": list(options),
        "env": dict(env or {}),
        "env_options": list(env_options),
    }


def _option(decision: str, flag: str, *, boolean: bool = False) -> dict[str, Any]:
    return {"decision": decision, "flag": flag, "boolean": boolean}


def _env_option(decision: str, name: str) -> dict[str, str]:
    return {"decision": decision, "name": name}


def _internal(name: str, *, tools: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"kind": "internal", "name": name, "tools": list(tools)}


def _browser(app: str, *, source: str, data_dir: str | None = None) -> dict[str, Any]:
    return {
        "kind": "browser",
        "app": app,
        "source": source,
        "data_dir": data_dir,
        "tools": ["docker", "npm"],
    }


def _eq(key: str, value: Any) -> dict[str, Any]:
    return {"decision": key, "eq": value}


def _gt(key: str, value: int) -> dict[str, Any]:
    return {"decision": key, "gt": value}


def _truthy(key: str) -> dict[str, Any]:
    return {"decision": key, "truthy": True}


def _all(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"all": list(conditions)}


def _any(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"any": list(conditions)}


def _condition_decisions(condition: dict[str, Any] | None) -> set[str]:
    if not condition:
        return set()
    if "decision" in condition:
        return {str(condition["decision"])}
    values: set[str] = set()
    for key in ("all", "any"):
        for child in condition.get(key, []):
            values.update(_condition_decisions(child))
    if "not" in condition:
        values.update(_condition_decisions(condition["not"]))
    return values


COMMON_RUNTIME_DECISIONS = [
    _decision(
        "device",
        "Execution device (auto, cpu, mps, or cuda)",
        kind="choice",
        choices=("auto", "cpu", "mps", "cuda"),
        scope="operational",
        why="Hardware selection affects compatibility, runtime, and reproducibility.",
    ),
    _decision(
        "runtime",
        "Browser runtime (docker or source)",
        kind="choice",
        choices=("docker", "source"),
        scope="operational",
        why="Browser applications can run from a pinned container or a development checkout.",
    ),
    _decision(
        "suite_root",
        "Suite checkout containing repos/ (source runtime only)",
        kind="path",
        scope="operational",
        required=False,
        why="Source mode needs the local application repositories; released Docker mode does not.",
        ask_when=_eq("runtime", "source"),
    ),
]


def _annotation_decisions(
    *, ask_when: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    review_applies = ask_when
    multi_reviewer = _gt("reviewer_count", 1)
    gold_applies = _all(ask_when, multi_reviewer) if ask_when else multi_reviewer
    selected_applies = (
        _all(ask_when, _eq("gold_policy", "selected_reviewer"))
        if ask_when
        else _eq("gold_policy", "selected_reviewer")
    )
    return [
        _decision(
            "review_mode",
            "Should reviewers see model suggestions?",
            kind="choice",
            choices=("assisted", "blinded"),
            choice_labels={
                "assisted": "Yes — assisted review with model suggestions",
                "blinded": "No — blinded manual review",
            },
            why="Showing model suggestions can speed review but changes the annotation protocol.",
            ask_when=review_applies,
        ),
        _decision(
            "reviewer_count",
            "Number of independent primary reviewers",
            kind="int",
            why="Independent review count determines whether reconciliation can apply.",
            ask_when=review_applies,
        ),
        _decision(
            "gold_policy",
            "How should several independent reviews become authoritative gold?",
            kind="choice",
            choices=("adjudicate", "selected_reviewer"),
            choice_labels={
                "adjudicate": "Reconcile disagreements in the curation viewer",
                "selected_reviewer": "Select one review as authoritative",
            },
            why="Several independent submissions cannot be silently collapsed.",
            ask_when=gold_applies,
        ),
        _decision(
            "selected_reviewer",
            "Reviewer identifier to use as authoritative gold",
            why="Selecting one reviewer must identify the retained submission.",
            ask_when=selected_applies,
        ),
        _decision(
            "selection_rationale",
            "Why is this reviewer authoritative?",
            why="Discarding other independent submissions requires an auditable rationale.",
            ask_when=selected_applies,
        ),
    ]


BENCHMARK_PRIMARY_REQUIRED = _any(
    _eq("input_role", "unlabelled"),
    _eq("input_role", "sealed_test"),
    _eq("re_review", True),
)


TEMPLATES: dict[str, dict[str, Any]] = {
    "inference": {
        "title": "Local inference",
        "summary": "De-identify one note, a canonical batch, or start the local service.",
        "decisions": [
            _decision(
                "inference_mode",
                "What do you want to run?",
                kind="choice",
                choices=("single", "batch", "service"),
                choice_labels={
                    "single": "De-identify one text file",
                    "batch": "De-identify a canonical JSONL batch",
                    "service": "Start the local service",
                },
                why="The input and operational safeguards differ for files, batches, and services.",
            ),
            _decision(
                "source",
                "Input file",
                kind="path",
                why="Single and batch inference require an explicit local input.",
                ask_when=_any(
                    _eq("inference_mode", "single"), _eq("inference_mode", "batch")
                ),
            ),
            _decision(
                "model",
                "Model bundle or Hub identifier",
                default="stighellemans/meddeid-dutch-synth",
                why="Reproducible inference records the exact model identity.",
            ),
            _decision(
                "revision",
                "Immutable model revision (optional)",
                required=False,
                why="A fixed revision prevents an upstream model update from changing results.",
            ),
            _decision(
                "language_profile",
                "Language profile for an ambiguous multi-profile model (optional)",
                required=False,
                why="MedDeID must not guess between locale-sensitive profiles.",
            ),
            _decision(
                "offline",
                "Require offline/local-only model loading?",
                kind="bool",
                default=False,
                why="Offline mode prevents runtime Hub access.",
            ),
            COMMON_RUNTIME_DECISIONS[0],
        ],
        "stages": [
            _stage(
                "inspect_input",
                "Inspect input",
                "Fail early when the selected input is unavailable or malformed.",
                decisions=("source",),
                applicable_when=_any(
                    _eq("inference_mode", "single"), _eq("inference_mode", "batch")
                ),
                action=_internal("inspect-input"),
            ),
            _stage(
                "single_inference",
                "De-identify one note",
                "Produce a local de-identified note and provenance record.",
                requires=("inspect_input",),
                decisions=("source", "model", "device"),
                applicable_when=_eq("inference_mode", "single"),
                action=_command(
                    "meddeid",
                    "deidentify",
                    "{source}",
                    "--output",
                    "{workspace}/artifacts/deidentified.txt",
                    "--model",
                    "{model}",
                    "--device",
                    "{device}",
                    tools=("meddeid",),
                    options=(
                        _option("revision", "--revision"),
                        _option("language_profile", "--language-profile"),
                        _option("offline", "--offline", boolean=True),
                    ),
                ),
                outputs=("artifacts/deidentified.txt",),
            ),
            _stage(
                "batch_inference",
                "Run batch inference",
                "Preserve document order and record model/timing lineage.",
                requires=("inspect_input",),
                decisions=("source", "model", "device"),
                applicable_when=_eq("inference_mode", "batch"),
                action=_command(
                    "meddeid",
                    "batch",
                    "{source}",
                    "--output",
                    "{workspace}/artifacts/predictions.jsonl",
                    "--model",
                    "{model}",
                    "--device",
                    "{device}",
                    tools=("meddeid",),
                    options=(
                        _option("revision", "--revision"),
                        _option("language_profile", "--language-profile"),
                        _option("offline", "--offline", boolean=True),
                    ),
                ),
                outputs=(
                    "artifacts/predictions.jsonl",
                    "artifacts/predictions.jsonl.manifest.json",
                ),
                expensive=True,
                allows_detach=True,
            ),
            _stage(
                "local_service",
                "Start local service",
                "Expose the hardened localhost service for interactive or internal clients.",
                decisions=("model", "device"),
                applicable_when=_eq("inference_mode", "service"),
                action=_command(
                    "meddeid-server",
                    tools=("meddeid-server",),
                    env={
                        "MEDDEID_MODEL": "{model}",
                        "MEDDEID_DEVICE": "{device}",
                        "MEDDEID_OFFLINE": "{offline}",
                    },
                    env_options=(
                        _env_option("revision", "MEDDEID_REVISION"),
                        _env_option("language_profile", "MEDDEID_LANGUAGE_PROFILE"),
                    ),
                ),
                human=True,
            ),
        ],
    },
    "dataset-review": {
        "title": "Dataset preparation and review",
        "summary": "Import source notes, prepare independent assignments, review them, and create authoritative annotations.",
        "decisions": [
            _decision(
                "source",
                "TXT directory, CSV, TSV, or Parquet source",
                kind="path",
                why="Project import needs an explicit source boundary.",
            ),
            _decision(
                "namespace",
                "Pseudonymous project namespace",
                why="Stable IDs must be scoped without exposing hospital source identifiers.",
            ),
            _decision(
                "language_profile",
                "Language profile",
                why="Import and later tools must use an explicit locale contract.",
            ),
            _decision(
                "create_split",
                "Create deterministic train/validation/test splits?",
                kind="bool",
                why="Split roles must be fixed before model development.",
            ),
            _decision(
                "split_seed",
                "Split seed",
                kind="int",
                default=20260508,
                why="The same seed reproduces the project split.",
            ),
            *_annotation_decisions(),
            *COMMON_RUNTIME_DECISIONS,
        ],
        "stages": [
            _stage(
                "import",
                "Import project",
                "Normalize source records and create protected stable document identifiers.",
                decisions=("source", "namespace", "language_profile"),
                action=_command(
                    "meddeid-data",
                    "project",
                    "create",
                    "{workspace}/project",
                    "{source}",
                    "--namespace",
                    "{namespace}",
                    "--language-profile",
                    "{language_profile}",
                    tools=("meddeid-data",),
                ),
                outputs=("project/project.json", "project/artifacts/annotations.jsonl"),
            ),
            _stage(
                "split",
                "Create fixed splits",
                "Fix development and test roles before downstream review or training.",
                requires=("import",),
                decisions=("split_seed",),
                enabled_when=_eq("create_split", True),
                action=_command(
                    "meddeid-data",
                    "project",
                    "split",
                    "{workspace}/project",
                    "--seed",
                    "{split_seed}",
                    tools=("meddeid-data",),
                ),
                outputs=("project/manifests/splits.json",),
                simple_label="Dataset splitting",
            ),
            _stage(
                "prepare_assignments",
                "Prepare isolated assignments",
                "Never let two reviewers write to the same file.",
                requires=("import", "split"),
                decisions=("reviewer_count", "review_mode"),
                action=_internal("prepare-assignments"),
                outputs=("assignments/assignment-manifest.json",),
            ),
            _stage(
                "preannotate",
                "Generate model suggestions",
                "Create starting spans only when assisted review was selected.",
                requires=("prepare_assignments",),
                decisions=("device",),
                enabled_when=_eq("review_mode", "assisted"),
                action=_internal("preannotate-assignments"),
                outputs=("artifacts/preannotation.json",),
                expensive=True,
                allows_detach=True,
                simple_label="Model suggestions",
            ),
            _stage(
                "primary_review",
                "Review primary spans",
                "A human inspects complete text and confirms, corrects, removes, or adds spans.",
                requires=("preannotate",),
                decisions=("runtime",),
                action=_browser("annotate", source="{next_assignment}"),
                outputs=("assignments",),
                human=True,
            ),
            _stage(
                "curate",
                "Reconcile reviewers",
                "Produce one auditable gold set from independent submissions.",
                requires=("primary_review",),
                decisions=("gold_policy", "runtime"),
                applicable_when=_gt("reviewer_count", 1),
                enabled_when=_eq("gold_policy", "adjudicate"),
                action=_browser(
                    "curate",
                    source="{workspace}/assignments",
                    data_dir="{workspace}/curation",
                ),
                outputs=(
                    "curation/exports/annotations.jsonl",
                    "curation/exports/manifest.json",
                ),
                human=True,
                simple_label="Reviewer reconciliation",
            ),
            _stage(
                "package",
                "Package authoritative annotations",
                "Pin the completed annotation file and its provenance for training or evaluation.",
                requires=("curate",),
                action=_internal("package-authoritative"),
                outputs=(
                    "artifacts/authoritative-annotations.jsonl",
                    "artifacts/authoritative-annotations.manifest.json",
                ),
            ),
        ],
    },
    "benchmark": {
        "title": "Evaluation benchmark creation",
        "summary": "Review primary gold when required, optionally reconcile it, add detailed subannotations, and export a checksummed bundle.",
        "decisions": [
            _decision(
                "source",
                "Source canonical JSONL",
                kind="path",
                why="Benchmark lineage starts from one explicit source artifact.",
            ),
            _decision(
                "input_role",
                "What does the source file contain?",
                kind="choice",
                choices=(
                    "unlabelled",
                    "completed_annotations",
                    "existing_gold",
                    "sealed_test",
                ),
                choice_labels={
                    "unlabelled": "Unlabelled documents that still need review",
                    "completed_annotations": "Completed annotations to accept or re-review",
                    "existing_gold": "Accepted gold annotations",
                    "sealed_test": "A sealed test set requiring controlled review",
                },
                why="The source role determines whether primary review is required or already complete.",
            ),
            _decision(
                "re_review",
                "Re-review existing/completed gold?",
                kind="bool",
                required=False,
                why="Existing gold is accepted or deliberately reopened; it is never reset implicitly.",
                ask_when=_any(
                    _eq("input_role", "existing_gold"),
                    _eq("input_role", "completed_annotations"),
                ),
            ),
            _decision(
                "profiles",
                "Comma-separated language profiles",
                kind="profiles",
                why="Detailed suggestions and export lineage must use each document's explicit locale.",
                ask_when=_eq("detailed_evaluation", True),
            ),
            *_annotation_decisions(ask_when=BENCHMARK_PRIMARY_REQUIRED),
            _decision(
                "detailed_evaluation",
                "Do you need detailed character-level/core-PII metrics?",
                kind="bool",
                why="Detailed core-PII metrics require reviewed character-level categories.",
            ),
            _decision(
                "score_predictions",
                "Score a prediction file?",
                kind="bool",
                why="Scoring is an explicit benchmark outcome.",
            ),
            _decision(
                "predictions",
                "Prediction JSONL to score",
                kind="path",
                why="Scoring must identify one exact prediction artifact.",
                ask_when=_eq("score_predictions", True),
            ),
            _decision(
                "plots",
                "Render comparison plots?",
                kind="bool",
                default=False,
                why="Plots are optional outputs rather than a hidden side effect.",
                ask_when=_eq("score_predictions", True),
            ),
            *COMMON_RUNTIME_DECISIONS,
        ],
        "stages": [
            _stage(
                "validate_source",
                "Validate source",
                "Validate canonical fields, labels, offsets, completion state, and locale metadata.",
                decisions=("source",),
                action=_command(
                    "meddeid-data", "validate", "{source}", tools=("meddeid-data",)
                ),
            ),
            _stage(
                "prepare_primary",
                "Prepare primary assignments",
                "Copy source gold safely and reset review state only when re-review is explicit.",
                requires=("validate_source",),
                decisions=("reviewer_count", "input_role"),
                applicable_when=BENCHMARK_PRIMARY_REQUIRED,
                action=_internal("prepare-benchmark-assignments"),
                outputs=("assignments/assignment-manifest.json",),
            ),
            _stage(
                "preannotate",
                "Generate model suggestions",
                "Initialize unlabelled assignments only under an assisted-review protocol.",
                requires=("prepare_primary",),
                decisions=("device",),
                applicable_when=BENCHMARK_PRIMARY_REQUIRED,
                enabled_when=_eq("review_mode", "assisted"),
                action=_internal("preannotate-assignments"),
                outputs=("artifacts/preannotation.json",),
                expensive=True,
                allows_detach=True,
                simple_label="Model suggestions",
            ),
            _stage(
                "primary_review",
                "Review primary gold",
                "Require deliberate whole-text human review when gold is new or explicitly reopened.",
                requires=("preannotate",),
                decisions=("runtime",),
                applicable_when=BENCHMARK_PRIMARY_REQUIRED,
                action=_browser("annotate", source="{next_assignment}"),
                outputs=("assignments",),
                human=True,
                simple_label="Primary human review",
            ),
            _stage(
                "curate",
                "Reconcile independent reviews",
                "Do not silently collapse several annotation sets.",
                requires=("primary_review",),
                decisions=("gold_policy", "runtime"),
                applicable_when=_all(
                    BENCHMARK_PRIMARY_REQUIRED, _gt("reviewer_count", 1)
                ),
                enabled_when=_eq("gold_policy", "adjudicate"),
                action=_browser(
                    "curate",
                    source="{workspace}/assignments",
                    data_dir="{workspace}/curation",
                ),
                outputs=(
                    "curation/exports/annotations.jsonl",
                    "curation/exports/manifest.json",
                ),
                human=True,
                simple_label="Reviewer reconciliation",
            ),
            _stage(
                "select_gold",
                "Select authoritative primary gold",
                "Resolve the exact completed source used by detailed evaluation.",
                requires=("curate",),
                action=_internal("package-authoritative"),
                outputs=(
                    "artifacts/authoritative-annotations.jsonl",
                    "artifacts/authoritative-annotations.manifest.json",
                ),
            ),
            _stage(
                "subannotate",
                "Review core-PII subannotations",
                "Mark the sensitive characters required by detailed evaluation metrics.",
                requires=("select_gold",),
                decisions=("profiles", "runtime"),
                enabled_when=_eq("detailed_evaluation", True),
                action=_browser(
                    "subannotate",
                    source="{workspace}/artifacts/authoritative-annotations.jsonl",
                    data_dir="{workspace}/subannotation",
                ),
                outputs=("subannotation/subannotations.jsonl",),
                human=True,
                simple_label="Detailed core-PII subannotation",
            ),
            _stage(
                "export_bundle",
                "Export evaluation bundle",
                "Verify completion and pin primary gold plus profile resources by checksum.",
                requires=("subannotate",),
                enabled_when=_eq("detailed_evaluation", True),
                action=_internal("export-subannotation-bundle"),
                outputs=("subannotation/evaluation-bundle/manifest.json",),
            ),
            _stage(
                "score",
                "Score predictions",
                "Compute metrics only against the explicitly selected gold and predictions.",
                requires=("export_bundle",),
                decisions=("predictions",),
                enabled_when=_eq("score_predictions", True),
                action=_command(
                    "meddeid-eval",
                    "score",
                    "--gold",
                    "{benchmark_gold}",
                    "--predictions",
                    "{predictions}",
                    "--output",
                    "{workspace}/artifacts/metrics.json",
                    tools=("meddeid-eval",),
                ),
                outputs=("artifacts/metrics.json",),
                simple_label="Prediction scoring and plots",
            ),
            _stage(
                "plots",
                "Render comparison plots",
                "Create figures only when they were requested.",
                requires=("score",),
                applicable_when=_eq("score_predictions", True),
                enabled_when=_eq("plots", True),
                action=_command(
                    "meddeid-eval",
                    "plot",
                    "--scores",
                    "{workspace}/artifacts/metrics.json",
                    "--output-dir",
                    "{workspace}/artifacts/plots",
                    tools=("meddeid-eval",),
                ),
                outputs=("artifacts/plots",),
            ),
        ],
    },
}


def _research_templates() -> dict[str, dict[str, Any]]:
    """Return the remaining templates without making the module-level table unreadable."""

    evaluation = {
        "title": "Evaluation and comparison",
        "summary": "Validate gold and predictions, score systems, optionally plot and run stability analysis.",
        "decisions": [
            _decision(
                "gold",
                "Gold JSONL",
                kind="path",
                why="Evaluation needs an explicit immutable reference.",
            ),
            _decision(
                "predictions",
                "Prediction JSONL",
                kind="path",
                why="Scores must be tied to one exact prediction artifact.",
            ),
            _decision(
                "plots",
                "Render plots?",
                kind="bool",
                why="Figure generation is optional.",
            ),
            _decision(
                "stability",
                "Run stability analysis?",
                kind="bool",
                why="Stability is a separate experimental outcome.",
            ),
            _decision(
                "stability_config",
                "Stability YAML configuration",
                kind="path",
                why="Perturbations and systems under test must be configured reproducibly.",
                ask_when=_eq("stability", True),
            ),
        ],
        "stages": [
            _stage(
                "validate",
                "Validate inputs",
                "Reject document, offset, or contract mismatches before scoring.",
                decisions=("gold", "predictions"),
                action=_internal("validate-evaluation-inputs"),
            ),
            _stage(
                "score",
                "Score predictions",
                "Compute exact, character, core-PII, and non-PII-redaction metrics supported by the gold.",
                requires=("validate",),
                action=_command(
                    "meddeid-eval",
                    "score",
                    "--gold",
                    "{gold}",
                    "--predictions",
                    "{predictions}",
                    "--output",
                    "{workspace}/artifacts/metrics.json",
                    tools=("meddeid-eval",),
                ),
                outputs=("artifacts/metrics.json",),
            ),
            _stage(
                "plots",
                "Render plots",
                "Create human-readable comparison figures when requested.",
                requires=("score",),
                enabled_when=_eq("plots", True),
                action=_command(
                    "meddeid-eval",
                    "plot",
                    "--scores",
                    "{workspace}/artifacts/metrics.json",
                    "--output-dir",
                    "{workspace}/artifacts/plots",
                    tools=("meddeid-eval",),
                ),
                outputs=("artifacts/plots",),
                simple_label="Comparison plots",
            ),
            _stage(
                "stability",
                "Run stability analysis",
                "Measure robustness under the configured privacy-safe perturbations.",
                requires=("score",),
                decisions=("stability_config",),
                enabled_when=_eq("stability", True),
                action=_internal("run-stability-analysis", tools=("meddeid-eval",)),
                outputs=("artifacts/stability.json",),
                expensive=True,
                allows_detach=True,
                simple_label="Stability analysis",
            ),
        ],
    }

    training_decisions = [
        _decision(
            "project",
            "Prepared MedDeID project",
            kind="path",
            why="Training views must be traceable to a reviewed project.",
        ),
        _decision(
            "development",
            "Reviewed development JSONL",
            kind="path",
            why="Development data is used for fitting and epoch selection.",
        ),
        _decision(
            "test_gold",
            "Sealed test gold JSONL",
            kind="path",
            why="Test data remains separate until final evaluation.",
        ),
        _decision(
            "config",
            "Training YAML",
            kind="path",
            why="Hyperparameters and profile contracts must be explicit.",
        ),
        _decision(
            "training_protocol",
            "How should training duration be chosen?",
            kind="choice",
            choices=("fit", "select_refit"),
            choice_labels={
                "fit": "Ordinary one-stage fit for exploratory work",
                "select_refit": "Select epochs on development data, then cleanly refit",
            },
            why="Exploratory fit and publication refit have different test-access rules.",
        ),
        COMMON_RUNTIME_DECISIONS[0],
    ]
    training_stages = [
        _stage(
            "prepare",
            "Prepare training views",
            "Create separate fit, selection, and refit views while sealing test answers.",
            decisions=("project", "development", "test_gold"),
            action=_command(
                "meddeid-data",
                "project",
                "prepare-training",
                "{project}",
                "--development",
                "{development}",
                "--test-gold",
                "{test_gold}",
                "--output",
                "{workspace}/prepared",
                tools=("meddeid-data",),
            ),
            outputs=("prepared/manifest.json",),
        ),
        _stage(
            "fit",
            "Run ordinary fit",
            "Perform one train/validation/test experiment for exploratory work.",
            requires=("prepare",),
            decisions=("config",),
            applicable_when=_eq("training_protocol", "fit"),
            action=_command(
                "meddeid-train",
                "fit",
                "--config",
                "{config}",
                "--data",
                "{workspace}/prepared/fit",
                "--run",
                "{workspace}/runs/fit",
                tools=("meddeid-train",),
            ),
            outputs=("runs/fit/train_metrics.json",),
            expensive=True,
            allows_detach=True,
        ),
        _stage(
            "select_epochs",
            "Select epoch count",
            "Choose training duration using development data without reading test answers.",
            requires=("prepare",),
            decisions=("config",),
            applicable_when=_eq("training_protocol", "select_refit"),
            action=_command(
                "meddeid-train",
                "select-epochs",
                "--config",
                "{config}",
                "--data",
                "{workspace}/prepared/selection",
                "--run",
                "{workspace}/runs/selection",
                "--selection-output",
                "{workspace}/runs/selection.json",
                tools=("meddeid-train",),
            ),
            outputs=("runs/selection.json",),
            expensive=True,
            allows_detach=True,
            simple_label="Epoch selection and clean refit",
        ),
        _stage(
            "refit",
            "Refit from the initial model",
            "Train on all development data for the selected duration and evaluate the sealed test once.",
            requires=("select_epochs",),
            decisions=("config",),
            applicable_when=_eq("training_protocol", "select_refit"),
            action=_command(
                "meddeid-train",
                "refit",
                "--config",
                "{config}",
                "--selection",
                "{workspace}/runs/selection.json",
                "--data",
                "{workspace}/prepared/refit",
                "--run",
                "{workspace}/runs/refit",
                tools=("meddeid-train",),
            ),
            outputs=("runs/refit/train_metrics.json",),
            expensive=True,
            allows_detach=True,
        ),
        _stage(
            "export",
            "Export model bundle",
            "Create the self-contained artifact required by inference.",
            requires=("fit", "refit"),
            action=_internal("export-trained-model"),
            outputs=("artifacts/model/bundle.json",),
        ),
        _stage(
            "smoke",
            "Smoke-test exported model",
            "Prove that the exported bundle loads and produces canonical predictions.",
            requires=("export",),
            decisions=("device",),
            action=_internal("smoke-model-bundle"),
            outputs=("artifacts/model-smoke.json",),
            expensive=True,
        ),
        _stage(
            "score",
            "Score exported model",
            "Evaluate the exact exported bundle, not an in-memory training checkpoint.",
            requires=("smoke",),
            action=_internal("score-exported-model"),
            outputs=("artifacts/model-metrics.json",),
            expensive=True,
            allows_detach=True,
        ),
    ]
    training = {
        "title": "Training and model export",
        "summary": "Prepare reviewed data, fit safely, export a bundle, and verify it.",
        "decisions": training_decisions,
        "stages": training_stages,
    }

    adaptation = {
        "title": "Domain adaptation",
        "summary": "Lock development/test roles, compare a fixed baseline, adapt, and evaluate on the same sealed test.",
        "decisions": [
            _decision(
                "project",
                "Target-domain MedDeID project",
                kind="path",
                why="Training views remain tied to one governed project.",
            ),
            _decision(
                "development",
                "Development annotations JSONL",
                kind="path",
                why="Only development data may influence epoch selection or fitting.",
            ),
            _decision(
                "test_gold",
                "Sealed test gold JSONL",
                kind="path",
                why="The fixed baseline and adapted model must use identical held-out gold.",
            ),
            _decision(
                "review_development",
                "Review the development annotations in this workflow?",
                kind="bool",
                why="Accepted completed gold is distinct from deliberately reopened annotations.",
            ),
            *_annotation_decisions(ask_when=_eq("review_development", True)),
            _decision(
                "baseline_model",
                "Fixed baseline model",
                why="The before/after comparison must use an immutable baseline.",
            ),
            _decision(
                "baseline_revision",
                "Immutable baseline revision",
                why="A moving baseline invalidates comparison.",
            ),
            _decision(
                "detailed_evaluation",
                "Do you need detailed character-level/core-PII test metrics?",
                kind="bool",
                why="Core-PII evaluation requires character-level test gold.",
            ),
            _decision(
                "profiles",
                "Test locale profiles",
                kind="profiles",
                why="Detailed evaluation must route every test document to an explicit unversioned locale.",
                ask_when=_eq("detailed_evaluation", True),
            ),
            _decision(
                "training_protocol",
                "How should adaptation duration be chosen?",
                kind="choice",
                choices=("fit", "select_refit"),
                choice_labels={
                    "fit": "Ordinary one-stage fit for exploratory work",
                    "select_refit": "Select epochs on development data, then cleanly refit",
                },
                why="Publication comparisons should use selection/refit.",
            ),
            _decision(
                "config",
                "Training YAML",
                kind="path",
                why="The adaptation configuration must be reproducible.",
            ),
            *COMMON_RUNTIME_DECISIONS,
        ],
        "stages": [
            _stage(
                "lock_roles",
                "Lock development/test roles",
                "Hash the disjoint development and test identities before inference or training.",
                decisions=("project", "development", "test_gold"),
                action=_internal("lock-adaptation-roles"),
                outputs=("artifacts/role-lock.json",),
            ),
            _stage(
                "baseline",
                "Run fixed baseline",
                "Save unchanged baseline predictions before adaptation.",
                requires=("lock_roles",),
                decisions=("baseline_model", "baseline_revision", "device"),
                action=_command(
                    "meddeid",
                    "batch",
                    "{test_gold}",
                    "--output",
                    "{workspace}/artifacts/baseline-predictions.jsonl",
                    "--model",
                    "{baseline_model}",
                    "--device",
                    "{device}",
                    tools=("meddeid",),
                    options=(_option("baseline_revision", "--revision"),),
                ),
                outputs=(
                    "artifacts/baseline-predictions.jsonl",
                    "artifacts/baseline-predictions.jsonl.manifest.json",
                ),
                expensive=True,
                allows_detach=True,
            ),
            _stage(
                "prepare_review",
                "Prepare development review assignments",
                "Reopen development annotations only after the review decision was explicitly selected.",
                requires=("baseline",),
                decisions=("reviewer_count", "review_mode"),
                applicable_when=_eq("review_development", True),
                action=_internal("prepare-adaptation-assignments"),
                outputs=("assignments/assignment-manifest.json",),
            ),
            _stage(
                "preannotate",
                "Generate development suggestions",
                "Show model suggestions only for assisted review.",
                requires=("prepare_review",),
                decisions=("device",),
                applicable_when=_eq("review_development", True),
                enabled_when=_eq("review_mode", "assisted"),
                action=_internal("preannotate-assignments"),
                outputs=("artifacts/preannotation.json",),
                expensive=True,
                allows_detach=True,
                simple_label="Model suggestions",
            ),
            _stage(
                "primary_review",
                "Review development annotations",
                "Require deliberate whole-text confirmation before training.",
                requires=("preannotate",),
                decisions=("runtime",),
                applicable_when=_eq("review_development", True),
                action=_browser("annotate", source="{next_assignment}"),
                outputs=("assignments",),
                human=True,
                simple_label="Development annotation review",
            ),
            _stage(
                "curate",
                "Reconcile development reviewers",
                "Do not collapse independent development reviews silently.",
                requires=("primary_review",),
                decisions=("gold_policy", "runtime"),
                applicable_when=_all(
                    _eq("review_development", True), _gt("reviewer_count", 1)
                ),
                enabled_when=_eq("gold_policy", "adjudicate"),
                action=_browser(
                    "curate",
                    source="{workspace}/assignments",
                    data_dir="{workspace}/curation",
                ),
                outputs=(
                    "curation/exports/annotations.jsonl",
                    "curation/exports/manifest.json",
                ),
                human=True,
                simple_label="Reviewer reconciliation",
            ),
            _stage(
                "package_development",
                "Package authoritative development gold",
                "Pin the reviewed development source used to prepare training views.",
                requires=("curate",),
                applicable_when=_eq("review_development", True),
                action=_internal("package-authoritative"),
                outputs=(
                    "artifacts/authoritative-annotations.jsonl",
                    "artifacts/authoritative-annotations.manifest.json",
                ),
            ),
            _stage(
                "subannotate_test",
                "Subannotate sealed test gold",
                "Add detailed labels to test gold without exposing it to epoch selection.",
                requires=("package_development",),
                enabled_when=_eq("detailed_evaluation", True),
                decisions=("profiles", "runtime"),
                action=_browser(
                    "subannotate",
                    source="{test_gold}",
                    data_dir="{workspace}/test-subannotation",
                ),
                outputs=("test-subannotation/subannotations.jsonl",),
                human=True,
                simple_label="Detailed core-PII test subannotation",
            ),
            _stage(
                "export_test_bundle",
                "Export detailed test bundle",
                "Pin the completed test subannotations and profile resources by checksum.",
                requires=("subannotate_test",),
                enabled_when=_eq("detailed_evaluation", True),
                action=_internal("export-test-subannotation-bundle"),
                outputs=("test-subannotation/evaluation-bundle/manifest.json",),
            ),
            _stage(
                "prepare",
                "Prepare adaptation training views",
                "Create fit, selection, and refit views while keeping sealed test answers unavailable to selection.",
                requires=("export_test_bundle",),
                decisions=("project", "development", "test_gold"),
                action=_command(
                    "meddeid-data",
                    "project",
                    "prepare-training",
                    "{project}",
                    "--development",
                    "{adaptation_development}",
                    "--test-gold",
                    "{adaptation_test_gold}",
                    "--output",
                    "{workspace}/prepared",
                    tools=("meddeid-data",),
                ),
                outputs=("prepared/manifest.json",),
            ),
            *training_stages[1:],
            _stage(
                "compare",
                "Compare baseline and adapted model",
                "Score both systems against the identical sealed test and render the requested comparison.",
                requires=("score",),
                action=_internal("compare-adaptation"),
                outputs=("artifacts/comparison.json", "artifacts/plots"),
            ),
        ],
    }

    deployment = {
        "title": "Local and production deployment preparation",
        "summary": "Start a hardened local service, verify it, and produce an operational readiness report.",
        "decisions": [
            _decision(
                "model",
                "Model bundle or immutable image",
                why="Deployment must pin the serving artifact.",
            ),
            _decision(
                "revision",
                "Immutable model revision (optional)",
                required=False,
                why="A fixed revision prevents upstream model drift.",
            ),
            _decision(
                "language_profile",
                "Default locale for an ambiguous multi-profile model (optional)",
                required=False,
                why="A service may set one locale default while still allowing trusted per-document metadata to override it.",
            ),
            _decision(
                "deployment_target",
                "Deployment target",
                kind="choice",
                choices=("local", "organization"),
                why="Organizational deployment needs controls beyond localhost evaluation.",
            ),
            _decision(
                "tls_boundary",
                "TLS/reverse-proxy boundary documented?",
                kind="bool",
                why="Network deployment requires authenticated encrypted transport.",
                ask_when=_eq("deployment_target", "organization"),
            ),
            _decision(
                "port",
                "Local service port",
                kind="int",
                scope="operational",
                default=8000,
                why="Health checks and clients need one explicit local port.",
            ),
            COMMON_RUNTIME_DECISIONS[0],
        ],
        "stages": [
            _stage(
                "preflight",
                "Deployment preflight",
                "Check model, container runtime, ports, secrets, logging, and storage before starting.",
                decisions=("model", "deployment_target", "language_profile"),
                action=_internal("deployment-preflight"),
                outputs=("artifacts/deployment-preflight.json",),
            ),
            _stage(
                "start",
                "Start local service",
                "Launch the pinned service with localhost-safe defaults.",
                requires=("preflight",),
                decisions=("device", "port", "revision", "language_profile"),
                action=_internal("start-deployment"),
                outputs=("artifacts/service.json",),
            ),
            _stage(
                "health",
                "Verify health and model identity",
                "Confirm readiness and the exact served model without logging patient content.",
                requires=("start",),
                action=_internal("verify-deployment-health"),
                outputs=("artifacts/health.json",),
            ),
            _stage(
                "readiness",
                "Write readiness report",
                "Separate technical evidence from governance controls still requiring organizational approval.",
                requires=("health",),
                action=_internal("deployment-readiness-report"),
                outputs=("artifacts/deployment-readiness.json",),
            ),
        ],
    }

    language_profile = {
        "title": "Language/profile contribution",
        "summary": "Scaffold a regional capability, audit resources, and prove cross-component conformance.",
        "decisions": [
            _decision(
                "package_name",
                "Language package name (meddeid-language-*)",
                why="Capabilities remain independently versioned packages.",
            ),
            _decision(
                "profiles",
                "Comma-separated regional profiles",
                kind="profiles",
                why="Use unversioned locale identities such as en-GB and en-US; bare language identifiers remain insufficient.",
            ),
            _decision(
                "output_dir",
                "New package directory",
                kind="path",
                path_role="output",
                why="Scaffolding never writes into an unrelated component.",
            ),
            _decision(
                "resource_mode",
                "Resource mode",
                kind="choice",
                choices=("none", "local", "remote"),
                why="Resource provenance and network use must be explicit.",
            ),
            _decision(
                "allow_remote",
                "Authorize remote resource retrieval?",
                kind="bool",
                why="Remote retrieval can create policy and licensing obligations.",
                ask_when=_eq("resource_mode", "remote"),
            ),
        ],
        "stages": [
            _stage(
                "scaffold",
                "Scaffold package",
                "Create Python/JavaScript profile boundaries and versioned resource manifests without copying another locale implementation.",
                decisions=("package_name", "profiles", "output_dir"),
                action=_internal("scaffold-language-package"),
                outputs=("{output_dir}/pyproject.toml", "{output_dir}/package.json"),
            ),
            _stage(
                "resources",
                "Build and audit resources",
                "Verify source licences, hashes, regional scope, counts, and deterministic transforms.",
                requires=("scaffold",),
                decisions=("resource_mode",),
                applicable_when={"not": _eq("resource_mode", "none")},
                action=_internal("audit-language-resources"),
                outputs=("{output_dir}/resources-audit.json",),
                external=True,
                simple_label="Language resource build",
            ),
            _stage(
                "conformance",
                "Run profile conformance",
                "Test parsing, post-processing, subannotation routing, packaging, and resource identity across consumers.",
                requires=("resources",),
                action=_internal("test-language-conformance"),
                outputs=("{output_dir}/conformance.json",),
            ),
        ],
    }

    synthetic = {
        "title": "Synthetic corpus production",
        "summary": "Generate resumably, apply quality gates, review documents, and create sealed splits.",
        "decisions": [
            _decision(
                "profiles",
                "Comma-separated generation profiles",
                kind="profiles",
                why="Every document must retain its locale-specific generation contract.",
            ),
            _decision(
                "count",
                "Target document count",
                kind="int",
                minimum=2,
                why="Cost, review, and disjoint development/benchmark splits depend on the intended corpus size.",
            ),
            _decision(
                "seed",
                "Generation seed",
                kind="int",
                default=20260508,
                why="Deterministic stages must be reproducible.",
            ),
            _decision(
                "generation_mode",
                "Generation mode",
                kind="choice",
                choices=("local", "remote"),
                why="Remote author/reviewer calls require explicit authorization and cost tracking.",
            ),
            _decision(
                "allow_remote",
                "Authorize paid/remote generation?",
                kind="bool",
                why="External calls can incur cost and data-boundary implications.",
                ask_when=_eq("generation_mode", "remote"),
            ),
            _decision(
                "paid_model_review",
                "Authorize a separate paid model review?",
                kind="bool",
                why="Document authorship and model review are separate external calls and authorizations.",
                ask_when=_eq("generation_mode", "remote"),
            ),
            _decision(
                "reviewer_provider",
                "Remote reviewer provider",
                kind="choice",
                choices=("openai",),
                why="Paid review must use an explicitly configured provider.",
                ask_when=_all(
                    _eq("generation_mode", "remote"), _eq("paid_model_review", True)
                ),
            ),
        ],
        "stages": [
            _stage(
                "generate",
                "Generate corpus",
                "Use bounded resumable workers and persist accepted documents plus attempt-level usage immediately.",
                decisions=(
                    "profiles",
                    "count",
                    "seed",
                    "generation_mode",
                    "allow_remote",
                    "paid_model_review",
                    "reviewer_provider",
                ),
                action=_internal("generate-synthetic-corpus"),
                outputs=("artifacts/generated.jsonl", "artifacts/usage-ledger.jsonl"),
                expensive=True,
                external=True,
                allows_detach=True,
            ),
            _stage(
                "quality",
                "Run quality gates",
                "Reject coverage, diversity, provenance, leakage, or locale-balance regressions before review.",
                requires=("generate",),
                action=_internal("validate-synthetic-quality"),
                outputs=("artifacts/quality-report.json",),
            ),
            _stage(
                "review",
                "Review generated documents",
                "Record one explicit pass/fail result per document rather than a batch-level assertion.",
                requires=("quality",),
                action=_internal("review-synthetic-documents"),
                outputs=("artifacts/review-report.jsonl",),
                human=True,
            ),
            _stage(
                "seal",
                "Create development and benchmark splits",
                "Independently construct and hash the benchmark before model development uses it.",
                requires=("review",),
                action=_internal("seal-synthetic-splits"),
                outputs=(
                    "artifacts/development.jsonl",
                    "artifacts/benchmark.jsonl",
                    "artifacts/splits.manifest.json",
                ),
            ),
        ],
    }

    model_bundle = {
        "title": "Model bundle contribution",
        "summary": "Train or adopt a checkpoint, export a self-contained multi-profile bundle, and verify every public interface.",
        "decisions": [
            _decision(
                "checkpoint",
                "Checkpoint or completed training run",
                kind="path",
                why="Export must identify the exact learned weights.",
            ),
            _decision(
                "profiles",
                "Supported locale profiles",
                kind="profiles",
                why="Multi-profile bundles declare stable locale and ruleset identities; package, resource, model, and Git revisions provide provenance.",
            ),
            _decision(
                "base_encoder",
                "Base encoder",
                why="Tokenizer/config reconstruction depends on immutable base identity.",
            ),
            _decision(
                "base_revision",
                "Immutable base revision",
                why="A moving encoder is not a reproducible bundle.",
            ),
            COMMON_RUNTIME_DECISIONS[0],
        ],
        "stages": [
            _stage(
                "validate",
                "Validate checkpoint metadata",
                "Reject missing profile, tokenizer, label-order, or base-revision lineage.",
                decisions=("checkpoint", "profiles", "base_encoder", "base_revision"),
                action=_internal("validate-model-checkpoint"),
                outputs=("artifacts/checkpoint-validation.json",),
            ),
            _stage(
                "export",
                "Export self-contained bundle",
                "Package weights, tokenizer, config, labels, profiles, notices, and checksums.",
                requires=("validate",),
                action=_command(
                    "meddeid-train",
                    "export",
                    "--checkpoint",
                    "{checkpoint}",
                    "--output",
                    "{workspace}/artifacts/model",
                    "--base-encoder",
                    "{base_encoder}",
                    "--base-revision",
                    "{base_revision}",
                    tools=("meddeid-train",),
                ),
                outputs=("artifacts/model/bundle.json",),
            ),
            _stage(
                "interfaces",
                "Test all inference interfaces",
                "Verify Python, CLI, batch, service, offline loading, and every supported profile.",
                requires=("export",),
                decisions=("device",),
                action=_internal("verify-model-interfaces"),
                outputs=("artifacts/model-verification.json",),
                expensive=True,
            ),
        ],
    }

    return {
        "evaluation": evaluation,
        "training": training,
        "domain-adaptation": adaptation,
        "deployment": deployment,
        "language-profile": language_profile,
        "synthetic-corpus": synthetic,
        "model-bundle": model_bundle,
    }


TEMPLATES.update(_research_templates())


GUIDE_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "deidentify",
        "title": "De-identify clinical text",
        "summary": "Use a model on one note, a batch, or as a local service.",
        "workflows": ("inference",),
    },
    {
        "id": "prepare-data",
        "title": "Review or prepare a dataset",
        "summary": "Review source documents, create a benchmark, or generate synthetic data.",
        "workflows": ("dataset-review", "benchmark", "synthetic-corpus"),
    },
    {
        "id": "train",
        "title": "Train or improve a model",
        "summary": "Fit a model or adapt it to a target domain.",
        "workflows": ("training", "domain-adaptation"),
    },
    {
        "id": "evaluate",
        "title": "Evaluate or compare models",
        "summary": "Score predictions and optionally create plots or stability evidence.",
        "workflows": ("evaluation",),
    },
    {
        "id": "deploy",
        "title": "Deploy a model",
        "summary": "Start a hardened local service and verify readiness.",
        "workflows": ("deployment",),
    },
    {
        "id": "contribute",
        "title": "Contribute a language or model package",
        "summary": "Build a language/profile capability or a self-contained model bundle.",
        "workflows": ("language-profile", "model-bundle"),
    },
)


def list_templates() -> list[dict[str, str]]:
    return [
        {"id": key, "title": value["title"], "summary": value["summary"]}
        for key, value in TEMPLATES.items()
    ]


def list_guide_groups() -> list[dict[str, Any]]:
    templates = {item["id"]: item for item in list_templates()}
    return [
        {
            "id": group["id"],
            "title": group["title"],
            "summary": group["summary"],
            "workflows": [templates[workflow_id] for workflow_id in group["workflows"]],
        }
        for group in GUIDE_GROUPS
    ]


def get_template(template_id: str) -> dict[str, Any]:
    try:
        template = deepcopy(TEMPLATES[template_id])
    except KeyError as exc:
        choices = ", ".join(TEMPLATES)
        raise ValueError(
            f"unknown workflow {template_id!r}; choose one of: {choices}"
        ) from exc
    template["id"] = template_id
    template["version"] = TEMPLATE_VERSION
    validate_template(template)
    return template
