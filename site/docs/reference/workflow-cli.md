# Workflow CLI reference

<span class="source-label">Owner: meddeid</span>

The workflow CLI coordinates multi-step work across MedDeID components. It
records important decisions, resolves the eligible stages, and runs one stage
at a time. If you have not selected a task yet, begin with
[Choose a workflow](../workflows/index.md).

The CLI does not make study-design decisions for you. Decide whether your work
requires independent review, detailed subannotation, stability analysis, or a
clean refit before accepting the proposed path.

## Create a workspace

Install the capability set required by your workflow, check the machine, and
start the interactive assistant. For example, a research workflow uses:

```bash
python -m pip install 'meddeid[research]'
meddeid doctor --workflow benchmark
meddeid start
```

`start` asks for a workspace location and then asks protocol-level questions
one at a time. It shows the resolved path before writing anything. Decisions
can include:

- whether existing gold is accepted or deliberately reopened;
- blinded or model-assisted primary review;
- reviewer count and the authoritative-gold policy;
- whether detailed subannotation is an evaluation outcome;
- ordinary fitting or epoch selection followed by a clean full refit; and
- whether remote or paid generation is permitted.

Remote document generation and a separate paid model-review pass are distinct
decisions. Selecting paid review also requires naming the provider; permission
for generation never silently authorizes review.

## Continue a workspace

The normal daily interaction is:

```bash
meddeid status ./work/my-benchmark
meddeid next ./work/my-benchmark
```

From inside the workspace, its path can be omitted:

```bash
cd ./work/my-benchmark
meddeid status
meddeid next
```

`status` shows overall progress, the next human-readable action, and important
exclusions. Add `--details` for the complete stage table. `next` validates
inputs, decisions, dependencies, and branch conditions before executing exactly
the first ready stage.

## Understand stage states

| State | Meaning |
|---|---|
| `pending` | An earlier stage is not finished. |
| `needs_input` | A decision required at this point is unanswered. |
| `ready` | Inputs, decisions, and prerequisites validate. |
| `running` | A foreground or detached process owns the stage. |
| `completed` | Required outputs exist, match their hashes, and pass registered artifact checks. |
| `skipped` | The user explicitly declined an applicable optional stage. |
| `not_applicable` | The selected study design excludes the stage. |
| `blocked` | A prerequisite or artifact is missing or invalid. |
| `failed` | The component command returned an error. |

Use `explain` when you need to understand why a stage has its current state:

```bash
meddeid workflow explain ./work/my-benchmark
```

## Inspect or run a specific stage

The workflow does not hide the component commands. Preview a resolved command
before running it:

```bash
meddeid workflow run ./work/my-benchmark subannotate --dry-run
meddeid workflow run ./work/my-benchmark score --dry-run
```

Use the component CLI directly when an advanced option is not represented by
the workflow. The component repositories remain independently usable.

Long inference, training, and generation stages support detached execution:

```bash
meddeid next ./work/my-benchmark --yes --detach
```

Human browser review remains in the foreground. Closing it before all records
are confirmed leaves the stage pending; the next run resumes from the saved
assignment.

## Automate workspace creation

Users and scripts that already know the workflow type can initialize it
non-interactively:

```bash
meddeid workflow init benchmark ./work/my-benchmark \
  --non-interactive \
  --set source=/data/gold.jsonl \
  --set input_role=existing_gold \
  --set re_review=false \
  --set detailed_evaluation=true \
  --set profiles=en-GB \
  --set score_predictions=false
```

Non-interactive mode never prompts. If a decision is missing, it exits with
code 3 and prints the exact `workflow configure --set ...` command required.

## Change a decision safely

```bash
meddeid workflow configure ./work/my-benchmark \
  --set profiles=en-GB,en-US \
  --reason 'change the selected locale coverage'
```

If the change would invalidate completed downstream work, the command stops and
lists the impact. Review it and repeat with `--yes`; MedDeID archives the exact
invalidated outputs and resets only affected stages. Source artifacts are never
overwritten.

Reviewer count, gold policy, profile routing, split roles, and training protocol
are handled this way. Multiple independent reviews can proceed only through
adjudication or an explicitly selected reviewer policy with a rationale.

## What the workspace records

`workflow.json` uses the `meddeid.workflow.v1` contract. It records the workflow
template and version, explicit decisions, ordered stage graph, input and output
paths and hashes, installed package and runtime versions, and a privacy-safe
event history. Status is recomputed from current artifacts instead of trusting
a stale completion flag.

Do not put patient text, names, or free-text clinical content in decision
reasons. Keep the workflow workspace outside every source or project directory
it records as an input. The CLI rejects a nested workspace because its own logs
and artifacts would otherwise change the checksum of that pinned input.
