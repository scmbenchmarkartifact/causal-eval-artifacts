# Anonymous Causal SCM Benchmark Artifact

This repository is an anonymized review artifact for a NeurIPS 2026 benchmark
submission. It contains the frozen causal SCM benchmark files, minimal prompt
and scoring code, and the offline analysis scripts needed to reproduce the
paper-level reports from embedded result records.

The artifact intentionally excludes unrelated induction, abduction, paper-build,
batch-job, and scratch-result code from the development repository.

## Contents

```text
data/
  benchmarks/      Frozen official runtime benchmark YAML files
  samples/         Small smoke-test benchmark and prediction files
  metadata/        Draft Croissant metadata for dataset-hosting submission
docs/
  benchmark_card.md
  data_schema.md
  evaluation_protocol.md
scripts/
  export_prompts.py
  score_predictions.py
  reproduce_report.py
  validate_paper_results.py
  verify_artifact.sh
artifact_manifest.json  File sizes and SHA-256 hashes for release contents
src/concept_synth/
  causal_reasoning/  Prompt rendering, SCM parsing, and scoring
  analysis/          Offline report reproduction scripts
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The core scoring path depends only on Python 3.10+, `ruamel.yaml`, and
`PyYAML`.

## Research Code Checklist

This artifact follows the NeurIPS/Papers with Code research-code checklist in
the form appropriate for a benchmark release:

| Checklist item | Status in this artifact |
| --- | --- |
| Dependencies | Declared in `pyproject.toml`; install command above. |
| Training code | Not applicable: this artifact releases a benchmark, scoring code, and frozen result records, not a trained model. |
| Evaluation code | `scripts/export_prompts.py`, `scripts/score_predictions.py`, `scripts/reproduce_report.py`, and `scripts/validate_paper_results.py`. |
| Pre-trained models | Not applicable: no learned model weights are introduced by the submission. |
| Results and commands | Representative aggregate results and exact validation commands are below. |

## Smoke Test

```bash
./scripts/verify_artifact.sh
```

This compiles the release code, exports prompts from the sample benchmark,
scores sample predictions, and runs the offline report bundle on the sample.

## Export Prompts

```bash
python scripts/export_prompts.py \
  --benchmark data/benchmarks/cind_a_scm_benchmark240_final_v1.runtime.yaml \
  --out-jsonl outputs/prompts.jsonl \
  --limit 10
```

Each JSONL record contains `instanceId`, `taskName`, `system`, and `prompt`.

## Score External Predictions

Predictions can be provided as JSONL:

```json
{"instanceId":"example-id","model":"my-model","response":"{\"mechanisms\":{...}}"}
```

Then score them:

```bash
python scripts/score_predictions.py \
  --benchmark data/benchmarks/cind_a_scm_benchmark240_final_v1.runtime.yaml \
  --predictions my_predictions.jsonl \
  --outdir outputs/my_predictions_scored
```

The scorer writes:

- `scored_predictions.jsonl`
- `scored_predictions.csv`
- `summary.json`

## Validate Paper Result Tables

The paper-table validator is replay-only: it reads stored `evaluation` records
inside the frozen runtime YAML files and compares aggregate values to
`data/reference/paper_model_results_expected.json`.

Representative frozen paper results from `topline_results_main` are:

| System | Ord-Full HE | Block HE | Hid-Full HE |
| --- | ---: | ---: | ---: |
| GPT-5.4 | 0.344 | 0.410 | 0.292 |
| Opus 4.6 | 0.292 | 0.240 | 0.164 |
| bnlearn+DSL | 0.596 | 0.620 | 0.620 |
| symbolic exact-search | 0.596 | 0.650 | 0.536 |

`HE` denotes held-out exact accuracy. The complete expected aggregate tables
are stored in `data/reference/paper_model_results_expected.json`.

```bash
python scripts/validate_paper_results.py \
  --expected data/reference/paper_model_results_expected.json \
  --outdir outputs/paper_result_validation
```

Rows for an in-progress model run can be excluded from the comparison by model
id or display name. For example, to validate the submitted tables while
excluding `grok4.3` and accepting last-digit display rounding:

```bash
python scripts/validate_paper_results.py \
  --expected data/reference/paper_model_results_expected.json \
  --ignore-model grok4.3 \
  --numeric-tolerance 0.001 \
  --outdir outputs/paper_result_validation_ignore_grok43
```

This command does not run LLM calls, symbolic solvers, SCM generation, or
re-scoring.

## Reproduce Offline Reports

For benchmark YAML files that already contain embedded `llmResults`, run:

```bash
python scripts/reproduce_report.py \
  --input data/benchmarks/cind_a_scm_benchmark240_final_v1.runtime.yaml \
  --outdir outputs/main240_report
```

This produces normalized records, CSV tables, and a Markdown report bundle.

## Frozen Benchmark Files

The full benchmark files are in `data/benchmarks/`:

- `cind_a_scm_benchmark240_final_v1.runtime.yaml`
- `cind_a_scm_ntopo_benchmark240_final_v1.runtime.yaml`
- `cind_a_scm_paired50_partial_order_mixed25_2block25_3block_v3.runtime.yaml`
- `cind_a_scm_root_unknown_paired90_ntopo_v1.runtime.yaml`
- `cind_a_scm_alt_exp_paired50_ordered_v1.runtime.yaml`
- `cind_a_scm_alt_exp_paired50_ntopo_v1.runtime.yaml`
- `cind_a_scm_ident_audit_ordered_pairs100_v1.runtime.yaml`
- `cind_a_scm_ident_audit_ntopo_pairs100_v1.runtime.yaml`
- `cind_a_scm_ident_counterexample_ordered_pairs100_v5.runtime.yaml`
- `cind_a_scm_ident_counterexample_ntopo_pairs100_v5.runtime.yaml`

These files are copied from the tracked repository snapshot used for submission.
They are not regenerated by this artifact.

## Release Notes

The live LLM provider runner is intentionally not included in this minimal
artifact. Reviewers can inspect prompts and score any externally generated
predictions through `scripts/score_predictions.py` without API keys or provider
accounts.

The dataset should be hosted on a dedicated dataset platform for NeurIPS
submission. If the code repository host has size limits, keep `data/samples/`
in the code repository and upload `data/benchmarks/` plus Croissant metadata to
the dataset host.

## Licenses

Code is released under the MIT License. Benchmark data is intended for public
release under CC BY 4.0; see `DATA_LICENSE.txt`.
