# Evaluation Protocol

## Prompting

Use `scripts/export_prompts.py` to materialize the exact prompts from a
benchmark file. The script writes JSONL with the system prompt and user prompt
for each problem.

## Prediction Collection

Run any model or reasoning system outside this artifact and save one prediction
per line:

```json
{"instanceId":"...","model":"system-name","response":"..."}
```

The artifact does not require API keys and does not include provider-specific
live model runners.

## Scoring

Use `scripts/score_predictions.py` to score prediction JSONL against a benchmark
file. The scorer parses candidate mechanisms, evaluates them on train and
held-out worlds, and writes per-instance and aggregate metrics.

## Reproducing Paper Tables

Use `scripts/reproduce_report.py` on benchmark files that include embedded
`llmResults`. This regenerates normalized records, CSV tables, and a Markdown
summary report.

## Determinism

Scoring is deterministic for a fixed benchmark file and prediction file.
Benchmark generation is not part of this minimal release snapshot.
