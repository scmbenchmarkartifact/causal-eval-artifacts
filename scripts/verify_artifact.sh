#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m compileall -q src scripts
python scripts/export_prompts.py \
  --benchmark data/samples/sample_runtime.yaml \
  --out-jsonl outputs/sample_prompts.jsonl \
  --limit 2
python scripts/score_predictions.py \
  --benchmark data/samples/sample_runtime.yaml \
  --predictions data/samples/sample_predictions.jsonl \
  --outdir outputs/sample_score
python scripts/reproduce_report.py \
  --input data/samples/sample_runtime.yaml \
  --outdir outputs/sample_report

echo "Artifact smoke verification completed."
