# Data Schema

Benchmark files are YAML files loaded through
`concept_synth.causal_reasoning.runtime_storage_io.load_causal_dataset`.

## Top-Level Structure

Files contain metadata plus a list of problem records. Runtime YAML files use a
compact representation that is expanded by the loader before prompting or
scoring.

## Problem Records

After loading, each record contains:

- `problem`: problem metadata, including `instanceId` and task parameters
- `problemDescription`: finite worlds and prompt-facing information
- `llmResults`: optional embedded model outputs and evaluations

## Model Results

The scorer accepts model results with:

- `model`: model or system name
- `response` or `rawResponse`: raw model text
- `extractedAnswer`: optional pre-parsed JSON object

If `extractedAnswer` is absent, the scorer extracts the first JSON object from
the model response.

## Answer Format

The expected answer is a JSON object with a `mechanisms` field:

```json
{
  "mechanisms": {
    "X1": "(and X0 X2)",
    "X2": "(xor X0 X1)"
  }
}
```

The exact variable names and allowed operators are problem-specific and are
provided in the prompt.
