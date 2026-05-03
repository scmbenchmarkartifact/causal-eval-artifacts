# Benchmark Card

## Name

Anonymous Causal SCM Benchmark

## Intended Use

The benchmark evaluates whether systems can infer compact structural causal
mechanisms from finite interventional observations and generalize those
mechanisms to held-out intervention worlds.

The benchmark is intended for evaluation, stress testing, and analysis of
causal reasoning behavior in language models and other symbolic or hybrid
reasoning systems.

## Tasks

The release contains several related SCM benchmark slices:

- Ordered SCM induction
- Non-topological-order SCM induction
- Partial-order disclosure
- Root-unknown non-topological SCM induction
- Alternative-explanation identification
- Identifiability audit
- Identifiability counterexample stress tests

## Input

Each problem provides finite Boolean worlds, observed endogenous variables,
intervention assignments, and a prompt asking the system to return a JSON object
containing candidate mechanisms.

## Output

Systems should return JSON with a `mechanisms` object mapping endogenous
variables to mechanism expressions in the benchmark DSL.

## Metrics

Primary metrics include:

- Valid JSON/mechanism output
- Problem-level correctness
- Train-world exactness
- Held-out-world exactness
- Train and held-out accuracy
- Parent-set recovery and mechanism-level diagnostics

## Limitations

The benchmark uses synthetic finite Boolean SCMs. This supports exact
inspection and controlled stress tests, but it is not a direct measurement of
performance on open-world causal discovery from observational scientific data.

## Responsible Release

The benchmark contains synthetic data and does not include personal data. The
artifact is anonymized for double-blind review.
