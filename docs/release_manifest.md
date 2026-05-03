# Release Manifest

## Included Code

The release includes only:

- YAML runtime loading
- Causal SCM prompt rendering
- Mechanism DSL parsing and evaluation
- Model-output scoring
- Offline report reproduction

## Excluded Code

The release excludes:

- Development batch runners
- Live LLM provider clients
- Construction-time benchmark augmentation scripts
- Old induction and abduction benchmarks
- Paper build artifacts and LaTeX auxiliary files
- Local scratch outputs and raw batch-job files

## Dataset Files

The official benchmark runtime files are stored in `data/benchmarks/`.
`data/samples/` contains a small sample for smoke tests only.
`data/metadata/croissant.json` contains the Croissant core and Responsible AI
metadata to upload with the OpenReview submission.

## Anonymization

This artifact removes author names, institutions, personal paths, personal
emails, and development-repository history from the released snapshot.
