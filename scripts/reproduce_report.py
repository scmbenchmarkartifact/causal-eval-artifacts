#!/usr/bin/env python3
"""Run the offline SCM report bundle on a benchmark file with embedded results."""

from __future__ import annotations

import argparse
from pathlib import Path

from concept_synth.analysis.scm_report import run_full_report


def _parse_models(value: str | None) -> set[str] | None:
    if not value:
        return None
    models = {piece.strip() for piece in value.split(",") if piece.strip()}
    return models or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce offline SCM analysis artifacts")
    parser.add_argument("--input", required=True, type=Path, help="Benchmark runtime YAML with embedded results")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--family", choices=["ordered", "ntopo"], default=None)
    parser.add_argument("--models", default=None, help="Optional comma-separated model filter")
    parser.add_argument("--baseline-input", default=None, help="Optional baseline YAML for subset comparisons")
    args = parser.parse_args()

    manifest = run_full_report(
        str(args.input),
        args.outdir,
        family=args.family,
        models=_parse_models(args.models),
        baseline_input=args.baseline_input,
    )
    print(f"Wrote report summary: {manifest['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
