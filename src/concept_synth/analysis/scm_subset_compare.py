from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    load_dataset,
    markdown_table,
    pct,
    rate,
    read_jsonl,
    stable_json_dumps,
    write_csv,
    write_markdown,
)
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)

SUMMARY_MD = "scm_subset_compare.md"
OVERVIEW_CSV = "scm_subset_overview.csv"
BY_SLICE_CSV = "scm_subset_by_slice.csv"
MANIFEST_JSON = "scm_subset_manifest.json"


def _parse_models_arg(models: Optional[str]) -> Optional[Set[str]]:
    if not models:
        return None
    items = {piece.strip() for piece in str(models).split(",") if piece.strip()}
    return items or None


def _baseline_instance_ids(path: Optional[str], *, family_filter: Optional[str] = None) -> Set[str]:
    if not path:
        return set()
    _payload, problems = load_dataset(path)
    out: Set[str] = set()
    for problem in problems:
        from concept_synth.analysis.scm_common import family, instance_id, is_a_scm_problem

        if not is_a_scm_problem(problem):
            continue
        if family_filter and family(problem) != family_filter:
            continue
        out.add(instance_id(problem))
    return out


def _selected_models(instance_records: Sequence[Dict[str, Any]], explicit_models: Optional[Set[str]] = None) -> List[str]:
    observed = sorted({str(row.get("model") or "unknown") for row in instance_records})
    if explicit_models is None:
        return observed
    return [model for model in observed if model in explicit_models]


def _instances_for_models(instance_records: Sequence[Dict[str, Any]], models: Sequence[str]) -> Dict[str, Set[str]]:
    by_model: Dict[str, Set[str]] = {model: set() for model in models}
    for row in instance_records:
        model = str(row.get("model") or "unknown")
        if model in by_model:
            by_model[model].add(str(row.get("instance_id") or "unknown"))
    return by_model


def _subset_specs(
    instance_records: Sequence[Dict[str, Any]],
    *,
    selected_models: Sequence[str],
    baseline_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Set[str], List[str]]]:
    by_model = _instances_for_models(instance_records, selected_models)
    specs: List[Tuple[str, Set[str], List[str]]] = []
    if selected_models:
        common_ids = set.intersection(*(ids for ids in by_model.values())) if by_model else set()
        specs.append(("common_all_models", common_ids, list(selected_models)))
    if baseline_ids is not None and baseline_ids:
        observed_ids = {str(row.get("instance_id") or "unknown") for row in instance_records}
        specs.append(("baseline_overlap", observed_ids & baseline_ids, list(selected_models)))
        specs.append(("new_only", observed_ids - baseline_ids, list(selected_models)))
    return [(name, ids, models) for name, ids, models in specs if ids]


def _metrics(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(group)
    return {
        "n": total,
        "valid_rate": rate(sum(1 for row in group if row.get("valid") is True), total),
        "correct_rate": rate(sum(1 for row in group if row.get("correct") is True), total),
        "train_exact_rate": rate(sum(1 for row in group if row.get("train_exact") is True), total),
        "heldout_exact_rate": rate(sum(1 for row in group if row.get("heldout_exact") is True), total),
        "heldout_only_error_rate": rate(sum(1 for row in group if row.get("split_error_bucket") == "heldout_only_error"), total),
        "train_and_heldout_error_rate": rate(sum(1 for row in group if row.get("split_error_bucket") == "train_and_heldout_error"), total),
    }


def analyze_subset_compare(
    instance_records: Sequence[Dict[str, Any]],
    *,
    selected_models: Optional[Set[str]] = None,
    baseline_ids: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    models = _selected_models(instance_records, selected_models)
    subset_specs = _subset_specs(instance_records, selected_models=models, baseline_ids=baseline_ids)

    overview_rows: List[Dict[str, Any]] = []
    by_slice_rows: List[Dict[str, Any]] = []

    for subset_name, instance_ids, subset_models in subset_specs:
        subset_rows = [
            row
            for row in instance_records
            if str(row.get("instance_id") or "unknown") in instance_ids and str(row.get("model") or "unknown") in subset_models
        ]
        for model in subset_models:
            group = [row for row in subset_rows if str(row.get("model") or "unknown") == model]
            if not group:
                continue
            payload = {
                "subset": subset_name,
                "model": model,
                "instance_count": len(instance_ids),
                "coverage": rate(len(group), len(instance_ids)),
            }
            payload.update(_metrics(group))
            overview_rows.append(payload)

        slices = sorted({str(row.get("slice") or "unknown") for row in subset_rows})
        for slice_name in slices:
            slice_rows = [row for row in subset_rows if str(row.get("slice") or "unknown") == slice_name]
            for model in subset_models:
                group = [row for row in slice_rows if str(row.get("model") or "unknown") == model]
                if not group:
                    continue
                payload = {
                    "subset": subset_name,
                    "slice": slice_name,
                    "model": model,
                    "instance_count": len({str(row.get("instance_id") or "unknown") for row in slice_rows}),
                    "coverage": rate(len(group), len({str(row.get("instance_id") or "unknown") for row in slice_rows})),
                }
                payload.update(_metrics(group))
                by_slice_rows.append(payload)

    return {
        "overview_rows": overview_rows,
        "by_slice_rows": by_slice_rows,
    }


def build_subset_compare_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    overview_rows = artifacts["overview_rows"]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in overview_rows:
        grouped[str(row.get("subset") or "unknown")].append(row)

    sections: List[str] = ["SCM Subset Comparison", ""]
    for subset in sorted(grouped.keys()):
        rows = []
        for row in sorted(grouped[subset], key=lambda item: str(item.get("model") or "")):
            rows.append(
                [
                    row.get("model"),
                    row.get("instance_count"),
                    pct(row.get("coverage")),
                    pct(row.get("valid_rate")),
                    pct(row.get("correct_rate")),
                    pct(row.get("train_exact_rate")),
                    pct(row.get("heldout_exact_rate")),
                ]
            )
        sections.extend(
            [
                f"### {subset}",
                "",
                markdown_table(
                    ["Model", "Instances", "Coverage", "Valid", "Correct", "TrainExact", "HeldoutExact"],
                    rows,
                )
                if rows
                else "No rows.",
                "",
            ]
        )
    if len(sections) == 2:
        sections.append("No subset-comparison rows available.")
    return "\n".join(sections).strip() + "\n"


def write_subset_compare_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "overview_csv": str(out / OVERVIEW_CSV),
        "by_slice_csv": str(out / BY_SLICE_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["overview_rows"], paths["overview_csv"])
    write_csv(artifacts["by_slice_rows"], paths["by_slice_csv"])
    write_markdown(build_subset_compare_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "files": paths,
        "counts": {key: len(value) for key, value in artifacts.items()},
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_instance_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.instance_records:
        return read_jsonl(args.instance_records)
    if not args.input:
        raise SystemExit("Either --input or --instance-records is required")
    _payload, problems = load_dataset(args.input)
    extracted = extract_scm_analysis_records(
        problems,
        source_name=str(args.input),
        models=_parse_models_arg(args.models),
        family_filter=args.family,
    )
    if args.extract_outdir:
        write_extract_artifacts(extracted, args.extract_outdir, source_name=str(args.input))
    return extracted["instance_records"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare A_SCM model performance on common / old / new subsets")
    parser.add_argument("--input", help="Benchmark YAML input")
    parser.add_argument("--instance-records", help="scm_instance_records.jsonl path")
    parser.add_argument("--baseline-input", help="Optional baseline benchmark YAML for old/new splits")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    parser.add_argument("--extract-outdir", help="Optional directory to write extracted records when reading --input")
    parser.add_argument("--outdir", required=True, help="Directory for subset comparison outputs")
    args = parser.parse_args(argv)

    instance_records = _load_instance_records(args)
    baseline_ids = _baseline_instance_ids(args.baseline_input, family_filter=args.family) if args.baseline_input else None
    artifacts = analyze_subset_compare(
        instance_records,
        selected_models=_parse_models_arg(args.models),
        baseline_ids=baseline_ids,
    )
    paths = write_subset_compare_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM subset comparison to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
