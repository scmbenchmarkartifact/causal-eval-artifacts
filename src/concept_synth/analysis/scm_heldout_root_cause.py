from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    load_dataset,
    markdown_table,
    pct,
    rate,
    read_jsonl,
    stable_json_dumps,
    write_csv,
    write_jsonl,
    write_markdown,
)
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)

SUMMARY_MD = "scm_heldout_root_cause.md"
ROOT_CAUSE_CSV = "scm_heldout_root_cause_by_instance.csv"
BY_MODEL_CSV = "scm_heldout_root_cause_by_model.csv"
BY_SLICE_CSV = "scm_heldout_root_cause_by_slice.csv"
EXAMPLES_JSONL = "scm_heldout_root_cause_examples.jsonl"
MANIFEST_JSON = "scm_heldout_root_cause_manifest.json"


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("instance_id") or "unknown"), str(row.get("model") or "unknown")


def _primary_bucket(
    instance_row: Dict[str, Any],
    *,
    non_exact_mechanisms: List[Dict[str, Any]],
    heldout_wrong_mechanisms: List[Dict[str, Any]],
) -> str:
    if instance_row.get("valid") is not True:
        bucket = str(instance_row.get("invalid_bucket") or "invalid")
        return f"invalid:{bucket}"
    if instance_row.get("heldout_exact") is True and instance_row.get("correct") is True:
        return "exact"

    if not non_exact_mechanisms:
        return "structurally_exact_but_world_wrong"

    if len(non_exact_mechanisms) == 1:
        if len(heldout_wrong_mechanisms) <= 1:
            return "single_mechanism_localized"
        return "single_mechanism_cascading"
    return "multi_mechanism_wrong"


def _build_instance_root_cause(
    instance_row: Dict[str, Any],
    mechanism_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    heldout_wrong = [
        row
        for row in mechanism_rows
        if row.get("heldout_var_accuracy") is not None and float(row.get("heldout_var_accuracy") or 0.0) < 0.999999
    ]
    train_wrong = [
        row
        for row in mechanism_rows
        if row.get("train_var_accuracy") is not None and float(row.get("train_var_accuracy") or 0.0) < 0.999999
    ]
    non_exact_mechanisms = [
        row
        for row in mechanism_rows
        if row.get("expr_exact") is False or row.get("parent_exact") is False
    ]
    parent_correct_formula_wrong = [
        row
        for row in mechanism_rows
        if row.get("expr_exact") is False and row.get("parent_exact") is True
    ]

    primary_bucket = _primary_bucket(
        instance_row,
        non_exact_mechanisms=non_exact_mechanisms,
        heldout_wrong_mechanisms=heldout_wrong,
    )
    first_wrong = None
    if heldout_wrong:
        ordered = sorted(
            heldout_wrong,
            key=lambda row: (
                int(row.get("target_topological_index") if row.get("target_topological_index") is not None else 10**9),
                str(row.get("target_var") or ""),
            ),
        )
        first_wrong = ordered[0]

    heldout_worlds = [row for row in world_rows if row.get("split") == "heldout"]
    dominant_world_mode = None
    if heldout_worlds:
        counts: Dict[str, int] = defaultdict(int)
        for row in heldout_worlds:
            if float(row.get("accuracy") or 0.0) < 0.999999:
                counts[str(row.get("intervention_mode") or "unknown")] += 1
        if counts:
            dominant_world_mode = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": str(instance_row.get("instance_id") or "unknown"),
        "model": str(instance_row.get("model") or "unknown"),
        "family": str(instance_row.get("family") or "unknown"),
        "slice": str(instance_row.get("slice") or "unknown"),
        "difficulty": str(instance_row.get("difficulty") or "unknown"),
        "valid": instance_row.get("valid"),
        "correct": instance_row.get("correct"),
        "train_exact": instance_row.get("train_exact"),
        "heldout_exact": instance_row.get("heldout_exact"),
        "split_error_bucket": instance_row.get("split_error_bucket"),
        "invalid_bucket": instance_row.get("invalid_bucket"),
        "primary_root_cause": primary_bucket,
        "heldout_intervention_sensitive": instance_row.get("heldout_mode_sensitivity_bucket") not in (None, "balanced", "unknown"),
        "high_novelty_generalization_failure": instance_row.get("heldout_novelty_sensitivity_bucket") == "high_novelty_sensitive",
        "parent_correct_formula_wrong": bool(parent_correct_formula_wrong),
        "non_exact_mechanism_count": len(non_exact_mechanisms),
        "parent_exact_count": sum(1 for row in mechanism_rows if row.get("parent_exact") is True),
        "formula_exact_count": sum(1 for row in mechanism_rows if row.get("expr_exact") is True),
        "canonical_formula_exact_count": sum(1 for row in mechanism_rows if row.get("expr_canonical_exact") is True),
        "heldout_wrong_mechanism_count": len(heldout_wrong),
        "train_wrong_mechanism_count": len(train_wrong),
        "first_wrong_variable": first_wrong.get("target_var") if first_wrong else None,
        "first_wrong_topological_index": first_wrong.get("target_topological_index") if first_wrong else None,
        "first_wrong_parent_exact": first_wrong.get("parent_exact") if first_wrong else None,
        "first_wrong_expr_exact": first_wrong.get("expr_exact") if first_wrong else None,
        "dominant_heldout_intervention_mode": dominant_world_mode,
        "heldout_mode_sensitivity_bucket": instance_row.get("heldout_mode_sensitivity_bucket"),
        "heldout_novelty_sensitivity_bucket": instance_row.get("heldout_novelty_sensitivity_bucket"),
        "replay_heldout_wrong_worlds": instance_row.get("replay_heldout_wrong_worlds"),
        "replay_heldout_wrong_cells": instance_row.get("replay_heldout_wrong_cells"),
        "heldout_wrong_variables": [str(row.get("target_var") or "") for row in heldout_wrong],
        "non_exact_mechanisms": [str(row.get("target_var") or "") for row in non_exact_mechanisms],
    }


def analyze_heldout_root_cause(
    instance_records: Sequence[Dict[str, Any]],
    mechanism_records: Sequence[Dict[str, Any]],
    world_records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    mech_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    world_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in mechanism_records:
        mech_by_key[_group_key(row)].append(row)
    for row in world_records:
        world_by_key[_group_key(row)].append(row)

    root_cause_rows: List[Dict[str, Any]] = []
    for row in instance_records:
        key = _group_key(row)
        root_cause_rows.append(_build_instance_root_cause(row, mech_by_key.get(key, []), world_by_key.get(key, [])))

    by_model_rows: List[Dict[str, Any]] = []
    by_slice_rows: List[Dict[str, Any]] = []

    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in root_cause_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
        grouped_slice[(str(row.get("slice") or "unknown"), str(row.get("model") or "unknown"))].append(row)

    for model in sorted(grouped_model.keys()):
        group = grouped_model[model]
        total = len(group)
        heldout_only = [row for row in group if row.get("split_error_bucket") == "heldout_only_error"]
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in heldout_only:
            bucket_counts[str(row.get("primary_root_cause") or "unknown")] += 1
        top_bucket = None
        top_bucket_share = None
        if bucket_counts and heldout_only:
            top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
            top_bucket_share = rate(top_count, len(heldout_only))
        by_model_rows.append(
            {
                "model": model,
                "n": total,
                "heldout_only_instances": len(heldout_only),
                "heldout_only_share": rate(len(heldout_only), total),
                "top_heldout_root_cause": top_bucket,
                "top_heldout_root_cause_share": top_bucket_share,
                "intervention_sensitive_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if row.get("heldout_intervention_sensitive")),
                    len(heldout_only),
                ),
                "high_novelty_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if row.get("high_novelty_generalization_failure")),
                    len(heldout_only),
                ),
                "parent_correct_formula_wrong_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if row.get("parent_correct_formula_wrong")),
                    len(heldout_only),
                ),
            }
        )

    for (slice_name, model), group in sorted(grouped_slice.items()):
        heldout_only = [row for row in group if row.get("split_error_bucket") == "heldout_only_error"]
        if not heldout_only:
            continue
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in heldout_only:
            bucket_counts[str(row.get("primary_root_cause") or "unknown")] += 1
        top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
        by_slice_rows.append(
            {
                "slice": slice_name,
                "model": model,
                "heldout_only_instances": len(heldout_only),
                "top_heldout_root_cause": top_bucket,
                "top_heldout_root_cause_share": rate(top_count, len(heldout_only)),
                "intervention_sensitive_share": rate(
                    sum(1 for row in heldout_only if row.get("heldout_intervention_sensitive")),
                    len(heldout_only),
                ),
                "high_novelty_share": rate(
                    sum(1 for row in heldout_only if row.get("high_novelty_generalization_failure")),
                    len(heldout_only),
                ),
            }
        )

    example_rows = [
        row
        for row in root_cause_rows
        if row.get("split_error_bucket") == "heldout_only_error" or str(row.get("primary_root_cause") or "").startswith("invalid:")
    ]

    return {
        "root_cause_rows": root_cause_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
        "example_rows": example_rows,
    }


def build_heldout_root_cause_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        rows.append(
            [
                row.get("model"),
                row.get("n"),
                row.get("heldout_only_instances"),
                pct(row.get("heldout_only_share")),
                row.get("top_heldout_root_cause") or "-",
                pct(row.get("top_heldout_root_cause_share")),
                pct(row.get("intervention_sensitive_share_among_heldout_only")),
                pct(row.get("high_novelty_share_among_heldout_only")),
            ]
        )
    lines = [
        "SCM Heldout Root Causes",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "Heldout-only N",
                "Heldout-only share",
                "Top root cause",
                "Top cause share",
                "Intervention-sensitive",
                "High-novelty",
            ],
            rows,
        )
        if rows
        else "No root-cause rows available.",
        "",
        "Interpretation notes:",
        "- `single_mechanism_localized` means one non-exact mechanism and only one heldout mechanism visibly wrong.",
        "- `single_mechanism_cascading` means one non-exact mechanism appears to propagate into downstream heldout mistakes.",
        "- `multi_mechanism_wrong` means multiple mechanisms differ structurally from gold.",
        "- `parent_correct_formula_wrong` is reported as a flag when formulas are wrong despite exact recovered parents.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_heldout_root_cause_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "root_cause_csv": str(out / ROOT_CAUSE_CSV),
        "by_model_csv": str(out / BY_MODEL_CSV),
        "by_slice_csv": str(out / BY_SLICE_CSV),
        "examples_jsonl": str(out / EXAMPLES_JSONL),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["root_cause_rows"], paths["root_cause_csv"])
    write_csv(artifacts["by_model_rows"], paths["by_model_csv"])
    write_csv(artifacts["by_slice_rows"], paths["by_slice_csv"])
    write_jsonl(artifacts["example_rows"], paths["examples_jsonl"])
    write_markdown(build_heldout_root_cause_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "files": paths,
        "counts": {key: len(value) for key, value in artifacts.items()},
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.instance_records and args.mechanism_records and args.world_records:
        return read_jsonl(args.instance_records), read_jsonl(args.mechanism_records), read_jsonl(args.world_records)
    if not args.input:
        raise SystemExit("Either --input or all of --instance-records/--mechanism-records/--world-records are required")
    _payload, problems = load_dataset(args.input)
    extracted = extract_scm_analysis_records(problems, source_name=str(args.input), models=None, family_filter=args.family)
    if args.extract_outdir:
        write_extract_artifacts(extracted, args.extract_outdir, source_name=str(args.input))
    return extracted["instance_records"], extracted["mechanism_records"], extracted["world_records"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze heldout-root-cause patterns for A_SCM model outputs")
    parser.add_argument("--input", help="Benchmark YAML input")
    parser.add_argument("--instance-records", help="scm_instance_records.jsonl path")
    parser.add_argument("--mechanism-records", help="scm_mechanism_records.jsonl path")
    parser.add_argument("--world-records", help="scm_world_records.jsonl path")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter when reading --input")
    parser.add_argument("--extract-outdir", help="Optional directory to write extracted records when reading --input")
    parser.add_argument("--outdir", required=True, help="Directory for heldout root-cause outputs")
    args = parser.parse_args(argv)

    instance_records, mechanism_records, world_records = _load_records(args)
    artifacts = analyze_heldout_root_cause(instance_records, mechanism_records, world_records)
    paths = write_heldout_root_cause_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM heldout root-cause analysis to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
