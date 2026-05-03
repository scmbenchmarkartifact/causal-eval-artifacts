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
from concept_synth.analysis.scm_subset_compare import _parse_models_arg

SUMMARY_MD = "scm_gap_decomposition_summary.md"
BY_INSTANCE_CSV = "scm_gap_decomposition_by_instance.csv"
BY_MODEL_CSV = "scm_gap_decomposition_by_model.csv"
BY_SLICE_CSV = "scm_gap_decomposition_by_slice.csv"
EXAMPLES_JSONL = "scm_gap_decomposition_examples.jsonl"
MANIFEST_JSON = "scm_gap_decomposition_manifest.json"


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("instance_id") or "unknown"), str(row.get("model") or "unknown")


def _bucket_for_row(
    instance_row: Dict[str, Any],
    amplification_row: Optional[Dict[str, Any]],
    equivalence_row: Optional[Dict[str, Any]],
) -> str:
    if instance_row.get("valid") is not True:
        return f"invalid:{str(instance_row.get('invalid_bucket') or 'invalid')}"
    if instance_row.get("train_exact") is True and instance_row.get("heldout_exact") is True:
        return "exact"
    if instance_row.get("train_exact") is False and instance_row.get("heldout_exact") is True:
        return "A_train_only_error"
    if instance_row.get("train_exact") is False and instance_row.get("heldout_exact") is False:
        amp_bucket = str((amplification_row or {}).get("amplification_bucket") or "no_clear_amplification")
        if amp_bucket in {"direct_amplification", "propagation_amplification", "mixed_amplification"}:
            return f"A_{amp_bucket}"
        return "A_no_clear_amplification"

    eq_bucket = str((equivalence_row or {}).get("training_equivalence_bucket") or "train_exact_not_shown_ambiguous_under_cap")
    if eq_bucket.startswith("train_exact_local_ambiguity"):
        return "B_local_ambiguity"
    if eq_bucket.startswith("train_exact_compensated_fit"):
        return "B_compensated_train_fit"
    if eq_bucket == "train_exact_bounded_ambiguity_fix_only":
        return "B_bounded_ambiguity_fix_only"
    return "B_not_shown_ambiguous_under_cap"


def analyze_gap_decomposition(
    instance_records: Sequence[Dict[str, Any]],
    amplification_instance_rows: Sequence[Dict[str, Any]],
    training_equivalence_instance_rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    amplification_by_key = {_group_key(row): row for row in amplification_instance_rows}
    equivalence_by_key = {_group_key(row): row for row in training_equivalence_instance_rows}

    instance_rows: List[Dict[str, Any]] = []
    for instance_row in instance_records:
        key = _group_key(instance_row)
        amplification_row = amplification_by_key.get(key)
        equivalence_row = equivalence_by_key.get(key)
        bucket = _bucket_for_row(instance_row, amplification_row, equivalence_row)
        instance_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "instance_id": instance_row.get("instance_id"),
                "model": instance_row.get("model"),
                "family": instance_row.get("family"),
                "slice": instance_row.get("slice"),
                "difficulty": instance_row.get("difficulty"),
                "valid": instance_row.get("valid"),
                "train_exact": instance_row.get("train_exact"),
                "heldout_exact": instance_row.get("heldout_exact"),
                "split_error_bucket": instance_row.get("split_error_bucket"),
                "invalid_bucket": instance_row.get("invalid_bucket"),
                "gap_bucket": bucket,
                "amplification_bucket": None if amplification_row is None else amplification_row.get("amplification_bucket"),
                "training_equivalence_bucket": None if equivalence_row is None else equivalence_row.get("training_equivalence_bucket"),
            }
        )

    by_model_rows: List[Dict[str, Any]] = []
    by_slice_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
        grouped_slice[(str(row.get("slice") or "unknown"), str(row.get("model") or "unknown"))].append(row)

    bucket_names = [
        "exact",
        "invalid",
        "A_direct_amplification",
        "A_propagation_amplification",
        "A_mixed_amplification",
        "A_no_clear_amplification",
        "A_train_only_error",
        "B_local_ambiguity",
        "B_compensated_train_fit",
        "B_bounded_ambiguity_fix_only",
        "B_not_shown_ambiguous_under_cap",
    ]

    def _row_counts(group: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        out = {name: 0 for name in bucket_names}
        for row in group:
            bucket = str(row.get("gap_bucket") or "")
            if bucket.startswith("invalid:"):
                out["invalid"] += 1
            elif bucket in out:
                out[bucket] += 1
        return out

    for model, group in sorted(grouped_model.items()):
        counts = _row_counts(group)
        total = len(group)
        by_model_rows.append(
            {
                "model": model,
                "n": total,
                "exact_share": rate(counts["exact"], total),
                "invalid_share": rate(counts["invalid"], total),
                "a_direct_amplification_share": rate(counts["A_direct_amplification"], total),
                "a_propagation_amplification_share": rate(counts["A_propagation_amplification"], total),
                "a_mixed_amplification_share": rate(counts["A_mixed_amplification"], total),
                "a_no_clear_amplification_share": rate(counts["A_no_clear_amplification"], total),
                "b_local_ambiguity_share": rate(counts["B_local_ambiguity"], total),
                "b_compensated_train_fit_share": rate(counts["B_compensated_train_fit"], total),
                "b_fix_only_share": rate(counts["B_bounded_ambiguity_fix_only"], total),
                "b_not_shown_ambiguous_share": rate(counts["B_not_shown_ambiguous_under_cap"], total),
            }
        )

    for (slice_value, model), group in sorted(grouped_slice.items()):
        counts = _row_counts(group)
        total = len(group)
        by_slice_rows.append(
            {
                "slice": slice_value,
                "model": model,
                "n": total,
                "a_amplified_share": rate(
                    counts["A_direct_amplification"] + counts["A_propagation_amplification"] + counts["A_mixed_amplification"],
                    total,
                ),
                "b_local_ambiguity_share": rate(counts["B_local_ambiguity"], total),
                "b_compensated_train_fit_share": rate(counts["B_compensated_train_fit"], total),
                "b_not_shown_ambiguous_share": rate(counts["B_not_shown_ambiguous_under_cap"], total),
            }
        )

    example_rows = [
        row
        for row in instance_rows
        if row.get("gap_bucket") in {
            "A_direct_amplification",
            "A_propagation_amplification",
            "A_mixed_amplification",
            "B_local_ambiguity",
            "B_compensated_train_fit",
            "B_not_shown_ambiguous_under_cap",
        }
    ]
    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
        "example_rows": example_rows,
    }


def build_gap_decomposition_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                pct(row.get("exact_share")),
                pct(row.get("invalid_share")),
                pct(row.get("a_direct_amplification_share")),
                pct(row.get("a_propagation_amplification_share")),
                pct(row.get("a_mixed_amplification_share")),
                pct(row.get("a_no_clear_amplification_share")),
                pct(row.get("b_local_ambiguity_share")),
                pct(row.get("b_compensated_train_fit_share")),
                pct(row.get("b_not_shown_ambiguous_share")),
            ]
        )
    lines = [
        "SCM Gap Decomposition",
        "",
        "This report decomposes the train-to-heldout gap into train-error amplification (A) and train-exact heldout misses (B).",
        "Bucket A is split into direct, propagation-driven, and mixed amplification. Bucket B is split into local train ambiguity, compensated train fit, and cases not shown ambiguous under the bounded search class.",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "Exact",
                "Invalid",
                "A direct",
                "A propagation",
                "A mixed",
                "A no-clear",
                "B local ambiguity",
                "B compensated",
                "B not shown ambiguous",
            ],
            model_rows,
        ) if model_rows else "No gap decomposition rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_gap_decomposition_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_instance_csv": str(out / BY_INSTANCE_CSV),
        "by_model_csv": str(out / BY_MODEL_CSV),
        "by_slice_csv": str(out / BY_SLICE_CSV),
        "examples_jsonl": str(out / EXAMPLES_JSONL),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["instance_rows"], paths["by_instance_csv"])
    write_csv(artifacts["by_model_rows"], paths["by_model_csv"])
    write_csv(artifacts["by_slice_rows"], paths["by_slice_csv"])
    write_jsonl(artifacts["example_rows"], paths["examples_jsonl"])
    write_markdown(build_gap_decomposition_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decompose the A_SCM train-to-heldout gap")
    parser.add_argument("--instance-records", required=True, help="Extracted instance JSONL")
    parser.add_argument("--amplification-records", required=True, help="Amplification by-instance CSV or JSONL exported as JSONL")
    parser.add_argument("--training-equivalence-records", required=True, help="Training-equivalence by-instance CSV or JSONL exported as JSONL")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    instance_rows = read_jsonl(args.instance_records)
    amplification_rows = read_jsonl(args.amplification_records)
    training_equivalence_rows = read_jsonl(args.training_equivalence_records)
    artifacts = analyze_gap_decomposition(instance_rows, amplification_rows, training_equivalence_rows)
    paths = write_gap_decomposition_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM gap decomposition artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
