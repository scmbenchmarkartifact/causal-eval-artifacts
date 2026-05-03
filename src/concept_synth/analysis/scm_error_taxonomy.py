from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    bucket_numeric,
    load_dataset,
    markdown_table,
    mean,
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

SUMMARY_MD = "scm_error_taxonomy_summary.md"
OVERVIEW_CSV = "scm_error_taxonomy_overview.csv"
INVALID_BY_MODEL_CSV = "scm_invalid_buckets_by_model.csv"
SPLIT_BY_MODEL_CSV = "scm_split_buckets_by_model.csv"
SLICE_BY_MODEL_CSV = "scm_error_taxonomy_by_slice.csv"
WORLD_MODE_CSV = "scm_world_errors_by_mode.csv"
WORLD_NOVELTY_CSV = "scm_world_errors_by_novelty.csv"
DOMINANT_VAR_CSV = "scm_dominant_wrong_variables.csv"
MANIFEST_JSON = "scm_error_taxonomy_manifest.json"


def _share_rows(rows: Iterable[Dict[str, Any]], *, group_key: str, bucket_key: str, count_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")][str(row.get(bucket_key) or "missing")] += int(row.get(count_key) or 0)
    out: List[Dict[str, Any]] = []
    for group in sorted(grouped.keys()):
        total = sum(grouped[group].values())
        for bucket, count in sorted(grouped[group].items(), key=lambda item: (-item[1], item[0])):
            out.append(
                {
                    group_key: group,
                    bucket_key: bucket,
                    "count": count,
                    "share": rate(count, total),
                    "group_total": total,
                }
            )
    return out


def _dominant_bucket(rows: Sequence[Dict[str, Any]], bucket_key: str) -> Optional[str]:
    if not rows:
        return None
    best = max(rows, key=lambda row: (int(row.get("count") or 0), str(row.get(bucket_key) or "")))
    return str(best.get(bucket_key)) if best else None


def _world_mode_rows(world_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in world_records:
        grouped[(
            str(row.get("model") or "unknown"),
            str(row.get("split") or "unknown"),
            str(row.get("intervention_mode") or "unknown"),
        )].append(row)

    out: List[Dict[str, Any]] = []
    for (model, split, mode), group in sorted(grouped.items()):
        exact = sum(1 for row in group if row.get("exact") is True)
        scored = [int(row.get("scored_cells") or 0) for row in group]
        wrong = [int(row.get("wrong_cells") or 0) for row in group]
        under = sum(int(row.get("under_count") or 0) for row in group)
        over = sum(int(row.get("over_count") or 0) for row in group)
        total_scored = sum(scored)
        total_wrong = sum(wrong)
        out.append(
            {
                "model": model,
                "split": split,
                "intervention_mode": mode,
                "worlds": len(group),
                "exact_rate": rate(exact, len(group)),
                "mean_accuracy": mean((float(row.get("accuracy") or 0.0) for row in group), default=0.0),
                "mean_wrong_cells": mean(wrong, default=0.0),
                "wrong_cell_rate": rate(total_wrong, total_scored),
                "under_share": rate(under, under + over),
                "over_share": rate(over, under + over),
            }
        )
    return out


def _world_novelty_rows(world_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in world_records:
        if row.get("split") != "heldout":
            continue
        novelty_bin = bucket_numeric(row.get("novelty_score"), [float("-inf"), 0.2, 0.4, 0.6, 0.8, float("inf")])
        grouped[(
            str(row.get("model") or "unknown"),
            str(row.get("split") or "heldout"),
            novelty_bin,
        )].append(row)

    out: List[Dict[str, Any]] = []
    for (model, split, novelty_bin), group in sorted(grouped.items()):
        exact = sum(1 for row in group if row.get("exact") is True)
        total_scored = sum(int(row.get("scored_cells") or 0) for row in group)
        total_wrong = sum(int(row.get("wrong_cells") or 0) for row in group)
        out.append(
            {
                "model": model,
                "split": split,
                "novelty_bin": novelty_bin,
                "worlds": len(group),
                "exact_rate": rate(exact, len(group)),
                "mean_accuracy": mean((float(row.get("accuracy") or 0.0) for row in group), default=0.0),
                "wrong_cell_rate": rate(total_wrong, total_scored),
                "mean_target_count": mean((float(row.get("intervention_target_count") or 0.0) for row in group), default=0.0),
            }
        )
    return out


def _dominant_variable_rows(instance_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in instance_records:
        if not row.get("valid"):
            continue
        for split in ("train", "heldout"):
            value = row.get(f"replay_{split}_dominant_wrong_variable")
            if not value:
                continue
            counts[(str(row.get("model") or "unknown"), split, str(value))] += 1
    by_group: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    for (model, split, variable), count in counts.items():
        by_group[(model, split)].append((variable, count))
    for (model, split), items in sorted(by_group.items()):
        total = sum(count for _variable, count in items)
        for variable, count in sorted(items, key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "variable": variable,
                    "count": count,
                    "share": rate(count, total),
                    "group_total": total,
                }
            )
    return rows


def analyze_error_taxonomy(
    instance_records: Sequence[Dict[str, Any]],
    world_records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_slice_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_records:
        model = str(row.get("model") or "unknown")
        by_model[model].append(row)
        by_slice_model[(str(row.get("slice") or "unknown"), model)].append(row)

    overview_rows: List[Dict[str, Any]] = []
    slice_rows: List[Dict[str, Any]] = []
    for model in sorted(by_model.keys()):
        group = by_model[model]
        total = len(group)
        valid = sum(1 for row in group if row.get("valid") is True)
        correct = sum(1 for row in group if row.get("correct") is True)
        train_exact = sum(1 for row in group if row.get("train_exact") is True)
        heldout_exact = sum(1 for row in group if row.get("heldout_exact") is True)
        invalid_rows = [row for row in group if row.get("valid") is not True]
        split_rows = [row for row in group if row.get("valid") is True]
        invalid_bucket = _dominant_bucket(
            _share_rows(
                ({"model": model, "invalid_bucket": row.get("invalid_bucket") or "missing", "count": 1} for row in invalid_rows),
                group_key="model",
                bucket_key="invalid_bucket",
                count_key="count",
            ),
            "invalid_bucket",
        )
        split_bucket = _dominant_bucket(
            _share_rows(
                ({"model": model, "split_error_bucket": row.get("split_error_bucket") or "missing", "count": 1} for row in split_rows),
                group_key="model",
                bucket_key="split_error_bucket",
                count_key="count",
            ),
            "split_error_bucket",
        )
        overview_rows.append(
            {
                "model": model,
                "n": total,
                "valid_rate": rate(valid, total),
                "correct_rate": rate(correct, total),
                "train_exact_rate": rate(train_exact, total),
                "heldout_exact_rate": rate(heldout_exact, total),
                "invalid_rate": rate(total - valid, total),
                "heldout_only_error_rate": rate(sum(1 for row in group if row.get("split_error_bucket") == "heldout_only_error"), total),
                "train_only_error_rate": rate(sum(1 for row in group if row.get("split_error_bucket") == "train_only_error"), total),
                "train_and_heldout_error_rate": rate(sum(1 for row in group if row.get("split_error_bucket") == "train_and_heldout_error"), total),
                "top_invalid_bucket": invalid_bucket,
                "top_split_bucket": split_bucket,
            }
        )

    for (slice_name, model), group in sorted(by_slice_model.items()):
        total = len(group)
        slice_rows.append(
            {
                "slice": slice_name,
                "model": model,
                "n": total,
                "valid_rate": rate(sum(1 for row in group if row.get("valid") is True), total),
                "correct_rate": rate(sum(1 for row in group if row.get("correct") is True), total),
                "train_exact_rate": rate(sum(1 for row in group if row.get("train_exact") is True), total),
                "heldout_exact_rate": rate(sum(1 for row in group if row.get("heldout_exact") is True), total),
                "mean_train_accuracy": mean((float(row.get("train_accuracy") or 0.0) for row in group), default=0.0),
                "mean_heldout_accuracy": mean((float(row.get("heldout_accuracy") or 0.0) for row in group), default=0.0),
                "dominant_split_bucket": _dominant_bucket(
                    _share_rows(
                        ({"slice": f"{slice_name}:{model}", "split_error_bucket": row.get("split_error_bucket") or "missing", "count": 1} for row in group),
                        group_key="slice",
                        bucket_key="split_error_bucket",
                        count_key="count",
                    ),
                    "split_error_bucket",
                ),
            }
        )

    invalid_rows = _share_rows(
        ({"model": str(row.get("model") or "unknown"), "invalid_bucket": row.get("invalid_bucket") or "missing", "count": 1} for row in instance_records if row.get("valid") is not True),
        group_key="model",
        bucket_key="invalid_bucket",
        count_key="count",
    )
    split_rows = _share_rows(
        ({"model": str(row.get("model") or "unknown"), "split_error_bucket": row.get("split_error_bucket") or "missing", "count": 1} for row in instance_records),
        group_key="model",
        bucket_key="split_error_bucket",
        count_key="count",
    )

    return {
        "overview_rows": overview_rows,
        "invalid_rows": invalid_rows,
        "split_rows": split_rows,
        "slice_rows": slice_rows,
        "world_mode_rows": _world_mode_rows(world_records),
        "world_novelty_rows": _world_novelty_rows(world_records),
        "dominant_variable_rows": _dominant_variable_rows(instance_records),
    }


def build_error_taxonomy_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    overview_rows = artifacts["overview_rows"]
    invalid_rows = artifacts["invalid_rows"]
    split_rows = artifacts["split_rows"]

    invalid_top: Dict[str, str] = {}
    for model in sorted({str(row.get("model")) for row in invalid_rows}):
        candidates = [row for row in invalid_rows if row.get("model") == model]
        if candidates:
            invalid_top[model] = str(max(candidates, key=lambda row: int(row.get("count") or 0)).get("invalid_bucket"))

    split_top: Dict[str, str] = {}
    for model in sorted({str(row.get("model")) for row in split_rows}):
        candidates = [row for row in split_rows if row.get("model") == model]
        if candidates:
            split_top[model] = str(max(candidates, key=lambda row: int(row.get("count") or 0)).get("split_error_bucket"))

    rows = []
    for row in overview_rows:
        model = str(row.get("model") or "unknown")
        rows.append(
            [
                model,
                row.get("n"),
                pct(row.get("valid_rate")),
                pct(row.get("correct_rate")),
                pct(row.get("train_exact_rate")),
                pct(row.get("heldout_exact_rate")),
                invalid_top.get(model, row.get("top_invalid_bucket") or "-"),
                split_top.get(model, row.get("top_split_bucket") or "-"),
            ]
        )

    lines = [
        "SCM Error Taxonomy",
        "",
        markdown_table(
            ["Model", "N", "Valid", "Correct", "TrainExact", "HeldoutExact", "Top invalid", "Top split bucket"],
            rows,
        ) if rows else "No instance records available.",
        "",
        "Interpretation notes:",
        "- `heldout_only_error` means the model fits train worlds exactly but misses heldout worlds.",
        "- `train_only_error` means heldout is exact despite train mistakes, usually a sign of localized train misspecification.",
        "- World-level CSVs break mistakes down by intervention mode and heldout novelty bands.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_error_taxonomy_artifacts(
    artifacts: Dict[str, List[Dict[str, Any]]],
    outdir: str | Path,
) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "overview_csv": str(out / OVERVIEW_CSV),
        "invalid_csv": str(out / INVALID_BY_MODEL_CSV),
        "split_csv": str(out / SPLIT_BY_MODEL_CSV),
        "slice_csv": str(out / SLICE_BY_MODEL_CSV),
        "world_mode_csv": str(out / WORLD_MODE_CSV),
        "world_novelty_csv": str(out / WORLD_NOVELTY_CSV),
        "dominant_variable_csv": str(out / DOMINANT_VAR_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["overview_rows"], paths["overview_csv"])
    write_csv(artifacts["invalid_rows"], paths["invalid_csv"])
    write_csv(artifacts["split_rows"], paths["split_csv"])
    write_csv(artifacts["slice_rows"], paths["slice_csv"])
    write_csv(artifacts["world_mode_rows"], paths["world_mode_csv"])
    write_csv(artifacts["world_novelty_rows"], paths["world_novelty_csv"])
    write_csv(artifacts["dominant_variable_rows"], paths["dominant_variable_csv"])
    write_markdown(build_error_taxonomy_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "files": paths,
        "counts": {key: len(value) for key, value in artifacts.items()},
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.instance_records and args.world_records:
        return read_jsonl(args.instance_records), read_jsonl(args.world_records)
    if not args.input:
        raise SystemExit("Either --input or both --instance-records and --world-records are required")
    _, problems = load_dataset(args.input)
    extracted = extract_scm_analysis_records(problems, source_name=str(args.input), models=None, family_filter=args.family)
    if args.extract_outdir:
        write_extract_artifacts(extracted, args.extract_outdir, source_name=str(args.input))
    return extracted["instance_records"], extracted["world_records"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze A_SCM error taxonomy from extracted records or benchmark YAML")
    parser.add_argument("--input", help="Benchmark YAML input")
    parser.add_argument("--instance-records", help="scm_instance_records.jsonl path")
    parser.add_argument("--world-records", help="scm_world_records.jsonl path")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter when reading --input")
    parser.add_argument("--extract-outdir", help="Optional directory to write extracted records when reading --input")
    parser.add_argument("--outdir", required=True, help="Directory for taxonomy outputs")
    args = parser.parse_args(argv)

    instance_records, world_records = _load_records(args)
    artifacts = analyze_error_taxonomy(instance_records, world_records)
    paths = write_error_taxonomy_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM error taxonomy to {args.outdir}")
    print(f"  overview: {paths['overview_csv']}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
