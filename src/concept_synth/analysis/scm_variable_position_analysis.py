from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    markdown_table,
    mean,
    pct,
    rate,
    read_jsonl,
    stable_json_dumps,
    write_csv,
    write_markdown,
)
from concept_synth.analysis.scm_common import load_dataset
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)

SUMMARY_MD = "scm_variable_position_summary.md"
BY_INDEX_CSV = "scm_variable_position_by_index.csv"
BY_FIRST_WRONG_CSV = "scm_variable_position_first_wrong.csv"
MANIFEST_JSON = "scm_variable_position_manifest.json"


def analyze_variable_position(
    mechanism_records: Sequence[Dict[str, Any]],
    cascade_rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped_index: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in mechanism_records:
        idx = row.get("target_endogenous_index")
        if idx is None:
            continue
        grouped_index[(str(row.get("model") or "unknown"), int(idx))].append(row)

    index_rows: List[Dict[str, Any]] = []
    for (model, idx), group in sorted(grouped_index.items()):
        total = len(group)
        index_rows.append(
            {
                "model": model,
                "target_endogenous_index": idx,
                "n": total,
                "parent_exact_rate": rate(sum(1 for row in group if row.get("parent_exact") is True), total),
                "expr_exact_rate": rate(sum(1 for row in group if row.get("expr_exact") is True), total),
                "mean_train_var_accuracy": mean((float(row.get("train_var_accuracy") or 0.0) for row in group), default=0.0),
                "mean_heldout_var_accuracy": mean((float(row.get("heldout_var_accuracy") or 0.0) for row in group), default=0.0),
                "wrong_share": rate(sum(1 for row in group if float(row.get("heldout_var_accuracy") or 0.0) < 0.999999), total),
            }
        )

    grouped_first_wrong: Dict[Tuple[str, int], int] = defaultdict(int)
    totals: Dict[str, int] = defaultdict(int)
    for row in cascade_rows:
        if row.get("split_error_bucket") != "heldout_only_error":
            continue
        model = str(row.get("model") or "unknown")
        idx = row.get("first_heldout_wrong_topological_index")
        if idx is None:
            continue
        grouped_first_wrong[(model, int(idx))] += 1
        totals[model] += 1

    first_wrong_rows: List[Dict[str, Any]] = []
    for (model, idx), count in sorted(grouped_first_wrong.items()):
        first_wrong_rows.append(
            {
                "model": model,
                "first_wrong_topological_index": idx,
                "count": count,
                "share": rate(count, totals.get(model, 0)),
            }
        )

    return {
        "index_rows": index_rows,
        "first_wrong_rows": first_wrong_rows,
    }


def build_variable_position_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    hardest_rows = []
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for row in artifacts["index_rows"]:
        model = str(row.get("model") or "unknown")
        current = best_by_model.get(model)
        if current is None or float(row.get("wrong_share") or 0.0) > float(current.get("wrong_share") or 0.0):
            best_by_model[model] = row
    for model, row in sorted(best_by_model.items()):
        hardest_rows.append(
            [
                model,
                row.get("target_endogenous_index"),
                row.get("n"),
                pct(row.get("parent_exact_rate")),
                pct(row.get("expr_exact_rate")),
                f"{float(row.get('mean_heldout_var_accuracy') or 0.0):.3f}",
                pct(row.get("wrong_share")),
            ]
        )

    first_wrong_rows = []
    for row in artifacts["first_wrong_rows"][:12]:
        first_wrong_rows.append(
            [
                row.get("model"),
                row.get("first_wrong_topological_index"),
                row.get("count"),
                pct(row.get("share")),
            ]
        )

    lines = [
        "SCM Variable Position Analysis",
        "",
        "This report measures where along the endogenous order models begin to fail.",
        "",
        "Hardest endogenous position by model:",
        markdown_table(
            ["Model", "Endogenous index", "N", "Parent exact", "Expr exact", "Heldout acc", "Wrong share"],
            hardest_rows,
        ) if hardest_rows else "No position rows.",
        "",
        "First heldout wrong position distribution:",
        markdown_table(["Model", "First wrong topo idx", "Count", "Share"], first_wrong_rows) if first_wrong_rows else "No first-wrong rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_variable_position_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_index_csv": str(out / BY_INDEX_CSV),
        "by_first_wrong_csv": str(out / BY_FIRST_WRONG_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["index_rows"], paths["by_index_csv"])
    write_csv(artifacts["first_wrong_rows"], paths["by_first_wrong_csv"])
    write_markdown(build_variable_position_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze A_SCM errors by endogenous-variable position")
    parser.add_argument("--mechanism-records", help="Path to scm_mechanism_records.jsonl")
    parser.add_argument("--cascade-records", help="Path to scm_cascade_attribution_by_instance.csv is not supported; use JSONL extract or --input")
    parser.add_argument("--input", help="Optional benchmark YAML; if set, records are extracted first")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    if args.input:
        _, problems = load_dataset(args.input)
        extracted = extract_scm_analysis_records(problems, source_name=str(args.input))
        write_extract_artifacts(extracted, Path(args.outdir) / "extract", source_name=str(args.input))
        from concept_synth.analysis.scm_cascade_attribution import analyze_cascade_attribution

        cascade = analyze_cascade_attribution(
            extracted["instance_records"],
            extracted["mechanism_records"],
            extracted["world_records"],
        )
        artifacts = analyze_variable_position(extracted["mechanism_records"], cascade["instance_rows"])
    else:
        if not args.mechanism_records:
            raise SystemExit("Provide either --input or --mechanism-records")
        if args.cascade_records:
            raise SystemExit("For now, use --input to generate cascade rows alongside mechanism rows")
        artifacts = analyze_variable_position(read_jsonl(args.mechanism_records), [])
    paths = write_variable_position_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM variable-position artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
