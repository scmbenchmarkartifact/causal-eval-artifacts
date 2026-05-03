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

SUMMARY_MD = "scm_structure_breakdown_summary.md"
GOLD_AST_CSV = "scm_structure_by_gold_ast_bin.csv"
PRED_AST_CSV = "scm_structure_by_candidate_ast_bin.csv"
AST_GAP_CSV = "scm_structure_by_ast_gap_bin.csv"
GOLD_OP_CSV = "scm_structure_by_gold_operator_profile.csv"
CANDIDATE_OP_CSV = "scm_structure_by_candidate_operator_profile.csv"
PARENT_REL_CSV = "scm_parent_relation_breakdown.csv"
MECH_AST_CSV = "scm_mechanism_by_gold_ast_bin.csv"
MECH_TEMPLATE_CSV = "scm_mechanism_template_breakdown.csv"
MANIFEST_JSON = "scm_structure_breakdown_manifest.json"


def _metrics(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(group)
    return {
        "n": total,
        "valid_rate": rate(sum(1 for row in group if row.get("valid") is True), total),
        "correct_rate": rate(sum(1 for row in group if row.get("correct") is True), total),
        "train_exact_rate": rate(sum(1 for row in group if row.get("train_exact") is True), total),
        "heldout_exact_rate": rate(sum(1 for row in group if row.get("heldout_exact") is True), total),
        "mean_train_accuracy": mean((float(row.get("train_accuracy") or 0.0) for row in group), default=0.0),
        "mean_heldout_accuracy": mean((float(row.get("heldout_accuracy") or 0.0) for row in group), default=0.0),
    }


def _group_rows(rows: Sequence[Dict[str, Any]], *, key_fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field) if row.get(field) is not None else "missing") for field in key_fields)
        grouped[key].append(row)
    return grouped


def _parent_relation_bucket(pred_parents: Sequence[str], gold_parents: Sequence[str]) -> str:
    pred = set(str(v) for v in pred_parents)
    gold = set(str(v) for v in gold_parents)
    if pred == gold:
        return "exact"
    if pred < gold:
        return "subset"
    if gold < pred:
        return "superset"
    if pred & gold:
        return "overlap"
    return "disjoint"


def _bin_rows(instance_records: Sequence[Dict[str, Any]], field: str, edges: Sequence[float], label: str) -> List[Dict[str, Any]]:
    enriched = []
    for row in instance_records:
        enriched.append({**row, label: bucket_numeric(row.get(field), edges)})
    out: List[Dict[str, Any]] = []
    for (model, bucket), group in sorted(_group_rows(enriched, key_fields=["model", label]).items()):
        payload = {"model": model, label: bucket}
        payload.update(_metrics(group))
        out.append(payload)
    return out


def _profile_rows(instance_records: Sequence[Dict[str, Any]], field: str, output_field: str) -> List[Dict[str, Any]]:
    grouped = _group_rows(instance_records, key_fields=["model", field])
    rows: List[Dict[str, Any]] = []
    for (model, profile), group in sorted(grouped.items()):
        payload = {"model": model, output_field: profile}
        payload.update(_metrics(group))
        rows.append(payload)
    return rows


def _mechanism_rows(mechanism_records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    enriched = []
    for row in mechanism_records:
        enriched.append(
            {
                **row,
                "gold_ast_bin": bucket_numeric(row.get("gold_ast"), [float("-inf"), 4, 6, 8, 10, 12, float("inf")]),
                "pred_ast_bin": bucket_numeric(row.get("pred_ast"), [float("-inf"), 4, 6, 8, 10, 12, float("inf")]),
                "parent_relation": _parent_relation_bucket(row.get("pred_parents") or [], row.get("gold_parents") or []),
            }
        )

    parent_rows: List[Dict[str, Any]] = []
    grouped_parent = _group_rows(enriched, key_fields=["model", "parent_relation"])
    for (model, relation), group in sorted(grouped_parent.items()):
        total = len(group)
        parent_rows.append(
            {
                "model": model,
                "parent_relation": relation,
                "n": total,
                "train_var_accuracy_mean": mean((float(row.get("train_var_accuracy") or 0.0) for row in group), default=0.0),
                "heldout_var_accuracy_mean": mean((float(row.get("heldout_var_accuracy") or 0.0) for row in group), default=0.0),
                "parent_exact_rate": rate(sum(1 for row in group if row.get("parent_exact") is True), total),
            }
        )

    mech_ast_rows: List[Dict[str, Any]] = []
    grouped_ast = _group_rows(enriched, key_fields=["model", "gold_ast_bin"])
    for (model, bucket), group in sorted(grouped_ast.items()):
        total = len(group)
        mech_ast_rows.append(
            {
                "model": model,
                "gold_ast_bin": bucket,
                "n": total,
                "train_var_accuracy_mean": mean((float(row.get("train_var_accuracy") or 0.0) for row in group), default=0.0),
                "heldout_var_accuracy_mean": mean((float(row.get("heldout_var_accuracy") or 0.0) for row in group), default=0.0),
                "ast_gap_mean": mean((float(row.get("ast_gap") or 0.0) for row in group), default=0.0),
                "parent_gap_mean": mean((float(row.get("parent_gap") or 0.0) for row in group), default=0.0),
            }
        )

    template_rows: List[Dict[str, Any]] = []
    grouped_templates = _group_rows(enriched, key_fields=["model", "pred_expr_canonical"])
    for (model, template), group in sorted(grouped_templates.items(), key=lambda item: (-len(item[1]), item[0])):
        if not template or template == "missing":
            continue
        total = len(group)
        template_rows.append(
            {
                "model": model,
                "pred_expr_canonical": template,
                "n": total,
                "train_var_accuracy_mean": mean((float(row.get("train_var_accuracy") or 0.0) for row in group), default=0.0),
                "heldout_var_accuracy_mean": mean((float(row.get("heldout_var_accuracy") or 0.0) for row in group), default=0.0),
                "parent_exact_rate": rate(sum(1 for row in group if row.get("parent_exact") is True), total),
            }
        )

    return {
        "parent_rows": parent_rows,
        "mech_ast_rows": mech_ast_rows,
        "template_rows": template_rows,
    }


def analyze_structure_breakdown(
    instance_records: Sequence[Dict[str, Any]],
    mechanism_records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "gold_ast_rows": _bin_rows(instance_records, "gold_modular_ast", [float("-inf"), 20, 28, 36, 44, 56, float("inf")], "gold_ast_bin"),
        "candidate_ast_rows": _bin_rows(instance_records, "candidate_ast", [float("-inf"), 20, 28, 36, 44, 56, float("inf")], "candidate_ast_bin"),
        "ast_gap_rows": _bin_rows(instance_records, "ast_gap", [float("-inf"), -12, -4, 0, 5, 13, float("inf")], "ast_gap_bin"),
        "gold_operator_rows": _profile_rows(instance_records, "gold_operator_profile", "gold_operator_profile"),
        "candidate_operator_rows": _profile_rows(instance_records, "candidate_operator_profile", "candidate_operator_profile"),
        **_mechanism_rows(mechanism_records),
    }


def build_structure_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    gold_ast_rows = artifacts["gold_ast_rows"]
    parent_rows = artifacts["parent_rows"]

    hardest_by_model: List[List[Any]] = []
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in gold_ast_rows:
        by_model[str(row.get("model") or "unknown")].append(row)
    for model in sorted(by_model.keys()):
        eligible = [row for row in by_model[model] if int(row.get("n") or 0) >= 1]
        if not eligible:
            continue
        hardest = min(eligible, key=lambda row: (float(row.get("heldout_exact_rate") or 0.0), str(row.get("gold_ast_bin") or "")))
        hardest_by_model.append(
            [
                model,
                hardest.get("gold_ast_bin"),
                hardest.get("n"),
                pct(hardest.get("correct_rate")),
                pct(hardest.get("heldout_exact_rate")),
            ]
        )

    parent_top: List[List[Any]] = []
    grouped_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        grouped_parent[str(row.get("model") or "unknown")].append(row)
    for model in sorted(grouped_parent.keys()):
        top = max(grouped_parent[model], key=lambda row: int(row.get("n") or 0))
        parent_top.append(
            [
                model,
                top.get("parent_relation"),
                top.get("n"),
                pct(top.get("parent_exact_rate")),
                f"{float(top.get('heldout_var_accuracy_mean') or 0.0):.3f}",
            ]
        )

    lines = [
        "SCM Structure Breakdown",
        "",
        "Hardest gold AST bins by model:",
        markdown_table(["Model", "Gold AST bin", "N", "Correct", "HeldoutExact"], hardest_by_model) if hardest_by_model else "No rows.",
        "",
        "Most common parent-relation regime by model:",
        markdown_table(["Model", "Parent relation", "N", "Parent exact", "Heldout var acc"], parent_top) if parent_top else "No mechanism rows.",
        "",
        "Interpretation notes:",
        "- Gold AST bins show how performance changes with latent SCM size.",
        "- AST gap bins compare predicted map size against gold modular AST.",
        "- Parent relation buckets help separate missing-parent vs extra-parent failure modes.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_structure_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "gold_ast_csv": str(out / GOLD_AST_CSV),
        "candidate_ast_csv": str(out / PRED_AST_CSV),
        "ast_gap_csv": str(out / AST_GAP_CSV),
        "gold_operator_csv": str(out / GOLD_OP_CSV),
        "candidate_operator_csv": str(out / CANDIDATE_OP_CSV),
        "parent_relation_csv": str(out / PARENT_REL_CSV),
        "mechanism_ast_csv": str(out / MECH_AST_CSV),
        "mechanism_template_csv": str(out / MECH_TEMPLATE_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["gold_ast_rows"], paths["gold_ast_csv"])
    write_csv(artifacts["candidate_ast_rows"], paths["candidate_ast_csv"])
    write_csv(artifacts["ast_gap_rows"], paths["ast_gap_csv"])
    write_csv(artifacts["gold_operator_rows"], paths["gold_operator_csv"])
    write_csv(artifacts["candidate_operator_rows"], paths["candidate_operator_csv"])
    write_csv(artifacts["parent_rows"], paths["parent_relation_csv"])
    write_csv(artifacts["mech_ast_rows"], paths["mechanism_ast_csv"])
    write_csv(artifacts["template_rows"], paths["mechanism_template_csv"])
    write_markdown(build_structure_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "files": paths,
        "counts": {key: len(value) for key, value in artifacts.items()},
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.instance_records and args.mechanism_records:
        return read_jsonl(args.instance_records), read_jsonl(args.mechanism_records)
    if not args.input:
        raise SystemExit("Either --input or both --instance-records and --mechanism-records are required")
    _, problems = load_dataset(args.input)
    extracted = extract_scm_analysis_records(problems, source_name=str(args.input), models=None, family_filter=args.family)
    if args.extract_outdir:
        write_extract_artifacts(extracted, args.extract_outdir, source_name=str(args.input))
    return extracted["instance_records"], extracted["mechanism_records"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze A_SCM performance against latent and predicted structure")
    parser.add_argument("--input", help="Benchmark YAML input")
    parser.add_argument("--instance-records", help="scm_instance_records.jsonl path")
    parser.add_argument("--mechanism-records", help="scm_mechanism_records.jsonl path")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter when reading --input")
    parser.add_argument("--extract-outdir", help="Optional directory to write extracted records when reading --input")
    parser.add_argument("--outdir", required=True, help="Directory for structure outputs")
    args = parser.parse_args(argv)

    instance_records, mechanism_records = _load_records(args)
    artifacts = analyze_structure_breakdown(instance_records, mechanism_records)
    paths = write_structure_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM structure breakdown to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
