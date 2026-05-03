from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
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
from concept_synth.analysis.scm_common import load_dataset
from concept_synth.causal_reasoning.mechanism_dsl import MechanismNode, parse_mechanism

SUMMARY_MD = "scm_parent_formula_taxonomy_summary.md"
BY_MODEL_CSV = "scm_parent_formula_taxonomy_by_model.csv"
BY_SLICE_CSV = "scm_parent_formula_taxonomy_by_slice.csv"
BY_MECH_CSV = "scm_parent_formula_taxonomy_by_mechanism.csv"
MANIFEST_JSON = "scm_parent_formula_taxonomy_manifest.json"


def _parent_set(row: Dict[str, Any], key: str) -> Set[str]:
    value = row.get(key) or []
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return set()


def _parent_relation_bucket(row: Dict[str, Any]) -> str:
    gold = _parent_set(row, "gold_parents")
    pred = _parent_set(row, "pred_parents")
    if gold == pred:
        return "exact"
    if pred and pred < gold:
        return "subset"
    if gold and gold < pred:
        return "superset"
    if gold & pred:
        return "overlap"
    return "disjoint"


def _parse_node(expr: Optional[str], allowed_vars: Set[str]) -> Optional[MechanismNode]:
    if not expr:
        return None
    try:
        return parse_mechanism(
            str(expr),
            allowed_operators={"not", "and", "or", "xor", "iff", "if"},
            allowed_variables=allowed_vars,
            allow_constants=True,
        )
    except Exception:
        return None


def _is_negation_of(left: Optional[MechanismNode], right: Optional[MechanismNode]) -> bool:
    if left is None or right is None:
        return False
    return left.kind == "op" and left.value == "not" and len(left.args) == 1 and left.args[0] == right


def _formula_bucket(row: Dict[str, Any]) -> str:
    relation = _parent_relation_bucket(row)
    if relation != "exact":
        return f"parent_{relation}"
    if row.get("expr_exact") is True:
        return "formula_exact"
    if row.get("expr_canonical_exact") is True:
        return "canonical_surface_variant"

    allowed_vars = _parent_set(row, "gold_parents") | _parent_set(row, "pred_parents")
    pred_node = _parse_node(row.get("pred_expr"), allowed_vars)
    gold_node = _parse_node(row.get("gold_expr"), allowed_vars)
    if _is_negation_of(pred_node, gold_node) or _is_negation_of(gold_node, pred_node):
        return "negation_toggle"

    pred_profile = str(row.get("pred_operator_profile") or "none")
    gold_profile = str(row.get("gold_operator_profile") or "none")
    pred_ast = row.get("pred_ast")
    gold_ast = row.get("gold_ast")
    if pred_profile == gold_profile and pred_ast is not None and gold_ast is not None:
        if int(pred_ast) < int(gold_ast):
            return "same_profile_under_composed"
        if int(pred_ast) > int(gold_ast):
            return "same_profile_over_composed"
        return "same_profile_recomposition"
    if pred_ast is not None and gold_ast is not None:
        if int(pred_ast) < int(gold_ast):
            return "operator_substitution_under_composed"
        if int(pred_ast) > int(gold_ast):
            return "operator_substitution_over_composed"
    return "operator_substitution"


def analyze_parent_formula_taxonomy(
    mechanism_records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    mech_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in mechanism_records:
        relation = _parent_relation_bucket(row)
        bucket = _formula_bucket(row)
        out = {
            "analysis_version": ANALYSIS_VERSION,
            "instance_id": str(row.get("instance_id") or "unknown"),
            "model": str(row.get("model") or "unknown"),
            "slice": str(row.get("slice") or "unknown"),
            "family": str(row.get("family") or "unknown"),
            "difficulty": str(row.get("difficulty") or "unknown"),
            "target_var": str(row.get("target_var") or ""),
            "target_topological_index": row.get("target_topological_index"),
            "heldout_var_accuracy": row.get("heldout_var_accuracy"),
            "train_var_accuracy": row.get("train_var_accuracy"),
            "parent_relation_bucket": relation,
            "formula_taxonomy_bucket": bucket,
            "expr_exact": row.get("expr_exact"),
            "expr_canonical_exact": row.get("expr_canonical_exact"),
            "parent_exact": row.get("parent_exact"),
            "pred_operator_profile": row.get("pred_operator_profile"),
            "gold_operator_profile": row.get("gold_operator_profile"),
            "ast_gap": row.get("ast_gap"),
            "parent_gap": row.get("parent_gap"),
            "split_error_bucket": row.get("split_error_bucket"),
        }
        mech_rows.append(out)
        grouped_model[out["model"]].append(out)
        grouped_slice[(out["slice"], out["model"])].append(out)

    def _aggregate(group: Sequence[Dict[str, Any]], *, model: str, slice_name: Optional[str] = None) -> Dict[str, Any]:
        total = len(group)
        parent_exact_wrong = [row for row in group if row.get("parent_exact") is True and row.get("expr_exact") is False]
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in parent_exact_wrong:
            bucket_counts[str(row.get("formula_taxonomy_bucket") or "unknown")] += 1
        top_bucket = None
        top_share = None
        if bucket_counts and parent_exact_wrong:
            top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
            top_share = rate(top_count, len(parent_exact_wrong))
        out = {
            "model": model,
            "n": total,
            "parent_exact_rate": rate(sum(1 for row in group if row.get("parent_exact") is True), total),
            "expr_exact_rate": rate(sum(1 for row in group if row.get("expr_exact") is True), total),
            "canonical_exact_rate": rate(sum(1 for row in group if row.get("expr_canonical_exact") is True), total),
            "parent_exact_formula_wrong_n": len(parent_exact_wrong),
            "top_parent_exact_formula_bucket": top_bucket,
            "top_parent_exact_formula_bucket_share": top_share,
            "subset_parent_rate": rate(sum(1 for row in group if row.get("parent_relation_bucket") == "subset"), total),
            "superset_parent_rate": rate(sum(1 for row in group if row.get("parent_relation_bucket") == "superset"), total),
            "overlap_parent_rate": rate(sum(1 for row in group if row.get("parent_relation_bucket") == "overlap"), total),
            "disjoint_parent_rate": rate(sum(1 for row in group if row.get("parent_relation_bucket") == "disjoint"), total),
        }
        if slice_name is not None:
            out["slice"] = slice_name
        return out

    by_model_rows = [_aggregate(group, model=model) for model, group in sorted(grouped_model.items())]
    by_slice_rows = [
        _aggregate(group, model=model, slice_name=slice_name)
        for (slice_name, model), group in sorted(grouped_slice.items())
    ]

    return {
        "mechanism_rows": mech_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
    }


def build_parent_formula_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                pct(row.get("parent_exact_rate")),
                pct(row.get("expr_exact_rate")),
                pct(row.get("canonical_exact_rate")),
                row.get("parent_exact_formula_wrong_n"),
                row.get("top_parent_exact_formula_bucket") or "-",
                pct(row.get("top_parent_exact_formula_bucket_share")),
                pct(row.get("subset_parent_rate")),
                pct(row.get("superset_parent_rate")),
                pct(row.get("overlap_parent_rate")),
                pct(row.get("disjoint_parent_rate")),
            ]
        )
    lines = [
        "SCM Parent And Formula Taxonomy",
        "",
        "This report separates parent-set recovery from same-parent formula induction errors.",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "Parent exact",
                "Expr exact",
                "Canonical exact",
                "Parent exact but formula wrong",
                "Top formula bucket",
                "Top share",
                "Parent subset",
                "Parent superset",
                "Parent overlap",
                "Parent disjoint",
            ],
            model_rows,
        ) if model_rows else "No mechanism rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_parent_formula_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_mechanism_csv": str(out / BY_MECH_CSV),
        "by_model_csv": str(out / BY_MODEL_CSV),
        "by_slice_csv": str(out / BY_SLICE_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["mechanism_rows"], paths["by_mechanism_csv"])
    write_csv(artifacts["by_model_rows"], paths["by_model_csv"])
    write_csv(artifacts["by_slice_rows"], paths["by_slice_csv"])
    write_markdown(build_parent_formula_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze parent-set versus formula-level mechanism errors for A_SCM")
    parser.add_argument("--mechanism-records", help="Path to scm_mechanism_records.jsonl")
    parser.add_argument("--input", help="Optional benchmark YAML; if set, records are extracted first")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    if args.input:
        _, problems = load_dataset(args.input)
        extracted = extract_scm_analysis_records(problems, source_name=str(args.input))
        write_extract_artifacts(extracted, Path(args.outdir) / "extract", source_name=str(args.input))
        artifacts = analyze_parent_formula_taxonomy(extracted["mechanism_records"])
    else:
        if not args.mechanism_records:
            raise SystemExit("Provide either --input or --mechanism-records")
        artifacts = analyze_parent_formula_taxonomy(read_jsonl(args.mechanism_records))
    paths = write_parent_formula_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM parent/formula taxonomy artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
