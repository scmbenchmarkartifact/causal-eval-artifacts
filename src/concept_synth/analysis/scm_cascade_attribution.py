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
    write_jsonl,
    write_markdown,
)
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)
from concept_synth.analysis.scm_common import load_dataset

SUMMARY_MD = "scm_cascade_attribution_summary.md"
BY_INSTANCE_CSV = "scm_cascade_attribution_by_instance.csv"
BY_MODEL_CSV = "scm_cascade_attribution_by_model.csv"
BY_SLICE_CSV = "scm_cascade_attribution_by_slice.csv"
EXAMPLES_JSONL = "scm_cascade_attribution_examples.jsonl"
MANIFEST_JSON = "scm_cascade_attribution_manifest.json"


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("instance_id") or "unknown"), str(row.get("model") or "unknown")


def _parent_set(row: Dict[str, Any], key: str) -> Set[str]:
    value = row.get(key) or []
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return set()


def _ancestor_closure(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    cache: Dict[str, Set[str]] = {}

    def _dfs(node: str, path: Set[str]) -> Set[str]:
        cached = cache.get(node)
        if cached is not None:
            return set(cached)
        out: Set[str] = set()
        for parent in graph.get(node, set()):
            if parent in path:
                continue
            out.add(parent)
            out.update(_dfs(parent, path | {parent}))
        cache[node] = set(out)
        return out

    return {node: _dfs(node, {node}) for node in graph.keys()}


def _primary_bucket(
    instance_row: Dict[str, Any],
    *,
    local_root_vars: List[str],
    downstream_vars: List[str],
    unexplained_vars: List[str],
    parent_formula_root_vars: List[str],
    parent_mismatch_root_vars: List[str],
) -> str:
    if instance_row.get("valid") is not True:
        return f"invalid:{str(instance_row.get('invalid_bucket') or 'invalid')}"
    if instance_row.get("train_exact") is True and instance_row.get("heldout_exact") is True:
        return "exact"
    if not local_root_vars:
        return "structurally_exact_world_failure"

    if len(local_root_vars) == 1:
        spread = "cascade" if downstream_vars else "localized"
        if parent_formula_root_vars and parent_formula_root_vars[0] == local_root_vars[0]:
            if unexplained_vars:
                return f"one_local_parent_exact_formula_wrong_{spread}_plus_unexplained"
            return f"one_local_parent_exact_formula_wrong_{spread}"
        if parent_mismatch_root_vars and parent_mismatch_root_vars[0] == local_root_vars[0]:
            if unexplained_vars:
                return f"one_local_parent_mismatch_{spread}_plus_unexplained"
            return f"one_local_parent_mismatch_{spread}"
        if unexplained_vars:
            return f"one_local_other_structural_{spread}_plus_unexplained"
        return f"one_local_other_structural_{spread}"

    if downstream_vars:
        return "multi_local_with_cascade"
    if unexplained_vars:
        return "multi_local_plus_unexplained"
    return "multi_local_independent"


def _build_cascade_row(
    instance_row: Dict[str, Any],
    mechanism_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_var = {str(row.get("target_var") or ""): row for row in mechanism_rows}
    structural_wrong_vars = sorted(
        [
            str(row.get("target_var") or "")
            for row in mechanism_rows
            if row.get("expr_exact") is False or row.get("parent_exact") is False
        ],
        key=lambda var: int((by_var.get(var) or {}).get("target_topological_index") or 10**9),
    )
    heldout_wrong_vars = sorted(
        [
            str(row.get("target_var") or "")
            for row in mechanism_rows
            if row.get("heldout_var_accuracy") is not None and float(row.get("heldout_var_accuracy") or 0.0) < 0.999999
        ],
        key=lambda var: int((by_var.get(var) or {}).get("target_topological_index") or 10**9),
    )
    train_wrong_vars = sorted(
        [
            str(row.get("target_var") or "")
            for row in mechanism_rows
            if row.get("train_var_accuracy") is not None and float(row.get("train_var_accuracy") or 0.0) < 0.999999
        ],
        key=lambda var: int((by_var.get(var) or {}).get("target_topological_index") or 10**9),
    )

    graph: Dict[str, Set[str]] = {}
    for row in mechanism_rows:
        var = str(row.get("target_var") or "")
        if not var:
            continue
        graph[var] = _parent_set(row, "pred_parents") | _parent_set(row, "gold_parents")
    ancestors = _ancestor_closure(graph)

    local_root_vars: List[str] = []
    for var in structural_wrong_vars:
        wrong_ancestors = [ancestor for ancestor in structural_wrong_vars if ancestor != var and ancestor in ancestors.get(var, set())]
        if not wrong_ancestors:
            local_root_vars.append(var)

    downstream_vars: List[str] = []
    unexplained_vars: List[str] = []
    for var in heldout_wrong_vars:
        if var in structural_wrong_vars:
            continue
        triggering_roots = [root for root in local_root_vars if root in ancestors.get(var, set())]
        if triggering_roots:
            downstream_vars.append(var)
        else:
            unexplained_vars.append(var)

    parent_formula_root_vars = [
        var
        for var in local_root_vars
        if (by_var.get(var) or {}).get("parent_exact") is True and (by_var.get(var) or {}).get("expr_exact") is False
    ]
    parent_mismatch_root_vars = [
        var
        for var in local_root_vars
        if (by_var.get(var) or {}).get("parent_exact") is False
    ]

    heldout_worlds = [row for row in world_rows if row.get("split") == "heldout"]
    wrong_heldout_worlds = [row for row in heldout_worlds if row.get("exact") is not True]
    mode_counts: Dict[str, int] = defaultdict(int)
    for row in wrong_heldout_worlds:
        mode_counts[str(row.get("intervention_mode") or "unknown")] += 1
    dominant_mode = None
    if mode_counts:
        dominant_mode = max(mode_counts.items(), key=lambda item: (item[1], item[0]))[0]

    first_local_root = None
    if local_root_vars:
        first_local_root = local_root_vars[0]

    first_heldout_wrong = None
    if heldout_wrong_vars:
        first_heldout_wrong = heldout_wrong_vars[0]

    primary_bucket = _primary_bucket(
        instance_row,
        local_root_vars=local_root_vars,
        downstream_vars=downstream_vars,
        unexplained_vars=unexplained_vars,
        parent_formula_root_vars=parent_formula_root_vars,
        parent_mismatch_root_vars=parent_mismatch_root_vars,
    )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": str(instance_row.get("instance_id") or "unknown"),
        "model": str(instance_row.get("model") or "unknown"),
        "family": str(instance_row.get("family") or "unknown"),
        "slice": str(instance_row.get("slice") or "unknown"),
        "difficulty": str(instance_row.get("difficulty") or "unknown"),
        "valid": instance_row.get("valid"),
        "train_exact": instance_row.get("train_exact"),
        "heldout_exact": instance_row.get("heldout_exact"),
        "split_error_bucket": instance_row.get("split_error_bucket"),
        "invalid_bucket": instance_row.get("invalid_bucket"),
        "primary_cascade_bucket": primary_bucket,
        "structural_wrong_count": len(structural_wrong_vars),
        "local_root_count": len(local_root_vars),
        "downstream_propagated_count": len(downstream_vars),
        "unexplained_wrong_count": len(unexplained_vars),
        "parent_exact_formula_wrong_root_count": len(parent_formula_root_vars),
        "parent_mismatch_root_count": len(parent_mismatch_root_vars),
        "heldout_wrong_mechanism_count": len(heldout_wrong_vars),
        "train_wrong_mechanism_count": len(train_wrong_vars),
        "first_local_root": first_local_root,
        "first_local_root_topological_index": (
            (by_var.get(first_local_root) or {}).get("target_topological_index")
            if first_local_root is not None
            else None
        ),
        "first_heldout_wrong_variable": first_heldout_wrong,
        "first_heldout_wrong_topological_index": (
            (by_var.get(first_heldout_wrong) or {}).get("target_topological_index")
            if first_heldout_wrong is not None
            else None
        ),
        "dominant_wrong_heldout_mode": dominant_mode,
        "local_root_vars": local_root_vars,
        "downstream_propagated_vars": downstream_vars,
        "unexplained_wrong_vars": unexplained_vars,
        "parent_exact_formula_wrong_roots": parent_formula_root_vars,
        "parent_mismatch_roots": parent_mismatch_root_vars,
    }


def analyze_cascade_attribution(
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

    instance_rows: List[Dict[str, Any]] = []
    for row in instance_records:
        key = _group_key(row)
        instance_rows.append(_build_cascade_row(row, mech_by_key.get(key, []), world_by_key.get(key, [])))

    by_model_rows: List[Dict[str, Any]] = []
    by_slice_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
        grouped_slice[(str(row.get("slice") or "unknown"), str(row.get("model") or "unknown"))].append(row)

    for model, group in sorted(grouped_model.items()):
        heldout_only = [row for row in group if row.get("split_error_bucket") == "heldout_only_error"]
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in heldout_only:
            bucket_counts[str(row.get("primary_cascade_bucket") or "unknown")] += 1
        top_bucket = None
        top_share = None
        if bucket_counts and heldout_only:
            top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
            top_share = rate(top_count, len(heldout_only))
        by_model_rows.append(
            {
                "model": model,
                "n": len(group),
                "heldout_only_instances": len(heldout_only),
                "heldout_only_share": rate(len(heldout_only), len(group)),
                "top_cascade_bucket": top_bucket,
                "top_cascade_bucket_share": top_share,
                "single_local_root_share": rate(
                    sum(1 for row in heldout_only if int(row.get("local_root_count") or 0) == 1),
                    len(heldout_only),
                ),
                "downstream_propagation_share": rate(
                    sum(1 for row in heldout_only if int(row.get("downstream_propagated_count") or 0) > 0),
                    len(heldout_only),
                ),
                "parent_exact_formula_wrong_root_share": rate(
                    sum(1 for row in heldout_only if int(row.get("parent_exact_formula_wrong_root_count") or 0) > 0),
                    len(heldout_only),
                ),
                "unexplained_share": rate(
                    sum(1 for row in heldout_only if int(row.get("unexplained_wrong_count") or 0) > 0),
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
            bucket_counts[str(row.get("primary_cascade_bucket") or "unknown")] += 1
        top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
        by_slice_rows.append(
            {
                "slice": slice_name,
                "model": model,
                "heldout_only_instances": len(heldout_only),
                "top_cascade_bucket": top_bucket,
                "top_cascade_bucket_share": rate(top_count, len(heldout_only)),
                "downstream_propagation_share": rate(
                    sum(1 for row in heldout_only if int(row.get("downstream_propagated_count") or 0) > 0),
                    len(heldout_only),
                ),
                "unexplained_share": rate(
                    sum(1 for row in heldout_only if int(row.get("unexplained_wrong_count") or 0) > 0),
                    len(heldout_only),
                ),
            }
        )

    examples = [
        row
        for row in instance_rows
        if row.get("split_error_bucket") == "heldout_only_error" or str(row.get("primary_cascade_bucket") or "").startswith("invalid:")
    ]

    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
        "example_rows": examples,
    }


def build_cascade_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                row.get("heldout_only_instances"),
                pct(row.get("heldout_only_share")),
                row.get("top_cascade_bucket") or "-",
                pct(row.get("top_cascade_bucket_share")),
                pct(row.get("single_local_root_share")),
                pct(row.get("downstream_propagation_share")),
                pct(row.get("parent_exact_formula_wrong_root_share")),
                pct(row.get("unexplained_share")),
            ]
        )
    slice_rows = []
    for row in sorted(
        artifacts["by_slice_rows"],
        key=lambda item: (
            -int(item.get("heldout_only_instances") or 0),
            str(item.get("slice") or ""),
            str(item.get("model") or ""),
        ),
    )[:12]:
        slice_rows.append(
            [
                row.get("slice"),
                row.get("model"),
                row.get("heldout_only_instances"),
                row.get("top_cascade_bucket") or "-",
                pct(row.get("top_cascade_bucket_share")),
                pct(row.get("downstream_propagation_share")),
                pct(row.get("unexplained_share")),
            ]
        )
    lines = [
        "SCM Cascade Attribution",
        "",
        "This report refines heldout root-cause attribution by distinguishing local structural mistakes from downstream propagation.",
        "",
        "By model:",
        markdown_table(
            [
                "Model",
                "N",
                "Heldout-only",
                "Heldout-only share",
                "Top bucket",
                "Top bucket share",
                "Single local root",
                "Cascade",
                "Parent-exact formula root",
                "Unexplained",
            ],
            model_rows,
        ) if model_rows else "No cascade rows.",
        "",
        "Most heldout-heavy slice/model pairs:",
        markdown_table(
            ["Slice", "Model", "Heldout-only", "Top bucket", "Top share", "Cascade", "Unexplained"],
            slice_rows,
        ) if slice_rows else "No heldout-only slice rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_cascade_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
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
    write_markdown(build_cascade_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_records(instance_path: str, mechanism_path: str, world_path: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "instance_records": read_jsonl(instance_path),
        "mechanism_records": read_jsonl(mechanism_path),
        "world_records": read_jsonl(world_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cascade-style heldout failures for A_SCM records")
    parser.add_argument("--instance-records", help="Path to scm_instance_records.jsonl")
    parser.add_argument("--mechanism-records", help="Path to scm_mechanism_records.jsonl")
    parser.add_argument("--world-records", help="Path to scm_world_records.jsonl")
    parser.add_argument("--input", help="Optional benchmark YAML; if set, records are extracted first")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    if args.input:
        _, problems = load_dataset(args.input)
        extracted = extract_scm_analysis_records(problems, source_name=str(args.input))
        write_extract_artifacts(extracted, Path(args.outdir) / "extract", source_name=str(args.input))
        artifacts = analyze_cascade_attribution(
            extracted["instance_records"],
            extracted["mechanism_records"],
            extracted["world_records"],
        )
    else:
        if not (args.instance_records and args.mechanism_records and args.world_records):
            raise SystemExit("Provide either --input or all of --instance-records/--mechanism-records/--world-records")
        loaded = _load_records(args.instance_records, args.mechanism_records, args.world_records)
        artifacts = analyze_cascade_attribution(
            loaded["instance_records"],
            loaded["mechanism_records"],
            loaded["world_records"],
        )
    paths = write_cascade_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM cascade attribution artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
