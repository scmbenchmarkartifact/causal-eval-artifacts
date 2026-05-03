from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    compute_local_rollout_split_stats,
    difficulty,
    family,
    instance_id,
    is_a_scm_problem,
    load_dataset,
    markdown_table,
    pct,
    rate,
    replay_candidate,
    slice_name,
    split_error_bucket,
    stable_json_dumps,
    task_name,
    write_csv,
    write_jsonl,
    write_markdown,
)
from concept_synth.analysis.scm_repairability import analyze_repairability
from concept_synth.analysis.scm_subset_compare import _parse_models_arg

SUMMARY_MD = "scm_training_equivalence_summary.md"
BY_INSTANCE_CSV = "scm_training_equivalence_by_instance.csv"
BY_MODEL_CSV = "scm_training_equivalence_by_model.csv"
BY_SLICE_CSV = "scm_training_equivalence_by_slice.csv"
EXAMPLES_JSONL = "scm_training_equivalence_examples.jsonl"
MANIFEST_JSON = "scm_training_equivalence_manifest.json"


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("instance_id") or "unknown"), str(row.get("model") or "unknown")


def _bucket_for_row(
    replay,
    *,
    local_root_count: int,
    local_train_equivalent_root_count: int,
    compensated_root_count: int,
    simple_bounded_fix: bool,
) -> str:
    if replay.candidate_parse.replayable is False:
        return f"invalid:{str(replay.candidate_parse.invalid_bucket or 'invalid')}"
    if replay.evaluation.get("trainExact") is True and replay.evaluation.get("heldoutExact") is True:
        return "exact"
    if replay.evaluation.get("trainExact") is not True:
        return "train_nonexact"

    if local_train_equivalent_root_count > 0:
        if simple_bounded_fix:
            return "train_exact_local_ambiguity_with_simple_fix"
        return "train_exact_local_ambiguity"
    if compensated_root_count > 0:
        if simple_bounded_fix:
            return "train_exact_compensated_fit_with_simple_fix"
        return "train_exact_compensated_fit"
    if local_root_count <= 0:
        if simple_bounded_fix:
            return "train_exact_bounded_ambiguity_fix_only"
        return "train_exact_not_shown_ambiguous_under_cap"
    if simple_bounded_fix:
        return "train_exact_bounded_ambiguity_fix_only"
    return "train_exact_not_shown_ambiguous_under_cap"


def _analyze_result(
    problem: Dict[str, Any],
    llm_result: Dict[str, Any],
    *,
    cascade_row: Optional[Dict[str, Any]],
    repair_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    replay = replay_candidate(problem, llm_result)
    parsed = replay.candidate_parse
    if not parsed.replayable:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "instance_id": instance_id(problem),
            "model": str(llm_result.get("model") or "unknown"),
            "family": family(problem),
            "slice": slice_name(problem),
            "difficulty": difficulty(problem),
            "task_name": task_name(problem),
            "valid": False,
            "train_exact": replay.evaluation.get("trainExact"),
            "heldout_exact": replay.evaluation.get("heldoutExact"),
            "split_error_bucket": split_error_bucket(replay.evaluation),
            "invalid_bucket": parsed.invalid_bucket,
            "training_equivalence_bucket": f"invalid:{str(parsed.invalid_bucket or 'invalid')}",
        }

    train_local = compute_local_rollout_split_stats(problem, parsed, split_name="train")
    local_root_vars = [str(var) for var in ((cascade_row or {}).get("local_root_vars") or [])]
    per_var = train_local.get("per_variable") or {}
    local_train_equivalent_root_vars = [
        var
        for var in local_root_vars
        if bool((per_var.get(var) or {}).get("local_exact"))
    ]
    compensated_root_vars = [
        var
        for var in local_root_vars
        if int((per_var.get(var) or {}).get("scored_cells") or 0) > 0 and not bool((per_var.get(var) or {}).get("local_exact"))
    ]
    simple_bounded_fix = bool(
        (repair_row or {}).get("local_functional_one_parent_repairable_under_cap")
        or (repair_row or {}).get("local_functional_one_mechanism_repairable_under_cap")
    )
    bucket = _bucket_for_row(
        replay,
        local_root_count=len(local_root_vars),
        local_train_equivalent_root_count=len(local_train_equivalent_root_vars),
        compensated_root_count=len(compensated_root_vars),
        simple_bounded_fix=simple_bounded_fix,
    )
    local_root_train_equivalent_share = rate(len(local_train_equivalent_root_vars), len(local_root_vars))
    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "family": family(problem),
        "slice": slice_name(problem),
        "difficulty": difficulty(problem),
        "task_name": task_name(problem),
        "valid": True,
        "train_exact": replay.evaluation.get("trainExact"),
        "heldout_exact": replay.evaluation.get("heldoutExact"),
        "split_error_bucket": split_error_bucket(replay.evaluation),
        "invalid_bucket": None,
        "training_equivalence_bucket": bucket,
        "local_root_count": len(local_root_vars),
        "local_train_equivalent_root_count": len(local_train_equivalent_root_vars),
        "compensated_root_count": len(compensated_root_vars),
        "local_root_train_equivalent_share": local_root_train_equivalent_share,
        "any_local_train_equivalent_root": bool(local_train_equivalent_root_vars),
        "any_compensated_root": bool(compensated_root_vars),
        "local_root_vars": local_root_vars,
        "local_train_equivalent_root_vars": local_train_equivalent_root_vars,
        "compensated_root_vars": compensated_root_vars,
        "simple_bounded_fix": simple_bounded_fix,
        "one_parent_fix": bool((repair_row or {}).get("local_functional_one_parent_repairable_under_cap")),
        "one_mechanism_fix": bool((repair_row or {}).get("local_functional_one_mechanism_repairable_under_cap")),
        "one_gold_mechanism_global": bool((repair_row or {}).get("global_one_gold_mechanism_repairable")),
        "local_functional_one_parent_target": (repair_row or {}).get("local_functional_one_parent_target"),
        "local_functional_one_mechanism_target": (repair_row or {}).get("local_functional_one_mechanism_target"),
        "global_one_gold_mechanism_target": (repair_row or {}).get("global_one_gold_mechanism_target"),
    }


def analyze_training_equivalence(
    problems: Sequence[Dict[str, Any]],
    cascade_instance_rows: Sequence[Dict[str, Any]],
    repairability_instance_rows: Sequence[Dict[str, Any]],
    *,
    models: Optional[set[str]] = None,
    family_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    cascade_by_key = {_group_key(row): row for row in cascade_instance_rows}
    repair_by_key = {_group_key(row): row for row in repairability_instance_rows}

    instance_rows: List[Dict[str, Any]] = []
    for problem in problems:
        if not is_a_scm_problem(problem):
            continue
        if family_filter and family(problem) != family_filter:
            continue
        problem_id = instance_id(problem)
        for llm_result in problem.get("llmResults") or []:
            if not isinstance(llm_result, dict):
                continue
            model = str(llm_result.get("model") or "unknown")
            if models and model not in models:
                continue
            key = (problem_id, model)
            instance_rows.append(
                _analyze_result(
                    problem,
                    llm_result,
                    cascade_row=cascade_by_key.get(key),
                    repair_row=repair_by_key.get(key),
                )
            )

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
            bucket_counts[str(row.get("training_equivalence_bucket") or "unknown")] += 1
        top_bucket = None
        top_bucket_share = None
        if bucket_counts and heldout_only:
            top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
            top_bucket_share = rate(top_count, len(heldout_only))
        by_model_rows.append(
            {
                "model": model,
                "n": len(group),
                "heldout_only_instances": len(heldout_only),
                "heldout_only_share": rate(len(heldout_only), len(group)),
                "top_training_equivalence_bucket": top_bucket,
                "top_training_equivalence_share": top_bucket_share,
                "local_ambiguity_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if str(row.get("training_equivalence_bucket") or "").startswith("train_exact_local_ambiguity")),
                    len(heldout_only),
                ),
                "compensated_train_fit_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if str(row.get("training_equivalence_bucket") or "").startswith("train_exact_compensated_fit")),
                    len(heldout_only),
                ),
                "simple_bounded_fix_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if row.get("simple_bounded_fix")),
                    len(heldout_only),
                ),
                "not_shown_ambiguous_under_cap_share_among_heldout_only": rate(
                    sum(1 for row in heldout_only if row.get("training_equivalence_bucket") == "train_exact_not_shown_ambiguous_under_cap"),
                    len(heldout_only),
                ),
            }
        )

    for (slice_value, model), group in sorted(grouped_slice.items()):
        heldout_only = [row for row in group if row.get("split_error_bucket") == "heldout_only_error"]
        if not heldout_only:
            continue
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in heldout_only:
            bucket_counts[str(row.get("training_equivalence_bucket") or "unknown")] += 1
        top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
        by_slice_rows.append(
            {
                "slice": slice_value,
                "model": model,
                "heldout_only_instances": len(heldout_only),
                "top_training_equivalence_bucket": top_bucket,
                "top_training_equivalence_share": rate(top_count, len(heldout_only)),
                "local_ambiguity_share": rate(
                    sum(1 for row in heldout_only if str(row.get("training_equivalence_bucket") or "").startswith("train_exact_local_ambiguity")),
                    len(heldout_only),
                ),
                "compensated_train_fit_share": rate(
                    sum(1 for row in heldout_only if str(row.get("training_equivalence_bucket") or "").startswith("train_exact_compensated_fit")),
                    len(heldout_only),
                ),
                "simple_bounded_fix_share": rate(
                    sum(1 for row in heldout_only if row.get("simple_bounded_fix")),
                    len(heldout_only),
                ),
            }
        )

    example_rows = [
        row
        for row in instance_rows
        if row.get("training_equivalence_bucket") in {
            "train_exact_local_ambiguity_with_simple_fix",
            "train_exact_local_ambiguity",
            "train_exact_compensated_fit_with_simple_fix",
            "train_exact_not_shown_ambiguous_under_cap",
        }
    ]
    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
        "example_rows": example_rows,
    }


def build_training_equivalence_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                row.get("heldout_only_instances"),
                pct(row.get("heldout_only_share")),
                row.get("top_training_equivalence_bucket") or "-",
                pct(row.get("top_training_equivalence_share")),
                pct(row.get("local_ambiguity_share_among_heldout_only")),
                pct(row.get("compensated_train_fit_share_among_heldout_only")),
                pct(row.get("simple_bounded_fix_share_among_heldout_only")),
                pct(row.get("not_shown_ambiguous_under_cap_share_among_heldout_only")),
            ]
        )
    lines = [
        "SCM Training Equivalence",
        "",
        "This report measures bucket B: train-exact but heldout-wrong cases.",
        "A local root mechanism counts as train-equivalent only when it fits the scored train rows on gold parent contexts exactly; otherwise train exactness is treated as compensated fit rather than local ambiguity.",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "Heldout-only",
                "Heldout-only share",
                "Top bucket",
                "Top share",
                "Local ambiguity",
                "Compensated fit",
                "Simple bounded fix",
                "Not shown ambiguous",
            ],
            model_rows,
        ) if model_rows else "No training-equivalence rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_training_equivalence_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
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
    write_markdown(build_training_equivalence_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze bounded training equivalence for A_SCM heldout failures")
    parser.add_argument("--input", required=True, help="Benchmark YAML input")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    args = parser.parse_args(argv)

    _, problems = load_dataset(args.input)
    models = _parse_models_arg(args.models)
    repairability = analyze_repairability(problems, models=models, family_filter=args.family)

    from concept_synth.analysis.scm_extract_eval_records import extract_scm_analysis_records
    from concept_synth.analysis.scm_cascade_attribution import analyze_cascade_attribution

    extract = extract_scm_analysis_records(problems, source_name=str(args.input), models=models, family_filter=args.family)
    cascade = analyze_cascade_attribution(extract["instance_records"], extract["mechanism_records"], extract["world_records"])
    artifacts = analyze_training_equivalence(
        problems,
        cascade["instance_rows"],
        repairability["instance_rows"],
        models=models,
        family_filter=args.family,
    )
    paths = write_training_equivalence_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM training-equivalence artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
