from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    bucket_numeric,
    canonical_template_from_node,
    canonicalize_node,
    classify_invalid_bucket,
    difficulty,
    ensure_evaluation,
    extracted_answer,
    family,
    generation_diagnostics,
    gold_allowed_operators,
    gold_allow_constants,
    gold_endogenous_vars,
    gold_root_vars,
    gold_scm,
    gold_topological_order,
    instance_id,
    is_a_scm_problem,
    load_dataset,
    map_record_from_replay,
    markdown_table,
    mean,
    median,
    mode_sensitivity_bucket,
    natural_key,
    novelty_sensitivity_bucket,
    parse_candidate,
    pct,
    rate,
    read_jsonl,
    replay_candidate,
    slice_name,
    split_error_bucket,
    stable_json_dumps,
    summarize_operator_profile,
    task_name,
    task_params,
    write_csv,
    write_jsonl,
    write_markdown,
)

from concept_synth.causal_reasoning.mechanism_dsl import MechanismNode, analyze_mechanism, parse_mechanism


INSTANCE_JSONL = "scm_instance_records.jsonl"
WORLD_JSONL = "scm_world_records.jsonl"
MECH_JSONL = "scm_mechanism_records.jsonl"
MAP_JSONL = "scm_map_records.jsonl"
SUMMARY_MD = "scm_extract_summary.md"
MANIFEST_JSON = "scm_manifest.json"


def _parse_model_filter(models: Optional[str]) -> Optional[set[str]]:
    if not models:
        return None
    out = {piece.strip() for piece in str(models).split(",") if piece.strip()}
    return out or None


def _gold_parse(problem: Dict[str, Any]):
    scm = gold_scm(problem)
    mechanisms = scm.get("mechanisms") or {}
    return parse_candidate(problem, {"extractedAnswer": {"mechanisms": mechanisms}})


def _gold_modular_ast(problem: Dict[str, Any]) -> Optional[int]:
    diag = generation_diagnostics(problem)
    if diag.get("gold_modular_total_ast") is not None:
        return int(diag.get("gold_modular_total_ast"))
    scm = gold_scm(problem)
    if scm.get("totalAst") is not None:
        return int(scm.get("totalAst"))
    stats = scm.get("mechanismStatsByVar") or {}
    if isinstance(stats, dict):
        return int(sum(int((stats.get(var) or {}).get("astSize", 0)) for var in gold_endogenous_vars(problem)))
    return None


def _gold_expanded_ast(problem: Dict[str, Any]) -> Optional[int]:
    diag = generation_diagnostics(problem)
    if diag.get("gold_expanded_total_ast") is not None:
        return int(diag.get("gold_expanded_total_ast"))
    scm = gold_scm(problem)
    stats = scm.get("expandedMechanismStatsByVar") or {}
    if isinstance(stats, dict):
        return int(sum(int((stats.get(var) or {}).get("astSize", 0)) for var in gold_endogenous_vars(problem)))
    return None


def _aggregate_operator_counts(stats_by_var: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(stats_by_var, dict):
        return out
    for payload in stats_by_var.values():
        if not isinstance(payload, dict):
            continue
        for op, count in ((payload.get("operatorCounts") or {}).items() if isinstance(payload.get("operatorCounts"), dict) else []):
            out[str(op)] = out.get(str(op), 0) + int(count)
    return out


def _gold_operator_profile(problem: Dict[str, Any]) -> str:
    return summarize_operator_profile(_aggregate_operator_counts((gold_scm(problem).get("mechanismStatsByVar") or {})))


def _shortcut_positive_count(problem: Dict[str, Any]) -> int:
    diag = generation_diagnostics(problem)
    local = diag.get("localShortcutDiagnosticsByEndogenous") or {}
    count = 0
    if isinstance(local, dict):
        for payload in local.values():
            if not isinstance(payload, dict):
                continue
            if int(payload.get("train_alt_count_under_cap", 0) or 0) > 0:
                count += 1
    return count


def _instance_record(
    problem: Dict[str, Any],
    llm_result: Dict[str, Any],
    replay,
    gold_parse,
    source_name: str,
) -> Dict[str, Any]:
    evaluation = replay.evaluation
    parsed = replay.candidate_parse
    candidate_stats = evaluation.get("candidateStats") or {}
    candidate_op_counts = candidate_stats.get("operatorCounts") or {}
    diag = generation_diagnostics(problem)
    train = replay.train_summary or {}
    heldout = replay.heldout_summary or {}
    train_worlds = [row for row in replay.world_records if row.get("split") == "train"]
    heldout_worlds = [row for row in replay.world_records if row.get("split") == "heldout"]

    gold_modular_ast = _gold_modular_ast(problem)
    gold_expanded_ast = _gold_expanded_ast(problem)
    candidate_ast = candidate_stats.get("astSize")
    candidate_depth = candidate_stats.get("maxDepth")
    candidate_parent_count = candidate_stats.get("parentCount")
    ast_gap = None if gold_modular_ast is None or candidate_ast is None else int(candidate_ast) - int(gold_modular_ast)

    invalid_bucket = parsed.invalid_bucket or classify_invalid_bucket(evaluation.get("parseError"), evaluation.get("failureExplanation"))
    map_record = map_record_from_replay(problem, llm_result, replay)
    record = {
        "analysis_version": ANALYSIS_VERSION,
        "source_name": source_name,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "task_name": task_name(problem),
        "family": family(problem),
        "slice": slice_name(problem),
        "difficulty": difficulty(problem),
        "prompt_variant": str(task_params(problem).get("scmPromptVariant") or ""),
        "valid": bool(evaluation.get("valid")),
        "correct": evaluation.get("correct"),
        "train_exact": evaluation.get("trainExact"),
        "heldout_exact": evaluation.get("heldoutExact"),
        "split_error_bucket": split_error_bucket(evaluation),
        "invalid_bucket": invalid_bucket,
        "parse_error": evaluation.get("parseError"),
        "failure_explanation": evaluation.get("failureExplanation"),
        "train_accuracy": evaluation.get("trainAccuracy"),
        "heldout_accuracy": evaluation.get("heldoutAccuracy"),
        "train_world_exact_accuracy": evaluation.get("trainWorldExactAccuracy"),
        "heldout_world_exact_accuracy": evaluation.get("heldoutWorldExactAccuracy"),
        "train_worlds": evaluation.get("trainWorlds"),
        "heldout_worlds": evaluation.get("heldoutWorlds"),
        "first_train_mismatch": evaluation.get("firstTrainMismatch"),
        "first_heldout_mismatch": evaluation.get("firstHeldoutMismatch"),
        "gold_modular_ast": gold_modular_ast,
        "gold_expanded_ast": gold_expanded_ast,
        "gold_operator_profile": _gold_operator_profile(problem),
        "gold_root_count": len(gold_root_vars(problem)),
        "gold_endogenous_count": len(gold_endogenous_vars(problem)),
        "candidate_ast": candidate_ast,
        "candidate_depth": candidate_depth,
        "candidate_parent_count": candidate_parent_count,
        "candidate_operator_profile": summarize_operator_profile(candidate_op_counts),
        "candidate_operator_counts": candidate_op_counts,
        "ast_gap": ast_gap,
        "bloat": evaluation.get("bloat"),
        "acc_gold_plus_25": ((evaluation.get("accGoldPlus") or {}).get("delta_25") if isinstance(evaluation.get("accGoldPlus"), dict) else None),
        "parent_f1": evaluation.get("parentF1"),
        "heldout_mean_novelty": diag.get("heldout_mean_novelty"),
        "heldout_max_novelty": diag.get("heldout_max_novelty"),
        "heldout_min_novelty": diag.get("heldout_min_novelty"),
        "min_scored_worlds_any_endogenous": diag.get("min_scored_worlds_any_endogenous"),
        "min_scored_cells_any_endogenous": diag.get("min_scored_cells_any_endogenous"),
        "max_intervened_worlds_any_endogenous": diag.get("max_intervened_worlds_any_endogenous"),
        "heldout_min_scored_worlds_any_endogenous": diag.get("heldout_min_scored_worlds_any_endogenous"),
        "heldout_min_scored_cells_any_endogenous": diag.get("heldout_min_scored_cells_any_endogenous"),
        "shortcut_positive_vars": _shortcut_positive_count(problem),
        "replayable": parsed.replayable,
        "replay_train_exact": train.get("exact"),
        "replay_heldout_exact": heldout.get("exact"),
        "replay_train_accuracy": train.get("accuracy"),
        "replay_heldout_accuracy": heldout.get("accuracy"),
        "replay_train_wrong_cells": train.get("wrongCells"),
        "replay_heldout_wrong_cells": heldout.get("wrongCells"),
        "replay_train_wrong_worlds": train.get("wrongWorldCount"),
        "replay_heldout_wrong_worlds": heldout.get("wrongWorldCount"),
        "replay_train_direction_bucket": train.get("directionBucket"),
        "replay_heldout_direction_bucket": heldout.get("directionBucket"),
        "replay_train_mismatch_scope_bucket": train.get("mismatchScopeBucket"),
        "replay_heldout_mismatch_scope_bucket": heldout.get("mismatchScopeBucket"),
        "replay_train_dominant_wrong_variable": train.get("dominantWrongVariable"),
        "replay_heldout_dominant_wrong_variable": heldout.get("dominantWrongVariable"),
        "heldout_mode_sensitivity_bucket": mode_sensitivity_bucket(heldout_worlds),
        "heldout_novelty_sensitivity_bucket": novelty_sensitivity_bucket(heldout_worlds),
        "exact_map_signature": map_record.get("exact_map_signature"),
        "canonical_map_signature": map_record.get("canonical_map_signature"),
        "gold_exact_map_signature": map_record.get("gold_exact_map_signature"),
        "gold_canonical_map_signature": map_record.get("gold_canonical_map_signature"),
        "mechanism_count": map_record.get("mechanism_count"),
        "required_mechanism_count": map_record.get("required_mechanism_count"),
        "candidate_parse_failure": parsed.failure_explanation,
        "gold_parse_replayable": gold_parse.replayable,
    }
    return record


def _mechanism_records(
    problem: Dict[str, Any],
    llm_result: Dict[str, Any],
    replay,
    gold_parse,
    source_name: str,
) -> List[Dict[str, Any]]:
    parsed = replay.candidate_parse
    if not parsed.replayable:
        return []
    gold = gold_scm(problem)
    gold_mechanisms = gold.get("mechanisms") or {}
    gold_stats_by_var = gold.get("mechanismStatsByVar") or {}
    gold_parents_by_var = gold.get("parentsByVar") or {}
    train_per_var = (replay.train_summary or {}).get("perVariable") or {}
    heldout_per_var = (replay.heldout_summary or {}).get("perVariable") or {}
    evaluation = replay.evaluation
    out: List[Dict[str, Any]] = []
    topo_index = {var: idx for idx, var in enumerate(parsed.eval_topological_order)}
    endogenous_index = {var: idx for idx, var in enumerate(parsed.endogenous_vars)}
    for target_var in parsed.endogenous_vars:
        pred_node = parsed.parsed_nodes_by_var.get(target_var)
        gold_node = gold_parse.parsed_nodes_by_var.get(target_var) if gold_parse.replayable else None
        pred_stats = parsed.candidate_stats_by_var.get(target_var) or (analyze_mechanism(pred_node) if pred_node is not None else {})
        gold_stats = (gold_stats_by_var.get(target_var) or {}) if isinstance(gold_stats_by_var, dict) else {}
        if gold_node is not None and not gold_stats:
            gold_stats = analyze_mechanism(gold_node)
        pred_expr = parsed.mechanisms.get(target_var)
        gold_expr = gold_mechanisms.get(target_var)
        pred_ast = pred_stats.get("astSize")
        gold_ast = gold_stats.get("astSize")
        parent_gap = None
        if pred_stats.get("parentCount") is not None and gold_stats.get("parentCount") is not None:
            parent_gap = int(pred_stats.get("parentCount")) - int(gold_stats.get("parentCount"))
        ast_gap = None
        if pred_ast is not None and gold_ast is not None:
            ast_gap = int(pred_ast) - int(gold_ast)
        gold_parents = sorted([str(v) for v in (gold_parents_by_var.get(target_var) or [])], key=natural_key)
        pred_parents = parsed.parent_by_var.get(target_var) or []
        out.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "source_name": source_name,
                "instance_id": instance_id(problem),
                "model": str(llm_result.get("model") or "unknown"),
                "family": family(problem),
                "slice": slice_name(problem),
                "difficulty": difficulty(problem),
                "target_var": target_var,
                "target_topological_index": topo_index.get(target_var),
                "target_endogenous_index": endogenous_index.get(target_var),
                "valid": bool(evaluation.get("valid")),
                "train_exact": evaluation.get("trainExact"),
                "heldout_exact": evaluation.get("heldoutExact"),
                "split_error_bucket": split_error_bucket(evaluation),
                "invalid_bucket": parsed.invalid_bucket,
                "gold_expr": gold_expr,
                "gold_expr_canonical": canonical_template_from_node(gold_node) if gold_node is not None else None,
                "gold_ast": gold_ast,
                "gold_depth": gold_stats.get("maxDepth"),
                "gold_parent_count": gold_stats.get("parentCount"),
                "gold_operator_counts": gold_stats.get("operatorCounts"),
                "gold_operator_profile": summarize_operator_profile(gold_stats.get("operatorCounts") or {}),
                "gold_parents": gold_parents,
                "pred_expr": pred_expr,
                "pred_expr_canonical": canonical_template_from_node(pred_node) if pred_node is not None else None,
                "pred_ast": pred_ast,
                "pred_depth": pred_stats.get("maxDepth"),
                "pred_parent_count": pred_stats.get("parentCount"),
                "pred_operator_counts": pred_stats.get("operatorCounts"),
                "pred_operator_profile": summarize_operator_profile(pred_stats.get("operatorCounts") or {}),
                "pred_parents": pred_parents,
                "ast_gap": ast_gap,
                "parent_gap": parent_gap,
                "expr_exact": pred_expr == gold_expr,
                "expr_canonical_exact": (
                    canonical_template_from_node(pred_node) == canonical_template_from_node(gold_node)
                    if pred_node is not None and gold_node is not None
                    else None
                ),
                "parent_exact": set(pred_parents) == set(gold_parents),
                "train_var_accuracy": (train_per_var.get(target_var) or {}).get("accuracy"),
                "heldout_var_accuracy": (heldout_per_var.get(target_var) or {}).get("accuracy"),
                "train_var_wrong_cells": (train_per_var.get(target_var) or {}).get("wrong_cells"),
                "heldout_var_wrong_cells": (heldout_per_var.get(target_var) or {}).get("wrong_cells"),
                "train_var_under_count": (train_per_var.get(target_var) or {}).get("under_count"),
                "train_var_over_count": (train_per_var.get(target_var) or {}).get("over_count"),
                "heldout_var_under_count": (heldout_per_var.get(target_var) or {}).get("under_count"),
                "heldout_var_over_count": (heldout_per_var.get(target_var) or {}).get("over_count"),
            }
        )
    return out


def extract_scm_analysis_records(
    problems: Sequence[Dict[str, Any]],
    *,
    source_name: str,
    models: Optional[set[str]] = None,
    family_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    instance_records: List[Dict[str, Any]] = []
    world_records: List[Dict[str, Any]] = []
    mechanism_records: List[Dict[str, Any]] = []
    map_records: List[Dict[str, Any]] = []

    for problem in problems:
        if not is_a_scm_problem(problem):
            continue
        if family_filter and family(problem) != family_filter:
            continue
        gold_parse = _gold_parse(problem)
        for llm_result in problem.get("llmResults") or []:
            if not isinstance(llm_result, dict):
                continue
            model = str(llm_result.get("model") or "unknown")
            if models and model not in models:
                continue
            replay = replay_candidate(problem, llm_result)
            instance_records.append(_instance_record(problem, llm_result, replay, gold_parse, source_name))
            world_records.extend(replay.world_records)
            mechanism_records.extend(_mechanism_records(problem, llm_result, replay, gold_parse, source_name))
            map_records.append(map_record_from_replay(problem, llm_result, replay))
    return {
        "instance_records": instance_records,
        "world_records": world_records,
        "mechanism_records": mechanism_records,
        "map_records": map_records,
    }


def _summary_markdown(artifacts: Dict[str, List[Dict[str, Any]]], source_name: str) -> str:
    instance_records = artifacts["instance_records"]
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_records:
        by_model[str(row.get("model") or "unknown")].append(row)

    rows: List[List[Any]] = []
    for model in sorted(by_model.keys()):
        group = by_model[model]
        total = len(group)
        valid = sum(1 for row in group if row.get("valid") is True)
        train_exact = sum(1 for row in group if row.get("train_exact") is True)
        heldout_exact = sum(1 for row in group if row.get("heldout_exact") is True)
        replayable = sum(1 for row in group if row.get("replayable") is True)
        rows.append([
            model,
            total,
            pct(rate(valid, total)),
            pct(rate(train_exact, total)),
            pct(rate(heldout_exact, total)),
            pct(rate(replayable, total)),
        ])

    lines = [
        "SCM Extraction Summary",
        "",
        f"Source: `{source_name}`",
        f"Instance records: {len(artifacts['instance_records'])}",
        f"World records: {len(artifacts['world_records'])}",
        f"Mechanism records: {len(artifacts['mechanism_records'])}",
        f"Map records: {len(artifacts['map_records'])}",
        "",
        markdown_table(
            ["Model", "N", "Valid", "TrainExact", "HeldoutExact", "Replayable"],
            rows,
        ) if rows else "No A_SCM model results found.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path, *, source_name: str) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "instance_jsonl": str(out / INSTANCE_JSONL),
        "world_jsonl": str(out / WORLD_JSONL),
        "mechanism_jsonl": str(out / MECH_JSONL),
        "map_jsonl": str(out / MAP_JSONL),
        "instance_csv": str(out / INSTANCE_JSONL.replace(".jsonl", ".csv")),
        "world_csv": str(out / WORLD_JSONL.replace(".jsonl", ".csv")),
        "mechanism_csv": str(out / MECH_JSONL.replace(".jsonl", ".csv")),
        "map_csv": str(out / MAP_JSONL.replace(".jsonl", ".csv")),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_jsonl(artifacts["instance_records"], paths["instance_jsonl"])
    write_jsonl(artifacts["world_records"], paths["world_jsonl"])
    write_jsonl(artifacts["mechanism_records"], paths["mechanism_jsonl"])
    write_jsonl(artifacts["map_records"], paths["map_jsonl"])
    write_csv(artifacts["instance_records"], paths["instance_csv"])
    write_csv(artifacts["world_records"], paths["world_csv"])
    write_csv(artifacts["mechanism_records"], paths["mechanism_csv"])
    write_csv(artifacts["map_records"], paths["map_csv"])
    write_markdown(_summary_markdown(artifacts, source_name), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "source_name": source_name,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract offline A_SCM analysis records from causal benchmark YAML")
    parser.add_argument("--input", required=True, help="Input causal benchmark YAML")
    parser.add_argument("--outdir", required=True, help="Output directory for JSONL/CSV/Markdown artifacts")
    parser.add_argument("--models", help="Comma-separated model filter")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    args = parser.parse_args(argv)

    _, problems = load_dataset(args.input)
    artifacts = extract_scm_analysis_records(
        problems,
        source_name=str(args.input),
        models=_parse_model_filter(args.models),
        family_filter=args.family,
    )
    paths = write_artifacts(artifacts, args.outdir, source_name=str(args.input))
    print(f"Wrote A_SCM analysis artifacts to {args.outdir}")
    print(f"  instance_records: {len(artifacts['instance_records'])}")
    print(f"  world_records: {len(artifacts['world_records'])}")
    print(f"  mechanism_records: {len(artifacts['mechanism_records'])}")
    print(f"  map_records: {len(artifacts['map_records'])}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
