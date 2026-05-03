from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    difficulty,
    family,
    gold_allow_constants,
    gold_allowed_operators,
    gold_endogenous_vars,
    gold_scm,
    gold_topological_order,
    instance_id,
    load_dataset,
    markdown_table,
    parse_candidate,
    pct,
    rate,
    replay_candidate,
    slice_name,
    stable_json_dumps,
    task_name,
    write_csv,
    write_jsonl,
    write_markdown,
)
from concept_synth.analysis.scm_subset_compare import _parse_models_arg
from concept_synth.causal_reasoning.cind_family import (
    _extract_world_intervention_targets,
    _find_small_shortcut_witness,
    _iter_world_rows,
    _worlds_by_split,
)

SUMMARY_MD = "scm_repairability_summary.md"
BY_INSTANCE_CSV = "scm_repairability_by_instance.csv"
BY_MODEL_CSV = "scm_repairability_by_model.csv"
EXAMPLES_JSONL = "scm_repairability_examples.jsonl"
MANIFEST_JSON = "scm_repairability_manifest.json"


def _full_exact(replay) -> bool:
    return bool(replay.evaluation.get("valid")) and bool(replay.evaluation.get("trainExact")) and bool(replay.evaluation.get("heldoutExact"))


def _build_rows_by_var(problem: Dict[str, Any], endogenous_vars: Sequence[str]) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
    rows_by_split: Dict[str, Dict[str, List[Dict[str, int]]]] = {
        "train": {str(var): [] for var in endogenous_vars},
        "heldout": {str(var): [] for var in endogenous_vars},
    }
    train_worlds, heldout_worlds = _worlds_by_split(problem.get("problem") or problem)
    for split_name, worlds in (("train", train_worlds), ("heldout", heldout_worlds)):
        for world in worlds:
            intervened = set(_extract_world_intervention_targets(world))
            rows = _iter_world_rows(world)
            for var in endogenous_vars:
                if str(var) in intervened:
                    continue
                rows_by_split[split_name][str(var)].extend(
                    {
                        str(key): int(value)
                        for key, value in row.items()
                        if isinstance(value, (bool, int)) or (isinstance(value, str) and value in {"0", "1"})
                    }
                    for row in rows
                    if str(var) in row
                )
    return rows_by_split


def _constant_witness(rows: Sequence[Dict[str, int]], target_var: str, *, allow_constants: bool) -> Optional[Dict[str, Any]]:
    if not allow_constants:
        return None
    values = {int(row[target_var]) for row in rows if target_var in row}
    if len(values) != 1:
        return None
    value = next(iter(values))
    return {
        "expr": str(int(value)),
        "ast": 1,
        "node": None,
        "stats": {"astSize": 1, "maxDepth": 1, "parentCount": 0, "operatorCounts": {}},
    }


def _functional_witness(
    *,
    rows: Sequence[Dict[str, int]],
    target_var: str,
    candidate_vars: Sequence[str],
    ast_cap: int,
    allowed_ops: Sequence[str],
    allow_constants: bool,
) -> Optional[Dict[str, Any]]:
    rows_list = [row for row in rows if target_var in row]
    if not rows_list:
        return None
    if not candidate_vars:
        return _constant_witness(rows_list, target_var, allow_constants=allow_constants)
    witness = _find_small_shortcut_witness(
        rows=rows_list,
        target_var=target_var,
        candidate_vars=list(candidate_vars),
        ast_cap=int(ast_cap),
        allowed_operators=list(allowed_ops),
        allow_constants=allow_constants,
    )
    if witness is not None:
        return witness
    return _constant_witness(rows_list, target_var, allow_constants=allow_constants)


def _replace_expr_and_replay(problem: Dict[str, Any], llm_result: Dict[str, Any], target_var: str, expr: str):
    updated = copy.deepcopy(llm_result)
    answer = updated.get("extractedAnswer")
    if not isinstance(answer, dict):
        answer = {}
        updated["extractedAnswer"] = answer
    mechanisms = answer.get("mechanisms")
    if not isinstance(mechanisms, dict):
        mechanisms = {}
        answer["mechanisms"] = mechanisms
    mechanisms[str(target_var)] = str(expr)
    updated.pop("evaluation", None)
    return replay_candidate(problem, updated)


def _candidate_var_priority(parsed, replay) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    train_per_var = (replay.train_summary or {}).get("perVariable") or {}
    heldout_per_var = (replay.heldout_summary or {}).get("perVariable") or {}
    for idx, var in enumerate(parsed.endogenous_vars):
        train_wrong = int((train_per_var.get(var) or {}).get("wrong_cells") or 0)
        heldout_wrong = int((heldout_per_var.get(var) or {}).get("wrong_cells") or 0)
        structural_wrong = 0
        # prioritize true errors and structure mismatches first
        if var in parsed.parent_by_var:
            structural_wrong = 1
        out.append((-(heldout_wrong + train_wrong + structural_wrong), idx, str(var)))
    out.sort()
    return out


def _parent_edit_variants(pred_parents: Sequence[str], predecessors: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    pred = tuple(sorted({str(v) for v in pred_parents}))
    pred_set = set(pred)
    predecessors_set = {str(v) for v in predecessors}
    variants: Dict[Tuple[str, ...], str] = {}

    for parent in pred:
        candidate = tuple(sorted(pred_set - {parent}))
        variants[candidate] = "remove"
    for parent in sorted(predecessors_set - pred_set):
        candidate = tuple(sorted(pred_set | {parent}))
        variants[candidate] = "add"
    for remove_parent in pred:
        for add_parent in sorted(predecessors_set - pred_set):
            candidate = tuple(sorted((pred_set - {remove_parent}) | {add_parent}))
            variants[candidate] = "swap"
    variants.pop(pred, None)
    return [(edit_type, parents) for parents, edit_type in sorted(variants.items(), key=lambda item: (len(item[0]), item[1], item[0]))]


def _analyze_result_repairability(
    problem: Dict[str, Any],
    llm_result: Dict[str, Any],
    *,
    functional_ast_cap: int,
) -> Dict[str, Any]:
    replay = replay_candidate(problem, llm_result)
    parsed = replay.candidate_parse
    gold_mechanisms = (gold_scm(problem).get("mechanisms") or {}) if isinstance(gold_scm(problem).get("mechanisms"), dict) else (gold_scm(problem).get("mechanisms") or {})
    if parsed.replayable is False:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "instance_id": instance_id(problem),
            "model": str(llm_result.get("model") or "unknown"),
            "family": family(problem),
            "slice": slice_name(problem),
            "difficulty": difficulty(problem),
            "task_name": task_name(problem),
            "valid": False,
            "repairability_bucket": "invalid",
            "local_functional_one_parent_repairable_under_cap": False,
            "local_functional_one_mechanism_repairable_under_cap": False,
            "global_one_gold_mechanism_repairable": False,
            "functional_ast_cap": int(functional_ast_cap),
            "invalid_bucket": parsed.invalid_bucket,
        }

    if _full_exact(replay):
        return {
            "analysis_version": ANALYSIS_VERSION,
            "instance_id": instance_id(problem),
            "model": str(llm_result.get("model") or "unknown"),
            "family": family(problem),
            "slice": slice_name(problem),
            "difficulty": difficulty(problem),
            "task_name": task_name(problem),
            "valid": True,
            "repairability_bucket": "exact",
            "local_functional_one_parent_repairable_under_cap": False,
            "local_functional_one_mechanism_repairable_under_cap": False,
            "global_one_gold_mechanism_repairable": False,
            "functional_ast_cap": int(functional_ast_cap),
            "invalid_bucket": None,
        }

    rows_by_split = _build_rows_by_var(problem, parsed.endogenous_vars)
    allowed_ops = list(gold_allowed_operators(problem))
    allow_constants = bool(gold_allow_constants(problem))
    topo = list(gold_topological_order(problem))
    topo_index = {str(var): idx for idx, var in enumerate(topo)}

    one_parent = None
    one_mech = None
    gold_global = None

    priority_vars = _candidate_var_priority(parsed, replay)
    for _neg_wrong, _idx, var in priority_vars:
        var_idx = topo_index.get(str(var), -1)
        predecessors = [candidate for candidate in topo[:var_idx] if candidate != var] if var_idx > 0 else []
        rows_combined = list(rows_by_split["train"].get(var, [])) + list(rows_by_split["heldout"].get(var, []))
        if not rows_combined:
            continue
        pred_parents = parsed.parent_by_var.get(var, [])
        for edit_type, parent_variant in _parent_edit_variants(pred_parents, predecessors):
            witness = _functional_witness(
                rows=rows_combined,
                target_var=var,
                candidate_vars=list(parent_variant),
                ast_cap=int(functional_ast_cap),
                allowed_ops=allowed_ops,
                allow_constants=allow_constants,
            )
            if witness is None:
                continue
            one_parent = {
                "target": var,
                "edit_type": edit_type,
                "parent_variant": list(parent_variant),
                "expr": str(witness.get("expr") or ""),
                "ast": int((witness.get("ast") or 0) if witness.get("ast") is not None else 0),
            }
            break
        if one_parent is not None:
            break

    if one_parent is not None:
        one_mech = {
            "target": one_parent["target"],
            "expr": one_parent["expr"],
            "ast": one_parent["ast"],
        }
    else:
        for _neg_wrong, _idx, var in priority_vars:
            var_idx = topo_index.get(str(var), -1)
            predecessors = [candidate for candidate in topo[:var_idx] if candidate != var] if var_idx > 0 else []
            rows_combined = list(rows_by_split["train"].get(var, [])) + list(rows_by_split["heldout"].get(var, []))
            if not rows_combined:
                continue
            witness = _functional_witness(
                rows=rows_combined,
                target_var=var,
                candidate_vars=predecessors,
                ast_cap=int(functional_ast_cap),
                allowed_ops=allowed_ops,
                allow_constants=allow_constants,
            )
            if witness is None:
                continue
            one_mech = {
                "target": var,
                "expr": str(witness.get("expr") or ""),
                "ast": int((witness.get("ast") or 0) if witness.get("ast") is not None else 0),
            }
            break

    for _neg_wrong, _idx, var in priority_vars:
        gold_expr = gold_mechanisms.get(var)
        pred_expr = parsed.mechanisms.get(var)
        if not isinstance(gold_expr, str) or not gold_expr.strip() or pred_expr == gold_expr:
            continue
        replay_gold = _replace_expr_and_replay(problem, llm_result, var, gold_expr)
        if _full_exact(replay_gold):
            gold_global = {
                "target": var,
                "expr": gold_expr,
            }
            break

    if one_parent is not None:
        bucket = "one_parent_functional_local_under_cap"
    elif one_mech is not None:
        bucket = "one_mechanism_functional_local_under_cap"
    elif gold_global is not None:
        bucket = "one_gold_mechanism_global"
    else:
        bucket = "not_one_step_repairable_under_cap"

    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "family": family(problem),
        "slice": slice_name(problem),
        "difficulty": difficulty(problem),
        "task_name": task_name(problem),
        "valid": True,
        "repairability_bucket": bucket,
        "local_functional_one_parent_repairable_under_cap": one_parent is not None,
        "local_functional_one_parent_target": None if one_parent is None else one_parent["target"],
        "local_functional_one_parent_edit_type": None if one_parent is None else one_parent["edit_type"],
        "local_functional_one_parent_parent_variant": None if one_parent is None else one_parent["parent_variant"],
        "local_functional_one_parent_expr": None if one_parent is None else one_parent["expr"],
        "local_functional_one_parent_ast": None if one_parent is None else one_parent["ast"],
        "local_functional_one_mechanism_repairable_under_cap": one_mech is not None,
        "local_functional_one_mechanism_target": None if one_mech is None else one_mech["target"],
        "local_functional_one_mechanism_expr": None if one_mech is None else one_mech["expr"],
        "local_functional_one_mechanism_ast": None if one_mech is None else one_mech["ast"],
        "global_one_gold_mechanism_repairable": gold_global is not None,
        "global_one_gold_mechanism_target": None if gold_global is None else gold_global["target"],
        "functional_ast_cap": int(functional_ast_cap),
        "invalid_bucket": None,
    }


def analyze_repairability(
    problems: Sequence[Dict[str, Any]],
    *,
    models: Optional[set[str]] = None,
    family_filter: Optional[str] = None,
    functional_ast_cap: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    instance_rows: List[Dict[str, Any]] = []
    for problem in problems:
        if family_filter and family(problem) != family_filter:
            continue
        for llm_result in problem.get("llmResults") or []:
            if not isinstance(llm_result, dict):
                continue
            model = str(llm_result.get("model") or "unknown")
            if models and model not in models:
                continue
            instance_rows.append(
                _analyze_result_repairability(problem, llm_result, functional_ast_cap=functional_ast_cap)
            )

    by_model_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
    for model, group in sorted(grouped_model.items()):
        total = len(group)
        valid_nonexact = [row for row in group if row.get("valid") is True and row.get("repairability_bucket") != "exact"]
        by_model_rows.append(
            {
                "model": model,
                "n": total,
                "exact_share": rate(sum(1 for row in group if row.get("repairability_bucket") == "exact"), total),
                "invalid_share": rate(sum(1 for row in group if row.get("repairability_bucket") == "invalid"), total),
                "local_functional_one_parent_share_among_valid_nonexact": rate(
                    sum(1 for row in valid_nonexact if row.get("local_functional_one_parent_repairable_under_cap")),
                    len(valid_nonexact),
                ),
                "local_functional_one_mechanism_share_among_valid_nonexact": rate(
                    sum(1 for row in valid_nonexact if row.get("local_functional_one_mechanism_repairable_under_cap")),
                    len(valid_nonexact),
                ),
                "global_one_gold_mechanism_share_among_valid_nonexact": rate(
                    sum(1 for row in valid_nonexact if row.get("global_one_gold_mechanism_repairable")),
                    len(valid_nonexact),
                ),
                "not_one_step_repairable_share_among_valid_nonexact": rate(
                    sum(1 for row in valid_nonexact if row.get("repairability_bucket") == "not_one_step_repairable_under_cap"),
                    len(valid_nonexact),
                ),
            }
        )

    example_rows = [
        row
        for row in instance_rows
        if row.get("repairability_bucket") in {
            "one_parent_functional_local_under_cap",
            "one_mechanism_functional_local_under_cap",
            "one_gold_mechanism_global",
            "not_one_step_repairable_under_cap",
        }
    ]
    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "example_rows": example_rows,
    }


def build_repairability_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                pct(row.get("exact_share")),
                pct(row.get("invalid_share")),
                pct(row.get("local_functional_one_parent_share_among_valid_nonexact")),
                pct(row.get("local_functional_one_mechanism_share_among_valid_nonexact")),
                pct(row.get("global_one_gold_mechanism_share_among_valid_nonexact")),
                pct(row.get("not_one_step_repairable_share_among_valid_nonexact")),
            ]
        )
    lines = [
        "SCM Repairability",
        "",
        "This report separates structural closeness to gold from bounded functional repairability.",
        "Functional one-step repairability asks whether a single local replacement can fit the scored train+heldout rows under a bounded search cap, even if the repaired formula differs from gold.",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "Exact",
                "Invalid",
                "1-parent functional",
                "1-mechanism functional",
                "1-gold-mechanism global",
                "Not 1-step under cap",
            ],
            model_rows,
        ) if model_rows else "No repairability rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_repairability_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_instance_csv": str(out / BY_INSTANCE_CSV),
        "by_model_csv": str(out / BY_MODEL_CSV),
        "examples_jsonl": str(out / EXAMPLES_JSONL),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["instance_rows"], paths["by_instance_csv"])
    write_csv(artifacts["by_model_rows"], paths["by_model_csv"])
    write_jsonl(artifacts["example_rows"], paths["examples_jsonl"])
    write_markdown(build_repairability_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one-step repairability for A_SCM results")
    parser.add_argument("--input", required=True, help="Benchmark YAML input")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    parser.add_argument("--functional-ast-cap", type=int, default=6, help="AST cap for bounded functional repair search")
    args = parser.parse_args(argv)

    _, problems = load_dataset(args.input)
    artifacts = analyze_repairability(
        problems,
        models=_parse_models_arg(args.models),
        family_filter=args.family,
        functional_ast_cap=int(args.functional_ast_cap),
    )
    paths = write_repairability_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM repairability artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
