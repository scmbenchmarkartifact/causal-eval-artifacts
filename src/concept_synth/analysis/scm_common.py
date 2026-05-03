from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from concept_synth.io_utils import load_from_yaml
from concept_synth.causal_reasoning.evaluator import evaluate_causal_llm_result
from concept_synth.causal_reasoning.mechanism_dsl import (
    MechanismEvalError,
    MechanismNode,
    MechanismParseError,
    analyze_mechanism,
    evaluate_parsed_mechanism,
    mechanism_variables,
    node_to_sexpr,
    parse_mechanism,
)
from concept_synth.causal_reasoning.cind_family import (
    DEFAULT_ALLOWED_OPERATORS,
    _canonicalize_node,
    _extract_world_intervention_assignments,
    _extract_world_intervention_mode,
    _extract_world_intervention_targets,
    _infer_ntopo_endogenous_order,
    _iter_world_rows,
    _worlds_by_split,
)

ANALYSIS_VERSION = "0.1.0"
COMMUTATIVE_OPS = {"and", "or", "xor", "iff"}
_DIGIT_RE = re.compile(r"(\d+)")


def natural_key(value: Any) -> List[Any]:
    text = str(value or "")
    parts = _DIGIT_RE.split(text)
    out: List[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def stable_json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted((_json_safe(v) for v in value), key=natural_key)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_jsonl(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(record))
            handle.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    out: List[Dict[str, Any]] = []
    if not target.exists():
        return out
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                out.append(payload)
    return out


def write_csv(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_record(record) for record in records]
    fieldnames: List[str] = []
    seen = set()
    for row in flat_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def write_markdown(text: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple, set)):
            out[key] = stable_json_dumps(value)
        else:
            out[key] = value
    return out


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def _fmt(values: Sequence[Any]) -> str:
        cells = []
        for idx, value in enumerate(values):
            cells.append(str(value).ljust(widths[idx]))
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("-" * max(3, width) for width in widths) + " |"
    lines = [_fmt(headers), sep]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def rate(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def mean(values: Iterable[float], default: float = 0.0) -> float:
    items = [float(v) for v in values]
    if not items:
        return float(default)
    return float(sum(items) / len(items))


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def bucket_numeric(value: Optional[float], edges: Sequence[float]) -> str:
    if value is None:
        return "missing"
    v = float(value)
    for left, right in zip(edges[:-1], edges[1:]):
        if v < right:
            if math.isinf(left):
                return f"<{right:g}"
            if math.isinf(right):
                return f">={left:g}"
            return f"[{left:g},{right:g})"
    return f">={edges[-1]:g}"


def summarize_operator_profile(op_counts: Dict[str, Any]) -> str:
    if not isinstance(op_counts, dict) or not op_counts:
        return "none"
    items = sorted(
        ((str(op), int(count)) for op, count in op_counts.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "none"
    if len(items) == 1:
        return items[0][0]
    top = [items[0][0], items[1][0]]
    return "+".join(top)


@dataclass
class CandidateParseResult:
    replayable: bool
    invalid_bucket: Optional[str]
    failure_explanation: Optional[str]
    mechanisms: Dict[str, str]
    parsed_nodes_by_var: Dict[str, MechanismNode]
    candidate_stats_by_var: Dict[str, Dict[str, Any]]
    parent_by_var: Dict[str, List[str]]
    eval_topological_order: List[str]
    variables: List[str]
    root_vars: List[str]
    endogenous_vars: List[str]
    topological_order: List[str]
    hide_topological_order: bool


@dataclass
class ReplayBundle:
    evaluation: Dict[str, Any]
    candidate_parse: CandidateParseResult
    train_summary: Optional[Dict[str, Any]]
    heldout_summary: Optional[Dict[str, Any]]
    world_records: List[Dict[str, Any]]


def load_dataset(path: str | Path) -> Tuple[Any, List[Dict[str, Any]]]:
    data = load_from_yaml(str(path))
    if isinstance(data, dict) and isinstance(data.get("problems"), list):
        return data, [p for p in data.get("problems", []) if isinstance(p, dict)]
    if isinstance(data, list):
        return data, [p for p in data if isinstance(p, dict)]
    return data, []


def task_name(problem: Dict[str, Any]) -> str:
    return str((((problem.get("problem") or {}).get("task") or {}).get("taskName")) or "")


def is_a_scm_problem(problem: Dict[str, Any]) -> bool:
    name = task_name(problem).upper()
    return name in {"CIND_A_SCM", "CIND_A_SCM_ALIAS"} or str(
        ((problem.get("problemDescription") or {}).get("scenarioType")) or ""
    ).upper() == "CIND_A_SCM"


def instance_id(problem: Dict[str, Any]) -> str:
    payload = problem.get("problem") or {}
    return str(payload.get("instanceId") or problem.get("instanceId") or "unknown")


def slice_name(problem: Dict[str, Any]) -> str:
    direct = problem.get("subslice")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    desc = problem.get("problemDescription") or {}
    if isinstance(desc, dict):
        extra = desc.get("extra") or {}
        if isinstance(extra, dict):
            value = extra.get("subslice") or extra.get("sweepConfig")
            if isinstance(value, str) and value.strip():
                return value.strip()
    params = task_params(problem)
    for key in ("subslice", "sweepConfig"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def family(problem: Dict[str, Any]) -> str:
    variant = str(task_params(problem).get("scmPromptVariant") or "").strip().lower()
    return "ntopo" if variant == "no_topological_order" else "ordered"


def difficulty(problem: Dict[str, Any]) -> str:
    return str(((problem.get("problemDescription") or {}).get("difficulty")) or "")


def task_params(problem: Dict[str, Any]) -> Dict[str, Any]:
    params = (((problem.get("problem") or {}).get("task") or {}).get("parameters") or {})
    return params if isinstance(params, dict) else {}


def problem_signature_variables(problem: Dict[str, Any]) -> List[str]:
    payload = (problem.get("problem") or {}).get("signature") or {}
    variables = payload.get("variables") or []
    return [str(v) for v in variables]


def gold_scm(problem: Dict[str, Any]) -> Dict[str, Any]:
    payload = ((problem.get("problem") or {}).get("goldAnswer") or {}).get("scm") or {}
    return payload if isinstance(payload, dict) else {}


def gold_root_vars(problem: Dict[str, Any]) -> List[str]:
    params = task_params(problem)
    value = params.get("rootVariables") or gold_scm(problem).get("rootVariables") or []
    return [str(v) for v in value]


def gold_endogenous_vars(problem: Dict[str, Any]) -> List[str]:
    params = task_params(problem)
    value = params.get("endogenousVariables") or gold_scm(problem).get("endogenousVariables") or []
    if value:
        return [str(v) for v in value]
    roots = set(gold_root_vars(problem))
    topo = gold_topological_order(problem)
    return [v for v in topo if v not in roots]


def gold_topological_order(problem: Dict[str, Any]) -> List[str]:
    params = task_params(problem)
    topo = params.get("topologicalOrder") or gold_scm(problem).get("topologicalOrder") or problem_signature_variables(problem)
    return [str(v) for v in topo]


def gold_allowed_operators(problem: Dict[str, Any]) -> List[str]:
    params = task_params(problem)
    ops = params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS)
    return [str(op) for op in ops]


def gold_allow_constants(problem: Dict[str, Any]) -> bool:
    params = task_params(problem)
    return bool(params.get("allowConstants", True))


def parse_gold_mechanisms(
    problem: Dict[str, Any],
) -> Tuple[Dict[str, MechanismNode], Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    mechanisms = (gold_scm(problem).get("mechanisms") or {}) if isinstance(gold_scm(problem), dict) else {}
    topo = gold_topological_order(problem)
    allowed_ops = set(gold_allowed_operators(problem))
    allow_constants = gold_allow_constants(problem)
    index_of = {var: idx for idx, var in enumerate(topo)}
    nodes_by_var: Dict[str, MechanismNode] = {}
    stats_by_var: Dict[str, Dict[str, Any]] = {}
    parents_by_var: Dict[str, List[str]] = {}
    for var in gold_endogenous_vars(problem):
        expr = str((mechanisms or {}).get(var) or "").strip()
        if not expr:
            continue
        idx = index_of.get(str(var), -1)
        if idx < 0:
            continue
        node = parse_mechanism(
            expr,
            allowed_operators=allowed_ops,
            allowed_variables=set(topo[:idx]),
            allow_constants=allow_constants,
        )
        nodes_by_var[str(var)] = node
        stats_by_var[str(var)] = analyze_mechanism(node)
        parents_by_var[str(var)] = sorted(mechanism_variables(node), key=natural_key)
    return nodes_by_var, stats_by_var, parents_by_var


def generation_diagnostics(problem: Dict[str, Any]) -> Dict[str, Any]:
    params = task_params(problem)
    diag = params.get("generationDiagnostics") or {}
    return diag if isinstance(diag, dict) else {}


def scored_rows_by_split_and_var(
    problem: Dict[str, Any],
    *,
    vars_to_track: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
    tracked = [str(var) for var in (vars_to_track or gold_endogenous_vars(problem))]
    payload = problem.get("problem") or problem
    train_worlds, heldout_worlds = _worlds_by_split(payload)
    out: Dict[str, Dict[str, List[Dict[str, int]]]] = {
        "train": {var: [] for var in tracked},
        "heldout": {var: [] for var in tracked},
    }
    for split_name, worlds in (("train", train_worlds), ("heldout", heldout_worlds)):
        for world in worlds:
            intervened = set(_extract_world_intervention_targets(world))
            rows = _iter_world_rows(world)
            for var in tracked:
                if var in intervened:
                    continue
                out[split_name][var].extend(
                    {
                        str(key): int(value)
                        for key, value in row.items()
                        if isinstance(value, (bool, int))
                        or (isinstance(value, str) and value in {"0", "1"})
                    }
                    for row in rows
                    if str(var) in row
                )
    return out


def ensure_evaluation(problem: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = llm_result.get("evaluation") or {}
    if isinstance(evaluation, dict) and evaluation:
        return evaluation
    result = evaluate_causal_llm_result(problem=problem, result=llm_result, task_name=task_name(problem))
    return result.to_dict()


def extracted_answer(llm_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    answer = llm_result.get("extractedAnswer")
    if isinstance(answer, dict):
        return answer
    evaluation = llm_result.get("evaluation") or {}
    if isinstance(evaluation, dict):
        answer = evaluation.get("extractedAnswer")
        if isinstance(answer, dict):
            return answer
    response_dict = llm_result.get("responseDict")
    if isinstance(response_dict, dict) and "mechanisms" in response_dict:
        return response_dict
    return None


def classify_invalid_bucket(parse_error: Optional[str], failure_explanation: Optional[str]) -> Optional[str]:
    parts = " ".join(str(x or "") for x in (parse_error, failure_explanation)).strip().lower()
    if not parts:
        return None
    if "unknown or non-endogenous keys" in parts:
        return "unknown_keys"
    if "missing endogenous variables" in parts:
        return "missing_endogenous_variables"
    if "expected top-level key 'mechanisms'" in parts:
        return "missing_mechanisms_object"
    if "cyclic_ntopo_dependencies" in parts or "cyclic; cannot derive" in parts:
        return "cyclic_ntopo_dependencies"
    if "references variables outside observed set or self" in parts:
        return "illegal_variable_reference"
    if "variable '" in parts and "is not allowed" in parts:
        return "mechanism_illegal_variable"
    if "must be a non-empty string" in parts:
        return "empty_mechanism_string"
    if "single-line without tabs/newlines" in parts:
        return "multiline_mechanism_string"
    if "could not parse json answer" in parts or "no json object found" in parts or "top-level json is not an object" in parts:
        return "answer_extraction_failed"
    if "mechanism parse failed" in parts:
        return "mechanism_parse_failed"
    if "evaluation failed" in parts or "simulation failed" in parts:
        return "simulation_failed"
    if "topological order metadata contains duplicate variables" in parts:
        return "bad_topological_metadata"
    return "other_invalid"


def split_error_bucket(evaluation: Dict[str, Any]) -> str:
    valid = bool(evaluation.get("valid"))
    if not valid:
        return "invalid"
    train_exact = evaluation.get("trainExact")
    heldout_exact = evaluation.get("heldoutExact")
    if train_exact is True and heldout_exact is True:
        return "train_and_heldout_exact"
    if train_exact is True and heldout_exact is False:
        return "heldout_only_error"
    if train_exact is False and heldout_exact is True:
        return "train_only_error"
    if train_exact is False and heldout_exact is False:
        return "train_and_heldout_error"
    if evaluation.get("correct") is True:
        return "train_exact_only"
    return "unknown"


def dominant_label_from_counts(counts: Dict[str, int]) -> Optional[str]:
    items = [(str(k), int(v)) for k, v in counts.items() if int(v) > 0]
    if not items:
        return None
    items.sort(key=lambda item: (-item[1], natural_key(item[0])))
    return items[0][0]


def mismatch_scope_bucket(counts: Dict[str, int]) -> str:
    positives = [(str(k), int(v)) for k, v in counts.items() if int(v) > 0]
    if not positives:
        return "none"
    total = sum(v for _k, v in positives)
    if len(positives) == 1:
        return "single_variable"
    top = max(v for _k, v in positives)
    if total > 0 and (top / total) >= 0.75:
        return "dominant_variable"
    return "multi_variable"


def direction_bucket(under_count: int, over_count: int) -> str:
    under = int(under_count)
    over = int(over_count)
    if under <= 0 and over <= 0:
        return "none"
    if under >= max(1, 2 * over):
        return "underpredict"
    if over >= max(1, 2 * under):
        return "overpredict"
    return "mixed"


def mode_sensitivity_bucket(world_records: Sequence[Dict[str, Any]]) -> str:
    by_mode: Dict[str, List[float]] = {}
    for row in world_records:
        mode = str(row.get("intervention_mode") or "none")
        scored = int(row.get("scored_cells") or 0)
        wrong = int(row.get("wrong_cells") or 0)
        if scored <= 0:
            continue
        by_mode.setdefault(mode, []).append(float(wrong) / float(scored))
    if not by_mode:
        return "unknown"
    overall = mean((rate for values in by_mode.values() for rate in values), default=0.0)
    ranked = sorted(((mode, mean(values, default=0.0)) for mode, values in by_mode.items()), key=lambda item: (-item[1], item[0]))
    top_mode, top_rate = ranked[0]
    if top_rate <= 0.0:
        return "balanced"
    if overall <= 0.0:
        return f"{top_mode}_sensitive"
    if top_rate >= overall * 1.4 and len(by_mode.get(top_mode, [])) >= 2:
        return f"{top_mode}_sensitive"
    return "balanced"


def novelty_sensitivity_bucket(world_records: Sequence[Dict[str, Any]]) -> str:
    low_rates: List[float] = []
    high_rates: List[float] = []
    for row in world_records:
        novelty = row.get("novelty_score")
        if novelty is None:
            continue
        scored = int(row.get("scored_cells") or 0)
        wrong = int(row.get("wrong_cells") or 0)
        if scored <= 0:
            continue
        value = float(wrong) / float(scored)
        if float(novelty) >= 0.5:
            high_rates.append(value)
        else:
            low_rates.append(value)
    if not low_rates and not high_rates:
        return "unknown"
    low_mean = mean(low_rates, default=0.0)
    high_mean = mean(high_rates, default=0.0)
    if high_mean >= low_mean + 0.15 and high_mean > 0.0:
        return "high_novelty_sensitive"
    if low_mean >= high_mean + 0.15 and low_mean > 0.0:
        return "low_novelty_sensitive"
    return "balanced"


def canonicalize_node(node: MechanismNode) -> MechanismNode:
    return _canonicalize_node(node)


def _rename_node_variables(node: MechanismNode, mapping: Dict[str, str]) -> MechanismNode:
    if node.kind == "var":
        return MechanismNode(kind="var", value=mapping.get(node.value, node.value), args=())
    if node.kind == "const":
        return node
    return MechanismNode(kind="op", value=node.value, args=tuple(_rename_node_variables(arg, mapping) for arg in node.args))


def canonical_template_from_node(node: MechanismNode) -> str:
    canonical = canonicalize_node(node)
    local_mapping: Dict[str, str] = {}

    def _assign(cur: MechanismNode) -> None:
        if cur.kind == "var":
            if cur.value not in local_mapping:
                local_mapping[cur.value] = f"V{len(local_mapping)}"
            return
        for arg in cur.args:
            _assign(arg)

    _assign(canonical)
    renamed = _rename_node_variables(canonical, local_mapping)
    return node_to_sexpr(renamed)


def canonical_template_from_expr(expr: str, *, allowed_operators: Optional[Sequence[str]] = None, allowed_variables: Optional[Sequence[str]] = None, allow_constants: bool = True) -> str:
    node = parse_mechanism(
        expr,
        allowed_operators=set(str(v) for v in (allowed_operators or DEFAULT_ALLOWED_OPERATORS)),
        allowed_variables=set(str(v) for v in allowed_variables) if allowed_variables is not None else None,
        allow_constants=allow_constants,
    )
    return canonical_template_from_node(node)


def exact_map_signature(mechanisms: Dict[str, str], endogenous_order: Sequence[str]) -> str:
    payload = {str(var): str(mechanisms.get(var, "")).strip() for var in endogenous_order if var in mechanisms}
    extras = sorted((str(var), str(expr).strip()) for var, expr in mechanisms.items() if str(var) not in payload)
    for var, expr in extras:
        payload[var] = expr
    return stable_json_dumps(payload)


def canonical_map_signature(problem: Dict[str, Any], parsed_nodes_by_var: Dict[str, MechanismNode]) -> str:
    variables = gold_topological_order(problem) or sorted(problem_signature_variables(problem), key=natural_key)
    mapping = {var: f"V{idx}" for idx, var in enumerate(variables)}
    payload: Dict[str, str] = {}
    for var in sorted(parsed_nodes_by_var.keys(), key=natural_key):
        payload[mapping.get(var, var)] = node_to_sexpr(_rename_node_variables(canonicalize_node(parsed_nodes_by_var[var]), mapping))
    return stable_json_dumps(payload)


def _normalize_mechanism_map(answer: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(answer, dict):
        return None
    mechanisms = answer.get("mechanisms")
    if not isinstance(mechanisms, dict):
        return None
    out: Dict[str, str] = {}
    for key, value in mechanisms.items():
        if isinstance(value, str):
            out[str(key)] = value.strip()
    return out


def parse_candidate(problem: Dict[str, Any], llm_result: Dict[str, Any]) -> CandidateParseResult:
    params = task_params(problem)
    variables = problem_signature_variables(problem) or gold_topological_order(problem)
    topological_order = gold_topological_order(problem)
    root_vars = gold_root_vars(problem)
    root_set = set(root_vars)
    endogenous_vars = gold_endogenous_vars(problem)
    endogenous_set = set(endogenous_vars)
    allowed_operators = gold_allowed_operators(problem)
    allow_constants = gold_allow_constants(problem)
    prompt_variant = str(params.get("scmPromptVariant") or params.get("promptVariant") or "with_topological_order").strip().lower()
    hide_topological_order = prompt_variant in {"no_topological_order", "unknown_topological_order", "blind_topological_order", "ntopo", "no_topo"}

    answer = extracted_answer(llm_result)
    mechanisms = _normalize_mechanism_map(answer) or {}
    if not isinstance(answer, dict) or not isinstance(answer.get("mechanisms"), dict):
        return CandidateParseResult(
            replayable=False,
            invalid_bucket="missing_mechanisms_object",
            failure_explanation="Expected top-level key 'mechanisms' with a JSON object value",
            mechanisms=mechanisms,
            parsed_nodes_by_var={},
            candidate_stats_by_var={},
            parent_by_var={},
            eval_topological_order=topological_order,
            variables=variables,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            topological_order=topological_order,
            hide_topological_order=hide_topological_order,
        )

    provided = set(mechanisms.keys())
    unknown = sorted([key for key in provided if key not in endogenous_set], key=natural_key)
    if unknown:
        return CandidateParseResult(
            replayable=False,
            invalid_bucket="unknown_keys",
            failure_explanation="Mechanism map contains unknown or non-endogenous keys",
            mechanisms=mechanisms,
            parsed_nodes_by_var={},
            candidate_stats_by_var={},
            parent_by_var={},
            eval_topological_order=topological_order,
            variables=variables,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            topological_order=topological_order,
            hide_topological_order=hide_topological_order,
        )

    missing = sorted([var for var in endogenous_vars if var not in provided], key=natural_key)
    if missing:
        return CandidateParseResult(
            replayable=False,
            invalid_bucket="missing_endogenous_variables",
            failure_explanation="Mechanism map is missing endogenous variables",
            mechanisms=mechanisms,
            parsed_nodes_by_var={},
            candidate_stats_by_var={},
            parent_by_var={},
            eval_topological_order=topological_order,
            variables=variables,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            topological_order=topological_order,
            hide_topological_order=hide_topological_order,
        )

    parsed_nodes_by_var: Dict[str, MechanismNode] = {}
    candidate_stats_by_var: Dict[str, Dict[str, Any]] = {}
    parent_by_var: Dict[str, List[str]] = {}
    index_of = {var: idx for idx, var in enumerate(topological_order)}
    observed_set = set(variables or topological_order)

    for var in endogenous_vars:
        expr = mechanisms.get(var, "")
        if not expr:
            return CandidateParseResult(
                replayable=False,
                invalid_bucket="empty_mechanism_string",
                failure_explanation=f"Mechanism for variable '{var}' must be a non-empty string",
                mechanisms=mechanisms,
                parsed_nodes_by_var={},
                candidate_stats_by_var={},
                parent_by_var={},
                eval_topological_order=topological_order,
                variables=variables,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                topological_order=topological_order,
                hide_topological_order=hide_topological_order,
            )
        if any(ch in expr for ch in ("\n", "\r", "\t")):
            return CandidateParseResult(
                replayable=False,
                invalid_bucket="multiline_mechanism_string",
                failure_explanation=f"Mechanism for '{var}' must be single-line without tabs/newlines",
                mechanisms=mechanisms,
                parsed_nodes_by_var={},
                candidate_stats_by_var={},
                parent_by_var={},
                eval_topological_order=topological_order,
                variables=variables,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                topological_order=topological_order,
                hide_topological_order=hide_topological_order,
            )
        idx = index_of.get(var, -1)
        if idx < 0:
            return CandidateParseResult(
                replayable=False,
                invalid_bucket="bad_topological_metadata",
                failure_explanation=f"Variable '{var}' not found in topological order metadata",
                mechanisms=mechanisms,
                parsed_nodes_by_var={},
                candidate_stats_by_var={},
                parent_by_var={},
                eval_topological_order=topological_order,
                variables=variables,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                topological_order=topological_order,
                hide_topological_order=hide_topological_order,
            )
        allowed_variables = [v for v in observed_set if v != var] if hide_topological_order else list(topological_order[:idx])
        try:
            node = parse_mechanism(
                expr,
                allowed_operators=set(allowed_operators),
                allowed_variables=set(allowed_variables),
                allow_constants=allow_constants,
            )
        except MechanismParseError as exc:
            bucket = classify_invalid_bucket(str(exc), f"Mechanism parse failed for '{var}': {exc}")
            return CandidateParseResult(
                replayable=False,
                invalid_bucket=bucket or "mechanism_parse_failed",
                failure_explanation=f"Mechanism parse failed for '{var}': {exc}",
                mechanisms=mechanisms,
                parsed_nodes_by_var={},
                candidate_stats_by_var={},
                parent_by_var={},
                eval_topological_order=topological_order,
                variables=variables,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                topological_order=topological_order,
                hide_topological_order=hide_topological_order,
            )
        parents = sorted(mechanism_variables(node), key=natural_key)
        if hide_topological_order:
            invalid_parents = [parent for parent in parents if parent not in observed_set or parent == var]
            if invalid_parents:
                return CandidateParseResult(
                    replayable=False,
                    invalid_bucket="illegal_variable_reference",
                    failure_explanation=f"Mechanism for '{var}' references variables outside observed set or self",
                    mechanisms=mechanisms,
                    parsed_nodes_by_var={},
                    candidate_stats_by_var={},
                    parent_by_var={},
                    eval_topological_order=topological_order,
                    variables=variables,
                    root_vars=root_vars,
                    endogenous_vars=endogenous_vars,
                    topological_order=topological_order,
                    hide_topological_order=hide_topological_order,
                )
        parsed_nodes_by_var[var] = node
        candidate_stats_by_var[var] = analyze_mechanism(node)
        parent_by_var[var] = parents

    eval_topological_order = list(topological_order)
    if hide_topological_order:
        inferred, cycle_nodes = _infer_ntopo_endogenous_order(
            topological_order=topological_order,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            parents_by_var=parent_by_var,
        )
        if inferred is None:
            return CandidateParseResult(
                replayable=False,
                invalid_bucket="cyclic_ntopo_dependencies",
                failure_explanation="NTopo mechanism dependencies are cyclic; cannot derive a valid acyclic SCM order",
                mechanisms=mechanisms,
                parsed_nodes_by_var=parsed_nodes_by_var,
                candidate_stats_by_var=candidate_stats_by_var,
                parent_by_var=parent_by_var,
                eval_topological_order=topological_order,
                variables=variables,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                topological_order=topological_order,
                hide_topological_order=hide_topological_order,
            )
        root_order = [var for var in topological_order if var in root_set]
        eval_topological_order = [*root_order, *inferred]

    return CandidateParseResult(
        replayable=True,
        invalid_bucket=None,
        failure_explanation=None,
        mechanisms=mechanisms,
        parsed_nodes_by_var=parsed_nodes_by_var,
        candidate_stats_by_var=candidate_stats_by_var,
        parent_by_var=parent_by_var,
        eval_topological_order=eval_topological_order,
        variables=variables,
        root_vars=root_vars,
        endogenous_vars=endogenous_vars,
        topological_order=topological_order,
        hide_topological_order=hide_topological_order,
    )


def _heldout_plan_by_world(problem: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diag = generation_diagnostics(problem)
    entries = diag.get("heldoutPlanDiagnostics") or []
    if not isinstance(entries, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        world_id = str(item.get("worldId") or "")
        if not world_id:
            continue
        out[world_id] = item
    return out


def replay_candidate(problem: Dict[str, Any], llm_result: Dict[str, Any]) -> ReplayBundle:
    evaluation = ensure_evaluation(problem, llm_result)
    parsed = parse_candidate(problem, llm_result)
    if not parsed.replayable:
        return ReplayBundle(
            evaluation=evaluation,
            candidate_parse=parsed,
            train_summary=None,
            heldout_summary=None,
            world_records=[],
        )

    train_worlds, heldout_worlds = _worlds_by_split(problem.get("problem") or problem)
    heldout_plan = _heldout_plan_by_world(problem)
    train_summary = _replay_split(
        problem=problem,
        llm_result=llm_result,
        split_name="train",
        worlds=train_worlds,
        parsed=parsed,
        heldout_plan={},
    )
    heldout_summary = _replay_split(
        problem=problem,
        llm_result=llm_result,
        split_name="heldout",
        worlds=heldout_worlds,
        parsed=parsed,
        heldout_plan=heldout_plan,
    )
    world_records = list(train_summary.get("worldRecords") or []) + list(heldout_summary.get("worldRecords") or [])
    return ReplayBundle(
        evaluation=evaluation,
        candidate_parse=parsed,
        train_summary=train_summary,
        heldout_summary=heldout_summary,
        world_records=world_records,
    )


def compute_local_rollout_split_stats(
    problem: Dict[str, Any],
    parsed: CandidateParseResult,
    *,
    split_name: str,
) -> Dict[str, Any]:
    train_worlds, heldout_worlds = _worlds_by_split(problem.get("problem") or problem)
    worlds = train_worlds if str(split_name) == "train" else heldout_worlds

    root_set = set(parsed.root_vars)
    endogenous_set = set(parsed.endogenous_vars)
    total_scored = 0
    direct_wrong = 0
    rollout_wrong = 0
    propagated_only_wrong = 0
    compensated_direct_wrong = 0
    per_var: Dict[str, Dict[str, Any]] = {
        str(var): {
            "scored_cells": 0,
            "direct_wrong_cells": 0,
            "rollout_wrong_cells": 0,
            "propagated_only_wrong_cells": 0,
            "compensated_direct_wrong_cells": 0,
        }
        for var in parsed.endogenous_vars
    }

    for world in worlds:
        mode = _extract_world_intervention_mode(world)
        targets = set(_extract_world_intervention_targets(world))
        const_assignments = _extract_world_intervention_assignments(world)
        rows = _iter_world_rows(world)

        for row in rows:
            gold_assignment = {
                str(key): int(value)
                for key, value in row.items()
                if isinstance(value, (bool, int)) or (isinstance(value, str) and value in {"0", "1"})
            }
            rollout_assignment: Dict[str, int] = {}
            for var in parsed.eval_topological_order:
                if mode == "hard_assigned" and var in targets:
                    rollout_assignment[var] = int(gold_assignment[var])
                    continue
                if var in const_assignments:
                    rollout_assignment[var] = int(const_assignments[var])
                    continue
                if var in root_set:
                    rollout_assignment[var] = int(gold_assignment[var])
                    continue
                node = parsed.parsed_nodes_by_var[var]
                rollout_assignment[var] = int(evaluate_parsed_mechanism(node, rollout_assignment))

            for var in parsed.eval_topological_order:
                if var not in endogenous_set or var in targets or var not in gold_assignment:
                    continue
                node = parsed.parsed_nodes_by_var[var]
                expected = int(gold_assignment[var])
                total_pred = int(rollout_assignment[var])
                total_wrong_cell = total_pred != expected
                try:
                    direct_pred = int(evaluate_parsed_mechanism(node, gold_assignment))
                    direct_wrong_cell = direct_pred != expected
                except MechanismEvalError:
                    direct_wrong_cell = True

                total_scored += 1
                bucket = per_var[str(var)]
                bucket["scored_cells"] += 1
                if direct_wrong_cell:
                    direct_wrong += 1
                    bucket["direct_wrong_cells"] += 1
                if total_wrong_cell:
                    rollout_wrong += 1
                    bucket["rollout_wrong_cells"] += 1
                if total_wrong_cell and not direct_wrong_cell:
                    propagated_only_wrong += 1
                    bucket["propagated_only_wrong_cells"] += 1
                if direct_wrong_cell and not total_wrong_cell:
                    compensated_direct_wrong += 1
                    bucket["compensated_direct_wrong_cells"] += 1

    for payload in per_var.values():
        scored = int(payload.get("scored_cells") or 0)
        payload["direct_wrong_rate"] = rate(int(payload.get("direct_wrong_cells") or 0), scored)
        payload["rollout_wrong_rate"] = rate(int(payload.get("rollout_wrong_cells") or 0), scored)
        payload["propagated_only_wrong_rate"] = rate(
            int(payload.get("propagated_only_wrong_cells") or 0),
            scored,
        )
        payload["compensated_direct_wrong_rate"] = rate(
            int(payload.get("compensated_direct_wrong_cells") or 0),
            scored,
        )
        payload["local_exact"] = scored > 0 and int(payload.get("direct_wrong_cells") or 0) == 0

    return {
        "split": str(split_name),
        "scored_cells": total_scored,
        "direct_wrong_cells": direct_wrong,
        "rollout_wrong_cells": rollout_wrong,
        "propagated_only_wrong_cells": propagated_only_wrong,
        "compensated_direct_wrong_cells": compensated_direct_wrong,
        "direct_wrong_rate": rate(direct_wrong, total_scored),
        "rollout_wrong_rate": rate(rollout_wrong, total_scored),
        "propagated_only_wrong_rate": rate(propagated_only_wrong, total_scored),
        "compensated_direct_wrong_rate": rate(compensated_direct_wrong, total_scored),
        "per_variable": per_var,
    }


def _replay_split(
    *,
    problem: Dict[str, Any],
    llm_result: Dict[str, Any],
    split_name: str,
    worlds: Sequence[Dict[str, Any]],
    parsed: CandidateParseResult,
    heldout_plan: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    root_set = set(parsed.root_vars)
    endogenous_set = set(parsed.endogenous_vars)
    topo_index = {str(var): idx for idx, var in enumerate(parsed.topological_order)}
    total = 0
    correct = 0
    under_count = 0
    over_count = 0
    first_mismatch = None
    world_total = 0
    world_exact_count = 0
    wrong_world_count = 0
    per_var_total: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
    per_var_correct: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
    per_var_under: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
    per_var_over: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
    world_records: List[Dict[str, Any]] = []

    for world in worlds:
        world_id = str(world.get("worldId") or "unknown")
        mode = _extract_world_intervention_mode(world)
        targets = sorted(_extract_world_intervention_targets(world), key=natural_key)
        target_topo_indices = sorted(
            [int(topo_index[target]) for target in targets if target in topo_index],
        )
        const_assignments = _extract_world_intervention_assignments(world)
        rows = _iter_world_rows(world)
        world_scored = 0
        world_correct = 0
        world_under = 0
        world_over = 0
        world_var_wrong: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
        world_var_scored: Dict[str, int] = {var: 0 for var in parsed.endogenous_vars}
        mismatch_examples: List[Dict[str, Any]] = []

        for row_index, row in enumerate(rows):
            assignment: Dict[str, int] = {}
            for var in parsed.eval_topological_order:
                if mode == "hard_assigned" and var in targets:
                    if var not in row:
                        raise ValueError(f"Missing assigned intervention value for '{var}' in row")
                    assignment[var] = int(row[var])
                    continue
                if var in const_assignments:
                    assignment[var] = int(const_assignments[var])
                    continue
                if var in root_set:
                    if var not in row:
                        raise ValueError(f"Missing root variable '{var}' in observed row")
                    assignment[var] = int(row[var])
                    continue
                node = parsed.parsed_nodes_by_var[var]
                try:
                    assignment[var] = int(evaluate_parsed_mechanism(node, assignment))
                except MechanismEvalError as exc:
                    raise ValueError(f"Evaluation failed for {var}: {exc}") from exc

            for var in parsed.eval_topological_order:
                if var not in endogenous_set or var in targets or var not in row:
                    continue
                expected = int(row[var])
                predicted = int(assignment[var])
                total += 1
                world_scored += 1
                world_var_scored[var] += 1
                per_var_total[var] += 1
                if predicted == expected:
                    correct += 1
                    world_correct += 1
                    per_var_correct[var] += 1
                else:
                    if predicted < expected:
                        under_count += 1
                        world_under += 1
                        per_var_under[var] += 1
                    else:
                        over_count += 1
                        world_over += 1
                        per_var_over[var] += 1
                    world_var_wrong[var] += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "worldId": world_id,
                            "rowIndex": row_index,
                            "variable": var,
                            "predicted": predicted,
                            "expected": expected,
                        }
                    if len(mismatch_examples) < 3:
                        mismatch_examples.append(
                            {
                                "rowIndex": row_index,
                                "variable": var,
                                "predicted": predicted,
                                "expected": expected,
                            }
                        )

        if world_scored > 0:
            world_total += 1
            exact = world_correct == world_scored
            if exact:
                world_exact_count += 1
            else:
                wrong_world_count += 1
            plan = heldout_plan.get(world_id, {}) if split_name == "heldout" else {}
            dominant_wrong_variable = dominant_label_from_counts(world_var_wrong)
            dominant_wrong_topological_index = (
                topo_index.get(str(dominant_wrong_variable))
                if dominant_wrong_variable is not None
                else None
            )
            world_records.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "instance_id": instance_id(problem),
                    "model": str(llm_result.get("model") or "unknown"),
                    "family": family(problem),
                    "slice": slice_name(problem),
                    "split": split_name,
                    "world_id": world_id,
                    "intervention_mode": mode,
                    "intervention_targets": targets,
                    "intervention_target_count": len(targets),
                    "intervention_target_topological_indices": target_topo_indices,
                    "intervention_target_min_topological_index": (
                        min(target_topo_indices) if target_topo_indices else None
                    ),
                    "intervention_target_max_topological_index": (
                        max(target_topo_indices) if target_topo_indices else None
                    ),
                    "intervention_target_mean_topological_index": (
                        (sum(target_topo_indices) / float(len(target_topo_indices)))
                        if target_topo_indices
                        else None
                    ),
                    "constant_assignments": const_assignments,
                    "scored_cells": world_scored,
                    "correct_cells": world_correct,
                    "wrong_cells": world_scored - world_correct,
                    "accuracy": rate(world_correct, world_scored),
                    "exact": exact,
                    "under_count": world_under,
                    "over_count": world_over,
                    "wrong_var_count": sum(1 for value in world_var_wrong.values() if value > 0),
                    "dominant_wrong_variable": dominant_wrong_variable,
                    "dominant_wrong_topological_index": dominant_wrong_topological_index,
                    "mismatch_scope_bucket": mismatch_scope_bucket(world_var_wrong),
                    "direction_bucket": direction_bucket(world_under, world_over),
                    "wrong_by_variable": {k: v for k, v in world_var_wrong.items() if v > 0},
                    "scored_by_variable": {k: v for k, v in world_var_scored.items() if v > 0},
                    "mismatch_examples": mismatch_examples,
                    "novelty_score": plan.get("noveltyScore"),
                    "nearest_train_world_id": plan.get("nearestTrainWorldId"),
                    "nearest_train_mode": plan.get("nearestTrainMode"),
                    "nearest_train_target_overlap": plan.get("nearestTrainTargetOverlap"),
                    "mode_match_nearest_train": plan.get("modeMatchNearestTrain"),
                    "mode_mismatch_nearest_train": plan.get("modeMismatchNearestTrain"),
                }
            )

    per_variable: Dict[str, Dict[str, Any]] = {}
    for var in parsed.endogenous_vars:
        scored = per_var_total[var]
        correct_count = per_var_correct[var]
        wrong = scored - correct_count
        per_variable[var] = {
            "scored_cells": scored,
            "correct_cells": correct_count,
            "wrong_cells": wrong,
            "accuracy": rate(correct_count, scored),
            "under_count": per_var_under[var],
            "over_count": per_var_over[var],
        }

    return {
        "split": split_name,
        "valid": True,
        "exact": bool(total > 0 and correct == total),
        "accuracy": rate(correct, total) or 0.0,
        "totalCells": total,
        "correctCells": correct,
        "wrongCells": total - correct,
        "underCount": under_count,
        "overCount": over_count,
        "firstMismatch": first_mismatch,
        "worldTotal": world_total,
        "worldExactCount": world_exact_count,
        "worldExactAccuracy": rate(world_exact_count, world_total) or 0.0,
        "wrongWorldCount": wrong_world_count,
        "perVariable": per_variable,
        "worldRecords": world_records,
        "dominantWrongVariable": dominant_label_from_counts({var: payload["wrong_cells"] for var, payload in per_variable.items()}),
        "mismatchScopeBucket": mismatch_scope_bucket({var: payload["wrong_cells"] for var, payload in per_variable.items()}),
        "directionBucket": direction_bucket(under_count, over_count),
    }


def map_record_from_replay(problem: Dict[str, Any], llm_result: Dict[str, Any], replay: ReplayBundle) -> Dict[str, Any]:
    evaluation = replay.evaluation
    parsed = replay.candidate_parse
    mechanisms = parsed.mechanisms
    gold = gold_scm(problem)
    endogenous_order = parsed.endogenous_vars
    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "family": family(problem),
        "slice": slice_name(problem),
        "difficulty": difficulty(problem),
        "valid": bool(evaluation.get("valid")),
        "train_exact": evaluation.get("trainExact"),
        "heldout_exact": evaluation.get("heldoutExact"),
        "invalid_bucket": parsed.invalid_bucket or classify_invalid_bucket(evaluation.get("parseError"), evaluation.get("failureExplanation")),
        "mechanism_count": len(mechanisms),
        "required_mechanism_count": len(endogenous_order),
        "exact_map_signature": exact_map_signature(mechanisms, endogenous_order),
        "canonical_map_signature": canonical_map_signature(problem, parsed.parsed_nodes_by_var) if parsed.replayable else None,
        "gold_exact_map_signature": exact_map_signature(gold.get("mechanisms") or {}, endogenous_order) if isinstance(gold.get("mechanisms"), dict) else None,
        "gold_canonical_map_signature": canonical_map_signature(problem, {var: parse_mechanism(expr, allowed_operators=set(gold_allowed_operators(problem)), allowed_variables=set(v for v in gold_topological_order(problem) if v != var), allow_constants=gold_allow_constants(problem)) for var, expr in (gold.get("mechanisms") or {}).items() if var in endogenous_order}) if isinstance(gold.get("mechanisms"), dict) else None,
    }
