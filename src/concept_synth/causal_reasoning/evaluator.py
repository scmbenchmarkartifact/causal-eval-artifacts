"""Evaluation scaffold for causal reasoning tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .task_registry import DEFAULT_CAUSAL_TASK_REGISTRY, CausalTaskRegistry
from .runtime_storage_io import load_causal_dataset, save_causal_dataset


@dataclass
class CausalEvaluationResult:
    """Task-agnostic evaluation result container."""

    valid: bool
    correct: Optional[bool]
    taskName: str
    parseError: Optional[str] = None
    failureExplanation: Optional[str] = None
    extractedAnswer: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "valid": self.valid,
            "correct": self.correct,
            "taskName": self.taskName,
            "parseError": self.parseError,
            "failureExplanation": self.failureExplanation,
            "extractedAnswer": self.extractedAnswer,
        }
        result.update(self.details)
        return result


def _ensure_default_registry_seeded(registry: CausalTaskRegistry) -> None:
    if registry is DEFAULT_CAUSAL_TASK_REGISTRY:
        from .cind_family import ensure_cind_family_tasks_registered

        ensure_cind_family_tasks_registered(registry)


def _extract_answer_fallback(response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not response:
        return None, "Empty response"

    try:
        payload = json.loads(response)
        if isinstance(payload, dict):
            return payload, None
        return None, "Top-level JSON is not an object"
    except Exception:
        pass

    # Lightweight extraction of the first JSON object in free-form text. This
    # fallback keeps the public scorer usable for plain JSON predictions without
    # depending on provider-specific response wrappers.
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        snippet = response[start : end + 1]
        try:
            payload = json.loads(snippet)
            if isinstance(payload, dict):
                return payload, None
            return None, "Extracted JSON is not an object"
        except Exception as e:
            return None, f"Could not parse JSON answer: {e}"

    return None, "No JSON object found in response"


def _infer_task_name(problem: Dict[str, Any], fallback: str = "generic") -> str:
    prob = problem.get("problem", problem)
    task = prob.get("task", {}) or {}
    return task.get("taskName", fallback)


def _extract_subslice(record: Dict[str, Any]) -> Optional[str]:
    if not isinstance(record, dict):
        return None

    direct = record.get("subslice")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    desc = record.get("problemDescription", {}) or {}
    if isinstance(desc, dict):
        extra = desc.get("extra", {}) or {}
        if isinstance(extra, dict):
            subslice = extra.get("subslice")
            if isinstance(subslice, str) and subslice.strip():
                return subslice.strip()

    problem = record.get("problem", {}) or {}
    if isinstance(problem, dict):
        task = problem.get("task", {}) or {}
        if isinstance(task, dict):
            params = task.get("parameters", {}) or {}
            if isinstance(params, dict):
                subslice = params.get("subslice")
                if isinstance(subslice, str) and subslice.strip():
                    return subslice.strip()

    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _normalize_usage_details(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _token_metrics_from_result(llm_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    usage = _normalize_usage_details(llm_result.get("usageDetails"))

    billed_tokens = _coerce_float(llm_result.get("billedTokens"))
    if billed_tokens is None:
        billed_tokens = _coerce_float(llm_result.get("billedOutputTokens"))
    if billed_tokens is None:
        billed_tokens = _coerce_float(llm_result.get("billed_output_tokens"))
    if billed_tokens is None:
        billed_tokens = (
            _coerce_float(usage.get("output_tokens"))
            or _coerce_float(usage.get("outputTokens"))
            or _coerce_float(usage.get("completion_tokens"))
            or _coerce_float(usage.get("completionTokens"))
        )

    thinking_tokens = _coerce_float(llm_result.get("thinkingTokens"))
    if thinking_tokens is None:
        thinking_tokens = (
            _coerce_float(usage.get("reasoning_tokens"))
            or _coerce_float(usage.get("reasoningTokens"))
        )

    input_tokens = (
        _coerce_float(usage.get("input_tokens"))
        or _coerce_float(usage.get("inputTokens"))
        or _coerce_float(usage.get("prompt_tokens"))
        or _coerce_float(usage.get("promptTokens"))
    )

    total_tokens = _coerce_float(usage.get("total_tokens")) or _coerce_float(usage.get("totalTokens"))

    return {
        "billedTokens": billed_tokens,
        "thinkingTokens": thinking_tokens,
        "inputTokens": input_tokens,
        "totalTokens": total_tokens,
    }


def _clone_unshared(value: Any) -> Any:
    """Recursively clone lists/dicts without preserving shared references."""
    if isinstance(value, dict):
        return {k: _clone_unshared(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_unshared(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_unshared(v) for v in value)
    return value


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_pct(rate: Optional[float]) -> str:
    if rate is None:
        return "-"
    return f"{100.0 * rate:.1f}%"


def _format_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _normalize_failure_mode(text: str, max_len: int = 96) -> str:
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return "unknown_failure"
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _failure_mode_label(eval_result: "CausalEvaluationResult") -> str:
    if eval_result.parseError:
        return "parse_error"
    if eval_result.failureExplanation:
        return _normalize_failure_mode(eval_result.failureExplanation)
    if eval_result.valid and eval_result.correct is False:
        return "incorrect_without_explanation"
    if not eval_result.valid:
        return "invalid_without_explanation"
    return "unknown_failure"


def _make_model_aggregate() -> Dict[str, Any]:
    return {
        "total": 0,
        "valid": 0,
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "train_exact_true": 0,
        "train_exact_present": 0,
        "heldout_exact_true": 0,
        "heldout_exact_present": 0,
        "train_acc_sum": 0.0,
        "train_acc_count": 0,
        "heldout_acc_sum": 0.0,
        "heldout_acc_count": 0,
        "train_world_exact_acc_sum": 0.0,
        "train_world_exact_acc_count": 0,
        "heldout_world_exact_acc_sum": 0.0,
        "heldout_world_exact_acc_count": 0,
        "latency_ms_sum": 0.0,
        "latency_ms_count": 0,
        "billed_tokens_sum": 0.0,
        "billed_tokens_count": 0,
        "thinking_tokens_sum": 0.0,
        "thinking_tokens_count": 0,
        "input_tokens_sum": 0.0,
        "input_tokens_count": 0,
        "total_tokens_sum": 0.0,
        "total_tokens_count": 0,
        "ast_values": [],
        "ast_depth_values": [],
        "ast_parent_count_values": [],
        "gold_ast_values": [],
        "ast_gap_values": [],
        "bloat_true": 0,
        "bloat_present": 0,
        "acc_gold_25_true": 0,
        "acc_gold_25_present": 0,
        "parent_f1_sum": 0.0,
        "parent_f1_count": 0,
        "failure_modes": {},
    }


def _stats_from_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
    }


def _update_model_aggregate(
    aggregate: Dict[str, Any],
    eval_result: "CausalEvaluationResult",
    llm_result: Dict[str, Any],
) -> None:
    aggregate["total"] += 1
    if eval_result.valid:
        aggregate["valid"] += 1
    if eval_result.correct is True:
        aggregate["correct"] += 1
    elif eval_result.correct is False:
        aggregate["incorrect"] += 1
    if eval_result.parseError:
        aggregate["parse_errors"] += 1

    latency_ms = _coerce_float(llm_result.get("latencyMs"))
    if latency_ms is not None:
        aggregate["latency_ms_sum"] += latency_ms
        aggregate["latency_ms_count"] += 1
    token_metrics = _token_metrics_from_result(llm_result)
    billed_tokens = token_metrics.get("billedTokens")
    if billed_tokens is not None:
        aggregate["billed_tokens_sum"] += billed_tokens
        aggregate["billed_tokens_count"] += 1
    thinking_tokens = token_metrics.get("thinkingTokens")
    if thinking_tokens is not None:
        aggregate["thinking_tokens_sum"] += thinking_tokens
        aggregate["thinking_tokens_count"] += 1
    input_tokens = token_metrics.get("inputTokens")
    if input_tokens is not None:
        aggregate["input_tokens_sum"] += input_tokens
        aggregate["input_tokens_count"] += 1
    total_tokens = token_metrics.get("totalTokens")
    if total_tokens is not None:
        aggregate["total_tokens_sum"] += total_tokens
        aggregate["total_tokens_count"] += 1

    details = eval_result.details or {}

    train_exact = details.get("trainExact")
    if isinstance(train_exact, bool):
        aggregate["train_exact_present"] += 1
        if train_exact:
            aggregate["train_exact_true"] += 1

    heldout_exact = details.get("heldoutExact")
    if isinstance(heldout_exact, bool):
        aggregate["heldout_exact_present"] += 1
        if heldout_exact:
            aggregate["heldout_exact_true"] += 1

    train_acc = _coerce_float(details.get("trainAccuracy"))
    if train_acc is not None:
        aggregate["train_acc_sum"] += train_acc
        aggregate["train_acc_count"] += 1

    heldout_acc = _coerce_float(details.get("heldoutAccuracy"))
    if heldout_acc is not None:
        aggregate["heldout_acc_sum"] += heldout_acc
        aggregate["heldout_acc_count"] += 1

    train_world_exact_acc = _coerce_float(details.get("trainWorldExactAccuracy"))
    if train_world_exact_acc is not None:
        aggregate["train_world_exact_acc_sum"] += train_world_exact_acc
        aggregate["train_world_exact_acc_count"] += 1

    heldout_world_exact_acc = _coerce_float(details.get("heldoutWorldExactAccuracy"))
    if heldout_world_exact_acc is not None:
        aggregate["heldout_world_exact_acc_sum"] += heldout_world_exact_acc
        aggregate["heldout_world_exact_acc_count"] += 1

    candidate_stats = details.get("candidateStats")
    if isinstance(candidate_stats, dict):
        ast_size = _coerce_float(candidate_stats.get("astSize"))
        if ast_size is not None:
            aggregate["ast_values"].append(ast_size)
        ast_depth = _coerce_float(candidate_stats.get("maxDepth"))
        if ast_depth is not None:
            aggregate["ast_depth_values"].append(ast_depth)
        parent_count = _coerce_float(candidate_stats.get("parentCount"))
        if parent_count is not None:
            aggregate["ast_parent_count_values"].append(parent_count)

    gold_ast = _coerce_float(details.get("goldAstSize"))
    if gold_ast is not None:
        aggregate["gold_ast_values"].append(gold_ast)

    if isinstance(candidate_stats, dict):
        ast_size = _coerce_float(candidate_stats.get("astSize"))
        if ast_size is not None and gold_ast is not None:
            aggregate["ast_gap_values"].append(ast_size - gold_ast)

    bloat = details.get("bloat")
    if isinstance(bloat, bool):
        aggregate["bloat_present"] += 1
        if bloat:
            aggregate["bloat_true"] += 1

    acc_gold_plus = details.get("accGoldPlus")
    if isinstance(acc_gold_plus, dict) and isinstance(acc_gold_plus.get("delta_25"), bool):
        aggregate["acc_gold_25_present"] += 1
        if acc_gold_plus["delta_25"]:
            aggregate["acc_gold_25_true"] += 1

    parent_f1 = _coerce_float(details.get("parentF1"))
    if parent_f1 is not None:
        aggregate["parent_f1_sum"] += parent_f1
        aggregate["parent_f1_count"] += 1

    if not (eval_result.valid and eval_result.correct is True):
        label = _failure_mode_label(eval_result)
        aggregate["failure_modes"][label] = aggregate["failure_modes"].get(label, 0) + 1


def _finalize_model_stats(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for model in sorted(aggregates.keys()):
        agg = aggregates[model]
        total = int(agg["total"])
        latency_avg = None
        if agg["latency_ms_count"] > 0:
            latency_avg = agg["latency_ms_sum"] / agg["latency_ms_count"]
        billed_tokens_avg = None
        if agg["billed_tokens_count"] > 0:
            billed_tokens_avg = agg["billed_tokens_sum"] / agg["billed_tokens_count"]
        thinking_tokens_avg = None
        if agg["thinking_tokens_count"] > 0:
            thinking_tokens_avg = agg["thinking_tokens_sum"] / agg["thinking_tokens_count"]
        input_tokens_avg = None
        if agg["input_tokens_count"] > 0:
            input_tokens_avg = agg["input_tokens_sum"] / agg["input_tokens_count"]
        total_tokens_avg = None
        if agg["total_tokens_count"] > 0:
            total_tokens_avg = agg["total_tokens_sum"] / agg["total_tokens_count"]

        train_acc_avg = None
        if agg["train_acc_count"] > 0:
            train_acc_avg = agg["train_acc_sum"] / agg["train_acc_count"]

        heldout_acc_avg = None
        if agg["heldout_acc_count"] > 0:
            heldout_acc_avg = agg["heldout_acc_sum"] / agg["heldout_acc_count"]

        train_world_exact_acc_avg = None
        if agg["train_world_exact_acc_count"] > 0:
            train_world_exact_acc_avg = (
                agg["train_world_exact_acc_sum"] / agg["train_world_exact_acc_count"]
            )

        heldout_world_exact_acc_avg = None
        if agg["heldout_world_exact_acc_count"] > 0:
            heldout_world_exact_acc_avg = (
                agg["heldout_world_exact_acc_sum"] / agg["heldout_world_exact_acc_count"]
            )

        parent_f1_avg = None
        if agg["parent_f1_count"] > 0:
            parent_f1_avg = agg["parent_f1_sum"] / agg["parent_f1_count"]

        failure_modes_sorted = sorted(
            agg["failure_modes"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        failure_modes = [{"mode": mode, "count": count} for mode, count in failure_modes_sorted]

        out[model] = {
            "total": total,
            "valid": int(agg["valid"]),
            "correct": int(agg["correct"]),
            "incorrect": int(agg["incorrect"]),
            "parseErrors": int(agg["parse_errors"]),
            "validRate": _rate(int(agg["valid"]), total),
            "correctRate": _rate(int(agg["correct"]), total),
            "parseErrorRate": _rate(int(agg["parse_errors"]), total),
            "trainExactRate": _rate(int(agg["train_exact_true"]), int(agg["train_exact_present"])),
            "heldoutExactRate": _rate(int(agg["heldout_exact_true"]), int(agg["heldout_exact_present"])),
            "trainAccuracyAvg": train_acc_avg,
            "heldoutAccuracyAvg": heldout_acc_avg,
            "trainWorldExactAccuracyAvg": train_world_exact_acc_avg,
            "heldoutWorldExactAccuracyAvg": heldout_world_exact_acc_avg,
            "latencyMsAvg": latency_avg,
            "billedTokensAvg": billed_tokens_avg,
            "billedTokensCount": int(agg["billed_tokens_count"]),
            "thinkingTokensAvg": thinking_tokens_avg,
            "thinkingTokensCount": int(agg["thinking_tokens_count"]),
            "inputTokensAvg": input_tokens_avg,
            "inputTokensCount": int(agg["input_tokens_count"]),
            "totalTokensAvg": total_tokens_avg,
            "totalTokensCount": int(agg["total_tokens_count"]),
            "candidateAst": _stats_from_values(list(agg["ast_values"])),
            "candidateDepth": _stats_from_values(list(agg["ast_depth_values"])),
            "candidateParentCount": _stats_from_values(list(agg["ast_parent_count_values"])),
            "goldAst": _stats_from_values(list(agg["gold_ast_values"])),
            "astGap": _stats_from_values(list(agg["ast_gap_values"])),
            "bloatRate": _rate(int(agg["bloat_true"]), int(agg["bloat_present"])),
            "accGoldPlus25Rate": _rate(int(agg["acc_gold_25_true"]), int(agg["acc_gold_25_present"])),
            "parentF1Avg": parent_f1_avg,
            "failureCount": sum(int(x["count"]) for x in failure_modes),
            "failureModes": failure_modes,
        }
    return out


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(min(len(row), len(widths))):
            widths[i] = max(widths[i], len(str(row[i])))

    def _pad_row(values: List[str]) -> str:
        padded = []
        for i in range(len(widths)):
            value = str(values[i]) if i < len(values) else ""
            padded.append(value.ljust(widths[i]))
        return "| " + " | ".join(padded) + " |"

    sep_cells = ["-" * max(3, w) for w in widths]
    lines = [_pad_row([str(h) for h in headers]), "| " + " | ".join(sep_cells) + " |"]
    for row in rows:
        lines.append(_pad_row([str(cell) for cell in row]))
    return "\n".join(lines)


def _format_num_with_count(
    value: Optional[float],
    count: Optional[int],
    digits: int = 1,
) -> str:
    if value is None or count is None or int(count) <= 0:
        return "-"
    return f"{value:.{digits}f} (n={int(count)})"


def format_causal_evaluation_report(
    stats: Dict[str, Any],
    max_failure_modes_per_model: int = 5,
) -> str:
    by_model = stats.get("by_model") or {}
    if not by_model:
        return "Causal Evaluation Report\n\nNo model results found."

    lines: List[str] = []
    lines.append("Causal Evaluation Report")
    lines.append("")

    perf_headers = [
        "Model",
        "N",
        "Valid",
        "Correct",
        "ParseErr",
        "TrainExact",
        "HeldoutExact",
        "TrainAcc",
        "HeldoutAcc",
        "TrainWorldExact",
        "HeldoutWorldExact",
        "LatencyMs",
    ]
    perf_rows: List[List[str]] = []
    model_order = sorted(
        by_model.keys(),
        key=lambda model: (
            -float(
                by_model.get(model, {}).get("correctRate")
                if by_model.get(model, {}).get("correctRate") is not None
                else -1.0
            ),
            -int(by_model.get(model, {}).get("total") or 0),
            str(model),
        ),
    )
    for model in model_order:
        row = by_model[model]
        perf_rows.append(
            [
                model,
                str(row.get("total", 0)),
                _format_pct(row.get("validRate")),
                _format_pct(row.get("correctRate")),
                _format_pct(row.get("parseErrorRate")),
                _format_pct(row.get("trainExactRate")),
                _format_pct(row.get("heldoutExactRate")),
                _format_num(row.get("trainAccuracyAvg"), digits=3),
                _format_num(row.get("heldoutAccuracyAvg"), digits=3),
                _format_pct(row.get("trainWorldExactAccuracyAvg")),
                _format_pct(row.get("heldoutWorldExactAccuracyAvg")),
                _format_num(row.get("latencyMsAvg"), digits=1),
            ]
        )

    lines.append("Performance by Model")
    lines.append(_markdown_table(perf_headers, perf_rows))
    lines.append("")

    by_slice_model = stats.get("by_slice_model") or {}
    if by_slice_model:
        slice_perf_headers = [
            "Slice",
            "Model",
            "N",
            "Valid",
            "Correct",
            "ParseErr",
            "TrainExact",
            "HeldoutExact",
            "TrainAcc",
            "HeldoutAcc",
        ]
        slice_perf_rows: List[List[str]] = []
        for slice_name in sorted(by_slice_model.keys()):
            slice_rows = by_slice_model.get(slice_name) or {}
            for i, model in enumerate(sorted(slice_rows.keys())):
                row = slice_rows[model]
                slice_perf_rows.append(
                    [
                        slice_name if i == 0 else "",
                        model,
                        str(row.get("total", 0)),
                        _format_pct(row.get("validRate")),
                        _format_pct(row.get("correctRate")),
                        _format_pct(row.get("parseErrorRate")),
                        _format_pct(row.get("trainExactRate")),
                        _format_pct(row.get("heldoutExactRate")),
                        _format_num(row.get("trainAccuracyAvg"), digits=3),
                        _format_num(row.get("heldoutAccuracyAvg"), digits=3),
                    ]
                )
        lines.append("Performance by Slice and Model")
        lines.append(_markdown_table(slice_perf_headers, slice_perf_rows))
        lines.append("")

    token_headers = [
        "Model",
        "BilledTok avg",
        "ThinkingTok avg",
        "InputTok avg",
        "TotalTok avg",
    ]
    token_rows: List[List[str]] = []
    for model in sorted(by_model.keys()):
        row = by_model[model]
        token_rows.append(
            [
                model,
                _format_num_with_count(
                    row.get("billedTokensAvg"),
                    row.get("billedTokensCount"),
                    digits=1,
                ),
                _format_num_with_count(
                    row.get("thinkingTokensAvg"),
                    row.get("thinkingTokensCount"),
                    digits=1,
                ),
                _format_num_with_count(
                    row.get("inputTokensAvg"),
                    row.get("inputTokensCount"),
                    digits=1,
                ),
                _format_num_with_count(
                    row.get("totalTokensAvg"),
                    row.get("totalTokensCount"),
                    digits=1,
                ),
            ]
        )
    lines.append("Token Usage by Model")
    lines.append(_markdown_table(token_headers, token_rows))
    lines.append("")

    ast_headers = [
        "Model",
        "AST n",
        "AST mean",
        "AST med",
        "AST min",
        "AST max",
        "GoldAST mean",
        "Gap mean",
        "Depth mean",
        "Parents mean",
        "Bloat",
        "Acc@g+25",
        "ParentF1",
    ]
    ast_rows: List[List[str]] = []
    for model in sorted(by_model.keys()):
        row = by_model[model]
        candidate_ast = row.get("candidateAst", {}) or {}
        gold_ast = row.get("goldAst", {}) or {}
        ast_gap = row.get("astGap", {}) or {}
        depth_stats = row.get("candidateDepth", {}) or {}
        parent_stats = row.get("candidateParentCount", {}) or {}
        ast_rows.append(
            [
                model,
                str(candidate_ast.get("count", 0)),
                _format_num(candidate_ast.get("mean")),
                _format_num(candidate_ast.get("median")),
                _format_num(candidate_ast.get("min")),
                _format_num(candidate_ast.get("max")),
                _format_num(gold_ast.get("mean")),
                _format_num(ast_gap.get("mean")),
                _format_num(depth_stats.get("mean")),
                _format_num(parent_stats.get("mean")),
                _format_pct(row.get("bloatRate")),
                _format_pct(row.get("accGoldPlus25Rate")),
                _format_num(row.get("parentF1Avg")),
            ]
        )
    lines.append("Parsimony / AST by Model")
    lines.append(_markdown_table(ast_headers, ast_rows))
    lines.append("")

    failure_headers = ["Model", "Failure mode", "Count", "Share"]
    failure_rows: List[List[str]] = []
    for model in sorted(by_model.keys()):
        row = by_model[model]
        modes = list(row.get("failureModes", []) or [])
        failure_count = int(row.get("failureCount", 0))
        if not modes:
            failure_rows.append([model, "(none)", "0", "-"])
            continue
        for i, mode in enumerate(modes[:max_failure_modes_per_model]):
            label = model if i == 0 else ""
            share = _rate(int(mode["count"]), failure_count)
            failure_rows.append(
                [
                    label,
                    str(mode["mode"]),
                    str(mode["count"]),
                    _format_pct(share),
                ]
            )

    lines.append("Failure Modes by Model")
    lines.append(_markdown_table(failure_headers, failure_rows))

    return "\n".join(lines).strip()


def evaluate_causal_llm_result(
    problem: Dict[str, Any],
    result: Dict[str, Any],
    task_name: Optional[str] = None,
    registry: CausalTaskRegistry = DEFAULT_CAUSAL_TASK_REGISTRY,
) -> CausalEvaluationResult:
    """Evaluate one already-collected model result against a causal task hook.

    The evaluator is deterministic: task hooks parse the supplied text or
    extracted answer and compare candidate mechanisms against frozen worlds.
    It does not call external models or mutate the benchmark problem.
    """
    _ensure_default_registry_seeded(registry)

    resolved_task_name = task_name or _infer_task_name(problem)
    task = registry.get(resolved_task_name)

    # Prefer pre-extracted answer if present.
    pre_extracted = result.get("extractedAnswer")
    if isinstance(pre_extracted, dict):
        answer = pre_extracted
        parse_error = None
    else:
        response = result.get("rawResponse") or result.get("response") or ""
        if task and task.extract_answer:
            # Task-specific extractors normalize output formats; they are local
            # parsers, not model-generation steps.
            answer, parse_error = task.extract_answer(response)
        else:
            answer, parse_error = _extract_answer_fallback(response)

    if answer is None:
        return CausalEvaluationResult(
            valid=False,
            correct=False,
            taskName=resolved_task_name,
            parseError=parse_error,
            failureExplanation="Answer extraction failed",
            extractedAnswer=None,
        )

    if task and task.evaluate_answer:
        eval_payload = task.evaluate_answer(problem, answer)
    else:
        eval_payload = {
            "valid": True,
            "correct": None,
            "failureExplanation": "No task-specific evaluator registered",
        }

    return CausalEvaluationResult(
        valid=bool(eval_payload.get("valid", True)),
        correct=eval_payload.get("correct"),
        taskName=resolved_task_name,
        parseError=parse_error,
        failureExplanation=eval_payload.get("failureExplanation"),
        extractedAnswer=answer,
        details={
            k: v
            for k, v in eval_payload.items()
            if k not in {"valid", "correct", "failureExplanation"}
        },
    )


def evaluate_causal_problem_file(
    input_file: str,
    output_file: Optional[str] = None,
    task_name: Optional[str] = None,
    registry: CausalTaskRegistry = DEFAULT_CAUSAL_TASK_REGISTRY,
) -> Dict[str, Any]:
    """Evaluate all llmResults in a causal dataset file."""
    _ensure_default_registry_seeded(registry)

    data, problems, wrapped, use_runtime_storage = load_causal_dataset(input_file)

    stats = {
        "total_problems": len(problems),
        "total_results": 0,
        "valid": 0,
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
    }
    model_aggregates: Dict[str, Dict[str, Any]] = {}
    slice_model_aggregates: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for record in problems:
        subslice = _extract_subslice(record)
        llm_results = record.get("llmResults", []) or []
        for llm_result in llm_results:
            eval_result = evaluate_causal_llm_result(
                problem=record,
                result=llm_result,
                task_name=task_name,
                registry=registry,
            )
            llm_result["evaluation"] = eval_result.to_dict()

            stats["total_results"] += 1
            if eval_result.valid:
                stats["valid"] += 1
            if eval_result.correct is True:
                stats["correct"] += 1
            elif eval_result.correct is False:
                stats["incorrect"] += 1
            if eval_result.parseError:
                stats["parse_errors"] += 1

            model = str(llm_result.get("model") or "unknown")
            aggregate = model_aggregates.setdefault(model, _make_model_aggregate())
            _update_model_aggregate(aggregate, eval_result, llm_result)
            if subslice:
                slice_aggregate = slice_model_aggregates.setdefault(str(subslice), {})
                model_slice_aggregate = slice_aggregate.setdefault(model, _make_model_aggregate())
                _update_model_aggregate(model_slice_aggregate, eval_result, llm_result)

    target = output_file or input_file
    save_causal_dataset(
        _clone_unshared(data),
        _clone_unshared(problems),
        wrapped,
        target,
        use_runtime_storage=use_runtime_storage,
    )

    stats["by_model"] = _finalize_model_stats(model_aggregates)
    stats["by_slice_model"] = {
        slice_name: _finalize_model_stats(model_map)
        for slice_name, model_map in sorted(slice_model_aggregates.items())
        if model_map
    }
    stats["report"] = format_causal_evaluation_report(stats)

    return stats
