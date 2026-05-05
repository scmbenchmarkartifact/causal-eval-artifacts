#!/usr/bin/env python3
"""Validate paper model-result tables from frozen runtime evaluations.

This is a replay-only validator. It reads the embedded ``evaluation`` records in
the released runtime YAML files, aggregates the paper-facing metrics, and
compares them to either a checked-in expected JSON file or the generated TeX
table sources from the paper build. It does not run LLMs, symbolic solvers,
SCM generation, or re-scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARKS_DIR = REPO_ROOT / "data" / "benchmarks"
DEFAULT_EXPECTED = REPO_ROOT / "data" / "reference" / "paper_model_results_expected.json"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "paper_result_validation"

UNDEFINED = "\u2013"

LLM_ORDER = [
    "gpt-5.4",
    "claude-opus-4-6",
    "deepseek4pro",
    "gemini-3.1-pro-preview",
    "grok4.2",
    "grok4",
    "grok4.3",
    "kimi-k2-thinking",
    "deepseek-reasoner",
]

MAIN_PAPER_EXCLUDED_MODELS = {
    "qwen-3.5",
    "gemini-3-pro-preview",
    "kimi-k2.6",
    "gpt-5.2",
}

TOPLINE_BASELINES = [
    "external-bnlearn-paper-pipeline-2026-04-15-r44fix",
    "symbolic-scm-bestof-portfolio-2026-04-15",
    "external-bnlearn-final-bestof-2026-04-01",
    "symbolic-scm-final-bestof-2026-04-01",
]

ROOT_BASELINES = [
    "external-bnlearn-paper-pipeline-2026-04-15-r44fix",
    "symbolic-scm-bestof-portfolio-2026-04-15",
]

MODEL_DISPLAY = {
    "claude-opus-4-6": "Opus 4.6",
    "deepseek-reasoner": "DSReasoner",
    "deepseek4pro": "DeepSeek4Pro",
    "external-bnlearn-final-bestof-2026-04-01": "bnlearn+DSL",
    "external-bnlearn-paper-pipeline-2026-04-15-r44fix": "bnlearn+DSL",
    "gemini-3.1-pro-preview": "Gemini3.1",
    "gpt-5.4": "GPT-5.4",
    "grok4": "Grok 4",
    "grok4.2": "Grok 4.20",
    "grok4.3": "Grok 4.3",
    "kimi-k2-thinking": "KimiK2t",
    "symbolic-scm-bestof-portfolio-2026-04-15": "symbolic exact-search",
    "symbolic-scm-final-bestof-2026-04-01": "symbolic exact-search",
}

BENCHMARKS = {
    "ordered_250": {
        "filename": "cind_a_scm_benchmark240_final_v1.runtime.yaml",
        "label": "Ordered 250",
        "short": "Ord-Full",
    },
    "ntopo_250": {
        "filename": "cind_a_scm_ntopo_benchmark240_final_v1.runtime.yaml",
        "label": "Ntopo 250",
        "short": "Hid-Full",
    },
    "partial_order_100": {
        "filename": "cind_a_scm_paired50_partial_order_mixed25_2block25_3block_v3.runtime.yaml",
        "label": "Partial Order 100",
        "short": "Block",
    },
    "root_unknown_ntopo_100": {
        "filename": "cind_a_scm_root_unknown_paired90_ntopo_v1.runtime.yaml",
        "label": "ROOT_UNKNOWN Ntopo 100",
        "short": "Hid-Roots",
    },
    "alt_exp_ordered_100": {
        "filename": "cind_a_scm_alt_exp_paired50_ordered_v1.runtime.yaml",
        "label": "ALT_EXP Ordered 100",
        "short": "Alt-Ord",
    },
    "alt_exp_ntopo_100": {
        "filename": "cind_a_scm_alt_exp_paired50_ntopo_v1.runtime.yaml",
        "label": "ALT_EXP Ntopo 100",
        "short": "Alt-Hid",
    },
    "ordered_audit_100": {
        "filename": "cind_a_scm_ident_audit_ordered_pairs100_v1.runtime.yaml",
        "label": "Ordered + Extra Worlds",
        "short": "Ord-Ext",
    },
    "ntopo_audit_100": {
        "filename": "cind_a_scm_ident_audit_ntopo_pairs100_v1.runtime.yaml",
        "label": "Hidden-order + Extra Worlds",
        "short": "Hid-Ext",
    },
    "ordered_counterexample_100": {
        "filename": "cind_a_scm_ident_counterexample_ordered_pairs100_v5.runtime.yaml",
        "label": "Ordered + Counterexample Audit",
        "short": "Ord-CEx",
    },
    "ntopo_counterexample_100": {
        "filename": "cind_a_scm_ident_counterexample_ntopo_pairs100_v5.runtime.yaml",
        "label": "Hidden-order + Counterexample Audit",
        "short": "Hid-CEx",
    },
}

TOPLINE_ORDER = ["ordered_250", "partial_order_100", "ntopo_250"]
COMMON_LADDER_ORDER = ["ordered_250", "partial_order_100", "ntopo_250", "root_unknown_ntopo_100"]
COMMON_LADDER_DISPLAY = {
    "ordered_250": "Ord-Match",
    "partial_order_100": "Block",
    "ntopo_250": "Hid-Match",
    "root_unknown_ntopo_100": "Hid-Roots",
}
SUPPORT_AUDIT_ORDER = [
    "ordered_audit_100",
    "ntopo_audit_100",
    "ordered_counterexample_100",
    "ntopo_counterexample_100",
]

COUNTEREXAMPLE_IGNORED_RESULT_MODELS = {
    "external-bnlearn-paper-pipeline-2026-04-23",
}


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.replace("-", " "))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def record_problem(record: dict[str, Any]) -> dict[str, Any]:
    problem = record.get("problem")
    return problem if isinstance(problem, dict) else record


def problem_description(record: dict[str, Any]) -> dict[str, Any]:
    desc = record.get("problemDescription")
    if isinstance(desc, dict):
        return desc
    nested = record_problem(record).get("problemDescription")
    return nested if isinstance(nested, dict) else {}


def record_extra(record: dict[str, Any]) -> dict[str, Any]:
    extra = problem_description(record).get("extra")
    return extra if isinstance(extra, dict) else {}


def record_gold(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("goldAnswer")
    if isinstance(gold, dict):
        return gold
    nested = record_problem(record).get("goldAnswer")
    return nested if isinstance(nested, dict) else {}


def record_instance_id(record: dict[str, Any]) -> str:
    return str(record.get("instanceId") or record_problem(record).get("instanceId") or "").strip()


def normalize_alignment_id(value: str) -> str:
    out = str(value or "").strip()
    out = re.sub(r"__(ALT_EXP|ROOT_UNKNOWN)$", "", out)
    out = re.sub(r"__PARTIAL_ORDER_(2BLOCK|3BLOCK)$", "", out)
    return out


def paired_source_id(record: dict[str, Any]) -> str:
    extra = record_extra(record)
    for key in ("pairedSourceInstanceId", "sourceOrderedInstanceId", "sourceNtopoInstanceId"):
        value = str(extra.get(key) or "").strip()
        if value:
            return normalize_alignment_id(value)
    return normalize_alignment_id(record_instance_id(record))


def alt_exp_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record_gold(record).get("scm_alt_exp")
    return payload if isinstance(payload, dict) else {}


def list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [part for part in value.split(";") if part]
    return []


def problem_row(benchmark_key: str, record: dict[str, Any]) -> dict[str, Any]:
    alt_payload = alt_exp_payload(record)
    support = alt_payload.get("alternativeSupport")
    if not isinstance(support, list):
        support = []
    support_models = sorted(
        {
            str(item.get("model") or "").strip()
            for item in support
            if isinstance(item, dict) and str(item.get("model") or "").strip()
        }
    )
    known_alternative = bool(
        alt_payload.get("knownAlternativeAvailable")
        or alt_payload.get("alternativeSupportCount")
        or support
        or alt_payload.get("otherMechanisms")
        or alt_payload.get("alternativeMechanisms")
    )
    return {
        "benchmark_key": benchmark_key,
        "benchmark_label": BENCHMARKS[benchmark_key]["label"],
        "instance_id": record_instance_id(record),
        "paired_source_id": paired_source_id(record),
        "known_alternative_available": known_alternative,
        "alternative_support_models": support_models,
    }


def strict_heldout_exact(evaluation: dict[str, Any]) -> bool:
    heldout_exact = evaluation.get("heldoutExact")
    if "heldoutAllWorldsExact" not in evaluation:
        heldout_exact = evaluation.get("trainExact") is True and heldout_exact is True
    return evaluation.get("valid") is True and heldout_exact is True


def strict_mechanism_heldout_exact(evaluation: dict[str, Any]) -> bool:
    heldout_exact = evaluation.get("mechanismHeldoutExact")
    if "mechanismHeldoutAllWorldsExact" not in evaluation:
        heldout_exact = evaluation.get("mechanismTrainExact") is True and heldout_exact is True
    return evaluation.get("valid") is True and heldout_exact is True


def iter_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    # Runtime snapshots exist in two equivalent encodings. Older files use a
    # list of llmResults; newer refreshes may use a model-keyed map.
    model_results = record.get("modelResults")
    if isinstance(model_results, dict):
        out = []
        for model, result in model_results.items():
            if isinstance(result, dict):
                item = dict(result)
                item.setdefault("model", model)
                out.append(item)
        return out
    llm_results = record.get("llmResults")
    if isinstance(llm_results, list):
        return [item for item in llm_results if isinstance(item, dict)]
    return []


def result_row(benchmark_key: str, record: dict[str, Any], result: dict[str, Any], global_index: int, local_index: int) -> dict[str, Any]:
    evaluation = dict(result.get("evaluation") or {})
    train_world = evaluation.get("trainWorldExactAccuracy")
    heldout_world = evaluation.get("heldoutWorldExactAccuracy")
    if benchmark_key in {"ordered_counterexample_100", "ntopo_counterexample_100"} and evaluation.get("valid") is not True:
        # Counterexample-audit tables treat invalid submissions as zero world
        # accuracy so that parse/format failures cannot inflate partial scores.
        train_world = 0.0
        heldout_world = 0.0
    return {
        "benchmark_key": benchmark_key,
        "benchmark_label": BENCHMARKS[benchmark_key]["label"],
        "instance_id": record_instance_id(record),
        "paired_source_id": paired_source_id(record),
        "model": str(result.get("model") or ""),
        "result_index_global": global_index,
        "result_index_within_problem": local_index,
        "valid": evaluation.get("valid"),
        "correct": evaluation.get("correct"),
        "train_exact": evaluation.get("trainExact"),
        "heldout_exact": strict_heldout_exact(evaluation),
        "train_world_exact_accuracy": train_world,
        "heldout_world_exact_accuracy": heldout_world,
        "alt_valid": evaluation.get("altValid"),
        "alt_train_exact": evaluation.get("altTrainExact"),
        "alt_semantically_distinct": evaluation.get("altSemanticallyDistinct"),
        "alt_success": evaluation.get("altSuccess"),
        "experiment_valid": evaluation.get("experimentValid"),
        "pair_separates": evaluation.get("pairSeparates"),
        "experiment_optimality": evaluation.get("experimentOptimality"),
        "witness_valid": evaluation.get("witnessValid"),
        "joint_success": evaluation.get("jointSuccess"),
        "root_set_exact": evaluation.get("rootSetExact"),
        "mechanism_train_exact": evaluation.get("mechanismTrainExact"),
        "mechanism_heldout_exact": strict_mechanism_heldout_exact(evaluation),
    }


def load_release_records(benchmarks_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    problems: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    global_index = 0
    for key, spec in BENCHMARKS.items():
        path = benchmarks_dir / spec["filename"]
        data = load_yaml(path)
        records = data.get("problems")
        if not isinstance(records, list):
            raise ValueError(f"{path} does not contain a problems list")
        counts[key] = len(records)
        for record in records:
            if not isinstance(record, dict):
                continue
            problems.append(problem_row(key, record))
            for local_index, result in enumerate(iter_results(record)):
                model = str(result.get("model") or "")
                if key in {"ordered_counterexample_100", "ntopo_counterexample_100"} and model in COUNTEREXAMPLE_IGNORED_RESULT_MODELS:
                    continue
                results.append(result_row(key, record, result, global_index, local_index))
                global_index += 1
    return problems, latest_result_rows(results), counts


def latest_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Some runtime files keep multiple attempts for the same
    # benchmark/problem/model after incremental refreshes. Paper tables use the
    # last embedded attempt, matching the order in the frozen YAML.
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("benchmark_key") or ""),
            str(row.get("instance_id") or ""),
            str(row.get("model") or ""),
        )
        previous = latest.get(key)
        if previous is None or int(row.get("result_index_global") or 0) >= int(previous.get("result_index_global") or 0):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("benchmark_key") or ""),
            str(row.get("instance_id") or ""),
            str(row.get("model") or ""),
        ),
    )


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) is True else 0.0 for row in rows) / float(len(rows)) if rows else 0.0


def mean_numeric(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(nums) / float(len(nums)) if nums else None


def fmt_rate(value: float | None) -> str:
    return UNDEFINED if value is None else f"{round(float(value), 4):.3f}"


def fmt_conditional(value: float | None, denominator: int) -> str:
    if denominator == 0:
        return UNDEFINED
    if 1 <= denominator <= 5:
        return "*"
    return fmt_rate(value)


def fmt_count(value: int) -> str:
    return str(int(value))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_exact_rows = [row for row in rows if row.get("train_exact") is True]
    return {
        "n": len(rows),
        "valid": rate(rows, "valid"),
        "train_exact": rate(rows, "train_exact"),
        "train_world": mean_numeric([row.get("train_world_exact_accuracy") for row in rows]),
        "heldout_world": mean_numeric([row.get("heldout_world_exact_accuracy") for row in rows]),
        "heldout_exact": rate(rows, "heldout_exact"),
        "cond_heldout_world": mean_numeric([row.get("heldout_world_exact_accuracy") for row in train_exact_rows]),
        "cond_heldout_exact": rate(train_exact_rows, "heldout_exact"),
        "train_exact_count": len(train_exact_rows),
    }


def rows_by_benchmark_model(results: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(str(row.get("benchmark_key") or ""), str(row.get("model") or ""))].append(row)
    return grouped


def topline_table(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped = rows_by_benchmark_model(results)
    out: list[dict[str, str]] = []
    for benchmark_key in TOPLINE_ORDER:
        for model in [*LLM_ORDER, *TOPLINE_BASELINES]:
            bucket = grouped.get((benchmark_key, model), [])
            if not bucket:
                continue
            summary = summarize(bucket)
            out.append(
                {
                    "setting": BENCHMARKS[benchmark_key]["short"],
                    "system": display_model(model),
                    "valid": fmt_rate(summary["valid"]),
                    "train_exact": fmt_rate(summary["train_exact"]),
                    "heldout_world": fmt_rate(summary["heldout_world"]),
                    "heldout_exact": fmt_rate(summary["heldout_exact"]),
                    "heldout_world_given_train_exact": fmt_conditional(
                        summary["cond_heldout_world"], summary["train_exact_count"]
                    ),
                    "heldout_exact_given_train_exact": fmt_conditional(
                        summary["cond_heldout_exact"], summary["train_exact_count"]
                    ),
                }
            )
    return out


def support_audit_summaries(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped = rows_by_benchmark_model(results)
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for benchmark_key in SUPPORT_AUDIT_ORDER:
        for model in LLM_ORDER:
            bucket = grouped.get((benchmark_key, model), [])
            if bucket:
                summaries[(benchmark_key, model)] = summarize(bucket)
    return summaries


def identifiability_audit_table(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    summaries = support_audit_summaries(results)
    out: list[dict[str, str]] = []
    for benchmark_key in SUPPORT_AUDIT_ORDER:
        for model in LLM_ORDER:
            summary = summaries.get((benchmark_key, model))
            if summary is None:
                continue
            out.append(
                {
                    "setting": BENCHMARKS[benchmark_key]["short"],
                    "model": display_model(model),
                    "n": fmt_count(summary["n"]),
                    "valid": fmt_rate(summary["valid"]),
                    "train_exact": fmt_rate(summary["train_exact"]),
                    "train_world": fmt_rate(summary["train_world"]),
                    "heldout_world": fmt_rate(summary["heldout_world"]),
                    "heldout_exact": fmt_rate(summary["heldout_exact"]),
                    "heldout_world_given_train_exact": fmt_conditional(
                        summary["cond_heldout_world"], summary["train_exact_count"]
                    ),
                }
            )
    return out


def common_pool_problem_ids(problems: list[dict[str, Any]]) -> set[str]:
    present: dict[str, set[str]] = defaultdict(set)
    for row in problems:
        key = str(row.get("benchmark_key") or "")
        if key in COMMON_LADDER_ORDER:
            present[str(row.get("paired_source_id") or "")].add(key)
    return {paired_id for paired_id, keys in present.items() if all(key in keys for key in COMMON_LADDER_ORDER)}


def information_ladder_table(problems: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, str]]:
    common_pool = common_pool_problem_ids(problems)
    by_model_bench_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        paired_id = str(row.get("paired_source_id") or "")
        bench = str(row.get("benchmark_key") or "")
        if paired_id in common_pool and bench in COMMON_LADDER_ORDER:
            by_model_bench_pair[(str(row.get("model") or ""), bench)][paired_id] = row

    audit_summaries = support_audit_summaries(results)
    out: list[dict[str, str]] = []
    for model in LLM_ORDER:
        row = {"model": display_model(model)}
        per_benchmark = {bench: by_model_bench_pair.get((model, bench), {}) for bench in COMMON_LADDER_ORDER}
        # The ladder compares settings on the same underlying problem pool, so
        # each model is restricted to paired ids present in every ladder slice.
        aligned_for_model = set(common_pool)
        for bench in COMMON_LADDER_ORDER:
            aligned_for_model &= set(per_benchmark[bench].keys())
        for bench in COMMON_LADDER_ORDER:
            bucket = [per_benchmark[bench][paired_id] for paired_id in sorted(aligned_for_model)]
            row[COMMON_LADDER_DISPLAY[bench]] = fmt_rate(rate(bucket, "train_exact")) if bucket else UNDEFINED
        for bench in SUPPORT_AUDIT_ORDER:
            summary = audit_summaries.get((bench, model))
            row[BENCHMARKS[bench]["short"]] = fmt_rate(summary["train_exact"]) if summary else UNDEFINED
        out.append(row)
    return out


def alt_relation(model: str, problem: dict[str, Any]) -> str:
    known = bool(problem.get("known_alternative_available"))
    support_models = set(list_field(problem.get("alternative_support_models")))
    if not known:
        return "open_ended"
    if not support_models:
        return "known_pair_no_source_metadata"
    if model in support_models:
        return "same_model_sourced"
    return "other_model_sourced"


def diagnostic_panel_a(problems: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, str]]:
    problem_lookup = {
        (str(row.get("benchmark_key") or ""), str(row.get("instance_id") or "")): row
        for row in problems
    }
    row_lookup = {
        (str(row.get("benchmark_key") or ""), str(row.get("paired_source_id") or ""), str(row.get("model") or "")): row
        for row in results
    }
    pairings = [
        ("ntopo_250", "alt_exp_ntopo_100", "Alt-Hid"),
        ("ordered_250", "alt_exp_ordered_100", "Alt-Ord"),
    ]
    out: list[dict[str, str]] = []
    for hidden_key, alt_key, display_setting in pairings:
        paired_ids = sorted(
            {
                str(row.get("paired_source_id") or "")
                for row in results
                if str(row.get("benchmark_key") or "") in {hidden_key, alt_key}
            }
        )
        for model in LLM_ORDER:
            triples = []
            other_model_sourced_rows = []
            for paired_id in paired_ids:
                hidden_row = row_lookup.get((hidden_key, paired_id, model))
                alt_row = row_lookup.get((alt_key, paired_id, model))
                if hidden_row is None or alt_row is None:
                    continue
                alt_problem = problem_lookup[(alt_key, str(alt_row.get("instance_id") or ""))]
                triples.append((hidden_row, alt_row, alt_problem))
                if alt_relation(model, alt_problem) == "other_model_sourced":
                    other_model_sourced_rows.append(alt_row)
            if not triples:
                continue
            hidden_rows = [triple[0] for triple in triples]
            alt_rows = [triple[1] for triple in triples]
            out.append(
                {
                    "setting": display_setting,
                    "system": display_model(model),
                    "paired_train_correct": fmt_conditional(rate(hidden_rows, "correct"), len(hidden_rows)),
                    "alt_scm_joint": fmt_conditional(rate(alt_rows, "joint_success"), len(alt_rows)),
                    "joint_leave_model_out": fmt_conditional(
                        rate(other_model_sourced_rows, "joint_success"), len(other_model_sourced_rows)
                    ),
                    "alt_train_exact": fmt_conditional(rate(alt_rows, "alt_success"), len(alt_rows)),
                    "experiment_witness": fmt_conditional(rate(alt_rows, "witness_valid"), len(alt_rows)),
                }
            )
    return out


def diagnostic_panel_b(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped = rows_by_benchmark_model(results)
    out: list[dict[str, str]] = []
    for model in [*LLM_ORDER, *ROOT_BASELINES]:
        bucket = grouped.get(("root_unknown_ntopo_100", model), [])
        if not bucket:
            continue
        root_exact_rows = [row for row in bucket if row.get("root_set_exact") is True]
        out.append(
            {
                "system": display_model(model),
                "root_exact": fmt_rate(rate(bucket, "root_set_exact")),
                "train_exact": fmt_rate(rate(bucket, "mechanism_train_exact")),
                "heldout_world": fmt_rate(mean_numeric([row.get("heldout_world_exact_accuracy") for row in bucket])),
                "heldout_exact": fmt_rate(rate(bucket, "mechanism_heldout_exact")),
                "train_exact_given_root_exact": fmt_conditional(
                    rate(root_exact_rows, "mechanism_train_exact"), len(root_exact_rows)
                ),
                "heldout_world_given_root_exact": fmt_conditional(
                    mean_numeric([row.get("heldout_world_exact_accuracy") for row in root_exact_rows]),
                    len(root_exact_rows),
                ),
                "heldout_exact_given_root_exact": fmt_conditional(
                    rate(root_exact_rows, "mechanism_heldout_exact"), len(root_exact_rows)
                ),
            }
        )
    return out


def actual_tables(benchmarks_dir: Path) -> dict[str, Any]:
    problems, results, counts = load_release_records(benchmarks_dir)
    return {
        "metadata": {
            "source": "runtime_yaml_embedded_evaluations",
            "benchmark_problem_counts": counts,
            "result_row_count": len(results),
            "note": "Replay-only aggregation from stored evaluation records; no generation or re-scoring.",
        },
        "tables": {
            "topline_results_main": topline_table(results),
            "identifiability_audit_performance": identifiability_audit_table(results),
            "information_ladder_common_pool": information_ladder_table(problems, results),
            "diagnostic_decomposition_main_panel_a": diagnostic_panel_a(problems, results),
            "diagnostic_decomposition_main_panel_b": diagnostic_panel_b(results),
        },
    }


def remove_latex_commands(cell: str) -> str:
    out = cell.strip()
    out = out.replace(r"\\", "").strip()
    previous = None
    while previous != out:
        previous = out
        out = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", out)
    out = out.replace("~", " ")
    out = out.replace(r"\%", "%")
    out = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?", "", out)
    out = out.strip("{} ")
    return out


def clean_cells(line: str) -> list[str]:
    line = line.split("%", 1)[0].strip()
    if line.endswith(r"\\"):
        line = line[:-2].strip()
    return [remove_latex_commands(cell) for cell in line.split("&")]


def parse_topline(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_setting = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line:
            continue
        cells = clean_cells(line)
        if len(cells) != 8 or cells[0] == "Setting":
            continue
        if cells[0]:
            current_setting = cells[0]
        rows.append(
            {
                "setting": current_setting,
                "system": cells[1],
                "valid": cells[2],
                "train_exact": cells[3],
                "heldout_world": cells[4],
                "heldout_exact": cells[5],
                "heldout_world_given_train_exact": cells[6],
                "heldout_exact_given_train_exact": cells[7],
            }
        )
    return rows


def parse_audit(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line:
            continue
        cells = clean_cells(line)
        if len(cells) != 9 or cells[0] == "Setting":
            continue
        rows.append(
            {
                "setting": cells[0],
                "model": cells[1],
                "n": cells[2],
                "valid": cells[3],
                "train_exact": cells[4],
                "train_world": cells[5],
                "heldout_world": cells[6],
                "heldout_exact": cells[7],
                "heldout_world_given_train_exact": cells[8],
            }
        )
    return rows


def parse_information_ladder(path: Path) -> list[dict[str, str]]:
    columns = ["model", "Ord-Match", "Block", "Hid-Match", "Hid-Roots", "Ord-Ext", "Hid-Ext", "Ord-CEx", "Hid-CEx"]
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line:
            continue
        cells = clean_cells(line)
        if len(cells) != len(columns) or cells[0] == "Model":
            continue
        rows.append(dict(zip(columns, cells)))
    return rows


def parse_diagnostic(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    panel_a: list[dict[str, str]] = []
    panel_b: list[dict[str, str]] = []
    active = ""
    in_data = False
    current_setting = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Panel A:" in line:
            active = "a"
            in_data = False
            continue
        if "Panel B:" in line:
            active = "b"
            in_data = False
            continue
        if line.strip() == r"\midrule":
            in_data = True
            continue
        if line.strip() == r"\bottomrule":
            in_data = False
            continue
        if not in_data or "&" not in line:
            continue
        cells = clean_cells(line)
        if active == "a" and len(cells) == 7 and cells[0] != "Setting":
            if cells[0]:
                current_setting = cells[0]
            panel_a.append(
                {
                    "setting": current_setting,
                    "system": cells[1],
                    "paired_train_correct": cells[2],
                    "alt_scm_joint": cells[3],
                    "joint_leave_model_out": cells[4],
                    "alt_train_exact": cells[5],
                    "experiment_witness": cells[6],
                }
            )
        elif active == "b" and len(cells) == 8 and cells[0] != "System":
            panel_b.append(
                {
                    "system": cells[0],
                    "root_exact": cells[1],
                    "train_exact": cells[2],
                    "heldout_world": cells[3],
                    "heldout_exact": cells[4],
                    "train_exact_given_root_exact": cells[5],
                    "heldout_world_given_root_exact": cells[6],
                    "heldout_exact_given_root_exact": cells[7],
                }
            )
    return panel_a, panel_b


def expected_from_paper_tables(tables_dir: Path) -> dict[str, Any]:
    panel_a, panel_b = parse_diagnostic(tables_dir / "diagnostic_decomposition_main.tex")
    return {
        "metadata": {
            "source": "paper_generated_tex_tables",
            "source_detail": "Parsed from paper generated TeX table files.",
        },
        "tables": {
            "topline_results_main": parse_topline(tables_dir / "topline_results_main.tex"),
            "identifiability_audit_performance": parse_audit(tables_dir / "identifiability_audit_performance.tex"),
            "information_ladder_common_pool": parse_information_ladder(tables_dir / "information_ladder_common_pool.tex"),
            "diagnostic_decomposition_main_panel_a": panel_a,
            "diagnostic_decomposition_main_panel_b": panel_b,
        },
    }


def load_expected(expected_path: Path | None, paper_tables_dir: Path | None) -> dict[str, Any] | None:
    if paper_tables_dir is not None:
        return expected_from_paper_tables(paper_tables_dir)
    if expected_path is None or not expected_path.exists():
        return None
    return json.loads(expected_path.read_text(encoding="utf-8"))


KEY_FIELDS = {
    "topline_results_main": ["setting", "system"],
    "identifiability_audit_performance": ["setting", "model"],
    "information_ladder_common_pool": ["model"],
    "diagnostic_decomposition_main_panel_a": ["setting", "system"],
    "diagnostic_decomposition_main_panel_b": ["system"],
}


def row_key(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in {"", "*", UNDEFINED, "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compare_values(actual_value: Any, expected_value: Any, tolerance: float) -> bool:
    actual_float = parse_float(actual_value)
    expected_float = parse_float(expected_value)
    if actual_float is not None and expected_float is not None:
        # Table entries are printed to three decimals; the small epsilon avoids
        # binary floating-point artifacts at the requested tolerance boundary.
        return abs(actual_float - expected_float) <= tolerance + 1e-12
    return str(actual_value) == str(expected_value)


def filter_ignored_models(payload: dict[str, Any], ignored_display_models: set[str]) -> dict[str, Any]:
    if not ignored_display_models:
        return payload
    filtered = json.loads(json.dumps(payload))
    for table_name, rows in filtered.get("tables", {}).items():
        key_fields = KEY_FIELDS.get(table_name, [])
        model_fields = [field for field in ("system", "model") if field in key_fields]
        if not model_fields:
            continue
        model_field = model_fields[0]
        filtered["tables"][table_name] = [
            row for row in rows if str(row.get(model_field) or "") not in ignored_display_models
        ]
    return filtered


def compare_tables(actual: dict[str, Any], expected: dict[str, Any] | None, *, tolerance: float = 0.0) -> list[dict[str, Any]]:
    if expected is None:
        return []
    mismatches: list[dict[str, Any]] = []
    actual_tables_map = actual.get("tables", {})
    expected_tables_map = expected.get("tables", {})
    for table_name, expected_rows in expected_tables_map.items():
        key_fields = KEY_FIELDS[table_name]
        actual_rows = actual_tables_map.get(table_name, [])
        actual_map = {row_key(row, key_fields): row for row in actual_rows}
        expected_map = {row_key(row, key_fields): row for row in expected_rows}
        for key in sorted(set(expected_map) - set(actual_map)):
            mismatches.append({"table": table_name, "row": key, "issue": "missing_actual_row"})
        for key in sorted(set(actual_map) - set(expected_map)):
            mismatches.append({"table": table_name, "row": key, "issue": "unexpected_actual_row"})
        for key in sorted(set(actual_map) & set(expected_map)):
            actual_row = actual_map[key]
            expected_row = expected_map[key]
            for field, expected_value in expected_row.items():
                if field in key_fields:
                    continue
                actual_value = actual_row.get(field)
                if not compare_values(actual_value, expected_value, tolerance):
                    mismatches.append(
                        {
                            "table": table_name,
                            "row": key,
                            "field": field,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
    return mismatches


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paper model results from embedded runtime evaluations")
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--paper-tables-dir", type=Path, default=None)
    parser.add_argument("--write-expected", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--ignore-model",
        action="append",
        default=[],
        help="Model id or display name to exclude from both computed and expected comparisons. May be repeated.",
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for displayed numeric table entries.",
    )
    args = parser.parse_args()

    actual = actual_tables(args.benchmarks_dir)
    expected = load_expected(args.expected, args.paper_tables_dir)
    ignored_display_models = {
        display_model(model) for model in args.ignore_model
    } | {str(model) for model in args.ignore_model}
    actual = filter_ignored_models(actual, ignored_display_models)
    if expected is not None:
        expected = filter_ignored_models(expected, ignored_display_models)
    if args.write_expected is not None:
        if expected is None:
            raise SystemExit("--write-expected requires --paper-tables-dir or an existing --expected file")
        write_json(args.write_expected, expected)

    mismatches = compare_tables(actual, expected, tolerance=args.numeric_tolerance)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "computed_paper_model_results.json", actual)
    if expected is not None:
        write_json(args.outdir / "expected_paper_model_results.json", expected)
    summary = {
        "status": "passed" if not mismatches else "failed",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "validated_tables": sorted(actual["tables"].keys()),
        "generation_or_rescoring": False,
        "ignored_models": sorted(ignored_display_models),
        "numeric_tolerance": args.numeric_tolerance,
    }
    write_json(args.outdir / "validation_summary.json", summary)
    print(f"Validated tables: {', '.join(summary['validated_tables'])}")
    print(f"Mismatch count: {len(mismatches)}")
    print(f"Wrote validation summary: {args.outdir / 'validation_summary.json'}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
