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
    mean,
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
from concept_synth.analysis.scm_subset_compare import _parse_models_arg

SUMMARY_MD = "scm_train_heldout_amplification_summary.md"
BY_INSTANCE_CSV = "scm_train_heldout_amplification_by_instance.csv"
BY_MODEL_CSV = "scm_train_heldout_amplification_by_model.csv"
BY_SLICE_CSV = "scm_train_heldout_amplification_by_slice.csv"
EXAMPLES_JSONL = "scm_train_heldout_amplification_examples.jsonl"
MANIFEST_JSON = "scm_train_heldout_amplification_manifest.json"

_RATE_EPS = 0.02


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("instance_id") or "unknown"), str(row.get("model") or "unknown")


def _classify_amplification(row: Dict[str, Any]) -> str:
    if row.get("valid") is not True:
        return f"invalid:{str(row.get('invalid_bucket') or 'invalid')}"
    if row.get("train_exact") is True and row.get("heldout_exact") is True:
        return "exact"
    if row.get("train_exact") is True and row.get("heldout_exact") is False:
        return "heldout_only_error"
    if row.get("train_exact") is False and row.get("heldout_exact") is True:
        return "train_only_error"

    total_delta = float(row.get("rollout_wrong_rate_delta") or 0.0)
    direct_delta = float(row.get("direct_wrong_rate_delta") or 0.0)
    propagation_delta = float(row.get("propagated_only_wrong_rate_delta") or 0.0)
    if total_delta <= _RATE_EPS:
        return "no_clear_amplification"

    direct_up = direct_delta > _RATE_EPS
    propagation_up = propagation_delta > _RATE_EPS
    if direct_up and propagation_up:
        return "mixed_amplification"
    if propagation_up and propagation_delta >= (direct_delta - (_RATE_EPS / 2.0)):
        return "propagation_amplification"
    if direct_up:
        return "direct_amplification"
    return "no_clear_amplification"


def _analyze_result(problem: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
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
            "correct": replay.evaluation.get("correct"),
            "train_exact": replay.evaluation.get("trainExact"),
            "heldout_exact": replay.evaluation.get("heldoutExact"),
            "split_error_bucket": split_error_bucket(replay.evaluation),
            "invalid_bucket": parsed.invalid_bucket,
            "amplification_bucket": f"invalid:{str(parsed.invalid_bucket or 'invalid')}",
        }

    train = compute_local_rollout_split_stats(problem, parsed, split_name="train")
    heldout = compute_local_rollout_split_stats(problem, parsed, split_name="heldout")
    row = {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "family": family(problem),
        "slice": slice_name(problem),
        "difficulty": difficulty(problem),
        "task_name": task_name(problem),
        "valid": bool(replay.evaluation.get("valid")),
        "correct": replay.evaluation.get("correct"),
        "train_exact": replay.evaluation.get("trainExact"),
        "heldout_exact": replay.evaluation.get("heldoutExact"),
        "split_error_bucket": split_error_bucket(replay.evaluation),
        "invalid_bucket": parsed.invalid_bucket,
        "train_scored_cells": train.get("scored_cells"),
        "heldout_scored_cells": heldout.get("scored_cells"),
        "train_direct_wrong_cells": train.get("direct_wrong_cells"),
        "heldout_direct_wrong_cells": heldout.get("direct_wrong_cells"),
        "train_rollout_wrong_cells": train.get("rollout_wrong_cells"),
        "heldout_rollout_wrong_cells": heldout.get("rollout_wrong_cells"),
        "train_propagated_only_wrong_cells": train.get("propagated_only_wrong_cells"),
        "heldout_propagated_only_wrong_cells": heldout.get("propagated_only_wrong_cells"),
        "train_compensated_direct_wrong_cells": train.get("compensated_direct_wrong_cells"),
        "heldout_compensated_direct_wrong_cells": heldout.get("compensated_direct_wrong_cells"),
        "train_direct_wrong_rate": train.get("direct_wrong_rate"),
        "heldout_direct_wrong_rate": heldout.get("direct_wrong_rate"),
        "train_rollout_wrong_rate": train.get("rollout_wrong_rate"),
        "heldout_rollout_wrong_rate": heldout.get("rollout_wrong_rate"),
        "train_propagated_only_wrong_rate": train.get("propagated_only_wrong_rate"),
        "heldout_propagated_only_wrong_rate": heldout.get("propagated_only_wrong_rate"),
        "train_compensated_direct_wrong_rate": train.get("compensated_direct_wrong_rate"),
        "heldout_compensated_direct_wrong_rate": heldout.get("compensated_direct_wrong_rate"),
        "direct_wrong_rate_delta": None,
        "rollout_wrong_rate_delta": None,
        "propagated_only_wrong_rate_delta": None,
        "compensated_direct_wrong_rate_delta": None,
    }
    for key in (
        "direct_wrong_rate",
        "rollout_wrong_rate",
        "propagated_only_wrong_rate",
        "compensated_direct_wrong_rate",
    ):
        train_value = train.get(key)
        heldout_value = heldout.get(key)
        delta_key = key.replace("train_", "")
        if train_value is None or heldout_value is None:
            continue
        row[f"{delta_key}_delta"] = float(heldout_value) - float(train_value)
    row["amplification_bucket"] = _classify_amplification(row)
    return row


def analyze_train_heldout_amplification(
    problems: Sequence[Dict[str, Any]],
    *,
    models: Optional[set[str]] = None,
    family_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    instance_rows: List[Dict[str, Any]] = []
    for problem in problems:
        if not is_a_scm_problem(problem):
            continue
        if family_filter and family(problem) != family_filter:
            continue
        for llm_result in problem.get("llmResults") or []:
            if not isinstance(llm_result, dict):
                continue
            model = str(llm_result.get("model") or "unknown")
            if models and model not in models:
                continue
            instance_rows.append(_analyze_result(problem, llm_result))

    by_model_rows: List[Dict[str, Any]] = []
    by_slice_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
        grouped_slice[(str(row.get("slice") or "unknown"), str(row.get("model") or "unknown"))].append(row)

    for model, group in sorted(grouped_model.items()):
        a_cases = [row for row in group if row.get("train_exact") is False and row.get("heldout_exact") is False and row.get("valid") is True]
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in a_cases:
            bucket_counts[str(row.get("amplification_bucket") or "unknown")] += 1
        top_bucket = None
        top_bucket_share = None
        if bucket_counts and a_cases:
            top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
            top_bucket_share = rate(top_count, len(a_cases))
        by_model_rows.append(
            {
                "model": model,
                "n": len(group),
                "a_cases": len(a_cases),
                "top_amplification_bucket": top_bucket,
                "top_amplification_share": top_bucket_share,
                "amplified_share_among_a_cases": rate(
                    sum(1 for row in a_cases if row.get("amplification_bucket") in {"direct_amplification", "propagation_amplification", "mixed_amplification"}),
                    len(a_cases),
                ),
                "mean_train_rollout_wrong_rate": mean((row.get("train_rollout_wrong_rate") or 0.0 for row in a_cases), default=0.0),
                "mean_heldout_rollout_wrong_rate": mean((row.get("heldout_rollout_wrong_rate") or 0.0 for row in a_cases), default=0.0),
                "mean_train_direct_wrong_rate": mean((row.get("train_direct_wrong_rate") or 0.0 for row in a_cases), default=0.0),
                "mean_heldout_direct_wrong_rate": mean((row.get("heldout_direct_wrong_rate") or 0.0 for row in a_cases), default=0.0),
                "mean_train_propagated_only_wrong_rate": mean((row.get("train_propagated_only_wrong_rate") or 0.0 for row in a_cases), default=0.0),
                "mean_heldout_propagated_only_wrong_rate": mean((row.get("heldout_propagated_only_wrong_rate") or 0.0 for row in a_cases), default=0.0),
            }
        )

    for (slice_value, model), group in sorted(grouped_slice.items()):
        a_cases = [row for row in group if row.get("train_exact") is False and row.get("heldout_exact") is False and row.get("valid") is True]
        if not a_cases:
            continue
        bucket_counts: Dict[str, int] = defaultdict(int)
        for row in a_cases:
            bucket_counts[str(row.get("amplification_bucket") or "unknown")] += 1
        top_bucket, top_count = max(bucket_counts.items(), key=lambda item: (item[1], item[0]))
        by_slice_rows.append(
            {
                "slice": slice_value,
                "model": model,
                "a_cases": len(a_cases),
                "top_amplification_bucket": top_bucket,
                "top_amplification_share": rate(top_count, len(a_cases)),
                "amplified_share": rate(
                    sum(1 for row in a_cases if row.get("amplification_bucket") in {"direct_amplification", "propagation_amplification", "mixed_amplification"}),
                    len(a_cases),
                ),
                "mean_rollout_wrong_rate_delta": mean((row.get("rollout_wrong_rate_delta") or 0.0 for row in a_cases), default=0.0),
                "mean_direct_wrong_rate_delta": mean((row.get("direct_wrong_rate_delta") or 0.0 for row in a_cases), default=0.0),
                "mean_propagated_only_wrong_rate_delta": mean((row.get("propagated_only_wrong_rate_delta") or 0.0 for row in a_cases), default=0.0),
            }
        )

    example_rows = [
        row
        for row in instance_rows
        if row.get("amplification_bucket") in {"direct_amplification", "propagation_amplification", "mixed_amplification", "heldout_only_error"}
    ]
    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "by_slice_rows": by_slice_rows,
        "example_rows": example_rows,
    }


def build_train_heldout_amplification_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                row.get("a_cases"),
                row.get("top_amplification_bucket") or "-",
                pct(row.get("top_amplification_share")),
                pct(row.get("amplified_share_among_a_cases")),
                pct(row.get("mean_train_rollout_wrong_rate")),
                pct(row.get("mean_heldout_rollout_wrong_rate")),
                pct(row.get("mean_train_propagated_only_wrong_rate")),
                pct(row.get("mean_heldout_propagated_only_wrong_rate")),
            ]
        )
    lines = [
        "SCM Train-Heldout Amplification",
        "",
        "This report measures bucket A: train-side errors that become worse on heldout.",
        "Direct local error evaluates each predicted mechanism on gold parent values; propagated-only error counts additional full-rollout misses caused by upstream corruption.",
        "",
        markdown_table(
            [
                "Model",
                "N",
                "A cases",
                "Top amplification",
                "Top share",
                "Amplified",
                "Train wrong",
                "Heldout wrong",
                "Train propagated-only",
                "Heldout propagated-only",
            ],
            model_rows,
        ) if model_rows else "No amplification rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_train_heldout_amplification_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
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
    write_markdown(build_train_heldout_amplification_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze train-to-heldout amplification for A_SCM results")
    parser.add_argument("--input", required=True, help="Benchmark YAML input")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    args = parser.parse_args(argv)

    _, problems = load_dataset(args.input)
    artifacts = analyze_train_heldout_amplification(
        problems,
        models=_parse_models_arg(args.models),
        family_filter=args.family,
    )
    paths = write_train_heldout_amplification_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM train-heldout amplification artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
