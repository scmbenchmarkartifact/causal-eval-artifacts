#!/usr/bin/env python3
"""Score external model predictions against a causal SCM benchmark file.

Prediction input may be JSONL or a JSON list. Each prediction must include an
instance identifier and either a raw response string or an extracted answer:

{"instanceId": "...", "model": "my-model", "response": "{\"mechanisms\": {...}}"}
{"instanceId": "...", "model": "my-model", "extractedAnswer": {"mechanisms": {...}}}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from concept_synth.causal_reasoning.evaluator import evaluate_causal_llm_result
from concept_synth.causal_reasoning.runtime_storage_io import load_causal_dataset


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_prediction_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON prediction file must contain a list of objects")
        return [row for row in payload if isinstance(row, dict)]

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Prediction line {line_no} is not an object")
            records.append(payload)
    return records


def _prediction_instance_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("instanceId", "problemId", "id"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _problem_instance_id(record: Dict[str, Any], fallback: str) -> str:
    problem = record.get("problem", {}) or {}
    return str(problem.get("instanceId") or record.get("problemId") or fallback)


def _to_llm_result(prediction: Dict[str, Any], default_model: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model": str(prediction.get("model") or default_model),
    }
    if "extractedAnswer" in prediction:
        result["extractedAnswer"] = prediction["extractedAnswer"]
    if "rawResponse" in prediction:
        result["rawResponse"] = prediction["rawResponse"]
    elif "response" in prediction:
        result["response"] = prediction["response"]
    elif "text" in prediction:
        result["response"] = prediction["text"]
    for key in ("latencyMs", "billedTokens", "thinkingTokens", "usageDetails"):
        if key in prediction:
            result[key] = prediction[key]
    return result


def _score_predictions(
    benchmark_path: Path,
    prediction_path: Path,
    *,
    default_model: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _, problems, _, _ = load_causal_dataset(str(benchmark_path))
    problems_by_id = {
        _problem_instance_id(record, f"idx_{idx}"): record
        for idx, record in enumerate(problems)
    }

    predictions = _load_prediction_records(prediction_path)
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    aggregates: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "valid": 0, "correct": 0, "incorrect": 0, "parse_errors": 0}
    )

    for idx, prediction in enumerate(predictions):
        instance_id = _prediction_instance_id(prediction)
        if not instance_id or instance_id not in problems_by_id:
            missing.append({"index": idx, "instanceId": instance_id})
            continue

        llm_result = _to_llm_result(prediction, default_model)
        evaluation = evaluate_causal_llm_result(
            problem=problems_by_id[instance_id],
            result=llm_result,
        )
        evaluation_dict = evaluation.to_dict()
        model = str(llm_result.get("model") or default_model)
        details = evaluation.details or {}

        row = {
            "instanceId": instance_id,
            "model": model,
            "valid": evaluation.valid,
            "correct": evaluation.correct,
            "parseError": evaluation.parseError,
            "failureExplanation": evaluation.failureExplanation,
            "trainExact": details.get("trainExact"),
            "heldoutExact": details.get("heldoutExact"),
            "trainAccuracy": details.get("trainAccuracy"),
            "heldoutAccuracy": details.get("heldoutAccuracy"),
            "parentF1": details.get("parentF1"),
            "evaluation": evaluation_dict,
        }
        rows.append(row)

        agg = aggregates[model]
        agg["total"] += 1
        if evaluation.valid:
            agg["valid"] += 1
        if evaluation.correct is True:
            agg["correct"] += 1
        elif evaluation.correct is False:
            agg["incorrect"] += 1
        if evaluation.parseError:
            agg["parse_errors"] += 1

    summary = {
        "benchmark": str(benchmark_path),
        "predictions": str(prediction_path),
        "problem_count": len(problems),
        "prediction_count": len(predictions),
        "scored_count": len(rows),
        "missing_predictions": missing,
        "by_model": {},
    }
    for model, agg in sorted(aggregates.items()):
        total = agg["total"]
        summary["by_model"][model] = {
            **agg,
            "valid_rate": agg["valid"] / total if total else None,
            "correct_rate": agg["correct"] / total if total else None,
            "parse_error_rate": agg["parse_errors"] / total if total else None,
        }
    return rows, summary


def _write_outputs(rows: List[Dict[str, Any]], summary: Dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "scored_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row))
            handle.write("\n")

    csv_path = outdir / "scored_predictions.csv"
    fieldnames = [
        "instanceId",
        "model",
        "valid",
        "correct",
        "parseError",
        "failureExplanation",
        "trainExact",
        "heldoutExact",
        "trainAccuracy",
        "heldoutAccuracy",
        "parentF1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    (outdir / "summary.json").write_text(_stable_json(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score external predictions against the causal SCM benchmark")
    parser.add_argument("--benchmark", required=True, type=Path, help="Benchmark runtime YAML")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSONL or JSON list")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for scored outputs")
    parser.add_argument("--default-model", default="submitted-model", help="Model name if omitted in predictions")
    args = parser.parse_args()

    rows, summary = _score_predictions(
        args.benchmark,
        args.predictions,
        default_model=args.default_model,
    )
    _write_outputs(rows, summary, args.outdir)

    print(f"Problems: {summary['problem_count']}")
    print(f"Predictions: {summary['prediction_count']}")
    print(f"Scored: {summary['scored_count']}")
    for model, stats in summary["by_model"].items():
        print(
            f"{model}: correct={stats['correct']}/{stats['total']} "
            f"valid={stats['valid']}/{stats['total']} parse_errors={stats['parse_errors']}"
        )
    if summary["missing_predictions"]:
        print(f"Missing instance ids: {len(summary['missing_predictions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
