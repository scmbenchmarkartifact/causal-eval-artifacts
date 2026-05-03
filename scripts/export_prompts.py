#!/usr/bin/env python3
"""Export benchmark prompts to JSONL for reviewer inspection or model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from concept_synth.causal_reasoning.prompting import build_causal_prompt
from concept_synth.causal_reasoning.runtime_storage_io import load_causal_dataset


def _instance_id(record: Dict[str, Any], fallback: str) -> str:
    problem = record.get("problem", {}) or {}
    return str(problem.get("instanceId") or record.get("problemId") or fallback)


def _task_name(record: Dict[str, Any], fallback: str = "CIND_A_SCM") -> str:
    task = ((record.get("problem", {}) or {}).get("task", {}) or {})
    return str(task.get("taskName") or fallback)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export causal SCM prompts from a benchmark YAML")
    parser.add_argument("--benchmark", required=True, type=Path, help="Benchmark runtime YAML")
    parser.add_argument("--out-jsonl", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of prompts")
    parser.add_argument("--family", choices=["ordered", "ntopo"], default=None, help="Optional prompt family filter")
    args = parser.parse_args()

    _, problems, _, _ = load_causal_dataset(str(args.benchmark))
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(problems):
            instance_id = _instance_id(record, f"idx_{idx}")
            task_name = _task_name(record)
            if args.family and args.family not in instance_id:
                continue
            prompt, system = build_causal_prompt(record, task_name)
            payload = {
                "instanceId": instance_id,
                "taskName": task_name,
                "system": system,
                "prompt": prompt,
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(f"Wrote {written} prompts to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
