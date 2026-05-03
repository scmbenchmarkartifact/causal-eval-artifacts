"""Format-aware dataset load/save helpers for causal runtime storage."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from concept_synth.io_utils import load_from_yaml, save_to_yaml

from .runtime_storage import RUNTIME_SCHEMA_VERSION, compact_problem_record, expand_runtime_problem_record


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def dataset_uses_runtime_storage(data: Any) -> bool:
    """Return True when a dataset payload is stored in compact runtime form."""
    if isinstance(data, dict):
        metadata = data.get("metadata")
        if isinstance(metadata, dict) and metadata.get("runtimeSchemaVersion") == RUNTIME_SCHEMA_VERSION:
            return True
        problems = data.get("problems")
        if isinstance(problems, list):
            return any(isinstance(record, dict) and "modelResults" in record for record in problems)
    if isinstance(data, list):
        return any(isinstance(record, dict) and "modelResults" in record for record in data)
    return False


def load_causal_dataset(path: str) -> Tuple[Any, List[Dict[str, Any]], bool, bool]:
    """Load a causal dataset and expand compact runtime records on the fly."""
    data = load_from_yaml(path)
    wrapped = isinstance(data, dict) and "problems" in data
    raw_problems: List[Dict[str, Any]] = data.get("problems", []) if wrapped else data
    use_runtime_storage = dataset_uses_runtime_storage(data)

    if use_runtime_storage:
        problems = [
            expand_runtime_problem_record(record) if isinstance(record, dict) else _clone(record)
            for record in raw_problems
        ]
    else:
        problems = raw_problems

    return data, problems, wrapped, use_runtime_storage


def require_runtime_benchmark_path(path: str, *, purpose: str = "run") -> str:
    """Reject live/archive companion paths when a runtime companion exists.

    This is intended for causal benchmark write paths only. Read-only operations
    such as evaluation may still use any companion path.
    """
    candidate = Path(path)
    name = candidate.name

    if name.endswith(".archive.yaml"):
        raise ValueError(
            f"Refusing to {purpose} against archive companion {path}. "
            f"Use the runtime benchmark file instead."
        )

    if name.endswith(".runtime.yaml"):
        return str(candidate)

    if name.endswith(".yaml"):
        runtime_path = candidate.with_name(f"{name[:-len('.yaml')]}.runtime.yaml")
        if runtime_path.exists():
            raise ValueError(
                f"Refusing to {purpose} against live companion {path}. "
                f"Use the runtime benchmark file instead: {runtime_path}"
            )

    return str(candidate)


def save_causal_dataset(
    data: Any,
    problems: List[Dict[str, Any]],
    wrapped: bool,
    output_path: str,
    *,
    use_runtime_storage: bool = False,
) -> None:
    """Save a causal dataset, compacting records when runtime storage is enabled."""
    problems_out = (
        [compact_problem_record(record) if isinstance(record, dict) else _clone(record) for record in problems]
        if use_runtime_storage
        else _clone(problems)
    )

    if wrapped:
        payload = _clone(data)
        payload["problems"] = problems_out
    else:
        payload = problems_out

    save_to_yaml(payload, output_path)
