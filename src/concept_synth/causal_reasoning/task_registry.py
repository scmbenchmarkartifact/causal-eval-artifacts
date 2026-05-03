"""Registry for task-specific causal reasoning hooks.

The scaffold is intentionally generic: concrete causal tasks can plug custom
problem generation, prompting, extraction, and evaluation behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


GenerateProblemFn = Callable[[int, str, Dict[str, Any]], Dict[str, Any]]
BuildPromptFn = Callable[[Dict[str, Any], Dict[str, str]], Tuple[str, str]]
ExtractAnswerFn = Callable[[str], Tuple[Optional[Dict[str, Any]], Optional[str]]]
EvaluateAnswerFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


@dataclass
class CausalTaskDefinition:
    """Task hooks for a concrete causal reasoning variant."""

    name: str
    description: str = ""
    generate_problem: Optional[GenerateProblemFn] = None
    build_prompt: Optional[BuildPromptFn] = None
    extract_answer: Optional[ExtractAnswerFn] = None
    evaluate_answer: Optional[EvaluateAnswerFn] = None


class CausalTaskRegistry:
    """In-memory registry for causal task implementations."""

    def __init__(self):
        self._tasks: Dict[str, CausalTaskDefinition] = {}

    def register(self, task: CausalTaskDefinition, overwrite: bool = False) -> None:
        key = task.name.strip()
        if not key:
            raise ValueError("Task name must be non-empty")
        if key in self._tasks and not overwrite:
            raise ValueError(f"Task already registered: {key}")
        self._tasks[key] = task

    def get(self, task_name: str) -> Optional[CausalTaskDefinition]:
        return self._tasks.get(task_name)

    def require(self, task_name: str) -> CausalTaskDefinition:
        task = self.get(task_name)
        if task is None:
            known = ", ".join(self.list_task_names()) or "(none)"
            raise KeyError(f"Unknown causal task '{task_name}'. Registered tasks: {known}")
        return task

    def list_task_names(self) -> List[str]:
        return sorted(self._tasks.keys())

    def clear(self) -> None:
        self._tasks.clear()


DEFAULT_CAUSAL_TASK_REGISTRY = CausalTaskRegistry()
