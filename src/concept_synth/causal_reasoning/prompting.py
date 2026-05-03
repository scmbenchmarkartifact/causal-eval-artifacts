"""Prompt loading and generic prompt construction for causal reasoning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .task_registry import DEFAULT_CAUSAL_TASK_REGISTRY, CausalTaskRegistry


@dataclass
class CausalPromptTemplates:
    """Prompt template bundle."""

    system_prompt: str = ""
    task_prompt: str = ""
    suffix_prompt: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "task_prompt": self.task_prompt,
            "suffix_prompt": self.suffix_prompt,
        }


def _read_text_or_empty(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_causal_prompt_templates(
    task_name: str,
    prompts_dir: Optional[str] = None,
) -> CausalPromptTemplates:
    """Load prompt templates for a causal task with generic fallback."""
    if prompts_dir is None:
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

    prefixes = [f"cr_{task_name.lower()}_scenario", "cr_generic_scenario"]

    for prefix in prefixes:
        system_path = os.path.join(prompts_dir, f"{prefix}_system.txt")
        task_path = os.path.join(prompts_dir, f"{prefix}_task.txt")
        suffix_path = os.path.join(prompts_dir, f"{prefix}_suffix.txt")

        system_prompt = _read_text_or_empty(system_path)
        task_prompt = _read_text_or_empty(task_path)
        suffix_prompt = _read_text_or_empty(suffix_path)

        if system_prompt or task_prompt or suffix_prompt:
            return CausalPromptTemplates(
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                suffix_prompt=suffix_prompt,
            )

    return CausalPromptTemplates()


def _resolve_prompt_template_task_name(problem: Dict[str, Any], task_name: str) -> str:
    """Resolve template namespace from per-problem metadata.

    Allows mixed prompt variants inside one dataset while keeping task registry key stable.
    """
    resolved = str(task_name or "").strip() or task_name
    prob = problem.get("problem", problem)
    task = prob.get("task", {}) if isinstance(prob, dict) else {}
    params = task.get("parameters", {}) if isinstance(task, dict) else {}
    prompt_variant = str(
        params.get("scmPromptVariant") or params.get("promptVariant") or ""
    ).strip().lower()
    if resolved.upper() == "CIND_A_SCM":
        if prompt_variant in {
            "no_topological_order",
            "unknown_topological_order",
            "blind_topological_order",
            "ntopo",
            "no_topo",
        }:
            return "CIND_A_SCM_NTOPO"
        if prompt_variant in {"partial_topological_order", "partial_order", "partial_topo"}:
            return "CIND_A_SCM_PARTIAL_ORDER"
        return resolved
    if resolved.upper() == "CIND_A_SCM_ALT_EXP":
        if prompt_variant in {
            "no_topological_order",
            "unknown_topological_order",
            "blind_topological_order",
            "ntopo",
            "no_topo",
        }:
            return "CIND_A_SCM_ALT_EXP_NTOPO"
        return resolved
    if resolved.upper() == "CIND_A_SCM_ROOT_UNKNOWN":
        if prompt_variant in {
            "no_topological_order",
            "unknown_topological_order",
            "blind_topological_order",
            "ntopo",
            "no_topo",
        }:
            return "CIND_A_SCM_ROOT_UNKNOWN_NTOPO"
        return resolved
    if resolved.upper() == "CIND_A_SCM_ID":
        if prompt_variant in {
            "no_topological_order",
            "unknown_topological_order",
            "blind_topological_order",
            "ntopo",
            "no_topo",
        }:
            return "CIND_A_SCM_ID_NTOPO"
    return resolved


def _extract_true_list(pred_data: Any) -> list:
    if isinstance(pred_data, dict):
        return list(pred_data.get("true", []) or [])
    if isinstance(pred_data, list):
        return list(pred_data)
    return []


def format_causal_world_for_prompt(world: Dict[str, Any]) -> str:
    """Render a world in a stable, task-agnostic text format."""
    lines = []
    world_id = world.get("worldId", "unknown")
    domain = world.get("domain", [])
    predicates = world.get("predicates", {})

    lines.append(f"### World: {world_id}")
    lines.append(f"Domain: {{{', '.join(domain)}}}")
    lines.append("")
    lines.append("Predicates (known true facts):")

    for pred_name in sorted(predicates.keys()):
        true_items = _extract_true_list(predicates.get(pred_name))
        values = ", ".join(str(x) for x in true_items) if true_items else "(none)"
        lines.append(f"- {pred_name}: {values}")

    target_labels = world.get("targetLabels", {})
    if target_labels:
        lines.append("")
        lines.append("Target Labels:")
        for key in sorted(target_labels.keys()):
            lines.append(f"- {key}: {target_labels[key]}")

    interventions = world.get("interventions", [])
    if interventions:
        lines.append("")
        lines.append(f"Interventions: {interventions}")

    events = world.get("events", {})
    if events:
        lines.append("")
        lines.append(f"Events: {events}")

    return "\n".join(lines)


def build_causal_prompt(
    problem: Dict[str, Any],
    task_name: str,
    registry: CausalTaskRegistry = DEFAULT_CAUSAL_TASK_REGISTRY,
) -> Tuple[str, str]:
    """Build prompt for causal reasoning; delegates to task hook when present."""
    template_task_name = _resolve_prompt_template_task_name(problem, task_name)
    templates = load_causal_prompt_templates(template_task_name)

    task = registry.get(task_name)
    if task and task.build_prompt:
        return task.build_prompt(problem, templates.as_dict())

    prob = problem.get("problem", problem)
    worlds = prob.get("worlds", [])
    task_spec = prob.get("task", {}) or {}
    query = task_spec.get("query", "")

    world_blocks = []
    for w in worlds:
        world_blocks.append(format_causal_world_for_prompt(w))
        world_blocks.append("")

    query_block = f"\nQuery: {query}\n" if query else ""

    prompt = (
        f"{templates.task_prompt}\n\n"
        f"{'\n'.join(world_blocks).strip()}\n"
        f"{query_block}\n"
        f"{templates.suffix_prompt}"
    ).strip()

    return prompt, templates.system_prompt
