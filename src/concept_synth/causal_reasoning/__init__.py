"""Causal SCM benchmark scoring and prompt utilities."""

from .cind_family import (
    CIND_FAMILY_A_TASKS,
    TASK_CIND_A_OOD,
    TASK_CIND_A_P,
    TASK_CIND_A_SCM,
    TASK_CIND_A_Y,
    ensure_cind_family_tasks_registered,
    register_cind_family_tasks,
)
from .evaluator import (
    CausalEvaluationResult,
    evaluate_causal_llm_result,
    evaluate_causal_problem_file,
    format_causal_evaluation_report,
)
from .mechanism_dsl import (
    DEFAULT_ALLOWED_OPERATORS,
    MechanismEvalError,
    MechanismNode,
    MechanismParseError,
    analyze_mechanism,
    evaluate_mechanism,
    parse_mechanism,
)
from .prompting import CausalPromptTemplates, build_causal_prompt, load_causal_prompt_templates
from .schema_cr import (
    CAUSAL_SCHEMA_VERSION,
    CausalInstance,
    CausalProblemDescription,
    CausalProblemRecord,
    CausalTaskSpec,
    CausalWorldView,
)
from .task_registry import (
    DEFAULT_CAUSAL_TASK_REGISTRY,
    CausalTaskDefinition,
    CausalTaskRegistry,
)

ensure_cind_family_tasks_registered(DEFAULT_CAUSAL_TASK_REGISTRY)

__all__ = [
    "CAUSAL_SCHEMA_VERSION",
    "CausalTaskSpec",
    "CausalWorldView",
    "CausalInstance",
    "CausalProblemDescription",
    "CausalProblemRecord",
    "CausalTaskDefinition",
    "CausalTaskRegistry",
    "DEFAULT_CAUSAL_TASK_REGISTRY",
    "CausalPromptTemplates",
    "load_causal_prompt_templates",
    "build_causal_prompt",
    "CausalEvaluationResult",
    "evaluate_causal_llm_result",
    "evaluate_causal_problem_file",
    "format_causal_evaluation_report",
    "MechanismNode",
    "MechanismParseError",
    "MechanismEvalError",
    "DEFAULT_ALLOWED_OPERATORS",
    "parse_mechanism",
    "evaluate_mechanism",
    "analyze_mechanism",
    "CIND_FAMILY_A_TASKS",
    "TASK_CIND_A_Y",
    "TASK_CIND_A_P",
    "TASK_CIND_A_SCM",
    "TASK_CIND_A_OOD",
    "register_cind_family_tasks",
    "ensure_cind_family_tasks_registered",
]
