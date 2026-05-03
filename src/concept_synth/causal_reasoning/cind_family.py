"""Family A (IMS / CIND) scaffold for mechanism induction under interventions."""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import time
from collections import OrderedDict
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .mechanism_dsl import (
    DEFAULT_ALLOWED_OPERATORS,
    MechanismNode,
    MechanismEvalError,
    MechanismParseError,
    analyze_mechanism,
    evaluate_parsed_mechanism,
    mechanism_variables,
    node_to_sexpr,
    parse_mechanism,
)
from .schema_cr import (
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

TASK_CIND_A_Y = "CIND_A_Y"
TASK_CIND_A_P = "CIND_A_P"
TASK_CIND_A_SCM = "CIND_A_SCM"
TASK_CIND_A_SCM_ROOT_UNKNOWN = "CIND_A_SCM_ROOT_UNKNOWN"
TASK_CIND_A_SCM_ID = "CIND_A_SCM_ID"
TASK_CIND_A_SCM_ALT_EXP = "CIND_A_SCM_ALT_EXP"
TASK_CIND_A_OOD = "CIND_A_OOD"

TASK_CIND_A_Y_ALIAS = "cind_a_y"
TASK_CIND_A_P_ALIAS = "cind_a_p"
TASK_CIND_A_SCM_ALIAS = "cind_a_scm"
TASK_CIND_A_SCM_ROOT_UNKNOWN_ALIAS = "cind_a_scm_root_unknown"
TASK_CIND_A_SCM_ID_ALIAS = "cind_a_scm_id"
TASK_CIND_A_SCM_ALT_EXP_ALIAS = "cind_a_scm_alt_exp"
TASK_CIND_A_OOD_ALIAS = "cind_a_ood"

WorldLike = Union[Dict[str, Any], CausalWorldView]
TruthTable = Dict[Tuple[int, ...], int]
MechanismTruthPayload = Tuple[Tuple[str, ...], TruthTable]
ScmLookupEntry = Union[
    Tuple[Tuple[str, ...], TruthTable],
    Tuple[Tuple[str, ...], TruthTable, Tuple[int, ...]],
]
ScmLookupTables = Dict[str, ScmLookupEntry]

CIND_FAMILY_A_TASKS = (
    TASK_CIND_A_Y,
    TASK_CIND_A_P,
    TASK_CIND_A_SCM,
    TASK_CIND_A_SCM_ROOT_UNKNOWN,
    TASK_CIND_A_SCM_ID,
    TASK_CIND_A_SCM_ALT_EXP,
    TASK_CIND_A_OOD,
)

TASK_TO_VARIANT = {
    TASK_CIND_A_Y: "A_Y",
    TASK_CIND_A_P: "A_P",
    TASK_CIND_A_SCM: "A_SCM",
    TASK_CIND_A_SCM_ROOT_UNKNOWN: "A_SCM_ROOT_UNKNOWN",
    TASK_CIND_A_SCM_ID: "A_SCM_ID",
    TASK_CIND_A_SCM_ALT_EXP: "A_SCM_ALT_EXP",
    TASK_CIND_A_OOD: "A_OOD",
    TASK_CIND_A_Y_ALIAS: "A_Y",
    TASK_CIND_A_P_ALIAS: "A_P",
    TASK_CIND_A_SCM_ALIAS: "A_SCM",
    TASK_CIND_A_SCM_ROOT_UNKNOWN_ALIAS: "A_SCM_ROOT_UNKNOWN",
    TASK_CIND_A_SCM_ID_ALIAS: "A_SCM_ID",
    TASK_CIND_A_SCM_ALT_EXP_ALIAS: "A_SCM_ALT_EXP",
    TASK_CIND_A_OOD_ALIAS: "A_OOD",
}

VARIANT_TO_CANONICAL_TASK = {
    "A_Y": TASK_CIND_A_Y,
    "A_P": TASK_CIND_A_P,
    "A_SCM": TASK_CIND_A_SCM,
    "A_SCM_ROOT_UNKNOWN": TASK_CIND_A_SCM_ROOT_UNKNOWN,
    "A_SCM_ID": TASK_CIND_A_SCM_ID,
    "A_SCM_ALT_EXP": TASK_CIND_A_SCM_ALT_EXP,
    "A_OOD": TASK_CIND_A_OOD,
}

SCM_PROMPT_VARIANT_ORDERED = "with_topological_order"
SCM_PROMPT_VARIANT_NTOPO = "no_topological_order"
SCM_PROMPT_VARIANT_PARTIAL = "partial_topological_order"

_SCM_PROMPT_VARIANT_NTOPO_ALIASES = {
    "no_topological_order",
    "unknown_topological_order",
    "blind_topological_order",
    "ntopo",
    "no_topo",
}
_SCM_PROMPT_VARIANT_PARTIAL_ALIASES = {
    "partial_topological_order",
    "partial_order",
    "partial_topo",
}

DEFAULT_BUDGET_DELTAS = (0, 10, 25)
_X_TOKEN_RE = re.compile(r"^X(\d+)$")

_DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "n": 5,
        "k": 4,
        "m": 8,
        "heldout_k": 2,
        "parent_max": 2,
        "gold_ast_min": 8,
        "gold_ast_max": 15,
    },
    "medium": {
        "n": 7,
        "k": 6,
        "m": 12,
        "heldout_k": 3,
        "parent_max": 3,
        "gold_ast_min": 12,
        "gold_ast_max": 20,
    },
    "hard": {
        "n": 9,
        "k": 8,
        "m": 16,
        "heldout_k": 4,
        "parent_max": 4,
        "gold_ast_min": 18,
        "gold_ast_max": 28,
    },
    "extreme": {
        "n": 11,
        "k": 10,
        "m": 24,
        "heldout_k": 5,
        "parent_max": 4,
        "gold_ast_min": 25,
        "gold_ast_max": 40,
    },
    "hard_small": {
        "n": 6,
        "k": 5,
        "m": 10,
        "heldout_k": 5,
        "parent_size": 3,
        "parent_max": 3,
        "gold_ast_min": 12,
        "gold_ast_max": 18,
        "gold_depth_target": 3,
        "allow_if": False,
        "require_all_parents_used": True,
        "require_each_parent_essential": True,
        "intervention_size_probs": {1: 0.6, 2: 0.4},
        "nonparent_shortcut_ast_cap": 7,
        "allvar_shortcut_ast_cap": 3,
        "max_extra_train_worlds_for_shortcut": 1,
        "use_cegis_lite": True,
        "distractor_ast_cap": 14,
        "cegis_candidate_interventions": 60,
        "min_total_kills": 1,
        "min_worlds_with_kills": 1,
        "min_parent_assignment_coverage": 0.75,
        "scm_intervention_mode_probs": {"hard_constant": 0.45, "hard_assigned": 0.55},
        "scm_env_levels": [0.2, 0.35, 0.5, 0.65, 0.8],
        "scm_shortcut_ast_cap_floor": 3,
        "scm_shortcut_ast_cap": 4,
        "scm_cegis_seed_worlds": 2,
        "scm_cegis_restarts": 6,
        "scm_target_survivors_small_max": 0,
        "scm_min_survivor_reduction_frac": 0.67,
        "scm_quality_retry_budget": 60,
    },
    "very_hard_small": {
        "n": 8,
        "k": 8,
        "m": 10,
        "heldout_k": 8,
        "parent_size": 4,
        "parent_max": 4,
        "gold_ast_min": 18,
        "gold_ast_max": 26,
        "gold_depth_target": 3,
        "allow_if": False,
        "require_all_parents_used": True,
        "require_each_parent_essential": True,
        "intervention_size_probs": {1: 0.15, 2: 0.55, 3: 0.30},
        "nonparent_shortcut_ast_cap": 9,
        "allvar_shortcut_ast_cap": None,
        "allvar_shortcut_ast_cap_from_gold": True,
        "max_extra_train_worlds_for_shortcut": 1,
        "use_cegis_lite": True,
        "distractor_ast_cap": 20,
        "cegis_candidate_interventions": 160,
        "cegis_exact_witness_ast_cap": 9,
        "require_survivors_small_or_gap": False,
        "target_survivors_small_min": None,
        "gap_to_second_best_min": -22,
        "gap_to_second_best_max": 2,
        "min_total_kills": 1,
        "min_worlds_with_kills": 1,
        "min_parent_assignment_coverage": 0.75,
        "scm_intervention_mode_probs": {"hard_constant": 0.4, "hard_assigned": 0.6},
        "scm_env_levels": [0.2, 0.35, 0.5, 0.65, 0.8],
        "scm_shortcut_ast_cap_floor": 3,
        "scm_shortcut_ast_cap": 4,
        "scm_cegis_seed_worlds": 2,
        "scm_cegis_restarts": 8,
        "scm_target_survivors_small_max": 1,
        "scm_min_survivor_reduction_frac": 0.5,
        "scm_quality_retry_budget": 80,
    },
}

_MIN_AST_TABLE_CACHE_MAX = max(
    16,
    int(os.environ.get("CS_MIN_AST_TABLE_CACHE_MAX", "1024")),
)
_TRUTH_TABLE_ONES_PATTERNS_CACHE_MAX = max(
    512,
    int(os.environ.get("CS_TRUTH_TABLE_ONES_PATTERNS_CACHE_MAX", "131072")),
)
_ESSENTIAL_PAIR_CACHE_MAX = max(
    8,
    int(os.environ.get("CS_ESSENTIAL_PAIR_CACHE_MAX", "256")),
)

_MIN_AST_TABLE_CACHE: "OrderedDict[Tuple[int, Tuple[str, ...], int, int, bool], Dict[int, int]]" = (
    OrderedDict()
)
_TRUTH_TABLE_VAR_MASKS_CACHE: Dict[int, Tuple[int, ...]] = {}
_TRUTH_TABLE_ONES_PATTERNS_CACHE: "OrderedDict[int, Tuple[int, Tuple[int, ...]]]" = OrderedDict()
_ESSENTIAL_PAIR_CACHE: "OrderedDict[int, Tuple[Tuple[Tuple[int, int], ...], ...]]" = OrderedDict()


def _lru_get(cache: OrderedDict, key: Any) -> Any:
    try:
        value = cache[key]
    except KeyError:
        return None
    cache.move_to_end(key, last=True)
    return value


def _lru_set(cache: OrderedDict, key: Any, value: Any, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key, last=True)
    while len(cache) > int(max_entries):
        cache.popitem(last=False)


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip()


def _clone_unshared(value: Any) -> Any:
    """Recursively clone lists/dicts without preserving shared references."""
    if isinstance(value, dict):
        return {k: _clone_unshared(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_unshared(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_unshared(v) for v in value)
    return value


def _resolve_variant(task_name: str) -> str:
    key = _normalize_task_name(task_name)
    return TASK_TO_VARIANT.get(key, "A_Y")


def _normalize_scm_prompt_variant(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _SCM_PROMPT_VARIANT_NTOPO_ALIASES:
        return SCM_PROMPT_VARIANT_NTOPO
    if raw in _SCM_PROMPT_VARIANT_PARTIAL_ALIASES:
        return SCM_PROMPT_VARIANT_PARTIAL
    return SCM_PROMPT_VARIANT_ORDERED


def _scm_variant_hides_topological_order(value: Any) -> bool:
    return _normalize_scm_prompt_variant(value) == SCM_PROMPT_VARIANT_NTOPO


def _scm_variant_uses_partial_order(value: Any) -> bool:
    return _normalize_scm_prompt_variant(value) == SCM_PROMPT_VARIANT_PARTIAL


def _compute_topological_layers(
    *,
    topological_order: Sequence[str],
    parents_by_var: Dict[str, Sequence[str]],
) -> List[List[str]]:
    topo = [str(v) for v in topological_order]
    if not topo:
        return []
    parent_map = {str(var): [str(parent) for parent in (parents or [])] for var, parents in parents_by_var.items()}
    level_by_var: Dict[str, int] = {}
    for var in topo:
        parents = [str(parent) for parent in (parent_map.get(str(var)) or []) if str(parent) in level_by_var]
        if not parents:
            level_by_var[str(var)] = 0
        else:
            level_by_var[str(var)] = 1 + max(int(level_by_var[parent]) for parent in parents)
    max_level = max(level_by_var.values()) if level_by_var else -1
    layers: List[List[str]] = []
    for level in range(max_level + 1):
        layer = [str(var) for var in topo if int(level_by_var.get(str(var), -1)) == level]
        if layer:
            layers.append(layer)
    return layers


def _normalize_topological_layers(
    raw_layers: Any,
    *,
    topological_order: Sequence[str],
) -> Optional[List[List[str]]]:
    topo = [str(v) for v in topological_order]
    if not topo:
        return None
    if not isinstance(raw_layers, list):
        return None
    layers: List[List[str]] = []
    seen: Set[str] = set()
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, list):
            return None
        layer = [str(v) for v in raw_layer]
        if not layer:
            return None
        if len(layer) != len(set(layer)):
            return None
        if any(var in seen for var in layer):
            return None
        seen.update(layer)
        layers.append(layer)
    if set(topo) != seen:
        return None
    topo_index = {str(var): idx for idx, var in enumerate(topo)}
    for layer in layers:
        layer.sort(key=lambda var: topo_index.get(str(var), 10**9))
    return layers


def _same_or_earlier_layer_variables(var: str, layers: Sequence[Sequence[str]]) -> Optional[List[str]]:
    target = str(var)
    out: List[str] = []
    for layer in layers:
        current = [str(v) for v in layer]
        if target in current:
            return [v for v in out + current if str(v) != target]
        out.extend(current)
    return None


def _normalize_intervention_size_probs(raw: Any) -> Optional[Dict[int, float]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None

    out: Dict[int, float] = {}
    for key, value in raw.items():
        try:
            k = int(key)
            v = float(value)
        except (TypeError, ValueError):
            continue
        if k <= 0 or v <= 0.0:
            continue
        out[k] = v

    if not out:
        return None

    total = sum(out.values())
    if total <= 0.0:
        return None

    return {k: (v / total) for k, v in sorted(out.items())}


def _normalize_mode_probs(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    c = float(raw.get("hard_constant", 0.0))
    a = float(raw.get("hard_assigned", 0.0))
    c = max(0.0, c)
    a = max(0.0, a)
    total = c + a
    if total <= 0.0:
        return None
    return {
        "hard_constant": c / total,
        "hard_assigned": a / total,
    }


def _resolve_generation_spec(task_config: Dict[str, Any], variant: str) -> Dict[str, Any]:
    difficulty = str(task_config.get("difficulty", "easy")).strip().lower() or "easy"
    preset = dict(_DIFFICULTY_PRESETS.get(difficulty, _DIFFICULTY_PRESETS["easy"]))

    for key in (
        "n",
        "k",
        "m",
        "heldout_k",
        "parent_max",
        "gold_ast_min",
        "gold_ast_max",
        "parent_size",
        "gold_depth_target",
        "scm_root_count",
        "scm_eq_ast_min",
        "scm_eq_ast_max",
        "scm_shortcut_ast_cap",
        "scm_shortcut_ast_cap_floor",
        "scm_min_n",
        "shortcut_ast_cap",
        "cegis_candidate_world_budget",
        "cegis_max_iters",
        "cegis_exact_witness_ast_cap",
        "cegis_exact_witness_time_budget_ms",
        "cegis_exact_witness_max_signatures_per_size",
        "shortcut_check_time_budget_ms",
        "shortcut_check_max_signatures_per_size",
        "shortcut_check_timeout_fallback_samples",
        "scm_cegis_seed_worlds",
        "scm_cegis_restarts",
        "scm_target_survivors_small_max",
        "scm_quality_retry_budget",
        "scm_max_predecessors_per_target",
        "scm_stage3_probe_size",
        "scm_stage3_probe_subsets_per_var",
        "scm_stage3_world_budget",
        "scm_stage3_candidate_world_budget",
        "world_max_attempts",
    ):
        if key in task_config:
            preset[key] = int(task_config[key])

    preset["difficulty"] = difficulty
    prompt_variant_raw = str(
        task_config.get("scm_prompt_variant", preset.get("scm_prompt_variant", "with_topological_order"))
    ).strip().lower()
    if prompt_variant_raw in {
        "no_topological_order",
        "unknown_topological_order",
        "blind_topological_order",
        "ntopo",
        "no_topo",
    }:
        preset["scm_prompt_variant"] = "no_topological_order"
    else:
        preset["scm_prompt_variant"] = "with_topological_order"
    preset["allow_if"] = bool(task_config.get("allow_if", preset.get("allow_if", False)))
    default_allow_constants = bool(variant != "A_SCM")
    preset["allow_constants"] = bool(
        task_config.get(
            "allow_constants",
            preset.get("allow_constants", default_allow_constants),
        )
    )
    preset["output_mode"] = str(task_config.get("output_mode", "parents_plus_mechanism"))
    preset["domain_values"] = [0, 1]
    preset["require_all_parents_used"] = bool(
        task_config.get("require_all_parents_used", preset.get("require_all_parents_used", False))
    )
    preset["require_each_parent_essential"] = bool(
        task_config.get("require_each_parent_essential", preset.get("require_each_parent_essential", False))
    )
    resolved_gold_depth_target = preset.get("gold_depth_target")
    default_depth_sampler_mode = "canonical"
    if variant in {"A_Y", "A_P", "A_OOD"} and resolved_gold_depth_target is not None:
        try:
            if int(resolved_gold_depth_target) >= 4:
                default_depth_sampler_mode = "stochastic"
        except (TypeError, ValueError):
            default_depth_sampler_mode = "canonical"
    depth_sampler_mode_raw = str(
        task_config.get(
            "gold_depth_sampler_mode",
            preset.get(
                "gold_depth_sampler_mode",
                default_depth_sampler_mode,
            ),
        )
        or "canonical"
    ).strip().lower()
    if depth_sampler_mode_raw in {"stochastic", "stochastic_depth", "random", "rand"}:
        preset["gold_depth_sampler_mode"] = "stochastic"
    else:
        preset["gold_depth_sampler_mode"] = "canonical"

    if "nonparent_shortcut_ast_cap" in task_config:
        shortcut_cap = task_config.get("nonparent_shortcut_ast_cap")
    else:
        shortcut_cap = preset.get("nonparent_shortcut_ast_cap")
    if shortcut_cap is None:
        preset["nonparent_shortcut_ast_cap"] = None
    else:
        preset["nonparent_shortcut_ast_cap"] = max(1, int(shortcut_cap))

    if "allvar_shortcut_ast_cap" in task_config:
        allvar_shortcut_cap = task_config.get("allvar_shortcut_ast_cap")
    else:
        allvar_shortcut_cap = preset.get("allvar_shortcut_ast_cap")
    if allvar_shortcut_cap is None:
        preset["allvar_shortcut_ast_cap"] = None
    else:
        preset["allvar_shortcut_ast_cap"] = max(1, int(allvar_shortcut_cap))

    preset["allvar_shortcut_ast_cap_from_gold"] = bool(
        task_config.get(
            "allvar_shortcut_ast_cap_from_gold",
            preset.get("allvar_shortcut_ast_cap_from_gold", False),
        )
    )

    if "intervention_size_probs" in task_config:
        preset["intervention_size_probs"] = _normalize_intervention_size_probs(
            task_config.get("intervention_size_probs")
        )
    else:
        preset["intervention_size_probs"] = _normalize_intervention_size_probs(
            preset.get("intervention_size_probs")
        )

    if "scm_intervention_mode_probs" in task_config:
        preset["scm_intervention_mode_probs"] = _normalize_mode_probs(
            task_config.get("scm_intervention_mode_probs")
        )
    else:
        preset["scm_intervention_mode_probs"] = _normalize_mode_probs(
            preset.get("scm_intervention_mode_probs")
        )

    if "scm_env_levels" in task_config:
        env_raw = task_config.get("scm_env_levels")
    else:
        env_raw = preset.get("scm_env_levels")
    if isinstance(env_raw, list):
        env_vals = [float(x) for x in env_raw if 0.0 < float(x) < 1.0]
        preset["scm_env_levels"] = env_vals if env_vals else None
    else:
        preset["scm_env_levels"] = None

    preset["use_cegis_lite"] = bool(task_config.get("use_cegis_lite", preset.get("use_cegis_lite", False)))

    if "distractor_ast_cap" in task_config:
        distractor_cap = task_config.get("distractor_ast_cap")
    else:
        distractor_cap = preset.get("distractor_ast_cap")
    if distractor_cap is None:
        preset["distractor_ast_cap"] = None
    else:
        preset["distractor_ast_cap"] = max(1, int(distractor_cap))

    if "distractor_pool_size" in task_config:
        pool_size = int(task_config.get("distractor_pool_size"))
    else:
        pool_size = int(preset.get("distractor_pool_size", 256))
    preset["distractor_pool_size"] = max(16, pool_size)

    if "cegis_candidate_interventions" in task_config:
        proposal_count = int(task_config.get("cegis_candidate_interventions"))
    else:
        proposal_count = int(preset.get("cegis_candidate_interventions", 30))
    preset["cegis_candidate_interventions"] = max(5, proposal_count)

    preset["require_survivors_small_or_gap"] = bool(
        task_config.get(
            "require_survivors_small_or_gap",
            preset.get("require_survivors_small_or_gap", False),
        )
    )

    def _normalize_optional_int(name: str) -> Optional[int]:
        if name in task_config:
            value = task_config.get(name)
        else:
            value = preset.get(name)
        if value is None:
            return None
        return int(value)

    def _normalize_optional_float(name: str) -> Optional[float]:
        if name in task_config:
            value = task_config.get(name)
        else:
            value = preset.get(name)
        if value is None:
            return None
        return float(value)

    preset["target_survivors_small_min"] = _normalize_optional_int("target_survivors_small_min")
    preset["target_survivors_small_max"] = _normalize_optional_int("target_survivors_small_max")
    preset["gap_to_second_best_min"] = _normalize_optional_int("gap_to_second_best_min")
    preset["gap_to_second_best_max"] = _normalize_optional_int("gap_to_second_best_max")
    preset["min_total_kills"] = _normalize_optional_int("min_total_kills")
    preset["min_worlds_with_kills"] = _normalize_optional_int("min_worlds_with_kills")
    preset["min_scored_worlds_per_endogenous"] = _normalize_optional_int(
        "min_scored_worlds_per_endogenous"
    )
    preset["min_scored_cells_per_endogenous"] = _normalize_optional_int(
        "min_scored_cells_per_endogenous"
    )
    preset["max_intervened_worlds_per_endogenous"] = _normalize_optional_int(
        "max_intervened_worlds_per_endogenous"
    )
    preset["min_hard_assigned_worlds"] = _normalize_optional_int("min_hard_assigned_worlds")
    preset["min_hard_constant_worlds"] = _normalize_optional_int("min_hard_constant_worlds")
    preset["heldout_min_scored_worlds_per_endogenous"] = _normalize_optional_int(
        "heldout_min_scored_worlds_per_endogenous"
    )
    preset["heldout_min_scored_cells_per_endogenous"] = _normalize_optional_int(
        "heldout_min_scored_cells_per_endogenous"
    )
    preset["max_extra_train_worlds_for_shortcut"] = _normalize_optional_int(
        "max_extra_train_worlds_for_shortcut"
    )
    preset["min_parent_assignment_coverage"] = _normalize_optional_float(
        "min_parent_assignment_coverage"
    )
    heldout_mode_raw = str(
        task_config.get(
            "heldout_mode",
            preset.get("heldout_mode", "iid_current"),
        )
        or "iid_current"
    ).strip().lower()
    if heldout_mode_raw in {"verify_balanced", "verify", "balanced"}:
        preset["heldout_mode"] = "verify_balanced"
    else:
        preset["heldout_mode"] = "iid_current"
    novelty_min = _normalize_optional_float("heldout_target_novelty_min")
    novelty_max = _normalize_optional_float("heldout_target_novelty_max")
    preset["heldout_target_novelty_min"] = (
        None if novelty_min is None else min(max(float(novelty_min), 0.0), 1.0)
    )
    preset["heldout_target_novelty_max"] = (
        None if novelty_max is None else min(max(float(novelty_max), 0.0), 1.0)
    )
    if "world_attempt_time_budget_sec" in task_config:
        world_attempt_time_budget_raw = task_config.get("world_attempt_time_budget_sec")
    else:
        world_attempt_time_budget_raw = preset.get("world_attempt_time_budget_sec")
    if world_attempt_time_budget_raw is None:
        if variant in {"A_Y", "A_P", "A_OOD"}:
            preset["world_attempt_time_budget_sec"] = 25.0
        else:
            preset["world_attempt_time_budget_sec"] = None
    else:
        try:
            world_attempt_time_budget = float(world_attempt_time_budget_raw)
        except (TypeError, ValueError):
            world_attempt_time_budget = 0.0
        preset["world_attempt_time_budget_sec"] = (
            None if world_attempt_time_budget <= 0.0 else world_attempt_time_budget
        )
    preset["scm_target_survivors_small_max"] = _normalize_optional_int(
        "scm_target_survivors_small_max"
    )
    scm_min_reduction_raw = (
        task_config.get("scm_min_survivor_reduction_frac")
        if "scm_min_survivor_reduction_frac" in task_config
        else preset.get("scm_min_survivor_reduction_frac")
    )
    if scm_min_reduction_raw is None:
        preset["scm_min_survivor_reduction_frac"] = None
    else:
        frac = float(scm_min_reduction_raw)
        preset["scm_min_survivor_reduction_frac"] = min(max(frac, 0.0), 1.0)
    preset["debug_generation"] = bool(task_config.get("debug_generation", False))
    preset["debug_every"] = max(1, int(task_config.get("debug_every", preset.get("debug_every", 10))))
    preset["use_scm_cegis_lite"] = bool(
        task_config.get("use_scm_cegis_lite", preset.get("use_scm_cegis_lite", True))
    )
    preset["use_scm_stage3_lite"] = bool(
        task_config.get(
            "use_scm_stage3_lite",
            preset.get("use_scm_stage3_lite", variant == "A_SCM"),
        )
    )
    scm_stage3_assigned_bias_raw = (
        task_config.get("scm_stage3_assigned_bias")
        if "scm_stage3_assigned_bias" in task_config
        else preset.get("scm_stage3_assigned_bias", 0.8 if variant == "A_SCM" else 0.5)
    )
    preset["scm_stage3_assigned_bias"] = min(
        max(float(scm_stage3_assigned_bias_raw), 0.0),
        1.0,
    )
    preset["scm_require_global_minimal_equiv"] = bool(
        task_config.get(
            "scm_require_global_minimal_equiv",
            preset.get("scm_require_global_minimal_equiv", variant == "A_SCM"),
        )
    )
    if variant == "A_SCM":
        # For SCM, depth targeting is an optional explicit knob.
        # Preset depth targets from non-SCM variants can overconstrain generation.
        if "gold_depth_target" not in task_config:
            preset["gold_depth_target"] = None
        if "scm_max_predecessors_per_target" not in task_config and (
            "scm_max_predecessors_per_target" not in preset
        ):
            preset["scm_max_predecessors_per_target"] = int(preset.get("parent_max", 3))
        if "scm_stage3_probe_size" not in task_config and "scm_stage3_probe_size" not in preset:
            preset["scm_stage3_probe_size"] = 3
        if (
            "scm_stage3_probe_subsets_per_var" not in task_config
            and "scm_stage3_probe_subsets_per_var" not in preset
        ):
            preset["scm_stage3_probe_subsets_per_var"] = 2
        if "scm_stage3_world_budget" not in task_config and "scm_stage3_world_budget" not in preset:
            preset["scm_stage3_world_budget"] = 1
        if (
            "scm_stage3_candidate_world_budget" not in task_config
            and "scm_stage3_candidate_world_budget" not in preset
        ):
            preset["scm_stage3_candidate_world_budget"] = 24

    # Keep knobs in a sane range.
    preset["n"] = max(3, preset["n"])
    preset["k"] = max(2, preset["k"])
    preset["m"] = max(2, preset["m"])
    preset["heldout_k"] = max(1, preset["heldout_k"])
    preset["scm_cegis_seed_worlds"] = max(
        2,
        int(preset.get("scm_cegis_seed_worlds", 3)),
    )
    preset["scm_cegis_restarts"] = max(
        1,
        int(preset.get("scm_cegis_restarts", 1)),
    )
    preset["scm_quality_retry_budget"] = max(
        1,
        int(preset.get("scm_quality_retry_budget", 40)),
    )
    parent_cap = preset["n"] - 1
    if variant in {"A_Y", "A_P", "A_OOD"}:
        parent_cap = min(parent_cap, 4)
    preset["parent_max"] = min(max(1, preset["parent_max"]), parent_cap)
    if preset.get("parent_size") is None:
        preset["parent_size"] = None
    else:
        preset["parent_size"] = min(max(1, int(preset["parent_size"])), parent_cap)
        preset["parent_max"] = max(preset["parent_max"], preset["parent_size"])

    if preset.get("cegis_exact_witness_ast_cap") is None:
        default_witness_cap = (
            preset.get("allvar_shortcut_ast_cap")
            if preset.get("allvar_shortcut_ast_cap") is not None
            else preset.get("nonparent_shortcut_ast_cap")
        )
        if default_witness_cap is None:
            preset["cegis_exact_witness_ast_cap"] = None
        else:
            preset["cegis_exact_witness_ast_cap"] = max(1, int(default_witness_cap))
    else:
        preset["cegis_exact_witness_ast_cap"] = max(1, int(preset["cegis_exact_witness_ast_cap"]))

    witness_time_budget_raw = preset.get("cegis_exact_witness_time_budget_ms", 120)
    if witness_time_budget_raw is None:
        preset["cegis_exact_witness_time_budget_ms"] = None
    else:
        witness_time_budget = int(witness_time_budget_raw)
        preset["cegis_exact_witness_time_budget_ms"] = (
            None if witness_time_budget <= 0 else witness_time_budget
        )

    witness_sig_cap_raw = preset.get("cegis_exact_witness_max_signatures_per_size", 2048)
    if witness_sig_cap_raw is None:
        preset["cegis_exact_witness_max_signatures_per_size"] = None
    else:
        witness_sig_cap = int(witness_sig_cap_raw)
        preset["cegis_exact_witness_max_signatures_per_size"] = (
            None if witness_sig_cap <= 0 else witness_sig_cap
        )

    shortcut_time_budget_raw = preset.get("shortcut_check_time_budget_ms", 60)
    if shortcut_time_budget_raw is None:
        preset["shortcut_check_time_budget_ms"] = None
    else:
        shortcut_time_budget = int(shortcut_time_budget_raw)
        preset["shortcut_check_time_budget_ms"] = (
            None if shortcut_time_budget <= 0 else shortcut_time_budget
        )

    shortcut_sig_cap_raw = preset.get("shortcut_check_max_signatures_per_size", 1024)
    if shortcut_sig_cap_raw is None:
        preset["shortcut_check_max_signatures_per_size"] = None
    else:
        shortcut_sig_cap = int(shortcut_sig_cap_raw)
        preset["shortcut_check_max_signatures_per_size"] = (
            None if shortcut_sig_cap <= 0 else shortcut_sig_cap
        )

    timeout_policy_raw = str(
        task_config.get(
            "shortcut_check_timeout_policy",
            preset.get("shortcut_check_timeout_policy", "sampled_fallback"),
        )
        or "sampled_fallback"
    ).strip().lower()
    if timeout_policy_raw in {"assume_shortcut", "shortcut", "fail_closed", "reject"}:
        preset["shortcut_check_timeout_policy"] = "assume_shortcut"
    elif timeout_policy_raw in {"assume_no_shortcut", "pass", "fail_open", "allow"}:
        preset["shortcut_check_timeout_policy"] = "assume_no_shortcut"
    else:
        preset["shortcut_check_timeout_policy"] = "sampled_fallback"

    fallback_samples_raw = preset.get("shortcut_check_timeout_fallback_samples", 192)
    if fallback_samples_raw is None:
        preset["shortcut_check_timeout_fallback_samples"] = 0
    else:
        preset["shortcut_check_timeout_fallback_samples"] = max(0, int(fallback_samples_raw))

    if preset["distractor_ast_cap"] is None:
        preset["distractor_ast_cap"] = max(1, min(int(preset["gold_ast_max"]), 14))

    if preset["min_parent_assignment_coverage"] is not None:
        cov = float(preset["min_parent_assignment_coverage"])
        preset["min_parent_assignment_coverage"] = min(max(cov, 0.0), 1.0)

    if preset["max_extra_train_worlds_for_shortcut"] is None:
        preset["max_extra_train_worlds_for_shortcut"] = 0
    else:
        preset["max_extra_train_worlds_for_shortcut"] = max(
            0,
            int(preset["max_extra_train_worlds_for_shortcut"]),
        )
    for key in (
        "min_scored_worlds_per_endogenous",
        "min_scored_cells_per_endogenous",
        "max_intervened_worlds_per_endogenous",
        "min_hard_assigned_worlds",
        "min_hard_constant_worlds",
        "heldout_min_scored_worlds_per_endogenous",
        "heldout_min_scored_cells_per_endogenous",
    ):
        value = preset.get(key)
        if value is not None:
            preset[key] = max(0, int(value))
    novelty_min = preset.get("heldout_target_novelty_min")
    novelty_max = preset.get("heldout_target_novelty_max")
    if novelty_min is not None and novelty_max is not None and float(novelty_min) > float(novelty_max):
        preset["heldout_target_novelty_min"], preset["heldout_target_novelty_max"] = (
            float(novelty_max),
            float(novelty_min),
        )
    preset["world_max_attempts"] = max(1, int(preset.get("world_max_attempts", 220)))

    if preset.get("gold_depth_target") is None:
        preset["gold_depth_target"] = None
    else:
        preset["gold_depth_target"] = max(1, int(preset["gold_depth_target"]))

    scm_max_pred = preset.get("scm_max_predecessors_per_target")
    if scm_max_pred is None:
        preset["scm_max_predecessors_per_target"] = None
    else:
        preset["scm_max_predecessors_per_target"] = max(1, int(scm_max_pred))
    preset["scm_stage3_probe_size"] = max(1, int(preset.get("scm_stage3_probe_size", 3)))
    preset["scm_stage3_probe_subsets_per_var"] = max(
        1,
        int(preset.get("scm_stage3_probe_subsets_per_var", 2)),
    )
    preset["scm_stage3_world_budget"] = max(0, int(preset.get("scm_stage3_world_budget", 1)))
    preset["scm_stage3_candidate_world_budget"] = max(
        4,
        int(preset.get("scm_stage3_candidate_world_budget", 24)),
    )

    # If user gave inverted bounds, fix them.
    if preset["gold_ast_min"] > preset["gold_ast_max"]:
        preset["gold_ast_min"], preset["gold_ast_max"] = (
            preset["gold_ast_max"],
            preset["gold_ast_min"],
        )

    return preset


def _make_var(name: str) -> MechanismNode:
    return MechanismNode(kind="var", value=name)


def _make_op(name: str, args: list[MechanismNode]) -> MechanismNode:
    return MechanismNode(kind="op", value=name, args=tuple(args))


def _natural_var_key(name: str) -> Tuple[int, int, str]:
    m = _X_TOKEN_RE.match(str(name))
    if m:
        return (0, int(m.group(1)), str(name))
    return (1, 10**9, str(name))


def _node_ast_size(node: MechanismNode) -> int:
    if node.kind in {"var", "const"}:
        return 1
    return 1 + sum(_node_ast_size(arg) for arg in node.args)


def _node_depth(node: MechanismNode) -> int:
    if node.kind in {"var", "const"}:
        return 1
    return 1 + max(_node_depth(arg) for arg in node.args)


def _node_sort_key(node: MechanismNode) -> Tuple[Any, ...]:
    if node.kind == "var":
        kind_key, idx, txt = _natural_var_key(node.value)
        return (0, kind_key, idx, txt)
    if node.kind == "const":
        return (1, int(node.value))
    return (2, node.value, node_to_sexpr(node))


def _const_node(bit: int) -> MechanismNode:
    return MechanismNode(kind="const", value="1" if int(bit) else "0")


def _is_const(node: MechanismNode, bit: Optional[int] = None) -> bool:
    if node.kind != "const":
        return False
    if bit is None:
        return True
    return int(node.value) == int(bit)


def _node_contains_const(node: MechanismNode) -> bool:
    if node.kind == "const":
        return True
    return any(_node_contains_const(arg) for arg in node.args)


def _canonicalize_node(node: MechanismNode) -> MechanismNode:
    if node.kind in {"var", "const"}:
        return node

    op = node.value
    args = [_canonicalize_node(arg) for arg in node.args]

    if op == "not":
        child = args[0]
        if _is_const(child):
            return _const_node(1 - int(child.value))
        if child.kind == "op" and child.value == "not":
            return child.args[0]
        return _make_op("not", [child])

    if op == "if":
        cond, left, right = args
        if _is_const(cond, 1):
            return left
        if _is_const(cond, 0):
            return right
        if node_to_sexpr(left) == node_to_sexpr(right):
            return left
        if _is_const(left, 1) and _is_const(right, 0):
            return cond
        if _is_const(left, 0) and _is_const(right, 1):
            return _canonicalize_node(_make_op("not", [cond]))
        return _make_op("if", [cond, left, right])

    if op in {"and", "or", "xor"}:
        flat: List[MechanismNode] = []
        for arg in args:
            if arg.kind == "op" and arg.value == op:
                flat.extend(list(arg.args))
            else:
                flat.append(arg)
        args = flat

    if op == "and":
        for arg in args:
            if _is_const(arg, 0):
                return _const_node(0)
        filtered = [arg for arg in args if not _is_const(arg, 1)]
        uniq: Dict[str, MechanismNode] = {}
        for arg in filtered:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if not out:
            return _const_node(1)
        if len(out) == 1:
            return out[0]
        return _make_op("and", out)

    if op == "or":
        for arg in args:
            if _is_const(arg, 1):
                return _const_node(1)
        filtered = [arg for arg in args if not _is_const(arg, 0)]
        uniq: Dict[str, MechanismNode] = {}
        for arg in filtered:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if not out:
            return _const_node(0)
        if len(out) == 1:
            return out[0]
        return _make_op("or", out)

    if op == "xor":
        parity = 0
        counts: Dict[str, Tuple[int, MechanismNode]] = {}
        for arg in args:
            if _is_const(arg):
                parity ^= int(arg.value)
                continue
            key = node_to_sexpr(arg)
            prev = counts.get(key)
            if prev is None:
                counts[key] = (1, arg)
            else:
                counts[key] = (prev[0] + 1, prev[1])
        out = [node for (count, node) in counts.values() if (count % 2) == 1]
        if parity == 1:
            out.append(_const_node(1))
        out = sorted(out, key=_node_sort_key)
        if not out:
            return _const_node(0)
        if len(out) == 1:
            return out[0]
        return _make_op("xor", out)

    if op == "iff":
        has_zero = any(_is_const(arg, 0) for arg in args)
        has_one = any(_is_const(arg, 1) for arg in args)
        if has_zero and has_one:
            return _const_node(0)
        uniq: Dict[str, MechanismNode] = {}
        for arg in args:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if len(out) <= 1:
            return _const_node(1)
        return _make_op("iff", out)

    return _make_op(op, args)


def _canonicalize_node_assuming_canonical_children(node: MechanismNode) -> MechanismNode:
    """Fast-path canonicalization when all child nodes are already canonical."""
    if node.kind in {"var", "const"}:
        return node

    op = node.value
    args = list(node.args)

    if op == "not":
        child = args[0]
        if _is_const(child):
            return _const_node(1 - int(child.value))
        if child.kind == "op" and child.value == "not":
            return child.args[0]
        return _make_op("not", [child])

    if op == "if":
        cond, left, right = args
        if _is_const(cond, 1):
            return left
        if _is_const(cond, 0):
            return right
        if node_to_sexpr(left) == node_to_sexpr(right):
            return left
        if _is_const(left, 1) and _is_const(right, 0):
            return cond
        if _is_const(left, 0) and _is_const(right, 1):
            if _is_const(cond):
                return _const_node(1 - int(cond.value))
            if cond.kind == "op" and cond.value == "not":
                return cond.args[0]
            return _make_op("not", [cond])
        return _make_op("if", [cond, left, right])

    if op in {"and", "or", "xor"}:
        flat: List[MechanismNode] = []
        for arg in args:
            if arg.kind == "op" and arg.value == op:
                flat.extend(list(arg.args))
            else:
                flat.append(arg)
        args = flat

    if op == "and":
        for arg in args:
            if _is_const(arg, 0):
                return _const_node(0)
        filtered = [arg for arg in args if not _is_const(arg, 1)]
        uniq: Dict[str, MechanismNode] = {}
        for arg in filtered:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if not out:
            return _const_node(1)
        if len(out) == 1:
            return out[0]
        return _make_op("and", out)

    if op == "or":
        for arg in args:
            if _is_const(arg, 1):
                return _const_node(1)
        filtered = [arg for arg in args if not _is_const(arg, 0)]
        uniq: Dict[str, MechanismNode] = {}
        for arg in filtered:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if not out:
            return _const_node(0)
        if len(out) == 1:
            return out[0]
        return _make_op("or", out)

    if op == "xor":
        parity = 0
        counts: Dict[str, Tuple[int, MechanismNode]] = {}
        for arg in args:
            if _is_const(arg):
                parity ^= int(arg.value)
                continue
            key = node_to_sexpr(arg)
            prev = counts.get(key)
            if prev is None:
                counts[key] = (1, arg)
            else:
                counts[key] = (prev[0] + 1, prev[1])
        out = [child for (count, child) in counts.values() if (count % 2) == 1]
        if parity == 1:
            out.append(_const_node(1))
        out = sorted(out, key=_node_sort_key)
        if not out:
            return _const_node(0)
        if len(out) == 1:
            return out[0]
        return _make_op("xor", out)

    if op == "iff":
        has_zero = any(_is_const(arg, 0) for arg in args)
        has_one = any(_is_const(arg, 1) for arg in args)
        if has_zero and has_one:
            return _const_node(0)
        uniq: Dict[str, MechanismNode] = {}
        for arg in args:
            uniq[node_to_sexpr(arg)] = arg
        out = sorted(uniq.values(), key=_node_sort_key)
        if len(out) <= 1:
            return _const_node(1)
        return _make_op("iff", out)

    return _make_op(op, args)


def _truth_table_for_node(node: MechanismNode, variables: List[str]) -> int:
    n = len(variables)
    assignment_count = 1 << n
    if assignment_count <= 0:
        return 0
    full_mask = (1 << assignment_count) - 1

    var_masks = _TRUTH_TABLE_VAR_MASKS_CACHE.get(int(n))
    if var_masks is None:
        built: List[int] = []
        for i in range(n):
            pattern = 0
            for assignment_idx in range(assignment_count):
                if (int(assignment_idx) >> int(i)) & 1:
                    pattern |= (1 << int(assignment_idx))
            built.append(int(pattern))
        var_masks = tuple(int(x) for x in built)
        _TRUTH_TABLE_VAR_MASKS_CACHE[int(n)] = var_masks

    var_to_mask: Dict[str, int] = {
        str(var): int(var_masks[idx]) for idx, var in enumerate(variables)
    }
    node_mask_cache: Dict[int, int] = {}

    def _eval_mask(cur: MechanismNode) -> int:
        node_id = int(id(cur))
        cached = node_mask_cache.get(node_id)
        if cached is not None:
            return int(cached)

        if cur.kind == "const":
            out = int(full_mask) if str(cur.value) == "1" else 0
        elif cur.kind == "var":
            var_name = str(cur.value)
            if var_name not in var_to_mask:
                raise MechanismEvalError(f"Missing variable '{var_name}' in assignment")
            out = int(var_to_mask[var_name])
        else:
            op = str(cur.value)
            child_masks = [_eval_mask(child) for child in cur.args]
            if op == "not":
                out = int(full_mask) ^ int(child_masks[0])
            elif op == "and":
                out = int(full_mask)
                for child_mask in child_masks:
                    out &= int(child_mask)
            elif op == "or":
                out = 0
                for child_mask in child_masks:
                    out |= int(child_mask)
            elif op == "xor":
                out = 0
                for child_mask in child_masks:
                    out ^= int(child_mask)
            elif op == "iff":
                head = int(child_masks[0])
                out = int(full_mask)
                for child_mask in child_masks[1:]:
                    out &= int(full_mask) ^ (int(head) ^ int(child_mask))
            elif op == "if":
                cond, left, right = child_masks
                out = (int(cond) & int(left)) | ((int(full_mask) ^ int(cond)) & int(right))
            else:
                raise MechanismEvalError(f"Unsupported operator '{op}'")

        out &= int(full_mask)
        node_mask_cache[node_id] = int(out)
        return int(out)

    return int(_eval_mask(node))


def _essential_pairs_for_n(n: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    cached = _lru_get(_ESSENTIAL_PAIR_CACHE, int(n))
    if cached is not None:
        return cached
    if n <= 0:
        out: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple()
        _lru_set(_ESSENTIAL_PAIR_CACHE, int(n), out, _ESSENTIAL_PAIR_CACHE_MAX)
        return out
    assignment_count = int(1 << int(n))
    per_var: List[Tuple[Tuple[int, int], ...]] = []
    for i in range(int(n)):
        bit = int(1 << int(i))
        pairs: List[Tuple[int, int]] = []
        for mask in range(assignment_count):
            flipped = int(mask) ^ int(bit)
            if int(mask) < int(flipped):
                pairs.append((int(mask), int(flipped)))
        per_var.append(tuple(pairs))
    out = tuple(per_var)
    _lru_set(_ESSENTIAL_PAIR_CACHE, int(n), out, _ESSENTIAL_PAIR_CACHE_MAX)
    return out


def _table_outputs_both_bits(table: int, n: int) -> bool:
    if n <= 0:
        return False
    assignment_count = int(1 << int(n))
    full_mask = int((1 << assignment_count) - 1)
    reduced = int(table) & int(full_mask)
    return int(reduced) != 0 and int(reduced) != int(full_mask)


def _table_all_parents_essential(table: int, n: int) -> bool:
    if n <= 0:
        return True
    table_i = int(table)
    for pairs in _essential_pairs_for_n(int(n)):
        essential = False
        for a, b in pairs:
            if ((table_i >> int(a)) & 1) != ((table_i >> int(b)) & 1):
                essential = True
                break
        if not essential:
            return False
    return True


def _weak_size_compositions(total: int, parts: int, min_part: int = 1) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    if parts <= 0:
        return out

    def _rec(remaining: int, slots: int, acc: List[int]) -> None:
        if slots == 1:
            if remaining >= min_part:
                out.append(tuple(acc + [remaining]))
            return
        max_first = remaining - min_part * (slots - 1)
        for first in range(min_part, max_first + 1):
            _rec(remaining - first, slots - 1, acc + [first])

    _rec(total, parts, [])
    return out


def _nondecreasing_size_compositions(total: int, parts: int, min_part: int = 1) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    if parts <= 0:
        return out

    def _rec(remaining: int, slots: int, start: int, acc: List[int]) -> None:
        if slots == 1:
            if remaining >= start and remaining >= min_part:
                out.append(tuple(acc + [remaining]))
            return
        max_first = remaining - start * (slots - 1)
        for first in range(start, max_first + 1):
            _rec(remaining - first, slots - 1, first, acc + [first])

    _rec(total, parts, max(1, min_part), [])
    return out


def _min_ast_by_truth_table_cached(
    num_vars: int,
    ast_limit: int,
    allowed_ops: Optional[Set[str]] = None,
    max_nary_arity: int = 3,
    allow_constants: bool = True,
) -> Dict[int, int]:
    if ast_limit < 1:
        return {}

    ops = tuple(sorted(allowed_ops or set(DEFAULT_ALLOWED_OPERATORS)))
    cache_key = (
        int(num_vars),
        ops,
        int(ast_limit),
        int(max_nary_arity),
        bool(allow_constants),
    )
    cached = _lru_get(_MIN_AST_TABLE_CACHE, cache_key)
    if cached is not None:
        return cached

    # Reuse any existing larger-limit cache for the same operator/arity/config.
    # For callers that only need to know whether a strictly smaller equivalent
    # formula exists, filtering the larger cache down to ast_limit is exact.
    covering_key: Optional[Tuple[int, Tuple[str, ...], int, int, bool]] = None
    for key in _MIN_AST_TABLE_CACHE.keys():
        if (
            int(key[0]) == int(num_vars)
            and tuple(key[1]) == ops
            and int(key[3]) == int(max_nary_arity)
            and bool(key[4]) == bool(allow_constants)
            and int(key[2]) >= int(ast_limit)
        ):
            if covering_key is None or int(key[2]) < int(covering_key[2]):
                covering_key = key
    if covering_key is not None:
        covering = _lru_get(_MIN_AST_TABLE_CACHE, covering_key)
        if covering is None:
            covering = {}
        return {int(table): int(ast) for table, ast in covering.items() if int(ast) <= int(ast_limit)}

    variables = [f"X{i}" for i in range(1, int(num_vars) + 1)]
    op_set = set(ops)

    by_size: Dict[int, Dict[str, Dict[str, Any]]] = {s: {} for s in range(1, ast_limit + 1)}
    min_ast: Dict[int, int] = {}
    table_cache: Dict[str, int] = {}
    stats_cache: Dict[str, Dict[str, Any]] = {}
    entry_cache: Dict[int, Tuple[MechanismNode, Dict[str, Any]]] = {}

    def _entry_for(node: MechanismNode) -> Dict[str, Any]:
        node_id = int(id(node))
        cached_pair = entry_cache.get(node_id)
        if cached_pair is not None and cached_pair[0] is node:
            return cached_pair[1]
        can = _canonicalize_node_assuming_canonical_children(node)
        can_id = int(id(can))
        cached_can_pair = entry_cache.get(can_id)
        if cached_can_pair is not None and cached_can_pair[0] is can:
            entry_cache[node_id] = (node, cached_can_pair[1])
            return cached_can_pair[1]
        sexpr = node_to_sexpr(can)
        if sexpr not in table_cache:
            table_cache[sexpr] = _truth_table_for_node(can, variables)
            stats_cache[sexpr] = analyze_mechanism(can)
        stats = stats_cache[sexpr]
        entry = {
            "node": can,
            "sexpr": sexpr,
            "table": int(table_cache[sexpr]),
            "ast": int(stats.get("astSize", _node_ast_size(can))),
        }
        entry_cache[node_id] = (node, entry)
        entry_cache[can_id] = (can, entry)
        return entry

    def _add_entry(e: Dict[str, Any]) -> None:
        ast = int(e["ast"])
        if ast < 1 or ast > ast_limit:
            return
        if (not allow_constants) and _node_contains_const(e["node"]):
            return
        sexpr = str(e["sexpr"])
        if sexpr in by_size[ast]:
            return
        by_size[ast][sexpr] = e
        table = int(e["table"])
        prev = min_ast.get(table)
        if prev is None or ast < prev:
            min_ast[table] = ast

    atoms: List[MechanismNode] = [_make_var(v) for v in variables]
    if allow_constants:
        atoms.extend([_const_node(0), _const_node(1)])
    for atom in atoms:
        _add_entry(_entry_for(atom))

    commutative_ops = [op for op in ("and", "or", "xor", "iff") if op in op_set]
    has_not = "not" in op_set

    for size in range(2, ast_limit + 1):
        if has_not:
            for child in list(by_size[size - 1].values()):
                _add_entry(_entry_for(_make_op("not", [child["node"]])))

        for op in commutative_ops:
            max_arity = min(size - 1, max(2, int(max_nary_arity)))
            for arity in range(2, max_arity + 1):
                partitions = _nondecreasing_size_compositions(total=size - 1, parts=arity, min_part=1)
                for part in partitions:
                    groups = [list(by_size[s].values()) for s in part]
                    if any(not g for g in groups):
                        continue
                    for combo in itertools.product(*groups):
                        sexprs = [str(e["sexpr"]) for e in combo]
                        if any(sexprs[i] > sexprs[i + 1] for i in range(len(sexprs) - 1)):
                            continue
                        _add_entry(_entry_for(_make_op(op, [e["node"] for e in combo])))

        if "if" in op_set and size >= 4:
            for part in _weak_size_compositions(total=size - 1, parts=3, min_part=1):
                groups = [list(by_size[s].values()) for s in part]
                if any(not g for g in groups):
                    continue
                for cond, left, right in itertools.product(*groups):
                    _add_entry(_entry_for(_make_op("if", [cond["node"], left["node"], right["node"]])))

    # Keep only non-dominated cache entries for this signature. A cache built
    # at a larger ast_limit subsumes all smaller-limit entries.
    for key in list(_MIN_AST_TABLE_CACHE.keys()):
        if key == cache_key:
            continue
        if (
            int(key[0]) == int(num_vars)
            and tuple(key[1]) == ops
            and int(key[3]) == int(max_nary_arity)
            and bool(key[4]) == bool(allow_constants)
            and int(key[2]) <= int(ast_limit)
        ):
            _MIN_AST_TABLE_CACHE.pop(key, None)

    _lru_set(_MIN_AST_TABLE_CACHE, cache_key, dict(min_ast), _MIN_AST_TABLE_CACHE_MAX)
    cached_new = _lru_get(_MIN_AST_TABLE_CACHE, cache_key)
    if cached_new is None:
        return {}
    return cached_new


def _sample_gold_mechanism_canonical_depth(
    rng: random.Random,
    parents: List[str],
    ast_min: int,
    ast_max: int,
    allow_if: bool,
    allow_constants: bool,
    depth_target: int,
    require_all_parents_used: bool,
    require_each_parent_essential: bool,
) -> Tuple[str, Dict[str, Any]]:
    if allow_if:
        raise RuntimeError("Depth-exact canonical sampling does not support IF yet")
    if depth_target < 1:
        raise RuntimeError("Depth target must be >=1")
    if not parents:
        raise RuntimeError("Parents must be non-empty for canonical sampling")

    allowed_ops = set(DEFAULT_ALLOWED_OPERATORS)
    table_cache: Dict[str, int] = {}
    entry_cache: Dict[int, Tuple[MechanismNode, Dict[str, Any]]] = {}

    def _entry_for_node(node: MechanismNode) -> Dict[str, Any]:
        node_id = int(id(node))
        cached_pair = entry_cache.get(node_id)
        if cached_pair is not None and cached_pair[0] is node:
            return cached_pair[1]
        can = _canonicalize_node_assuming_canonical_children(node)
        can_id = int(id(can))
        cached_can_pair = entry_cache.get(can_id)
        if cached_can_pair is not None and cached_can_pair[0] is can:
            entry_cache[node_id] = (node, cached_can_pair[1])
            return cached_can_pair[1]
        sexpr = node_to_sexpr(can)
        if sexpr not in table_cache:
            table_cache[sexpr] = _truth_table_for_node(can, parents)
        entry = {
            "node": can,
            "sexpr": sexpr,
            "table": table_cache[sexpr],
            "ast": int(_node_ast_size(can)),
            "depth": int(_node_depth(can)),
        }
        entry_cache[node_id] = (node, entry)
        entry_cache[can_id] = (can, entry)
        return entry

    def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if int(a["ast"]) != int(b["ast"]):
            return int(a["ast"]) < int(b["ast"])
        return str(a["sexpr"]) < str(b["sexpr"])

    exact_depth: Dict[int, Dict[int, Dict[str, Any]]] = {d: {} for d in range(1, depth_target + 1)}

    # Base atoms at depth 1.
    atoms: List[MechanismNode] = [_make_var(v) for v in parents]
    if allow_constants:
        atoms.extend([_const_node(0), _const_node(1)])
    for atom in atoms:
        e = _entry_for_node(atom)
        if int(e["depth"]) != 1:
            continue
        prev = exact_depth[1].get(int(e["table"]))
        if prev is None or _better(e, prev):
            exact_depth[1][int(e["table"])] = e

    nary_ops = [op for op in ("and", "or", "xor", "iff") if op in allowed_ops]

    for depth in range(2, depth_target + 1):
        out: Dict[int, Dict[str, Any]] = {}
        lower: List[Dict[str, Any]] = []
        for d in range(1, depth):
            lower.extend(exact_depth[d].values())
        lower = sorted(lower, key=lambda e: (int(e["ast"]), str(e["sexpr"])))

        # Unary NOT over exact previous depth.
        if "not" in allowed_ops:
            for child in exact_depth[depth - 1].values():
                e = _entry_for_node(_make_op("not", [child["node"]]))
                if int(e["depth"]) != depth:
                    continue
                prev = out.get(int(e["table"]))
                if prev is None or _better(e, prev):
                    out[int(e["table"])] = e

        # N-ary operators over <=depth-1 children, with max child depth = depth-1.
        # Include arity-3 so depth=3 can still express 5-parent mechanisms.
        for op in nary_ops:
            # arity 2
            for i, left in enumerate(lower):
                for right in lower[i:]:
                    if max(int(left["depth"]), int(right["depth"])) != (depth - 1):
                        continue
                    e = _entry_for_node(_make_op(op, [left["node"], right["node"]]))
                    if int(e["depth"]) != depth:
                        continue
                    prev = out.get(int(e["table"]))
                    if prev is None or _better(e, prev):
                        out[int(e["table"])] = e

            # arity 3
            for i, a in enumerate(lower):
                for j, b in enumerate(lower[i:], start=i):
                    for c in lower[j:]:
                        if max(int(a["depth"]), int(b["depth"]), int(c["depth"])) != (depth - 1):
                            continue
                        e = _entry_for_node(_make_op(op, [a["node"], b["node"], c["node"]]))
                        if int(e["depth"]) != depth:
                            continue
                        prev = out.get(int(e["table"]))
                        if prev is None or _better(e, prev):
                            out[int(e["table"])] = e

        exact_depth[depth] = out

    pool = list(exact_depth.get(depth_target, {}).values())
    if not pool:
        raise RuntimeError("No canonical formulas found at requested depth")

    strong: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []
    parent_set = set(parents)
    parent_count = int(len(parents))
    for e in pool:
        node = e["node"]
        if (not allow_constants) and _node_contains_const(node):
            continue
        used = set(mechanism_variables(node))
        if not used or not used.issubset(parent_set):
            continue
        if require_all_parents_used and used != parent_set:
            continue
        if not _table_outputs_both_bits(int(e["table"]), int(parent_count)):
            continue
        if require_each_parent_essential and not _table_all_parents_essential(
            int(e["table"]),
            int(parent_count),
        ):
            continue
        if len(used) < min(2, len(parents)):
            continue
        if int(e["ast"]) < max(5, len(parents) + 2):
            continue

        soft.append(e)
        if ast_min <= int(e["ast"]) <= ast_max:
            strong.append(e)

    candidates = strong if strong else soft
    if not candidates:
        raise RuntimeError("No canonical formulas satisfy parent/essential constraints")

    min_candidate_ast = min(int(e["ast"]) for e in candidates)
    candidates = [e for e in candidates if int(e["ast"]) == min_candidate_ast]

    if min_candidate_ast > 1:
        smaller_min_map = _min_ast_by_truth_table_cached(
            num_vars=len(parents),
            ast_limit=min_candidate_ast - 1,
            allowed_ops=allowed_ops,
            max_nary_arity=3,
            allow_constants=allow_constants,
        )
        minimal_equiv = [
            e
            for e in candidates
            if int(smaller_min_map.get(int(e["table"]), int(e["ast"]))) >= int(e["ast"])
        ]
        if not minimal_equiv:
            raise RuntimeError(
                "No depth-target canonical candidate is globally minimal in AST "
                "among equivalent formulas"
            )
        candidates = minimal_equiv

    candidates = sorted(candidates, key=lambda e: (int(e["ast"]), str(e["sexpr"])))
    chosen = rng.choice(candidates)
    chosen_stats = analyze_mechanism(chosen["node"])
    return str(chosen["sexpr"]), dict(chosen_stats)


def _sample_gold_mechanism_stochastic_depth(
    rng: random.Random,
    parents: List[str],
    ast_min: int,
    ast_max: int,
    allow_if: bool,
    allow_constants: bool,
    depth_target: int,
    require_all_parents_used: bool,
    require_each_parent_essential: bool,
) -> Tuple[str, Dict[str, Any]]:
    if depth_target < 1:
        raise RuntimeError("Depth target must be >=1")
    if not parents:
        raise RuntimeError("Parents must be non-empty for stochastic depth sampling")

    parent_set = set(parents)
    strict_requirements = bool(require_all_parents_used or require_each_parent_essential)
    max_expr_attempts = 16000 if strict_requirements else 7000
    seen_expr: Set[str] = set()

    for _ in range(max_expr_attempts):
        node = _canonicalize_node(
            _rand_expr_tree_exact_depth(
                rng=rng,
                variables=parents,
                depth=int(depth_target),
                allow_if=allow_if,
                max_nary_arity=5 if strict_requirements else 4,
            )
        )
        if (not allow_constants) and _node_contains_const(node):
            continue

        stats = analyze_mechanism(node)
        used = set(stats.get("variables", []))
        if not used or not used.issubset(parent_set):
            continue
        if require_all_parents_used and used != parent_set:
            continue
        if not _node_outputs_both_bits(node, sorted(used)):
            continue
        if require_each_parent_essential and not _all_parents_essential(node, parents):
            continue
        if len(used) < min(2, len(parents)):
            continue
        ast_size = int(stats.get("astSize", _node_ast_size(node)))
        if ast_size < ast_min or ast_size > ast_max:
            continue
        if ast_size < max(5, len(parents) + 2):
            continue
        if int(stats.get("maxDepth", _node_depth(node))) != int(depth_target):
            continue

        expr = node_to_sexpr(node)
        if expr in seen_expr:
            continue
        seen_expr.add(expr)
        return expr, stats

    raise RuntimeError("No stochastic depth-target formulas satisfy parent/essential constraints")


def _node_outputs_both_bits(node: MechanismNode, vars_used: List[str]) -> bool:
    if not vars_used:
        return False
    outputs = set()
    n = len(vars_used)
    for mask in range(1 << n):
        assignment = {}
        for i, var in enumerate(vars_used):
            assignment[var] = (mask >> i) & 1
        outputs.add(evaluate_parsed_mechanism(node, assignment))
        if len(outputs) > 1:
            return True
    return False


def _all_parents_essential(node: MechanismNode, parents: List[str]) -> bool:
    if not parents:
        return True

    outputs: Dict[int, int] = {}
    n = len(parents)
    for mask in range(1 << n):
        assignment = {}
        for i, var in enumerate(parents):
            assignment[var] = (mask >> i) & 1
        outputs[mask] = evaluate_parsed_mechanism(node, assignment)

    for i in range(n):
        bit = 1 << i
        essential = False
        for mask in range(1 << n):
            flipped = mask ^ bit
            if mask < flipped and outputs[mask] != outputs[flipped]:
                essential = True
                break
        if not essential:
            return False

    return True


def _rename_node_variables(node: MechanismNode, var_map: Dict[str, str]) -> MechanismNode:
    if node.kind == "var":
        return _make_var(str(var_map.get(str(node.value), str(node.value))))
    if node.kind == "const":
        return node
    return _make_op(str(node.value), [_rename_node_variables(arg, var_map) for arg in node.args])


def _expand_scm_node(
    var: str,
    local_nodes_by_var: Dict[str, MechanismNode],
    root_set: Set[str],
    _memo: Optional[Dict[str, MechanismNode]] = None,
) -> MechanismNode:
    memo = _memo if _memo is not None else {}
    if var in memo:
        return memo[var]
    if var in root_set:
        out = _make_var(var)
        memo[var] = out
        return out

    node = local_nodes_by_var[var]
    out = _expand_scm_node_from_node(node, local_nodes_by_var, root_set, memo)
    memo[var] = out
    return out


def _expand_scm_node_from_node(
    node: MechanismNode,
    local_nodes_by_var: Dict[str, MechanismNode],
    root_set: Set[str],
    memo: Optional[Dict[str, MechanismNode]] = None,
) -> MechanismNode:
    if node.kind in {"const"}:
        return node
    if node.kind == "var":
        name = str(node.value)
        if name in root_set:
            return _make_var(name)
        return _expand_scm_node(name, local_nodes_by_var, root_set, memo)
    return _make_op(
        str(node.value),
        [_expand_scm_node_from_node(arg, local_nodes_by_var, root_set, memo) for arg in node.args],
    )


def _rand_expr_tree(
    rng: random.Random,
    variables: List[str],
    max_depth: int,
    allow_if: bool,
) -> MechanismNode:
    if max_depth <= 1 or rng.random() < 0.35:
        return _make_var(rng.choice(variables))

    op_choices = ["not", "and", "or", "xor", "iff"]
    if allow_if:
        op_choices.append("if")

    op = rng.choice(op_choices)
    if op == "not":
        return _make_op(op, [_rand_expr_tree(rng, variables, max_depth - 1, allow_if)])

    if op == "if":
        return _make_op(
            op,
            [
                _rand_expr_tree(rng, variables, max_depth - 1, allow_if),
                _rand_expr_tree(rng, variables, max_depth - 1, allow_if),
                _rand_expr_tree(rng, variables, max_depth - 1, allow_if),
            ],
        )

    arity = 2 if rng.random() < 0.8 else 3
    args = [_rand_expr_tree(rng, variables, max_depth - 1, allow_if) for _ in range(arity)]
    return _make_op(op, args)


def _rand_expr_tree_exact_depth(
    rng: random.Random,
    variables: List[str],
    depth: int,
    allow_if: bool,
    max_nary_arity: int = 5,
    banned_root_ops: Optional[Set[str]] = None,
) -> MechanismNode:
    """Sample a random expression whose pre-canonical depth is exactly `depth`."""
    if depth <= 1:
        return _make_var(rng.choice(variables))

    banned = banned_root_ops or set()
    op_choices: List[str] = []

    if "not" not in banned:
        # Keep unary expansion possible, but lower weight than n-ary operators.
        op_choices.extend(["not"])

    for op in ("and", "or", "xor", "iff"):
        if op not in banned:
            op_choices.extend([op, op, op])

    if allow_if and "if" not in banned:
        op_choices.extend(["if", "if"])

    if not op_choices:
        return _make_var(rng.choice(variables))

    op = rng.choice(op_choices)
    if op == "not":
        child = _rand_expr_tree_exact_depth(
            rng=rng,
            variables=variables,
            depth=depth - 1,
            allow_if=allow_if,
            max_nary_arity=max_nary_arity,
            banned_root_ops={"not"},
        )
        return _make_op("not", [child])

    if op == "if":
        child_depths = [rng.randint(1, depth - 1) for _ in range(3)]
        child_depths[rng.randrange(3)] = depth - 1
        args: List[MechanismNode] = []
        for child_depth in child_depths:
            child_banned = {"if"} if child_depth == (depth - 1) else None
            args.append(
                _rand_expr_tree_exact_depth(
                    rng=rng,
                    variables=variables,
                    depth=child_depth,
                    allow_if=allow_if,
                    max_nary_arity=max_nary_arity,
                    banned_root_ops=child_banned,
                )
            )
        return _make_op("if", args)

    arity = rng.randint(2, max(2, int(max_nary_arity)))
    child_depths = [rng.randint(1, depth - 1) for _ in range(arity)]
    child_depths[rng.randrange(arity)] = depth - 1
    args = []
    for child_depth in child_depths:
        child_banned = {op} if child_depth == (depth - 1) and op in {"and", "or", "xor"} else None
        args.append(
            _rand_expr_tree_exact_depth(
                rng=rng,
                variables=variables,
                depth=child_depth,
                allow_if=allow_if,
                max_nary_arity=max_nary_arity,
                banned_root_ops=child_banned,
            )
        )
    return _make_op(op, args)


def _sample_gold_mechanism(
    rng: random.Random,
    parents: List[str],
    ast_min: int,
    ast_max: int,
    allow_if: bool,
    allow_constants: bool = True,
    depth_target: Optional[int] = None,
    depth_sampler_mode: str = "canonical",
    require_all_parents_used: bool = False,
    require_each_parent_essential: bool = False,
    require_global_minimal_equiv: bool = False,
    allow_fallback: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    if depth_target is not None:
        mode = str(depth_sampler_mode or "canonical").strip().lower()
        if mode in {"stochastic", "stochastic_depth", "random", "rand"}:
            return _sample_gold_mechanism_stochastic_depth(
                rng=rng,
                parents=parents,
                ast_min=ast_min,
                ast_max=ast_max,
                allow_if=allow_if,
                allow_constants=allow_constants,
                depth_target=int(depth_target),
                require_all_parents_used=require_all_parents_used,
                require_each_parent_essential=require_each_parent_essential,
            )
        return _sample_gold_mechanism_canonical_depth(
            rng=rng,
            parents=parents,
            ast_min=ast_min,
            ast_max=ast_max,
            allow_if=allow_if,
            allow_constants=allow_constants,
            depth_target=int(depth_target),
            require_all_parents_used=require_all_parents_used,
            require_each_parent_essential=require_each_parent_essential,
        )

    allowed_ops = set(DEFAULT_ALLOWED_OPERATORS)
    if allow_if:
        allowed_ops.add("if")

    strict_requirements = bool(require_all_parents_used or require_each_parent_essential)
    max_expr_attempts = 1400 if strict_requirements else 700
    for _ in range(max_expr_attempts):
        depth = rng.randint(3, 7)
        node = _canonicalize_node(_rand_expr_tree(rng, parents, depth, allow_if=allow_if))
        if (not allow_constants) and _node_contains_const(node):
            continue
        sexpr = node_to_sexpr(node)

        try:
            parsed = parse_mechanism(
                sexpr,
                allowed_operators=allowed_ops,
                allowed_variables=set(parents),
                allow_constants=allow_constants,
            )
            stats = analyze_mechanism(parsed)
        except (MechanismParseError, MechanismEvalError):
            continue

        used = set(stats["variables"])
        if not used or not used.issubset(set(parents)):
            continue

        if require_all_parents_used and used != set(parents):
            continue

        # Reject semantically constant expressions.
        if not _node_outputs_both_bits(parsed, sorted(used)):
            continue

        if require_each_parent_essential and not _all_parents_essential(parsed, parents):
            continue

        ast_size = stats["astSize"]
        if ast_size < ast_min or ast_size > ast_max:
            continue

        # Prefer multi-parent mechanisms in v1.
        if len(used) < min(2, len(parents)):
            continue

        if require_global_minimal_equiv and int(ast_size) > 1:
            table = _truth_table_for_node(parsed, sorted(parents))
            smaller_min_map = _min_ast_by_truth_table_cached(
                num_vars=len(parents),
                ast_limit=int(ast_size) - 1,
                allowed_ops=allowed_ops,
                max_nary_arity=3,
                allow_constants=allow_constants,
            )
            if int(smaller_min_map.get(int(table), int(ast_size))) < int(ast_size):
                continue

        return sexpr, stats

    if not allow_fallback:
        raise RuntimeError(
            "Could not sample a gold mechanism satisfying the requested constraints"
        )

    # Deterministic fallback if random search fails and fallback is allowed.
    if len(parents) >= 2:
        fallback = f"(xor {parents[0]} {parents[1]})"
    else:
        fallback = parents[0]
    parsed_fallback = parse_mechanism(
        fallback,
        allowed_operators=allowed_ops,
        allowed_variables=set(parents),
        allow_constants=allow_constants,
    )
    return fallback, analyze_mechanism(parsed_fallback)


def _sample_unit_contexts(
    rng: random.Random,
    units: List[str],
    input_vars: List[str],
    must_vary_vars: List[str],
) -> Dict[str, Dict[str, int]]:
    for _ in range(200):
        contexts: Dict[str, Dict[str, int]] = {}
        for uid in units:
            row = {v: rng.randint(0, 1) for v in input_vars}
            contexts[uid] = row

        if all(len({contexts[u][v] for u in units}) > 1 for v in must_vary_vars):
            return contexts

    # fallback: force variation on the first two units
    contexts = {}
    for i, uid in enumerate(units):
        row = {v: rng.randint(0, 1) for v in input_vars}
        contexts[uid] = row
    if len(units) >= 2:
        for v in must_vary_vars:
            contexts[units[0]][v] = 0
            contexts[units[1]][v] = 1
    return contexts


def _assignments_key(assignments: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in assignments.items()))


def _sample_intervention(
    rng: random.Random,
    input_vars: List[str],
    force_var: Optional[str] = None,
    min_targets: int = 1,
    max_targets: int = 2,
    size_probs: Optional[Dict[int, float]] = None,
) -> Dict[str, int]:
    candidates = list(input_vars)
    if not candidates:
        return {}

    k = None
    if size_probs:
        weighted_sizes = [
            (int(size), float(prob))
            for size, prob in size_probs.items()
            if int(size) > 0 and float(prob) > 0.0 and int(size) <= len(candidates)
        ]
        if weighted_sizes:
            sizes = [s for s, _ in weighted_sizes]
            weights = [w for _, w in weighted_sizes]
            k = int(rng.choices(sizes, weights=weights, k=1)[0])

    if k is None:
        k = rng.randint(min_targets, max_targets)
        k = min(k, len(candidates))

    picked = set(rng.sample(candidates, k))
    if force_var is not None:
        picked.add(force_var)

    return {v: rng.randint(0, 1) for v in sorted(picked)}


def _sample_root_thresholds(
    rng: random.Random,
    units: List[str],
    root_vars: List[str],
) -> Dict[str, Dict[str, float]]:
    return {
        uid: {rv: rng.random() for rv in root_vars}
        for uid in units
    }


def _sample_world_env_profile(
    rng: random.Random,
    root_vars: List[str],
    levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    support = [0.2, 0.35, 0.5, 0.65, 0.8]
    if levels:
        cleaned = [float(x) for x in levels if 0.0 < float(x) < 1.0]
        if cleaned:
            support = cleaned
    profile = {rv: float(rng.choice(support)) for rv in root_vars}
    return profile


def _materialize_root_contexts(
    units: List[str],
    root_vars: List[str],
    unit_thresholds: Dict[str, Dict[str, float]],
    env_profile: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for uid in units:
        th = unit_thresholds.get(uid, {})
        out[uid] = {
            rv: int(float(th.get(rv, 0.5)) < float(env_profile.get(rv, 0.5)))
            for rv in root_vars
        }
    return out


def _sample_intervention_mode(
    rng: random.Random,
    mode_probs: Optional[Dict[str, float]] = None,
) -> str:
    probs = mode_probs or {"hard_constant": 0.5, "hard_assigned": 0.5}
    c = float(probs.get("hard_constant", 0.0))
    a = float(probs.get("hard_assigned", 0.0))
    if c <= 0 and a <= 0:
        return "hard_constant"
    if c <= 0:
        return "hard_assigned"
    if a <= 0:
        return "hard_constant"
    return str(rng.choices(["hard_constant", "hard_assigned"], weights=[c, a], k=1)[0])


def _sample_assigned_values_by_unit(
    rng: random.Random,
    units: List[str],
    targets: List[str],
    base_bias: float = 0.5,
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {uid: {} for uid in units}
    bias = min(max(float(base_bias), 0.05), 0.95)
    for t in targets:
        vals = [1 if rng.random() < bias else 0 for _ in units]
        if len(units) >= 2 and len(set(vals)) == 1:
            vals[0] = 0
            vals[1] = 1
        for uid, value in zip(units, vals):
            out[uid][t] = int(value)
    return out


def _intervention_plan_key(
    mode: str,
    assignments: Dict[str, int],
    targets: List[str],
) -> Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]:
    return (
        str(mode).strip().lower(),
        tuple(sorted((str(k), int(v)) for k, v in assignments.items())),
        tuple(sorted(str(t) for t in targets)),
    )


def _intervention_plan_key_cached(
    plan: Dict[str, Any],
) -> Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]:
    cached = plan.get("_plan_key")
    if isinstance(cached, tuple) and len(cached) == 3:
        return cached  # type: ignore[return-value]
    key = _intervention_plan_key(
        mode=str(plan.get("mode", "")),
        assignments=dict(plan.get("assignments") or {}),
        targets=list(plan.get("targets") or []),
    )
    plan["_plan_key"] = key
    return key


def _intervention_plan_target_set(plan: Dict[str, Any]) -> AbstractSet[str]:
    cached = plan.get("_plan_target_set")
    if isinstance(cached, (set, frozenset)):
        return cached
    targets = frozenset(str(v) for v in (plan.get("targets") or []))
    plan["_plan_target_set"] = targets
    return targets


def _mechanism_matches_rows(
    node: MechanismNode,
    rows: List[Dict[str, Any]],
    target_var: str,
) -> bool:
    for row in rows:
        if target_var not in row:
            continue
        y_true = int(row[target_var])
        y_hat = int(evaluate_parsed_mechanism(node, row))
        if y_true != y_hat:
            return False
    return True


def _mechanism_mismatch_count(
    node: MechanismNode,
    rows: List[Dict[str, Any]],
    target_var: str,
) -> int:
    mismatches = 0
    for row in rows:
        if target_var not in row:
            continue
        y_true = int(row[target_var])
        y_hat = int(evaluate_parsed_mechanism(node, row))
        if y_true != y_hat:
            mismatches += 1
    return mismatches


def _compile_mechanism_truth_table(
    node: MechanismNode,
) -> MechanismTruthPayload:
    deps = tuple(sorted(mechanism_variables(node), key=_natural_var_key))
    table: Dict[Tuple[int, ...], int] = {}
    for bits in itertools.product((0, 1), repeat=len(deps)):
        env = {dep: int(bit) for dep, bit in zip(deps, bits)}
        table[tuple(int(bit) for bit in bits)] = int(evaluate_parsed_mechanism(node, env))
    return deps, table


def _truth_table_ones_patterns_cached(
    truth_table: TruthTable,
) -> Tuple[int, ...]:
    table_id = int(id(truth_table))
    cached = _lru_get(_TRUTH_TABLE_ONES_PATTERNS_CACHE, table_id)
    current_size = int(len(truth_table))
    if cached is not None and int(cached[0]) == int(current_size):
        return tuple(int(x) for x in cached[1])
    if cached is not None:
        # Guard against id reuse with changed table size.
        _TRUTH_TABLE_ONES_PATTERNS_CACHE.pop(table_id, None)

    ones_patterns: List[int] = []
    for bits, value in truth_table.items():
        if int(value) != 1:
            continue
        pattern = 0
        for bit_idx, bit in enumerate(bits):
            if int(bit):
                pattern |= (1 << int(bit_idx))
        ones_patterns.append(int(pattern))

    out = tuple(int(x) for x in ones_patterns)
    _lru_set(
        _TRUTH_TABLE_ONES_PATTERNS_CACHE,
        table_id,
        (int(current_size), out),
        _TRUTH_TABLE_ONES_PATTERNS_CACHE_MAX,
    )
    return out


def _mechanism_mismatch_count_from_truth_table(
    *,
    dep_order: Tuple[str, ...],
    truth_table: TruthTable,
    rows: List[Dict[str, Any]],
    target_var: str,
) -> int:
    mismatches = 0
    for row in rows:
        key = tuple(int(row[dep]) for dep in dep_order)
        y_hat = int(truth_table[key])
        if int(row[target_var]) != y_hat:
            mismatches += 1
    return mismatches


def _truth_table_output_mask_from_bitsets(
    *,
    dep_order: Tuple[str, ...],
    truth_table: TruthTable,
    var_bitsets: Dict[str, int],
    full_units_mask: int,
) -> int:
    out_mask = 0
    ones_patterns = _truth_table_ones_patterns_cached(truth_table)
    dep_masks = tuple(
        int(var_bitsets.get(str(dep), 0)) & int(full_units_mask)
        for dep in dep_order
    )
    for pattern in ones_patterns:
        match_mask = int(full_units_mask)
        for dep_idx, dep_mask in enumerate(dep_masks):
            if (int(pattern) >> int(dep_idx)) & 1:
                match_mask &= int(dep_mask)
            else:
                match_mask &= int((~int(dep_mask)) & int(full_units_mask))
            if int(match_mask) == 0:
                break
        out_mask |= int(match_mask)
        if int(out_mask) == int(full_units_mask):
            break
    return int(out_mask) & int(full_units_mask)


def _mechanism_mismatch_count_from_truth_table_bitsets(
    *,
    dep_order: Tuple[str, ...],
    truth_table: TruthTable,
    var_bitsets: Dict[str, int],
    target_var: str,
    full_units_mask: int,
) -> int:
    target_mask = int(var_bitsets.get(str(target_var), 0)) & int(full_units_mask)
    pred_mask = _truth_table_output_mask_from_bitsets(
        dep_order=dep_order,
        truth_table=truth_table,
        var_bitsets=var_bitsets,
        full_units_mask=int(full_units_mask),
    )
    return int((int(target_mask) ^ int(pred_mask)) & int(full_units_mask)).bit_count()


def _add_distractor_candidate(
    pool: Dict[str, Dict[str, Any]],
    expr: str,
    category_hint: str,
    input_vars: List[str],
    parents: List[str],
    allowed_ops: List[str],
    ast_cap: int,
    gold_expr: str,
    *,
    allow_constants: bool = True,
) -> None:
    expr_txt = str(expr).strip()
    if not expr_txt or expr_txt == gold_expr or expr_txt in pool:
        return

    try:
        node = parse_mechanism(
            expr_txt,
            allowed_operators=set(allowed_ops),
            allowed_variables=set(input_vars),
            allow_constants=allow_constants,
        )
        stats = analyze_mechanism(node)
    except (MechanismParseError, MechanismEvalError):
        return

    ast_size = int(stats.get("astSize", 0))
    if ast_size < 1 or ast_size > int(ast_cap):
        return

    vars_used = set(stats.get("variables", []))
    parent_set = set(parents)
    nonparent_set = set(input_vars) - parent_set

    category = category_hint
    if vars_used and vars_used.issubset(nonparent_set):
        category = "nonparent_only"
    elif vars_used and vars_used.issubset(parent_set) and vars_used != parent_set:
        category = "parent_subset"
    elif vars_used & parent_set and vars_used & nonparent_set:
        category = "mixed"
    elif category == "":
        category = "other"

    pool[expr_txt] = {
        "expr": expr_txt,
        "node": node,
        "ast": ast_size,
        "vars": vars_used,
        "category": category,
    }


def _build_distractor_pool(
    seed: int,
    input_vars: List[str],
    parents: List[str],
    allowed_ops: List[str],
    gold_expr: str,
    ast_cap: int,
    pool_size: int,
    allow_if: bool,
    allow_constants: bool = True,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    parent_set = set(parents)
    nonparents = [v for v in input_vars if v not in parent_set]
    pool: Dict[str, Dict[str, Any]] = {}

    def _add(expr: str, category: str) -> None:
        _add_distractor_candidate(
            pool,
            expr,
            category,
            input_vars,
            parents,
            allowed_ops,
            ast_cap,
            gold_expr,
            allow_constants=allow_constants,
        )

    # Small baseline families.
    if allow_constants:
        _add("0", "baseline")
        _add("1", "baseline")
    for v in input_vars:
        _add(v, "baseline")
        _add(f"(not {v})", "baseline")

    pair_ops = [op for op in ("and", "or", "xor", "iff") if op in set(allowed_ops)]
    for i, a in enumerate(input_vars):
        for b in input_vars[i + 1 :]:
            for op in pair_ops:
                _add(f"({op} {a} {b})", "baseline")
            _add(f"(and {a} (not {b}))", "baseline")
            _add(f"(and (not {a}) {b})", "baseline")

    # Limited ternary baseline family for stronger distractors while keeping pool compact.
    max_ternary = max(0, min(80, len(input_vars) ** 3))
    ternary_count = 0
    for combo in itertools.combinations(input_vars, 3):
        if ternary_count >= max_ternary:
            break
        a, b, c = combo
        for op in pair_ops:
            _add(f"({op} {a} {b} {c})", "baseline")
        ternary_count += 1

    # Randomized candidates in required categories.
    category_targets = {
        "nonparent_only": max(24, pool_size // 8),
        "parent_subset": max(24, pool_size // 8),
        "mixed": max(24, pool_size // 8),
    }
    category_counts = {key: 0 for key in category_targets}

    def _candidate_vars(mode: str) -> List[str]:
        if mode == "nonparent_only":
            return nonparents if nonparents else input_vars
        if mode == "parent_subset":
            if len(parents) <= 1:
                return parents if parents else input_vars
            subset_size = rng.randint(1, len(parents) - 1)
            return sorted(rng.sample(parents, subset_size))
        # mixed
        have_parent = sorted(rng.sample(parents, min(1, len(parents)))) if parents else []
        have_nonparent = (
            sorted(rng.sample(nonparents, min(1, len(nonparents)))) if nonparents else []
        )
        remaining = [v for v in input_vars if v not in set(have_parent + have_nonparent)]
        extra_n = rng.randint(0, min(2, len(remaining)))
        extras = sorted(rng.sample(remaining, extra_n)) if extra_n else []
        vars_mixed = sorted(set(have_parent + have_nonparent + extras))
        return vars_mixed if vars_mixed else input_vars

    allowed_ops_set = set(allowed_ops)
    if allow_if:
        allowed_ops_set.add("if")

    random_attempts = 0
    max_random_attempts = max(600, pool_size * 20)
    while random_attempts < max_random_attempts and any(
        category_counts[c] < category_targets[c] for c in category_targets
    ):
        random_attempts += 1
        category = min(
            category_targets.keys(),
            key=lambda c: (category_counts[c] / max(1, category_targets[c]), c),
        )
        vars_for_expr = _candidate_vars(category)
        if not vars_for_expr:
            continue
        depth = rng.randint(2, 6)
        expr_node = _rand_expr_tree(rng, vars_for_expr, depth, allow_if=allow_if)
        expr = node_to_sexpr(expr_node)

        before = len(pool)
        _add_distractor_candidate(
            pool,
            expr,
            category,
            input_vars,
            parents,
            list(allowed_ops_set),
            ast_cap,
            gold_expr,
            allow_constants=allow_constants,
        )
        if len(pool) > before:
            added = pool[expr]
            added_cat = str(added.get("category", ""))
            if added_cat in category_counts:
                category_counts[added_cat] += 1

    # Build final pool with category coverage first, then fill by smallest AST.
    all_entries = list(pool.values())
    for entry in all_entries:
        entry["vars"] = set(entry.get("vars", set()))

    by_cat: Dict[str, List[Dict[str, Any]]] = {
        "nonparent_only": [],
        "parent_subset": [],
        "mixed": [],
        "baseline": [],
        "other": [],
    }
    for entry in all_entries:
        cat = str(entry.get("category", "other"))
        if cat not in by_cat:
            cat = "other"
        by_cat[cat].append(entry)

    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: (int(e["ast"]), str(e["expr"])))

    selected: List[Dict[str, Any]] = []
    selected_exprs: Set[str] = set()

    for cat in ("nonparent_only", "parent_subset", "mixed", "baseline"):
        quota = max(8, pool_size // 12)
        for entry in by_cat.get(cat, [])[:quota]:
            expr = str(entry["expr"])
            if expr in selected_exprs:
                continue
            selected.append(entry)
            selected_exprs.add(expr)

    remaining = sorted(
        all_entries,
        key=lambda e: (int(e["ast"]), str(e["expr"])),
    )
    for entry in remaining:
        if len(selected) >= pool_size:
            break
        expr = str(entry["expr"])
        if expr in selected_exprs:
            continue
        selected.append(entry)
        selected_exprs.add(expr)

    selected.sort(key=lambda e: (int(e["ast"]), str(e["expr"])))
    return selected


def _compute_survivor_diagnostics(
    distractors: List[Dict[str, Any]],
    survivor_indices: List[int],
    gold_ast_size: int,
) -> Dict[str, Any]:
    survivor_entries = [distractors[i] for i in survivor_indices]
    survivors_total = len(survivor_entries)
    survivors_small = sum(1 for e in survivor_entries if int(e["ast"]) <= int(gold_ast_size))

    min_ast_survivor = None
    if survivor_entries:
        min_ast_survivor = min(int(e["ast"]) for e in survivor_entries)

    gap_to_second_best = None
    if min_ast_survivor is not None:
        gap_to_second_best = int(min_ast_survivor) - int(gold_ast_size)

    return {
        "survivors_total": survivors_total,
        "survivors_small": survivors_small,
        "min_ast_survivor_excluding_gold": min_ast_survivor,
        "gap_to_second_best": gap_to_second_best,
    }


def _propose_intervention_candidates(
    rng: random.Random,
    input_vars: List[str],
    seen_interventions: Set[Tuple[Tuple[str, int], ...]],
    force_var: Optional[str],
    num_candidates: int,
    min_targets: int,
    max_targets: int,
    size_probs: Optional[Dict[int, float]],
) -> List[Dict[str, int]]:
    proposals: List[Dict[str, int]] = []
    seen_local: Set[Tuple[Tuple[str, int], ...]] = set()

    attempts = 0
    max_attempts = max(40, num_candidates * 12)
    while len(proposals) < num_candidates and attempts < max_attempts:
        attempts += 1
        ints = _sample_intervention(
            rng,
            input_vars,
            force_var=force_var,
            min_targets=min_targets,
            max_targets=max_targets,
            size_probs=size_probs,
        )
        key = _assignments_key(ints)
        if key in seen_interventions or key in seen_local:
            continue
        seen_local.add(key)
        proposals.append(ints)

    return proposals


def _build_train_worlds_cegis_lite(
    rng: random.Random,
    units: List[str],
    input_vars: List[str],
    target_var: str,
    base_contexts: Dict[str, Dict[str, int]],
    gold_node: MechanismNode,
    parents: List[str],
    k: int,
    intervention_size_probs: Optional[Dict[int, float]],
    candidate_interventions: int,
    distractors: List[Dict[str, Any]],
    gold_ast_size: int,
    min_survivors_small_to_keep: int = 0,
    exact_witness_ast_cap: Optional[int] = None,
    exact_witness_time_budget_ms: Optional[int] = None,
    exact_witness_max_signatures_per_size: Optional[int] = None,
    allowed_ops: Optional[List[str]] = None,
    allow_constants: bool = True,
) -> Tuple[List[CausalWorldView], Set[Tuple[Tuple[str, int], ...]], List[int], Dict[str, Any]]:
    train_worlds: List[CausalWorldView] = []
    seen_interventions: Set[Tuple[Tuple[str, int], ...]] = set()

    # Baseline observational world.
    baseline = {}
    baseline_world = _make_world(
        world_id="train_00",
        split="train",
        units=units,
        input_vars=input_vars,
        target_var=target_var,
        base_contexts=base_contexts,
        interventions=baseline,
        gold_node=gold_node,
    )
    train_worlds.append(baseline_world)
    seen_interventions.add(_assignments_key(baseline))

    baseline_rows = _iter_world_rows(baseline_world)
    survivor_indices = [
        i
        for i, cand in enumerate(distractors)
        if _mechanism_matches_rows(cand["node"], baseline_rows, target_var)
    ]

    force_queue = parents.copy()
    rng.shuffle(force_queue)
    kill_trace: List[int] = []
    exact_witness_kills_trace: List[int] = []
    exact_witness_ast_trace: List[Optional[int]] = []
    distractor_truth_cache: Dict[int, Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = {}

    while len(train_worlds) < k:
        force_var = force_queue.pop() if force_queue else None
        max_targets = 2 if rng.random() < 0.8 else 3
        proposals = _propose_intervention_candidates(
            rng=rng,
            input_vars=input_vars,
            seen_interventions=seen_interventions,
            force_var=force_var,
            num_candidates=candidate_interventions,
            min_targets=1,
            max_targets=max_targets,
            size_probs=intervention_size_probs,
        )

        if not proposals:
            # Fallback to ensure progress.
            proposals = _propose_intervention_candidates(
                rng=rng,
                input_vars=input_vars,
                seen_interventions=seen_interventions,
                force_var=None,
                num_candidates=1,
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
            )
            if not proposals:
                break

        smallest_ast_survivors: List[int] = []
        if survivor_indices:
            min_ast = min(int(distractors[idx]["ast"]) for idx in survivor_indices)
            smallest_ast_survivors = [
                idx for idx in survivor_indices if int(distractors[idx]["ast"]) == min_ast
            ]

        exact_witness = None
        exact_witness_node = None
        exact_witness_truth_payload: Optional[Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = None
        if exact_witness_ast_cap is not None:
            all_rows_current: List[Dict[str, Any]] = []
            for world in train_worlds:
                all_rows_current.extend(_iter_world_rows(world))
            exact_witness = _find_small_shortcut_witness(
                rows=all_rows_current,
                target_var=target_var,
                candidate_vars=input_vars,
                ast_cap=int(exact_witness_ast_cap),
                allowed_operators=allowed_ops,
                allow_constants=allow_constants,
                time_budget_ms=exact_witness_time_budget_ms,
                max_signatures_per_size=exact_witness_max_signatures_per_size,
            )
            if exact_witness is not None:
                exact_witness_node = exact_witness["node"]
                exact_witness_truth_payload = _compile_mechanism_truth_table(exact_witness_node)

        best = None
        best_feasible = None
        best_positive = None
        best_positive_feasible = None
        for ints in proposals:
            candidate_world = _make_world(
                world_id="candidate",
                split="train",
                units=units,
                input_vars=input_vars,
                target_var=target_var,
                base_contexts=base_contexts,
                interventions=ints,
                gold_node=gold_node,
            )
            rows = _iter_world_rows(candidate_world)
            survivors_after: List[int] = []
            killed = 0
            mismatch_by_idx: Dict[int, int] = {}
            for idx in survivor_indices:
                truth_payload = distractor_truth_cache.get(int(idx))
                if truth_payload is None:
                    truth_payload = _compile_mechanism_truth_table(distractors[idx]["node"])
                    distractor_truth_cache[int(idx)] = truth_payload
                mismatch = _mechanism_mismatch_count_from_truth_table(
                    dep_order=truth_payload[0],
                    truth_table=truth_payload[1],
                    rows=rows,
                    target_var=target_var,
                )
                mismatch_by_idx[int(idx)] = int(mismatch)
                if int(mismatch) == 0:
                    survivors_after.append(idx)
                else:
                    killed += 1

            disagreement = 0
            for idx in smallest_ast_survivors:
                disagreement += int(mismatch_by_idx.get(int(idx), 0))

            exact_witness_mismatch = 0
            exact_witness_killed = 0
            if exact_witness_node is not None:
                if exact_witness_truth_payload is not None:
                    exact_witness_mismatch = _mechanism_mismatch_count_from_truth_table(
                        dep_order=exact_witness_truth_payload[0],
                        truth_table=exact_witness_truth_payload[1],
                        rows=rows,
                        target_var=target_var,
                    )
                else:
                    exact_witness_mismatch = _mechanism_mismatch_count(
                        exact_witness_node, rows, target_var
                    )
                exact_witness_killed = int(exact_witness_mismatch > 0)

            if exact_witness_node is not None:
                score = (
                    int(exact_witness_killed),
                    int(exact_witness_mismatch),
                    int(killed),
                    int(disagreement),
                )
            else:
                score = (int(killed), int(disagreement))
            key = _assignments_key(ints)
            tie_key = tuple(sorted((k, v) for k, v in key))
            survivors_small_after = sum(
                1 for idx in survivors_after if int(distractors[idx]["ast"]) <= int(gold_ast_size)
            )
            feasible = survivors_small_after >= int(min_survivors_small_to_keep)

            if (
                best is None
                or score > best["score"]
                or (score == best["score"] and tie_key < best["tie_key"])
            ):
                best = {
                    "ints": ints,
                    "rows": rows,
                    "survivors_after": survivors_after,
                    "killed": killed,
                    "exact_witness_killed": exact_witness_killed,
                    "exact_witness_mismatch": exact_witness_mismatch,
                    "score": score,
                    "tie_key": tie_key,
                }
            if feasible and (
                best_feasible is None
                or score > best_feasible["score"]
                or (score == best_feasible["score"] and tie_key < best_feasible["tie_key"])
            ):
                best_feasible = {
                    "ints": ints,
                    "rows": rows,
                    "survivors_after": survivors_after,
                    "killed": killed,
                    "exact_witness_killed": exact_witness_killed,
                    "exact_witness_mismatch": exact_witness_mismatch,
                    "score": score,
                    "tie_key": tie_key,
                }
            if int(killed) > 0 and (
                best_positive is None
                or score > best_positive["score"]
                or (score == best_positive["score"] and tie_key < best_positive["tie_key"])
            ):
                best_positive = {
                    "ints": ints,
                    "rows": rows,
                    "survivors_after": survivors_after,
                    "killed": killed,
                    "exact_witness_killed": exact_witness_killed,
                    "exact_witness_mismatch": exact_witness_mismatch,
                    "score": score,
                    "tie_key": tie_key,
                }
            if int(killed) > 0 and feasible and (
                best_positive_feasible is None
                or score > best_positive_feasible["score"]
                or (score == best_positive_feasible["score"] and tie_key < best_positive_feasible["tie_key"])
            ):
                best_positive_feasible = {
                    "ints": ints,
                    "rows": rows,
                    "survivors_after": survivors_after,
                    "killed": killed,
                    "exact_witness_killed": exact_witness_killed,
                    "exact_witness_mismatch": exact_witness_mismatch,
                    "score": score,
                    "tie_key": tie_key,
                }

        selected = None
        # If survivors remain, prioritize worlds that kill at least one distractor.
        if survivor_indices:
            selected = (
                best_positive_feasible
                if best_positive_feasible is not None
                else (
                    best_positive
                    if best_positive is not None
                    else (best_feasible if best_feasible is not None else best)
                )
            )
        else:
            selected = best_feasible if best_feasible is not None else best
        assert selected is not None
        ints = selected["ints"]
        key = _assignments_key(ints)
        seen_interventions.add(key)

        world_id = f"train_{len(train_worlds):02d}"
        selected_world = _make_world(
            world_id=world_id,
            split="train",
            units=units,
            input_vars=input_vars,
            target_var=target_var,
            base_contexts=base_contexts,
            interventions=ints,
            gold_node=gold_node,
        )
        train_worlds.append(selected_world)
        survivor_indices = list(selected["survivors_after"])
        kill_trace.append(int(selected["killed"]))
        exact_witness_kills_trace.append(int(selected.get("exact_witness_killed", 0)))
        if exact_witness is None:
            exact_witness_ast_trace.append(None)
        else:
            exact_witness_ast_trace.append(int(exact_witness.get("ast", 0)))

    return train_worlds, seen_interventions, survivor_indices, {
        "kills_per_world": kill_trace,
        "exact_witness_kills_per_world": exact_witness_kills_trace,
        "exact_witness_ast_per_world": exact_witness_ast_trace,
        "exact_witness_cap": (
            None if exact_witness_ast_cap is None else int(exact_witness_ast_cap)
        ),
        "exact_witness_time_budget_ms": (
            None
            if exact_witness_time_budget_ms is None
            else int(exact_witness_time_budget_ms)
        ),
        "exact_witness_max_signatures_per_size": (
            None
            if exact_witness_max_signatures_per_size is None
            else int(exact_witness_max_signatures_per_size)
        ),
    }


def _format_do(assignments: Dict[str, int]) -> str:
    if not assignments:
        return "none"
    parts = [f"{k}={v}" for k, v in sorted(assignments.items())]
    return "do(" + ", ".join(parts) + ")"


def _format_do_mode(
    mode: str,
    assignments: Dict[str, int],
    targets: List[str],
) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key == "hard_assigned":
        if not targets:
            return "none"
        parts = [f"{t}=assigned_per_row" for t in sorted(set(targets), key=_natural_var_key)]
        return "do(" + ", ".join(parts) + ")"
    return _format_do(assignments)


def _normalize_structured_intervention_fields(world: Dict[str, Any]) -> Dict[str, Any]:
    extra = world.get("extra") or {}
    assignments_old = _extract_world_intervention_assignments(world, prefer_structured=False)
    targets_old = _extract_world_intervention_targets(world, prefer_structured=False)
    mode_old = _extract_world_intervention_mode(world, prefer_structured=False)

    mode_key = str(extra.get("InterventionMode", mode_old)).strip().lower()
    assigned_raw = extra.get("InterventionTargetsAssigned")
    constant_raw = extra.get("InterventionTargetsConstant")
    all_raw = extra.get("InterventionTargetsAll")

    structured_present = any(
        key in extra
        for key in (
            "InterventionMode",
            "InterventionTargetsAssigned",
            "InterventionTargetsConstant",
            "InterventionTargetsAll",
        )
    )

    assigned: List[str]
    if isinstance(assigned_raw, list):
        assigned = _sorted_unique_natural([str(v) for v in assigned_raw])
    elif structured_present:
        assigned = []
    elif mode_key == "hard_assigned":
        assigned = _sorted_unique_natural(list(targets_old))
    else:
        assigned = []

    constant: Dict[str, int]
    if isinstance(constant_raw, dict):
        constant = {str(k): int(v) for k, v in constant_raw.items()}
    elif structured_present:
        constant = {}
    elif mode_key == "hard_constant":
        constant = {str(k): int(v) for k, v in assignments_old.items()}
    else:
        constant = {}

    if isinstance(all_raw, list):
        all_targets = _sorted_unique_natural([str(v) for v in all_raw])
    elif structured_present:
        all_targets = _sorted_unique_natural([*assigned, *constant.keys()])
    elif mode_key == "hard_assigned":
        all_targets = _sorted_unique_natural([*assigned, *constant.keys()])
    else:
        all_targets = _sorted_unique_natural(list(constant.keys()) or list(targets_old))

    if not all_targets and not constant:
        mode_norm = "none"
        assigned = []
        constant = {}
        all_targets = []
    else:
        mode_norm = mode_key if mode_key in {"hard_constant", "hard_assigned"} else "hard_constant"
        if mode_norm == "hard_constant":
            assigned = []
            all_targets = _sorted_unique_natural(list(constant.keys()))
        elif mode_norm == "hard_assigned":
            assigned = _sorted_unique_natural(assigned)
            all_targets = _sorted_unique_natural([*assigned, *constant.keys()])

    return {
        "InterventionMode": mode_norm,
        "InterventionTargetsAssigned": assigned,
        "InterventionTargetsConstant": constant,
        "InterventionTargetsAll": all_targets,
    }


def _make_world(
    world_id: str,
    split: str,
    units: List[str],
    input_vars: List[str],
    target_var: str,
    base_contexts: Dict[str, Dict[str, int]],
    interventions: Dict[str, int],
    gold_node: MechanismNode,
) -> CausalWorldView:
    rows = []
    for uid in units:
        values = dict(base_contexts[uid])
        values.update(interventions)
        values[target_var] = evaluate_parsed_mechanism(gold_node, values)
        rows.append({"unitId": uid, "values": values})

    return CausalWorldView(
        worldId=world_id,
        domain=units,
        domainSize=len(units),
        observationMode="panel_full",
        interventions=[{"type": "hard_do", "assignments": interventions}],
        extra={
            "split": split,
            "rows": rows,
            "do": _format_do(interventions),
        },
    )


def _world_extra(world: WorldLike) -> Dict[str, Any]:
    if isinstance(world, CausalWorldView):
        extra = world.extra or {}
    elif isinstance(world, dict):
        extra = world.get("extra") or {}
    else:
        extra = {}
    return extra if isinstance(extra, dict) else {}


def _world_interventions(world: WorldLike) -> List[Dict[str, Any]]:
    if isinstance(world, CausalWorldView):
        interventions = world.interventions or []
    elif isinstance(world, dict):
        interventions = world.get("interventions") or []
    else:
        interventions = []
    if not isinstance(interventions, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in interventions:
        if isinstance(it, dict):
            out.append(it)
    return out


def _world_id(world: WorldLike) -> str:
    if isinstance(world, CausalWorldView):
        return str(world.worldId)
    if isinstance(world, dict):
        return str(world.get("worldId", ""))
    return ""


def _extract_world_intervention_assignments(
    world: WorldLike,
    *,
    prefer_structured: bool = True,
) -> Dict[str, int]:
    if prefer_structured:
        extra = _world_extra(world)
        if "InterventionTargetsConstant" in extra:
            raw = extra.get("InterventionTargetsConstant")
            if isinstance(raw, dict):
                return {str(k): int(v) for k, v in raw.items()}
            return {}

    out: Dict[str, int] = {}
    for it in _world_interventions(world):
        assignments = (it or {}).get("assignments", {}) or {}
        if not isinstance(assignments, dict):
            continue
        for key, value in assignments.items():
            out[str(key)] = int(value)
    return out


def _extract_world_intervention_targets(
    world: WorldLike,
    *,
    prefer_structured: bool = True,
) -> Set[str]:
    if prefer_structured:
        extra = _world_extra(world)
        if "InterventionTargetsAll" in extra:
            listed = extra.get("InterventionTargetsAll") or []
            if isinstance(listed, list):
                return {str(v) for v in listed}
            return set()
        if "InterventionTargetsAssigned" in extra or "InterventionTargetsConstant" in extra:
            out: Set[str] = set()
            assigned = extra.get("InterventionTargetsAssigned") or []
            constant = extra.get("InterventionTargetsConstant") or {}
            if isinstance(assigned, list):
                out.update(str(v) for v in assigned)
            if isinstance(constant, dict):
                out.update(str(k) for k in constant.keys())
            return out

    targets: Set[str] = set()
    for it in _world_interventions(world):
        obj = it or {}
        assignments = obj.get("assignments", {}) or {}
        if isinstance(assignments, dict):
            for key in assignments.keys():
                targets.add(str(key))
        listed = obj.get("targets") or []
        if isinstance(listed, list):
            for key in listed:
                targets.add(str(key))
    return targets


def _extract_world_intervention_mode(
    world: WorldLike,
    *,
    prefer_structured: bool = True,
) -> str:
    if prefer_structured:
        extra = _world_extra(world)
        mode_structured = str(extra.get("InterventionMode", "")).strip().lower()
        if mode_structured in {"none", "hard_constant", "hard_assigned"}:
            return mode_structured

    for it in _world_interventions(world):
        mode = str((it or {}).get("mode", "")).strip().lower()
        if mode in {"hard_constant", "hard_assigned"}:
            return mode
    return (
        "none"
        if not _extract_world_intervention_targets(world, prefer_structured=False)
        else "hard_constant"
    )


def _jaccard_distance(lhs: AbstractSet[str], rhs: AbstractSet[str]) -> float:
    union = set(lhs).union(rhs)
    if not union:
        return 0.0
    overlap = len(set(lhs).intersection(rhs))
    return 1.0 - (float(overlap) / float(len(union)))


def _scm_compute_split_scoring_coverage(
    worlds: List[WorldLike],
    endogenous_vars: List[str],
) -> Dict[str, Any]:
    payload_cache: Dict[int, Tuple[Set[str], List[Dict[str, Any]]]] = {}
    by_var: Dict[str, Dict[str, int]] = {
        str(var): {
            "scored_worlds": 0,
            "scored_cells": 0,
            "intervened_worlds": 0,
        }
        for var in endogenous_vars
    }
    mode_counts: Dict[str, int] = {}

    for world in worlds:
        intervened, rows = _scm_world_rows_and_intervened(world, payload_cache=payload_cache)
        row_count = int(len(rows))
        mode = _extract_world_intervention_mode(world)
        mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
        for var in endogenous_vars:
            var_key = str(var)
            stats = by_var[var_key]
            if var_key in intervened:
                stats["intervened_worlds"] = int(stats["intervened_worlds"]) + 1
                continue
            if row_count > 0:
                stats["scored_worlds"] = int(stats["scored_worlds"]) + 1
            stats["scored_cells"] = int(stats["scored_cells"]) + int(row_count)

    min_scored_worlds = (
        min(int(payload["scored_worlds"]) for payload in by_var.values()) if by_var else 0
    )
    min_scored_cells = (
        min(int(payload["scored_cells"]) for payload in by_var.values()) if by_var else 0
    )
    max_intervened_worlds = (
        max(int(payload["intervened_worlds"]) for payload in by_var.values()) if by_var else 0
    )
    return {
        "by_var": by_var,
        "summary": {
            "min_scored_worlds_any_endogenous": int(min_scored_worlds),
            "min_scored_cells_any_endogenous": int(min_scored_cells),
            "max_intervened_worlds_any_endogenous": int(max_intervened_worlds),
        },
        "mode_counts": {
            key: int(value)
            for key, value in sorted(mode_counts.items(), key=lambda kv: str(kv[0]))
        },
    }


def _scm_build_scoring_coverage_diagnostics(
    train_worlds: List[WorldLike],
    heldout_worlds: List[WorldLike],
    endogenous_vars: List[str],
) -> Dict[str, Any]:
    train_cov = _scm_compute_split_scoring_coverage(train_worlds, endogenous_vars)
    heldout_cov = _scm_compute_split_scoring_coverage(heldout_worlds, endogenous_vars)
    by_var: Dict[str, Dict[str, int]] = {}
    for var in sorted({str(v) for v in endogenous_vars}, key=_natural_var_key):
        train_stats = (train_cov.get("by_var") or {}).get(var, {})
        heldout_stats = (heldout_cov.get("by_var") or {}).get(var, {})
        by_var[var] = {
            "scored_worlds": int(train_stats.get("scored_worlds", 0)),
            "scored_cells": int(train_stats.get("scored_cells", 0)),
            "intervened_worlds": int(train_stats.get("intervened_worlds", 0)),
            "heldout_scored_worlds": int(heldout_stats.get("scored_worlds", 0)),
            "heldout_scored_cells": int(heldout_stats.get("scored_cells", 0)),
        }

    heldout_min_scored_worlds = (
        min(int(payload["heldout_scored_worlds"]) for payload in by_var.values()) if by_var else 0
    )
    heldout_min_scored_cells = (
        min(int(payload["heldout_scored_cells"]) for payload in by_var.values()) if by_var else 0
    )
    summary = dict(train_cov.get("summary") or {})
    summary.update(
        {
            "heldout_min_scored_worlds_any_endogenous": int(heldout_min_scored_worlds),
            "heldout_min_scored_cells_any_endogenous": int(heldout_min_scored_cells),
        }
    )
    return {
        "by_var": by_var,
        "summary": summary,
        "train_mode_counts": dict(train_cov.get("mode_counts") or {}),
        "heldout_mode_counts": dict(heldout_cov.get("mode_counts") or {}),
    }


def _scm_training_coverage_failure_reason(
    coverage_diag: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[str]:
    summary = coverage_diag.get("summary") or {}
    train_mode_counts = coverage_diag.get("train_mode_counts") or {}

    min_scored_worlds = spec.get("min_scored_worlds_per_endogenous")
    if min_scored_worlds is not None and int(summary.get("min_scored_worlds_any_endogenous", 0)) < int(
        min_scored_worlds
    ):
        return "min_scored_worlds_per_endogenous"

    min_scored_cells = spec.get("min_scored_cells_per_endogenous")
    if min_scored_cells is not None and int(summary.get("min_scored_cells_any_endogenous", 0)) < int(
        min_scored_cells
    ):
        return "min_scored_cells_per_endogenous"

    max_intervened_worlds = spec.get("max_intervened_worlds_per_endogenous")
    if max_intervened_worlds is not None and int(
        summary.get("max_intervened_worlds_any_endogenous", 0)
    ) > int(max_intervened_worlds):
        return "max_intervened_worlds_per_endogenous"

    min_assigned = spec.get("min_hard_assigned_worlds")
    if min_assigned is not None and int(train_mode_counts.get("hard_assigned", 0)) < int(min_assigned):
        return "min_hard_assigned_worlds"

    min_constant = spec.get("min_hard_constant_worlds")
    if min_constant is not None and int(train_mode_counts.get("hard_constant", 0)) < int(min_constant):
        return "min_hard_constant_worlds"

    return None


def _scm_build_heldout_plan_diagnostics(
    train_worlds: List[WorldLike],
    heldout_worlds: List[WorldLike],
) -> Dict[str, Any]:
    train_plans: List[Dict[str, Any]] = []
    for world in train_worlds:
        train_plans.append(
            {
                "worldId": _world_id(world),
                "targets": set(_extract_world_intervention_targets(world)),
                "mode": _extract_world_intervention_mode(world),
            }
        )

    per_world: List[Dict[str, Any]] = []
    novelty_values: List[float] = []
    for world in heldout_worlds:
        targets = set(_extract_world_intervention_targets(world))
        mode = _extract_world_intervention_mode(world)
        nearest_plan: Optional[Dict[str, Any]] = None
        nearest_score: Optional[Tuple[float, int, int, str]] = None
        nearest_distance = 1.0
        nearest_overlap = 0
        for train_plan in train_plans:
            overlap = int(len(targets.intersection(train_plan["targets"])))
            distance = float(_jaccard_distance(targets, train_plan["targets"]))
            tie_score = (
                float(distance),
                int(-overlap),
                0 if str(mode) == str(train_plan["mode"]) else 1,
                str(train_plan["worldId"]),
            )
            if nearest_score is None or tie_score < nearest_score:
                nearest_score = tie_score
                nearest_plan = train_plan
                nearest_distance = float(distance)
                nearest_overlap = int(overlap)
        novelty_values.append(float(nearest_distance))
        per_world.append(
            {
                "worldId": _world_id(world),
                "interventionTargetCount": int(len(targets)),
                "nearestTrainWorldId": None if nearest_plan is None else str(nearest_plan["worldId"]),
                "nearestTrainMode": None if nearest_plan is None else str(nearest_plan["mode"]),
                "nearestTrainTargetOverlap": int(nearest_overlap),
                "noveltyScore": float(nearest_distance),
                "modeMatchNearestTrain": (
                    None if nearest_plan is None else bool(str(mode) == str(nearest_plan["mode"]))
                ),
                "modeMismatchNearestTrain": (
                    None if nearest_plan is None else bool(str(mode) != str(nearest_plan["mode"]))
                ),
            }
        )

    mean_novelty = (
        float(sum(novelty_values) / float(len(novelty_values))) if novelty_values else 0.0
    )
    return {
        "by_world": per_world,
        "summary": {
            "heldout_mean_novelty": float(mean_novelty),
            "heldout_max_novelty": float(max(novelty_values)) if novelty_values else 0.0,
            "heldout_min_novelty": float(min(novelty_values)) if novelty_values else 0.0,
        },
    }


def _scm_heldout_calibration_failure_reason(
    coverage_diag: Dict[str, Any],
    heldout_plan_diag: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[str]:
    if str(spec.get("heldout_mode", "iid_current")) != "verify_balanced":
        return None

    plan_summary = heldout_plan_diag.get("summary") or {}
    novelty_min = spec.get("heldout_target_novelty_min")
    if novelty_min is not None and float(plan_summary.get("heldout_min_novelty", 0.0)) + 1e-12 < float(
        novelty_min
    ):
        return "heldout_target_novelty_min"

    novelty_max = spec.get("heldout_target_novelty_max")
    if novelty_max is not None and float(plan_summary.get("heldout_max_novelty", 0.0)) - 1e-12 > float(
        novelty_max
    ):
        return "heldout_target_novelty_max"

    coverage_summary = coverage_diag.get("summary") or {}
    heldout_min_scored_worlds = spec.get("heldout_min_scored_worlds_per_endogenous")
    if heldout_min_scored_worlds is not None and int(
        coverage_summary.get("heldout_min_scored_worlds_any_endogenous", 0)
    ) < int(heldout_min_scored_worlds):
        return "heldout_min_scored_worlds_per_endogenous"

    heldout_min_scored_cells = spec.get("heldout_min_scored_cells_per_endogenous")
    if heldout_min_scored_cells is not None and int(
        coverage_summary.get("heldout_min_scored_cells_any_endogenous", 0)
    ) < int(heldout_min_scored_cells):
        return "heldout_min_scored_cells_per_endogenous"

    return None


def _mechanism_matches_rows(
    node: Optional[MechanismNode],
    rows: List[Dict[str, Any]],
    target_var: str,
) -> Optional[bool]:
    if node is None:
        return None
    rows_eval = [row for row in rows if target_var in row]
    if not rows_eval:
        return None
    try:
        for row in rows_eval:
            env = {str(key): int(value) for key, value in row.items()}
            if int(evaluate_parsed_mechanism(node, env)) != int(row[target_var]):
                return False
    except (MechanismEvalError, TypeError, ValueError, KeyError):
        return False
    return True


def _scm_build_local_shortcut_diagnostics(
    train_worlds: List[WorldLike],
    heldout_worlds: List[WorldLike],
    *,
    topological_order: List[str],
    endogenous_vars: List[str],
    mechanism_stats_by_var: Dict[str, Dict[str, Any]],
    allowed_ops: List[str],
    ast_cap_floor: int,
    ast_cap_max: Optional[int],
    allow_constants: bool,
    max_predecessors_per_target: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    diagnostics: Dict[str, Dict[str, Any]] = {
        str(var): {
            "smallest_train_fitting_alt_ast": None,
            "train_alt_count_under_cap": 0,
            "train_alt_fits_heldout_scored_cells": None,
        }
        for var in endogenous_vars
    }
    if ast_cap_max is None:
        return diagnostics

    payload_cache: Dict[int, Tuple[Set[str], List[Dict[str, Any]]]] = {}
    train_rows_by_var: Dict[str, List[Dict[str, Any]]] = {str(v): [] for v in endogenous_vars}
    heldout_rows_by_var: Dict[str, List[Dict[str, Any]]] = {str(v): [] for v in endogenous_vars}
    for world in train_worlds:
        intervened, rows = _scm_world_rows_and_intervened(world, payload_cache=payload_cache)
        _scm_extend_rows_by_var_cache(
            rows_by_var_cache=train_rows_by_var,
            vars_to_track=[str(v) for v in endogenous_vars],
            intervened=intervened,
            rows=rows,
        )
    for world in heldout_worlds:
        intervened, rows = _scm_world_rows_and_intervened(world, payload_cache=payload_cache)
        _scm_extend_rows_by_var_cache(
            rows_by_var_cache=heldout_rows_by_var,
            vars_to_track=[str(v) for v in endogenous_vars],
            intervened=intervened,
            rows=rows,
        )

    witnesses = _scm_collect_shortcut_witnesses(
        train_worlds=[world for world in train_worlds if world is not None],
        topological_order=topological_order,
        endogenous_vars=endogenous_vars,
        mechanism_stats_by_var=mechanism_stats_by_var,
        allowed_ops=allowed_ops,
        ast_cap_floor=ast_cap_floor,
        ast_cap_max=int(ast_cap_max),
        check_vars=endogenous_vars,
        precomputed_rows_by_var=train_rows_by_var,
        allow_constants=allow_constants,
        max_predecessors_per_target=max_predecessors_per_target,
    )
    for var in sorted({str(v) for v in endogenous_vars}, key=_natural_var_key):
        witness = witnesses.get(var)
        if witness is None:
            continue
        diagnostics[var] = {
            "smallest_train_fitting_alt_ast": int((witness or {}).get("ast", 0)),
            "train_alt_count_under_cap": 1,
            "train_alt_fits_heldout_scored_cells": _mechanism_matches_rows(
                witness.get("node"),
                heldout_rows_by_var.get(var, []),
                var,
            ),
        }
    return diagnostics


def _build_scm_mechanism_lookup_tables(
    *,
    topological_order: List[str],
    root_vars: List[str],
    gold_nodes_by_var: Dict[str, MechanismNode],
) -> ScmLookupTables:
    """Compile SCM mechanisms into compact truth-table lookups."""
    tables: ScmLookupTables = {}
    root_set = set(root_vars)
    index_of = {v: i for i, v in enumerate(topological_order)}
    for var in topological_order:
        if var in root_set:
            continue
        node = gold_nodes_by_var.get(var)
        if node is None:
            continue
        deps = sorted(
            mechanism_variables(node),
            key=lambda name: (int(index_of.get(name, 10**9)), str(name)),
        )
        dep_tuple = tuple(str(v) for v in deps)
        table: Dict[Tuple[int, ...], int] = {}
        ones_patterns: List[int] = []
        for bits in itertools.product((0, 1), repeat=len(dep_tuple)):
            env = {dep: int(bit) for dep, bit in zip(dep_tuple, bits)}
            bits_tuple = tuple(int(bit) for bit in bits)
            out = int(evaluate_parsed_mechanism(node, env))
            table[bits_tuple] = int(out)
            if int(out) == 1:
                pattern = 0
                for bit_idx, bit in enumerate(bits_tuple):
                    if int(bit):
                        pattern |= (1 << int(bit_idx))
                ones_patterns.append(int(pattern))
        tables[str(var)] = (
            dep_tuple,
            table,
            tuple(int(p) for p in ones_patterns),
        )
    return tables


def _build_root_masks_from_contexts(
    *,
    units: List[str],
    root_vars: List[str],
    base_root_contexts: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    out: Dict[str, int] = {str(rv): 0 for rv in root_vars}
    for unit_idx, uid in enumerate(units):
        row = base_root_contexts.get(uid, {})
        bit = 1 << int(unit_idx)
        for rv in root_vars:
            if int(row.get(rv, 0)):
                out[str(rv)] = int(out.get(str(rv), 0)) | int(bit)
    return out


def _simulate_rows_bitsets_full_scm(
    *,
    units: List[str],
    output_vars: List[str],
    topological_order: List[str],
    root_vars: List[str],
    base_root_contexts: Dict[str, Dict[str, int]],
    base_root_masks: Optional[Dict[str, int]],
    interventions: Dict[str, int],
    gold_nodes_by_var: Dict[str, MechanismNode],
    intervention_mode: str,
    intervention_targets: Optional[List[str]],
    assigned_values_by_unit: Optional[Dict[str, Dict[str, int]]],
    mechanism_lookup_tables: Optional[ScmLookupTables],
) -> Tuple[Dict[str, int], int]:
    topo_vars = [str(v) for v in topological_order]
    unit_count = int(len(units))
    full_units_mask = (1 << unit_count) - 1 if unit_count > 0 else 0
    root_set = set(str(v) for v in root_vars)
    mode_raw = intervention_mode
    if isinstance(mode_raw, str):
        mode_norm = mode_raw if mode_raw in {"hard_constant", "hard_assigned", "none"} else mode_raw.strip().lower()
    else:
        mode_norm = str(mode_raw).strip().lower()
    mode = mode_norm or "hard_constant"
    assignments_const = {str(k): int(v) for k, v in dict(interventions).items()}
    targets = (
        set(str(v) for v in intervention_targets)
        if intervention_targets is not None
        else set(assignments_const.keys())
    )
    lookup_tables = mechanism_lookup_tables or {}
    assigned_values_by_unit_map = assigned_values_by_unit or {}
    output_vars_list = [str(v) for v in output_vars]

    root_masks = (
        base_root_masks
        if base_root_masks is not None
        else _build_root_masks_from_contexts(
            units=units,
            root_vars=list(root_vars),
            base_root_contexts=base_root_contexts,
        )
    )

    var_plan: List[Tuple[str, Any]] = []
    for var in topo_vars:
        if mode == "hard_assigned" and var in targets:
            var_plan.append(("assigned", str(var)))
            continue
        if var in assignments_const:
            var_plan.append(("constant", int(assignments_const[var])))
            continue
        if var in root_set:
            var_plan.append(("root", str(var)))
            continue
        lookup_entry = lookup_tables.get(var)
        if lookup_entry is not None:
            dep_order = tuple(str(dep) for dep in lookup_entry[0])
            truth_table = lookup_entry[1]
            ones_patterns = (
                tuple(int(p) for p in lookup_entry[2])
                if len(lookup_entry) >= 3
                else _truth_table_ones_patterns_cached(truth_table)
            )
            var_plan.append(
                (
                    "lookup",
                    (
                        dep_order,
                        truth_table,
                        ones_patterns,
                    ),
                )
            )
            continue
        var_plan.append(("node", (str(var), gold_nodes_by_var[var])))

    masks_by_var: Dict[str, int] = {}
    assigned_masks_cache: Dict[str, int] = {}

    def _truth_table_mask(
        dep_order: Tuple[str, ...],
        ones_patterns: Tuple[int, ...],
    ) -> int:
        out = 0
        dep_masks = tuple(
            int(masks_by_var.get(str(dep), 0)) & int(full_units_mask)
            for dep in dep_order
        )
        for pattern in ones_patterns:
            match_mask = int(full_units_mask)
            for dep_idx, dep_mask in enumerate(dep_masks):
                if (int(pattern) >> int(dep_idx)) & 1:
                    match_mask &= int(dep_mask)
                else:
                    match_mask &= int((~int(dep_mask)) & int(full_units_mask))
                if int(match_mask) == 0:
                    break
            out |= int(match_mask)
            if int(out) == int(full_units_mask):
                break
        return int(out) & int(full_units_mask)

    for var_idx, var in enumerate(topo_vars):
        plan_kind, plan_payload = var_plan[var_idx]
        if plan_kind == "assigned":
            assigned_var = str(plan_payload)
            cached = assigned_masks_cache.get(assigned_var)
            if cached is None:
                mask = 0
                for unit_idx, uid in enumerate(units):
                    assigned_row = assigned_values_by_unit_map.get(uid, {})
                    if int(assigned_row.get(assigned_var, 0)):
                        mask |= (1 << int(unit_idx))
                cached = int(mask) & int(full_units_mask)
                assigned_masks_cache[assigned_var] = int(cached)
            masks_by_var[var] = int(cached)
        elif plan_kind == "constant":
            masks_by_var[var] = int(full_units_mask) if int(plan_payload) else 0
        elif plan_kind == "root":
            masks_by_var[var] = int(root_masks.get(str(plan_payload), 0)) & int(full_units_mask)
        elif plan_kind == "lookup":
            dep_order, _truth_table, ones_patterns = plan_payload
            masks_by_var[var] = _truth_table_mask(dep_order, ones_patterns)
        else:
            # Safety fallback; this path should be rare for SCM.
            var_name, node = plan_payload
            mask = 0
            deps = tuple(sorted(mechanism_variables(node), key=_natural_var_key))
            for unit_idx, _uid in enumerate(units):
                env: Dict[str, int] = {}
                for dep in deps:
                    dep_mask = int(masks_by_var.get(str(dep), 0))
                    env[str(dep)] = 1 if ((dep_mask >> int(unit_idx)) & 1) else 0
                if int(evaluate_parsed_mechanism(node, env)):
                    mask |= (1 << int(unit_idx))
            masks_by_var[var_name] = int(mask) & int(full_units_mask)

    out_masks: Dict[str, int] = {}
    for var in output_vars_list:
        out_masks[str(var)] = int(masks_by_var.get(str(var), 0)) & int(full_units_mask)
    return out_masks, int(unit_count)


def _simulate_rows_values_full_scm(
    *,
    units: List[str],
    variables: List[str],
    topological_order: List[str],
    root_vars: List[str],
    base_root_contexts: Dict[str, Dict[str, int]],
    interventions: Dict[str, int],
    gold_nodes_by_var: Dict[str, MechanismNode],
    intervention_mode: str,
    intervention_targets: Optional[List[str]],
    assigned_values_by_unit: Optional[Dict[str, Dict[str, int]]],
    mechanism_lookup_tables: Optional[ScmLookupTables],
) -> List[Dict[str, int]]:
    root_set = set(root_vars)
    mode = str(intervention_mode).strip().lower() or "hard_constant"
    targets = set(intervention_targets or list(interventions.keys()))
    assignments_const = {str(k): int(v) for k, v in dict(interventions).items()}
    lookup_tables = mechanism_lookup_tables or {}
    assigned_values_by_unit_map = assigned_values_by_unit or {}
    topo_vars = [str(v) for v in topological_order]
    topo_index = {var: idx for idx, var in enumerate(topo_vars)}
    variables_list = [str(v) for v in variables]
    variable_output_index: List[Tuple[str, int]] = [
        (v, int(topo_index[v])) for v in variables_list if v in topo_index
    ]
    variable_output_missing: List[str] = [v for v in variables_list if v not in topo_index]

    var_plan: List[Tuple[int, str, Any]] = []
    for var_idx, var in enumerate(topo_vars):
        if mode == "hard_assigned" and var in targets:
            var_plan.append((var_idx, "assigned", str(var)))
            continue
        if var in assignments_const:
            var_plan.append((var_idx, "constant", int(assignments_const[var])))
            continue
        if var in root_set:
            var_plan.append((var_idx, "root", str(var)))
            continue
        lookup_entry = lookup_tables.get(var)
        if lookup_entry is not None:
            dep_order = tuple(str(dep) for dep in lookup_entry[0])
            truth_table = lookup_entry[1]
            try:
                dep_indices = tuple(int(topo_index[dep]) for dep in dep_order)
            except KeyError:
                dep_indices = ()
            if dep_indices or not dep_order:
                var_plan.append((var_idx, "lookup", (dep_indices, truth_table)))
                continue
        var_plan.append((var_idx, "node", (str(var), gold_nodes_by_var[var])))

    row_values_list: List[Dict[str, int]] = []
    for uid in units:
        assignment_values: List[int] = [0 for _ in topo_vars]
        root_values = base_root_contexts.get(uid, {})
        assigned_row = assigned_values_by_unit_map.get(uid, {})
        for var_idx, plan_kind, plan_payload in var_plan:
            if plan_kind == "assigned":
                var_name = str(plan_payload)
                assignment_values[var_idx] = int(assigned_row.get(var_name, 0))
            elif plan_kind == "constant":
                assignment_values[var_idx] = int(plan_payload)
            elif plan_kind == "root":
                var_name = str(plan_payload)
                assignment_values[var_idx] = int(root_values.get(var_name, 0))
            elif plan_kind == "lookup":
                dep_indices, truth_table = plan_payload
                key = tuple(int(assignment_values[dep_idx]) for dep_idx in dep_indices)
                assignment_values[var_idx] = int(truth_table[key])
            else:
                var_name, node = plan_payload
                env = {topo_vars[i]: int(assignment_values[i]) for i in range(var_idx)}
                assignment_values[var_idx] = int(evaluate_parsed_mechanism(node, env))

        row_values = {var: int(assignment_values[idx]) for var, idx in variable_output_index}
        for var in variable_output_missing:
            row_values[var] = 0
        row_values_list.append(row_values)

    return row_values_list


def _make_world_full_scm(
    world_id: str,
    split: str,
    units: List[str],
    variables: List[str],
    topological_order: List[str],
    root_vars: List[str],
    base_root_contexts: Dict[str, Dict[str, int]],
    interventions: Dict[str, int],
    gold_nodes_by_var: Dict[str, MechanismNode],
    intervention_mode: str = "hard_constant",
    intervention_targets: Optional[List[str]] = None,
    assigned_values_by_unit: Optional[Dict[str, Dict[str, int]]] = None,
    env_profile: Optional[Dict[str, float]] = None,
    mechanism_lookup_tables: Optional[ScmLookupTables] = None,
) -> CausalWorldView:
    mode = str(intervention_mode).strip().lower() or "hard_constant"
    targets = set(intervention_targets or list(interventions.keys()))
    targets_sorted = sorted(targets, key=_natural_var_key)
    assignments_const = {str(k): int(v) for k, v in dict(interventions).items()}
    row_values_list = _simulate_rows_values_full_scm(
        units=units,
        variables=variables,
        topological_order=topological_order,
        root_vars=root_vars,
        base_root_contexts=base_root_contexts,
        interventions=interventions,
        gold_nodes_by_var=gold_nodes_by_var,
        intervention_mode=mode,
        intervention_targets=list(targets),
        assigned_values_by_unit=assigned_values_by_unit,
        mechanism_lookup_tables=mechanism_lookup_tables,
    )
    if not targets_sorted and not assignments_const:
        mode_structured = "none"
        targets_assigned_structured: List[str] = []
        targets_constant_structured: Dict[str, int] = {}
        targets_all_structured: List[str] = []
    elif mode == "hard_assigned":
        mode_structured = "hard_assigned"
        targets_assigned_structured = list(targets_sorted)
        targets_constant_structured = dict(assignments_const)
        targets_all_structured = sorted(
            set(targets_assigned_structured).union(targets_constant_structured.keys()),
            key=_natural_var_key,
        )
    else:
        mode_structured = "hard_constant"
        targets_assigned_structured = []
        targets_constant_structured = dict(assignments_const)
        targets_all_structured = sorted(targets_constant_structured.keys(), key=_natural_var_key)

    rows: List[Dict[str, Any]] = [
        {"unitId": str(uid), "values": dict(row_values)}
        for uid, row_values in zip(units, row_values_list)
    ]

    return CausalWorldView(
        worldId=world_id,
        domain=units,
        domainSize=len(units),
        observationMode="panel_full",
        interventions=[
            {
                "type": "hard_do",
                "mode": mode,
                "assignments": dict(interventions),
                "targets": list(targets_sorted),
            }
        ],
        extra={
            "split": split,
            "rows": rows,
            "do": _format_do_mode(mode, interventions, list(targets_sorted)),
            "interventionMode": mode,
            "InterventionMode": mode_structured,
            "InterventionTargetsAssigned": list(targets_assigned_structured),
            "InterventionTargetsConstant": dict(targets_constant_structured),
            "InterventionTargetsAll": list(targets_all_structured),
            "envProfile": dict(env_profile or {}),
        },
    )


def _sample_scm_intervention_plan(
    rng: random.Random,
    *,
    variables: List[str],
    units: List[str],
    mode_probs: Dict[str, float],
    force_var: Optional[str],
    min_targets: int,
    max_targets: int,
    size_probs: Optional[Dict[int, float]],
) -> Dict[str, Any]:
    sampled_ints = _sample_intervention(
        rng,
        variables,
        force_var=force_var,
        min_targets=min_targets,
        max_targets=max_targets,
        size_probs=size_probs,
    )
    targets = sorted(sampled_ints.keys(), key=_natural_var_key)
    mode = _sample_intervention_mode(rng, mode_probs)
    if mode == "hard_assigned":
        ints: Dict[str, int] = {}
        assigned_by_unit = _sample_assigned_values_by_unit(
            rng,
            units=units,
            targets=targets,
            base_bias=float(rng.choice([0.3, 0.5, 0.7])),
        )
    else:
        ints = dict(sampled_ints)
        assigned_by_unit = None
    return {
        "mode": str(mode),
        "assignments": dict(ints),
        "targets": list(targets),
        "assigned_values_by_unit": assigned_by_unit,
    }


def _propose_scm_intervention_plans(
    rng: random.Random,
    *,
    variables: List[str],
    units: List[str],
    seen_interventions: Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]],
    mode_probs: Dict[str, float],
    force_var: Optional[str],
    num_candidates: int,
    min_targets: int,
    max_targets: int,
    size_probs: Optional[Dict[int, float]],
    materialize_assigned_values: bool = True,
) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    seen_local: Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]] = set()
    candidate_vars = [str(v) for v in variables]
    var_count = len(candidate_vars)
    if var_count == 0:
        return proposals

    weighted_sizes: List[int] = []
    weighted_size_weights: List[float] = []
    if size_probs:
        for size, prob in size_probs.items():
            s = int(size)
            p = float(prob)
            if s > 0 and p > 0.0 and s <= var_count:
                weighted_sizes.append(s)
                weighted_size_weights.append(p)

    mode_prob_constant = float((mode_probs or {}).get("hard_constant", 0.0))
    mode_prob_assigned = float((mode_probs or {}).get("hard_assigned", 0.0))
    mode_threshold = 1.0
    if mode_prob_constant > 0.0 and mode_prob_assigned > 0.0:
        mode_threshold = mode_prob_constant / (mode_prob_constant + mode_prob_assigned)

    attempts = 0
    max_attempts = max(50, int(num_candidates) * 16)
    while len(proposals) < int(num_candidates) and attempts < max_attempts:
        attempts += 1
        if weighted_sizes:
            sampled_k = int(rng.choices(weighted_sizes, weights=weighted_size_weights, k=1)[0])
        else:
            sampled_k = int(rng.randint(min_targets, max_targets))
        sampled_k = max(1, min(sampled_k, var_count))

        picked = set(rng.sample(candidate_vars, sampled_k))
        if force_var is not None:
            picked.add(str(force_var))
        targets = [v for v in candidate_vars if v in picked]
        if force_var is not None:
            fv = str(force_var)
            if fv in picked and fv not in targets:
                targets.append(fv)
        key_targets = tuple(sorted(str(t) for t in targets))
        if mode_prob_constant <= 0.0 and mode_prob_assigned <= 0.0:
            mode = "hard_constant"
        elif mode_prob_constant <= 0.0:
            mode = "hard_assigned"
        elif mode_prob_assigned <= 0.0:
            mode = "hard_constant"
        else:
            mode = "hard_constant" if rng.random() < mode_threshold else "hard_assigned"
        if mode == "hard_assigned":
            assignments: Dict[str, int] = {}
            key_assignments: Tuple[Tuple[str, int], ...] = ()
        else:
            assignments = {v: int(rng.randint(0, 1)) for v in targets}
            key_assignments = tuple((t, int(assignments.get(t, 0))) for t in key_targets)
        key = (str(mode), key_assignments, key_targets)
        if key in seen_interventions or key in seen_local:
            continue
        seen_local.add(key)
        if mode == "hard_assigned":
            assigned_bias = float(rng.choice([0.3, 0.5, 0.7]))
            assigned_by_unit = (
                _sample_assigned_values_by_unit(
                    rng,
                    units=units,
                    targets=targets,
                    base_bias=assigned_bias,
                )
                if materialize_assigned_values
                else None
            )
        else:
            assigned_bias = 0.5
            assigned_by_unit = None
        proposals.append(
            {
                "mode": str(mode),
                "assignments": dict(assignments),
                "targets": list(targets),
                "assigned_values_by_unit": assigned_by_unit,
                "assigned_base_bias": float(assigned_bias),
                "_plan_key": key,
                "_plan_target_set": frozenset(key_targets),
            }
        )
    return proposals


def _materialize_assigned_values_for_plan(
    rng: random.Random,
    *,
    plan: Dict[str, Any],
    units: List[str],
) -> Dict[str, Dict[str, int]]:
    existing = plan.get("assigned_values_by_unit")
    if isinstance(existing, dict):
        return existing
    targets = [str(v) for v in (plan.get("targets") or [])]
    base_bias = float(plan.get("assigned_base_bias", rng.choice([0.3, 0.5, 0.7])))
    assigned = _sample_assigned_values_by_unit(
        rng,
        units=units,
        targets=targets,
        base_bias=base_bias,
    )
    plan["assigned_values_by_unit"] = assigned
    return assigned


def _sample_heldout_worlds_full_scm(
    rng: random.Random,
    *,
    heldout_k: int,
    train_worlds: List[CausalWorldView],
    seen_interventions: Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]],
    units: List[str],
    variables: List[str],
    topological_order: List[str],
    root_vars: List[str],
    gold_nodes_by_var: Dict[str, MechanismNode],
    unit_root_thresholds: Dict[str, Dict[str, float]],
    endogenous_vars: List[str],
    intervention_mode_probs: Dict[str, float],
    intervention_size_probs: Optional[Dict[int, float]],
    env_levels: Optional[List[float]],
    mechanism_lookup_tables: Optional[ScmLookupTables],
    spec: Dict[str, Any],
) -> Tuple[List[CausalWorldView], Dict[str, Any], Dict[str, Any], Optional[str]]:
    heldout_mode = str(spec.get("heldout_mode", "iid_current") or "iid_current")
    max_set_attempts = 1 if heldout_mode == "iid_current" else max(12, int(heldout_k) * 10)
    last_reason = "heldout_sampling_incomplete"

    for _attempt in range(max_set_attempts):
        heldout_worlds: List[CausalWorldView] = []
        seen_for_heldout = set(seen_interventions)
        sample_guard = 0
        sample_guard_max = max(40, int(heldout_k) * 40)
        while len(heldout_worlds) < int(heldout_k) and sample_guard < sample_guard_max:
            sample_guard += 1
            plan = _sample_scm_intervention_plan(
                rng,
                variables=variables,
                units=units,
                mode_probs=intervention_mode_probs,
                force_var=None,
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
            )
            key = _intervention_plan_key_cached(plan)
            if key in seen_for_heldout:
                continue
            seen_for_heldout.add(key)
            env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
            world_root_contexts = _materialize_root_contexts(
                units=units,
                root_vars=root_vars,
                unit_thresholds=unit_root_thresholds,
                env_profile=env_profile,
            )
            heldout_worlds.append(
                _make_world_full_scm(
                    world_id=f"heldout_{len(heldout_worlds):02d}",
                    split="heldout",
                    units=units,
                    variables=variables,
                    topological_order=topological_order,
                    root_vars=root_vars,
                    base_root_contexts=world_root_contexts,
                    interventions=dict(plan["assignments"]),
                    gold_nodes_by_var=gold_nodes_by_var,
                    intervention_mode=str(plan["mode"]),
                    intervention_targets=list(plan["targets"]),
                    assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                    env_profile=env_profile,
                    mechanism_lookup_tables=mechanism_lookup_tables,
                )
            )

        if len(heldout_worlds) < int(heldout_k):
            last_reason = "heldout_sampling_incomplete"
            continue

        coverage_diag = _scm_build_scoring_coverage_diagnostics(
            train_worlds=train_worlds,
            heldout_worlds=heldout_worlds,
            endogenous_vars=endogenous_vars,
        )
        plan_diag = _scm_build_heldout_plan_diagnostics(
            train_worlds=train_worlds,
            heldout_worlds=heldout_worlds,
        )
        calibration_failure = _scm_heldout_calibration_failure_reason(
            coverage_diag=coverage_diag,
            heldout_plan_diag=plan_diag,
            spec=spec,
        )
        if calibration_failure is None:
            return heldout_worlds, coverage_diag, plan_diag, None
        last_reason = calibration_failure

    return [], {}, {}, last_reason


def _build_train_worlds_scm_cegis_lite(
    rng: random.Random,
    *,
    units: List[str],
    variables: List[str],
    topological_order: List[str],
    root_vars: List[str],
    endogenous_vars: List[str],
    unit_root_thresholds: Dict[str, Dict[str, float]],
    gold_nodes_by_var: Dict[str, MechanismNode],
    mechanism_stats_by_var: Dict[str, Dict[str, Any]],
    k: int,
    intervention_size_probs: Optional[Dict[int, float]],
    intervention_mode_probs: Dict[str, float],
    env_levels: Optional[List[float]],
    allowed_ops: List[str],
    shortcut_ast_cap_floor: int,
    shortcut_ast_cap_max: Optional[int],
    candidate_world_budget: int,
    max_iters: int,
    restarts: int = 1,
    seed_target_worlds: int = 3,
    allow_constants: bool = True,
    max_predecessors_per_target: Optional[int] = None,
    use_stage3_lite: bool = True,
    stage3_probe_size: int = 3,
    stage3_probe_subsets_per_var: int = 2,
    stage3_world_budget: int = 1,
    stage3_candidate_world_budget: int = 24,
    stage3_assigned_bias: float = 0.8,
    adaptive_candidate_budget: bool = True,
    adaptive_candidate_min_candidates: int = 32,
    early_stop_on_quality_met: bool = True,
    target_survivors_small_max: Optional[int] = None,
    min_survivor_reduction_frac: Optional[float] = None,
    mechanism_lookup_tables: Optional[ScmLookupTables] = None,
) -> Tuple[
    List[CausalWorldView],
    Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]],
    Dict[str, Any],
]:
    endogenous_sorted = sorted(endogenous_vars, key=_natural_var_key)
    if mechanism_lookup_tables is None:
        mechanism_lookup_tables = _build_scm_mechanism_lookup_tables(
            topological_order=topological_order,
            root_vars=root_vars,
            gold_nodes_by_var=gold_nodes_by_var,
        )
    seed_target_worlds_effective = min(int(k), max(2, int(seed_target_worlds)))
    restart_count = max(1, int(restarts))
    stage3_enabled = bool(use_stage3_lite) and int(stage3_world_budget) > 0
    stage3_probe_size_eff = max(1, int(stage3_probe_size))
    stage3_probe_subsets_eff = max(1, int(stage3_probe_subsets_per_var))
    stage3_world_budget_eff = max(0, int(stage3_world_budget))
    stage3_candidate_budget_eff = max(4, int(stage3_candidate_world_budget))
    stage3_assigned_bias_eff = min(max(float(stage3_assigned_bias), 0.0), 1.0)
    adaptive_candidates_enabled = bool(adaptive_candidate_budget)
    adaptive_candidate_min_eff = max(8, int(adaptive_candidate_min_candidates))
    target_survivors_small_max_eff = (
        None
        if target_survivors_small_max is None
        else max(0, int(target_survivors_small_max))
    )
    min_survivor_reduction_frac_eff = (
        None
        if min_survivor_reduction_frac is None
        else min(max(float(min_survivor_reduction_frac), 0.0), 1.0)
    )
    # Reserve capacity for Stage-3-lite so it can actually run.
    stage3_reserved_slots = (
        min(int(stage3_world_budget_eff), max(0, int(k) - int(seed_target_worlds_effective)))
        if stage3_enabled
        else 0
    )
    cegis_target_worlds = max(
        int(seed_target_worlds_effective),
        int(k) - int(stage3_reserved_slots),
    )

    def _witness_ast_map(witnesses: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        return {
            str(var): int((payload or {}).get("ast", 0))
            for var, payload in sorted(witnesses.items(), key=lambda kv: _natural_var_key(kv[0]))
        }

    def _witness_cap_map(witnesses: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        return {
            str(var): int((payload or {}).get("cap", 0))
            for var, payload in sorted(witnesses.items(), key=lambda kv: _natural_var_key(kv[0]))
        }

    def _gap_and_ratio_from_ast_map(ast_map: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, float]]:
        gaps: Dict[str, int] = {}
        ratios: Dict[str, float] = {}
        for var in sorted(ast_map.keys(), key=_natural_var_key):
            gold_ast = int((mechanism_stats_by_var.get(var) or {}).get("astSize", 0))
            if gold_ast <= 0:
                continue
            witness_ast = int(ast_map.get(var, 0))
            gaps[var] = max(0, int(gold_ast) - int(witness_ast))
            ratios[var] = float(witness_ast) / float(gold_ast)
        return gaps, ratios

    def _effective_min_reduction_for_initial(initial_survivors: int) -> Optional[float]:
        if min_survivor_reduction_frac_eff is None:
            return None
        effective = float(min_survivor_reduction_frac_eff)
        if initial_survivors > 0 and target_survivors_small_max_eff is not None:
            max_reachable_frac = max(
                0.0,
                min(
                    1.0,
                    float(initial_survivors - int(target_survivors_small_max_eff))
                    / float(initial_survivors),
                ),
            )
            effective = min(float(effective), float(max_reachable_frac))
        return float(effective)

    def _restart_quality_met(
        train_worlds_len: int,
        diag: Dict[str, Any],
    ) -> bool:
        if int(train_worlds_len) < int(k):
            return False
        initial_survivors = int(diag.get("initial_survivors_small_estimate", 0))
        final_survivors = int(diag.get("survivors_small_estimate", 0))
        reduction_abs = int(
            diag.get(
                "survivor_reduction_abs",
                max(0, int(initial_survivors) - int(final_survivors)),
            )
        )
        reduction_frac = (
            float(reduction_abs) / float(initial_survivors)
            if int(initial_survivors) > 0
            else 1.0
        )

        survivors_ok = (
            True
            if target_survivors_small_max_eff is None
            else int(final_survivors) <= int(target_survivors_small_max_eff)
        )
        effective_min_reduction = _effective_min_reduction_for_initial(initial_survivors)
        reduction_ok = (
            True
            if effective_min_reduction is None
            else (
                int(initial_survivors) == 0
                or float(reduction_frac) + 1e-12 >= float(effective_min_reduction)
            )
        )

        # If no explicit quality constraints were requested, only early-stop on
        # the strongest condition: all shortcut witnesses under cap resolved.
        if target_survivors_small_max_eff is None and effective_min_reduction is None:
            return int(final_survivors) == 0
        return bool(survivors_ok and reduction_ok)

    def _adaptive_budget_for_survivors(
        base_budget: int,
        survivor_count: int,
        *,
        min_candidates: int,
    ) -> int:
        base = max(4, int(base_budget))
        if not adaptive_candidates_enabled:
            return base
        s = max(0, int(survivor_count))
        if s <= 1:
            cap = 40
        elif s <= 2:
            cap = 64
        elif s <= 3:
            cap = 96
        elif s <= 4:
            cap = 128
        else:
            cap = base
        return max(4, min(base, max(int(min_candidates), int(cap))))

    def _select_proposals_for_full_eval(
        proposals: List[Dict[str, Any]],
        *,
        witness_risk_weight: Dict[str, int],
        priority_set: Set[str],
        eval_cap: int,
        random_tail_cap: int,
    ) -> List[Dict[str, Any]]:
        if not proposals:
            return []
        cap = max(1, min(int(eval_cap), len(proposals)))
        if len(proposals) <= cap:
            return list(proposals)

        risk_by_var = {str(v): int(w) for v, w in witness_risk_weight.items()}
        total_risk = int(sum(risk_by_var.values()))
        priority_vars = {str(v) for v in priority_set}
        total_priority = int(len(priority_vars))
        ranked: List[Tuple[int, Tuple[int, int, int, int], Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]]] = []
        for idx, plan in enumerate(proposals):
            target_set = _intervention_plan_target_set(plan)
            touched_risk = int(sum(int(risk_by_var.get(v, 0)) for v in target_set))
            untouched_risk = int(total_risk - touched_risk)
            touched_priority = int(sum(1 for v in target_set if v in priority_vars))
            untouched_priority = int(total_priority - touched_priority)
            mode_pref = 1 if str(plan.get("mode", "")).strip().lower() == "hard_assigned" else 0
            quick_score = (
                int(untouched_risk),
                int(untouched_priority),
                int(mode_pref),
                int(-len(target_set)),
            )
            tie_key = _intervention_plan_key_cached(plan)
            ranked.append((int(idx), quick_score, tie_key))

        ranked.sort(
            key=lambda item: (
                -int(item[1][0]),
                -int(item[1][1]),
                -int(item[1][2]),
                -int(item[1][3]),
                item[2],
            )
        )

        tail = min(max(0, int(random_tail_cap)), max(0, cap - 1), max(0, len(ranked) - 1))
        deterministic = max(1, int(cap) - int(tail))
        selected_indices: List[int] = [int(item[0]) for item in ranked[:deterministic]]
        remaining = [int(item[0]) for item in ranked[deterministic:]]
        if tail > 0 and remaining:
            selected_indices.extend(rng.sample(remaining, min(int(tail), len(remaining))))

        selected_set = set(int(i) for i in selected_indices)
        ordered_indices = [int(item[0]) for item in ranked if int(item[0]) in selected_set]
        return [proposals[i] for i in ordered_indices[:cap]]

    shared_root_context_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Dict[str, int]]] = {}
    shared_root_mask_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, int]] = {}

    def _run_one_restart(restart_idx: int) -> Tuple[
        List[CausalWorldView],
        Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]],
        Dict[str, Any],
        Tuple[int, int, int, int, int, float, float, int],
    ]:
        train_worlds: List[CausalWorldView] = []
        seen_interventions: Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]] = set()
        rows_by_var_cache: Dict[str, List[Dict[str, Any]]] = {v: [] for v in endogenous_vars}
        world_payload_cache: Dict[int, Tuple[Set[str], List[Dict[str, Any]]]] = {}
        full_units_mask = (1 << int(len(units))) - 1 if units else 0

        def _root_contexts_for_env(env_profile: Dict[str, float]) -> Dict[str, Dict[str, int]]:
            key = tuple((rv, float(env_profile.get(rv, 0.5))) for rv in root_vars)
            cached = shared_root_context_cache.get(key)
            if cached is not None:
                return cached
            computed = _materialize_root_contexts(
                units=units,
                root_vars=root_vars,
                unit_thresholds=unit_root_thresholds,
                env_profile=env_profile,
            )
            shared_root_context_cache[key] = computed
            return computed

        def _root_masks_for_env(env_profile: Dict[str, float]) -> Dict[str, int]:
            key = tuple((rv, float(env_profile.get(rv, 0.5))) for rv in root_vars)
            cached = shared_root_mask_cache.get(key)
            if cached is not None:
                return cached
            contexts = _root_contexts_for_env(env_profile)
            masks = _build_root_masks_from_contexts(
                units=units,
                root_vars=root_vars,
                base_root_contexts=contexts,
            )
            shared_root_mask_cache[key] = masks
            return masks

        def _refresh_witnesses_incremental(
            current_witnesses: Dict[str, Dict[str, Any]],
            intervened_targets: Set[str],
        ) -> Dict[str, Dict[str, Any]]:
            if shortcut_ast_cap_max is None or not current_witnesses:
                return dict(current_witnesses)
            intervened_set = {str(v) for v in intervened_targets}
            vars_to_recheck = [
                str(v)
                for v in sorted(current_witnesses.keys(), key=_natural_var_key)
                if str(v) not in intervened_set
            ]
            if not vars_to_recheck:
                return dict(current_witnesses)
            refreshed = _scm_collect_shortcut_witnesses(
                train_worlds=train_worlds,
                topological_order=topological_order,
                endogenous_vars=endogenous_vars,
                mechanism_stats_by_var=mechanism_stats_by_var,
                allowed_ops=allowed_ops,
                ast_cap_floor=shortcut_ast_cap_floor,
                ast_cap_max=int(shortcut_ast_cap_max),
                check_vars=vars_to_recheck,
                precomputed_rows_by_var=rows_by_var_cache,
                allow_constants=allow_constants,
                max_predecessors_per_target=max_predecessors_per_target,
            )
            updated = dict(current_witnesses)
            for var in vars_to_recheck:
                payload = refreshed.get(str(var))
                if payload is None:
                    updated.pop(str(var), None)
                else:
                    updated[str(var)] = payload
            return updated

        baseline_env = {rv: 0.5 for rv in root_vars}
        baseline_root_contexts = _root_contexts_for_env(baseline_env)
        baseline_world = _make_world_full_scm(
            world_id="train_00",
            split="train",
            units=units,
            variables=variables,
            topological_order=topological_order,
            root_vars=root_vars,
            base_root_contexts=baseline_root_contexts,
            interventions={},
            gold_nodes_by_var=gold_nodes_by_var,
            intervention_mode="hard_constant",
            intervention_targets=[],
            assigned_values_by_unit=None,
            env_profile=baseline_env,
            mechanism_lookup_tables=mechanism_lookup_tables,
        )
        train_worlds.append(baseline_world)
        seen_interventions.add(_intervention_plan_key(mode="hard_constant", assignments={}, targets=[]))
        baseline_intervened, baseline_rows = _scm_world_rows_and_intervened(
            baseline_world,
            payload_cache=world_payload_cache,
        )
        _scm_extend_rows_by_var_cache(
            rows_by_var_cache=rows_by_var_cache,
            vars_to_track=endogenous_vars,
            intervened=baseline_intervened,
            rows=baseline_rows,
        )

        # Add a small seed set of interventions before shortcut-guided planning.
        seed_guard = 0
        seed_max_guard = max(20, int(candidate_world_budget) * 3)
        while len(train_worlds) < seed_target_worlds_effective and seed_guard < seed_max_guard:
            seed_guard += 1
            force_var = rng.choice(endogenous_vars) if endogenous_vars else None
            proposals = _propose_scm_intervention_plans(
                rng,
                variables=variables,
                units=units,
                seen_interventions=seen_interventions,
                mode_probs=intervention_mode_probs,
                force_var=force_var,
                num_candidates=max(2, min(8, int(candidate_world_budget) // 2)),
                min_targets=1,
                max_targets=2,
                size_probs=intervention_size_probs,
            )
            if not proposals:
                continue
            proposals.sort(
                key=lambda p: (
                    0 if str(p["mode"]) == "hard_assigned" else 1,
                    _intervention_plan_key_cached(p),
                )
            )
            plan = proposals[0]
            env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
            world_root_contexts = _root_contexts_for_env(env_profile)
            world = _make_world_full_scm(
                world_id=f"train_{len(train_worlds):02d}",
                split="train",
                units=units,
                variables=variables,
                topological_order=topological_order,
                root_vars=root_vars,
                base_root_contexts=world_root_contexts,
                interventions=dict(plan["assignments"]),
                gold_nodes_by_var=gold_nodes_by_var,
                intervention_mode=str(plan["mode"]),
                intervention_targets=list(plan["targets"]),
                assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                env_profile=env_profile,
                mechanism_lookup_tables=mechanism_lookup_tables,
            )
            key = _intervention_plan_key_cached(plan)
            seen_interventions.add(key)
            train_worlds.append(world)
            world_intervened, world_rows = _scm_world_rows_and_intervened(
                world,
                payload_cache=world_payload_cache,
            )
            _scm_extend_rows_by_var_cache(
                rows_by_var_cache=rows_by_var_cache,
                vars_to_track=endogenous_vars,
                intervened=world_intervened,
                rows=world_rows,
            )

        stage3_probe_subsets = (
            _scm_build_stage3_probe_subsets(
                topological_order=topological_order,
                endogenous_vars=endogenous_vars,
                max_predecessors_per_target=max_predecessors_per_target,
                probe_size=stage3_probe_size_eff,
                subsets_per_var=stage3_probe_subsets_eff,
            )
            if stage3_enabled
            else {}
        )
        stage3_probe_specs = _scm_compile_stage3_probe_specs(stage3_probe_subsets)
        stage3_coverage = _scm_compute_stage3_probe_coverage(
            rows_by_var_cache=rows_by_var_cache,
            probe_subsets_by_var=stage3_probe_subsets,
        )
        stage3_initial_summary = _scm_stage3_coverage_summary(
            probe_subsets_by_var=stage3_probe_subsets,
            coverage=stage3_coverage,
        )
        stage3_total_new_patterns = 0
        stage3_candidate_plan_evals = 0
        stage3_worlds_added = 0
        stage3_kills_by_var: Dict[str, int] = {v: 0 for v in endogenous_sorted}
        stage3_coverage_history: List[float] = [
            float(stage3_initial_summary.get("coverage_ratio", 1.0))
        ]

        shortcuts_killed_by_var: Dict[str, int] = {v: 0 for v in endogenous_sorted}
        candidate_plan_evals = 0
        cegis_iters = 0
        kills_per_iter: List[int] = []

        witnesses = (
            _scm_collect_shortcut_witnesses(
                train_worlds=train_worlds,
                topological_order=topological_order,
                endogenous_vars=endogenous_vars,
                mechanism_stats_by_var=mechanism_stats_by_var,
                allowed_ops=allowed_ops,
                ast_cap_floor=shortcut_ast_cap_floor,
                ast_cap_max=int(shortcut_ast_cap_max),
                check_vars=endogenous_vars,
                precomputed_rows_by_var=rows_by_var_cache,
                allow_constants=allow_constants,
                max_predecessors_per_target=max_predecessors_per_target,
            )
            if shortcut_ast_cap_max is not None
            else {}
        )
        initial_witnesses = dict(witnesses)
        survivor_count_history: List[int] = [int(len(witnesses))]

        index_of = {v: i for i, v in enumerate(topological_order)}
        while (
            len(train_worlds) < int(cegis_target_worlds)
            and int(cegis_iters) < int(max_iters)
            and bool(witnesses)
        ):
            cegis_iters += 1
            min_ast = min(int((w or {}).get("ast", 10**9)) for w in witnesses.values())
            priority_vars = [
                v
                for v, w in witnesses.items()
                if int((w or {}).get("ast", 10**9)) == int(min_ast)
            ]
            priority_set = set(priority_vars)
            focus_var = sorted(priority_vars, key=_natural_var_key)[0]
            focus_idx = int(index_of.get(focus_var, -1))
            focus_predecessors = [
                v for v in topological_order[:focus_idx] if v not in set(root_vars)
            ]
            force_var = rng.choice(focus_predecessors) if focus_predecessors else None

            witness_risk_weight: Dict[str, int] = {}
            for var, witness in witnesses.items():
                witness_ast = int((witness or {}).get("ast", 10**9))
                gold_ast = int((mechanism_stats_by_var.get(var) or {}).get("astSize", witness_ast))
                ast_gap = max(0, int(gold_ast) - int(witness_ast))
                severity = max(1, int(ast_gap) + 1)
                if int(witness_ast) <= int(shortcut_ast_cap_floor):
                    severity += 2
                witness_risk_weight[str(var)] = int(severity)

            witness_truth_by_var: Dict[str, Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = {}
            witness_truth_cache_by_node_id: Dict[int, Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = {}
            for var, witness in witnesses.items():
                node = witness.get("node")
                if node is None:
                    continue
                node_key = id(node)
                truth_payload = witness_truth_cache_by_node_id.get(node_key)
                if truth_payload is None:
                    truth_payload = _compile_mechanism_truth_table(node)
                    witness_truth_cache_by_node_id[node_key] = truth_payload
                witness_truth_by_var[str(var)] = truth_payload

            dynamic_candidate_budget = _adaptive_budget_for_survivors(
                base_budget=int(candidate_world_budget),
                survivor_count=len(witnesses),
                min_candidates=adaptive_candidate_min_eff,
            )
            proposals = _propose_scm_intervention_plans(
                rng,
                variables=variables,
                units=units,
                seen_interventions=seen_interventions,
                mode_probs=intervention_mode_probs,
                force_var=force_var,
                num_candidates=max(5, int(dynamic_candidate_budget)),
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
                materialize_assigned_values=False,
            )
            if not proposals:
                break
            survivors_n = max(0, int(len(witnesses)))
            if survivors_n <= 1:
                proposal_eval_cap_max = 40
                proposal_eval_cap = max(12, int(round(float(len(proposals)) * 0.45)))
            elif survivors_n <= 3:
                proposal_eval_cap_max = 56
                proposal_eval_cap = max(20, int(round(float(len(proposals)) * 0.55)))
            else:
                proposal_eval_cap_max = 72
                proposal_eval_cap = max(28, int(round(float(len(proposals)) * 0.65)))
            proposal_eval_cap = min(len(proposals), int(min(proposal_eval_cap_max, proposal_eval_cap)))
            proposal_tail_cap = min(8, max(2, int(proposal_eval_cap) // 8))
            proposals_for_eval = _select_proposals_for_full_eval(
                proposals,
                witness_risk_weight=witness_risk_weight,
                priority_set=priority_set,
                eval_cap=int(proposal_eval_cap),
                random_tail_cap=int(proposal_tail_cap),
            )
            eval_vars: Set[str] = set()
            for witness_var, truth_payload in witness_truth_by_var.items():
                eval_vars.add(str(witness_var))
                dep_order, _truth_table = truth_payload
                for dep in dep_order:
                    eval_vars.add(str(dep))
            for specs in stage3_probe_specs.values():
                for _key, subset, _support_mask in specs:
                    for dep in subset:
                        eval_vars.add(str(dep))
            eval_vars_sorted = sorted(eval_vars, key=_natural_var_key)

            best_any: Optional[Dict[str, Any]] = None
            best_positive: Optional[Dict[str, Any]] = None
            for plan in proposals_for_eval:
                candidate_plan_evals += 1
                if str(plan.get("mode", "")).strip().lower() == "hard_assigned":
                    _materialize_assigned_values_for_plan(
                        rng,
                        plan=plan,
                        units=units,
                    )
                env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
                world_root_contexts = _root_contexts_for_env(env_profile)
                world_root_masks = _root_masks_for_env(env_profile)
                row_bitsets, _unit_count = _simulate_rows_bitsets_full_scm(
                    units=units,
                    output_vars=eval_vars_sorted,
                    topological_order=topological_order,
                    root_vars=root_vars,
                    interventions=dict(plan["assignments"]),
                    base_root_contexts=world_root_contexts,
                    base_root_masks=world_root_masks,
                    gold_nodes_by_var=gold_nodes_by_var,
                    intervention_mode=str(plan["mode"]),
                    intervention_targets=list(plan["targets"]),
                    assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                    mechanism_lookup_tables=mechanism_lookup_tables,
                )
                targets = _intervention_plan_target_set(plan)
                killed_vars: List[str] = []
                total_mismatch = 0
                priority_mismatch = 0
                for var, witness in witnesses.items():
                    if var in targets:
                        continue
                    truth_payload = witness_truth_by_var.get(str(var))
                    if truth_payload is None:
                        node = witness.get("node")
                        if node is None:
                            continue
                        truth_payload = _compile_mechanism_truth_table(node)
                        witness_truth_by_var[str(var)] = truth_payload
                    dep_order, truth_table = truth_payload
                    mismatch = _mechanism_mismatch_count_from_truth_table_bitsets(
                        dep_order=dep_order,
                        truth_table=truth_table,
                        var_bitsets=row_bitsets,
                        target_var=var,
                        full_units_mask=int(full_units_mask),
                    )

                    total_mismatch += int(mismatch)
                    if var in priority_set:
                        priority_mismatch += int(mismatch)
                    if int(mismatch) > 0:
                        killed_vars.append(var)

                (
                    coverage_gain,
                    coverage_new_by_key,
                    coverage_gain_by_var,
                ) = (
                    _scm_stage3_coverage_gain_for_bitsets(
                        var_bitsets=row_bitsets,
                        full_units_mask=int(full_units_mask),
                        targets=targets,
                        probe_specs_by_var=stage3_probe_specs,
                        coverage=stage3_coverage,
                    )
                    if stage3_probe_specs
                    else (0, {}, {})
                )

                killed_priority = sum(1 for v in killed_vars if v in priority_set)
                killed_risk = int(sum(int(witness_risk_weight.get(v, 1)) for v in killed_vars))
                remaining_risk = int(
                    sum(int(witness_risk_weight.get(v, 1)) for v in witnesses.keys() if v not in set(killed_vars))
                )
                mode_pref = 1 if str(plan["mode"]) == "hard_assigned" else 0
                score = (
                    int(killed_risk),
                    int(killed_priority),
                    int(len(killed_vars)),
                    int(priority_mismatch),
                    int(total_mismatch),
                    int(-remaining_risk),
                    int(coverage_gain),
                    int(len(coverage_gain_by_var)),
                    int(mode_pref),
                )
                tie_key = _intervention_plan_key_cached(plan)
                candidate = {
                    "plan": plan,
                    "env_profile": dict(env_profile),
                    "base_root_contexts": world_root_contexts,
                    "killed_vars": list(sorted(set(killed_vars), key=_natural_var_key)),
                    "score": score,
                    "tie_key": tie_key,
                    "coverage_gain": int(coverage_gain),
                    "coverage_new_by_key": dict(coverage_new_by_key),
                    "coverage_gain_by_var": dict(coverage_gain_by_var),
                }
                if (
                    best_any is None
                    or candidate["score"] > best_any["score"]
                    or (
                        candidate["score"] == best_any["score"]
                        and candidate["tie_key"] < best_any["tie_key"]
                    )
                ):
                    best_any = candidate
                if killed_vars and (
                    best_positive is None
                    or candidate["score"] > best_positive["score"]
                    or (
                        candidate["score"] == best_positive["score"]
                        and candidate["tie_key"] < best_positive["tie_key"]
                    )
                ):
                    best_positive = candidate

            selected = best_positive if best_positive is not None else best_any
            if selected is None:
                break
            plan = selected["plan"]
            world = _make_world_full_scm(
                world_id=f"train_{len(train_worlds):02d}",
                split="train",
                units=units,
                variables=variables,
                topological_order=topological_order,
                root_vars=root_vars,
                base_root_contexts=selected["base_root_contexts"],
                interventions=dict(plan["assignments"]),
                gold_nodes_by_var=gold_nodes_by_var,
                intervention_mode=str(plan["mode"]),
                intervention_targets=list(plan["targets"]),
                assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                env_profile=selected.get("env_profile"),
                mechanism_lookup_tables=mechanism_lookup_tables,
            )
            train_worlds.append(world)
            key = _intervention_plan_key_cached(plan)
            seen_interventions.add(key)
            world_intervened, world_rows = _scm_world_rows_and_intervened(
                world,
                payload_cache=world_payload_cache,
            )
            _scm_extend_rows_by_var_cache(
                rows_by_var_cache=rows_by_var_cache,
                vars_to_track=endogenous_vars,
                intervened=world_intervened,
                rows=world_rows,
            )
            for var in selected["killed_vars"]:
                shortcuts_killed_by_var[var] = int(shortcuts_killed_by_var.get(var, 0)) + 1
            kills_per_iter.append(int(len(selected["killed_vars"])))
            _scm_apply_stage3_coverage_gain(
                coverage=stage3_coverage,
                new_patterns_by_key=selected.get("coverage_new_by_key") or {},
            )
            if int(selected.get("coverage_gain", 0)) > 0:
                stage3_total_new_patterns += int(selected.get("coverage_gain", 0))
                stage3_coverage_history.append(
                    float(
                        _scm_stage3_coverage_summary(
                            probe_subsets_by_var=stage3_probe_subsets,
                            coverage=stage3_coverage,
                        ).get("coverage_ratio", 1.0)
                    )
                )

            selected_targets = _intervention_plan_target_set(plan)
            witnesses = _refresh_witnesses_incremental(
                current_witnesses=witnesses,
                intervened_targets=selected_targets,
            )
            survivor_count_history.append(int(len(witnesses)))

        # Stage-3-lite: add a small number of coverage-focused worlds.
        stage3_mode_probs = {
            "hard_constant": max(0.0, 1.0 - float(stage3_assigned_bias_eff)),
            "hard_assigned": max(0.0, float(stage3_assigned_bias_eff)),
        }
        stage3_slots_remaining = min(
            int(stage3_world_budget_eff),
            max(0, int(k) - len(train_worlds)),
        )
        if not witnesses and float(stage3_coverage_history[-1] if stage3_coverage_history else 1.0) >= 1.0 - 1e-12:
            stage3_slots_remaining = 0
        stage3_guard = 0
        stage3_guard_max = max(20, int(stage3_slots_remaining) * max(4, int(stage3_candidate_budget_eff)))
        stage3_cached_witnesses: Dict[str, Dict[str, Any]] = dict(witnesses)
        while (
            stage3_enabled
            and stage3_slots_remaining > 0
            and len(train_worlds) < int(k)
            and stage3_guard < stage3_guard_max
        ):
            stage3_guard += 1
            stage3_current_witnesses = dict(stage3_cached_witnesses)
            stage3_current_coverage_ratio = float(
                stage3_coverage_history[-1] if stage3_coverage_history else 1.0
            )
            if (not stage3_current_witnesses) and (
                (not stage3_probe_subsets) or stage3_current_coverage_ratio >= 1.0 - 1e-12
            ):
                break
            stage3_witness_risk_weight: Dict[str, int] = {}
            for var, witness in stage3_current_witnesses.items():
                witness_ast = int((witness or {}).get("ast", 10**9))
                gold_ast = int((mechanism_stats_by_var.get(var) or {}).get("astSize", witness_ast))
                stage3_witness_risk_weight[str(var)] = max(1, int(max(0, gold_ast - witness_ast)) + 1)

            dynamic_stage3_budget = _adaptive_budget_for_survivors(
                base_budget=int(stage3_candidate_budget_eff),
                survivor_count=len(stage3_current_witnesses),
                min_candidates=max(8, int(min(stage3_candidate_budget_eff, adaptive_candidate_min_eff // 2))),
            )
            proposals = _propose_scm_intervention_plans(
                rng,
                variables=variables,
                units=units,
                seen_interventions=seen_interventions,
                mode_probs=stage3_mode_probs,
                force_var=None,
                num_candidates=max(4, int(dynamic_stage3_budget)),
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
                materialize_assigned_values=False,
            )
            if not proposals:
                break
            stage3_witness_truth_by_var: Dict[str, Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = {}
            stage3_truth_cache_by_node_id: Dict[int, Tuple[Tuple[str, ...], Dict[Tuple[int, ...], int]]] = {}
            for var, witness in stage3_current_witnesses.items():
                node = witness.get("node")
                if node is None:
                    continue
                node_key = id(node)
                truth_payload = stage3_truth_cache_by_node_id.get(node_key)
                if truth_payload is None:
                    truth_payload = _compile_mechanism_truth_table(node)
                    stage3_truth_cache_by_node_id[node_key] = truth_payload
                stage3_witness_truth_by_var[str(var)] = truth_payload
            stage3_survivors_n = max(0, int(len(stage3_current_witnesses)))
            if stage3_survivors_n <= 1:
                stage3_eval_cap_max = 16
                stage3_eval_cap = max(8, int(round(float(len(proposals)) * 0.50)))
            elif stage3_survivors_n <= 3:
                stage3_eval_cap_max = 24
                stage3_eval_cap = max(12, int(round(float(len(proposals)) * 0.60)))
            else:
                stage3_eval_cap_max = 32
                stage3_eval_cap = max(16, int(round(float(len(proposals)) * 0.70)))
            stage3_eval_cap = min(len(proposals), int(min(stage3_eval_cap_max, stage3_eval_cap)))
            stage3_tail_cap = min(4, max(1, int(stage3_eval_cap) // 8))
            proposals_for_eval = _select_proposals_for_full_eval(
                proposals,
                witness_risk_weight=stage3_witness_risk_weight,
                priority_set=set(),
                eval_cap=int(stage3_eval_cap),
                random_tail_cap=int(stage3_tail_cap),
            )
            stage3_eval_vars: Set[str] = set()
            for witness_var, truth_payload in stage3_witness_truth_by_var.items():
                stage3_eval_vars.add(str(witness_var))
                dep_order, _truth_table = truth_payload
                for dep in dep_order:
                    stage3_eval_vars.add(str(dep))
            for specs in stage3_probe_specs.values():
                for _key, subset, _support_mask in specs:
                    for dep in subset:
                        stage3_eval_vars.add(str(dep))
            stage3_eval_vars_sorted = sorted(stage3_eval_vars, key=_natural_var_key)

            best_stage3: Optional[Dict[str, Any]] = None
            for plan in proposals_for_eval:
                stage3_candidate_plan_evals += 1
                if str(plan.get("mode", "")).strip().lower() == "hard_assigned":
                    _materialize_assigned_values_for_plan(
                        rng,
                        plan=plan,
                        units=units,
                    )
                env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
                world_root_contexts = _root_contexts_for_env(env_profile)
                world_root_masks = _root_masks_for_env(env_profile)
                row_bitsets, _unit_count = _simulate_rows_bitsets_full_scm(
                    units=units,
                    output_vars=stage3_eval_vars_sorted,
                    topological_order=topological_order,
                    root_vars=root_vars,
                    interventions=dict(plan["assignments"]),
                    base_root_contexts=world_root_contexts,
                    base_root_masks=world_root_masks,
                    gold_nodes_by_var=gold_nodes_by_var,
                    intervention_mode=str(plan["mode"]),
                    intervention_targets=list(plan["targets"]),
                    assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                    mechanism_lookup_tables=mechanism_lookup_tables,
                )
                targets = _intervention_plan_target_set(plan)

                coverage_gain, coverage_new_by_key, coverage_gain_by_var = (
                    _scm_stage3_coverage_gain_for_bitsets(
                        var_bitsets=row_bitsets,
                        full_units_mask=int(full_units_mask),
                        targets=targets,
                        probe_specs_by_var=stage3_probe_specs,
                        coverage=stage3_coverage,
                    )
                    if stage3_probe_specs
                    else (0, {}, {})
                )

                killed_vars: List[str] = []
                for var, witness in stage3_current_witnesses.items():
                    if var in targets:
                        continue
                    truth_payload = stage3_witness_truth_by_var.get(str(var))
                    if truth_payload is None:
                        node = witness.get("node")
                        if node is None:
                            continue
                        truth_payload = _compile_mechanism_truth_table(node)
                        stage3_witness_truth_by_var[str(var)] = truth_payload
                    dep_order, truth_table = truth_payload
                    mismatch = _mechanism_mismatch_count_from_truth_table_bitsets(
                        dep_order=dep_order,
                        truth_table=truth_table,
                        var_bitsets=row_bitsets,
                        target_var=var,
                        full_units_mask=int(full_units_mask),
                    )
                    if int(mismatch) > 0:
                        killed_vars.append(var)
                killed_risk = int(
                    sum(int(stage3_witness_risk_weight.get(v, 1)) for v in killed_vars)
                )
                mode_pref = 1 if str(plan["mode"]) == "hard_assigned" else 0
                score = (
                    int(killed_risk),
                    int(coverage_gain),
                    int(len(killed_vars)),
                    int(len(coverage_gain_by_var)),
                    int(mode_pref),
                    int(-len(targets)),
                )
                tie_key = _intervention_plan_key_cached(plan)
                candidate = {
                    "plan": plan,
                    "env_profile": dict(env_profile),
                    "base_root_contexts": world_root_contexts,
                    "score": score,
                    "tie_key": tie_key,
                    "coverage_gain": int(coverage_gain),
                    "coverage_new_by_key": dict(coverage_new_by_key),
                    "coverage_gain_by_var": dict(coverage_gain_by_var),
                    "killed_vars": list(sorted(set(killed_vars), key=_natural_var_key)),
                    "killed_risk": int(killed_risk),
                }
                if (
                    best_stage3 is None
                    or candidate["score"] > best_stage3["score"]
                    or (
                        candidate["score"] == best_stage3["score"]
                        and candidate["tie_key"] < best_stage3["tie_key"]
                    )
                ):
                    best_stage3 = candidate

            if best_stage3 is None:
                break
            if (
                int(best_stage3.get("coverage_gain", 0)) <= 0
                and int(best_stage3.get("killed_risk", 0)) <= 0
            ):
                break

            plan = best_stage3["plan"]
            world = _make_world_full_scm(
                world_id=f"train_{len(train_worlds):02d}",
                split="train",
                units=units,
                variables=variables,
                topological_order=topological_order,
                root_vars=root_vars,
                base_root_contexts=best_stage3["base_root_contexts"],
                interventions=dict(plan["assignments"]),
                gold_nodes_by_var=gold_nodes_by_var,
                intervention_mode=str(plan["mode"]),
                intervention_targets=list(plan["targets"]),
                assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                env_profile=best_stage3.get("env_profile"),
                mechanism_lookup_tables=mechanism_lookup_tables,
            )
            train_worlds.append(world)
            key = _intervention_plan_key_cached(plan)
            seen_interventions.add(key)
            world_intervened, world_rows = _scm_world_rows_and_intervened(
                world,
                payload_cache=world_payload_cache,
            )
            _scm_extend_rows_by_var_cache(
                rows_by_var_cache=rows_by_var_cache,
                vars_to_track=endogenous_vars,
                intervened=world_intervened,
                rows=world_rows,
            )
            _scm_apply_stage3_coverage_gain(
                coverage=stage3_coverage,
                new_patterns_by_key=best_stage3.get("coverage_new_by_key") or {},
            )
            stage3_total_new_patterns += int(best_stage3.get("coverage_gain", 0))
            for var in best_stage3.get("killed_vars", []):
                shortcuts_killed_by_var[var] = int(shortcuts_killed_by_var.get(var, 0)) + 1
                stage3_kills_by_var[var] = int(stage3_kills_by_var.get(var, 0)) + 1
            stage3_worlds_added += 1
            stage3_slots_remaining -= 1
            stage3_selected_targets = _intervention_plan_target_set(plan)
            stage3_cached_witnesses = _refresh_witnesses_incremental(
                current_witnesses=stage3_cached_witnesses,
                intervened_targets=stage3_selected_targets,
            )
            stage3_coverage_history.append(
                float(
                    _scm_stage3_coverage_summary(
                        probe_subsets_by_var=stage3_probe_subsets,
                        coverage=stage3_coverage,
                    ).get("coverage_ratio", 1.0)
                )
            )

        # Fill remaining worlds if shortcuts were resolved early.
        fill_guard = 0
        fill_max_guard = max(60, int(candidate_world_budget) * max(2, int(k)))
        while len(train_worlds) < int(k) and fill_guard < fill_max_guard:
            fill_guard += 1
            proposals = _propose_scm_intervention_plans(
                rng,
                variables=variables,
                units=units,
                seen_interventions=seen_interventions,
                mode_probs=intervention_mode_probs,
                force_var=None,
                num_candidates=max(2, min(8, int(candidate_world_budget) // 2)),
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
            )
            if not proposals:
                continue
            proposals.sort(
                key=lambda p: (
                    0 if str(p["mode"]) == "hard_assigned" else 1,
                    _intervention_plan_key_cached(p),
                )
            )
            plan = proposals[0]
            env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
            world_root_contexts = _root_contexts_for_env(env_profile)
            world = _make_world_full_scm(
                world_id=f"train_{len(train_worlds):02d}",
                split="train",
                units=units,
                variables=variables,
                topological_order=topological_order,
                root_vars=root_vars,
                base_root_contexts=world_root_contexts,
                interventions=dict(plan["assignments"]),
                gold_nodes_by_var=gold_nodes_by_var,
                intervention_mode=str(plan["mode"]),
                intervention_targets=list(plan["targets"]),
                assigned_values_by_unit=plan.get("assigned_values_by_unit"),
                env_profile=env_profile,
                mechanism_lookup_tables=mechanism_lookup_tables,
            )
            key = _intervention_plan_key_cached(plan)
            seen_interventions.add(key)
            train_worlds.append(world)
            world_intervened, world_rows = _scm_world_rows_and_intervened(
                world,
                payload_cache=world_payload_cache,
            )
            _scm_extend_rows_by_var_cache(
                rows_by_var_cache=rows_by_var_cache,
                vars_to_track=endogenous_vars,
                intervened=world_intervened,
                rows=world_rows,
            )

        final_witnesses = (
            _scm_collect_shortcut_witnesses(
                train_worlds=train_worlds,
                topological_order=topological_order,
                endogenous_vars=endogenous_vars,
                mechanism_stats_by_var=mechanism_stats_by_var,
                allowed_ops=allowed_ops,
                ast_cap_floor=shortcut_ast_cap_floor,
                ast_cap_max=int(shortcut_ast_cap_max),
                check_vars=endogenous_vars,
                precomputed_rows_by_var=rows_by_var_cache,
                allow_constants=allow_constants,
                max_predecessors_per_target=max_predecessors_per_target,
            )
            if shortcut_ast_cap_max is not None
            else {}
        )
        if not survivor_count_history or survivor_count_history[-1] != int(len(final_witnesses)):
            survivor_count_history.append(int(len(final_witnesses)))

        initial_ast_by_var = _witness_ast_map(initial_witnesses)
        final_ast_by_var = _witness_ast_map(final_witnesses)
        initial_cap_by_var = _witness_cap_map(initial_witnesses)
        final_cap_by_var = _witness_cap_map(final_witnesses)
        initial_gap_by_var, initial_ratio_by_var = _gap_and_ratio_from_ast_map(initial_ast_by_var)
        final_gap_by_var, final_ratio_by_var = _gap_and_ratio_from_ast_map(final_ast_by_var)

        initial_survivors = int(len(initial_witnesses))
        final_survivors = int(len(final_witnesses))
        reduction_abs = int(max(0, initial_survivors - final_survivors))
        reduction_ratio = (
            float(reduction_abs) / float(initial_survivors)
            if int(initial_survivors) > 0
            else 1.0
        )
        total_kills = int(sum(int(shortcuts_killed_by_var.get(v, 0)) for v in endogenous_sorted))
        vars_with_kills = [v for v in endogenous_sorted if int(shortcuts_killed_by_var.get(v, 0)) > 0]
        kill_coverage_ratio = (
            float(len(vars_with_kills)) / float(max(1, len(endogenous_sorted)))
            if endogenous_sorted
            else 0.0
        )
        final_gap_sum = int(sum(int(final_gap_by_var.get(v, 0)) for v in sorted(final_gap_by_var.keys(), key=_natural_var_key)))
        final_ratio_mean = (
            float(sum(float(final_ratio_by_var[v]) for v in sorted(final_ratio_by_var.keys(), key=_natural_var_key)))
            / float(len(final_ratio_by_var))
            if final_ratio_by_var
            else 1.0
        )
        stage3_coverage = _scm_compute_stage3_probe_coverage(
            rows_by_var_cache=rows_by_var_cache,
            probe_subsets_by_var=stage3_probe_subsets,
        )
        stage3_final_summary = _scm_stage3_coverage_summary(
            probe_subsets_by_var=stage3_probe_subsets,
            coverage=stage3_coverage,
        )
        stage3_initial_ratio = float(stage3_initial_summary.get("coverage_ratio", 1.0))
        stage3_final_ratio = float(stage3_final_summary.get("coverage_ratio", 1.0))
        stage3_coverage_gain_ratio = max(0.0, float(stage3_final_ratio - stage3_initial_ratio))
        stage3_initial_seen = int(stage3_initial_summary.get("seen_patterns", 0))
        stage3_final_seen = int(stage3_final_summary.get("seen_patterns", 0))
        stage3_total_patterns = int(stage3_final_summary.get("total_patterns", 0))

        # Minimize remaining shortcuts first, then shortcut simplicity gap, then maximize progress/kills.
        objective = (
            0 if len(train_worlds) >= int(k) else 1,
            int(final_survivors),
            int(final_gap_sum),
            int(-reduction_abs),
            int(-total_kills),
            float(max(0.0, 1.0 - stage3_final_ratio)),
            float(final_ratio_mean),
            int(candidate_plan_evals),
        )

        restart_diag = {
            "attempts": int(candidate_plan_evals),
            "cegis_iters": int(cegis_iters),
            "shortcuts_killed_by_var": {
                var: int(shortcuts_killed_by_var.get(var, 0))
                for var in endogenous_sorted
            },
            "kills_per_iter": list(kills_per_iter),
            "total_shortcuts_killed": int(total_kills),
            "vars_with_kills": list(vars_with_kills),
            "kill_coverage_ratio": float(kill_coverage_ratio),
            "initial_survivors_small_estimate": int(initial_survivors),
            "survivors_small_estimate": int(final_survivors),
            "survivor_reduction_abs": int(reduction_abs),
            "survivor_reduction_ratio": float(reduction_ratio),
            "initial_survivor_vars": sorted(initial_witnesses.keys(), key=_natural_var_key),
            "survivor_vars": sorted(final_witnesses.keys(), key=_natural_var_key),
            "survivor_count_history": list(survivor_count_history),
            "initial_witness_ast_by_var": dict(initial_ast_by_var),
            "final_witness_ast_by_var": dict(final_ast_by_var),
            "initial_witness_cap_by_var": dict(initial_cap_by_var),
            "final_witness_cap_by_var": dict(final_cap_by_var),
            "initial_shortcut_ast_gap_by_var": dict(initial_gap_by_var),
            "final_shortcut_ast_gap_by_var": dict(final_gap_by_var),
            "initial_shortcut_ast_ratio_by_var": dict(initial_ratio_by_var),
            "final_shortcut_ast_ratio_by_var": dict(final_ratio_by_var),
            "final_shortcut_gap_sum": int(final_gap_sum),
            "final_shortcut_ratio_mean": float(final_ratio_mean),
            "resolved_all_shortcuts_under_cap": bool(final_survivors == 0),
            "cegis_restart_index": int(restart_idx),
            "stage3_lite_enabled": bool(stage3_enabled),
            "stage3_lite_probe_size": int(stage3_probe_size_eff),
            "stage3_lite_probe_subsets_per_var": int(stage3_probe_subsets_eff),
            "stage3_lite_world_budget": int(stage3_world_budget_eff),
            "stage3_lite_reserved_slots": int(stage3_reserved_slots),
            "cegis_target_worlds_before_stage3": int(cegis_target_worlds),
            "stage3_lite_worlds_added": int(stage3_worlds_added),
            "stage3_lite_candidate_plan_evals": int(stage3_candidate_plan_evals),
            "stage3_lite_assigned_bias": float(stage3_assigned_bias_eff),
            "stage3_lite_total_new_patterns": int(stage3_total_new_patterns),
            "stage3_lite_initial_patterns_seen": int(stage3_initial_seen),
            "stage3_lite_final_patterns_seen": int(stage3_final_seen),
            "stage3_lite_total_patterns": int(stage3_total_patterns),
            "stage3_lite_initial_coverage_ratio": float(stage3_initial_ratio),
            "stage3_lite_final_coverage_ratio": float(stage3_final_ratio),
            "stage3_lite_coverage_gain_ratio": float(stage3_coverage_gain_ratio),
            "stage3_lite_initial_coverage_by_var": dict(
                stage3_initial_summary.get("coverage_by_var", {})
            ),
            "stage3_lite_final_coverage_by_var": dict(
                stage3_final_summary.get("coverage_by_var", {})
            ),
            "stage3_lite_coverage_history": [float(v) for v in stage3_coverage_history],
            "stage3_lite_kills_by_var": {
                var: int(stage3_kills_by_var.get(var, 0))
                for var in endogenous_sorted
            },
        }
        return train_worlds, seen_interventions, restart_diag, objective

    best_bundle: Optional[
        Tuple[
            List[CausalWorldView],
            Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]],
            Dict[str, Any],
            Tuple[int, int, int, int, int, float, float, int],
            int,
        ]
    ] = None
    restart_summaries: List[Dict[str, Any]] = []
    restarts_executed = 0

    for restart_idx in range(restart_count):
        restarts_executed += 1
        train_worlds_i, seen_i, diag_i, objective_i = _run_one_restart(restart_idx)
        quality_met_i = _restart_quality_met(train_worlds_len=len(train_worlds_i), diag=diag_i)
        restart_summaries.append(
            {
                "restart": int(restart_idx),
                "objective": [
                    int(objective_i[0]),
                    int(objective_i[1]),
                    int(objective_i[2]),
                    int(objective_i[3]),
                    int(objective_i[4]),
                    float(objective_i[5]),
                    float(objective_i[6]),
                    int(objective_i[7]),
                ],
                "initial_survivors": int(diag_i.get("initial_survivors_small_estimate", 0)),
                "final_survivors": int(diag_i.get("survivors_small_estimate", 0)),
                "reduction_abs": int(diag_i.get("survivor_reduction_abs", 0)),
                "reduction_ratio": float(diag_i.get("survivor_reduction_ratio", 0.0)),
                "total_kills": int(diag_i.get("total_shortcuts_killed", 0)),
                "stage3_final_coverage": float(diag_i.get("stage3_lite_final_coverage_ratio", 1.0)),
                "stage3_worlds_added": int(diag_i.get("stage3_lite_worlds_added", 0)),
                "quality_met": bool(quality_met_i),
            }
        )
        if best_bundle is None or objective_i < best_bundle[3]:
            best_bundle = (
                train_worlds_i,
                seen_i,
                diag_i,
                objective_i,
                int(restart_idx),
            )
        if bool(early_stop_on_quality_met) and bool(quality_met_i):
            break

    if best_bundle is None:
        return ([], set(), {"attempts": 0, "cegis_iters": 0, "shortcuts_killed_by_var": {}})

    best_train_worlds, best_seen_interventions, best_diag, _best_objective, best_restart_index = best_bundle
    best_diag = dict(best_diag)
    best_diag["cegis_restart_count"] = int(restarts_executed)
    best_diag["cegis_selected_restart_index"] = int(best_restart_index)
    best_diag["cegis_restart_summaries"] = list(restart_summaries)
    best_diag["cegis_restart_early_stopped"] = bool(
        bool(early_stop_on_quality_met) and int(restarts_executed) < int(restart_count)
    )

    return best_train_worlds, best_seen_interventions, best_diag


def _iter_world_rows(world: WorldLike) -> List[Dict[str, Any]]:
    rows = (_world_extra(world).get("rows")) or []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = row.get("values")
        if isinstance(values, dict):
            out.append(values)
    return out


def _infer_split(world: WorldLike) -> str:
    extra = _world_extra(world)
    split = str(extra.get("split", "")).strip().lower()
    if split in {"train", "heldout"}:
        return split

    wid = _world_id(world).lower()
    if wid.startswith("heldout") or wid.startswith("test"):
        return "heldout"
    return "train"


def _worlds_by_split(problem_payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    worlds = problem_payload.get("worlds", []) or []
    train, heldout = [], []
    for world in worlds:
        split = _infer_split(world)
        if split == "heldout":
            heldout.append(world)
        else:
            train.append(world)
    return train, heldout


def _compress_rows_for_shortcut_search(
    rows_eval: List[Dict[str, Any]],
    target_var: str,
    candidate_vars: List[str],
) -> Tuple[List[str], List[Dict[str, int]], Dict[str, int], int, int, int, bool]:
    """Deduplicate rows by candidate assignment while preserving exact semantics.

    If two rows share the same candidate assignment but disagree on target value,
    the target is not representable as a deterministic mechanism over candidates.
    """
    candidate_unique = list(dict.fromkeys(str(v) for v in candidate_vars))
    assignment_to_target: Dict[int, int] = {}

    for row in rows_eval:
        bits = 0
        for idx, var in enumerate(candidate_unique):
            if int(row.get(var, 0)):
                bits |= (1 << int(idx))
        target_val = int(row.get(target_var, 0)) & 1
        prev = assignment_to_target.get(int(bits))
        if prev is None:
            assignment_to_target[int(bits)] = int(target_val)
        elif int(prev) != int(target_val):
            return candidate_unique, [], {}, 0, 0, 0, True

    assignment_items = list(assignment_to_target.items())
    row_count = int(len(assignment_items))
    if row_count <= 0:
        return candidate_unique, [], {}, 0, 0, 0, False

    full_mask = (1 << row_count) - 1
    target_mask = 0
    var_masks: List[int] = [0 for _ in candidate_unique]
    compressed_rows: List[Dict[str, int]] = []

    for row_idx, (assignment_bits, target_val) in enumerate(assignment_items):
        if int(target_val):
            target_mask |= (1 << int(row_idx))
        row_payload: Dict[str, int] = {str(target_var): int(target_val)}
        for var_idx, var in enumerate(candidate_unique):
            value = 1 if ((int(assignment_bits) >> int(var_idx)) & 1) else 0
            row_payload[str(var)] = int(value)
            if int(value):
                var_masks[int(var_idx)] |= (1 << int(row_idx))
        compressed_rows.append(row_payload)

    var_mask_lookup = {str(var): int(var_masks[idx]) for idx, var in enumerate(candidate_unique)}
    return (
        candidate_unique,
        compressed_rows,
        var_mask_lookup,
        int(row_count),
        int(full_mask),
        int(target_mask),
        False,
    )


def _has_small_shortcut(
    rows: List[Dict[str, Any]],
    target_var: str,
    candidate_vars: List[str],
    ast_cap: int,
    allowed_operators: Optional[List[str]] = None,
    allow_constants: bool = True,
    time_budget_ms: Optional[int] = None,
    max_signatures_per_size: Optional[int] = None,
    timeout_policy: str = "sampled_fallback",
    timeout_fallback_samples: int = 192,
) -> bool:
    cap = int(ast_cap)
    if cap < 1 or not rows:
        return False

    rows_eval = [row for row in rows if target_var in row]
    if not rows_eval:
        return False

    (
        candidate_vars_unique,
        rows_eval_shortcut,
        var_mask_lookup,
        row_count,
        full_mask,
        target_mask,
        has_conflict,
    ) = _compress_rows_for_shortcut_search(
        rows_eval=rows_eval,
        target_var=target_var,
        candidate_vars=candidate_vars,
    )
    if has_conflict:
        return False
    if row_count <= 0:
        return False

    ops = set(allowed_operators or list(DEFAULT_ALLOWED_OPERATORS))
    allow_not = "not" in ops
    allow_nary = [op for op in ("and", "or", "xor", "iff") if op in ops]
    commutative_ops = {"and", "or", "xor", "iff"}
    signature_cap = (
        None
        if max_signatures_per_size is None or int(max_signatures_per_size) <= 0
        else int(max_signatures_per_size)
    )

    timeout_policy_norm = str(timeout_policy or "sampled_fallback").strip().lower()

    deadline = (
        None
        if time_budget_ms is None or int(time_budget_ms) <= 0
        else (time.perf_counter() + (int(time_budget_ms) / 1000.0))
    )
    op_counter = 0

    def _time_exceeded() -> bool:
        nonlocal op_counter
        if deadline is None:
            return False
        op_counter += 1
        if (op_counter & 127) != 0:
            return False
        return time.perf_counter() >= deadline

    def _sampled_timeout_fallback() -> bool:
        sample_n = max(0, int(timeout_fallback_samples))
        if sample_n <= 0:
            return False

        allow_if = "if" in ops
        max_depth = max(2, min(6, int(cap)))
        seed = (
            int(target_mask)
            ^ (int(cap) << 7)
            ^ (len(candidate_vars_unique) << 13)
            ^ (len(rows_eval) << 19)
        ) & 0xFFFFFFFF
        rng = random.Random(seed)
        for _ in range(sample_n):
            depth = rng.randint(2, max_depth)
            node = _canonicalize_node(
                _rand_expr_tree(
                    rng=rng,
                    variables=candidate_vars_unique,
                    max_depth=depth,
                    allow_if=allow_if,
                )
            )
            if (not allow_constants) and _node_contains_const(node):
                continue
            stats = analyze_mechanism(node)
            ast = int(stats.get("astSize", _node_ast_size(node)))
            if ast < 1 or ast > int(cap):
                continue
            out_mask = 0
            for idx, row in enumerate(rows_eval_shortcut):
                if int(evaluate_parsed_mechanism(node, row)):
                    out_mask |= (1 << idx)
            if int(out_mask) == int(target_mask):
                return True
        return False

    def _on_timeout() -> bool:
        if timeout_policy_norm in {"assume_shortcut", "shortcut", "fail_closed", "reject"}:
            return True
        if timeout_policy_norm in {"assume_no_shortcut", "pass", "fail_open", "allow"}:
            return False
        # Default: bounded sampled fallback.
        return _sampled_timeout_fallback()

    by_size: Dict[int, Set[int]] = {s: set() for s in range(1, cap + 1)}

    def _add(size: int, sig: int) -> bool:
        if size < 1 or size > cap:
            return False
        if sig == target_mask:
            return True
        if sig in by_size[size]:
            return False
        if signature_cap is not None and len(by_size[size]) >= int(signature_cap):
            return False
        by_size[size].add(sig)
        return False

    # Base atoms: constants and candidate variables.
    if allow_constants:
        if _add(1, 0):
            return True
        if _add(1, full_mask):
            return True
    for v in candidate_vars_unique:
        sig = int(var_mask_lookup.get(str(v), 0))
        if _add(1, sig):
            return True

    binary_cache: Dict[
        Tuple[str, int, int],
        int,
    ] = {}
    ternary_cache: Dict[
        Tuple[str, int, int, int],
        int,
    ] = {}

    def _apply_binary(op: str, left: int, right: int) -> int:
        if op in commutative_ops and right < left:
            left, right = right, left
        key = (op, left, right)
        cached = binary_cache.get(key)
        if cached is not None:
            return int(cached)
        if op == "and":
            out = left & right
        elif op == "or":
            out = left | right
        elif op == "xor":
            out = left ^ right
        else:
            # iff: all-equal semantics for binary is equivalence.
            out = full_mask ^ (left ^ right)
        out = int(out) & int(full_mask)
        binary_cache[key] = out
        return int(out)

    def _apply_ternary(
        op: str,
        a_sig: int,
        b_sig: int,
        c_sig: int,
    ) -> int:
        if op in commutative_ops:
            a_sig, b_sig, c_sig = tuple(sorted((a_sig, b_sig, c_sig)))
        key = (op, a_sig, b_sig, c_sig)
        cached = ternary_cache.get(key)
        if cached is not None:
            return int(cached)
        if op == "and":
            out = a_sig & b_sig & c_sig
        elif op == "or":
            out = a_sig | b_sig | c_sig
        elif op == "xor":
            out = a_sig ^ b_sig ^ c_sig
        else:
            # iff: all three equal.
            out = full_mask ^ ((a_sig ^ b_sig) | (a_sig ^ c_sig))
        out = int(out) & int(full_mask)
        ternary_cache[key] = out
        return int(out)

    for size in range(2, cap + 1):
        if _time_exceeded():
            return _on_timeout()
        # Lower-size buckets are immutable while constructing this exact size.
        sigs_by_size: Dict[int, Tuple[int, ...]] = {
            s: tuple(by_size[s]) for s in range(1, size)
        }
        if allow_not:
            for sig in sigs_by_size.get(size - 1, ()):
                if _time_exceeded():
                    return _on_timeout()
                out = full_mask ^ int(sig)
                if _add(size, out):
                    return True

        # Binary n-ary operators (2-arg case): size = 1 + a + b.
        for op in allow_nary:
            for a in range(1, size - 1):
                b = size - 1 - a
                if op in commutative_ops and a > b:
                    continue
                left_sigs = sigs_by_size.get(a, ())
                right_sigs = sigs_by_size.get(b, ())
                if not left_sigs or not right_sigs:
                    continue
                same_bucket = (a == b) and (op in commutative_ops)
                for left_idx, left in enumerate(left_sigs):
                    if _time_exceeded():
                        return _on_timeout()
                    right_start = left_idx if same_bucket else 0
                    for right in right_sigs[right_start:]:
                        if _time_exceeded():
                            return _on_timeout()
                        out = _apply_binary(op, left, right)
                        if _add(size, out):
                            return True

        # Ternary n-ary operators (3-arg case): size = 1 + a + b + c.
        if size >= 4:
            for op in allow_nary:
                for a in range(1, size - 2):
                    for b in range(1, size - 1 - a):
                        c = size - 1 - a - b
                        if c < 1:
                            continue
                        if op in commutative_ops and not (a <= b <= c):
                            continue
                        a_sigs = sigs_by_size.get(a, ())
                        b_sigs = sigs_by_size.get(b, ())
                        c_sigs = sigs_by_size.get(c, ())
                        if not a_sigs or not b_sigs or not c_sigs:
                            continue
                        for ai, a_sig in enumerate(a_sigs):
                            if _time_exceeded():
                                return _on_timeout()
                            bj_start = ai if (op in commutative_ops and a == b) else 0
                            for bj in range(bj_start, len(b_sigs)):
                                if _time_exceeded():
                                    return _on_timeout()
                                b_sig = b_sigs[bj]
                                ck_start = bj if (op in commutative_ops and b == c) else 0
                                for c_sig in c_sigs[ck_start:]:
                                    if _time_exceeded():
                                        return _on_timeout()
                                    out = _apply_ternary(op, a_sig, b_sig, c_sig)
                                    if _add(size, out):
                                        return True

    return False


def _find_small_shortcut_witness(
    rows: List[Dict[str, Any]],
    target_var: str,
    candidate_vars: List[str],
    ast_cap: int,
    allowed_operators: Optional[List[str]] = None,
    allow_constants: bool = True,
    time_budget_ms: Optional[int] = None,
    max_signatures_per_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return one smallest shortcut witness expression (if any) up to AST cap."""
    cap = int(ast_cap)
    if cap < 1 or not rows or not candidate_vars:
        return None

    rows_eval = [row for row in rows if target_var in row]
    if not rows_eval:
        return None

    (
        candidate_vars_unique,
        _rows_eval_shortcut,
        var_mask_lookup,
        row_count,
        full_mask,
        target_mask,
        has_conflict,
    ) = _compress_rows_for_shortcut_search(
        rows_eval=rows_eval,
        target_var=target_var,
        candidate_vars=candidate_vars,
    )
    if has_conflict:
        return None
    if row_count <= 0:
        return None
    if not candidate_vars_unique:
        return None

    ops = set(allowed_operators or list(DEFAULT_ALLOWED_OPERATORS))
    allow_not = "not" in ops
    allow_nary = [op for op in ("and", "or", "xor", "iff") if op in ops]
    commutative_ops = {"and", "or", "xor", "iff"}
    signature_cap = (
        None
        if max_signatures_per_size is None or int(max_signatures_per_size) <= 0
        else int(max_signatures_per_size)
    )
    deadline = (
        None
        if time_budget_ms is None or int(time_budget_ms) <= 0
        else (time.perf_counter() + (int(time_budget_ms) / 1000.0))
    )
    op_counter = 0

    def _time_exceeded() -> bool:
        nonlocal op_counter
        if deadline is None:
            return False
        op_counter += 1
        if (op_counter & 127) != 0:
            return False
        return time.perf_counter() >= deadline

    # Store only signatures in the hot loop; reconstruct one witness lazily.
    by_size: List[Dict[int, int]] = [dict() for _ in range(cap + 1)]
    by_size_items: List[List[Tuple[int, int]]] = [[] for _ in range(cap + 1)]
    node_specs: List[Tuple[str, str, Tuple[int, ...]]] = []
    op_node_cache: Dict[Tuple[str, Tuple[int, ...]], int] = {}

    def _make_atom(kind: str, value: str) -> int:
        node_specs.append((kind, str(value), tuple()))
        return len(node_specs) - 1

    def _make_op_node(op: str, child_ids: Tuple[int, ...]) -> int:
        key = (str(op), child_ids)
        cached = op_node_cache.get(key)
        if cached is not None:
            return int(cached)
        node_specs.append(("op", str(op), child_ids))
        idx = len(node_specs) - 1
        op_node_cache[key] = idx
        return idx

    def _materialize(size: int, node_id: int) -> Optional[Dict[str, Any]]:
        node_cache: Dict[int, MechanismNode] = {}

        def _build(nid: int) -> MechanismNode:
            cached = node_cache.get(int(nid))
            if cached is not None:
                return cached
            kind, value, children = node_specs[int(nid)]
            if kind == "var":
                out = _make_var(str(value))
            elif kind == "const":
                out = _const_node(int(value))
            else:
                out = _make_op(str(value), [_build(int(cid)) for cid in children])
            node_cache[int(nid)] = out
            return out

        node = _build(int(node_id))
        stats = analyze_mechanism(node)
        return {
            "expr": node_to_sexpr(node),
            "node": node,
            "ast": int(stats.get("astSize", size)),
            "stats": stats,
        }

    def _add(size: int, sig: int, node_id: int) -> Optional[Dict[str, Any]]:
        if size < 1 or size > cap:
            return None
        if sig == target_mask:
            return _materialize(size, int(node_id))
        if sig in by_size[size]:
            return None
        if signature_cap is not None and len(by_size[size]) >= int(signature_cap):
            return None
        by_size[size][sig] = int(node_id)
        by_size_items[size].append((int(sig), int(node_id)))
        return None

    # Base atoms: constants and candidate variables.
    if allow_constants:
        witness = _add(1, 0, _make_atom("const", "0"))
        if witness is not None:
            return witness
        witness = _add(1, full_mask, _make_atom("const", "1"))
        if witness is not None:
            return witness
    for v in sorted(set(candidate_vars_unique), key=_natural_var_key):
        sig = int(var_mask_lookup.get(str(v), 0))
        witness = _add(1, sig, _make_atom("var", str(v)))
        if witness is not None:
            return witness

    binary_caches: Dict[str, Dict[Tuple[int, int], int]] = {str(op): {} for op in allow_nary}
    ternary_caches: Dict[str, Dict[Tuple[int, int, int], int]] = {str(op): {} for op in allow_nary}

    def _apply_binary(op: str, left: int, right: int) -> int:
        if op in commutative_ops and right < left:
            left, right = right, left
        cache = binary_caches[op]
        key = (left, right)
        cached = cache.get(key)
        if cached is not None:
            return int(cached)
        if op == "and":
            out = left & right
        elif op == "or":
            out = left | right
        elif op == "xor":
            out = left ^ right
        else:
            # iff: all-equal semantics for binary is equivalence.
            out = full_mask ^ (left ^ right)
        out = int(out) & int(full_mask)
        cache[key] = out
        return int(out)

    def _apply_ternary(
        op: str,
        a_sig: int,
        b_sig: int,
        c_sig: int,
    ) -> int:
        if op in commutative_ops:
            a_sig, b_sig, c_sig = tuple(sorted((a_sig, b_sig, c_sig)))
        cache = ternary_caches[op]
        key = (a_sig, b_sig, c_sig)
        cached = cache.get(key)
        if cached is not None:
            return int(cached)
        if op == "and":
            out = a_sig & b_sig & c_sig
        elif op == "or":
            out = a_sig | b_sig | c_sig
        elif op == "xor":
            out = a_sig ^ b_sig ^ c_sig
        else:
            # iff: all three equal.
            out = full_mask ^ ((a_sig ^ b_sig) | (a_sig ^ c_sig))
        out = int(out) & int(full_mask)
        cache[key] = out
        return int(out)

    for size in range(2, cap + 1):
        if _time_exceeded():
            return None
        if allow_not:
            for sig, child_id in by_size_items[size - 1]:
                if _time_exceeded():
                    return None
                out = full_mask ^ int(sig)
                if out in by_size[size]:
                    continue
                node_id = _make_op_node("not", (int(child_id),))
                witness = _add(size, out, node_id)
                if witness is not None:
                    return witness

        # Binary n-ary operators (2-arg case): size = 1 + a + b.
        for op in allow_nary:
            for a in range(1, size - 1):
                b = size - 1 - a
                if op in commutative_ops and a > b:
                    continue
                left_items = by_size_items[a]
                right_items = by_size_items[b]
                if not left_items or not right_items:
                    continue
                same_bucket = (a == b) and (op in commutative_ops)
                for left_idx, (left_sig, left_id) in enumerate(left_items):
                    if _time_exceeded():
                        return None
                    right_start = left_idx if same_bucket else 0
                    for right_idx in range(int(right_start), len(right_items)):
                        right_sig, right_id = right_items[right_idx]
                        if _time_exceeded():
                            return None
                        out = _apply_binary(op, left_sig, right_sig)
                        if out in by_size[size]:
                            continue
                        node_id = _make_op_node(op, (int(left_id), int(right_id)))
                        witness = _add(size, out, node_id)
                        if witness is not None:
                            return witness

        # Ternary n-ary operators (3-arg case): size = 1 + a + b + c.
        if size >= 4:
            for op in allow_nary:
                for a in range(1, size - 2):
                    for b in range(1, size - 1 - a):
                        c = size - 1 - a - b
                        if c < 1:
                            continue
                        if op in commutative_ops and not (a <= b <= c):
                            continue
                        a_items = by_size_items[a]
                        b_items = by_size_items[b]
                        c_items = by_size_items[c]
                        if not a_items or not b_items or not c_items:
                            continue
                        for ai, (a_sig, a_id) in enumerate(a_items):
                            if _time_exceeded():
                                return None
                            bj_start = ai if (op in commutative_ops and a == b) else 0
                            for bj in range(int(bj_start), len(b_items)):
                                if _time_exceeded():
                                    return None
                                b_sig, b_id = b_items[bj]
                                ck_start = bj if (op in commutative_ops and b == c) else 0
                                for ck in range(int(ck_start), len(c_items)):
                                    c_sig, c_id = c_items[ck]
                                    if _time_exceeded():
                                        return None
                                    out = _apply_ternary(op, a_sig, b_sig, c_sig)
                                    if out in by_size[size]:
                                        continue
                                    node_id = _make_op_node(
                                        op,
                                        (int(a_id), int(b_id), int(c_id)),
                                    )
                                    witness = _add(size, out, node_id)
                                    if witness is not None:
                                        return witness

    return None


def _has_nonparent_shortcut(
    rows: List[Dict[str, Any]],
    target_var: str,
    nonparent_vars: List[str],
    ast_cap: int,
    allowed_operators: Optional[List[str]] = None,
    allow_constants: bool = True,
    time_budget_ms: Optional[int] = None,
    max_signatures_per_size: Optional[int] = None,
    timeout_policy: str = "sampled_fallback",
    timeout_fallback_samples: int = 192,
) -> bool:
    return _has_small_shortcut(
        rows=rows,
        target_var=target_var,
        candidate_vars=nonparent_vars,
        ast_cap=ast_cap,
        allowed_operators=allowed_operators,
        allow_constants=allow_constants,
        time_budget_ms=time_budget_ms,
        max_signatures_per_size=max_signatures_per_size,
        timeout_policy=timeout_policy,
        timeout_fallback_samples=timeout_fallback_samples,
    )


def _anti_shortcut_failure_reason(
    train_worlds: List[CausalWorldView],
    parents: List[str],
    input_vars: List[str],
    target_var: str,
    nonparent_shortcut_ast_cap: Optional[int] = None,
    allvar_shortcut_ast_cap: Optional[int] = None,
    allowed_ops: Optional[List[str]] = None,
    cegis_survivor_diagnostics: Optional[Dict[str, Any]] = None,
    target_survivors_small_min: Optional[int] = None,
    target_survivors_small_max: Optional[int] = None,
    gap_to_second_best_min: Optional[int] = None,
    gap_to_second_best_max: Optional[int] = None,
    require_survivors_small_or_gap: bool = False,
    min_total_kills: Optional[int] = None,
    min_worlds_with_kills: Optional[int] = None,
    min_parent_assignment_coverage: Optional[float] = None,
    allow_constants: bool = True,
    shortcut_check_time_budget_ms: Optional[int] = None,
    shortcut_check_max_signatures_per_size: Optional[int] = None,
    shortcut_check_timeout_policy: str = "sampled_fallback",
    shortcut_check_timeout_fallback_samples: int = 192,
) -> Optional[str]:
    all_rows = []
    for world in train_worlds:
        all_rows.extend(_iter_world_rows(world))

    if not all_rows:
        return "no_train_rows"

    # Reject degenerate targets: require both labels in training.
    y_values = {int(row[target_var]) for row in all_rows if target_var in row}
    if len(y_values) <= 1:
        return "degenerate_target_constant"

    parent_intervened = {p: False for p in parents}
    touched_parent = False
    for world in train_worlds:
        ints = world.interventions or []
        for it in ints:
            assignments = (it or {}).get("assignments", {})
            for p in parents:
                if p in assignments:
                    parent_intervened[p] = True
                    touched_parent = True
    if not touched_parent:
        return "no_parent_intervened"

    # Every parent must either vary in pooled rows or be intervened at least once.
    for p in parents:
        values = {int(row[p]) for row in all_rows if p in row}
        parent_varies = len(values) > 1
        if not (parent_varies or parent_intervened.get(p, False)):
            return "parent_noninformative"

    if min_parent_assignment_coverage is not None and parents:
        parent_rows = [row for row in all_rows if all(p in row for p in parents)]
        if not parent_rows:
            return "no_complete_parent_rows"
        observed_parent_assignments = {
            tuple(int(row[p]) for p in parents) for row in parent_rows
        }
        total_parent_assignments = 1 << len(parents)
        coverage = len(observed_parent_assignments) / float(total_parent_assignments)
        if coverage + 1e-12 < float(min_parent_assignment_coverage):
            return "parent_assignment_coverage_too_low"

    # Avoid trivial equality shortcuts: Y == Xi for all rows.
    for x in input_vars:
        if x in parents:
            continue
        if all(int(row.get(x, -1)) == int(row.get(target_var, -2)) for row in all_rows):
            return "trivial_nonparent_equals_target"

    # Require at least one intervention world that changes Y distribution
    # relative to the observational baseline.
    def _has_assignments(world: CausalWorldView) -> bool:
        for it in world.interventions or []:
            assignments = (it or {}).get("assignments", {})
            if assignments:
                return True
        return False

    baseline_world = next((w for w in train_worlds if not _has_assignments(w)), train_worlds[0])

    def _y_true_count(world: CausalWorldView) -> int:
        rows = _iter_world_rows(world)
        return sum(int(row[target_var]) for row in rows if target_var in row)

    baseline_count = _y_true_count(baseline_world)
    found_distribution_shift = False
    for world in train_worlds:
        if world.worldId == baseline_world.worldId:
            continue
        if not _has_assignments(world):
            continue
        if _y_true_count(world) != baseline_count:
            found_distribution_shift = True
            break

    if not found_distribution_shift:
        return "no_target_distribution_shift"

    # Reject if any small non-parent-only mechanism can exactly fit train rows.
    if nonparent_shortcut_ast_cap is not None:
        nonparents = [x for x in input_vars if x not in set(parents)]
        if _has_small_shortcut(
            rows=all_rows,
            target_var=target_var,
            candidate_vars=nonparents,
            ast_cap=int(nonparent_shortcut_ast_cap),
            allowed_operators=allowed_ops,
            allow_constants=allow_constants,
            time_budget_ms=shortcut_check_time_budget_ms,
            max_signatures_per_size=shortcut_check_max_signatures_per_size,
            timeout_policy=shortcut_check_timeout_policy,
            timeout_fallback_samples=shortcut_check_timeout_fallback_samples,
        ):
            return "nonparent_shortcut_found"

    # Reject if any small mechanism over all observed X variables fits train rows.
    if allvar_shortcut_ast_cap is not None:
        if _has_small_shortcut(
            rows=all_rows,
            target_var=target_var,
            candidate_vars=input_vars,
            ast_cap=int(allvar_shortcut_ast_cap),
            allowed_operators=allowed_ops,
            allow_constants=allow_constants,
            time_budget_ms=shortcut_check_time_budget_ms,
            max_signatures_per_size=shortcut_check_max_signatures_per_size,
            timeout_policy=shortcut_check_timeout_policy,
            timeout_fallback_samples=shortcut_check_timeout_fallback_samples,
        ):
            return "allvar_shortcut_found"

    if cegis_survivor_diagnostics is not None:
        survivors_small = cegis_survivor_diagnostics.get("survivors_small")
        survivors_total = cegis_survivor_diagnostics.get("survivors_total")
        gap = cegis_survivor_diagnostics.get("gap_to_second_best")
        kills_per_world = [
            int(v) for v in (cegis_survivor_diagnostics.get("kills_per_world") or []) if v is not None
        ]

        survivors_exhausted = survivors_total is not None and int(survivors_total) == 0
        if not survivors_exhausted:
            if min_total_kills is not None and sum(kills_per_world) < int(min_total_kills):
                return "cegis_total_kills_below_min"
            if min_worlds_with_kills is not None:
                worlds_with_kills = sum(1 for v in kills_per_world if int(v) > 0)
                if worlds_with_kills < int(min_worlds_with_kills):
                    return "cegis_worlds_with_kills_below_min"

        # If no distractors survive, gap-related constraints are vacuous.
        if not survivors_exhausted:
            if require_survivors_small_or_gap:
                min_survivors = (
                    int(target_survivors_small_min)
                    if target_survivors_small_min is not None
                    else 1
                )
                min_gap = int(gap_to_second_best_min) if gap_to_second_best_min is not None else -10**9
                max_gap = int(gap_to_second_best_max) if gap_to_second_best_max is not None else 10**9
                cond_survivors = survivors_small is not None and int(survivors_small) >= min_survivors
                cond_gap = gap is not None and min_gap <= int(gap) <= max_gap
                if not (cond_survivors or cond_gap):
                    return "cegis_survivors_or_gap_unsatisfied"
            else:
                if target_survivors_small_min is not None:
                    if survivors_small is None or int(survivors_small) < int(target_survivors_small_min):
                        return "cegis_survivors_small_below_min"

                if target_survivors_small_max is not None:
                    if survivors_small is None or int(survivors_small) > int(target_survivors_small_max):
                        return "cegis_survivors_small_above_max"

                if gap_to_second_best_min is not None:
                    if gap is None or int(gap) < int(gap_to_second_best_min):
                        return "cegis_gap_below_min"

                if gap_to_second_best_max is not None:
                    if gap is None or int(gap) > int(gap_to_second_best_max):
                        return "cegis_gap_above_max"

    return None


def _anti_shortcut_checks(
    train_worlds: List[CausalWorldView],
    parents: List[str],
    input_vars: List[str],
    target_var: str,
    nonparent_shortcut_ast_cap: Optional[int] = None,
    allvar_shortcut_ast_cap: Optional[int] = None,
    allowed_ops: Optional[List[str]] = None,
    cegis_survivor_diagnostics: Optional[Dict[str, Any]] = None,
    target_survivors_small_min: Optional[int] = None,
    target_survivors_small_max: Optional[int] = None,
    gap_to_second_best_min: Optional[int] = None,
    gap_to_second_best_max: Optional[int] = None,
    require_survivors_small_or_gap: bool = False,
    min_total_kills: Optional[int] = None,
    min_worlds_with_kills: Optional[int] = None,
    min_parent_assignment_coverage: Optional[float] = None,
    allow_constants: bool = True,
    shortcut_check_time_budget_ms: Optional[int] = None,
    shortcut_check_max_signatures_per_size: Optional[int] = None,
    shortcut_check_timeout_policy: str = "sampled_fallback",
    shortcut_check_timeout_fallback_samples: int = 192,
) -> bool:
    return (
        _anti_shortcut_failure_reason(
            train_worlds=train_worlds,
            parents=parents,
            input_vars=input_vars,
            target_var=target_var,
            nonparent_shortcut_ast_cap=nonparent_shortcut_ast_cap,
            allvar_shortcut_ast_cap=allvar_shortcut_ast_cap,
            allowed_ops=allowed_ops,
            cegis_survivor_diagnostics=cegis_survivor_diagnostics,
            target_survivors_small_min=target_survivors_small_min,
            target_survivors_small_max=target_survivors_small_max,
            gap_to_second_best_min=gap_to_second_best_min,
            gap_to_second_best_max=gap_to_second_best_max,
            require_survivors_small_or_gap=require_survivors_small_or_gap,
            min_total_kills=min_total_kills,
            min_worlds_with_kills=min_worlds_with_kills,
            min_parent_assignment_coverage=min_parent_assignment_coverage,
            allow_constants=allow_constants,
            shortcut_check_time_budget_ms=shortcut_check_time_budget_ms,
            shortcut_check_max_signatures_per_size=shortcut_check_max_signatures_per_size,
            shortcut_check_timeout_policy=shortcut_check_timeout_policy,
            shortcut_check_timeout_fallback_samples=shortcut_check_timeout_fallback_samples,
        )
        is None
    )


def _scm_training_signal_failure_reason(
    train_worlds: List[CausalWorldView],
    endogenous_vars: List[str],
) -> Optional[str]:
    all_rows: List[Dict[str, Any]] = []
    for world in train_worlds:
        all_rows.extend(_iter_world_rows(world))
    if not all_rows:
        return "no_train_rows"

    varied = 0
    for var in endogenous_vars:
        vals = {int(row[var]) for row in all_rows if var in row}
        if len(vals) > 1:
            varied += 1
    if varied == 0:
        return "degenerate_all_endogenous_constant"

    min_varied = max(1, len(endogenous_vars) // 2)
    if varied < min_varied:
        return "insufficient_endogenous_variation"

    baseline_world = next(
        (w for w in train_worlds if not _extract_world_intervention_targets(w)),
        train_worlds[0],
    )
    baseline_rows = _iter_world_rows(baseline_world)
    baseline_counts = {
        var: sum(int(row.get(var, 0)) for row in baseline_rows if var in row)
        for var in endogenous_vars
    }

    shifted = False
    for world in train_worlds:
        if world.worldId == baseline_world.worldId:
            continue
        if not _extract_world_intervention_targets(world):
            continue
        rows = _iter_world_rows(world)
        for var in endogenous_vars:
            c = sum(int(row.get(var, 0)) for row in rows if var in row)
            if c != baseline_counts.get(var, c):
                shifted = True
                break
        if shifted:
            break
    if not shifted:
        return "no_endogenous_distribution_shift"
    return None


def _scm_small_shortcut_failure_reason(
    train_worlds: List[CausalWorldView],
    topological_order: List[str],
    endogenous_vars: List[str],
    mechanism_stats_by_var: Dict[str, Dict[str, Any]],
    allowed_ops: List[str],
    ast_cap_floor: int = 3,
    ast_cap_max: int = 6,
    check_vars: Optional[List[str]] = None,
    allow_constants: bool = True,
    max_predecessors_per_target: Optional[int] = None,
) -> Optional[str]:
    witnesses = _scm_collect_shortcut_witnesses(
        train_worlds=train_worlds,
        topological_order=topological_order,
        endogenous_vars=endogenous_vars,
        mechanism_stats_by_var=mechanism_stats_by_var,
        allowed_ops=allowed_ops,
        ast_cap_floor=ast_cap_floor,
        ast_cap_max=ast_cap_max,
        check_vars=check_vars,
        allow_constants=allow_constants,
        max_predecessors_per_target=max_predecessors_per_target,
    )
    if not witnesses:
        return None
    first_var = next(iter(witnesses.keys()))
    return f"scm_small_shortcut_found_{first_var}"


def _scm_world_rows_and_intervened(
    world: WorldLike,
    *,
    payload_cache: Optional[Dict[int, Tuple[Set[str], List[Dict[str, Any]]]]] = None,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    cache_key = id(world)
    if payload_cache is not None:
        cached = payload_cache.get(cache_key)
        if cached is not None:
            return cached
    payload = (_extract_world_intervention_targets(world), _iter_world_rows(world))
    if payload_cache is not None:
        payload_cache[cache_key] = payload
    return payload


def _scm_extend_rows_by_var_cache(
    rows_by_var_cache: Dict[str, List[Dict[str, Any]]],
    vars_to_track: List[str],
    intervened: Set[str],
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return
    for var in vars_to_track:
        if var in intervened:
            continue
        rows_by_var_cache.setdefault(var, []).extend(rows)


def _scm_build_stage3_probe_subsets(
    topological_order: List[str],
    endogenous_vars: List[str],
    *,
    max_predecessors_per_target: Optional[int],
    probe_size: int,
    subsets_per_var: int,
) -> Dict[str, List[Tuple[str, ...]]]:
    index_of = {v: i for i, v in enumerate(topological_order)}
    target_probe_size = max(1, int(probe_size))
    max_sets = max(1, int(subsets_per_var))
    probe_subsets: Dict[str, List[Tuple[str, ...]]] = {}

    for var in endogenous_vars:
        idx = int(index_of.get(var, -1))
        if idx <= 0:
            continue
        predecessors = _bounded_predecessors(
            topological_order[:idx],
            max_predecessors=max_predecessors_per_target,
        )
        if not predecessors:
            continue
        subset_size = min(target_probe_size, len(predecessors))
        combos = list(itertools.combinations(predecessors, subset_size))
        if not combos:
            continue

        if len(combos) <= max_sets:
            selected = combos
        elif max_sets == 1:
            selected = [combos[0]]
        else:
            picked_indices: List[int] = []
            denom = max(1, max_sets - 1)
            for i in range(max_sets):
                idx_f = int(round(float(i) * float(len(combos) - 1) / float(denom)))
                idx_i = min(max(0, idx_f), len(combos) - 1)
                if idx_i not in picked_indices:
                    picked_indices.append(idx_i)
            selected = [combos[i] for i in picked_indices]

        if selected:
            probe_subsets[str(var)] = [tuple(str(v) for v in subset) for subset in selected]
    return probe_subsets


def _scm_compile_stage3_probe_specs(
    probe_subsets_by_var: Dict[str, List[Tuple[str, ...]]],
) -> Dict[str, List[Tuple[Tuple[str, Tuple[str, ...]], Tuple[str, ...], int]]]:
    out: Dict[str, List[Tuple[Tuple[str, Tuple[str, ...]], Tuple[str, ...], int]]] = {}
    for var, subsets in probe_subsets_by_var.items():
        specs: List[Tuple[Tuple[str, Tuple[str, ...]], Tuple[str, ...], int]] = []
        for subset in subsets:
            subset_tuple = tuple(str(v) for v in subset)
            key = (str(var), subset_tuple)
            support_mask = (1 << len(subset_tuple)) - 1
            specs.append((key, subset_tuple, int(support_mask)))
        out[str(var)] = specs
    return out


def _scm_compute_stage3_probe_coverage(
    rows_by_var_cache: Dict[str, List[Dict[str, Any]]],
    probe_subsets_by_var: Dict[str, List[Tuple[str, ...]]],
) -> Dict[Tuple[str, Tuple[str, ...]], int]:
    coverage: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    for var, subsets in probe_subsets_by_var.items():
        rows = rows_by_var_cache.get(str(var), [])
        for subset in subsets:
            key = (str(var), tuple(str(v) for v in subset))
            mask = 0
            support_mask = (1 << len(subset)) - 1
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if any(s not in row for s in subset):
                    continue
                pattern_id = 0
                for idx, s in enumerate(subset):
                    if int(row[s]):
                        pattern_id |= (1 << idx)
                mask |= (1 << int(pattern_id))
                if int(mask) == int(support_mask):
                    break
            coverage[key] = int(mask)
    return coverage


def _scm_stage3_pattern_mask_from_dep_masks(
    *,
    dep_masks: Tuple[int, ...],
    full_units_mask: int,
) -> int:
    dep_count = int(len(dep_masks))
    if dep_count <= 0:
        return 0
    full = int(full_units_mask)
    if dep_count == 1:
        a = int(dep_masks[0]) & int(full)
        out = 0
        if int((~int(a)) & int(full)) != 0:
            out |= 1
        if int(a) != 0:
            out |= 2
        return int(out)
    if dep_count == 2:
        a = int(dep_masks[0]) & int(full)
        b = int(dep_masks[1]) & int(full)
        na = int((~int(a)) & int(full))
        nb = int((~int(b)) & int(full))
        out = 0
        if int(na & nb) != 0:
            out |= 1 << 0
        if int(a & nb) != 0:
            out |= 1 << 1
        if int(na & b) != 0:
            out |= 1 << 2
        if int(a & b) != 0:
            out |= 1 << 3
        return int(out)
    if dep_count == 3:
        a = int(dep_masks[0]) & int(full)
        b = int(dep_masks[1]) & int(full)
        c = int(dep_masks[2]) & int(full)
        na = int((~int(a)) & int(full))
        nb = int((~int(b)) & int(full))
        nc = int((~int(c)) & int(full))
        out = 0
        if int(na & nb & nc) != 0:
            out |= 1 << 0
        if int(a & nb & nc) != 0:
            out |= 1 << 1
        if int(na & b & nc) != 0:
            out |= 1 << 2
        if int(a & b & nc) != 0:
            out |= 1 << 3
        if int(na & nb & c) != 0:
            out |= 1 << 4
        if int(a & nb & c) != 0:
            out |= 1 << 5
        if int(na & b & c) != 0:
            out |= 1 << 6
        if int(a & b & c) != 0:
            out |= 1 << 7
        return int(out)

    # Fallback for larger probe sizes.
    support_mask = (1 << dep_count) - 1
    pattern_mask = 0
    for pattern_id in range(1 << dep_count):
        match_mask = int(full)
        for idx, dep_mask in enumerate(dep_masks):
            if (int(pattern_id) >> int(idx)) & 1:
                match_mask &= int(dep_mask)
            else:
                match_mask &= int((~int(dep_mask)) & int(full))
            if int(match_mask) == 0:
                break
        if int(match_mask) != 0:
            pattern_mask |= (1 << int(pattern_id))
            if int(pattern_mask) == int(support_mask):
                break
    return int(pattern_mask) & int(support_mask)


def _scm_stage3_pattern_mask_from_bitsets(
    *,
    subset: Tuple[str, ...],
    var_bitsets: Dict[str, int],
    full_units_mask: int,
) -> int:
    if not subset:
        return 0
    dep_masks = tuple(
        int(var_bitsets.get(str(var), 0)) & int(full_units_mask)
        for var in subset
    )
    return _scm_stage3_pattern_mask_from_dep_masks(
        dep_masks=dep_masks,
        full_units_mask=int(full_units_mask),
    )


def _scm_stage3_coverage_summary(
    probe_subsets_by_var: Dict[str, List[Tuple[str, ...]]],
    coverage: Dict[Tuple[str, Tuple[str, ...]], int],
) -> Dict[str, Any]:
    total_patterns = 0
    seen_patterns = 0
    coverage_by_var: Dict[str, float] = {}
    seen_by_var: Dict[str, int] = {}
    total_by_var: Dict[str, int] = {}

    for var in sorted(probe_subsets_by_var.keys(), key=_natural_var_key):
        subsets = probe_subsets_by_var.get(var, [])
        var_total = 0
        var_seen = 0
        for subset in subsets:
            support = 1 << len(subset)
            key = (str(var), tuple(str(v) for v in subset))
            observed = int(int(coverage.get(key, 0)) & int(support - 1)).bit_count()
            var_total += int(support)
            var_seen += int(observed)
        total_patterns += int(var_total)
        seen_patterns += int(var_seen)
        total_by_var[str(var)] = int(var_total)
        seen_by_var[str(var)] = int(var_seen)
        coverage_by_var[str(var)] = (
            float(var_seen) / float(var_total)
            if var_total > 0
            else 1.0
        )

    coverage_ratio = (
        float(seen_patterns) / float(total_patterns)
        if total_patterns > 0
        else 1.0
    )
    return {
        "coverage_ratio": float(coverage_ratio),
        "seen_patterns": int(seen_patterns),
        "total_patterns": int(total_patterns),
        "coverage_by_var": coverage_by_var,
        "seen_patterns_by_var": seen_by_var,
        "total_patterns_by_var": total_by_var,
    }


def _scm_stage3_coverage_gain_for_rows(
    rows: List[Dict[str, Any]],
    *,
    targets: Set[str],
    probe_subsets_by_var: Dict[str, List[Tuple[str, ...]]],
    coverage: Dict[Tuple[str, Tuple[str, ...]], int],
) -> Tuple[int, Dict[Tuple[str, Tuple[str, ...]], int], Dict[str, int]]:
    total_gain = 0
    new_patterns_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    gain_by_var: Dict[str, int] = {}

    for var, subsets in probe_subsets_by_var.items():
        if str(var) in targets:
            continue
        var_gain = 0
        for subset in subsets:
            key = (str(var), tuple(str(v) for v in subset))
            support_mask = (1 << len(subset)) - 1
            already_mask = int(coverage.get(key, 0)) & int(support_mask)
            discovered_mask = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if any(s not in row for s in subset):
                    continue
                pattern_id = 0
                for idx, s in enumerate(subset):
                    if int(row[s]):
                        pattern_id |= (1 << idx)
                discovered_mask |= (1 << int(pattern_id))
                if int((already_mask | discovered_mask) & support_mask) == int(support_mask):
                    break
            discovered_mask = int(discovered_mask) & int(support_mask) & (~int(already_mask))
            if discovered_mask:
                new_patterns_by_key[key] = int(discovered_mask)
                gain = int(discovered_mask).bit_count()
                var_gain += int(gain)
                total_gain += int(gain)
        if var_gain > 0:
            gain_by_var[str(var)] = int(var_gain)
    return int(total_gain), new_patterns_by_key, gain_by_var


def _scm_stage3_coverage_gain_for_bitsets(
    *,
    var_bitsets: Dict[str, int],
    full_units_mask: int,
    targets: Set[str],
    probe_specs_by_var: Dict[str, List[Tuple[Tuple[str, Tuple[str, ...]], Tuple[str, ...], int]]],
    coverage: Dict[Tuple[str, Tuple[str, ...]], int],
) -> Tuple[int, Dict[Tuple[str, Tuple[str, ...]], int], Dict[str, int]]:
    total_gain = 0
    new_patterns_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    gain_by_var: Dict[str, int] = {}

    for var, specs in probe_specs_by_var.items():
        if str(var) in targets:
            continue
        var_gain = 0
        for key, subset, support_mask in specs:
            already_mask = int(coverage.get(key, 0)) & int(support_mask)
            if int(already_mask) == int(support_mask):
                continue
            dep_masks = tuple(
                int(var_bitsets.get(str(dep), 0)) & int(full_units_mask)
                for dep in subset
            )
            observed_mask = _scm_stage3_pattern_mask_from_dep_masks(
                dep_masks=dep_masks,
                full_units_mask=int(full_units_mask),
            )
            discovered_mask = int(observed_mask) & int(support_mask) & (~int(already_mask))
            if int(discovered_mask) == 0:
                continue
            new_patterns_by_key[key] = int(discovered_mask)
            gain = int(discovered_mask).bit_count()
            var_gain += int(gain)
            total_gain += int(gain)
        if int(var_gain) > 0:
            gain_by_var[str(var)] = int(var_gain)
    return int(total_gain), new_patterns_by_key, gain_by_var


def _scm_apply_stage3_coverage_gain(
    coverage: Dict[Tuple[str, Tuple[str, ...]], int],
    new_patterns_by_key: Dict[Tuple[str, Tuple[str, ...]], int],
) -> None:
    for key, new_mask in new_patterns_by_key.items():
        if int(new_mask) == 0:
            continue
        coverage[key] = int(coverage.get(key, 0)) | int(new_mask)


def _scm_collect_shortcut_witnesses(
    train_worlds: List[CausalWorldView],
    topological_order: List[str],
    endogenous_vars: List[str],
    mechanism_stats_by_var: Dict[str, Dict[str, Any]],
    allowed_ops: List[str],
    ast_cap_floor: int = 3,
    ast_cap_max: int = 6,
    check_vars: Optional[List[str]] = None,
    precomputed_rows_by_var: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    allow_constants: bool = True,
    max_predecessors_per_target: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    index_of = {v: i for i, v in enumerate(topological_order)}
    endogenous_set = set(endogenous_vars)
    vars_to_check = [v for v in (check_vars or endogenous_vars) if v in endogenous_set]
    witnesses: Dict[str, Dict[str, Any]] = {}

    rows_by_var: Dict[str, List[Dict[str, Any]]] = {}
    if precomputed_rows_by_var is not None:
        rows_by_var = precomputed_rows_by_var
    else:
        # Cache per-world targets/rows once for this call.
        payload_cache: Dict[int, Tuple[Set[str], List[Dict[str, Any]]]] = {}
        world_payloads: List[Tuple[Set[str], List[Dict[str, Any]]]] = [
            _scm_world_rows_and_intervened(world, payload_cache=payload_cache)
            for world in train_worlds
        ]
        rows_by_var = {v: [] for v in vars_to_check}
        for intervened, rows in world_payloads:
            _scm_extend_rows_by_var_cache(
                rows_by_var_cache=rows_by_var,
                vars_to_track=vars_to_check,
                intervened=intervened,
                rows=rows,
            )

    for var in vars_to_check:
        idx = index_of.get(var, -1)
        if idx <= 0:
            continue
        candidate_vars = _bounded_predecessors(
            topological_order[:idx],
            max_predecessors=max_predecessors_per_target,
        )
        if not candidate_vars:
            continue

        rows = rows_by_var.get(var, [])

        if len(rows) < 2:
            continue

        gold_ast = int((mechanism_stats_by_var.get(var) or {}).get("astSize", 1))
        # Shortcuts must be strictly smaller than gold AST.
        cap = min(int(ast_cap_max), int(gold_ast) - 1)
        if cap < int(ast_cap_floor):
            continue

        # Keep this feasible: skip very large predecessor sets.
        if len(candidate_vars) > 7:
            continue

        witness = _find_small_shortcut_witness(
            rows=rows,
            target_var=var,
            candidate_vars=candidate_vars,
            ast_cap=cap,
            allowed_operators=allowed_ops,
            allow_constants=allow_constants,
        )
        if witness is not None:
            witnesses[var] = {
                **witness,
                "cap": int(cap),
                "rows": int(len(rows)),
                "candidateVars": list(candidate_vars),
            }
    return witnesses


def _build_cind_problem_full_scm(
    seed: int,
    instance_id: str,
    task_name: str,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    rng = random.Random(seed)
    t0_total = time.perf_counter()
    difficulty_key = str(spec.get("difficulty", "")).strip().lower()

    n_requested = int(spec["n"])
    min_n = max(0, int(spec.get("scm_min_n", 0)))
    n = max(min_n, n_requested)
    k = int(spec["k"])
    m = int(spec["m"])
    heldout_k = int(spec["heldout_k"])
    debug_generation = bool(spec.get("debug_generation", False))
    debug_every = max(1, int(spec.get("debug_every", 10)))
    debug_world_attempts = bool(spec.get("debug_world_attempts", False))
    debug_sampling_details = bool(spec.get("debug_sampling_details", False))

    def _dbg(msg: str) -> None:
        if debug_generation:
            print(f"[A_SCM debug] {instance_id}: {msg}")

    _dbg(
        f"start difficulty={spec.get('difficulty')} n={n} (requested={n_requested}) "
        f"k={k} m={m} heldout_k={heldout_k} seed={seed}"
    )

    # A_SCM has no distinguished target variable; all observed variables are peers.
    all_vars = [f"X{i}" for i in range(1, n + 1)]
    input_vars = list(all_vars)
    if len(all_vars) < 3:
        raise RuntimeError("A_SCM generation requires at least 3 observed variables")

    slot_order = [f"A{i}" for i in range(1, n + 1)]
    max_root_count = max(1, len(slot_order) - 2)
    default_root_count_by_difficulty = {
        "easy": 2,
        "medium": 3,
        "hard": 3,
        "extreme": 4,
        "hard_small": 3,
        "very_hard_small": 3,
    }
    default_root_count = min(
        int(default_root_count_by_difficulty.get(difficulty_key, 3)),
        max_root_count,
    )
    requested_root_count = int(spec.get("scm_root_count", default_root_count))
    root_count = min(max(1, requested_root_count), max_root_count)
    root_slots = list(slot_order[:root_count])
    root_slot_set = set(root_slots)
    endogenous_slots = [v for v in slot_order if v not in root_slot_set]

    allowed_ops = list(DEFAULT_ALLOWED_OPERATORS)
    if bool(spec.get("allow_if", False)):
        allowed_ops.append("if")
    allow_constants = bool(spec.get("allow_constants", False))
    scm_max_predecessors_per_target = spec.get("scm_max_predecessors_per_target")
    if scm_max_predecessors_per_target is not None:
        scm_max_predecessors_per_target = max(1, int(scm_max_predecessors_per_target))

    # Keep A_SCM gold mechanisms compact; hardness should come from interventions/invariance.
    default_eq_ast_by_difficulty: Dict[str, Tuple[int, int]] = {
        "easy": (3, 6),
        "medium": (3, 7),
        "hard": (4, 8),
        "extreme": (5, 10),
        "hard_small": (4, 8),
        "very_hard_small": (4, 10),
    }
    default_eq_ast_min, default_eq_ast_max = default_eq_ast_by_difficulty.get(
        difficulty_key, (4, 10)
    )
    eq_ast_min = int(spec.get("scm_eq_ast_min", default_eq_ast_min))
    eq_ast_max = int(spec.get("scm_eq_ast_max", default_eq_ast_max))
    if eq_ast_max < eq_ast_min:
        eq_ast_max = eq_ast_min

    mechanism_by_var: Dict[str, str] = {}
    parent_by_var: Dict[str, List[str]] = {}
    mechanism_stats_by_var: Dict[str, Dict[str, Any]] = {}
    expanded_mechanism_stats_by_var: Dict[str, Dict[str, Any]] = {}
    gold_nodes_by_var: Dict[str, MechanismNode] = {}
    topological_order: List[str] = []
    root_vars: List[str] = []
    endogenous_vars: List[str] = []
    slot_to_obs: Dict[str, str] = {}

    scm_sample_attempts = 0
    scm_sample_max_attempts = 200
    scm_rejection_counts: Dict[str, int] = {}
    eq_attempts_used_by_slot: Dict[str, List[int]] = {v: [] for v in endogenous_slots}

    def _bump_scm_rejection(reason: str) -> None:
        scm_rejection_counts[reason] = scm_rejection_counts.get(reason, 0) + 1

    while scm_sample_attempts < scm_sample_max_attempts:
        scm_sample_attempts += 1
        failed = False
        failed_slot: Optional[str] = None

        local_nodes_by_slot: Dict[str, MechanismNode] = {}
        expanded_nodes_by_slot: Dict[str, MechanismNode] = {}
        expanded_stats_by_slot: Dict[str, Dict[str, Any]] = {}

        for slot in endogenous_slots:
            idx = slot_order.index(slot)
            predecessors = _bounded_predecessors(
                slot_order[:idx],
                max_predecessors=scm_max_predecessors_per_target,
            )
            if not predecessors:
                _bump_scm_rejection("var_no_predecessors")
                failed = True
                failed_slot = slot
                break

            parent_upper = min(max(1, int(spec.get("parent_max", 3))), len(predecessors))
            parent_lower = 2 if parent_upper >= 2 else 1
            sampled = False
            for local_attempt in range(1, 101):
                parent_count = rng.randint(parent_lower, parent_upper)
                selected_parents = _sorted_unique_natural(rng.sample(predecessors, parent_count))
                try:
                    expr, _stats = _sample_gold_mechanism(
                        rng,
                        parents=selected_parents,
                        ast_min=eq_ast_min,
                        ast_max=eq_ast_max,
                        allow_if=bool(spec.get("allow_if", False)),
                        allow_constants=allow_constants,
                        depth_target=spec.get("gold_depth_target"),
                        depth_sampler_mode=str(spec.get("gold_depth_sampler_mode", "canonical")),
                        require_all_parents_used=True,
                        require_each_parent_essential=True,
                        require_global_minimal_equiv=bool(
                            spec.get("scm_require_global_minimal_equiv", True)
                        ),
                        allow_fallback=True,
                    )
                    node = parse_mechanism(
                        expr,
                        allowed_operators=set(allowed_ops),
                        allowed_variables=set(predecessors),
                        allow_constants=allow_constants,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    _bump_scm_rejection("eq_sampler_runtime_error")
                    if "No canonical formulas found at requested depth" in msg:
                        _bump_scm_rejection("eq_no_formula_at_depth")
                    elif "No canonical formulas satisfy parent/essential constraints" in msg:
                        _bump_scm_rejection("eq_no_candidate_constraints")
                    elif "globally minimal in AST" in msg:
                        _bump_scm_rejection("eq_no_global_minimal_equiv")
                    else:
                        _bump_scm_rejection("eq_runtime_other")
                    continue
                except MechanismParseError:
                    _bump_scm_rejection("eq_parse_error")
                    continue

                node = _canonicalize_node(node)
                used = _sorted_unique_natural(list(mechanism_variables(node)))
                if not used:
                    _bump_scm_rejection("eq_empty_used_variables")
                    continue

                local_nodes_by_slot[slot] = node
                eq_attempts_used_by_slot.setdefault(slot, []).append(local_attempt)
                sampled = True
                break

            if not sampled:
                _bump_scm_rejection("var_unsampled_after_100")
                _bump_scm_rejection(f"var_unsampled_{slot}")
                failed = True
                failed_slot = slot
                break

        if not failed:
            expanded_memo: Dict[str, MechanismNode] = {}
            for slot in endogenous_slots:
                expanded = _canonicalize_node(
                    _expand_scm_node(slot, local_nodes_by_slot, root_slot_set, expanded_memo)
                )
                expanded_nodes_by_slot[slot] = expanded
                expanded_stats_by_slot[slot] = analyze_mechanism(expanded)

            if spec.get("gold_depth_target") is not None:
                target_full_depth = max(1, int(spec.get("gold_depth_target")))
                eligible_for_target = [
                    slot
                    for slot in endogenous_slots
                    if int((expanded_stats_by_slot.get(slot) or {}).get("maxDepth", 0)) == target_full_depth
                ]
                if not eligible_for_target:
                    _bump_scm_rejection("no_slot_at_full_depth_target")
                    failed = True
            else:
                eligible_for_target = list(endogenous_slots)

        if not failed:
            if not eligible_for_target:
                _bump_scm_rejection("empty_depth_target_slot_pool")
                failed = True
            else:
                shuffled_observed = list(all_vars)
                rng.shuffle(shuffled_observed)
                slot_to_obs = {slot: observed for slot, observed in zip(slot_order, shuffled_observed)}

                if len(slot_to_obs) != len(slot_order):
                    _bump_scm_rejection("overlay_mapping_incomplete")
                    failed = True

        if not failed:
            topological_order = [slot_to_obs[s] for s in slot_order]
            root_vars = [slot_to_obs[s] for s in root_slots]
            endogenous_vars = [slot_to_obs[s] for s in endogenous_slots]

            if topological_order == all_vars:
                _bump_scm_rejection("identity_overlay_order")
                failed = True

        if not failed:
            mechanism_by_var = {}
            parent_by_var = {}
            mechanism_stats_by_var = {}
            expanded_mechanism_stats_by_var = {}
            gold_nodes_by_var = {}

            for slot in endogenous_slots:
                observed_var = slot_to_obs[slot]
                local_node_obs = _canonicalize_node(
                    _rename_node_variables(local_nodes_by_slot[slot], slot_to_obs)
                )
                mechanism_by_var[observed_var] = node_to_sexpr(local_node_obs)
                parent_by_var[observed_var] = _sorted_unique_natural(
                    list(mechanism_variables(local_node_obs))
                )
                mechanism_stats_by_var[observed_var] = analyze_mechanism(local_node_obs)
                gold_nodes_by_var[observed_var] = local_node_obs

                expanded_node_obs = _canonicalize_node(
                    _rename_node_variables(expanded_nodes_by_slot[slot], slot_to_obs)
                )
                expanded_mechanism_stats_by_var[observed_var] = analyze_mechanism(expanded_node_obs)

        if failed:
            _bump_scm_rejection("scm_attempt_failed")
            if debug_generation and (
                scm_sample_attempts == 1 or scm_sample_attempts % debug_every == 0
            ):
                top = sorted(scm_rejection_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
                top_txt = ", ".join(f"{k}={v}" for k, v in top)
                _dbg(
                    f"scm attempt {scm_sample_attempts}/{scm_sample_max_attempts} failed "
                    f"(failed_slot={failed_slot or 'unknown'}). top_reasons: {top_txt}"
                )
            continue

        if len(mechanism_by_var) == len(endogenous_vars):
            break

    if len(mechanism_by_var) != len(endogenous_vars):
        top_reasons = sorted(scm_rejection_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        reason_txt = ", ".join(f"{name}={count}" for name, count in top_reasons)
        if not reason_txt:
            reason_txt = "none_captured"
        raise RuntimeError(
            f"Failed to sample full SCM mechanisms after {scm_sample_max_attempts} attempts "
            f"for instance {instance_id} (task={task_name}, seed={seed}). "
            f"Top rejection reasons: {reason_txt}."
        )

    if debug_generation and debug_sampling_details:
        sampled_summaries: List[str] = []
        for obs_var in endogenous_vars:
            slot = next((s for s in endogenous_slots if slot_to_obs.get(s) == obs_var), None)
            attempts = (eq_attempts_used_by_slot.get(slot) or []) if slot is not None else []
            if attempts:
                sampled_summaries.append(f"{obs_var}:attempts={attempts[-1]}")
        _dbg(
            "scm mechanisms sampled successfully after "
            f"{scm_sample_attempts} outer attempts; per-var last attempts: "
            + ", ".join(sampled_summaries)
        )
        _dbg(f"scm sampling stage time={time.perf_counter() - t0_total:.2f}s")

    units = [f"u{i:02d}" for i in range(m)]
    intervention_size_probs = spec.get("intervention_size_probs")
    intervention_mode_probs = spec.get("scm_intervention_mode_probs") or {
        "hard_constant": 0.5,
        "hard_assigned": 0.5,
    }
    env_levels = spec.get("scm_env_levels")
    use_scm_cegis_lite = bool(spec.get("use_scm_cegis_lite", True))
    shortcut_ast_cap_raw = spec.get("shortcut_ast_cap")
    if shortcut_ast_cap_raw is None:
        shortcut_ast_cap_raw = spec.get("scm_shortcut_ast_cap")
    if shortcut_ast_cap_raw is None and difficulty_key in {"hard_small", "very_hard_small"}:
        shortcut_ast_cap_raw = 4
    shortcut_ast_cap_floor = max(2, int(spec.get("scm_shortcut_ast_cap_floor", 3)))
    shortcut_ast_cap_max = (
        None
        if shortcut_ast_cap_raw is None
        else max(shortcut_ast_cap_floor, int(shortcut_ast_cap_raw))
    )
    cegis_candidate_world_budget = max(
        5,
        int(spec.get("cegis_candidate_world_budget", spec.get("cegis_candidate_interventions", 30))),
    )
    cegis_max_iters = max(1, int(spec.get("cegis_max_iters", max(2, k))))
    scm_cegis_seed_worlds = min(int(k), max(2, int(spec.get("scm_cegis_seed_worlds", 3))))
    scm_cegis_restarts = max(1, int(spec.get("scm_cegis_restarts", 1)))
    use_scm_stage3_lite = bool(spec.get("use_scm_stage3_lite", True))
    scm_stage3_probe_size = max(1, int(spec.get("scm_stage3_probe_size", 3)))
    scm_stage3_probe_subsets_per_var = max(1, int(spec.get("scm_stage3_probe_subsets_per_var", 2)))
    scm_stage3_world_budget = max(0, int(spec.get("scm_stage3_world_budget", 1)))
    scm_stage3_candidate_world_budget = max(
        4,
        int(spec.get("scm_stage3_candidate_world_budget", max(8, cegis_candidate_world_budget))),
    )
    scm_stage3_assigned_bias = min(
        max(float(spec.get("scm_stage3_assigned_bias", 0.8)), 0.0),
        1.0,
    )
    scm_cegis_adaptive_candidate_budget = bool(
        spec.get("scm_cegis_adaptive_candidate_budget", True)
    )
    scm_cegis_adaptive_min_candidates = max(
        8,
        int(spec.get("scm_cegis_adaptive_min_candidates", 32)),
    )
    scm_cegis_early_stop_on_quality_met = bool(
        spec.get("scm_cegis_early_stop_on_quality_met", True)
    )
    scm_target_survivors_small_max = spec.get("scm_target_survivors_small_max")
    if scm_target_survivors_small_max is not None:
        scm_target_survivors_small_max = max(0, int(scm_target_survivors_small_max))
    scm_min_survivor_reduction_frac = spec.get("scm_min_survivor_reduction_frac")
    if scm_min_survivor_reduction_frac is not None:
        scm_min_survivor_reduction_frac = min(max(float(scm_min_survivor_reduction_frac), 0.0), 1.0)
    scm_quality_retry_budget = max(1, int(spec.get("scm_quality_retry_budget", 40)))
    max_attempts = 160
    world_attempts_used = 0
    rejection_reason_counts: Dict[str, int] = {}
    train_worlds: List[CausalWorldView] = []
    heldout_worlds: List[CausalWorldView] = []
    scoring_coverage_diag_final: Dict[str, Any] = {}
    heldout_plan_diag_final: Dict[str, Any] = {}
    local_shortcut_diag_final: Dict[str, Dict[str, Any]] = {}
    accepted = False
    scm_cegis_diag_final: Dict[str, Any] = {
        "attempts": 0,
        "cegis_iters": 0,
        "shortcuts_killed_by_var": {v: 0 for v in endogenous_vars},
        "kills_per_iter": [],
        "total_shortcuts_killed": 0,
        "vars_with_kills": [],
        "kill_coverage_ratio": 0.0,
        "initial_survivors_small_estimate": 0,
        "survivors_small_estimate": 0,
        "survivor_reduction_abs": 0,
        "survivor_reduction_ratio": 0.0,
        "initial_survivor_vars": [],
        "survivor_vars": [],
        "survivor_count_history": [],
        "initial_witness_ast_by_var": {},
        "final_witness_ast_by_var": {},
        "initial_witness_cap_by_var": {},
        "final_witness_cap_by_var": {},
        "initial_shortcut_ast_gap_by_var": {},
        "final_shortcut_ast_gap_by_var": {},
        "initial_shortcut_ast_ratio_by_var": {},
        "final_shortcut_ast_ratio_by_var": {},
        "final_shortcut_gap_sum": 0,
        "final_shortcut_ratio_mean": 1.0,
        "resolved_all_shortcuts_under_cap": False,
        "cegis_restart_count": int(scm_cegis_restarts),
        "cegis_selected_restart_index": 0,
        "cegis_restart_summaries": [],
        "cegis_restart_early_stopped": False,
        "stage3_lite_enabled": bool(use_scm_stage3_lite),
        "stage3_lite_probe_size": int(scm_stage3_probe_size),
        "stage3_lite_probe_subsets_per_var": int(scm_stage3_probe_subsets_per_var),
        "stage3_lite_world_budget": int(scm_stage3_world_budget),
        "stage3_lite_worlds_added": 0,
        "stage3_lite_candidate_plan_evals": 0,
        "stage3_lite_assigned_bias": float(scm_stage3_assigned_bias),
        "stage3_lite_total_new_patterns": 0,
        "stage3_lite_initial_patterns_seen": 0,
        "stage3_lite_final_patterns_seen": 0,
        "stage3_lite_total_patterns": 0,
        "stage3_lite_initial_coverage_ratio": 1.0,
        "stage3_lite_final_coverage_ratio": 1.0,
        "stage3_lite_coverage_gain_ratio": 0.0,
        "stage3_lite_initial_coverage_by_var": {},
        "stage3_lite_final_coverage_by_var": {},
        "stage3_lite_coverage_history": [],
        "stage3_lite_kills_by_var": {v: 0 for v in endogenous_vars},
    }
    mechanism_lookup_tables = _build_scm_mechanism_lookup_tables(
        topological_order=topological_order,
        root_vars=root_vars,
        gold_nodes_by_var=gold_nodes_by_var,
    )

    def _bump_rejection(reason: str) -> None:
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

    for attempt in range(1, max_attempts + 1):
        world_attempts_used = attempt
        t0_world_attempt = time.perf_counter()
        should_log_world_attempt = bool(
            debug_generation
            and debug_world_attempts
            and (
                attempt == 1
                or (attempt % debug_every == 0)
                or attempt == max_attempts
            )
        )
        unit_root_thresholds = _sample_root_thresholds(rng, units, root_vars)
        seen_interventions: Set[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]] = set()
        scm_cegis_diag_iter: Dict[str, Any] = {
            "attempts": 0,
            "cegis_iters": 0,
            "shortcuts_killed_by_var": {v: 0 for v in endogenous_vars},
            "kills_per_iter": [],
            "total_shortcuts_killed": 0,
            "vars_with_kills": [],
            "kill_coverage_ratio": 0.0,
            "initial_survivors_small_estimate": 0,
            "survivors_small_estimate": 0,
            "survivor_reduction_abs": 0,
            "survivor_reduction_ratio": 0.0,
            "initial_survivor_vars": [],
            "survivor_vars": [],
            "survivor_count_history": [],
            "initial_witness_ast_by_var": {},
            "final_witness_ast_by_var": {},
            "initial_witness_cap_by_var": {},
            "final_witness_cap_by_var": {},
            "initial_shortcut_ast_gap_by_var": {},
            "final_shortcut_ast_gap_by_var": {},
            "initial_shortcut_ast_ratio_by_var": {},
            "final_shortcut_ast_ratio_by_var": {},
            "final_shortcut_gap_sum": 0,
            "final_shortcut_ratio_mean": 1.0,
            "resolved_all_shortcuts_under_cap": False,
            "cegis_restart_count": int(scm_cegis_restarts),
            "cegis_selected_restart_index": 0,
            "cegis_restart_summaries": [],
            "cegis_restart_early_stopped": False,
            "stage3_lite_enabled": bool(use_scm_stage3_lite),
            "stage3_lite_probe_size": int(scm_stage3_probe_size),
            "stage3_lite_probe_subsets_per_var": int(scm_stage3_probe_subsets_per_var),
            "stage3_lite_world_budget": int(scm_stage3_world_budget),
            "stage3_lite_worlds_added": 0,
            "stage3_lite_candidate_plan_evals": 0,
            "stage3_lite_assigned_bias": float(scm_stage3_assigned_bias),
            "stage3_lite_total_new_patterns": 0,
            "stage3_lite_initial_patterns_seen": 0,
            "stage3_lite_final_patterns_seen": 0,
            "stage3_lite_total_patterns": 0,
            "stage3_lite_initial_coverage_ratio": 1.0,
            "stage3_lite_final_coverage_ratio": 1.0,
            "stage3_lite_coverage_gain_ratio": 0.0,
            "stage3_lite_initial_coverage_by_var": {},
            "stage3_lite_final_coverage_by_var": {},
            "stage3_lite_coverage_history": [],
            "stage3_lite_kills_by_var": {v: 0 for v in endogenous_vars},
        }
        if use_scm_cegis_lite:
            (
                train_worlds,
                seen_interventions,
                scm_cegis_diag_iter,
            ) = _build_train_worlds_scm_cegis_lite(
                rng=rng,
                units=units,
                variables=all_vars,
                topological_order=topological_order,
                root_vars=root_vars,
                endogenous_vars=endogenous_vars,
                unit_root_thresholds=unit_root_thresholds,
                gold_nodes_by_var=gold_nodes_by_var,
                mechanism_stats_by_var=mechanism_stats_by_var,
                k=k,
                intervention_size_probs=intervention_size_probs,
                intervention_mode_probs=intervention_mode_probs,
                env_levels=env_levels,
                allowed_ops=allowed_ops,
                shortcut_ast_cap_floor=shortcut_ast_cap_floor,
                shortcut_ast_cap_max=shortcut_ast_cap_max,
                candidate_world_budget=cegis_candidate_world_budget,
                max_iters=cegis_max_iters,
                restarts=scm_cegis_restarts,
                seed_target_worlds=scm_cegis_seed_worlds,
                allow_constants=allow_constants,
                max_predecessors_per_target=scm_max_predecessors_per_target,
                use_stage3_lite=use_scm_stage3_lite,
                stage3_probe_size=scm_stage3_probe_size,
                stage3_probe_subsets_per_var=scm_stage3_probe_subsets_per_var,
                stage3_world_budget=scm_stage3_world_budget,
                stage3_candidate_world_budget=scm_stage3_candidate_world_budget,
                stage3_assigned_bias=scm_stage3_assigned_bias,
                adaptive_candidate_budget=scm_cegis_adaptive_candidate_budget,
                adaptive_candidate_min_candidates=scm_cegis_adaptive_min_candidates,
                early_stop_on_quality_met=scm_cegis_early_stop_on_quality_met,
                target_survivors_small_max=scm_target_survivors_small_max,
                min_survivor_reduction_frac=scm_min_survivor_reduction_frac,
                mechanism_lookup_tables=mechanism_lookup_tables,
            )
            if len(train_worlds) < k:
                _bump_rejection("scm_cegis_short_train_worlds")
                if should_log_world_attempt:
                    _dbg(
                        f"world attempt {attempt}/{max_attempts} rejected at cegis build: "
                        f"scm_cegis_short_train_worlds (elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                    )
                continue
            if (
                should_log_world_attempt
                and int(scm_cegis_diag_iter.get("survivors_small_estimate", 0)) > 0
            ):
                _dbg(
                    f"world attempt {attempt}/{max_attempts} cegis reached budget with survivors="
                    f"{int(scm_cegis_diag_iter.get('survivors_small_estimate', 0))} "
                    f"(elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                )
            initial_survivors_iter = int(
                scm_cegis_diag_iter.get("initial_survivors_small_estimate", 0)
            )
            final_survivors_iter = int(scm_cegis_diag_iter.get("survivors_small_estimate", 0))
            reduction_abs_iter = int(
                scm_cegis_diag_iter.get(
                    "survivor_reduction_abs",
                    max(0, initial_survivors_iter - final_survivors_iter),
                )
            )
            reduction_frac_iter = (
                float(reduction_abs_iter) / float(initial_survivors_iter)
                if initial_survivors_iter > 0
                else 1.0
            )
            effective_min_reduction_frac_iter = (
                None
                if scm_min_survivor_reduction_frac is None
                else float(scm_min_survivor_reduction_frac)
            )
            if (
                effective_min_reduction_frac_iter is not None
                and initial_survivors_iter > 0
                and scm_target_survivors_small_max is not None
            ):
                # If a hard survivor target is set, clamp reduction requirement to what can be
                # achieved when that survivor target is met for this attempt.
                max_reachable_frac = max(
                    0.0,
                    min(
                        1.0,
                        float(initial_survivors_iter - int(scm_target_survivors_small_max))
                        / float(initial_survivors_iter),
                    ),
                )
                effective_min_reduction_frac_iter = min(
                    float(effective_min_reduction_frac_iter),
                    float(max_reachable_frac),
                )
            if (
                scm_target_survivors_small_max is not None
                and final_survivors_iter > int(scm_target_survivors_small_max)
            ):
                _bump_rejection("scm_cegis_survivors_above_target")
                if should_log_world_attempt:
                    _dbg(
                        f"world attempt {attempt}/{max_attempts} rejected at cegis quality: "
                        f"scm_cegis_survivors_above_target "
                        f"(final={final_survivors_iter}, target={int(scm_target_survivors_small_max)}, "
                        f"elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                    )
                quality_rejections = int(rejection_reason_counts.get("scm_cegis_survivors_above_target", 0)) + int(
                    rejection_reason_counts.get("scm_cegis_reduction_below_target", 0)
                )
                if quality_rejections >= int(scm_quality_retry_budget):
                    if should_log_world_attempt:
                        _dbg(
                            f"world attempt {attempt}/{max_attempts} terminating early after "
                            f"{quality_rejections} cegis-quality rejections "
                            f"(budget={int(scm_quality_retry_budget)})"
                        )
                    break
                continue
            if (
                effective_min_reduction_frac_iter is not None
                and initial_survivors_iter > 0
                and reduction_frac_iter < float(effective_min_reduction_frac_iter)
            ):
                _bump_rejection("scm_cegis_reduction_below_target")
                if should_log_world_attempt:
                    _dbg(
                        f"world attempt {attempt}/{max_attempts} rejected at cegis quality: "
                        f"scm_cegis_reduction_below_target "
                        f"(frac={reduction_frac_iter:.3f}, "
                        f"target={float(effective_min_reduction_frac_iter):.3f}, "
                        f"elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                    )
                quality_rejections = int(rejection_reason_counts.get("scm_cegis_survivors_above_target", 0)) + int(
                    rejection_reason_counts.get("scm_cegis_reduction_below_target", 0)
                )
                if quality_rejections >= int(scm_quality_retry_budget):
                    if should_log_world_attempt:
                        _dbg(
                            f"world attempt {attempt}/{max_attempts} terminating early after "
                            f"{quality_rejections} cegis-quality rejections "
                            f"(budget={int(scm_quality_retry_budget)})"
                        )
                    break
                continue
        else:
            baseline_env_profile = {rv: 0.5 for rv in root_vars}
            baseline_root_contexts = _materialize_root_contexts(
                units=units,
                root_vars=root_vars,
                unit_thresholds=unit_root_thresholds,
                env_profile=baseline_env_profile,
            )

            train_worlds = []
            baseline = {}
            baseline_targets: List[str] = []
            baseline_mode = "hard_constant"
            train_worlds.append(
                _make_world_full_scm(
                    world_id="train_00",
                    split="train",
                    units=units,
                    variables=all_vars,
                    topological_order=topological_order,
                    root_vars=root_vars,
                    base_root_contexts=baseline_root_contexts,
                    interventions=baseline,
                    gold_nodes_by_var=gold_nodes_by_var,
                    intervention_mode=baseline_mode,
                    intervention_targets=baseline_targets,
                    assigned_values_by_unit=None,
                    env_profile=baseline_env_profile,
                    mechanism_lookup_tables=mechanism_lookup_tables,
                )
            )
            seen_interventions.add(
                _intervention_plan_key(
                    mode=baseline_mode,
                    assignments=baseline,
                    targets=baseline_targets,
                )
            )

            force_queue = list(endogenous_vars)
            rng.shuffle(force_queue)
            while len(train_worlds) < k:
                force_var = force_queue.pop() if force_queue else None
                sampled_ints = _sample_intervention(
                    rng,
                    all_vars,
                    force_var=force_var,
                    min_targets=1,
                    max_targets=2 if rng.random() < 0.8 else 3,
                    size_probs=intervention_size_probs,
                )
                targets = sorted(sampled_ints.keys(), key=_natural_var_key)
                mode = _sample_intervention_mode(rng, intervention_mode_probs)
                if mode == "hard_assigned":
                    ints: Dict[str, int] = {}
                    assigned_by_unit = _sample_assigned_values_by_unit(
                        rng,
                        units=units,
                        targets=targets,
                        base_bias=float(rng.choice([0.3, 0.5, 0.7])),
                    )
                else:
                    ints = dict(sampled_ints)
                    assigned_by_unit = None

                key = _intervention_plan_key(mode=mode, assignments=ints, targets=targets)
                if key in seen_interventions:
                    continue
                seen_interventions.add(key)

                env_profile = _sample_world_env_profile(rng, root_vars, levels=env_levels)
                world_root_contexts = _materialize_root_contexts(
                    units=units,
                    root_vars=root_vars,
                    unit_thresholds=unit_root_thresholds,
                    env_profile=env_profile,
                )
                train_worlds.append(
                    _make_world_full_scm(
                        world_id=f"train_{len(train_worlds):02d}",
                        split="train",
                        units=units,
                        variables=all_vars,
                        topological_order=topological_order,
                        root_vars=root_vars,
                        base_root_contexts=world_root_contexts,
                        interventions=ints,
                        gold_nodes_by_var=gold_nodes_by_var,
                        intervention_mode=mode,
                            intervention_targets=targets,
                            assigned_values_by_unit=assigned_by_unit,
                            env_profile=env_profile,
                            mechanism_lookup_tables=mechanism_lookup_tables,
                        )
                    )

        failure_reason = _scm_training_signal_failure_reason(train_worlds, endogenous_vars)
        if failure_reason is not None:
            _bump_rejection(failure_reason)
            if should_log_world_attempt:
                _dbg(
                    f"world attempt {attempt}/{max_attempts} rejected at training-signal check: "
                    f"{failure_reason} (elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                )
            continue

        train_coverage_diag = _scm_build_scoring_coverage_diagnostics(
            train_worlds=train_worlds,
            heldout_worlds=[],
            endogenous_vars=endogenous_vars,
        )
        coverage_failure_reason = _scm_training_coverage_failure_reason(
            coverage_diag=train_coverage_diag,
            spec=spec,
        )
        if coverage_failure_reason is not None:
            _bump_rejection(coverage_failure_reason)
            if should_log_world_attempt:
                _dbg(
                    f"world attempt {attempt}/{max_attempts} rejected at coverage check: "
                    f"{coverage_failure_reason} (elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                )
            continue

        if shortcut_ast_cap_max is not None and not use_scm_cegis_lite:
            shortcut_reason = _scm_small_shortcut_failure_reason(
                train_worlds=train_worlds,
                topological_order=topological_order,
                endogenous_vars=endogenous_vars,
                mechanism_stats_by_var=mechanism_stats_by_var,
                allowed_ops=allowed_ops,
                ast_cap_floor=shortcut_ast_cap_floor,
                ast_cap_max=shortcut_ast_cap_max,
                check_vars=endogenous_vars,
                allow_constants=allow_constants,
                max_predecessors_per_target=scm_max_predecessors_per_target,
            )
            if shortcut_reason is not None:
                _bump_rejection(shortcut_reason)
                if should_log_world_attempt:
                    _dbg(
                        f"world attempt {attempt}/{max_attempts} rejected at shortcut check: "
                        f"{shortcut_reason} (elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                    )
                continue

        (
            heldout_worlds,
            scoring_coverage_diag_iter,
            heldout_plan_diag_iter,
            heldout_failure_reason,
        ) = _sample_heldout_worlds_full_scm(
            rng,
            heldout_k=heldout_k,
            train_worlds=train_worlds,
            seen_interventions=seen_interventions,
            units=units,
            variables=all_vars,
            topological_order=topological_order,
            root_vars=root_vars,
            gold_nodes_by_var=gold_nodes_by_var,
            unit_root_thresholds=unit_root_thresholds,
            endogenous_vars=endogenous_vars,
            intervention_mode_probs=intervention_mode_probs,
            intervention_size_probs=intervention_size_probs,
            env_levels=env_levels,
            mechanism_lookup_tables=mechanism_lookup_tables,
            spec=spec,
        )
        if heldout_failure_reason is not None:
            _bump_rejection(heldout_failure_reason)
            if should_log_world_attempt:
                _dbg(
                    f"world attempt {attempt}/{max_attempts} rejected at heldout check: "
                    f"{heldout_failure_reason} (elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
                )
            continue

        local_shortcut_diag_iter = _scm_build_local_shortcut_diagnostics(
            train_worlds=train_worlds,
            heldout_worlds=heldout_worlds,
            topological_order=topological_order,
            endogenous_vars=endogenous_vars,
            mechanism_stats_by_var=mechanism_stats_by_var,
            allowed_ops=allowed_ops,
            ast_cap_floor=shortcut_ast_cap_floor,
            ast_cap_max=shortcut_ast_cap_max,
            allow_constants=allow_constants,
            max_predecessors_per_target=scm_max_predecessors_per_target,
        )

        accepted = True
        scoring_coverage_diag_final = dict(scoring_coverage_diag_iter)
        heldout_plan_diag_final = dict(heldout_plan_diag_iter)
        local_shortcut_diag_final = dict(local_shortcut_diag_iter)
        if use_scm_cegis_lite:
            scm_cegis_diag_final = {
                "attempts": int(scm_cegis_diag_iter.get("attempts", 0)),
                "cegis_iters": int(scm_cegis_diag_iter.get("cegis_iters", 0)),
                "shortcuts_killed_by_var": dict(scm_cegis_diag_iter.get("shortcuts_killed_by_var", {})),
                "kills_per_iter": [
                    int(v) for v in (scm_cegis_diag_iter.get("kills_per_iter") or []) if v is not None
                ],
                "total_shortcuts_killed": int(scm_cegis_diag_iter.get("total_shortcuts_killed", 0)),
                "vars_with_kills": list(scm_cegis_diag_iter.get("vars_with_kills", [])),
                "kill_coverage_ratio": float(scm_cegis_diag_iter.get("kill_coverage_ratio", 0.0)),
                "initial_survivors_small_estimate": int(
                    scm_cegis_diag_iter.get("initial_survivors_small_estimate", 0)
                ),
                "survivors_small_estimate": int(scm_cegis_diag_iter.get("survivors_small_estimate", 0)),
                "survivor_reduction_abs": int(scm_cegis_diag_iter.get("survivor_reduction_abs", 0)),
                "survivor_reduction_ratio": float(scm_cegis_diag_iter.get("survivor_reduction_ratio", 0.0)),
                "initial_survivor_vars": list(scm_cegis_diag_iter.get("initial_survivor_vars", [])),
                "survivor_vars": list(scm_cegis_diag_iter.get("survivor_vars", [])),
                "survivor_count_history": [
                    int(v)
                    for v in (scm_cegis_diag_iter.get("survivor_count_history") or [])
                    if v is not None
                ],
                "initial_witness_ast_by_var": dict(
                    scm_cegis_diag_iter.get("initial_witness_ast_by_var", {})
                ),
                "final_witness_ast_by_var": dict(
                    scm_cegis_diag_iter.get("final_witness_ast_by_var", {})
                ),
                "initial_witness_cap_by_var": dict(
                    scm_cegis_diag_iter.get("initial_witness_cap_by_var", {})
                ),
                "final_witness_cap_by_var": dict(
                    scm_cegis_diag_iter.get("final_witness_cap_by_var", {})
                ),
                "initial_shortcut_ast_gap_by_var": dict(
                    scm_cegis_diag_iter.get("initial_shortcut_ast_gap_by_var", {})
                ),
                "final_shortcut_ast_gap_by_var": dict(
                    scm_cegis_diag_iter.get("final_shortcut_ast_gap_by_var", {})
                ),
                "initial_shortcut_ast_ratio_by_var": dict(
                    scm_cegis_diag_iter.get("initial_shortcut_ast_ratio_by_var", {})
                ),
                "final_shortcut_ast_ratio_by_var": dict(
                    scm_cegis_diag_iter.get("final_shortcut_ast_ratio_by_var", {})
                ),
                "final_shortcut_gap_sum": int(scm_cegis_diag_iter.get("final_shortcut_gap_sum", 0)),
                "final_shortcut_ratio_mean": float(
                    scm_cegis_diag_iter.get("final_shortcut_ratio_mean", 1.0)
                ),
                "resolved_all_shortcuts_under_cap": bool(
                    scm_cegis_diag_iter.get("resolved_all_shortcuts_under_cap", False)
                ),
                "cegis_restart_count": int(scm_cegis_diag_iter.get("cegis_restart_count", 1)),
                "cegis_selected_restart_index": int(
                    scm_cegis_diag_iter.get("cegis_selected_restart_index", 0)
                ),
                "cegis_restart_summaries": list(
                    scm_cegis_diag_iter.get("cegis_restart_summaries", [])
                ),
                "cegis_restart_early_stopped": bool(
                    scm_cegis_diag_iter.get("cegis_restart_early_stopped", False)
                ),
                "stage3_lite_enabled": bool(scm_cegis_diag_iter.get("stage3_lite_enabled", False)),
                "stage3_lite_probe_size": int(scm_cegis_diag_iter.get("stage3_lite_probe_size", 3)),
                "stage3_lite_probe_subsets_per_var": int(
                    scm_cegis_diag_iter.get("stage3_lite_probe_subsets_per_var", 2)
                ),
                "stage3_lite_world_budget": int(scm_cegis_diag_iter.get("stage3_lite_world_budget", 0)),
                "stage3_lite_worlds_added": int(scm_cegis_diag_iter.get("stage3_lite_worlds_added", 0)),
                "stage3_lite_candidate_plan_evals": int(
                    scm_cegis_diag_iter.get("stage3_lite_candidate_plan_evals", 0)
                ),
                "stage3_lite_assigned_bias": float(
                    scm_cegis_diag_iter.get("stage3_lite_assigned_bias", 0.8)
                ),
                "stage3_lite_total_new_patterns": int(
                    scm_cegis_diag_iter.get("stage3_lite_total_new_patterns", 0)
                ),
                "stage3_lite_initial_patterns_seen": int(
                    scm_cegis_diag_iter.get("stage3_lite_initial_patterns_seen", 0)
                ),
                "stage3_lite_final_patterns_seen": int(
                    scm_cegis_diag_iter.get("stage3_lite_final_patterns_seen", 0)
                ),
                "stage3_lite_total_patterns": int(scm_cegis_diag_iter.get("stage3_lite_total_patterns", 0)),
                "stage3_lite_initial_coverage_ratio": float(
                    scm_cegis_diag_iter.get("stage3_lite_initial_coverage_ratio", 1.0)
                ),
                "stage3_lite_final_coverage_ratio": float(
                    scm_cegis_diag_iter.get("stage3_lite_final_coverage_ratio", 1.0)
                ),
                "stage3_lite_coverage_gain_ratio": float(
                    scm_cegis_diag_iter.get("stage3_lite_coverage_gain_ratio", 0.0)
                ),
                "stage3_lite_initial_coverage_by_var": dict(
                    scm_cegis_diag_iter.get("stage3_lite_initial_coverage_by_var", {})
                ),
                "stage3_lite_final_coverage_by_var": dict(
                    scm_cegis_diag_iter.get("stage3_lite_final_coverage_by_var", {})
                ),
                "stage3_lite_coverage_history": [
                    float(v)
                    for v in (scm_cegis_diag_iter.get("stage3_lite_coverage_history") or [])
                    if v is not None
                ],
                "stage3_lite_kills_by_var": dict(
                    scm_cegis_diag_iter.get("stage3_lite_kills_by_var", {})
                ),
            }
        if should_log_world_attempt:
            _dbg(
                f"world attempt {attempt}/{max_attempts} accepted "
                f"(elapsed={time.perf_counter() - t0_world_attempt:.2f}s)"
            )
        break

    if not accepted:
        top_reasons = sorted(rejection_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        reason_txt = ", ".join(f"{name}={count}" for name, count in top_reasons)
        if not reason_txt:
            reason_txt = "none_captured"
        attempts_used_txt = int(max(1, world_attempts_used))
        raise RuntimeError(
            f"Failed to generate non-degenerate full-SCM worlds after {attempts_used_txt} attempts "
            f"for instance {instance_id} (task={task_name}, seed={seed}). "
            f"Top rejection reasons: {reason_txt}."
        )

    if debug_generation:
        _dbg(
            f"generation complete: scm_outer_attempts={scm_sample_attempts}, "
            f"world_attempts={world_attempts_used}, total_elapsed={time.perf_counter() - t0_total:.2f}s"
        )

    world_ids_train = [w.worldId for w in train_worlds]
    world_ids_heldout = [w.worldId for w in heldout_worlds]
    mode_counts_train: Dict[str, int] = {}
    mode_counts_heldout: Dict[str, int] = {}
    for w in train_worlds:
        mode = _extract_world_intervention_mode(w)
        mode_counts_train[mode] = mode_counts_train.get(mode, 0) + 1
    for w in heldout_worlds:
        mode = _extract_world_intervention_mode(w)
        mode_counts_heldout[mode] = mode_counts_heldout.get(mode, 0) + 1

    total_gold_ast = int(
        sum(int((mechanism_stats_by_var[v] or {}).get("astSize", 0)) for v in endogenous_vars)
    )
    total_expanded_gold_ast = int(
        sum(int((expanded_mechanism_stats_by_var[v] or {}).get("astSize", 0)) for v in endogenous_vars)
    )
    total_gold_edges = int(
        sum(int((mechanism_stats_by_var[v] or {}).get("parentCount", 0)) for v in endogenous_vars)
    )
    scoring_coverage_summary = dict(scoring_coverage_diag_final.get("summary") or {})
    heldout_plan_summary = dict(heldout_plan_diag_final.get("summary") or {})
    scm_prompt_variant = _normalize_scm_prompt_variant(spec.get("scm_prompt_variant", SCM_PROMPT_VARIANT_ORDERED))
    topological_layers = _compute_topological_layers(
        topological_order=topological_order,
        parents_by_var=parent_by_var,
    )
    prompt_variable_order = list(all_vars)
    if scm_prompt_variant == SCM_PROMPT_VARIANT_NTOPO:
        prompt_order_rng = random.Random((int(seed) * 1000003) ^ (len(all_vars) * 7919) ^ 0x5A17)
        prompt_order_rng.shuffle(prompt_variable_order)
        if len(prompt_variable_order) > 1 and prompt_variable_order == topological_order:
            prompt_variable_order = prompt_variable_order[1:] + prompt_variable_order[:1]
    elif scm_prompt_variant == SCM_PROMPT_VARIANT_PARTIAL:
        prompt_order_rng = random.Random((int(seed) * 1000003) ^ (len(all_vars) * 7919) ^ 0x31A5)
        prompt_variable_order = []
        for layer in topological_layers:
            shuffled = list(layer)
            prompt_order_rng.shuffle(shuffled)
            prompt_variable_order.extend(shuffled)
    if scm_prompt_variant == SCM_PROMPT_VARIANT_NTOPO:
        task_query_text = (
            "Infer a full mechanism map for all endogenous variables under interventions; "
            "simulate as one latent acyclic SCM with root variables treated as exogenous per-row context. "
            "Intervention worlds may use hard_constant or hard_assigned (externally assigned per-row) modes."
        )
    elif scm_prompt_variant == SCM_PROMPT_VARIANT_PARTIAL:
        task_query_text = (
            "Infer a full mechanism map for all endogenous variables under interventions; "
            "simulate using the disclosed partial topological layers, where variables may depend on the same or earlier layers. "
            "Intervention worlds may use hard_constant or hard_assigned (externally assigned per-row) modes."
        )
    else:
        task_query_text = (
            "Infer a full mechanism map for all endogenous variables under interventions; "
            "simulate in topological order with root variables treated as exogenous per-row context. "
            "Intervention worlds may use hard_constant or hard_assigned (externally assigned per-row) modes."
        )

    diag_initial_survivors = int(scm_cegis_diag_final.get("initial_survivors_small_estimate", 0))
    diag_reduction_ratio = float(scm_cegis_diag_final.get("survivor_reduction_ratio", 0.0))
    effective_min_survivor_reduction_frac_diag = (
        None
        if scm_min_survivor_reduction_frac is None
        else float(scm_min_survivor_reduction_frac)
    )
    if (
        effective_min_survivor_reduction_frac_diag is not None
        and scm_target_survivors_small_max is not None
        and diag_initial_survivors > 0
    ):
        max_reachable_frac_diag = max(
            0.0,
            min(
                1.0,
                float(diag_initial_survivors - int(scm_target_survivors_small_max))
                / float(diag_initial_survivors),
            ),
        )
        effective_min_survivor_reduction_frac_diag = min(
            float(effective_min_survivor_reduction_frac_diag),
            float(max_reachable_frac_diag),
        )

    task = CausalTaskSpec(
        taskName=task_name,
        taskVersion="0.3.0",
        query=task_query_text,
        expectedOutputSchema='json:{"mechanisms":{"Xk":"(sexpr)",...}}',
        parameters={
            "family": "IMS_CIND_A",
            "variant": "A_SCM",
            "variables": all_vars,
            "inputVariables": input_vars,
            "rootVariables": root_vars,
            "endogenousVariables": endogenous_vars,
            "topologicalOrder": topological_order,
            "scmPromptVariant": scm_prompt_variant,
            "topologicalLayers": topological_layers,
            "promptVariableOrder": prompt_variable_order,
            "maxPredecessorsPerMechanism": scm_max_predecessors_per_target,
            "allowedOperators": allowed_ops,
            "allowConstants": allow_constants,
            "budgetDeltas": list(DEFAULT_BUDGET_DELTAS),
            "worldSplits": {
                "train": world_ids_train,
                "heldout": world_ids_heldout,
            },
            "difficulty": spec["difficulty"],
            "PanelSemantics": True,
            "scmEvaluationMode": "simulate_endogenous_nonintervened",
            "generationDiagnostics": {
                "scmSampleAttempts": scm_sample_attempts,
                "worldAttempts": world_attempts_used,
                "worldRetries": max(0, world_attempts_used - 1),
                "attempts": world_attempts_used,
                "overlayRandomized": True,
                "use_scm_cegis_lite": use_scm_cegis_lite,
                "cegis_iters": int(scm_cegis_diag_final.get("cegis_iters", 0)),
                "cegisCandidatePlanEvals": int(scm_cegis_diag_final.get("attempts", 0)),
                "shortcuts_killed_by_var": dict(scm_cegis_diag_final.get("shortcuts_killed_by_var", {})),
                "kills_per_iter": [
                    int(v) for v in (scm_cegis_diag_final.get("kills_per_iter") or []) if v is not None
                ],
                "total_shortcuts_killed": int(scm_cegis_diag_final.get("total_shortcuts_killed", 0)),
                "vars_with_kills": list(scm_cegis_diag_final.get("vars_with_kills", [])),
                "kill_coverage_ratio": float(scm_cegis_diag_final.get("kill_coverage_ratio", 0.0)),
                "initial_survivors_small_estimate": int(
                    scm_cegis_diag_final.get("initial_survivors_small_estimate", 0)
                ),
                "survivors_small_estimate": int(scm_cegis_diag_final.get("survivors_small_estimate", 0)),
                "survivor_reduction_abs": int(scm_cegis_diag_final.get("survivor_reduction_abs", 0)),
                "survivor_reduction_ratio": float(
                    scm_cegis_diag_final.get("survivor_reduction_ratio", 0.0)
                ),
                "initial_survivor_vars": list(scm_cegis_diag_final.get("initial_survivor_vars", [])),
                "survivor_vars": list(scm_cegis_diag_final.get("survivor_vars", [])),
                "survivor_count_history": [
                    int(v)
                    for v in (scm_cegis_diag_final.get("survivor_count_history") or [])
                    if v is not None
                ],
                "initial_witness_ast_by_var": dict(
                    scm_cegis_diag_final.get("initial_witness_ast_by_var", {})
                ),
                "final_witness_ast_by_var": dict(
                    scm_cegis_diag_final.get("final_witness_ast_by_var", {})
                ),
                "initial_witness_cap_by_var": dict(
                    scm_cegis_diag_final.get("initial_witness_cap_by_var", {})
                ),
                "final_witness_cap_by_var": dict(
                    scm_cegis_diag_final.get("final_witness_cap_by_var", {})
                ),
                "initial_shortcut_ast_gap_by_var": dict(
                    scm_cegis_diag_final.get("initial_shortcut_ast_gap_by_var", {})
                ),
                "final_shortcut_ast_gap_by_var": dict(
                    scm_cegis_diag_final.get("final_shortcut_ast_gap_by_var", {})
                ),
                "initial_shortcut_ast_ratio_by_var": dict(
                    scm_cegis_diag_final.get("initial_shortcut_ast_ratio_by_var", {})
                ),
                "final_shortcut_ast_ratio_by_var": dict(
                    scm_cegis_diag_final.get("final_shortcut_ast_ratio_by_var", {})
                ),
                "final_shortcut_gap_sum": int(scm_cegis_diag_final.get("final_shortcut_gap_sum", 0)),
                "final_shortcut_ratio_mean": float(
                    scm_cegis_diag_final.get("final_shortcut_ratio_mean", 1.0)
                ),
                "resolved_all_shortcuts_under_cap": bool(
                    scm_cegis_diag_final.get("resolved_all_shortcuts_under_cap", False)
                ),
                "cegis_restart_count": int(scm_cegis_diag_final.get("cegis_restart_count", 1)),
                "cegis_selected_restart_index": int(
                    scm_cegis_diag_final.get("cegis_selected_restart_index", 0)
                ),
                "cegis_restart_summaries": list(
                    scm_cegis_diag_final.get("cegis_restart_summaries", [])
                ),
                "cegis_restart_early_stopped": bool(
                    scm_cegis_diag_final.get("cegis_restart_early_stopped", False)
                ),
                "stage3_lite_enabled": bool(scm_cegis_diag_final.get("stage3_lite_enabled", False)),
                "stage3_lite_probe_size": int(scm_cegis_diag_final.get("stage3_lite_probe_size", 3)),
                "stage3_lite_probe_subsets_per_var": int(
                    scm_cegis_diag_final.get("stage3_lite_probe_subsets_per_var", 2)
                ),
                "stage3_lite_world_budget": int(scm_cegis_diag_final.get("stage3_lite_world_budget", 0)),
                "stage3_lite_worlds_added": int(scm_cegis_diag_final.get("stage3_lite_worlds_added", 0)),
                "stage3_lite_candidate_plan_evals": int(
                    scm_cegis_diag_final.get("stage3_lite_candidate_plan_evals", 0)
                ),
                "stage3_lite_assigned_bias": float(
                    scm_cegis_diag_final.get("stage3_lite_assigned_bias", 0.8)
                ),
                "stage3_lite_total_new_patterns": int(
                    scm_cegis_diag_final.get("stage3_lite_total_new_patterns", 0)
                ),
                "stage3_lite_initial_patterns_seen": int(
                    scm_cegis_diag_final.get("stage3_lite_initial_patterns_seen", 0)
                ),
                "stage3_lite_final_patterns_seen": int(
                    scm_cegis_diag_final.get("stage3_lite_final_patterns_seen", 0)
                ),
                "stage3_lite_total_patterns": int(
                    scm_cegis_diag_final.get("stage3_lite_total_patterns", 0)
                ),
                "stage3_lite_initial_coverage_ratio": float(
                    scm_cegis_diag_final.get("stage3_lite_initial_coverage_ratio", 1.0)
                ),
                "stage3_lite_final_coverage_ratio": float(
                    scm_cegis_diag_final.get("stage3_lite_final_coverage_ratio", 1.0)
                ),
                "stage3_lite_coverage_gain_ratio": float(
                    scm_cegis_diag_final.get("stage3_lite_coverage_gain_ratio", 0.0)
                ),
                "stage3_lite_initial_coverage_by_var": dict(
                    scm_cegis_diag_final.get("stage3_lite_initial_coverage_by_var", {})
                ),
                "stage3_lite_final_coverage_by_var": dict(
                    scm_cegis_diag_final.get("stage3_lite_final_coverage_by_var", {})
                ),
                "stage3_lite_coverage_history": [
                    float(v)
                    for v in (scm_cegis_diag_final.get("stage3_lite_coverage_history") or [])
                    if v is not None
                ],
                "stage3_lite_kills_by_var": dict(
                    scm_cegis_diag_final.get("stage3_lite_kills_by_var", {})
                ),
                "cegis_quality_targets": {
                    "target_survivors_small_max": scm_target_survivors_small_max,
                    "min_survivor_reduction_frac": scm_min_survivor_reduction_frac,
                    "min_survivor_reduction_frac_effective": effective_min_survivor_reduction_frac_diag,
                    "quality_retry_budget": int(scm_quality_retry_budget),
                },
                "cegis_quality_met": {
                    "target_survivors_small_max": (
                        True
                        if scm_target_survivors_small_max is None
                        else int(scm_cegis_diag_final.get("survivors_small_estimate", 0))
                        <= int(scm_target_survivors_small_max)
                    ),
                    "min_survivor_reduction_frac": (
                        True
                        if effective_min_survivor_reduction_frac_diag is None
                        else float(diag_reduction_ratio)
                        >= float(effective_min_survivor_reduction_frac_diag)
                    ),
                },
                "interventionModeCounts": {
                    "train": dict(sorted(mode_counts_train.items())),
                    "heldout": dict(sorted(mode_counts_heldout.items())),
                },
                "scoringCoverageByEndogenous": dict(scoring_coverage_diag_final.get("by_var") or {}),
                "scoringCoverageSummary": dict(scoring_coverage_summary),
                "min_scored_worlds_any_endogenous": int(
                    scoring_coverage_summary.get("min_scored_worlds_any_endogenous", 0)
                ),
                "min_scored_cells_any_endogenous": int(
                    scoring_coverage_summary.get("min_scored_cells_any_endogenous", 0)
                ),
                "max_intervened_worlds_any_endogenous": int(
                    scoring_coverage_summary.get("max_intervened_worlds_any_endogenous", 0)
                ),
                "heldout_min_scored_worlds_any_endogenous": int(
                    scoring_coverage_summary.get("heldout_min_scored_worlds_any_endogenous", 0)
                ),
                "heldout_min_scored_cells_any_endogenous": int(
                    scoring_coverage_summary.get("heldout_min_scored_cells_any_endogenous", 0)
                ),
                "heldoutPlanDiagnostics": list(heldout_plan_diag_final.get("by_world") or []),
                "heldoutPlanSummary": dict(heldout_plan_summary),
                "heldout_mean_novelty": float(heldout_plan_summary.get("heldout_mean_novelty", 0.0)),
                "heldout_max_novelty": float(heldout_plan_summary.get("heldout_max_novelty", 0.0)),
                "heldout_min_novelty": float(heldout_plan_summary.get("heldout_min_novelty", 0.0)),
                "localShortcutDiagnosticsByEndogenous": dict(local_shortcut_diag_final),
                "gold_modular_total_ast": int(total_gold_ast),
                "gold_expanded_total_ast": int(total_expanded_gold_ast),
                "rejectionReasonCounts": {
                    key: value
                    for key, value in sorted(
                        rejection_reason_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                },
                "scmSamplingRejectionCounts": {
                    key: value
                    for key, value in sorted(
                        scm_rejection_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                },
            },
            "knobs": {
                "n": n,
                "n_requested": n_requested,
                "k": k,
                "m": m,
                "heldout_k": heldout_k,
                "parent_max": spec["parent_max"],
                "gold_depth_target": spec.get("gold_depth_target"),
                "gold_depth_sampler_mode": spec.get("gold_depth_sampler_mode"),
                "scm_root_count": root_count,
                "scm_eq_ast_min": eq_ast_min,
                "scm_eq_ast_max": eq_ast_max,
                "allow_constants": allow_constants,
                "scm_require_global_minimal_equiv": bool(
                    spec.get("scm_require_global_minimal_equiv", True)
                ),
                "scm_max_predecessors_per_target": scm_max_predecessors_per_target,
                "intervention_size_probs": spec.get("intervention_size_probs"),
                "scm_intervention_mode_probs": intervention_mode_probs,
                "scm_env_levels": env_levels,
                "scm_shortcut_ast_cap_floor": shortcut_ast_cap_floor,
                "scm_shortcut_ast_cap": shortcut_ast_cap_max,
                "shortcut_ast_cap": shortcut_ast_cap_max,
                "use_scm_cegis_lite": use_scm_cegis_lite,
                "cegis_candidate_world_budget": cegis_candidate_world_budget,
                "cegis_max_iters": cegis_max_iters,
                "scm_cegis_seed_worlds": scm_cegis_seed_worlds,
                "scm_cegis_restarts": scm_cegis_restarts,
                "scm_cegis_early_stop_on_quality_met": bool(
                    scm_cegis_early_stop_on_quality_met
                ),
                "scm_cegis_adaptive_candidate_budget": bool(
                    scm_cegis_adaptive_candidate_budget
                ),
                "scm_cegis_adaptive_min_candidates": int(
                    scm_cegis_adaptive_min_candidates
                ),
                "use_scm_stage3_lite": bool(use_scm_stage3_lite),
                "scm_stage3_probe_size": int(scm_stage3_probe_size),
                "scm_stage3_probe_subsets_per_var": int(scm_stage3_probe_subsets_per_var),
                "scm_stage3_world_budget": int(scm_stage3_world_budget),
                "scm_stage3_candidate_world_budget": int(scm_stage3_candidate_world_budget),
                "scm_stage3_assigned_bias": float(scm_stage3_assigned_bias),
                "scm_target_survivors_small_max": scm_target_survivors_small_max,
                "scm_min_survivor_reduction_frac": scm_min_survivor_reduction_frac,
                "scm_quality_retry_budget": int(scm_quality_retry_budget),
                "scm_prompt_variant": scm_prompt_variant,
                "min_scored_worlds_per_endogenous": spec.get("min_scored_worlds_per_endogenous"),
                "min_scored_cells_per_endogenous": spec.get("min_scored_cells_per_endogenous"),
                "max_intervened_worlds_per_endogenous": spec.get("max_intervened_worlds_per_endogenous"),
                "min_hard_assigned_worlds": spec.get("min_hard_assigned_worlds"),
                "min_hard_constant_worlds": spec.get("min_hard_constant_worlds"),
                "heldout_mode": spec.get("heldout_mode"),
                "heldout_target_novelty_min": spec.get("heldout_target_novelty_min"),
                "heldout_target_novelty_max": spec.get("heldout_target_novelty_max"),
                "heldout_min_scored_worlds_per_endogenous": spec.get(
                    "heldout_min_scored_worlds_per_endogenous"
                ),
                "heldout_min_scored_cells_per_endogenous": spec.get(
                    "heldout_min_scored_cells_per_endogenous"
                ),
            },
        },
    )

    instance = CausalInstance(
        instanceId=instance_id,
        scenario=TASK_CIND_A_SCM,
        signature={
            "variables": all_vars,
            "domains": {v: [0, 1] for v in all_vars},
            "observationStructure": "panel_same_units_across_worlds",
            "PanelSemantics": True,
            "topologicalOrder": topological_order,
            "rootVariables": root_vars,
            "endogenousVariables": endogenous_vars,
            "variableOverlayMode": "anonymous_structure_random_permutation",
        },
        backgroundAxioms=[
            "All worlds share one acyclic structural model.",
            "Root variables are exogenous; world-level environment settings may shift their distributions across worlds.",
            "Interventions follow hard do-semantics and override assigned variables.",
        ],
        worlds=[*train_worlds, *heldout_worlds],
        task=task,
        goldAnswer={
            "family": "IMS_CIND_A",
            "variant": "A_SCM",
            "budgetDeltas": list(DEFAULT_BUDGET_DELTAS),
            "worldSplits": {
                "train": world_ids_train,
                "heldout": world_ids_heldout,
            },
            "scm": {
                "rootVariables": root_vars,
                "endogenousVariables": endogenous_vars,
                "topologicalOrder": topological_order,
                "topologicalLayers": topological_layers,
                "mechanisms": mechanism_by_var,
                "parentsByVar": parent_by_var,
                "mechanismStatsByVar": mechanism_stats_by_var,
                "expandedMechanismStatsByVar": expanded_mechanism_stats_by_var,
                "totalAst": total_gold_ast,
                "gold_modular_total_ast": total_gold_ast,
                "gold_expanded_total_ast": total_expanded_gold_ast,
                "totalEdges": total_gold_edges,
            },
        },
    )

    problem_description = CausalProblemDescription(
        scenarioType=TASK_CIND_A_SCM,
        scenarioDescription="Family A CIND benchmark: induce full acyclic mechanism map under interventions",
        difficulty=spec["difficulty"],
        observationMode="panel_full",
        seed=seed,
        generatorVersion="0.3.0",
        tags=["causal", "interventions", "mechanism-induction", "ims", "cind", "A_SCM"],
        extra={
            "methodology": {
                "trainCorrectness": "exact_match_all_endogenous_nonintervened_cells",
                "heldout": "evaluate_on_unseen_interventions",
                "parsimony": "total_ast_acc_gold_plus_delta_and_bloat",
            },
            "PanelSemantics": True,
            "scmPromptVariant": scm_prompt_variant,
            "topologicalLayers": topological_layers,
            "promptVariableOrder": prompt_variable_order,
            "knobs": task.parameters["knobs"],
            "generationDiagnostics": task.parameters["generationDiagnostics"],
        },
    )

    record = CausalProblemRecord(problem=instance, problemDescription=problem_description).to_dict()
    return _clone_unshared(record)


def _build_cind_problem(
    seed: int,
    instance_id: str,
    task_name: str,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    rng = random.Random(seed)
    variant = _resolve_variant(task_name)
    spec = _resolve_generation_spec(task_config, variant=variant)

    if variant == "A_SCM":
        return _build_cind_problem_full_scm(
            seed=seed,
            instance_id=instance_id,
            task_name=task_name,
            spec=spec,
        )

    n = spec["n"]
    k = spec["k"]
    m = spec["m"]
    heldout_k = spec["heldout_k"]

    input_vars = [f"X{i}" for i in range(1, n)]
    target_var = "Y"
    all_vars = input_vars + [target_var]

    strict_gold_requirements = bool(
        spec.get("require_all_parents_used") or spec.get("require_each_parent_essential")
    )
    allow_constants = bool(spec.get("allow_constants", True))
    parent_gold_max_attempts = 200 if strict_gold_requirements else 1
    parent_gold_attempts_used = 0
    parents: List[str] = []
    gold_expr = ""
    gold_stats: Dict[str, Any] = {}
    parent_size = spec.get("parent_size")

    for attempt in range(1, parent_gold_max_attempts + 1):
        if parent_size is not None:
            parent_count = min(max(1, int(parent_size)), len(input_vars))
        else:
            parent_count = rng.randint(1, min(spec["parent_max"], len(input_vars)))

        sampled_parents = sorted(rng.sample(input_vars, parent_count))
        try:
                sampled_expr, sampled_stats = _sample_gold_mechanism(
                    rng,
                    parents=sampled_parents,
                    ast_min=spec["gold_ast_min"],
                    ast_max=spec["gold_ast_max"],
                    allow_if=spec["allow_if"],
                    allow_constants=allow_constants,
                    depth_target=spec.get("gold_depth_target"),
                    depth_sampler_mode=str(spec.get("gold_depth_sampler_mode", "canonical")),
                    require_all_parents_used=bool(spec.get("require_all_parents_used", False)),
                    require_each_parent_essential=bool(spec.get("require_each_parent_essential", False)),
                    allow_fallback=not strict_gold_requirements,
                )
        except RuntimeError:
            continue

        parents = sampled_parents
        gold_expr = sampled_expr
        gold_stats = sampled_stats
        parent_gold_attempts_used = attempt
        break

    if not parents or not gold_expr:
        raise RuntimeError(
            f"Failed to sample parents+gold mechanism after {parent_gold_max_attempts} attempts "
            f"for instance {instance_id} (task={task_name}, seed={seed})."
        )

    allowed_ops = list(DEFAULT_ALLOWED_OPERATORS)
    if spec["allow_if"]:
        allowed_ops.append("if")

    gold_node = parse_mechanism(
        gold_expr,
        allowed_operators=set(allowed_ops),
        allowed_variables=set(parents),
        allow_constants=allow_constants,
    )

    units = [f"u{i:02d}" for i in range(m)]
    use_cegis_lite = bool(spec.get("use_cegis_lite", False))
    distractor_ast_cap = int(spec.get("distractor_ast_cap", max(1, min(spec["gold_ast_max"], 14))))
    distractor_pool_size = int(spec.get("distractor_pool_size", 256))
    cegis_candidate_interventions = int(spec.get("cegis_candidate_interventions", 30))
    allvar_shortcut_ast_cap = spec.get("allvar_shortcut_ast_cap")
    if bool(spec.get("allvar_shortcut_ast_cap_from_gold", False)):
        allvar_shortcut_ast_cap = max(1, int(gold_stats.get("astSize", 1)) - 1)
    cegis_exact_witness_ast_cap = spec.get("cegis_exact_witness_ast_cap")
    if cegis_exact_witness_ast_cap is None:
        if allvar_shortcut_ast_cap is not None:
            cegis_exact_witness_ast_cap = int(allvar_shortcut_ast_cap)
        elif spec.get("nonparent_shortcut_ast_cap") is not None:
            cegis_exact_witness_ast_cap = int(spec.get("nonparent_shortcut_ast_cap"))
    if cegis_exact_witness_ast_cap is not None:
        cegis_exact_witness_ast_cap = max(1, int(cegis_exact_witness_ast_cap))
    cegis_exact_witness_time_budget_ms = spec.get("cegis_exact_witness_time_budget_ms")
    if cegis_exact_witness_time_budget_ms is not None:
        cegis_exact_witness_time_budget_ms = max(1, int(cegis_exact_witness_time_budget_ms))
    cegis_exact_witness_max_signatures_per_size = spec.get(
        "cegis_exact_witness_max_signatures_per_size"
    )
    if cegis_exact_witness_max_signatures_per_size is not None:
        cegis_exact_witness_max_signatures_per_size = max(
            64, int(cegis_exact_witness_max_signatures_per_size)
        )
    shortcut_check_time_budget_ms = spec.get("shortcut_check_time_budget_ms")
    if shortcut_check_time_budget_ms is not None:
        shortcut_check_time_budget_ms = max(1, int(shortcut_check_time_budget_ms))
    shortcut_check_max_signatures_per_size = spec.get("shortcut_check_max_signatures_per_size")
    if shortcut_check_max_signatures_per_size is not None:
        shortcut_check_max_signatures_per_size = max(
            64, int(shortcut_check_max_signatures_per_size)
        )
    shortcut_check_timeout_policy = str(
        spec.get("shortcut_check_timeout_policy", "sampled_fallback") or "sampled_fallback"
    ).strip().lower()
    shortcut_check_timeout_fallback_samples = max(
        0,
        int(spec.get("shortcut_check_timeout_fallback_samples", 192) or 0),
    )
    max_extra_train_worlds_for_shortcut = int(spec.get("max_extra_train_worlds_for_shortcut", 0) or 0)

    distractors: List[Dict[str, Any]] = []
    if use_cegis_lite:
        parent_seed_term = sum(int(str(p).replace("X", "")) for p in parents) if parents else 0
        distractor_seed = int(seed) * 7919 + parent_seed_term * 37 + int(gold_stats.get("astSize", 0))
        distractors = _build_distractor_pool(
            seed=distractor_seed,
            input_vars=input_vars,
            parents=parents,
            allowed_ops=allowed_ops,
            gold_expr=gold_expr,
            ast_cap=distractor_ast_cap,
            pool_size=distractor_pool_size,
            allow_if=bool(spec["allow_if"]),
            allow_constants=allow_constants,
        )

    train_worlds: List[CausalWorldView] = []
    heldout_worlds: List[CausalWorldView] = []
    accepted = False
    extra_train_worlds_added_final = 0

    max_attempts = max(1, int(spec.get("world_max_attempts", 220)))
    world_attempt_time_budget_sec = spec.get("world_attempt_time_budget_sec")
    if world_attempt_time_budget_sec is not None:
        world_attempt_time_budget_sec = max(0.25, float(world_attempt_time_budget_sec))
    world_attempt_budget_hit = False
    world_attempts_used = 0
    intervention_size_probs = spec.get("intervention_size_probs")
    rejection_reason_counts: Dict[str, int] = {}

    def _bump_rejection(reason: str) -> None:
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

    cegis_diag_final = {
        "survivors_total": None,
        "survivors_small": None,
        "min_ast_survivor_excluding_gold": None,
        "gap_to_second_best": None,
        "distractor_pool_size": len(distractors) if use_cegis_lite else None,
        "kills_per_world": [],
        "exact_witness_cap": (
            int(cegis_exact_witness_ast_cap)
            if cegis_exact_witness_ast_cap is not None
            else None
        ),
        "exact_witness_kills_per_world": [],
        "exact_witness_ast_per_world": [],
    }
    shortcut_failure_reasons = {"nonparent_shortcut_found", "allvar_shortcut_found"}
    world_attempt_budget_t0 = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        if (
            world_attempt_time_budget_sec is not None
            and attempt > 1
            and (time.perf_counter() - world_attempt_budget_t0) >= float(world_attempt_time_budget_sec)
        ):
            world_attempt_budget_hit = True
            _bump_rejection("world_attempt_time_budget_exceeded")
            break
        world_attempts_used = attempt
        base_contexts = _sample_unit_contexts(rng, units, input_vars, must_vary_vars=parents)
        accepted_train_worlds: List[CausalWorldView] = []
        accepted_seen_interventions: Set[Tuple[Tuple[str, int], ...]] = set()
        accepted_cegis_diag_iter: Dict[str, Any] = {}
        selected_train_world_count = k
        attempt_accepted = False

        for extra_added in range(max_extra_train_worlds_for_shortcut + 1):
            train_world_count = k + extra_added
            selected_train_world_count = train_world_count
            cegis_diag_iter: Dict[str, Any] = {
                "survivors_total": None,
                "survivors_small": None,
                "min_ast_survivor_excluding_gold": None,
                "gap_to_second_best": None,
                "distractor_pool_size": len(distractors) if use_cegis_lite else None,
                "kills_per_world": [],
                "exact_witness_cap": (
                    int(cegis_exact_witness_ast_cap)
                    if cegis_exact_witness_ast_cap is not None
                    else None
                ),
                "exact_witness_kills_per_world": [],
                "exact_witness_ast_per_world": [],
            }

            if use_cegis_lite:
                # Identifiability-first mode: never preserve small survivors during
                # intervention selection; always maximize elimination pressure.
                min_survivors_small_to_keep = 0
                train_worlds, seen_interventions, survivor_indices, trace = _build_train_worlds_cegis_lite(
                    rng=rng,
                    units=units,
                    input_vars=input_vars,
                    target_var=target_var,
                    base_contexts=base_contexts,
                    gold_node=gold_node,
                    parents=parents,
                    k=train_world_count,
                    intervention_size_probs=intervention_size_probs,
                    candidate_interventions=cegis_candidate_interventions,
                    distractors=distractors,
                    gold_ast_size=int(gold_stats.get("astSize", 0)),
                    min_survivors_small_to_keep=min_survivors_small_to_keep,
                    exact_witness_ast_cap=cegis_exact_witness_ast_cap,
                    exact_witness_time_budget_ms=cegis_exact_witness_time_budget_ms,
                    exact_witness_max_signatures_per_size=(
                        cegis_exact_witness_max_signatures_per_size
                    ),
                    allowed_ops=allowed_ops,
                    allow_constants=allow_constants,
                )
                if len(train_worlds) < train_world_count:
                    _bump_rejection("cegis_short_train_worlds")
                    break
                survivor_diag = _compute_survivor_diagnostics(
                    distractors=distractors,
                    survivor_indices=survivor_indices,
                    gold_ast_size=int(gold_stats.get("astSize", 0)),
                )
                cegis_diag_iter.update(survivor_diag)
                cegis_diag_iter["kills_per_world"] = list(trace.get("kills_per_world", []))
                cegis_diag_iter["exact_witness_cap"] = trace.get("exact_witness_cap")
                cegis_diag_iter["exact_witness_kills_per_world"] = list(
                    trace.get("exact_witness_kills_per_world", [])
                )
                cegis_diag_iter["exact_witness_ast_per_world"] = list(
                    trace.get("exact_witness_ast_per_world", [])
                )
            else:
                train_worlds = []
                seen_interventions = set()

                # Baseline observational world.
                baseline = {}
                train_worlds.append(
                    _make_world(
                        world_id="train_00",
                        split="train",
                        units=units,
                        input_vars=input_vars,
                        target_var=target_var,
                        base_contexts=base_contexts,
                        interventions=baseline,
                        gold_node=gold_node,
                    )
                )
                seen_interventions.add(_assignments_key(baseline))

                force_queue = parents.copy()
                rng.shuffle(force_queue)

                while len(train_worlds) < train_world_count:
                    force_var = force_queue.pop() if force_queue else None
                    ints = _sample_intervention(
                        rng,
                        input_vars,
                        force_var=force_var,
                        min_targets=1,
                        max_targets=2 if rng.random() < 0.8 else 3,
                        size_probs=intervention_size_probs,
                    )
                    key = _assignments_key(ints)
                    if key in seen_interventions:
                        continue
                    seen_interventions.add(key)

                    world_id = f"train_{len(train_worlds):02d}"
                    train_worlds.append(
                        _make_world(
                            world_id=world_id,
                            split="train",
                            units=units,
                            input_vars=input_vars,
                            target_var=target_var,
                            base_contexts=base_contexts,
                            interventions=ints,
                            gold_node=gold_node,
                        )
                    )

            if len(train_worlds) < train_world_count:
                _bump_rejection("random_short_train_worlds")
                break

            failure_reason = _anti_shortcut_failure_reason(
                train_worlds=train_worlds,
                parents=parents,
                input_vars=input_vars,
                target_var=target_var,
                nonparent_shortcut_ast_cap=spec.get("nonparent_shortcut_ast_cap"),
                allvar_shortcut_ast_cap=allvar_shortcut_ast_cap,
                allowed_ops=allowed_ops,
                cegis_survivor_diagnostics=cegis_diag_iter if use_cegis_lite else None,
                target_survivors_small_min=spec.get("target_survivors_small_min"),
                target_survivors_small_max=spec.get("target_survivors_small_max"),
                gap_to_second_best_min=spec.get("gap_to_second_best_min"),
                gap_to_second_best_max=spec.get("gap_to_second_best_max"),
                require_survivors_small_or_gap=bool(spec.get("require_survivors_small_or_gap", False)),
                min_total_kills=spec.get("min_total_kills"),
                min_worlds_with_kills=spec.get("min_worlds_with_kills"),
                min_parent_assignment_coverage=spec.get("min_parent_assignment_coverage"),
                allow_constants=allow_constants,
                shortcut_check_time_budget_ms=shortcut_check_time_budget_ms,
                shortcut_check_max_signatures_per_size=shortcut_check_max_signatures_per_size,
                shortcut_check_timeout_policy=shortcut_check_timeout_policy,
                shortcut_check_timeout_fallback_samples=shortcut_check_timeout_fallback_samples,
            )
            if failure_reason is None:
                accepted_train_worlds = train_worlds
                accepted_seen_interventions = set(seen_interventions)
                accepted_cegis_diag_iter = cegis_diag_iter
                extra_train_worlds_added_final = max(0, train_world_count - k)
                selected_train_world_count = train_world_count
                attempt_accepted = True
                break

            if (
                failure_reason in shortcut_failure_reasons
                and extra_added < max_extra_train_worlds_for_shortcut
            ):
                continue

            _bump_rejection(failure_reason)
            break

        if not attempt_accepted:
            continue

        train_worlds = accepted_train_worlds
        cegis_diag_final = accepted_cegis_diag_iter

        heldout_worlds = []
        seen_interventions_for_heldout = set(accepted_seen_interventions)
        while len(heldout_worlds) < heldout_k:
            ints = _sample_intervention(
                rng,
                input_vars,
                force_var=None,
                min_targets=1,
                max_targets=3,
                size_probs=intervention_size_probs,
            )
            key = _assignments_key(ints)
            if key in seen_interventions_for_heldout:
                continue
            seen_interventions_for_heldout.add(key)

            world_id = f"heldout_{len(heldout_worlds):02d}"
            heldout_worlds.append(
                _make_world(
                    world_id=world_id,
                    split="heldout",
                    units=units,
                    input_vars=input_vars,
                    target_var=target_var,
                    base_contexts=base_contexts,
                    interventions=ints,
                    gold_node=gold_node,
                )
            )

        if len(train_worlds) != selected_train_world_count:
            _bump_rejection("train_world_count_mismatch")
            continue
        accepted = True
        break

    if not accepted:
        top_reasons = sorted(rejection_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        reason_txt = ", ".join(f"{name}={count}" for name, count in top_reasons)
        if not reason_txt:
            reason_txt = "none_captured"
        raise RuntimeError(
            f"Failed to generate non-degenerate CIND worlds after {max_attempts} attempts "
            f"for instance {instance_id} (task={task_name}, seed={seed}). "
            f"Top rejection reasons: {reason_txt}."
        )

    world_ids_train = [w.worldId for w in train_worlds]
    world_ids_heldout = [w.worldId for w in heldout_worlds]

    query_by_variant = {
        "A_Y": "Infer Pa(Y) and one mechanism f_Y that is invariant across all training interventions.",
        "A_P": "Infer Pa(Y) only from interventional evidence.",
        "A_SCM": (
            "Infer a full mechanism map for endogenous variables under interventions; "
            "evaluate by simulation over all endogenous non-intervened variables."
        ),
        "A_OOD": "Infer one invariant mechanism for Y from training worlds; evaluation emphasizes held-out interventions.",
    }

    output_schema_by_variant = {
        "A_Y": "json:{\"parents\":[...],\"mechanism\":\"(sexpr)\"}",
        "A_P": "json:{\"parents\":[...]}",
        "A_SCM": "json:{\"mechanisms\":{\"Xk\":\"(sexpr)\",...}}",
        "A_OOD": "json:{\"parents\":[...],\"mechanism\":\"(sexpr)\"}",
    }

    task = CausalTaskSpec(
        taskName=task_name,
        taskVersion="0.2.0",
        query=query_by_variant.get(variant, query_by_variant["A_Y"]),
        expectedOutputSchema=output_schema_by_variant.get(variant, output_schema_by_variant["A_Y"]),
        parameters={
            "family": "IMS_CIND_A",
            "variant": variant,
            "target": target_var,
            "variables": all_vars,
            "inputVariables": input_vars,
            "allowedOperators": allowed_ops,
            "allowConstants": allow_constants,
            "budgetDeltas": list(DEFAULT_BUDGET_DELTAS),
            "worldSplits": {
                "train": world_ids_train,
                "heldout": world_ids_heldout,
            },
            "difficulty": spec["difficulty"],
            "generationDiagnostics": {
                "parentGoldAttempts": parent_gold_attempts_used,
                "worldAttempts": world_attempts_used,
                "worldRetries": max(0, world_attempts_used - 1),
                "worldAttemptBudgetHit": bool(world_attempt_budget_hit),
                "worldAttemptTimeBudgetSec": (
                    float(world_attempt_time_budget_sec)
                    if world_attempt_time_budget_sec is not None
                    else None
                ),
                "use_cegis_lite": use_cegis_lite,
                "survivors_total": cegis_diag_final.get("survivors_total"),
                "survivors_small": cegis_diag_final.get("survivors_small"),
                "min_ast_survivor_excluding_gold": cegis_diag_final.get("min_ast_survivor_excluding_gold"),
                "gap_to_second_best": cegis_diag_final.get("gap_to_second_best"),
                "distractor_pool_size": cegis_diag_final.get("distractor_pool_size"),
                "kills_per_world": list(cegis_diag_final.get("kills_per_world", [])),
                "exact_witness_cap": cegis_diag_final.get("exact_witness_cap"),
                "exact_witness_time_budget_ms": cegis_diag_final.get("exact_witness_time_budget_ms"),
                "exact_witness_max_signatures_per_size": cegis_diag_final.get(
                    "exact_witness_max_signatures_per_size"
                ),
                "shortcut_check_time_budget_ms": (
                    None
                    if shortcut_check_time_budget_ms is None
                    else int(shortcut_check_time_budget_ms)
                ),
                "shortcut_check_max_signatures_per_size": (
                    None
                    if shortcut_check_max_signatures_per_size is None
                    else int(shortcut_check_max_signatures_per_size)
                ),
                "shortcut_check_timeout_policy": str(shortcut_check_timeout_policy),
                "shortcut_check_timeout_fallback_samples": int(
                    shortcut_check_timeout_fallback_samples
                ),
                "exact_witness_kills_per_world": list(
                    cegis_diag_final.get("exact_witness_kills_per_world", [])
                ),
                "exact_witness_ast_per_world": list(
                    cegis_diag_final.get("exact_witness_ast_per_world", [])
                ),
                "extraTrainWorldsAdded": int(extra_train_worlds_added_final),
                "rejectionReasonCounts": {
                    key: value
                    for key, value in sorted(
                        rejection_reason_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                },
            },
            "knobs": {
                "n": n,
                "k": k,
                "m": m,
                "parent_max": spec["parent_max"],
                "parent_size": spec.get("parent_size"),
                "gold_ast_min": spec["gold_ast_min"],
                "gold_ast_max": spec["gold_ast_max"],
                "gold_depth_target": spec.get("gold_depth_target"),
                "gold_depth_sampler_mode": spec.get("gold_depth_sampler_mode"),
                "allow_constants": allow_constants,
                "require_all_parents_used": bool(spec.get("require_all_parents_used", False)),
                "require_each_parent_essential": bool(spec.get("require_each_parent_essential", False)),
                "nonparent_shortcut_ast_cap": spec.get("nonparent_shortcut_ast_cap"),
                "allvar_shortcut_ast_cap": allvar_shortcut_ast_cap,
                "allvar_shortcut_ast_cap_from_gold": bool(spec.get("allvar_shortcut_ast_cap_from_gold", False)),
                "max_extra_train_worlds_for_shortcut": max_extra_train_worlds_for_shortcut,
                "world_max_attempts": int(max_attempts),
                "world_attempt_time_budget_sec": (
                    float(world_attempt_time_budget_sec)
                    if world_attempt_time_budget_sec is not None
                    else None
                ),
                "intervention_size_probs": spec.get("intervention_size_probs"),
                "use_cegis_lite": use_cegis_lite,
                "distractor_ast_cap": distractor_ast_cap,
                "distractor_pool_size": distractor_pool_size,
                "cegis_candidate_interventions": cegis_candidate_interventions,
                "cegis_exact_witness_ast_cap": (
                    int(cegis_exact_witness_ast_cap)
                    if cegis_exact_witness_ast_cap is not None
                    else None
                ),
                "cegis_exact_witness_time_budget_ms": (
                    int(cegis_exact_witness_time_budget_ms)
                    if cegis_exact_witness_time_budget_ms is not None
                    else None
                ),
                "cegis_exact_witness_max_signatures_per_size": (
                    int(cegis_exact_witness_max_signatures_per_size)
                    if cegis_exact_witness_max_signatures_per_size is not None
                    else None
                ),
                "shortcut_check_time_budget_ms": (
                    int(shortcut_check_time_budget_ms)
                    if shortcut_check_time_budget_ms is not None
                    else None
                ),
                "shortcut_check_max_signatures_per_size": (
                    int(shortcut_check_max_signatures_per_size)
                    if shortcut_check_max_signatures_per_size is not None
                    else None
                ),
                "shortcut_check_timeout_policy": str(shortcut_check_timeout_policy),
                "shortcut_check_timeout_fallback_samples": int(
                    shortcut_check_timeout_fallback_samples
                ),
                "target_survivors_small_min": spec.get("target_survivors_small_min"),
                "target_survivors_small_max": spec.get("target_survivors_small_max"),
                "gap_to_second_best_min": spec.get("gap_to_second_best_min"),
                "gap_to_second_best_max": spec.get("gap_to_second_best_max"),
                "require_survivors_small_or_gap": bool(spec.get("require_survivors_small_or_gap", False)),
                "min_total_kills": spec.get("min_total_kills"),
                "min_worlds_with_kills": spec.get("min_worlds_with_kills"),
                "min_parent_assignment_coverage": spec.get("min_parent_assignment_coverage"),
            },
        },
    )

    instance = CausalInstance(
        instanceId=instance_id,
        scenario=VARIANT_TO_CANONICAL_TASK.get(variant, TASK_CIND_A_Y),
        signature={
            "variables": all_vars,
            "target": target_var,
            "domains": {v: [0, 1] for v in all_vars},
            "observationStructure": "panel_same_units_across_worlds",
        },
        backgroundAxioms=[
            "All worlds share the same underlying SCM.",
            "Interventions follow hard do-semantics and override assigned variables.",
        ],
        worlds=[*train_worlds, *heldout_worlds],
        task=task,
        goldAnswer={
            "family": "IMS_CIND_A",
            "variant": variant,
            "target": target_var,
            "parents": parents,
            "mechanism": gold_expr,
            "mechanismStats": gold_stats,
            "budgetDeltas": list(DEFAULT_BUDGET_DELTAS),
            "worldSplits": {
                "train": world_ids_train,
                "heldout": world_ids_heldout,
            },
        },
    )

    problem_description = CausalProblemDescription(
        scenarioType=VARIANT_TO_CANONICAL_TASK.get(variant, TASK_CIND_A_Y),
        scenarioDescription=(
            "Family A CIND benchmark: infer invariant mechanism from interventional worlds"
        ),
        difficulty=spec["difficulty"],
        observationMode="panel_full",
        seed=seed,
        generatorVersion="0.2.0",
        tags=["causal", "interventions", "mechanism-induction", "ims", "cind", variant],
        extra={
            "methodology": {
                "trainCorrectness": "exact_match_all_training_worlds",
                "heldout": "evaluate_on_unseen_interventions",
                "parsimony": "acc_gold_plus_delta_and_bloat",
            },
            "knobs": task.parameters["knobs"],
            "generationDiagnostics": task.parameters["generationDiagnostics"],
        },
    )

    record = CausalProblemRecord(problem=instance, problemDescription=problem_description).to_dict()
    return _clone_unshared(record)


def _try_parse_json_obj(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "Empty response"

    candidate = text.strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload, None
        return None, "Top-level JSON must be an object"
    except Exception:
        pass

    # Robust fallback: scan every '{' position and attempt raw JSON decode.
    # This handles responses that include prose with brace-like tokens
    # (e.g. LaTeX-style \text{...}) before the final JSON object.
    decoder = json.JSONDecoder()
    parsed_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, ch in enumerate(candidate):
        if ch != "{":
            continue
        try:
            payload, end = decoder.raw_decode(candidate, i)
        except Exception:
            continue
        if isinstance(payload, dict):
            parsed_candidates.append((i, end, payload))

    if parsed_candidates:
        def _candidate_score(obj: Dict[str, Any]) -> int:
            score = 0
            if isinstance(obj.get("mechanisms"), dict):
                score += 100
            if isinstance(obj.get("parents"), list):
                score += 50
            if isinstance(obj.get("mechanism"), str):
                score += 50
            return score

        best = max(
            parsed_candidates,
            key=lambda item: (
                _candidate_score(item[2]),
                len(item[2]),
                item[1] - item[0],
                -item[0],
            ),
        )
        return best[2], None

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        snippet = candidate[start : end + 1]
        try:
            payload = json.loads(snippet)
            if isinstance(payload, dict):
                return payload, None
            return None, "Extracted JSON is not an object"
        except Exception as e:
            return None, f"Could not parse JSON object: {e}"

    return None, "No JSON object found"


def _extract_cind_answer(response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return _try_parse_json_obj(response)


def _extract_parents(answer: Dict[str, Any]) -> Optional[List[str]]:
    keys = ("parents", "pa_hat", "Pa_hat", "parent_set", "parentSet")
    for key in keys:
        value = answer.get(key)
        if isinstance(value, list):
            out = [str(x).strip() for x in value if str(x).strip()]
            return out
    return None


def _natural_var_sort_key(token: str) -> Tuple[int, int, str]:
    match = _X_TOKEN_RE.match(str(token))
    if match:
        return (0, int(match.group(1)), "")
    return (1, 0, str(token))


def _sorted_unique_natural(values: List[str] | Set[str]) -> List[str]:
    uniq = {str(v) for v in values}
    return sorted(uniq, key=_natural_var_sort_key)


def _bounded_predecessors(
    ordered_predecessors: List[str],
    max_predecessors: Optional[int],
) -> List[str]:
    if max_predecessors is None:
        return list(ordered_predecessors)
    cap = max(1, int(max_predecessors))
    if len(ordered_predecessors) <= cap:
        return list(ordered_predecessors)
    return list(ordered_predecessors[-cap:])


def _extract_mechanism(answer: Dict[str, Any]) -> Optional[str]:
    keys = ("mechanism", "fhat_Y", "f_hat", "expression", "formula")
    for key in keys:
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parent_scores(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "parentExact": pred_set == gold_set,
        "parentPrecision": precision,
        "parentRecall": recall,
        "parentF1": f1,
        "parentTP": tp,
        "parentFP": fp,
        "parentFN": fn,
    }


def _evaluate_on_worlds(
    expression: str,
    worlds: List[Dict[str, Any]],
    target_var: str,
    allowed_operators: List[str],
    allowed_variables: Optional[List[str]] = None,
    allow_constants: bool = True,
) -> Dict[str, Any]:
    try:
        node = parse_mechanism(
            expression,
            allowed_operators=set(allowed_operators),
            allowed_variables=set(allowed_variables) if allowed_variables is not None else None,
            allow_constants=allow_constants,
        )
    except MechanismParseError as e:
        return {
            "valid": False,
            "parseError": str(e),
            "exact": False,
            "accuracy": 0.0,
            "totalRows": 0,
            "correctRows": 0,
            "firstMismatch": None,
            "parsedNode": None,
            "stats": None,
            "worldTotal": 0,
            "worldExactCount": 0,
            "worldExactAccuracy": 0.0,
        }

    total = 0
    correct = 0
    first_mismatch = None
    world_total = 0
    world_exact_count = 0

    for world in worlds:
        world_id = world.get("worldId", "unknown")
        rows = _iter_world_rows(world)
        world_rows = 0
        world_correct_rows = 0
        for i, values in enumerate(rows):
            if target_var not in values:
                continue
            total += 1
            world_rows += 1
            y_true = int(values[target_var])
            try:
                y_hat = int(evaluate_parsed_mechanism(node, values))
            except MechanismEvalError as e:
                return {
                    "valid": False,
                    "parseError": f"Evaluation failed: {e}",
                    "exact": False,
                    "accuracy": 0.0,
                    "totalRows": total,
                    "correctRows": correct,
                    "firstMismatch": {
                        "worldId": world_id,
                        "rowIndex": i,
                        "reason": str(e),
                    },
                    "parsedNode": None,
                    "stats": None,
                    "worldTotal": world_total,
                    "worldExactCount": world_exact_count,
                    "worldExactAccuracy": (world_exact_count / world_total) if world_total else 0.0,
                }

            if y_hat == y_true:
                correct += 1
                world_correct_rows += 1
            elif first_mismatch is None:
                first_mismatch = {
                    "worldId": world_id,
                    "rowIndex": i,
                    "predicted": y_hat,
                    "expected": y_true,
                }

        if world_rows > 0:
            world_total += 1
            if world_correct_rows == world_rows:
                world_exact_count += 1

    accuracy = (correct / total) if total else 0.0
    world_exact_accuracy = (world_exact_count / world_total) if world_total else 0.0
    return {
        "valid": True,
        "parseError": None,
        "exact": bool(total > 0 and correct == total),
        "accuracy": accuracy,
        "totalRows": total,
        "correctRows": correct,
        "firstMismatch": first_mismatch,
        "parsedNode": node,
        "stats": analyze_mechanism(node),
        "worldTotal": world_total,
        "worldExactCount": world_exact_count,
        "worldExactAccuracy": world_exact_accuracy,
    }


def _evaluate_scm_on_worlds(
    parsed_nodes_by_var: Dict[str, MechanismNode],
    worlds: List[Dict[str, Any]],
    topological_order: List[str],
    root_vars: List[str],
    endogenous_vars: List[str],
) -> Dict[str, Any]:
    root_set = set(root_vars)
    endogenous_set = set(endogenous_vars)
    total = 0
    correct = 0
    first_mismatch = None

    per_var_total: Dict[str, int] = {v: 0 for v in endogenous_vars}
    per_var_correct: Dict[str, int] = {v: 0 for v in endogenous_vars}
    world_total = 0
    world_exact_count = 0

    for world in worlds:
        world_id = world.get("worldId", "unknown")
        ints = _extract_world_intervention_assignments(world)
        intervention_targets = _extract_world_intervention_targets(world)
        mode = _extract_world_intervention_mode(world)
        rows = _iter_world_rows(world)
        world_scored = 0
        world_correct = 0
        for i, row in enumerate(rows):
            assignment: Dict[str, int] = {}
            for var in topological_order:
                if mode == "hard_assigned" and var in intervention_targets:
                    if var not in row:
                        return {
                            "valid": False,
                            "parseError": f"Missing assigned intervention value for '{var}' in row",
                            "exact": False,
                            "accuracy": 0.0,
                            "totalRows": total,
                            "correctRows": correct,
                            "firstMismatch": {
                                "worldId": world_id,
                                "rowIndex": i,
                                "variable": var,
                                "reason": "missing_assigned_intervention_value",
                            },
                            "worldTotal": world_total,
                            "worldExactCount": world_exact_count,
                            "worldExactAccuracy": (world_exact_count / world_total) if world_total else 0.0,
                        }
                    assignment[var] = int(row[var])
                    continue
                if var in ints:
                    assignment[var] = int(ints[var])
                    continue
                if var in root_set:
                    if var not in row:
                        return {
                            "valid": False,
                            "parseError": f"Missing root variable '{var}' in observed row",
                            "exact": False,
                            "accuracy": 0.0,
                            "totalRows": total,
                            "correctRows": correct,
                            "firstMismatch": {
                                "worldId": world_id,
                                "rowIndex": i,
                                "variable": var,
                                "reason": "missing_root_value",
                            },
                            "worldTotal": world_total,
                            "worldExactCount": world_exact_count,
                            "worldExactAccuracy": (world_exact_count / world_total) if world_total else 0.0,
                        }
                    assignment[var] = int(row[var])
                    continue

                node = parsed_nodes_by_var.get(var)
                if node is None:
                    return {
                        "valid": False,
                        "parseError": f"Missing parsed mechanism for endogenous variable '{var}'",
                        "exact": False,
                        "accuracy": 0.0,
                        "totalRows": total,
                        "correctRows": correct,
                        "firstMismatch": {
                            "worldId": world_id,
                            "rowIndex": i,
                            "variable": var,
                            "reason": "missing_mechanism",
                        },
                        "worldTotal": world_total,
                        "worldExactCount": world_exact_count,
                        "worldExactAccuracy": (world_exact_count / world_total) if world_total else 0.0,
                    }
                try:
                    assignment[var] = int(evaluate_parsed_mechanism(node, assignment))
                except MechanismEvalError as e:
                    return {
                        "valid": False,
                        "parseError": f"Evaluation failed for {var}: {e}",
                        "exact": False,
                        "accuracy": 0.0,
                        "totalRows": total,
                        "correctRows": correct,
                        "firstMismatch": {
                            "worldId": world_id,
                            "rowIndex": i,
                            "variable": var,
                            "reason": str(e),
                        },
                        "worldTotal": world_total,
                        "worldExactCount": world_exact_count,
                        "worldExactAccuracy": (world_exact_count / world_total) if world_total else 0.0,
                    }

            for var in topological_order:
                if var not in endogenous_set:
                    continue
                if var in intervention_targets:
                    # Intervened endogenous variables are not scored as mechanism predictions.
                    continue
                if var not in row:
                    continue
                total += 1
                world_scored += 1
                per_var_total[var] = per_var_total.get(var, 0) + 1
                expected = int(row[var])
                predicted = int(assignment[var])
                if predicted == expected:
                    correct += 1
                    world_correct += 1
                    per_var_correct[var] = per_var_correct.get(var, 0) + 1
                elif first_mismatch is None:
                    first_mismatch = {
                        "worldId": world_id,
                        "rowIndex": i,
                        "variable": var,
                        "predicted": predicted,
                        "expected": expected,
                    }

        if world_scored > 0:
            world_total += 1
            if world_correct == world_scored:
                world_exact_count += 1

    accuracy = (correct / total) if total else 0.0
    per_var_accuracy = {
        var: (per_var_correct[var] / per_var_total[var]) if per_var_total[var] else None
        for var in endogenous_vars
    }
    world_exact_accuracy = (world_exact_count / world_total) if world_total else 0.0
    return {
        "valid": True,
        "parseError": None,
        "exact": bool(total > 0 and correct == total),
        "accuracy": accuracy,
        "totalRows": total,
        "correctRows": correct,
        "firstMismatch": first_mismatch,
        "perVariableAccuracy": per_var_accuracy,
        "worldTotal": world_total,
        "worldExactCount": world_exact_count,
        "worldExactAccuracy": world_exact_accuracy,
    }


def _infer_ntopo_endogenous_order(
    topological_order: List[str],
    root_vars: List[str],
    endogenous_vars: List[str],
    parents_by_var: Dict[str, List[str]],
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Infer an evaluation order for endogenous variables from predicted dependencies.

    Returns:
      - endogenous_order if acyclic, else None
      - list of cycle-involved endogenous vars if cyclic, else None
    """
    topo_index = {str(v): i for i, v in enumerate(topological_order)}
    root_set = {str(v) for v in root_vars}
    endo_order_seed = [str(v) for v in topological_order if str(v) in set(endogenous_vars)]
    if len(endo_order_seed) != len(set(endo_order_seed)):
        return None, _sorted_unique_natural(endo_order_seed)

    endo_set = set(endo_order_seed)
    indegree: Dict[str, int] = {v: 0 for v in endo_order_seed}
    children: Dict[str, Set[str]] = {v: set() for v in endo_order_seed}

    for child in endo_order_seed:
        for parent in parents_by_var.get(child, []):
            p = str(parent)
            if p in root_set:
                continue
            if p in endo_set and child not in children[p]:
                children[p].add(child)
                indegree[child] += 1

    def _order_key(v: str) -> Tuple[int, int, str]:
        idx = topo_index.get(v, 10**9)
        nat = _natural_var_sort_key(v)
        return (idx, nat[0], nat[1] if len(nat) > 1 else 0, nat[2] if len(nat) > 2 else "")

    ready = [v for v in endo_order_seed if indegree[v] == 0]
    ready.sort(key=_order_key)

    out: List[str] = []
    while ready:
        current = ready.pop(0)
        out.append(current)
        for child in sorted(children[current], key=_order_key):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=_order_key)

    if len(out) != len(endo_order_seed):
        cycle_nodes = [v for v in endo_order_seed if indegree.get(v, 0) > 0]
        return None, _sorted_unique_natural(cycle_nodes)

    return out, None


def _infer_partial_order_endogenous_order(
    *,
    topological_order: List[str],
    root_vars: List[str],
    endogenous_vars: List[str],
    parents_by_var: Dict[str, List[str]],
    topological_layers: Sequence[Sequence[str]],
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Infer an endogenous order that respects disclosed partial-order layers."""

    topo_index = {str(v): i for i, v in enumerate(topological_order)}
    root_set = {str(v) for v in root_vars}
    endo_order_seed = [str(v) for v in topological_order if str(v) in set(endogenous_vars)]
    if len(endo_order_seed) != len(set(endo_order_seed)):
        return None, _sorted_unique_natural(endo_order_seed)

    layer_index: Dict[str, int] = {}
    for layer_idx, layer in enumerate(topological_layers):
        for var in layer:
            layer_index[str(var)] = int(layer_idx)

    missing_layers = [v for v in [*root_vars, *endo_order_seed] if str(v) not in layer_index]
    if missing_layers:
        return None, _sorted_unique_natural(missing_layers)

    endo_set = set(endo_order_seed)
    indegree: Dict[str, int] = {v: 0 for v in endo_order_seed}
    children: Dict[str, Set[str]] = {v: set() for v in endo_order_seed}

    for child in endo_order_seed:
        child_layer = int(layer_index[str(child)])
        for parent in parents_by_var.get(child, []):
            p = str(parent)
            parent_layer = layer_index.get(p)
            if parent_layer is not None and int(parent_layer) > child_layer:
                return None, _sorted_unique_natural([p, child])
            if p in root_set:
                continue
            if p in endo_set and child not in children[p]:
                children[p].add(child)
                indegree[child] += 1

    def _order_key(v: str) -> Tuple[int, int, int, str]:
        nat = _natural_var_sort_key(v)
        return (
            int(layer_index.get(str(v), 10**9)),
            int(topo_index.get(str(v), 10**9)),
            nat[1] if len(nat) > 1 else 0,
            nat[2] if len(nat) > 2 else "",
        )

    ready = [v for v in endo_order_seed if indegree[v] == 0]
    ready.sort(key=_order_key)

    out: List[str] = []
    while ready:
        current = ready.pop(0)
        out.append(current)
        for child in sorted(children[current], key=_order_key):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=_order_key)

    if len(out) != len(endo_order_seed):
        cycle_nodes = [v for v in endo_order_seed if indegree.get(v, 0) > 0]
        return None, _sorted_unique_natural(cycle_nodes)

    return out, None


def _evaluate_variant_a_y_like(
    problem: Dict[str, Any],
    answer: Dict[str, Any],
    variant: str,
) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}
    gold = record_problem.get("goldAnswer", {}) or {}

    target_var = str(params.get("target", "Y"))
    allowed_ops = list(params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS))
    allow_constants = bool(params.get("allowConstants", True))
    deltas = list(params.get("budgetDeltas") or list(DEFAULT_BUDGET_DELTAS))

    input_variables = list(params.get("inputVariables") or [])
    if not input_variables:
        all_vars = list(params.get("variables") or [])
        input_variables = [v for v in all_vars if str(v) != target_var]
    allowed_var_set = {str(v) for v in input_variables}

    train_worlds, heldout_worlds = _worlds_by_split(record_problem)

    pred_parents = _extract_parents(answer)
    if pred_parents is None:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Missing parent set in answer",
        }

    pred_parents_normalized = [str(p) for p in pred_parents]
    if len(pred_parents_normalized) != len(set(pred_parents_normalized)):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Parents must not contain duplicates",
            "declaredParents": pred_parents_normalized,
        }

    expected_parent_order = _sorted_unique_natural(pred_parents_normalized)
    if pred_parents_normalized != expected_parent_order:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": (
                "Parents must be sorted in natural variable order "
                "(e.g., X1 < X2 < ... < X10)"
            ),
            "declaredParents": pred_parents_normalized,
            "expectedOrder": expected_parent_order,
        }

    pred_parent_set = set(pred_parents_normalized)
    if target_var in pred_parent_set:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": f"Target variable '{target_var}' is not allowed in parents",
        }

    unknown_parents = _sorted_unique_natural([p for p in pred_parent_set if p not in allowed_var_set])
    if unknown_parents:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Parents contain variables outside allowed input variables",
            "unknownParents": unknown_parents,
            "allowedInputVariables": _sorted_unique_natural(allowed_var_set),
        }

    mechanism = _extract_mechanism(answer)
    if not mechanism:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Missing mechanism expression in answer",
        }
    if any(ch in mechanism for ch in ("\n", "\r", "\t")):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Mechanism must be a single-line string without tab/newline characters",
        }

    train_eval = _evaluate_on_worlds(
        mechanism,
        train_worlds,
        target_var,
        allowed_ops,
        allowed_variables=input_variables,
        allow_constants=allow_constants,
    )
    if not train_eval["valid"]:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": train_eval.get("parseError") or "Mechanism parse/eval failed",
            "parseError": train_eval.get("parseError"),
        }

    heldout_eval = _evaluate_on_worlds(
        mechanism,
        heldout_worlds,
        target_var,
        allowed_ops,
        allowed_variables=input_variables,
        allow_constants=allow_constants,
    )
    if not heldout_eval["valid"]:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": heldout_eval.get("parseError") or "Held-out evaluation failed",
            "parseError": heldout_eval.get("parseError"),
        }

    used_vars = set(train_eval["stats"]["variables"])
    if target_var in used_vars:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": f"Target variable '{target_var}' cannot appear as mechanism input",
            "mechanismVariables": _sorted_unique_natural(used_vars),
        }

    if not used_vars.issubset(allowed_var_set):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Mechanism references variables outside allowed input variables",
            "mechanismVariables": _sorted_unique_natural(used_vars),
            "allowedInputVariables": _sorted_unique_natural(allowed_var_set),
        }

    # Strict parent rule: parents must exactly equal sorted unique mechanism variables.
    expected_mech_parents = _sorted_unique_natural(used_vars)
    if expected_parent_order != expected_mech_parents:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Strict parent rule violated: parents must exactly match mechanism variables",
            "expectedParents": expected_mech_parents,
            "declaredParents": expected_parent_order,
        }

    gold_parents = list(gold.get("parents", []))
    parent_metrics = _parent_scores(gold_parents, pred_parents)

    gold_ast = (gold.get("mechanismStats") or {}).get("astSize")
    if gold_ast is None and isinstance(gold.get("mechanism"), str):
        gold_ast = analyze_mechanism(
            gold["mechanism"],
            allowed_operators=allowed_ops,
            allowed_variables=input_variables,
        ).get("astSize")

    candidate_ast = train_eval["stats"]["astSize"]
    acc_gold_plus = {}
    for delta in deltas:
        key = f"delta_{int(delta)}"
        passes_budget = gold_ast is None or candidate_ast <= int(gold_ast) + int(delta)
        acc_gold_plus[key] = bool(train_eval["exact"] and passes_budget)

    bloat = False
    if gold_ast is not None:
        bloat = bool(train_eval["exact"] and candidate_ast > int(gold_ast) + 25)

    heldout_all_worlds_exact = bool(heldout_eval["exact"])
    strict_heldout_exact = bool(train_eval["exact"] and heldout_all_worlds_exact)

    result = {
        "valid": True,
        "correct": bool(train_eval["exact"]),
        "trainExact": bool(train_eval["exact"]),
        "trainAccuracy": train_eval["accuracy"],
        "trainWorldExactAccuracy": train_eval.get("worldExactAccuracy"),
        "trainWorlds": train_eval.get("worldTotal"),
        "trainWorldExactCount": train_eval.get("worldExactCount"),
        "heldoutAllWorldsExact": heldout_all_worlds_exact,
        "heldoutExact": strict_heldout_exact,
        "heldoutAccuracy": heldout_eval["accuracy"],
        "heldoutWorldExactAccuracy": heldout_eval.get("worldExactAccuracy"),
        "heldoutWorlds": heldout_eval.get("worldTotal"),
        "heldoutWorldExactCount": heldout_eval.get("worldExactCount"),
        "heldoutRows": heldout_eval["totalRows"],
        "firstTrainMismatch": train_eval["firstMismatch"],
        "firstHeldoutMismatch": heldout_eval["firstMismatch"],
        "candidateStats": train_eval["stats"],
        "goldAstSize": gold_ast,
        "accGoldPlus": acc_gold_plus,
        "bloat": bloat,
    }
    result.update(parent_metrics)

    if variant == "A_OOD":
        result["oodHeadline"] = {
            "heldoutAllWorldsExact": heldout_all_worlds_exact,
            "heldoutExact": strict_heldout_exact,
            "heldoutAccuracy": heldout_eval["accuracy"],
        }

    return result

def _evaluate_variant_a_p(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}
    gold = record_problem.get("goldAnswer", {}) or {}

    target_var = str(params.get("target", "Y"))
    input_variables = list(params.get("inputVariables") or [])
    if not input_variables:
        all_vars = list(params.get("variables") or [])
        input_variables = [v for v in all_vars if str(v) != target_var]
    allowed_var_set = {str(v) for v in input_variables}

    pred_parents = _extract_parents(answer)
    if pred_parents is None:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Missing parent set in answer",
        }

    pred_parents_normalized = [str(p) for p in pred_parents]
    if len(pred_parents_normalized) != len(set(pred_parents_normalized)):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Parents must not contain duplicates",
            "declaredParents": pred_parents_normalized,
        }

    expected_parent_order = _sorted_unique_natural(pred_parents_normalized)
    if pred_parents_normalized != expected_parent_order:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": (
                "Parents must be sorted in natural variable order "
                "(e.g., X1 < X2 < ... < X10)"
            ),
            "declaredParents": pred_parents_normalized,
            "expectedOrder": expected_parent_order,
        }

    pred_parent_set = set(pred_parents_normalized)
    if target_var in pred_parent_set:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": f"Target variable '{target_var}' is not allowed in parents",
        }

    unknown_parents = _sorted_unique_natural([p for p in pred_parent_set if p not in allowed_var_set])
    if unknown_parents:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Parents contain variables outside allowed input variables",
            "unknownParents": unknown_parents,
            "allowedInputVariables": _sorted_unique_natural(allowed_var_set),
        }

    gold_parents = list(gold.get("parents", []))
    metrics = _parent_scores(gold_parents, pred_parents_normalized)
    result = {
        "valid": True,
        "correct": bool(metrics["parentExact"]),
        "goldParents": _sorted_unique_natural(gold_parents),
        "predictedParents": expected_parent_order,
    }
    result.update(metrics)
    return result

def _evaluate_variant_a_scm(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    signature = record_problem.get("signature", {}) or {}
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}
    gold = record_problem.get("goldAnswer", {}) or {}

    variables = [str(v) for v in list(signature.get("variables", []))]
    target_var = str(params.get("target", "Y"))
    topological_order = [str(v) for v in list(params.get("topologicalOrder") or variables)]
    if not topological_order:
        topological_order = list(variables)
    if len(topological_order) != len(set(topological_order)):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Topological order metadata contains duplicate variables",
        }
    scm_prompt_variant = _normalize_scm_prompt_variant(
        params.get("scmPromptVariant") or params.get("promptVariant") or SCM_PROMPT_VARIANT_ORDERED
    )
    hide_topological_order = _scm_variant_hides_topological_order(scm_prompt_variant)
    partial_topological_order = _scm_variant_uses_partial_order(scm_prompt_variant)
    topological_layers = _normalize_topological_layers(
        params.get("topologicalLayers"),
        topological_order=topological_order,
    )
    if partial_topological_order and not topological_layers:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Partial-order prompt variant requires valid topologicalLayers metadata",
        }

    root_vars = [str(v) for v in list(params.get("rootVariables") or [])]
    root_set = set(root_vars)
    endogenous_vars = [str(v) for v in list(params.get("endogenousVariables") or [])]
    if not endogenous_vars:
        endogenous_vars = [v for v in topological_order if v not in root_set]
    endogenous_set = set(endogenous_vars)
    allowed_operators = list(params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS))
    allow_constants = bool(params.get("allowConstants", True))
    mechanisms = answer.get("mechanisms")
    if not isinstance(mechanisms, dict):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Expected top-level key 'mechanisms' with a JSON object value",
        }

    provided = {str(k) for k in mechanisms.keys()}
    unknown = _sorted_unique_natural([k for k in provided if k not in endogenous_set])
    if unknown:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Mechanism map contains unknown or non-endogenous keys",
            "unknownMechanismKeys": unknown,
            "expectedKeys": _sorted_unique_natural(endogenous_set),
        }

    missing = _sorted_unique_natural([v for v in endogenous_vars if v not in provided])
    if missing:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Mechanism map is missing endogenous variables",
            "missingMechanismKeys": missing,
            "expectedKeys": _sorted_unique_natural(endogenous_set),
        }

    parsed_nodes_by_var: Dict[str, MechanismNode] = {}
    candidate_stats_by_var: Dict[str, Dict[str, Any]] = {}
    parent_by_var: Dict[str, List[str]] = {}
    observed_variables = [str(v) for v in (variables or topological_order)]
    observed_set = set(observed_variables)

    index_of = {v: i for i, v in enumerate(topological_order)}
    for var in endogenous_vars:
        expr = mechanisms.get(var)
        if not isinstance(expr, str) or not expr.strip():
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": f"Mechanism for variable '{var}' must be a non-empty string",
            }
        expr = expr.strip()
        if any(ch in expr for ch in ("\n", "\r", "\t")):
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": f"Mechanism for '{var}' must be single-line without tabs/newlines",
            }

        idx = index_of.get(var, -1)
        if idx < 0:
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": f"Variable '{var}' not found in topological order metadata",
            }
        # Match prompt contract:
        # - ntopo: any observed variable except self
        # - partial: only same-or-earlier displayed block variables except self
        # - ordered: only full topological predecessors
        if hide_topological_order:
            allowed_variables = [v for v in observed_variables if v != var]
        elif partial_topological_order:
            allowed_variables = _same_or_earlier_layer_variables(var, topological_layers or [])
            if allowed_variables is None:
                return {
                    "valid": False,
                    "correct": False,
                    "failureExplanation": f"Variable '{var}' not found in topologicalLayers metadata",
                }
        else:
            # Evaluation-side legality is based on full topological precedence.
            # Do not apply generation-time predecessor capping here.
            allowed_variables = list(topological_order[:idx])
        try:
            node = parse_mechanism(
                expr,
                allowed_operators=set(allowed_operators),
                allowed_variables=set(allowed_variables),
                allow_constants=allow_constants,
            )
        except MechanismParseError as e:
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": f"Mechanism parse failed for '{var}': {e}",
                "parseError": str(e),
            }
        parsed_nodes_by_var[var] = node
        stats = analyze_mechanism(node)
        candidate_stats_by_var[var] = stats
        parent_by_var[var] = _sorted_unique_natural(list(mechanism_variables(node)))
        if hide_topological_order:
            invalid_parents = [p for p in parent_by_var[var] if p not in observed_set or p == var]
            if invalid_parents:
                return {
                    "valid": False,
                    "correct": False,
                    "failureExplanation": (
                        f"Mechanism for '{var}' references variables outside observed set or self"
                    ),
                    "invalidVariables": _sorted_unique_natural(invalid_parents),
                    "allowedVariables": _sorted_unique_natural([v for v in observed_set if v != var]),
                }

    eval_topological_order = list(topological_order)
    if hide_topological_order:
        inferred_endo_order, cycle_nodes = _infer_ntopo_endogenous_order(
            topological_order=topological_order,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            parents_by_var=parent_by_var,
        )
        if inferred_endo_order is None:
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": (
                    "NTopo mechanism dependencies are cyclic; cannot derive a valid acyclic SCM order"
                ),
                "parseError": "cyclic_ntopo_dependencies",
                "cycleVariables": cycle_nodes or [],
            }

        root_order = [v for v in topological_order if v in root_set]
        eval_topological_order = [*root_order, *inferred_endo_order]
    elif partial_topological_order:
        inferred_endo_order, cycle_nodes = _infer_partial_order_endogenous_order(
            topological_order=topological_order,
            root_vars=root_vars,
            endogenous_vars=endogenous_vars,
            parents_by_var=parent_by_var,
            topological_layers=topological_layers or [],
        )
        if inferred_endo_order is None:
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": (
                    "Partial-order mechanism dependencies are cyclic or violate disclosed layers; "
                    "cannot derive a valid layer-consistent SCM order"
                ),
                "parseError": "invalid_partial_order_dependencies",
                "cycleVariables": cycle_nodes or [],
            }

        root_order = [v for v in topological_order if v in root_set]
        eval_topological_order = [*root_order, *inferred_endo_order]

    train_worlds, heldout_worlds = _worlds_by_split(record_problem)
    train_eval = _evaluate_scm_on_worlds(
        parsed_nodes_by_var=parsed_nodes_by_var,
        worlds=train_worlds,
        topological_order=eval_topological_order,
        root_vars=root_vars,
        endogenous_vars=endogenous_vars,
    )
    if not train_eval.get("valid", False):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": train_eval.get("parseError") or "SCM simulation failed on train worlds",
            "parseError": train_eval.get("parseError"),
        }

    heldout_eval = _evaluate_scm_on_worlds(
        parsed_nodes_by_var=parsed_nodes_by_var,
        worlds=heldout_worlds,
        topological_order=eval_topological_order,
        root_vars=root_vars,
        endogenous_vars=endogenous_vars,
    )
    if not heldout_eval.get("valid", False):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": heldout_eval.get("parseError") or "SCM simulation failed on held-out worlds",
            "parseError": heldout_eval.get("parseError"),
        }

    candidate_total_ast = int(
        sum(int((candidate_stats_by_var[v] or {}).get("astSize", 0)) for v in endogenous_vars)
    )
    candidate_total_edges = int(
        sum(int((candidate_stats_by_var[v] or {}).get("parentCount", 0)) for v in endogenous_vars)
    )
    candidate_max_depth = 0
    op_counts_total: Dict[str, int] = {}
    for var in endogenous_vars:
        stats = candidate_stats_by_var.get(var, {})
        candidate_max_depth = max(candidate_max_depth, int(stats.get("maxDepth", 0)))
        for op, count in (stats.get("operatorCounts") or {}).items():
            op_counts_total[str(op)] = op_counts_total.get(str(op), 0) + int(count)

    gold_scm = (gold.get("scm") or {}) if isinstance(gold.get("scm"), dict) else {}
    gold_total_ast = gold_scm.get("totalAst")
    if gold_total_ast is None:
        gold_stats_by_var = gold_scm.get("mechanismStatsByVar") or {}
        if isinstance(gold_stats_by_var, dict):
            gold_total_ast = sum(
                int((gold_stats_by_var.get(v) or {}).get("astSize", 0)) for v in endogenous_vars
            )
    if gold_total_ast is None:
        gold_total_ast = None

    deltas = list(params.get("budgetDeltas") or list(DEFAULT_BUDGET_DELTAS))
    acc_gold_plus: Dict[str, bool] = {}
    for delta in deltas:
        key = f"delta_{int(delta)}"
        passes_budget = gold_total_ast is None or candidate_total_ast <= int(gold_total_ast) + int(delta)
        acc_gold_plus[key] = bool(train_eval["exact"] and passes_budget)

    bloat = False
    if gold_total_ast is not None:
        bloat = bool(train_eval["exact"] and candidate_total_ast > int(gold_total_ast) + 25)

    heldout_all_worlds_exact = bool(heldout_eval["exact"])
    strict_heldout_exact = bool(train_eval["exact"] and heldout_all_worlds_exact)

    return {
        "valid": True,
        "correct": bool(train_eval["exact"]),
        "trainExact": bool(train_eval["exact"]),
        "trainAccuracy": train_eval["accuracy"],
        "trainWorldExactAccuracy": train_eval.get("worldExactAccuracy"),
        "trainWorlds": train_eval.get("worldTotal"),
        "trainWorldExactCount": train_eval.get("worldExactCount"),
        "heldoutAllWorldsExact": heldout_all_worlds_exact,
        "heldoutExact": strict_heldout_exact,
        "heldoutAccuracy": heldout_eval["accuracy"],
        "heldoutWorldExactAccuracy": heldout_eval.get("worldExactAccuracy"),
        "heldoutWorlds": heldout_eval.get("worldTotal"),
        "heldoutWorldExactCount": heldout_eval.get("worldExactCount"),
        "heldoutRows": heldout_eval["totalRows"],
        "firstTrainMismatch": train_eval.get("firstMismatch"),
        "firstHeldoutMismatch": heldout_eval.get("firstMismatch"),
        "candidateStats": {
            "astSize": candidate_total_ast,
            "maxDepth": candidate_max_depth,
            "parentCount": candidate_total_edges,
            "operatorCounts": op_counts_total,
            "equationCount": len(endogenous_vars),
            "perVariable": candidate_stats_by_var,
            "parentsByVar": parent_by_var,
        },
        "goldAstSize": gold_total_ast,
        "accGoldPlus": acc_gold_plus,
        "bloat": bloat,
        "fullScmCoverage": True,
        "providedMechanismCount": len(provided),
        "requiredMechanismCount": len(endogenous_vars),
        "fullScmVerification": "endogenous_full_simulation",
        "perVariableTrainAccuracy": train_eval.get("perVariableAccuracy"),
        "perVariableHeldoutAccuracy": heldout_eval.get("perVariableAccuracy"),
    }


def _evaluate_variant_a_scm_root_unknown(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    signature = record_problem.get("signature", {}) or {}
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}

    variables = [str(v) for v in list(signature.get("variables", []))]
    topological_order = [str(v) for v in list(params.get("topologicalOrder") or variables)]
    if not topological_order:
        topological_order = list(variables)
    observed_variables = [str(v) for v in list(params.get("variables") or variables or topological_order)]
    if not observed_variables:
        observed_variables = list(topological_order)
    observed_set = set(observed_variables)
    gold_root_vars = [str(v) for v in list(params.get("rootVariables") or [])]
    expected_root_count = int(params.get("rootCount") or len(gold_root_vars))

    roots = answer.get("roots")
    mechanisms = answer.get("mechanisms")
    if not isinstance(roots, list):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Expected top-level key 'roots' with a JSON array value",
        }
    if not isinstance(mechanisms, dict):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Expected top-level key 'mechanisms' with a JSON object value",
        }

    predicted_roots = [str(v) for v in roots]
    if len(predicted_roots) != len(set(predicted_roots)):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Roots must not contain duplicates",
            "predictedRoots": predicted_roots,
        }

    unknown_roots = _sorted_unique_natural([v for v in predicted_roots if v not in observed_set])
    if unknown_roots:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Roots contain variables outside ObservedVariables",
            "unknownRoots": unknown_roots,
            "observedVariables": _sorted_unique_natural(list(observed_set)),
        }

    if len(predicted_roots) != expected_root_count:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": f"Expected exactly {expected_root_count} roots",
            "predictedRootCount": len(predicted_roots),
            "expectedRootCount": expected_root_count,
        }

    predicted_root_set = set(predicted_roots)
    predicted_root_order = [v for v in topological_order if v in predicted_root_set]
    predicted_endogenous = [v for v in topological_order if v not in predicted_root_set]
    predicted_endogenous_set = set(predicted_endogenous)

    provided_keys = {str(k) for k in mechanisms.keys()}
    unknown_mech = _sorted_unique_natural([k for k in provided_keys if k not in predicted_endogenous_set])
    missing_mech = _sorted_unique_natural([v for v in predicted_endogenous if v not in provided_keys])
    if unknown_mech or missing_mech:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Mechanism keys must be exactly the complement of the declared roots",
            "missingMechanismKeys": missing_mech,
            "unknownMechanismKeys": unknown_mech,
            "expectedMechanismKeys": predicted_endogenous,
        }

    root_precision = 0.0
    root_recall = 0.0
    gold_root_set = set(gold_root_vars)
    if predicted_roots:
        root_precision = len(predicted_root_set & gold_root_set) / float(len(predicted_root_set))
    if gold_root_set:
        root_recall = len(predicted_root_set & gold_root_set) / float(len(gold_root_set))
    root_set_exact = bool(predicted_root_set == gold_root_set)

    derived_problem = _clone_unshared(problem)
    derived_record_problem = derived_problem.get("problem", derived_problem)
    derived_task = derived_record_problem.get("task", {}) or {}
    derived_params = derived_task.get("parameters", {}) or {}
    derived_params["rootVariables"] = predicted_root_order
    derived_params["endogenousVariables"] = predicted_endogenous
    derived_task["parameters"] = derived_params
    derived_record_problem["task"] = derived_task
    if isinstance(derived_record_problem.get("goldAnswer"), dict):
        gold_scm = (derived_record_problem["goldAnswer"].get("scm") or {})
        if isinstance(gold_scm, dict):
            gold_scm["rootVariables"] = predicted_root_order
            gold_scm["endogenousVariables"] = predicted_endogenous
            derived_record_problem["goldAnswer"]["scm"] = gold_scm

    scm_eval = _evaluate_variant_a_scm(derived_problem, {"mechanisms": mechanisms})
    result: Dict[str, Any] = {
        **dict(scm_eval),
        "rootSetExact": root_set_exact,
        "rootPrecision": float(root_precision),
        "rootRecall": float(root_recall),
        "predictedRoots": predicted_root_order,
        "goldRoots": _sorted_unique_natural(gold_root_vars),
        "predictedRootCount": len(predicted_roots),
        "expectedRootCount": expected_root_count,
        "predictedEndogenousVariables": predicted_endogenous,
        "mechanismTrainExact": bool(scm_eval.get("trainExact")),
        "mechanismHeldoutAllWorldsExact": bool(scm_eval.get("heldoutAllWorldsExact")),
        "mechanismHeldoutExact": bool(scm_eval.get("heldoutExact")),
    }
    result["correct"] = bool(result.get("valid")) and bool(result.get("trainExact")) and root_set_exact
    if not result["correct"] and result.get("valid") and not result.get("failureExplanation") and not root_set_exact:
        result["failureExplanation"] = "Predicted root set does not match the true root set"
    return result


def _gold_scm_id_payload(gold: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(gold.get("scm_id"), dict):
        return dict(gold.get("scm_id") or {})
    if isinstance(gold.get("scmId"), dict):
        return dict(gold.get("scmId") or {})
    certification = gold.get("certification")
    if certification is None:
        return {}
    return {"certification": certification}


def _gold_scm_alt_exp_payload(gold: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(gold.get("scm_alt_exp"), dict):
        return dict(gold.get("scm_alt_exp") or {})
    if isinstance(gold.get("scmAltExp"), dict):
        return dict(gold.get("scmAltExp") or {})
    return {}


def _canonicalize_scm_mechanism_map(
    mechanisms: Dict[str, Any],
    *,
    endogenous_vars: Sequence[str],
    observed_variables: Sequence[str],
    allowed_operators: Sequence[str],
    allow_constants: bool,
) -> Tuple[Optional[str], Optional[str]]:
    rendered: Dict[str, str] = {}
    observed = [str(v) for v in observed_variables]
    for var in endogenous_vars:
        expr = mechanisms.get(str(var))
        if not isinstance(expr, str) or not expr.strip():
            return None, f"Mechanism for variable '{var}' must be a non-empty string"
        try:
            node = parse_mechanism(
                expr.strip(),
                allowed_operators=set(str(op) for op in allowed_operators),
                allowed_variables={str(v) for v in observed if str(v) != str(var)},
                allow_constants=allow_constants,
            )
        except MechanismParseError as e:
            return None, f"Mechanism parse failed for '{var}': {e}"
        rendered[str(var)] = node_to_sexpr(_canonicalize_node(node))
    return json.dumps({"mechanisms": rendered}, sort_keys=True, separators=(",", ":")), None


def _parse_scm_mechanisms_semantic(
    problem: Dict[str, Any],
    mechanisms: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    record_problem = problem.get("problem", problem)
    signature = record_problem.get("signature", {}) or {}
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}

    variables = [str(v) for v in list(signature.get("variables", []))]
    topological_order = [str(v) for v in list(params.get("topologicalOrder") or variables)]
    if not topological_order:
        topological_order = list(variables)
    root_vars = [str(v) for v in list(params.get("rootVariables") or [])]
    root_set = set(root_vars)
    endogenous_vars = [str(v) for v in list(params.get("endogenousVariables") or [])]
    if not endogenous_vars:
        endogenous_vars = [v for v in topological_order if v not in root_set]
    endogenous_set = set(endogenous_vars)
    allowed_operators = [str(op) for op in list(params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS))]
    allow_constants = bool(params.get("allowConstants", True))
    observed_variables = [str(v) for v in (variables or topological_order)]
    observed_set = set(observed_variables)

    if not isinstance(mechanisms, dict):
        return None, "Expected top-level key 'mechanisms' with a JSON object value"

    provided = {str(k) for k in mechanisms.keys()}
    unknown = _sorted_unique_natural([k for k in provided if k not in endogenous_set])
    if unknown:
        return None, "Mechanism map contains unknown or non-endogenous keys"
    missing = _sorted_unique_natural([v for v in endogenous_vars if v not in provided])
    if missing:
        return None, "Mechanism map is missing endogenous variables"

    parsed_nodes_by_var: Dict[str, MechanismNode] = {}
    candidate_stats_by_var: Dict[str, Dict[str, Any]] = {}
    parent_by_var: Dict[str, List[str]] = {}

    for var in endogenous_vars:
        expr = mechanisms.get(var)
        if not isinstance(expr, str) or not expr.strip():
            return None, f"Mechanism for variable '{var}' must be a non-empty string"
        expr = expr.strip()
        if any(ch in expr for ch in ("\n", "\r", "\t")):
            return None, f"Mechanism for '{var}' must be single-line without tabs/newlines"
        try:
            node = parse_mechanism(
                expr,
                allowed_operators=set(allowed_operators),
                allowed_variables={str(v) for v in observed_variables if str(v) != str(var)},
                allow_constants=allow_constants,
            )
        except MechanismParseError as e:
            return None, f"Mechanism parse failed for '{var}': {e}"
        parsed_nodes_by_var[var] = node
        candidate_stats_by_var[var] = analyze_mechanism(node)
        used_vars = _sorted_unique_natural(list(mechanism_variables(node)))
        if any(p not in observed_set or p == var for p in used_vars):
            return None, f"Mechanism for '{var}' references variables outside observed set or self"
        parent_by_var[var] = used_vars

    inferred_endo_order, cycle_nodes = _infer_ntopo_endogenous_order(
        topological_order=topological_order,
        root_vars=root_vars,
        endogenous_vars=endogenous_vars,
        parents_by_var=parent_by_var,
    )
    if inferred_endo_order is None:
        cycle_txt = ", ".join(cycle_nodes or [])
        return None, (
            "Mechanism dependencies are cyclic; cannot derive a valid acyclic SCM order"
            + (f" ({cycle_txt})" if cycle_txt else "")
        )

    root_order = [v for v in topological_order if v in root_set]
    eval_topological_order = [*root_order, *inferred_endo_order]
    return {
        "observedVariables": observed_variables,
        "rootVariables": root_vars,
        "endogenousVariables": endogenous_vars,
        "allowedOperators": allowed_operators,
        "allowConstants": allow_constants,
        "topologicalOrder": topological_order,
        "evalTopologicalOrder": eval_topological_order,
        "parsedNodesByVar": parsed_nodes_by_var,
        "candidateStatsByVar": candidate_stats_by_var,
        "parentsByVar": parent_by_var,
    }, None


def _effective_parents_from_truth_table(
    dep_order: Sequence[str],
    truth_table: TruthTable,
) -> Tuple[str, ...]:
    deps = tuple(str(v) for v in dep_order)
    if not deps:
        return tuple()
    effective: List[str] = []
    for dep_idx, dep in enumerate(deps):
        changed = False
        for bits, out in truth_table.items():
            flipped = list(bits)
            flipped[dep_idx] = 1 - int(flipped[dep_idx])
            flipped_out = truth_table.get(tuple(int(bit) for bit in flipped))
            if flipped_out is not None and int(flipped_out) != int(out):
                changed = True
                break
        if changed:
            effective.append(dep)
    return tuple(effective)


def _project_truth_table_to_effective_deps(
    dep_order: Sequence[str],
    truth_table: TruthTable,
    effective_deps: Sequence[str],
) -> str:
    deps = tuple(str(v) for v in dep_order)
    eff = tuple(str(v) for v in effective_deps)
    if not eff:
        any_value = next(iter(truth_table.values())) if truth_table else 0
        return str(int(any_value))
    dep_index = {str(dep): idx for idx, dep in enumerate(deps)}
    bits_out: List[str] = []
    for eff_bits in itertools.product((0, 1), repeat=len(eff)):
        outputs: Set[int] = set()
        for full_bits, value in truth_table.items():
            matches = True
            for eff_idx, dep in enumerate(eff):
                if int(full_bits[dep_index[dep]]) != int(eff_bits[eff_idx]):
                    matches = False
                    break
            if matches:
                outputs.add(int(value))
        if not outputs:
            bits_out.append("0")
        elif len(outputs) == 1:
            bits_out.append(str(next(iter(outputs))))
        else:
            bits_out.append("X")
    return "".join(bits_out)


def _semantic_payload_for_scm_mechanisms(
    problem: Dict[str, Any],
    mechanisms: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    parsed_payload, parse_error = _parse_scm_mechanisms_semantic(problem, mechanisms)
    if parsed_payload is None:
        return None, parse_error

    endogenous_vars = [str(v) for v in list(parsed_payload.get("endogenousVariables") or [])]
    parsed_nodes_by_var = parsed_payload.get("parsedNodesByVar") or {}
    per_var: Dict[str, Dict[str, Any]] = {}
    total_effective_parent_count = 0
    total_ast_size = 0
    max_var_ast_size = 0
    total_ast_depth = 0
    for var in endogenous_vars:
        node = parsed_nodes_by_var.get(var)
        if node is None:
            return None, f"Missing parsed mechanism for '{var}'"
        dep_order, truth_table = _compile_mechanism_truth_table(node)
        effective_deps = _effective_parents_from_truth_table(dep_order, truth_table)
        truth_bits = _project_truth_table_to_effective_deps(dep_order, truth_table, effective_deps)
        stats = dict((parsed_payload.get("candidateStatsByVar") or {}).get(var) or {})
        ast_size = int(stats.get("astSize", 0))
        ast_depth = int(stats.get("maxDepth", 0))
        total_effective_parent_count += int(len(effective_deps))
        total_ast_size += ast_size
        max_var_ast_size = max(max_var_ast_size, ast_size)
        total_ast_depth += ast_depth
        per_var[str(var)] = {
            "effectiveParents": list(effective_deps),
            "truthBits": truth_bits,
            "astSize": ast_size,
            "astDepth": ast_depth,
        }
    signature_payload = {
        "perVariable": OrderedDict(
            (
                str(var),
                {
                    "effectiveParents": list(per_var[str(var)]["effectiveParents"]),
                    "truthBits": str(per_var[str(var)]["truthBits"]),
                },
            )
            for var in sorted(endogenous_vars, key=_natural_var_key)
        )
    }
    return {
        **parsed_payload,
        "semanticPerVariable": per_var,
        "semanticSignature": json.dumps(signature_payload, sort_keys=True, separators=(",", ":")),
        "simplicity": {
            "totalEffectiveParentCount": int(total_effective_parent_count),
            "totalAstSize": int(total_ast_size),
            "maxVariableAstSize": int(max_var_ast_size),
            "totalAstDepth": int(total_ast_depth),
        },
    }, None


def scm_semantic_signature(
    problem: Dict[str, Any],
    mechanisms: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Return the benchmark semantic signature for an SCM mechanism map.

    The signature identifies SCM semantics by effective parents and truth bits
    only. AST shape differences that preserve those semantics do not change the
    signature.
    """
    payload, error = _semantic_payload_for_scm_mechanisms(problem, mechanisms)
    if payload is None:
        return None, error
    return str(payload.get("semanticSignature") or ""), None


def scm_semantically_equivalent(
    problem: Dict[str, Any],
    left_mechanisms: Dict[str, Any],
    right_mechanisms: Dict[str, Any],
) -> Tuple[Optional[bool], Optional[str]]:
    """Return whether two SCM mechanism maps are semantically equivalent."""
    left_signature, left_error = scm_semantic_signature(problem, left_mechanisms)
    if left_signature is None:
        return None, left_error
    right_signature, right_error = scm_semantic_signature(problem, right_mechanisms)
    if right_signature is None:
        return None, right_error
    return bool(left_signature == right_signature), None


def _scm_semantically_distinct(
    problem: Dict[str, Any],
    left_mechanisms: Dict[str, Any],
    right_mechanisms: Dict[str, Any],
) -> Tuple[Optional[bool], Optional[str]]:
    equivalent, error = scm_semantically_equivalent(problem, left_mechanisms, right_mechanisms)
    if equivalent is None:
        return None, error
    return bool(not equivalent), None


def _scm_distance_from_gold(
    gold_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
) -> Dict[str, int]:
    endogenous_vars = [str(v) for v in list(gold_payload.get("endogenousVariables") or [])]
    gold_nodes = dict(gold_payload.get("parsedNodesByVar") or {})
    cand_nodes = dict(candidate_payload.get("parsedNodesByVar") or {})
    gold_sem = dict(gold_payload.get("semanticPerVariable") or {})
    cand_sem = dict(candidate_payload.get("semanticPerVariable") or {})

    changed_variable_count = 0
    total_truth_table_hamming_distance = 0
    total_effective_parent_symmetric_difference = 0

    for var in endogenous_vars:
        gold_var = dict(gold_sem.get(var) or {})
        cand_var = dict(cand_sem.get(var) or {})
        gold_eff = [str(v) for v in list(gold_var.get("effectiveParents") or [])]
        cand_eff = [str(v) for v in list(cand_var.get("effectiveParents") or [])]
        if gold_var != cand_var:
            changed_variable_count += 1
        total_effective_parent_symmetric_difference += len(set(gold_eff) ^ set(cand_eff))
        union_vars = sorted(
            set(mechanism_variables(gold_nodes[var])) | set(mechanism_variables(cand_nodes[var])),
            key=_natural_var_key,
        )
        if not union_vars:
            gold_out = int(evaluate_parsed_mechanism(gold_nodes[var], {}))
            cand_out = int(evaluate_parsed_mechanism(cand_nodes[var], {}))
            total_truth_table_hamming_distance += int(gold_out != cand_out)
            continue
        for bits in itertools.product((0, 1), repeat=len(union_vars)):
            env = {str(name): int(bit) for name, bit in zip(union_vars, bits)}
            gold_out = int(evaluate_parsed_mechanism(gold_nodes[var], env))
            cand_out = int(evaluate_parsed_mechanism(cand_nodes[var], env))
            total_truth_table_hamming_distance += int(gold_out != cand_out)

    return {
        "changedVariableCount": int(changed_variable_count),
        "totalTruthTableHammingDistance": int(total_truth_table_hamming_distance),
        "totalEffectiveParentSymmetricDifference": int(total_effective_parent_symmetric_difference),
    }


def _enumerate_root_assignments(root_vars: Sequence[str]) -> List[Dict[str, int]]:
    roots = [str(v) for v in root_vars]
    out: List[Dict[str, int]] = []
    for bits in itertools.product((0, 1), repeat=len(roots)):
        out.append({str(var): int(bit) for var, bit in zip(roots, bits)})
    return out


def _simulate_scm_under_assignment(
    *,
    payload: Dict[str, Any],
    root_assignment: Dict[str, int],
    experiment_assignments: Dict[str, int],
) -> Dict[str, int]:
    root_vars = [str(v) for v in list(payload.get("rootVariables") or [])]
    eval_topological_order = [str(v) for v in list(payload.get("evalTopologicalOrder") or [])]
    parsed_nodes_by_var = dict(payload.get("parsedNodesByVar") or {})

    assignment: Dict[str, int] = {}
    for var in root_vars:
        if var in experiment_assignments:
            assignment[var] = int(experiment_assignments[var])
        else:
            assignment[var] = int(root_assignment[var])
    for var in eval_topological_order:
        if var in assignment:
            continue
        if var in experiment_assignments:
            assignment[var] = int(experiment_assignments[var])
            continue
        assignment[var] = int(evaluate_parsed_mechanism(parsed_nodes_by_var[var], assignment))
    return assignment


def _normalize_single_do_experiment(
    experiment: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(experiment, dict):
        return None, "Experiment must be a JSON object"
    exp_type = str(experiment.get("type") or "").strip()
    if exp_type != "hard_do":
        return None, "Experiment type must be 'hard_do'"
    assignments = experiment.get("assignments")
    if not isinstance(assignments, dict) or len(assignments) != 1:
        return None, "Experiment assignments must contain exactly one variable"
    var, value = next(iter(assignments.items()))
    sval = str(var)
    if value not in (0, 1, "0", "1", False, True):
        return None, "Experiment assignment value must be binary 0/1"
    return {
        "type": "hard_do",
        "assignments": {sval: int(value)},
    }, None


def _canonical_experiment_sort_key(experiment: Dict[str, Any]) -> Tuple[Any, ...]:
    normalized, _ = _normalize_single_do_experiment(experiment)
    if normalized is None:
        return (10**9, "", 1)
    var, value = next(iter((normalized.get("assignments") or {}).items()))
    nat = _natural_var_sort_key(str(var))
    return (nat[0], nat[1] if len(nat) > 1 else 0, int(value))


def _pairwise_experiment_metrics(
    *,
    reference_payload: Dict[str, Any],
    alternative_payload: Dict[str, Any],
) -> Dict[str, Any]:
    variables = [str(v) for v in list(reference_payload.get("observedVariables") or [])]
    endogenous_set = {str(v) for v in list(reference_payload.get("endogenousVariables") or [])}
    root_assignments = _enumerate_root_assignments(reference_payload.get("rootVariables") or [])

    experiments = [
        {"type": "hard_do", "assignments": {str(var): int(bit)}}
        for var in sorted(variables, key=_natural_var_key)
        for bit in (0, 1)
    ]
    experiment_rows: List[Dict[str, Any]] = []
    for exp in experiments:
        exp_assignments = dict(exp.get("assignments") or {})
        target_var = next(iter(exp_assignments.keys()))
        disagreement_assignments = 0
        differing_cells = 0
        scored_cells = 0
        first_witness = None
        for root_assignment in root_assignments:
            ref_values = _simulate_scm_under_assignment(
                payload=reference_payload,
                root_assignment=root_assignment,
                experiment_assignments=exp_assignments,
            )
            alt_values = _simulate_scm_under_assignment(
                payload=alternative_payload,
                root_assignment=root_assignment,
                experiment_assignments=exp_assignments,
            )
            local_diffs: List[Dict[str, int]] = []
            for var in sorted(endogenous_set, key=_natural_var_key):
                if var == target_var:
                    continue
                scored_cells += 1
                ref_val = int(ref_values[var])
                alt_val = int(alt_values[var])
                if ref_val != alt_val:
                    differing_cells += 1
                    local_diffs.append(
                        {
                            "variable": str(var),
                            "reference": ref_val,
                            "alternative": alt_val,
                        }
                    )
            if local_diffs:
                disagreement_assignments += 1
                if first_witness is None:
                    first_witness = {
                        "rootAssignment": {str(k): int(v) for k, v in sorted(root_assignment.items(), key=lambda kv: _natural_var_sort_key(kv[0]))},
                        "differences": local_diffs,
                    }
        pair_disagreement_rate = (
            float(disagreement_assignments) / float(len(root_assignments))
            if root_assignments
            else 0.0
        )
        cell_difference_rate = (
            float(differing_cells) / float(scored_cells)
            if scored_cells > 0
            else 0.0
        )
        experiment_rows.append(
            {
                "experiment": exp,
                "pairSeparates": bool(disagreement_assignments > 0),
                "pairDisagreementRate": pair_disagreement_rate,
                "cellDifferenceRate": cell_difference_rate,
                "witness": first_witness,
            }
        )

    experiment_rows.sort(
        key=lambda row: (
            -float(row.get("pairDisagreementRate") or 0.0),
            -float(row.get("cellDifferenceRate") or 0.0),
            _canonical_experiment_sort_key(row.get("experiment") or {}),
        )
    )
    best = experiment_rows[0] if experiment_rows else None
    return {
        "experiments": experiment_rows,
        "bestExperiment": dict(best.get("experiment") or {}) if best else None,
        "bestPairSeparates": bool(best.get("pairSeparates")) if best else False,
        "bestPairDisagreementRate": float(best.get("pairDisagreementRate") or 0.0) if best else 0.0,
        "bestCellDifferenceRate": float(best.get("cellDifferenceRate") or 0.0) if best else 0.0,
        "bestWitness": _clone_unshared(best.get("witness")) if best else None,
    }


def _evaluate_variant_a_scm_id(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}
    gold = record_problem.get("goldAnswer", {}) or {}
    gold_scm_id = _gold_scm_id_payload(gold)
    gold_certification = str(gold_scm_id.get("certification") or "").strip().upper()
    if gold_certification not in {"UNIQUE", "AMBIGUOUS"}:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "SCM_ID task is missing gold certification metadata",
        }

    certification = str(answer.get("certification") or "").strip().upper()
    if certification not in {"UNIQUE", "AMBIGUOUS"}:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Expected top-level key 'certification' with value UNIQUE or AMBIGUOUS",
        }

    result: Dict[str, Any] = {
        "valid": True,
        "correct": bool(certification == gold_certification),
        "certification": certification,
        "goldCertification": gold_certification,
        "falseCertainty": bool(certification == "UNIQUE" and gold_certification == "AMBIGUOUS"),
    }

    if certification == "UNIQUE":
        return result

    witnesses = answer.get("witnesses")
    if not isinstance(witnesses, list) or len(witnesses) < 2:
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "AMBIGUOUS certification requires at least two witness SCMs",
            "certification": certification,
            "goldCertification": gold_certification,
        }

    allowed_operators = list(params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS))
    allow_constants = bool(params.get("allowConstants", True))
    signature = record_problem.get("signature", {}) or {}
    observed_variables = [str(v) for v in list(signature.get("variables", []))]
    root_vars = [str(v) for v in list(params.get("rootVariables") or [])]
    endogenous_vars = [str(v) for v in list(params.get("endogenousVariables") or [])]
    if not endogenous_vars:
        topological_order = [str(v) for v in list(params.get("topologicalOrder") or observed_variables)]
        endogenous_vars = [v for v in topological_order if v not in set(root_vars)]

    witness_evals: List[Dict[str, Any]] = []
    canonical_forms: List[str] = []
    for idx, witness in enumerate(witnesses, start=1):
        mechanisms = witness.get("mechanisms") if isinstance(witness, dict) else None
        if not isinstance(mechanisms, dict):
            return {
                "valid": False,
                "correct": False,
                "failureExplanation": f"Witness {idx} must be an object with a 'mechanisms' map",
                "certification": certification,
                "goldCertification": gold_certification,
            }
        witness_eval = _evaluate_variant_a_scm(problem, {"mechanisms": mechanisms})
        witness_row = {
            "index": idx,
            "valid": bool(witness_eval.get("valid")),
            "trainExact": bool(witness_eval.get("trainExact")),
            "heldoutExact": bool(witness_eval.get("heldoutExact")),
        }
        if not bool(witness_eval.get("valid")) or not bool(witness_eval.get("trainExact")):
            witness_row["failureExplanation"] = witness_eval.get("failureExplanation")
            witness_evals.append(witness_row)
            continue
        canonical_form, canonical_error = _canonicalize_scm_mechanism_map(
            mechanisms,
            endogenous_vars=endogenous_vars,
            observed_variables=observed_variables,
            allowed_operators=allowed_operators,
            allow_constants=allow_constants,
        )
        if canonical_form is None:
            witness_row["failureExplanation"] = canonical_error
            witness_evals.append(witness_row)
            continue
        witness_row["canonicalMechanismMap"] = canonical_form
        canonical_forms.append(canonical_form)
        witness_evals.append(witness_row)

    valid_witness_count = sum(1 for row in witness_evals if bool(row.get("valid")))
    train_exact_witness_count = sum(1 for row in witness_evals if bool(row.get("trainExact")))
    distinct_canonical_count = len(set(canonical_forms))
    result["witnessEvaluations"] = witness_evals
    result["witnessValidCount"] = int(valid_witness_count)
    result["witnessTrainExactCount"] = int(train_exact_witness_count)
    result["witnessDistinctCanonicalCount"] = int(distinct_canonical_count)

    if train_exact_witness_count < 2:
        return {
            **result,
            "valid": False,
            "correct": False,
            "failureExplanation": "AMBIGUOUS certification requires two train-consistent witness SCMs",
        }
    if distinct_canonical_count < 2:
        return {
            **result,
            "valid": False,
            "correct": False,
            "failureExplanation": "Witness SCMs must be canonically distinct",
        }

    result["correct"] = bool(gold_certification == "AMBIGUOUS")
    return result


def _evaluate_variant_a_scm_alt_exp(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}
    gold = record_problem.get("goldAnswer", {}) or {}
    gold_alt = _gold_scm_alt_exp_payload(gold)

    reference_mechanisms = gold_alt.get("referenceMechanisms")
    if not isinstance(reference_mechanisms, dict):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "SCM_ALT_EXP task is missing reference mechanism metadata",
        }

    alternative = answer.get("alternative")
    if not isinstance(alternative, dict) or not isinstance(alternative.get("mechanisms"), dict):
        return {
            "valid": False,
            "correct": False,
            "failureExplanation": "Expected top-level key 'alternative' with a 'mechanisms' map",
        }
    alt_mechanisms = dict(alternative.get("mechanisms") or {})

    alt_eval = _evaluate_variant_a_scm(problem, {"mechanisms": alt_mechanisms})
    result: Dict[str, Any] = {
        "valid": bool(alt_eval.get("valid")),
        "correct": False,
        "altValid": bool(alt_eval.get("valid")),
        "altTrainExact": bool(alt_eval.get("trainExact")),
        "altHeldoutExact": bool(alt_eval.get("heldoutExact")),
        "trainExact": bool(alt_eval.get("trainExact")),
        "heldoutExact": bool(alt_eval.get("heldoutExact")),
        "trainAccuracy": alt_eval.get("trainAccuracy"),
        "heldoutAccuracy": alt_eval.get("heldoutAccuracy"),
        "trainWorldExactAccuracy": alt_eval.get("trainWorldExactAccuracy"),
        "heldoutWorldExactAccuracy": alt_eval.get("heldoutWorldExactAccuracy"),
        "altTrainWorldExactAccuracy": alt_eval.get("trainWorldExactAccuracy"),
        "altHeldoutWorldExactAccuracy": alt_eval.get("heldoutWorldExactAccuracy"),
        "candidateStats": alt_eval.get("candidateStats"),
        "goldAstSize": alt_eval.get("goldAstSize"),
        "accGoldPlus": alt_eval.get("accGoldPlus"),
        "bloat": alt_eval.get("bloat"),
        "firstTrainMismatch": alt_eval.get("firstTrainMismatch"),
        "firstHeldoutMismatch": alt_eval.get("firstHeldoutMismatch"),
    }

    if not bool(alt_eval.get("valid")):
        result["failureExplanation"] = alt_eval.get("failureExplanation")
        result["parseError"] = alt_eval.get("parseError")
        return result

    alt_payload, alt_payload_error = _semantic_payload_for_scm_mechanisms(problem, alt_mechanisms)
    if alt_payload is None:
        result["valid"] = False
        result["failureExplanation"] = alt_payload_error
        return result
    ref_payload, ref_payload_error = _semantic_payload_for_scm_mechanisms(problem, reference_mechanisms)
    if ref_payload is None:
        return {
            **result,
            "valid": False,
            "failureExplanation": ref_payload_error or "Invalid reference SCM payload",
        }

    alt_semantically_distinct = bool(
        alt_payload.get("semanticSignature") != ref_payload.get("semanticSignature")
    )
    result["altSemanticallyDistinct"] = alt_semantically_distinct
    result["altSuccess"] = bool(
        result["altValid"] and result["altTrainExact"] and alt_semantically_distinct
    )
    result["referenceRole"] = gold_alt.get("referenceRole")

    experiment = answer.get("experiment")
    normalized_experiment, experiment_error = _normalize_single_do_experiment(experiment)
    result["experimentValid"] = normalized_experiment is not None
    result["experiment"] = normalized_experiment if normalized_experiment is not None else experiment

    witness = answer.get("witness")
    result["witnessProvided"] = isinstance(witness, dict)

    if not result["altSuccess"]:
        result["failureExplanation"] = (
            alt_eval.get("failureExplanation")
            or (
                "Alternative SCM must be semantically distinct from the presented reference mechanism"
                if not alt_semantically_distinct
                else "Alternative SCM must exactly fit the training worlds"
            )
        )
        return result

    result["valid"] = True

    if normalized_experiment is None:
        result["failureExplanation"] = experiment_error
        return result

    experiment_metrics = _pairwise_experiment_metrics(
        reference_payload=ref_payload,
        alternative_payload=alt_payload,
    )
    proposed_key = json.dumps(normalized_experiment, sort_keys=True, separators=(",", ":"))
    proposed_row = None
    for row in experiment_metrics.get("experiments") or []:
        row_exp = row.get("experiment") or {}
        row_key = json.dumps(row_exp, sort_keys=True, separators=(",", ":"))
        if row_key == proposed_key:
            proposed_row = row
            break
    if proposed_row is None:
        result["failureExplanation"] = "Proposed experiment is not in the admissible single-variable hard-do menu"
        return result

    result["pairSeparates"] = bool(proposed_row.get("pairSeparates"))
    result["pairDisagreementRate"] = float(proposed_row.get("pairDisagreementRate") or 0.0)
    result["cellDifferenceRate"] = float(proposed_row.get("cellDifferenceRate") or 0.0)
    best_rate = float(experiment_metrics.get("bestPairDisagreementRate") or 0.0)
    if best_rate > 0:
        result["experimentOptimality"] = float(result["pairDisagreementRate"]) / best_rate
    else:
        result["experimentOptimality"] = 0.0
    result["experimentResolves"] = bool(result["pairSeparates"])
    result["bestPairwiseExperiment"] = experiment_metrics.get("bestExperiment")
    result["bestPairwiseDisagreementRate"] = best_rate
    result["bestPairwiseCellDifferenceRate"] = float(experiment_metrics.get("bestCellDifferenceRate") or 0.0)

    witness_valid = False
    if isinstance(witness, dict):
        root_vars = [str(v) for v in list(ref_payload.get("rootVariables") or [])]
        root_assignment = witness.get("rootAssignment")
        differences = witness.get("differences")
        if isinstance(root_assignment, dict) and isinstance(differences, list) and differences:
            normalized_root_assignment = {
                str(k): int(v)
                for k, v in root_assignment.items()
                if v in (0, 1, "0", "1", False, True)
            }
            if set(normalized_root_assignment.keys()) == set(root_vars):
                exp_assignments = dict(normalized_experiment.get("assignments") or {})
                ref_values = _simulate_scm_under_assignment(
                    payload=ref_payload,
                    root_assignment=normalized_root_assignment,
                    experiment_assignments=exp_assignments,
                )
                alt_values = _simulate_scm_under_assignment(
                    payload=alt_payload,
                    root_assignment=normalized_root_assignment,
                    experiment_assignments=exp_assignments,
                )
                target_var = next(iter(exp_assignments.keys()))
                valid_differences = 0
                for diff in differences:
                    if not isinstance(diff, dict):
                        continue
                    var = str(diff.get("variable") or "")
                    if not var or var == target_var or var not in set(ref_payload.get("endogenousVariables") or []):
                        continue
                    ref_claim = diff.get("reference")
                    alt_claim = diff.get("alternative")
                    if ref_claim not in (0, 1, "0", "1", False, True):
                        continue
                    if alt_claim not in (0, 1, "0", "1", False, True):
                        continue
                    ref_val = int(ref_values[var])
                    alt_val = int(alt_values[var])
                    if ref_val == int(ref_claim) and alt_val == int(alt_claim) and ref_val != alt_val:
                        valid_differences += 1
                witness_valid = bool(valid_differences > 0)
    result["witnessValid"] = bool(witness_valid)
    result["jointSuccess"] = bool(
        result["altSuccess"] and result["experimentValid"] and result["pairSeparates"] and result["witnessValid"]
    )
    result["correct"] = bool(result["jointSuccess"])
    if not result["correct"] and not result.get("failureExplanation"):
        if not result["pairSeparates"]:
            result["failureExplanation"] = "Proposed experiment does not separate the presented reference and returned alternative"
        elif not result["witnessValid"]:
            result["failureExplanation"] = "Witness does not correctly demonstrate a separating prediction difference"
    return result

def _evaluate_cind_answer(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    variant = _resolve_variant(task.get("taskName", TASK_CIND_A_Y))

    if variant in {"A_Y", "A_OOD"}:
        return _evaluate_variant_a_y_like(problem, answer, variant=variant)
    if variant == "A_P":
        return _evaluate_variant_a_p(problem, answer)
    if variant == "A_SCM":
        return _evaluate_variant_a_scm(problem, answer)
    if variant == "A_SCM_ROOT_UNKNOWN":
        return _evaluate_variant_a_scm_root_unknown(problem, answer)
    if variant == "A_SCM_ID":
        return _evaluate_variant_a_scm_id(problem, answer)
    if variant == "A_SCM_ALT_EXP":
        return _evaluate_variant_a_scm_alt_exp(problem, answer)

    return {
        "valid": False,
        "correct": False,
        "failureExplanation": f"Unsupported CIND variant: {variant}",
    }


def _compact_world_lines(
    world: Dict[str, Any],
    input_variables: List[str],
    target_variable: str,
    structured_interventions: bool = False,
) -> List[str]:
    wid = world.get("worldId", "unknown")
    extra = world.get("extra") or {}
    lines: List[str]
    if structured_interventions:
        structured = _normalize_structured_intervention_fields(world)
        if isinstance(extra.get("do"), str) and str(extra.get("do")).strip():
            do_txt = str(extra.get("do"))
        else:
            do_txt = _format_do_mode(
                structured["InterventionMode"],
                structured["InterventionTargetsConstant"],
                structured["InterventionTargetsAssigned"],
            )
        lines = [f"WorldId: {wid}", f"Intervention: {do_txt}"]
        lines.append(f"InterventionMode: {structured['InterventionMode']}")
        lines.append(
            "InterventionTargetsAssigned: "
            + json.dumps(list(structured["InterventionTargetsAssigned"]))
        )
        lines.append(
            "InterventionTargetsConstant: "
            + json.dumps(dict(structured["InterventionTargetsConstant"]), sort_keys=True)
        )
        lines.append("InterventionTargetsAll: " + json.dumps(list(structured["InterventionTargetsAll"])))
    else:
        if isinstance(extra.get("do"), str) and str(extra.get("do")).strip():
            do_txt = str(extra.get("do"))
        else:
            ints = ((world.get("interventions") or [{}])[0] or {}).get("assignments", {})
            do_txt = _format_do({str(k): int(v) for k, v in ints.items()})
        mode = str(extra.get("interventionMode", "")).strip()
        lines = [f"WorldId: {wid}", f"Intervention: {do_txt}"]
        if mode:
            lines.append(f"InterventionMode: {mode}")

    lines.append("Rows:")
    rows = (((world.get("extra") or {}).get("rows")) or [])
    row_order = list(input_variables)
    if target_variable and target_variable not in row_order:
        row_order.append(target_variable)
    for row in rows:
        if not isinstance(row, dict):
            continue
        unit_id = str(row.get("unitId", "?"))
        values = row.get("values") if isinstance(row.get("values"), dict) else {}
        parts = [f"{v}={int(values[v])}" for v in row_order if v in values]
        lines.append(f"- {unit_id}: " + " ".join(parts))
    return lines


def _ordered_subset(order: List[str], values: List[str]) -> List[str]:
    seen: Set[str] = set()
    value_set = {str(v) for v in values}
    out: List[str] = []
    for item in order:
        token = str(item)
        if token in value_set and token not in seen:
            out.append(token)
            seen.add(token)
    for item in values:
        token = str(item)
        if token not in seen:
            out.append(token)
            seen.add(token)
    return out


def _build_problem_instance_block(
    task_name: str,
    variant: str,
    target: str,
    variables: List[str],
    input_variables: List[str],
    allowed_ops: List[str],
    task_query: str,
    train_worlds: List[Dict[str, Any]],
    heldout_worlds: List[Dict[str, Any]],
    output_example: str,
    root_variables: Optional[List[str]] = None,
    endogenous_variables: Optional[List[str]] = None,
    topological_order: Optional[List[str]] = None,
    topological_layers: Optional[List[List[str]]] = None,
    panel_semantics: Optional[bool] = None,
    hide_topological_order: bool = False,
    partial_topological_order: bool = False,
    reference_mechanisms: Optional[Dict[str, Any]] = None,
    hide_root_roles: bool = False,
    root_count: Optional[int] = None,
) -> str:
    if variant in {"A_SCM", "A_SCM_ROOT_UNKNOWN", "A_SCM_ID", "A_SCM_ALT_EXP"}:
        if hide_topological_order:
            allowed_mechanism_vars_line = (
                "- AllowedMechanismVariables: use only variable names in ObservedVariables."
            )
        elif partial_topological_order:
            allowed_mechanism_vars_line = (
                "- AllowedMechanismVariables: for each equation of V, only variables from the same or earlier TopologicalLayers, excluding V itself."
            )
        else:
            allowed_mechanism_vars_line = (
                "- AllowedMechanismVariables: for each equation of V, only predecessor variables in TopologicalOrder"
            )
    else:
        allowed_mechanism_vars_line = f"- AllowedMechanismVariables: {', '.join(input_variables)}"

    lines: List[str] = [
        "Problem Metadata:",
        f"- Task: {task_name}",
        f"- Variant: {variant}",
        f"- InputVariables: {', '.join(input_variables)}",
        f"- ObservedVariables: {', '.join(variables)}",
        allowed_mechanism_vars_line,
        f"- AllowedOperators: {', '.join(allowed_ops)}",
        f"- TrainWorldCount: {len(train_worlds)}",
        f"- HeldoutWorldCount: {len(heldout_worlds)}",
        f"- Query: {task_query}",
    ]
    if variant not in {"A_SCM", "A_SCM_ROOT_UNKNOWN", "A_SCM_ID", "A_SCM_ALT_EXP"}:
        lines.insert(3, f"- Target: {target}")
    if root_count is not None and hide_root_roles:
        lines.append(f"- RootCount: {int(root_count)}")
    elif root_variables:
        lines.append(f"- RootVariables: {', '.join(root_variables)}")
    if endogenous_variables and not hide_root_roles:
        lines.append(f"- EndogenousVariables: {', '.join(endogenous_variables)}")
    if topological_layers and partial_topological_order:
        rendered_layers = ["{" + ", ".join(layer) + "}" for layer in topological_layers]
        lines.append(f"- TopologicalLayers: {' < '.join(rendered_layers)}")
    elif topological_order and not hide_topological_order:
        lines.append(f"- TopologicalOrder: {', '.join(topological_order)}")
    if panel_semantics is not None:
        lines.append(f"- PanelSemantics: {'true' if bool(panel_semantics) else 'false'}")
    if variant in {"A_SCM", "A_SCM_ROOT_UNKNOWN", "A_SCM_ID", "A_SCM_ALT_EXP"}:
        lines.append("- InterventionModes: hard_constant (fixed do value) and hard_assigned (value assigned per-row)")
        lines.append(
            "- Do NOT parse the textual Intervention string; use the structured "
            "InterventionMode, InterventionTargetsAssigned, and InterventionTargetsConstant fields."
        )
        if bool(panel_semantics):
            lines.append(
                "PanelSemantics: Unit IDs refer to the same unit across worlds. "
                "A given unit's exogenous/root context variables are intended to be stable "
                "across worlds unless that variable is explicitly intervened."
            )
        else:
            lines.append(
                "PanelSemantics: Unit IDs are local row labels within each world. "
                "Do not assume u00 in different worlds refers to the same unit."
            )
        if hide_topological_order:
            lines.append(
                "TopologicalOrder is intentionally hidden in this prompt variant. "
                "Infer one acyclic mechanism map that globally explains all training worlds."
            )
        elif partial_topological_order:
            lines.append(
                "TopologicalLayers disclose only block-level precedence constraints. "
                "A mechanism may use variables from the same displayed block or any earlier block, but never from a later block."
            )
        if hide_root_roles:
            lines.append(
                "Root variable identities are intentionally hidden in this prompt variant. "
                "Infer which observed variables are roots, and provide mechanisms only for the remaining variables."
            )
    if variant == "A_SCM_ALT_EXP":
        lines.append("- AdmissibleExperimentFamily: single-variable hard_do with exactly one binary assignment do(V=0) or do(V=1)")
        lines.append(
            "- WitnessRequirement: provide one complete RootVariables assignment and at least one non-intervened EndogenousVariable whose prediction differs between the presented reference SCM and your returned alternative under the proposed experiment."
        )
        if reference_mechanisms:
            lines.extend(["", "ReferenceSCM:"])
            for var in endogenous_variables:
                if var in reference_mechanisms:
                    lines.append(f"- {var}: {str(reference_mechanisms[var])}")
    lines.extend(
        [
            "",
            "OutputSchemaExample:",
            output_example,
            "",
            "Training Worlds:",
        ]
    )

    for world in train_worlds:
        lines.extend(
            _compact_world_lines(
                world,
                input_variables,
                target,
                structured_interventions=(variant in {"A_SCM", "A_SCM_ROOT_UNKNOWN", "A_SCM_ID", "A_SCM_ALT_EXP"}),
            )
        )
        lines.append("")

    if heldout_worlds:
        lines.extend(
            [
                "Held-out Worlds:",
                "- Held-out interventions exist and are withheld during fitting.",
                "- Use one invariant mechanism inferred from training worlds.",
            ]
        )

    return "\n".join(lines).strip()


def _format_icl_example(icl_raw: Any, variant: str, output_example: str) -> str:
    default_by_variant = {
        "A_Y": "{" + "\"parents\":[\"X1\"],\"mechanism\":\"(not X1)\"" + "}",
        "A_P": "{" + "\"parents\":[\"X1\"]" + "}",
        "A_SCM": "{" + "\"mechanisms\":{\"X3\":\"(not X1)\",\"X4\":\"(xor X2 X3)\"}" + "}",
        "A_SCM_ROOT_UNKNOWN": "{" + "\"roots\":[\"X1\",\"X4\",\"X7\"],\"mechanisms\":{\"X2\":\"(not X1)\",\"X3\":\"(xor X2 X4)\"}" + "}",
        "A_SCM_ID": "{" + "\"certification\":\"UNIQUE\"" + "}",
        "A_SCM_ALT_EXP": "{" + "\"alternative\":{\"mechanisms\":{\"X3\":\"(not X1)\",\"X4\":\"(xor X2 X3)\"}},\"experiment\":{\"type\":\"hard_do\",\"assignments\":{\"X2\":1}},\"witness\":{\"rootAssignment\":{\"X1\":0},\"differences\":[{\"variable\":\"X4\",\"reference\":0,\"alternative\":1}]}}" + "}",
        "A_OOD": "{" + "\"parents\":[\"X1\"],\"mechanism\":\"(not X1)\"" + "}",
    }

    default_output = default_by_variant.get(variant, output_example)
    lines = [
        "Format-only ICL (structure only; this is not the current task instance):",
        "- Always return exactly one JSON object on one line.",
        "- Use exactly the required top-level keys for the task variant.",
        "Example output:",
        default_output,
    ]

    if icl_raw is None:
        return "\n".join(lines)

    custom = ""
    if isinstance(icl_raw, str):
        custom = icl_raw.strip()
    elif isinstance(icl_raw, dict):
        parts: List[str] = []
        title = str(icl_raw.get("title", "Additional Example")).strip() or "Additional Example"
        parts.append(f"{title}:")
        input_block = icl_raw.get("problem_instance") or icl_raw.get("input")
        if input_block:
            parts.append("Input:")
            parts.append(str(input_block).strip())
        output_block = icl_raw.get("answer") or icl_raw.get("output")
        if output_block:
            parts.append("Output:")
            if isinstance(output_block, (dict, list)):
                parts.append(json.dumps(output_block, separators=(",", ":")))
            else:
                parts.append(str(output_block).strip())
        custom = "\n".join(parts).strip()

    if custom:
        lines.extend(["", custom])

    return "\n".join(lines)

def _render_template_placeholders(template_text: str, values: Dict[str, str]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _build_cind_prompt(problem: Dict[str, Any], templates: Dict[str, str]) -> Tuple[str, str]:
    record_problem = problem.get("problem", problem)
    task = record_problem.get("task", {}) or {}
    params = task.get("parameters", {}) or {}

    task_name = str(task.get("taskName", TASK_CIND_A_Y))
    variant = _resolve_variant(task_name)
    target = str(params.get("target", "Y"))
    variables = list(params.get("variables", []))
    allowed_ops = list(params.get("allowedOperators") or list(DEFAULT_ALLOWED_OPERATORS))
    input_variables = list(params.get("inputVariables") or [])
    root_variables = [str(v) for v in list(params.get("rootVariables") or [])]
    endogenous_variables = [str(v) for v in list(params.get("endogenousVariables") or [])]
    topological_order = [str(v) for v in list(params.get("topologicalOrder") or [])]
    topological_layers = _normalize_topological_layers(
        params.get("topologicalLayers"),
        topological_order=topological_order,
    )
    panel_semantics = params.get("PanelSemantics")
    prompt_variant = _normalize_scm_prompt_variant(
        params.get("scmPromptVariant") or params.get("promptVariant") or SCM_PROMPT_VARIANT_ORDERED
    )
    scm_style_variant = variant in {"A_SCM", "A_SCM_ROOT_UNKNOWN", "A_SCM_ID", "A_SCM_ALT_EXP"}
    hide_root_roles = variant == "A_SCM_ROOT_UNKNOWN"
    hide_topological_order = scm_style_variant and _scm_variant_hides_topological_order(prompt_variant)
    partial_topological_order = scm_style_variant and _scm_variant_uses_partial_order(prompt_variant)
    prompt_variable_order = [str(v) for v in list(params.get("promptVariableOrder") or [])]
    gold = record_problem.get("goldAnswer", {}) or {}
    gold_alt = _gold_scm_alt_exp_payload(gold) if variant == "A_SCM_ALT_EXP" else {}
    reference_mechanisms = gold_alt.get("referenceMechanisms") if isinstance(gold_alt, dict) else None

    train_worlds, heldout_worlds = _worlds_by_split(record_problem)

    if not variables:
        sig = record_problem.get("signature", {}) or {}
        variables = list(sig.get("variables", []))
    else:
        sig = record_problem.get("signature", {}) or {}

    if panel_semantics is None:
        obs_struct = str(sig.get("observationStructure", "")).strip().lower()
        if obs_struct == "panel_same_units_across_worlds":
            panel_semantics = True
        elif obs_struct:
            panel_semantics = False

    if not variables:
        variables = []

    if not input_variables:
        input_variables = [v for v in variables if str(v) != target]

    display_order = list(variables)
    if prompt_variable_order:
        order_set = {str(v) for v in prompt_variable_order}
        variable_set = {str(v) for v in variables}
        if order_set == variable_set:
            display_order = list(prompt_variable_order)

    if scm_style_variant:
        input_variables = _ordered_subset(display_order, [str(v) for v in input_variables])
        root_variables = _ordered_subset(display_order, [str(v) for v in root_variables])
        endogenous_variables = _ordered_subset(display_order, [str(v) for v in endogenous_variables])
        variables = _ordered_subset(display_order, [str(v) for v in variables])

    output_schema = {
        "A_Y": '{"parents":["X1","X2"],"mechanism":"(xor X1 X2)"}',
        "A_P": '{"parents":["X1","X2"]}',
        "A_SCM": '{"mechanisms":{"X3":"...","X4":"(xor X1 X2)"}}',
        "A_SCM_ROOT_UNKNOWN": '{"roots":["X1","X4","X7"],"mechanisms":{"X2":"...","X3":"(xor X1 X2)"}}',
        "A_SCM_ID": '{"certification":"UNIQUE"}',
        "A_SCM_ALT_EXP": '{"alternative":{"mechanisms":{"X3":"...","X4":"(xor X1 X2)"}},"experiment":{"type":"hard_do","assignments":{"X2":1}},"witness":{"rootAssignment":{"X1":0},"differences":[{"variable":"X4","reference":0,"alternative":1}]}}',
        "A_OOD": '{"parents":["X1","X2"],"mechanism":"(xor X1 X2)"}',
    }
    output_example = output_schema.get(variant, output_schema["A_Y"])

    task_query = str(task.get("query", "")).strip()

    problem_instance_block = _build_problem_instance_block(
        task_name=task_name,
        variant=variant,
        target=target,
        variables=variables,
        input_variables=input_variables,
        allowed_ops=allowed_ops,
        task_query=task_query,
        train_worlds=train_worlds,
        heldout_worlds=heldout_worlds,
        output_example=output_example,
        root_variables=root_variables,
        endogenous_variables=endogenous_variables,
        topological_order=(None if hide_topological_order else topological_order),
        topological_layers=(topological_layers if partial_topological_order else None),
        panel_semantics=bool(panel_semantics) if panel_semantics is not None else None,
        hide_topological_order=hide_topological_order,
        partial_topological_order=partial_topological_order,
        reference_mechanisms=(reference_mechanisms if isinstance(reference_mechanisms, dict) else None),
        hide_root_roles=hide_root_roles,
        root_count=(len(root_variables) if hide_root_roles else None),
    )

    icl_block = _format_icl_example(params.get("iclExample"), variant, output_example)

    placeholders = {
        "TASK_NAME": task_name,
        "VARIANT": variant,
        "TARGET_VARIABLE": target,
        "VARIABLES_LIST": ", ".join(variables),
        "INPUT_VARIABLES_LIST": ", ".join(input_variables),
        "ALLOWED_OPERATORS": ", ".join(allowed_ops),
        "ROOT_VARIABLES": ", ".join(root_variables),
        "ENDOGENOUS_VARIABLES": ", ".join(endogenous_variables),
        "TOPOLOGICAL_ORDER": "(hidden)"
        if hide_topological_order
        else ", ".join(topological_order),
        "TRAIN_WORLD_COUNT": str(len(train_worlds)),
        "HELDOUT_WORLD_COUNT": str(len(heldout_worlds)),
        "TASK_QUERY": task_query,
        "OUTPUT_SCHEMA_EXAMPLE": output_example,
        "PROBLEM_INSTANCE": problem_instance_block,
        "ICL_EXAMPLE": icl_block,
    }

    task_template = (templates.get("task_prompt") or "").strip()
    if task_template:
        task_body = _render_template_placeholders(task_template, placeholders)
        if "{{PROBLEM_INSTANCE}}" not in task_template:
            task_body = task_body + "\n\n## Problem Instance\n" + problem_instance_block
    else:
        fallback_lines = [
            f"Task: {task_name} ({variant})",
            f"Target variable: {target}",
            f"Allowed operators: {', '.join(allowed_ops)}",
            "",
            "## Problem Instance",
            problem_instance_block,
        ]
        task_body = "\n".join(fallback_lines).strip()

    suffix_template = (templates.get("suffix_prompt") or "").strip()
    suffix_body = _render_template_placeholders(suffix_template, placeholders) if suffix_template else ""

    if suffix_body:
        prompt = (task_body + "\n\n" + suffix_body).strip()
    else:
        prompt = task_body.strip()

    system_template = (templates.get("system_prompt") or "").strip()
    system_prompt = _render_template_placeholders(system_template, placeholders)
    return prompt, system_prompt

def _task_generate_fn(task_name: str):
    def _generate(seed: int, instance_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        return _build_cind_problem(seed, instance_id, task_name=task_name, task_config=task_config)

    return _generate


def _task_eval_fn(_task_name: str):
    def _evaluate(problem: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_cind_answer(problem, answer)

    return _evaluate


def _build_task_definitions(include_aliases: bool = True) -> List[CausalTaskDefinition]:
    defs = [
        CausalTaskDefinition(
            name=TASK_CIND_A_Y,
            description="Family A, variant A-Y: infer Pa(Y) and mechanism f_Y",
            generate_problem=_task_generate_fn(TASK_CIND_A_Y),
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_Y),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_P,
            description="Family A, variant A-P: parent discovery for Y",
            generate_problem=_task_generate_fn(TASK_CIND_A_P),
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_P),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_SCM,
            description="Family A, variant A-SCM: full SCM induction with endogenous-variable simulation",
            generate_problem=_task_generate_fn(TASK_CIND_A_SCM),
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_SCM_ROOT_UNKNOWN,
            description="Family A, variant A-SCM-ROOT-UNKNOWN: full SCM induction when root variable identities are hidden",
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ROOT_UNKNOWN),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_SCM_ID,
            description="Family A, variant A-SCM-ID: certify whether training data uniquely identifies one SCM",
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ID),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_SCM_ALT_EXP,
            description="Family A, variant A-SCM-ALT-EXP: construct a semantically distinct alternative SCM and propose a distinguishing experiment",
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ALT_EXP),
        ),
        CausalTaskDefinition(
            name=TASK_CIND_A_OOD,
            description="Family A, variant A-OOD: held-out intervention generalization",
            generate_problem=_task_generate_fn(TASK_CIND_A_OOD),
            build_prompt=_build_cind_prompt,
            extract_answer=_extract_cind_answer,
            evaluate_answer=_task_eval_fn(TASK_CIND_A_OOD),
        ),
    ]

    if include_aliases:
        defs.extend(
            [
                CausalTaskDefinition(
                    name=TASK_CIND_A_Y_ALIAS,
                    description="Alias of CIND_A_Y",
                    generate_problem=_task_generate_fn(TASK_CIND_A_Y_ALIAS),
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_Y_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_P_ALIAS,
                    description="Alias of CIND_A_P",
                    generate_problem=_task_generate_fn(TASK_CIND_A_P_ALIAS),
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_P_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_SCM_ALIAS,
                    description="Alias of CIND_A_SCM",
                    generate_problem=_task_generate_fn(TASK_CIND_A_SCM_ALIAS),
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_SCM_ROOT_UNKNOWN_ALIAS,
                    description="Alias of CIND_A_SCM_ROOT_UNKNOWN",
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ROOT_UNKNOWN_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_SCM_ID_ALIAS,
                    description="Alias of CIND_A_SCM_ID",
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ID_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_SCM_ALT_EXP_ALIAS,
                    description="Alias of CIND_A_SCM_ALT_EXP",
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_SCM_ALT_EXP_ALIAS),
                ),
                CausalTaskDefinition(
                    name=TASK_CIND_A_OOD_ALIAS,
                    description="Alias of CIND_A_OOD",
                    generate_problem=_task_generate_fn(TASK_CIND_A_OOD_ALIAS),
                    build_prompt=_build_cind_prompt,
                    extract_answer=_extract_cind_answer,
                    evaluate_answer=_task_eval_fn(TASK_CIND_A_OOD_ALIAS),
                ),
            ]
        )

    return defs


def register_cind_family_tasks(
    registry: CausalTaskRegistry = DEFAULT_CAUSAL_TASK_REGISTRY,
    overwrite: bool = False,
    include_aliases: bool = True,
) -> List[str]:
    """Register Family A CIND tasks into a registry."""
    registered: List[str] = []
    for task_def in _build_task_definitions(include_aliases=include_aliases):
        existing = registry.get(task_def.name)
        if existing is not None and not overwrite:
            continue
        registry.register(task_def, overwrite=overwrite)
        registered.append(task_def.name)
    return registered


def ensure_cind_family_tasks_registered(
    registry: CausalTaskRegistry = DEFAULT_CAUSAL_TASK_REGISTRY,
) -> None:
    """Idempotently ensure Family A CIND tasks are available."""
    register_cind_family_tasks(registry=registry, overwrite=False, include_aliases=True)
