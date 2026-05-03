"""Helpers for compact causal runtime storage.

These helpers are pure conversions only. They do not change runner behavior.
The intent is to support a future migration from verbose problem records with
embedded ``llmResults`` into a compact runtime format with:

- compact problem/world encoding
- one current result slot per model
- lossless expansion back to the current in-memory problem form

The current module is intentionally conservative:

- problem/task/gold payloads are preserved exactly
- world rows are compacted into bitstrings
- legacy ``llmResults`` collapse into ``modelResults`` by model, preferring the last valid result
- build metadata is left untouched at the record level
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional


RUNTIME_SCHEMA_VERSION = "cind-a-scm-runtime-v2"
PROBLEM_ENCODING_VERSION = "scm-panel-binary-rows-v1"
RESULT_ENCODING_VERSION = "scm-current-model-results-v1"

_VAR_TOKEN_RE = re.compile(r"^([A-Za-z_]+)(\d+)$")
_WORLD_EXTRA_SUMMARY_KEYS = (
    "do",
    "interventionMode",
    "InterventionMode",
    "InterventionTargetsAssigned",
    "InterventionTargetsConstant",
    "InterventionTargetsAll",
)
_WORLD_EXTRA_COMPACT_KEYS = _WORLD_EXTRA_SUMMARY_KEYS + ("rows", "split", "envProfile")


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _natural_var_key(token: Any) -> tuple[Any, ...]:
    text = str(token)
    match = _VAR_TOKEN_RE.match(text)
    if match:
        return (match.group(1), int(match.group(2)))
    return (text, -1)


def _is_binary_domains_dict(domains: Any, variables: List[str]) -> bool:
    if not isinstance(domains, dict):
        return False
    if set(str(k) for k in domains.keys()) != {str(v) for v in variables}:
        return False
    for var in variables:
        values = domains.get(var)
        if not isinstance(values, list) or values != [0, 1]:
            return False
    return True


def _compact_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
    out = _clone(signature)
    variables = [str(v) for v in list(out.get("variables") or [])]
    if _is_binary_domains_dict(out.get("domains"), variables):
        out["domains"] = "binary"
    return out


def _expand_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
    out = _clone(signature)
    variables = [str(v) for v in list(out.get("variables") or [])]
    if out.get("domains") == "binary":
        out["domains"] = {var: [0, 1] for var in variables}
    return out


def _shared_panel_units(worlds: List[Dict[str, Any]]) -> Optional[List[str]]:
    if not worlds:
        return None
    first = worlds[0].get("domain")
    if not isinstance(first, list):
        return None
    first_units = [str(v) for v in first]
    for world in worlds[1:]:
        domain = world.get("domain")
        if not isinstance(domain, list) or [str(v) for v in domain] != first_units:
            return None
    return first_units


def _encode_bit(value: Any) -> str:
    return "1" if int(value) else "0"


def _encode_rows(rows: Iterable[Dict[str, Any]], variables: List[str], panel_units: List[str]) -> List[str]:
    by_unit: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        unit_id = str(row.get("unitId", ""))
        values = row.get("values")
        if unit_id and isinstance(values, dict):
            by_unit[unit_id] = values

    encoded: List[str] = []
    for unit_id in panel_units:
        values = by_unit.get(unit_id)
        if values is None:
            raise ValueError(f"Missing row for unit {unit_id!r}")
        encoded.append("".join(_encode_bit(values[var]) for var in variables))
    return encoded


def _decode_rows(bit_rows: Iterable[str], variables: List[str], panel_units: List[str]) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    bit_rows_list = list(bit_rows)
    if len(bit_rows_list) != len(panel_units):
        raise ValueError("Bit row count does not match panel unit count")

    width = len(variables)
    for unit_id, bits in zip(panel_units, bit_rows_list):
        bit_text = str(bits)
        if len(bit_text) != width:
            raise ValueError("Bit row width does not match variable count")
        if any(ch not in {"0", "1"} for ch in bit_text):
            raise ValueError("Bit rows must contain only 0/1")
        values = {var: int(bit) for var, bit in zip(variables, bit_text)}
        rows_out.append({"unitId": str(unit_id), "values": values})
    return rows_out


def _derive_intervention_summary(world: Dict[str, Any]) -> Dict[str, Any]:
    extra = world.get("extra") or {}
    summary: Dict[str, Any] = {}
    for key in _WORLD_EXTRA_SUMMARY_KEYS:
        if key in extra:
            summary[key] = _clone(extra[key])
    if summary:
        return summary

    interventions = world.get("interventions") or []
    first = interventions[0] if isinstance(interventions, list) and interventions else {}
    if not isinstance(first, dict):
        first = {}

    mode = str(first.get("mode") or "").strip().lower()
    assignments_raw = first.get("assignments") or {}
    assignments = (
        {str(k): int(v) for k, v in assignments_raw.items()}
        if isinstance(assignments_raw, dict)
        else {}
    )
    targets_raw = first.get("targets") or []
    targets = (
        sorted({str(v) for v in targets_raw}, key=_natural_var_key)
        if isinstance(targets_raw, list)
        else []
    )

    if not targets and not assignments:
        return {
            "do": "none",
            "interventionMode": "hard_constant",
            "InterventionMode": "none",
            "InterventionTargetsAssigned": [],
            "InterventionTargetsConstant": {},
            "InterventionTargetsAll": [],
        }

    if mode == "hard_assigned":
        parts = [f"{target}=assigned_per_row" for target in targets]
        do_text = "do(" + ", ".join(parts) + ")" if parts else "none"
        return {
            "do": do_text,
            "interventionMode": "hard_assigned",
            "InterventionMode": "hard_assigned",
            "InterventionTargetsAssigned": list(targets),
            "InterventionTargetsConstant": {},
            "InterventionTargetsAll": list(targets),
        }

    do_text = "none"
    if assignments:
        parts = [f"{key}={value}" for key, value in sorted(assignments.items())]
        do_text = "do(" + ", ".join(parts) + ")"
    return {
        "do": do_text,
        "interventionMode": "hard_constant",
        "InterventionMode": "hard_constant" if assignments else "none",
        "InterventionTargetsAssigned": [],
        "InterventionTargetsConstant": dict(assignments),
        "InterventionTargetsAll": sorted(assignments.keys(), key=_natural_var_key),
    }


def _compact_world(
    world: Dict[str, Any],
    variables: List[str],
    shared_panel_units: Optional[List[str]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"worldId": str(world.get("worldId", ""))}

    if "observationMode" in world:
        out["observationMode"] = world.get("observationMode")

    if world.get("predicates"):
        out["predicates"] = _clone(world.get("predicates"))
    if world.get("events"):
        out["events"] = _clone(world.get("events"))
    if world.get("targetLabels"):
        out["targetLabels"] = _clone(world.get("targetLabels"))

    interventions = world.get("interventions")
    if isinstance(interventions, list) and len(interventions) == 1 and isinstance(interventions[0], dict):
        out["intervention"] = _clone(interventions[0])
    elif interventions:
        out["interventions"] = _clone(interventions)

    domain = world.get("domain")
    if not (shared_panel_units is not None and isinstance(domain, list) and [str(v) for v in domain] == shared_panel_units):
        if isinstance(domain, list):
            out["domain"] = [str(v) for v in domain]
            out["domainSize"] = int(world.get("domainSize", len(domain)))

    extra = world.get("extra") or {}
    if "split" in extra:
        out["split"] = extra.get("split")
    if "envProfile" in extra:
        out["envProfile"] = _clone(extra.get("envProfile"))
    rows = extra.get("rows")
    if isinstance(rows, list) and shared_panel_units is not None:
        out["rows"] = _encode_rows(rows, variables, shared_panel_units)
    elif isinstance(rows, list):
        out["rowsVerbose"] = _clone(rows)

    summary = _derive_intervention_summary(world)
    if summary:
        out["interventionSummary"] = summary

    residual_extra = {k: _clone(v) for k, v in extra.items() if k not in _WORLD_EXTRA_COMPACT_KEYS}
    if residual_extra:
        out["extra"] = residual_extra

    return out


def _expand_world(
    world: Dict[str, Any],
    variables: List[str],
    shared_panel_units: Optional[List[str]],
) -> Dict[str, Any]:
    if "worldId" not in world:
        raise ValueError("Compact world is missing worldId")

    domain = [str(v) for v in world.get("domain", shared_panel_units or [])]
    out: Dict[str, Any] = {
        "worldId": str(world.get("worldId", "")),
        "domain": domain,
        "domainSize": int(world.get("domainSize", len(domain))),
        "observationMode": world.get("observationMode", "panel_full"),
        "predicates": _clone(world.get("predicates", {})) or {},
        "events": _clone(world.get("events", {})) or {},
        "targetLabels": _clone(world.get("targetLabels", {})) or {},
    }

    if "intervention" in world:
        out["interventions"] = [_clone(world.get("intervention"))]
    else:
        out["interventions"] = _clone(world.get("interventions", [])) or []

    extra = _clone(world.get("extra", {})) or {}
    if "split" in world:
        extra["split"] = world.get("split")
    if "envProfile" in world:
        extra["envProfile"] = _clone(world.get("envProfile"))

    if "rows" in world:
        if shared_panel_units is None:
            raise ValueError("Compact rows require shared panel units")
        extra["rows"] = _decode_rows(world.get("rows") or [], variables, shared_panel_units)
    elif "rowsVerbose" in world:
        extra["rows"] = _clone(world.get("rowsVerbose"))

    summary = _clone(world.get("interventionSummary", {})) or {}
    for key in _WORLD_EXTRA_SUMMARY_KEYS:
        if key in summary:
            extra[key] = _clone(summary[key])

    out["extra"] = extra
    return out


def compact_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Compact a verbose SCM problem into runtime storage form."""
    out = _clone(problem)
    signature = out.get("signature")
    if not isinstance(signature, dict):
        return out

    variables = [str(v) for v in list(signature.get("variables") or [])]
    worlds = out.get("worlds")
    if not isinstance(worlds, list):
        return out

    shared_panel_units = _shared_panel_units([w for w in worlds if isinstance(w, dict)])
    out["signature"] = _compact_signature(signature)
    out["problemEncoding"] = PROBLEM_ENCODING_VERSION
    if shared_panel_units is not None:
        out["panelUnits"] = list(shared_panel_units)
    out["worlds"] = [
        _compact_world(world, variables, shared_panel_units) if isinstance(world, dict) else _clone(world)
        for world in worlds
    ]
    return out


def expand_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a compact runtime problem into the current verbose problem form."""
    out = _clone(problem)
    signature = out.get("signature")
    if not isinstance(signature, dict):
        return out

    variables = [str(v) for v in list(signature.get("variables") or [])]
    shared_panel_units = out.get("panelUnits")
    if isinstance(shared_panel_units, list):
        shared_panel_units = [str(v) for v in shared_panel_units]
    else:
        shared_panel_units = None

    worlds = out.get("worlds")
    if isinstance(worlds, list):
        out["worlds"] = [
            _expand_world(world, variables, shared_panel_units) if isinstance(world, dict) else _clone(world)
            for world in worlds
        ]

    out["signature"] = _expand_signature(signature)
    out.pop("panelUnits", None)
    out.pop("problemEncoding", None)
    return out


def _compact_current_result(result: Dict[str, Any]) -> Dict[str, Any]:
    compact = _clone(result)
    compact.pop("responseDict", None)
    compact.pop("thinkingSummary", None)
    compact.pop("llmCallMeta", None)

    usage: Dict[str, Any] = {}
    for source_key in ("thinkingTokens", "billedTokens", "usageDetails"):
        if source_key in compact:
            usage[source_key] = _clone(compact.pop(source_key))
    if usage:
        compact["usage"] = usage

    evaluation = compact.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(compact.get("extractedAnswer"), dict):
        eval_copy = _clone(evaluation)
        eval_copy.pop("extractedAnswer", None)
        compact["evaluation"] = eval_copy

    return compact


def _expand_current_result(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    expanded = _clone(result)
    expanded["model"] = model
    usage = expanded.pop("usage", None)
    if isinstance(usage, dict):
        for source_key in ("thinkingTokens", "billedTokens", "usageDetails"):
            if source_key in usage:
                expanded[source_key] = _clone(usage[source_key])
    return expanded


def _result_valid_flag(result: Dict[str, Any]) -> bool:
    if result.get("valid") is True:
        return True
    evaluation = result.get("evaluation")
    return isinstance(evaluation, dict) and evaluation.get("valid") is True


def legacy_llm_results_to_model_results(llm_results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Collapse legacy llmResults into current-state modelResults.

    Selection rule for duplicate models:

    - if any valid result exists, use the last valid result
    - otherwise use the last result overall

    This preserves current runner skip semantics, which depend on the existence
    of a valid result for a model on a problem.
    """
    out: Dict[str, Dict[str, Any]] = {}
    last_any: Dict[str, Dict[str, Any]] = {}
    last_valid: Dict[str, Dict[str, Any]] = {}
    for result in llm_results:
        if not isinstance(result, dict):
            continue
        model = str(result.get("model") or "").strip()
        if not model:
            continue
        compact = _compact_current_result(result)
        last_any[model] = compact
        if _result_valid_flag(result):
            last_valid[model] = compact

    for model in sorted(last_any.keys()):
        out[model] = _clone(last_valid.get(model, last_any[model]))
    return out


def model_results_to_legacy_llm_results(model_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Materialize legacy-style llmResults from runtime modelResults."""
    out: List[Dict[str, Any]] = []
    for model, result in (model_results or {}).items():
        if not isinstance(result, dict):
            continue
        out.append(_expand_current_result(result, str(model)))
    return out


def compact_problem_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy causal record into the compact runtime record form."""
    out = _clone(record)
    problem = out.get("problem")
    if isinstance(problem, dict):
        out["problem"] = compact_problem(problem)
    llm_results = out.pop("llmResults", None)
    if isinstance(llm_results, list):
        out["modelResults"] = legacy_llm_results_to_model_results(llm_results)
    return out


def expand_runtime_problem_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a compact runtime record back into the legacy record form."""
    out = _clone(record)
    problem = out.get("problem")
    if isinstance(problem, dict):
        out["problem"] = expand_problem(problem)
    model_results = out.pop("modelResults", None)
    if isinstance(model_results, dict):
        out["llmResults"] = model_results_to_legacy_llm_results(model_results)
    return out
