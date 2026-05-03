from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    family,
    is_a_scm_problem,
    instance_id,
    load_dataset,
    markdown_table,
    pct,
    rate,
    stable_json_dumps,
    write_csv,
    write_jsonl,
    write_markdown,
)
from concept_synth.analysis.scm_subset_compare import _parse_models_arg
from concept_synth.analysis.scm_common import parse_candidate, task_name

SUMMARY_MD = "scm_ntopo_cycle_motifs_summary.md"
BY_INSTANCE_CSV = "scm_ntopo_cycle_motifs_by_instance.csv"
BY_MODEL_CSV = "scm_ntopo_cycle_motifs_by_model.csv"
BY_MOTIF_CSV = "scm_ntopo_cycle_motif_counts.csv"
EXAMPLES_JSONL = "scm_ntopo_cycle_examples.jsonl"
MANIFEST_JSON = "scm_ntopo_cycle_motifs_manifest.json"


def _sccs(graph: Dict[str, Set[str]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    components: List[List[str]] = []

    def _strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for nxt in sorted(graph.get(node, set())):
            if nxt not in indices:
                _strongconnect(nxt)
                lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
            elif nxt in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[nxt])

        if lowlinks[node] == indices[node]:
            component: List[str] = []
            while stack:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node in sorted(graph.keys()):
        if node not in indices:
            _strongconnect(node)
    return components


def _cycle_row(problem: Dict[str, Any], llm_result: Dict[str, Any], parsed) -> Optional[Dict[str, Any]]:
    if parsed.invalid_bucket != "cyclic_ntopo_dependencies":
        return None
    topo = parsed.topological_order
    topo_index = {str(var): idx for idx, var in enumerate(topo)}
    endo = set(parsed.endogenous_vars)
    graph: Dict[str, Set[str]] = {}
    for var, parents in parsed.parent_by_var.items():
        graph[str(var)] = {str(parent) for parent in parents if str(parent) in endo}
    components = _sccs(graph)
    if not components:
        components = [sorted([str(v) for v in (llm_result.get("evaluation") or {}).get("cycleVariables") or []])]
        components = [component for component in components if len(component) > 1]
    if not components:
        return None
    primary = min(
        components,
        key=lambda component: (
            len(component),
            max(int(topo_index.get(node, 10**9)) for node in component)
            - min(int(topo_index.get(node, 10**9)) for node in component),
            component,
        ),
    )
    indices = [int(topo_index.get(node, 10**9)) for node in primary if node in topo_index]
    span = max(indices) - min(indices) if indices else None
    locality = "adjacent_or_local" if span is not None and span <= max(1, len(primary)) else "long_range"
    motif = f"len{len(primary)}_{locality}"
    return {
        "analysis_version": ANALYSIS_VERSION,
        "instance_id": instance_id(problem),
        "model": str(llm_result.get("model") or "unknown"),
        "slice": str((problem.get("subslice") or ((problem.get("problemDescription") or {}).get("extra") or {}).get("subslice") or "unknown")),
        "family": family(problem),
        "task_name": task_name(problem),
        "cycle_component_count": len(components),
        "primary_cycle_nodes": primary,
        "primary_cycle_length": len(primary),
        "primary_cycle_span": span,
        "primary_cycle_locality": locality,
        "primary_cycle_motif": motif,
    }


def analyze_ntopo_cycle_motifs(
    problems: Sequence[Dict[str, Any]],
    *,
    models: Optional[set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    instance_rows: List[Dict[str, Any]] = []
    motif_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for problem in problems:
        if not is_a_scm_problem(problem) or family(problem) != "ntopo":
            continue
        for llm_result in problem.get("llmResults") or []:
            if not isinstance(llm_result, dict):
                continue
            model = str(llm_result.get("model") or "unknown")
            if models and model not in models:
                continue
            parsed = parse_candidate(problem, llm_result)
            row = _cycle_row(problem, llm_result, parsed)
            if row is None:
                continue
            instance_rows.append(row)
            motif_counts[(row["model"], row["primary_cycle_motif"])] += 1

    by_model_rows: List[Dict[str, Any]] = []
    grouped_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped_model[str(row.get("model") or "unknown")].append(row)
    for model, group in sorted(grouped_model.items()):
        motif_counter: Dict[str, int] = defaultdict(int)
        for row in group:
            motif_counter[str(row.get("primary_cycle_motif") or "unknown")] += 1
        top_motif = None
        top_motif_share = None
        if motif_counter:
            top_motif, top_count = max(motif_counter.items(), key=lambda item: (item[1], item[0]))
            top_motif_share = rate(top_count, len(group))
        by_model_rows.append(
            {
                "model": model,
                "n": len(group),
                "mean_primary_cycle_length": sum(int(row.get("primary_cycle_length") or 0) for row in group) / float(len(group)),
                "mean_primary_cycle_span": sum(float(row.get("primary_cycle_span") or 0.0) for row in group) / float(len(group)),
                "top_primary_cycle_motif": top_motif,
                "top_primary_cycle_motif_share": top_motif_share,
                "local_cycle_share": rate(sum(1 for row in group if row.get("primary_cycle_locality") == "adjacent_or_local"), len(group)),
            }
        )

    motif_rows = [
        {
            "model": model,
            "primary_cycle_motif": motif,
            "count": count,
        }
        for (model, motif), count in sorted(motif_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    return {
        "instance_rows": instance_rows,
        "by_model_rows": by_model_rows,
        "motif_rows": motif_rows,
        "example_rows": instance_rows[:50],
    }


def build_ntopo_cycle_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_rows = []
    for row in sorted(artifacts["by_model_rows"], key=lambda item: str(item.get("model") or "")):
        model_rows.append(
            [
                row.get("model"),
                row.get("n"),
                row.get("mean_primary_cycle_length"),
                row.get("mean_primary_cycle_span"),
                row.get("top_primary_cycle_motif") or "-",
                pct(row.get("top_primary_cycle_motif_share")),
                pct(row.get("local_cycle_share")),
            ]
        )
    motif_rows = []
    for row in artifacts["motif_rows"][:12]:
        motif_rows.append([row.get("model"), row.get("primary_cycle_motif"), row.get("count")])
    lines = [
        "SCM NTopo Cycle Motifs",
        "",
        "This report characterizes invalid NTopo answers that fail by introducing cyclic endogenous dependencies.",
        "",
        markdown_table(
            ["Model", "N", "Mean cycle len", "Mean span", "Top motif", "Top share", "Local-cycle share"],
            model_rows,
        ) if model_rows else "No NTopo cycle failures.",
        "",
        "Most frequent motifs:",
        markdown_table(["Model", "Motif", "Count"], motif_rows) if motif_rows else "No motif rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_ntopo_cycle_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_instance_csv": str(out / BY_INSTANCE_CSV),
        "by_model_csv": str(out / BY_MODEL_CSV),
        "by_motif_csv": str(out / BY_MOTIF_CSV),
        "examples_jsonl": str(out / EXAMPLES_JSONL),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["instance_rows"], paths["by_instance_csv"])
    write_csv(artifacts["by_model_rows"], paths["by_model_csv"])
    write_csv(artifacts["motif_rows"], paths["by_motif_csv"])
    write_jsonl(artifacts["example_rows"], paths["examples_jsonl"])
    write_markdown(build_ntopo_cycle_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cyclic NTopo invalid outputs for A_SCM")
    parser.add_argument("--input", required=True, help="Benchmark YAML input")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    args = parser.parse_args(argv)

    _, problems = load_dataset(args.input)
    artifacts = analyze_ntopo_cycle_motifs(problems, models=_parse_models_arg(args.models))
    paths = write_ntopo_cycle_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM NTopo cycle motif artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
