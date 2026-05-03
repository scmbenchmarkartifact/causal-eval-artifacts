from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from concept_synth.analysis.scm_common import ANALYSIS_VERSION, load_dataset, markdown_table, stable_json_dumps, write_markdown
from concept_synth.analysis.scm_error_taxonomy import (
    analyze_error_taxonomy,
    build_error_taxonomy_summary,
    write_error_taxonomy_artifacts,
)
from concept_synth.analysis.scm_cascade_attribution import (
    analyze_cascade_attribution,
    build_cascade_summary,
    write_cascade_artifacts,
)
from concept_synth.analysis.scm_extract_eval_records import extract_scm_analysis_records, write_artifacts as write_extract_artifacts
from concept_synth.analysis.scm_favorite_formulas import (
    analyze_favorite_formulas,
    build_favorites_summary,
    write_favorite_formula_artifacts,
)
from concept_synth.analysis.scm_heldout_root_cause import (
    analyze_heldout_root_cause,
    build_heldout_root_cause_summary,
    write_heldout_root_cause_artifacts,
)
from concept_synth.analysis.scm_heldout_stress import (
    analyze_heldout_stress,
    build_heldout_stress_summary,
    write_heldout_stress_artifacts,
)
from concept_synth.analysis.scm_ntopo_cycle_motifs import (
    analyze_ntopo_cycle_motifs,
    build_ntopo_cycle_summary,
    write_ntopo_cycle_artifacts,
)
from concept_synth.analysis.scm_parent_formula_taxonomy import (
    analyze_parent_formula_taxonomy,
    build_parent_formula_summary,
    write_parent_formula_artifacts,
)
from concept_synth.analysis.scm_repairability import (
    analyze_repairability,
    build_repairability_summary,
    write_repairability_artifacts,
)
from concept_synth.analysis.scm_train_heldout_amplification import (
    analyze_train_heldout_amplification,
    build_train_heldout_amplification_summary,
    write_train_heldout_amplification_artifacts,
)
from concept_synth.analysis.scm_training_equivalence import (
    analyze_training_equivalence,
    build_training_equivalence_summary,
    write_training_equivalence_artifacts,
)
from concept_synth.analysis.scm_gap_decomposition import (
    analyze_gap_decomposition,
    build_gap_decomposition_summary,
    write_gap_decomposition_artifacts,
)
from concept_synth.analysis.scm_structure_breakdown import (
    analyze_structure_breakdown,
    build_structure_summary,
    write_structure_artifacts,
)
from concept_synth.analysis.scm_subset_compare import (
    _baseline_instance_ids,
    _parse_models_arg,
    analyze_subset_compare,
    build_subset_compare_summary,
    write_subset_compare_artifacts,
)
from concept_synth.analysis.scm_variable_position_analysis import (
    analyze_variable_position,
    build_variable_position_summary,
    write_variable_position_artifacts,
)

SUMMARY_MD = "scm_report.md"
MANIFEST_JSON = "scm_report_manifest.json"


def _section_from_summary(title: str, summary_text: str) -> str:
    lines = summary_text.strip().splitlines()
    if lines and lines[0].strip() == title:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    body = "\n".join(lines).strip()
    if not body:
        return f"## {title}\n"
    return f"## {title}\n\n{body}\n"


def build_report_summary(
    source_name: str,
    extract_artifacts: Dict[str, List[Dict[str, Any]]],
    taxonomy_artifacts: Dict[str, List[Dict[str, Any]]],
    structure_artifacts: Dict[str, List[Dict[str, Any]]],
    favorite_artifacts: Dict[str, List[Dict[str, Any]]],
    subset_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    heldout_root_cause_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    cascade_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    parent_formula_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    heldout_stress_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    variable_position_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ntopo_cycle_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    repairability_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    train_heldout_amplification_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    training_equivalence_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gap_decomposition_artifacts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    instance_records = extract_artifacts["instance_records"]
    models = sorted({str(row.get("model") or "unknown") for row in instance_records})
    overview_by_model = {str(row.get("model") or "unknown"): row for row in taxonomy_artifacts["overview_rows"]}

    overview_rows = []
    for model in models:
        row = overview_by_model.get(model, {})
        overview_rows.append(
            [
                model,
                row.get("n", 0),
                f"{100.0 * float(row.get('valid_rate') or 0.0):.1f}%",
                f"{100.0 * float(row.get('correct_rate') or 0.0):.1f}%",
                f"{100.0 * float(row.get('train_exact_rate') or 0.0):.1f}%",
                f"{100.0 * float(row.get('heldout_exact_rate') or 0.0):.1f}%",
            ]
        )

    hard_ast_rows = []
    by_model_ast: Dict[str, List[Dict[str, Any]]] = {}
    for row in structure_artifacts["gold_ast_rows"]:
        by_model_ast.setdefault(str(row.get("model") or "unknown"), []).append(row)
    for model in models:
        candidates = by_model_ast.get(model, [])
        if not candidates:
            continue
        hardest = min(candidates, key=lambda row: (float(row.get("heldout_exact_rate") or 0.0), str(row.get("gold_ast_bin") or "")))
        hard_ast_rows.append(
            [
                model,
                hardest.get("gold_ast_bin"),
                hardest.get("n"),
                f"{100.0 * float(hardest.get('heldout_exact_rate') or 0.0):.1f}%",
            ]
        )

    favorite_rows = []
    favorites_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in favorite_artifacts["formula_canonical_rows"]:
        favorites_by_model.setdefault(str(row.get("model") or "unknown"), []).append(row)
    for model in models:
        group = favorites_by_model.get(model, [])
        if not group:
            continue
        top = group[0]
        favorite_rows.append(
            [
                model,
                top.get("formula_canonical"),
                top.get("count"),
                f"{100.0 * float(top.get('share') or 0.0):.1f}%",
                f"{100.0 * float(top.get('correct_rate') or 0.0):.1f}%",
            ]
        )

    overview_text = "\n".join(
        [
            "## Overview",
            "",
            f"Source: `{source_name}`",
            f"Models analyzed: {len(models)}",
            f"Instance records: {len(extract_artifacts['instance_records'])}",
            f"World records: {len(extract_artifacts['world_records'])}",
            f"Mechanism records: {len(extract_artifacts['mechanism_records'])}",
            "",
            "Topline model performance:",
            markdown_table(["Model", "N", "Valid", "Correct", "TrainExact", "HeldoutExact"], overview_rows) if overview_rows else "No overview rows.",
            "",
            "Hardest gold-AST bins by model:",
            markdown_table(["Model", "Gold AST bin", "N", "HeldoutExact"], hard_ast_rows) if hard_ast_rows else "No structure rows.",
            "",
            "Favorite canonical mechanism template by model:",
            markdown_table(["Model", "Template", "Count", "Share", "Correct"], favorite_rows) if favorite_rows else "No favorite rows.",
        ]
    ).strip()

    artifact_index_text = "\n".join(
        [
            "## Artifact Index",
            "",
            "- `extract/`: normalized instance/world/mechanism/map records in JSONL and CSV.",
            "- `subsets/`: apples-to-apples common-subset comparisons, with optional baseline-overlap and new-only splits.",
            "- `taxonomy/`: invalid buckets, train-vs-heldout error buckets, intervention-mode and novelty sensitivity.",
            "- `structure/`: AST bins, operator profiles, parent-relation regimes, and mechanism-level structure breakdown.",
            "- `favorites/`: exact and canonical favorite formulas and full mechanism maps by model.",
            "- `heldout_root_cause/`: per-instance heldout-failure attribution and aggregated heldout-only root-cause summaries.",
            "- `cascade/`: refined heldout cascade attribution, separating local structural errors from downstream propagation.",
            "- `parent_formula/`: parent-set recovery versus same-parent formula-induction errors.",
            "- `heldout_stress/`: heldout difficulty by intervention mode, novelty, target count, and target position.",
            "- `variable_position/`: performance by endogenous-variable position and first-wrong depth.",
            "- `ntopo_cycles/`: cyclic invalid-output motifs for NTopo answers.",
            "- `repairability/`: bounded functional one-step repairability and one-gold-mechanism repair probes.",
            "- `train_heldout_amplification/`: direct-vs-propagated amplification from train errors into heldout misses.",
            "- `training_equivalence/`: heldout-only misses split into local train ambiguity versus compensated train fit.",
            "- `gap_decomposition/`: unified A/B decomposition of the train-to-heldout gap.",
        ]
    ).strip()

    lines = [
        "# SCM Analysis Report",
        "",
        overview_text,
        "",
        _section_from_summary("SCM Subset Comparison", build_subset_compare_summary(subset_artifacts)).strip()
        if subset_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Error Taxonomy", build_error_taxonomy_summary(taxonomy_artifacts)).strip(),
        "",
        _section_from_summary("SCM Structure Breakdown", build_structure_summary(structure_artifacts)).strip(),
        "",
        _section_from_summary("SCM Favorite Formulas", build_favorites_summary(favorite_artifacts)).strip(),
        "",
        _section_from_summary("SCM Heldout Root Causes", build_heldout_root_cause_summary(heldout_root_cause_artifacts)).strip()
        if heldout_root_cause_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Cascade Attribution", build_cascade_summary(cascade_artifacts)).strip()
        if cascade_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Parent And Formula Taxonomy", build_parent_formula_summary(parent_formula_artifacts)).strip()
        if parent_formula_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Heldout Stress", build_heldout_stress_summary(heldout_stress_artifacts)).strip()
        if heldout_stress_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Variable Position Analysis", build_variable_position_summary(variable_position_artifacts)).strip()
        if variable_position_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM NTopo Cycle Motifs", build_ntopo_cycle_summary(ntopo_cycle_artifacts)).strip()
        if ntopo_cycle_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Repairability", build_repairability_summary(repairability_artifacts)).strip()
        if repairability_artifacts is not None
        else "",
        "",
        _section_from_summary(
            "SCM Train-Heldout Amplification",
            build_train_heldout_amplification_summary(train_heldout_amplification_artifacts),
        ).strip()
        if train_heldout_amplification_artifacts is not None
        else "",
        "",
        _section_from_summary(
            "SCM Training Equivalence",
            build_training_equivalence_summary(training_equivalence_artifacts),
        ).strip()
        if training_equivalence_artifacts is not None
        else "",
        "",
        _section_from_summary("SCM Gap Decomposition", build_gap_decomposition_summary(gap_decomposition_artifacts)).strip()
        if gap_decomposition_artifacts is not None
        else "",
        "",
        artifact_index_text,
    ]
    return "\n".join(lines).strip() + "\n"


def run_full_report(
    input_path: str,
    outdir: str | Path,
    *,
    family: Optional[str] = None,
    models: Optional[set[str]] = None,
    baseline_input: Optional[str] = None,
) -> Dict[str, Any]:
    _, problems = load_dataset(input_path)
    extract_artifacts = extract_scm_analysis_records(
        problems,
        source_name=str(input_path),
        models=models,
        family_filter=family,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    extract_paths = write_extract_artifacts(extract_artifacts, out / "extract", source_name=str(input_path))

    taxonomy_artifacts = analyze_error_taxonomy(extract_artifacts["instance_records"], extract_artifacts["world_records"])
    taxonomy_paths = write_error_taxonomy_artifacts(taxonomy_artifacts, out / "taxonomy")

    structure_artifacts = analyze_structure_breakdown(extract_artifacts["instance_records"], extract_artifacts["mechanism_records"])
    structure_paths = write_structure_artifacts(structure_artifacts, out / "structure")

    favorite_artifacts = analyze_favorite_formulas(extract_artifacts["mechanism_records"], extract_artifacts["map_records"])
    favorite_paths = write_favorite_formula_artifacts(favorite_artifacts, out / "favorites")

    subset_artifacts = analyze_subset_compare(
        extract_artifacts["instance_records"],
        selected_models=models,
        baseline_ids=_baseline_instance_ids(baseline_input, family_filter=family) if baseline_input else None,
    )
    subset_paths = write_subset_compare_artifacts(subset_artifacts, out / "subsets")

    heldout_root_cause_artifacts = analyze_heldout_root_cause(
        extract_artifacts["instance_records"],
        extract_artifacts["mechanism_records"],
        extract_artifacts["world_records"],
    )
    heldout_root_cause_paths = write_heldout_root_cause_artifacts(heldout_root_cause_artifacts, out / "heldout_root_cause")

    cascade_artifacts = analyze_cascade_attribution(
        extract_artifacts["instance_records"],
        extract_artifacts["mechanism_records"],
        extract_artifacts["world_records"],
    )
    cascade_paths = write_cascade_artifacts(cascade_artifacts, out / "cascade")

    parent_formula_artifacts = analyze_parent_formula_taxonomy(extract_artifacts["mechanism_records"])
    parent_formula_paths = write_parent_formula_artifacts(parent_formula_artifacts, out / "parent_formula")

    heldout_stress_artifacts = analyze_heldout_stress(extract_artifacts["world_records"])
    heldout_stress_paths = write_heldout_stress_artifacts(heldout_stress_artifacts, out / "heldout_stress")

    variable_position_artifacts = analyze_variable_position(
        extract_artifacts["mechanism_records"],
        cascade_artifacts["instance_rows"],
    )
    variable_position_paths = write_variable_position_artifacts(variable_position_artifacts, out / "variable_position")

    ntopo_cycle_artifacts = analyze_ntopo_cycle_motifs(problems, models=models)
    ntopo_cycle_paths = write_ntopo_cycle_artifacts(ntopo_cycle_artifacts, out / "ntopo_cycles")

    repairability_artifacts = analyze_repairability(
        problems,
        models=models,
        family_filter=family,
    )
    repairability_paths = write_repairability_artifacts(repairability_artifacts, out / "repairability")

    train_heldout_amplification_artifacts = analyze_train_heldout_amplification(
        problems,
        models=models,
        family_filter=family,
    )
    train_heldout_amplification_paths = write_train_heldout_amplification_artifacts(
        train_heldout_amplification_artifacts,
        out / "train_heldout_amplification",
    )

    training_equivalence_artifacts = analyze_training_equivalence(
        problems,
        cascade_artifacts["instance_rows"],
        repairability_artifacts["instance_rows"],
        models=models,
        family_filter=family,
    )
    training_equivalence_paths = write_training_equivalence_artifacts(
        training_equivalence_artifacts,
        out / "training_equivalence",
    )

    gap_decomposition_artifacts = analyze_gap_decomposition(
        extract_artifacts["instance_records"],
        train_heldout_amplification_artifacts["instance_rows"],
        training_equivalence_artifacts["instance_rows"],
    )
    gap_decomposition_paths = write_gap_decomposition_artifacts(
        gap_decomposition_artifacts,
        out / "gap_decomposition",
    )

    summary_path = out / SUMMARY_MD
    write_markdown(
        build_report_summary(
            str(input_path),
            extract_artifacts,
            taxonomy_artifacts,
            structure_artifacts,
            favorite_artifacts,
            subset_artifacts=subset_artifacts,
            heldout_root_cause_artifacts=heldout_root_cause_artifacts,
            cascade_artifacts=cascade_artifacts,
            parent_formula_artifacts=parent_formula_artifacts,
            heldout_stress_artifacts=heldout_stress_artifacts,
            variable_position_artifacts=variable_position_artifacts,
            ntopo_cycle_artifacts=ntopo_cycle_artifacts,
            repairability_artifacts=repairability_artifacts,
            train_heldout_amplification_artifacts=train_heldout_amplification_artifacts,
            training_equivalence_artifacts=training_equivalence_artifacts,
            gap_decomposition_artifacts=gap_decomposition_artifacts,
        ),
        summary_path,
    )

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "input": str(input_path),
        "family": family,
        "models": sorted(models) if models else None,
        "baseline_input": baseline_input,
        "extract_files": extract_paths,
        "subset_files": subset_paths,
        "taxonomy_files": taxonomy_paths,
        "structure_files": structure_paths,
        "favorites_files": favorite_paths,
        "heldout_root_cause_files": heldout_root_cause_paths,
        "cascade_files": cascade_paths,
        "parent_formula_files": parent_formula_paths,
        "heldout_stress_files": heldout_stress_paths,
        "variable_position_files": variable_position_paths,
        "ntopo_cycle_files": ntopo_cycle_paths,
        "repairability_files": repairability_paths,
        "train_heldout_amplification_files": train_heldout_amplification_paths,
        "training_equivalence_files": training_equivalence_paths,
        "gap_decomposition_files": gap_decomposition_paths,
        "summary_md": str(summary_path),
    }
    manifest_path = out / MANIFEST_JSON
    manifest_path.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def _parse_models(value: Optional[str]) -> Optional[set[str]]:
    return _parse_models_arg(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full offline A_SCM analysis suite and emit JSON/CSV/Markdown artifacts")
    parser.add_argument("--input", required=True, help="Benchmark YAML input")
    parser.add_argument("--outdir", required=True, help="Output directory for the report bundle")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter")
    parser.add_argument("--models", help="Optional comma-separated model filter")
    parser.add_argument("--baseline-input", help="Optional baseline benchmark YAML for old/new subset comparisons")
    args = parser.parse_args(argv)

    manifest = run_full_report(
        args.input,
        args.outdir,
        family=args.family,
        models=_parse_models(args.models),
        baseline_input=args.baseline_input,
    )
    print(f"Wrote SCM report bundle to {args.outdir}")
    print(f"  summary: {manifest['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
