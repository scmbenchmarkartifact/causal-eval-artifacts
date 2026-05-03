from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    bucket_numeric,
    markdown_table,
    mean,
    pct,
    rate,
    read_jsonl,
    stable_json_dumps,
    write_csv,
    write_markdown,
)
from concept_synth.analysis.scm_common import load_dataset
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)

SUMMARY_MD = "scm_heldout_stress_summary.md"
BY_MODE_CSV = "scm_heldout_stress_by_mode.csv"
BY_TARGET_COUNT_CSV = "scm_heldout_stress_by_target_count.csv"
BY_NOVELTY_MODE_CSV = "scm_heldout_stress_by_novelty_mode.csv"
BY_TARGET_POSITION_CSV = "scm_heldout_stress_by_target_position.csv"
MANIFEST_JSON = "scm_heldout_stress_manifest.json"


def _aggregate(group: Sequence[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    total = len(group)
    exact = sum(1 for row in group if row.get("exact") is True)
    total_scored = sum(int(row.get("scored_cells") or 0) for row in group)
    total_wrong = sum(int(row.get("wrong_cells") or 0) for row in group)
    return {
        **extra,
        "worlds": total,
        "exact_rate": rate(exact, total),
        "mean_accuracy": mean((float(row.get("accuracy") or 0.0) for row in group), default=0.0),
        "wrong_cell_rate": rate(total_wrong, total_scored),
        "mean_target_count": mean((float(row.get("intervention_target_count") or 0.0) for row in group), default=0.0),
        "mean_target_min_topological_index": mean(
            (
                float(row.get("intervention_target_min_topological_index"))
                for row in group
                if row.get("intervention_target_min_topological_index") is not None
            ),
            default=0.0,
        ),
        "mean_dominant_wrong_topological_index": mean(
            (
                float(row.get("dominant_wrong_topological_index"))
                for row in group
                if row.get("dominant_wrong_topological_index") is not None
            ),
            default=0.0,
        ),
        "mode_match_share": rate(
            sum(1 for row in group if row.get("mode_match_nearest_train") is True),
            sum(1 for row in group if row.get("mode_match_nearest_train") is not None),
        ),
        "mean_nearest_train_target_overlap": mean(
            (float(row.get("nearest_train_target_overlap") or 0.0) for row in group if row.get("nearest_train_target_overlap") is not None),
            default=0.0,
        ),
    }


def analyze_heldout_stress(world_records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    heldout_rows = [row for row in world_records if row.get("split") == "heldout"]
    mode_rows: List[Dict[str, Any]] = []
    target_count_rows: List[Dict[str, Any]] = []
    novelty_mode_rows: List[Dict[str, Any]] = []
    target_position_rows: List[Dict[str, Any]] = []

    grouped_mode: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_target_count: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_novelty_mode: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_target_position: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in heldout_rows:
        model = str(row.get("model") or "unknown")
        mode = str(row.get("intervention_mode") or "unknown")
        target_count_bin = bucket_numeric(row.get("intervention_target_count"), [float("-inf"), 1.5, 2.5, 3.5, float("inf")])
        novelty_bin = bucket_numeric(row.get("novelty_score"), [float("-inf"), 0.2, 0.4, 0.6, 0.8, float("inf")])
        target_position_bin = bucket_numeric(
            row.get("intervention_target_min_topological_index"),
            [float("-inf"), 2.0, 4.0, 6.0, 8.0, float("inf")],
        )
        grouped_mode[(model, mode)].append(row)
        grouped_target_count[(model, target_count_bin)].append(row)
        grouped_novelty_mode[(model, novelty_bin, mode)].append(row)
        grouped_target_position[(model, target_position_bin)].append(row)

    for (model, mode), group in sorted(grouped_mode.items()):
        mode_rows.append(_aggregate(group, model=model, intervention_mode=mode))
    for (model, target_count_bin), group in sorted(grouped_target_count.items()):
        target_count_rows.append(_aggregate(group, model=model, target_count_bin=target_count_bin))
    for (model, novelty_bin, mode), group in sorted(grouped_novelty_mode.items()):
        novelty_mode_rows.append(_aggregate(group, model=model, novelty_bin=novelty_bin, intervention_mode=mode))
    for (model, target_position_bin), group in sorted(grouped_target_position.items()):
        target_position_rows.append(_aggregate(group, model=model, target_position_bin=target_position_bin))

    return {
        "by_mode_rows": mode_rows,
        "by_target_count_rows": target_count_rows,
        "by_novelty_mode_rows": novelty_mode_rows,
        "by_target_position_rows": target_position_rows,
    }


def build_heldout_stress_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    model_hardest_mode: Dict[str, Dict[str, Any]] = {}
    for row in artifacts["by_mode_rows"]:
        model = str(row.get("model") or "unknown")
        current = model_hardest_mode.get(model)
        if current is None or float(row.get("wrong_cell_rate") or 0.0) > float(current.get("wrong_cell_rate") or 0.0):
            model_hardest_mode[model] = row

    hardest_rows = []
    for model, row in sorted(model_hardest_mode.items()):
        hardest_rows.append(
            [
                model,
                row.get("intervention_mode"),
                row.get("worlds"),
                pct(row.get("exact_rate")),
                pct(row.get("wrong_cell_rate")),
                row.get("mean_target_count"),
                pct(row.get("mode_match_share")),
                row.get("mean_nearest_train_target_overlap"),
            ]
        )

    novelty_rows = []
    for row in sorted(
        artifacts["by_novelty_mode_rows"],
        key=lambda item: (
            -float(item.get("wrong_cell_rate") or 0.0),
            str(item.get("model") or ""),
            str(item.get("novelty_bin") or ""),
            str(item.get("intervention_mode") or ""),
        ),
    )[:12]:
        novelty_rows.append(
            [
                row.get("model"),
                row.get("novelty_bin"),
                row.get("intervention_mode"),
                row.get("worlds"),
                pct(row.get("exact_rate")),
                pct(row.get("wrong_cell_rate")),
            ]
        )

    lines = [
        "SCM Heldout Stress Analysis",
        "",
        "This report asks which heldout intervention regimes are most stressful by mode, novelty, target count, and intervention position.",
        "",
        "Hardest heldout mode by model:",
        markdown_table(
            [
                "Model",
                "Mode",
                "Worlds",
                "Exact",
                "Wrong cell rate",
                "Mean target count",
                "Mode-match share",
                "Mean nearest-train overlap",
            ],
            hardest_rows,
        ) if hardest_rows else "No heldout worlds.",
        "",
        "Hardest novelty/mode combinations:",
        markdown_table(
            ["Model", "Novelty bin", "Mode", "Worlds", "Exact", "Wrong cell rate"],
            novelty_rows,
        ) if novelty_rows else "No novelty rows.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_heldout_stress_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_mode_csv": str(out / BY_MODE_CSV),
        "by_target_count_csv": str(out / BY_TARGET_COUNT_CSV),
        "by_novelty_mode_csv": str(out / BY_NOVELTY_MODE_CSV),
        "by_target_position_csv": str(out / BY_TARGET_POSITION_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["by_mode_rows"], paths["by_mode_csv"])
    write_csv(artifacts["by_target_count_rows"], paths["by_target_count_csv"])
    write_csv(artifacts["by_novelty_mode_rows"], paths["by_novelty_mode_csv"])
    write_csv(artifacts["by_target_position_rows"], paths["by_target_position_csv"])
    write_markdown(build_heldout_stress_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "counts": {key: len(value) for key, value in artifacts.items()},
        "files": paths,
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze heldout intervention stress regimes for A_SCM")
    parser.add_argument("--world-records", help="Path to scm_world_records.jsonl")
    parser.add_argument("--input", help="Optional benchmark YAML; if set, records are extracted first")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    if args.input:
        _, problems = load_dataset(args.input)
        extracted = extract_scm_analysis_records(problems, source_name=str(args.input))
        write_extract_artifacts(extracted, Path(args.outdir) / "extract", source_name=str(args.input))
        artifacts = analyze_heldout_stress(extracted["world_records"])
    else:
        if not args.world_records:
            raise SystemExit("Provide either --input or --world-records")
        artifacts = analyze_heldout_stress(read_jsonl(args.world_records))
    paths = write_heldout_stress_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM heldout stress artifacts to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
