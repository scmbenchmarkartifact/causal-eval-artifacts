from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concept_synth.analysis.scm_common import (
    ANALYSIS_VERSION,
    load_dataset,
    markdown_table,
    pct,
    rate,
    read_jsonl,
    stable_json_dumps,
    write_csv,
    write_markdown,
)
from concept_synth.analysis.scm_extract_eval_records import (
    extract_scm_analysis_records,
    write_artifacts as write_extract_artifacts,
)

SUMMARY_MD = "scm_favorite_formulas_summary.md"
FORMULA_EXACT_CSV = "scm_favorite_formulas_exact.csv"
FORMULA_CANONICAL_CSV = "scm_favorite_formulas_canonical.csv"
MAP_EXACT_CSV = "scm_favorite_maps_exact.csv"
MAP_CANONICAL_CSV = "scm_favorite_maps_canonical.csv"
OPERATOR_PROFILE_CSV = "scm_operator_profile_bias.csv"
MANIFEST_JSON = "scm_favorite_formulas_manifest.json"


def _aggregate_preferences(
    rows: Sequence[Dict[str, Any]],
    *,
    key_field: str,
    value_field: str,
    gold_field: Optional[str] = None,
    min_count: int = 1,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    totals: Dict[str, int] = defaultdict(int)
    for row in rows:
        model = str(row.get("model") or "unknown")
        value = row.get(value_field)
        if value in (None, "", "missing"):
            continue
        key = (model, str(value))
        grouped[key].append(row)
        totals[model] += 1

    out: List[Dict[str, Any]] = []
    for (model, value), group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        count = len(group)
        if count < min_count:
            continue
        total = totals[model]
        out.append(
            {
                key_field: value,
                "model": model,
                "count": count,
                "share": rate(count, total),
                "valid_rate": rate(sum(1 for row in group if row.get("valid") is True), count),
                "correct_rate": rate(sum(1 for row in group if row.get("correct") is True), count),
                "train_exact_rate": rate(sum(1 for row in group if row.get("train_exact") is True), count),
                "heldout_exact_rate": rate(sum(1 for row in group if row.get("heldout_exact") is True), count),
                "gold_match_rate": rate(sum(1 for row in group if gold_field and row.get(value_field) == row.get(gold_field)), count) if gold_field else None,
            }
        )
    return out


def _operator_profile_rows(mechanism_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    totals: Dict[str, int] = defaultdict(int)
    for row in mechanism_records:
        profile = str(row.get("pred_operator_profile") or "none")
        model = str(row.get("model") or "unknown")
        grouped[(model, profile)].append(row)
        totals[model] += 1
    out: List[Dict[str, Any]] = []
    for (model, profile), group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        count = len(group)
        total = totals[model]
        out.append(
            {
                "model": model,
                "pred_operator_profile": profile,
                "count": count,
                "share": rate(count, total),
                "heldout_var_accuracy_mean": rate(
                    sum(float(row.get("heldout_var_accuracy") or 0.0) for row in group),
                    count,
                ),
                "gold_match_rate": rate(
                    sum(1 for row in group if row.get("pred_operator_profile") == row.get("gold_operator_profile")),
                    count,
                ),
            }
        )
    return out


def analyze_favorite_formulas(
    mechanism_records: Sequence[Dict[str, Any]],
    map_records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "formula_exact_rows": _aggregate_preferences(
            mechanism_records,
            key_field="formula",
            value_field="pred_expr",
            gold_field="gold_expr",
        ),
        "formula_canonical_rows": _aggregate_preferences(
            mechanism_records,
            key_field="formula_canonical",
            value_field="pred_expr_canonical",
            gold_field="gold_expr_canonical",
        ),
        "map_exact_rows": _aggregate_preferences(
            map_records,
            key_field="map_signature",
            value_field="exact_map_signature",
            gold_field="gold_exact_map_signature",
        ),
        "map_canonical_rows": _aggregate_preferences(
            map_records,
            key_field="map_signature_canonical",
            value_field="canonical_map_signature",
            gold_field="gold_canonical_map_signature",
        ),
        "operator_profile_rows": _operator_profile_rows(mechanism_records),
    }


def _top_rows(rows: Sequence[Dict[str, Any]], *, value_field: str, limit: int = 3) -> List[List[Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model") or "unknown")].append(row)
    out: List[List[Any]] = []
    for model in sorted(by_model.keys()):
        for row in by_model[model][:limit]:
            out.append(
                [
                    model,
                    row.get(value_field),
                    row.get("count"),
                    pct(row.get("share")),
                    pct(row.get("correct_rate")),
                    pct(row.get("gold_match_rate")),
                ]
            )
    return out


def build_favorites_summary(artifacts: Dict[str, List[Dict[str, Any]]]) -> str:
    exact_rows = _top_rows(artifacts["formula_exact_rows"], value_field="formula")
    canonical_rows = _top_rows(artifacts["formula_canonical_rows"], value_field="formula_canonical")
    map_rows = _top_rows(artifacts["map_canonical_rows"], value_field="map_signature_canonical")
    lines = [
        "SCM Favorite Formulas",
        "",
        "Most repeated exact mechanism strings by model:",
        markdown_table(["Model", "Formula", "Count", "Share", "Correct", "Gold match"], exact_rows) if exact_rows else "No formula rows.",
        "",
        "Most repeated canonical mechanism templates by model:",
        markdown_table(["Model", "Template", "Count", "Share", "Correct", "Gold match"], canonical_rows) if canonical_rows else "No canonical rows.",
        "",
        "Most repeated canonical mechanism maps by model:",
        markdown_table(["Model", "Map template", "Count", "Share", "Correct", "Gold match"], map_rows) if map_rows else "No map rows.",
        "",
        "Interpretation notes:",
        "- Exact favorites show literal reuse, including variable names.",
        "- Canonical favorites collapse commutative reorderings and variable renamings into a template-level preference.",
        "- Gold-match rates show whether a favorite is often the right answer or an overused wrong template.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_favorite_formula_artifacts(artifacts: Dict[str, List[Dict[str, Any]]], outdir: str | Path) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "formula_exact_csv": str(out / FORMULA_EXACT_CSV),
        "formula_canonical_csv": str(out / FORMULA_CANONICAL_CSV),
        "map_exact_csv": str(out / MAP_EXACT_CSV),
        "map_canonical_csv": str(out / MAP_CANONICAL_CSV),
        "operator_profile_csv": str(out / OPERATOR_PROFILE_CSV),
        "summary_md": str(out / SUMMARY_MD),
        "manifest_json": str(out / MANIFEST_JSON),
    }
    write_csv(artifacts["formula_exact_rows"], paths["formula_exact_csv"])
    write_csv(artifacts["formula_canonical_rows"], paths["formula_canonical_csv"])
    write_csv(artifacts["map_exact_rows"], paths["map_exact_csv"])
    write_csv(artifacts["map_canonical_rows"], paths["map_canonical_csv"])
    write_csv(artifacts["operator_profile_rows"], paths["operator_profile_csv"])
    write_markdown(build_favorites_summary(artifacts), paths["summary_md"])
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "files": paths,
        "counts": {key: len(value) for key, value in artifacts.items()},
    }
    Path(paths["manifest_json"]).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return paths


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.mechanism_records and args.map_records:
        return read_jsonl(args.mechanism_records), read_jsonl(args.map_records)
    if not args.input:
        raise SystemExit("Either --input or both --mechanism-records and --map-records are required")
    _, problems = load_dataset(args.input)
    extracted = extract_scm_analysis_records(problems, source_name=str(args.input), models=None, family_filter=args.family)
    if args.extract_outdir:
        write_extract_artifacts(extracted, args.extract_outdir, source_name=str(args.input))
    return extracted["mechanism_records"], extracted["map_records"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze favorite A_SCM formulas and mechanism maps")
    parser.add_argument("--input", help="Benchmark YAML input")
    parser.add_argument("--mechanism-records", help="scm_mechanism_records.jsonl path")
    parser.add_argument("--map-records", help="scm_map_records.jsonl path")
    parser.add_argument("--family", choices=["ordered", "ntopo"], help="Optional family filter when reading --input")
    parser.add_argument("--extract-outdir", help="Optional directory to write extracted records when reading --input")
    parser.add_argument("--outdir", required=True, help="Directory for favorites outputs")
    args = parser.parse_args(argv)

    mechanism_records, map_records = _load_records(args)
    artifacts = analyze_favorite_formulas(mechanism_records, map_records)
    paths = write_favorite_formula_artifacts(artifacts, args.outdir)
    print(f"Wrote SCM favorite-formula analysis to {args.outdir}")
    print(f"  summary: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
