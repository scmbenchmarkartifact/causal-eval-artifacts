from pathlib import Path

from concept_synth.causal_reasoning.prompting import build_causal_prompt
from concept_synth.causal_reasoning.runtime_storage_io import load_causal_dataset


def test_sample_benchmark_loads_and_exports_prompt():
    root = Path(__file__).resolve().parents[1]
    _, problems, _, _ = load_causal_dataset(str(root / "data/samples/sample_runtime.yaml"))
    assert len(problems) == 2

    task = ((problems[0].get("problem", {}) or {}).get("task", {}) or {})
    prompt, system = build_causal_prompt(problems[0], task.get("taskName", "CIND_A_SCM"))
    assert prompt.strip()
    assert isinstance(system, str)
