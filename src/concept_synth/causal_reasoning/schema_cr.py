"""Schema scaffold for causal reasoning tasks.

This module is intentionally task-agnostic. It defines a stable envelope
for multi-world causal reasoning records while leaving task internals open.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CAUSAL_SCHEMA_VERSION = "fol-causal-reasoning-v1"


@dataclass
class CausalTaskSpec:
    """Task-level metadata and free-form task payload."""

    taskName: str
    taskVersion: str = "0.1.0"
    query: str = ""
    expectedOutputSchema: str = "json"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskName": self.taskName,
            "taskVersion": self.taskVersion,
            "query": self.query,
            "expectedOutputSchema": self.expectedOutputSchema,
            "parameters": copy.deepcopy(self.parameters),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalTaskSpec":
        return cls(
            taskName=d.get("taskName", "unknown"),
            taskVersion=d.get("taskVersion", "0.1.0"),
            query=d.get("query", ""),
            expectedOutputSchema=d.get("expectedOutputSchema", "json"),
            parameters=d.get("parameters", {}) or {},
        )


@dataclass
class CausalWorldView:
    """World-level payload for causal reasoning.

    The fields below are intentionally broad and optional because concrete
    causal task variants are not finalized yet.
    """

    worldId: str
    domain: List[str]
    domainSize: int
    observationMode: str = "full"
    predicates: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    events: Dict[str, Any] = field(default_factory=dict)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    targetLabels: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worldId": self.worldId,
            "domain": list(self.domain),
            "domainSize": self.domainSize,
            "observationMode": self.observationMode,
            "predicates": copy.deepcopy(self.predicates),
            "events": copy.deepcopy(self.events),
            "interventions": copy.deepcopy(self.interventions),
            "targetLabels": copy.deepcopy(self.targetLabels),
            "extra": copy.deepcopy(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalWorldView":
        return cls(
            worldId=d.get("worldId", "unknown"),
            domain=d.get("domain", []),
            domainSize=d.get("domainSize", len(d.get("domain", []))),
            observationMode=d.get("observationMode", "full"),
            predicates=d.get("predicates", {}) or {},
            events=d.get("events", {}) or {},
            interventions=d.get("interventions", []) or [],
            targetLabels=d.get("targetLabels", {}) or {},
            extra=d.get("extra", {}) or {},
        )


@dataclass
class CausalInstance:
    """Problem payload for causal reasoning."""

    instanceId: str
    schemaVersion: str = CAUSAL_SCHEMA_VERSION
    scenario: str = "CR_GENERIC"
    signature: Dict[str, Any] = field(default_factory=dict)
    backgroundAxioms: List[str] = field(default_factory=list)
    worlds: List[CausalWorldView] = field(default_factory=list)
    task: CausalTaskSpec = field(default_factory=lambda: CausalTaskSpec(taskName="generic"))
    goldAnswer: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "instanceId": self.instanceId,
            "schemaVersion": self.schemaVersion,
            "scenario": self.scenario,
            "signature": copy.deepcopy(self.signature),
            "backgroundAxioms": list(self.backgroundAxioms),
            "worlds": [w.to_dict() if isinstance(w, CausalWorldView) else w for w in self.worlds],
            "task": self.task.to_dict() if isinstance(self.task, CausalTaskSpec) else self.task,
        }
        if self.goldAnswer is not None:
            result["goldAnswer"] = copy.deepcopy(self.goldAnswer)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalInstance":
        worlds = [
            CausalWorldView.from_dict(w) if isinstance(w, dict) else w
            for w in d.get("worlds", [])
        ]
        task = d.get("task", {})
        if isinstance(task, dict):
            task = CausalTaskSpec.from_dict(task)
        return cls(
            instanceId=d.get("instanceId", "unknown"),
            schemaVersion=d.get("schemaVersion", CAUSAL_SCHEMA_VERSION),
            scenario=d.get("scenario", "CR_GENERIC"),
            signature=d.get("signature", {}) or {},
            backgroundAxioms=d.get("backgroundAxioms", []) or [],
            worlds=worlds,
            task=task,
            goldAnswer=d.get("goldAnswer"),
        )


@dataclass
class CausalProblemDescription:
    """Metadata for a causal reasoning problem instance."""

    scenarioType: str = "CR_GENERIC"
    scenarioDescription: str = "Task-agnostic causal reasoning scaffold"
    difficulty: str = "unassigned"
    observationMode: str = "full"
    seed: Optional[int] = None
    generatorVersion: str = "0.1.0"
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "scenarioType": self.scenarioType,
            "scenarioDescription": self.scenarioDescription,
            "difficulty": self.difficulty,
            "observationMode": self.observationMode,
            "generatorVersion": self.generatorVersion,
            "tags": list(self.tags),
            "extra": copy.deepcopy(self.extra),
        }
        if self.seed is not None:
            result["seed"] = self.seed
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalProblemDescription":
        return cls(
            scenarioType=d.get("scenarioType", "CR_GENERIC"),
            scenarioDescription=d.get(
                "scenarioDescription", "Task-agnostic causal reasoning scaffold"
            ),
            difficulty=d.get("difficulty", "unassigned"),
            observationMode=d.get("observationMode", "full"),
            seed=d.get("seed"),
            generatorVersion=d.get("generatorVersion", "0.1.0"),
            tags=d.get("tags", []) or [],
            extra=d.get("extra", {}) or {},
        )


@dataclass
class CausalProblemRecord:
    """Top-level record, aligned with existing project structure."""

    problem: CausalInstance
    problemDescription: CausalProblemDescription
    problemType: str = "foCausalReasoning"
    llmResults: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict() if isinstance(self.problem, CausalInstance) else self.problem,
            "problemDescription": (
                self.problemDescription.to_dict()
                if isinstance(self.problemDescription, CausalProblemDescription)
                else self.problemDescription
            ),
            "problemType": self.problemType,
            "llmResults": copy.deepcopy(self.llmResults),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalProblemRecord":
        problem = d.get("problem", {})
        if isinstance(problem, dict):
            problem = CausalInstance.from_dict(problem)

        desc = d.get("problemDescription", {})
        if isinstance(desc, dict):
            desc = CausalProblemDescription.from_dict(desc)

        return cls(
            problem=problem,
            problemDescription=desc,
            problemType=d.get("problemType", "foCausalReasoning"),
            llmResults=d.get("llmResults", []) or [],
        )


def create_causal_problem_entry(
    problem: Dict[str, Any],
    problem_description: Dict[str, Any],
    problem_type: str = "foCausalReasoning",
) -> Dict[str, Any]:
    """Create a top-level causal problem entry."""
    return {
        "problem": copy.deepcopy(problem),
        "problemDescription": copy.deepcopy(problem_description),
        "problemType": problem_type,
        "llmResults": [],
    }
