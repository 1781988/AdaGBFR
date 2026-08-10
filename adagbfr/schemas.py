from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    ANSWER = "answer"
    EVIDENCE_ABSENT = "evidence_absent"
    GRAPH_COVERAGE = "graph_coverage"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONFLICT = "conflict"
    INVALID_PLAN = "invalid_plan"
    EXECUTION_ERROR = "execution_error"


@dataclass
class Constraints:
    temporal_scope: Optional[str] = None
    entity_scope: Optional[str] = None
    currency: Optional[str] = None
    scale: Optional[str] = None
    unit: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> "Constraints":
        obj = obj or {}
        meta = obj.get("meta") or {}
        return cls(
            temporal_scope=_none(obj.get("temporal_scope")),
            entity_scope=_none(obj.get("entity_scope")),
            currency=_none(meta.get("currency", obj.get("currency"))),
            scale=_none(meta.get("scale", obj.get("scale"))),
            unit=_none(meta.get("unit", obj.get("unit"))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FormulaSpec:
    expression: str
    dependencies: List[str]
    confidence: float = 1.0
    source_ids: List[str] = field(default_factory=list)
    origin: str = "static"

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "FormulaSpec":
        return cls(
            expression=str(obj.get("expression", "")).strip(),
            dependencies=[str(x).strip() for x in obj.get("dependencies", []) if str(x).strip()],
            confidence=float(obj.get("confidence", 1.0)),
            source_ids=[str(x) for x in obj.get("source_ids", [])],
            origin=str(obj.get("origin", "static")),
        )


@dataclass
class MetricNode:
    name: str
    aliases: List[str] = field(default_factory=list)
    definition: str = ""
    formulas: List[FormulaSpec] = field(default_factory=list)
    origin: str = "static"
    confidence: float = 1.0
    source_ids: List[str] = field(default_factory=list)

    @property
    def atomic(self) -> bool:
        return not self.formulas

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "MetricNode":
        formulas = [FormulaSpec.from_dict(x) for x in obj.get("formulas", [])]
        return cls(
            name=str(obj.get("name", "")).strip(),
            aliases=[str(x).strip() for x in obj.get("aliases", []) if str(x).strip()],
            definition=str(obj.get("definition", "") or ""),
            formulas=formulas,
            origin=str(obj.get("origin", "static")),
            confidence=float(obj.get("confidence", 1.0)),
            source_ids=[str(x) for x in obj.get("source_ids", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    metric: str
    value: float
    raw_value: str
    snippet: str
    confidence: float
    method: str
    constraints: Constraints = field(default_factory=Constraints)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["constraints"] = self.constraints.to_dict()
        return out


@dataclass
class Subgoal:
    metric: str
    constraints: Constraints = field(default_factory=Constraints)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Subgoal":
        return cls(metric=str(obj.get("metric", "")).strip(), constraints=Constraints.from_dict(obj.get("constraints")))


@dataclass
class QueryPlan:
    reasoning_type: str
    operator: str
    subgoals: List[Subgoal]
    precision: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "QueryPlan":
        return cls(
            reasoning_type=str(obj.get("reasoning_type", "DirectRetrieval")),
            operator=str(obj.get("operator", "identity")),
            subgoals=[Subgoal.from_dict(x) for x in obj.get("subgoals", [])],
            precision=_none(obj.get("precision")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_type": self.reasoning_type,
            "operator": self.operator,
            "subgoals": [{"metric": x.metric, "constraints": x.constraints.to_dict()} for x in self.subgoals],
            "precision": self.precision,
        }


@dataclass
class PathCandidate:
    metric: str
    formula: FormulaSpec
    score: float
    estimated_depth: int
    evidence_coverage: float
    trust: float
    estimated_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResolutionResult:
    metric: str
    status: Status
    value: Optional[float] = None
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None
    origin: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == Status.ANSWER and self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status.value,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "trace": self.trace,
            "reason": self.reason,
            "origin": self.origin,
        }


@dataclass
class RunResult:
    status: Status
    value: Optional[float]
    answer: Optional[str]
    plan: Optional[QueryPlan]
    subgoal_results: List[ResolutionResult]
    budget: Dict[str, Any]
    overlay_metrics: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "value": self.value,
            "answer": self.answer,
            "plan": self.plan.to_dict() if self.plan else None,
            "subgoal_results": [x.to_dict() for x in self.subgoal_results],
            "budget": self.budget,
            "overlay_metrics": self.overlay_metrics,
            "reason": self.reason,
        }


def _none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"unknown", "null", "none", "n/a"}:
        return None
    return s
