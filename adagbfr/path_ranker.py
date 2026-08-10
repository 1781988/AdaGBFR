from __future__ import annotations

import re
from typing import Any, Dict, List

from .knowledge import MetricStore
from .schemas import FormulaSpec, MetricNode, PathCandidate


class BudgetAwarePathRanker:
    """Cheap path ranking before recursive evidence/LLM work."""
    def __init__(self, store: MetricStore, config: Dict[str, Any] | None = None):
        self.store = store; cfg = config or {}
        self.w_evidence = float(cfg.get("w_evidence", 0.52)); self.w_trust = float(cfg.get("w_trust", 0.28))
        self.w_depth = float(cfg.get("w_depth", 0.10)); self.w_cost = float(cfg.get("w_cost", 0.10))
        self.max_paths = int(cfg.get("max_paths_per_metric", 3)); self.min_score = float(cfg.get("min_path_score", -1.0))

    def rank(self, metric: MetricNode, context: str) -> List[PathCandidate]:
        candidates: List[PathCandidate] = []
        for formula in metric.formulas:
            coverage = self._coverage(formula, context)
            depth = max([self.store.estimate_depth(d) for d in formula.dependencies] + [0]) + 1
            origin_trust = 1.0 if formula.origin == "static" and metric.origin == "static" else 0.82
            trust = max(0.0, min(1.0, formula.confidence * metric.confidence * origin_trust))
            estimated_cost = len(formula.dependencies) + 0.65 * depth
            score = (self.w_evidence * coverage + self.w_trust * trust
                     - self.w_depth * min(1.0, depth / 6.0)
                     - self.w_cost * min(1.0, estimated_cost / 8.0))
            candidates.append(PathCandidate(metric=metric.name, formula=formula, score=score,
                                            estimated_depth=depth, evidence_coverage=coverage,
                                            trust=trust, estimated_cost=estimated_cost))
        candidates.sort(key=lambda x: (x.score, x.evidence_coverage, x.trust), reverse=True)
        return [x for x in candidates if x.score >= self.min_score][:self.max_paths]

    def _coverage(self, formula: FormulaSpec, context: str) -> float:
        if not formula.dependencies: return 0.0
        low = context.lower(); hits = 0.0
        for dep in formula.dependencies:
            found = any(alias and re.search(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)", low)
                        for alias in self.store.aliases_for(dep))
            hits += 1.0 if found else 0.0
        return hits / len(formula.dependencies)
