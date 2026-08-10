from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .budget import BudgetExceeded, BudgetTracker
from .dynamic_expansion import DynamicMetricExpander
from .evidence import HybridEvidenceExtractor
from .formula import FormulaError, SafeFormulaExecutor, normalize_metric_name
from .knowledge import MetricStore
from .path_ranker import BudgetAwarePathRanker
from .schemas import Constraints, Evidence, ResolutionResult, Status
from .verification import AdaptiveVerifier


@dataclass
class _CandidateResult:
    value: float
    confidence: float
    score: float
    evidence: List[Evidence]
    trace: List[Dict[str, Any]]
    origin: str


class AdaptiveResolver:
    def __init__(self, store: MetricStore, evidence_extractor: HybridEvidenceExtractor,
                 ranker: BudgetAwarePathRanker, executor: SafeFormulaExecutor,
                 verifier: AdaptiveVerifier, expander: Optional[DynamicMetricExpander],
                 budget: BudgetTracker, config: Dict[str, Any] | None = None):
        self.store = store; self.evidence_extractor = evidence_extractor; self.ranker = ranker
        self.executor = executor; self.verifier = verifier; self.expander = expander; self.budget = budget
        cfg = config or {}
        self.max_depth = int(cfg.get("max_depth", 6))
        self.early_stop_confidence = float(cfg.get("early_stop_confidence", 0.97))
        self.near_tie_margin = float(cfg.get("near_tie_margin", 0.08))
        self.consistency_rtol = float(cfg.get("consistency_rtol", 1e-4))
        self.consistency_atol = float(cfg.get("consistency_atol", 1e-8))
        self.enable_early_stop = bool(cfg.get("enable_early_stop", True))
        self.cache: Dict[Tuple[str, str], ResolutionResult] = {}

    def resolve(self, metric_name: str, constraints: Constraints, context: str, query: str,
                depth: int = 0, visiting: Optional[Set[str]] = None) -> ResolutionResult:
        visiting = set(visiting or set())
        cache_key = (normalize_metric_name(metric_name), repr(constraints.to_dict()))
        if cache_key in self.cache: return self.cache[cache_key]
        if depth > self.max_depth:
            return ResolutionResult(metric_name, Status.GRAPH_COVERAGE, reason="max_depth_exceeded")
        key = normalize_metric_name(metric_name)
        if key in visiting:
            return ResolutionResult(metric_name, Status.GRAPH_COVERAGE, reason="cycle_detected")
        visiting.add(key)

        metric = self.store.get(metric_name)
        if metric is None:
            from .schemas import MetricNode
            direct_probe = MetricNode(name=metric_name, aliases=[], formulas=[], origin="unknown", confidence=1.0)
            evidence = self.evidence_extractor.find(direct_probe, constraints, context)
            if evidence is not None:
                result = ResolutionResult(metric=metric_name, status=Status.ANSWER, value=evidence.value,
                    confidence=evidence.confidence, evidence=[evidence],
                    trace=[{"type":"direct_evidence","metric":metric_name,"value":evidence.value}], origin="direct_context")
                self.cache[cache_key] = result; return result
            if self.expander is not None:
                metric = self.expander.expand(metric_name, query=query, context=context)
            if metric is None:
                result = ResolutionResult(metric_name, Status.GRAPH_COVERAGE,
                                          reason="metric_missing_and_dynamic_expansion_failed")
                self.cache[cache_key] = result; return result

        evidence = self.evidence_extractor.find(metric, constraints, context)
        if evidence is not None:
            result = ResolutionResult(metric=metric.name, status=Status.ANSWER, value=evidence.value,
                confidence=evidence.confidence, evidence=[evidence],
                trace=[{"type":"direct_evidence","metric":metric.name,"value":evidence.value}], origin=metric.origin)
            self.cache[cache_key] = result; return result
        if metric.atomic:
            result = ResolutionResult(metric.name, Status.EVIDENCE_ABSENT,
                                      reason="atomic_metric_not_explicitly_grounded", origin=metric.origin)
            self.cache[cache_key] = result; return result

        try:
            paths = self.ranker.rank(metric, context)
            if not paths:
                result = ResolutionResult(metric.name, Status.GRAPH_COVERAGE,
                                          reason="no_candidate_formula_after_pruning", origin=metric.origin)
                self.cache[cache_key] = result; return result
            candidate_results: List[_CandidateResult] = []
            for idx, path in enumerate(paths):
                self.budget.record_path_considered()
                child_values: Dict[str, float] = {}; all_evidence: List[Evidence] = []
                child_trace: List[Dict[str, Any]] = []; child_confidences: List[float] = []
                path_failed = False
                for dep in path.formula.dependencies:
                    child = self.resolve(dep, constraints, context, query, depth + 1, visiting)
                    child_trace.extend(child.trace)
                    if not child.ok:
                        path_failed = True; break
                    child_values[dep] = float(child.value); all_evidence.extend(child.evidence)
                    child_confidences.append(child.confidence)
                if path_failed: continue
                self.budget.record_path_executed()
                try:
                    value = self.executor.execute(path.formula.expression, child_values, path.formula.dependencies)
                except (FormulaError, KeyError, ValueError, OverflowError):
                    continue
                trace = child_trace + [{"type":"formula_execution","metric":metric.name,
                    "formula":path.formula.expression,"inputs":child_values,"value":value,"path_score":path.score}]
                valid, verify_conf, verify_reason = self.verifier.verify(metric.name, path.formula, value, all_evidence, trace)
                if not valid: continue
                confidence = min([path.trust, verify_conf] + child_confidences) if child_confidences else min(path.trust, verify_conf)
                candidate_results.append(_CandidateResult(value=value, confidence=confidence, score=path.score,
                    evidence=all_evidence, trace=trace + [{"type":"verification","valid":True,"reason":verify_reason}],
                    origin=path.formula.origin))
                if self.enable_early_stop and confidence >= self.early_stop_confidence:
                    remaining = paths[idx + 1:]
                    near_tie = any(p.score >= path.score - self.near_tie_margin for p in remaining)
                    if not near_tie: break
            result = self._consensus(metric.name, candidate_results, metric.origin)
            self.cache[cache_key] = result; return result
        except BudgetExceeded:
            result = ResolutionResult(metric.name, Status.BUDGET_EXHAUSTED,
                                      reason="reasoning_budget_exhausted", origin=metric.origin)
            self.cache[cache_key] = result; return result

    def _consensus(self, metric: str, candidates: List[_CandidateResult], origin: str) -> ResolutionResult:
        if not candidates:
            return ResolutionResult(metric, Status.EVIDENCE_ABSENT, reason="all_derivation_paths_failed", origin=origin)
        if len(candidates) == 1:
            c = candidates[0]
            return ResolutionResult(metric, Status.ANSWER, c.value, c.confidence, c.evidence, c.trace, origin=origin)
        clusters: List[List[_CandidateResult]] = []
        for cand in candidates:
            for cluster in clusters:
                if math.isclose(cand.value, cluster[0].value, rel_tol=self.consistency_rtol, abs_tol=self.consistency_atol):
                    cluster.append(cand); break
            else:
                clusters.append([cand])
        clusters.sort(key=lambda c: (len(c), sum(x.confidence for x in c)), reverse=True)
        best = clusters[0]
        if len(clusters) > 1 and len(best) == len(clusters[1]):
            return ResolutionResult(metric, Status.CONFLICT, reason="cross_path_conflict", origin=origin)
        representative = max(best, key=lambda x: x.confidence)
        return ResolutionResult(metric=metric, status=Status.ANSWER, value=representative.value,
            confidence=min(1.0, representative.confidence + 0.01 * (len(best) - 1)),
            evidence=representative.evidence,
            trace=representative.trace + [{"type":"cross_path_consensus","support":len(best),"candidate_count":len(candidates)}],
            origin=origin)
