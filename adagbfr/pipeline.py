from __future__ import annotations

from typing import Any, Dict, List, Optional

from .budget import BudgetExceeded, BudgetTracker
from .question import QuestionAnalyzer
from .resolver import AdaptiveResolver
from .schemas import QueryPlan, ResolutionResult, RunResult, Status


class AdaGBFRPipeline:
    def __init__(self, analyzer: QuestionAnalyzer, resolver: AdaptiveResolver,
                 budget: BudgetTracker, store):
        self.analyzer = analyzer; self.resolver = resolver; self.budget = budget; self.store = store

    def run(self, query: str, context: str, plan_override: Optional[Dict[str, Any]] = None) -> RunResult:
        plan: Optional[QueryPlan] = None; results: List[ResolutionResult] = []
        try:
            plan = self.analyzer.analyze(query, context, override=plan_override)
            for subgoal in plan.subgoals:
                result = self.resolver.resolve(subgoal.metric, subgoal.constraints, context, query)
                results.append(result)
                if not result.ok:
                    return RunResult(status=result.status, value=None, answer=None, plan=plan,
                        subgoal_results=results, budget=self.budget.to_dict(),
                        overlay_metrics=[x.to_dict() for x in self.store.overlay_metrics()],
                        reason=f"subgoal '{subgoal.metric}' failed: {result.reason}")
            value = aggregate(plan.operator, [float(x.value) for x in results])
            return RunResult(status=Status.ANSWER, value=value, answer=format_answer(value, plan.precision), plan=plan,
                subgoal_results=results, budget=self.budget.to_dict(),
                overlay_metrics=[x.to_dict() for x in self.store.overlay_metrics()])
        except BudgetExceeded as e:
            return RunResult(Status.BUDGET_EXHAUSTED, None, None, plan, results, self.budget.to_dict(),
                             [x.to_dict() for x in self.store.overlay_metrics()], str(e))
        except (ValueError, KeyError) as e:
            return RunResult(Status.INVALID_PLAN, None, None, plan, results, self.budget.to_dict(),
                             [x.to_dict() for x in self.store.overlay_metrics()], str(e))
        except Exception as e:
            return RunResult(Status.EXECUTION_ERROR, None, None, plan, results, self.budget.to_dict(),
                             [x.to_dict() for x in self.store.overlay_metrics()], f"{type(e).__name__}: {e}")


def aggregate(operator: str, values: List[float]) -> float:
    op = (operator or "identity").strip().lower().replace(" ", "_")
    if not values: raise ValueError("No resolved values to aggregate")
    if op in {"identity", "direct", "direct_retrieval"}:
        if len(values) != 1: raise ValueError("identity operator expects one value")
        return values[0]
    if op == "sum": return sum(values)
    if op == "difference":
        if len(values) < 2: raise ValueError("difference expects at least two values")
        out = values[0]
        for v in values[1:]: out -= v
        return out
    if op == "product":
        out = 1.0
        for v in values: out *= v
        return out
    if op in {"division", "ratio"}:
        if len(values) != 2 or values[1] == 0: raise ValueError(f"{op} expects two values and nonzero denominator")
        return values[0] / values[1]
    if op in {"change_ratio", "change", "growth_rate"}:
        if len(values) != 2 or values[1] == 0: raise ValueError("change_ratio expects target and base values")
        return (values[0] - values[1]) / values[1]
    if op in {"range", "compare"}:
        if len(values) != 2: raise ValueError(f"{op} expects two values")
        return values[0] - values[1]
    if op == "average": return sum(values) / len(values)
    if op == "counting": return float(len(values))
    if op == "time":
        if len(values) != 2: raise ValueError("time expects two numerical boundaries")
        return values[1] - values[0]
    raise ValueError(f"Unsupported operator: {operator}")


def format_answer(value: float, precision: Optional[str]) -> str:
    if precision:
        p = precision.lower()
        if "percent" in p or "percentage" in p or "%" in p:
            digits = _precision_digits(p, default=2); return f"{value * 100:.{digits}f}%"
        digits = _precision_digits(p, default=None)
        if digits is not None: return f"{value:.{digits}f}"
        if "nearest integer" in p: return str(round(value))
    return f"{value:.10g}"


def _precision_digits(text: str, default: Optional[int]) -> Optional[int]:
    import re
    words = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}
    m = re.search(r"(\d+)\s*(?:decimal|place)", text)
    if m: return int(m.group(1))
    for word, n in words.items():
        if f"{word} decimal" in text or f"{word} place" in text: return n
    return default
