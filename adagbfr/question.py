from __future__ import annotations

import json
from typing import Any, Optional

from .schemas import QueryPlan


class QuestionAnalyzer:
    def __init__(self, llm: Any = None):
        self.llm = llm

    def analyze(self, query: str, context: str, override: Optional[dict] = None) -> QueryPlan:
        if override is not None:
            plan = QueryPlan.from_dict(override)
            if not plan.subgoals: raise ValueError("plan_override must contain at least one subgoal")
            return plan
        if self.llm is None:
            raise ValueError("LLM question analysis is disabled; provide plan_override")
        system = ("Convert a financial question into a small symbolic plan. Do not solve it and do not invent operands. "
                  "Use context only to disambiguate entity, time, currency, scale and unit.")
        user = f"""Question: {query}
Context:
{context[:10000]}
Choose reasoning_type from DirectRetrieval, ArithmeticCalculation, ComparativeReasoning, StatisticalAggregation, TemporalReasoning.
Choose operator from identity, sum, difference, product, division, ratio, change_ratio, range, compare, average, counting, time.
Return JSON only:
{{"reasoning_type":"...","operator":"...","subgoals":[{{"metric":"canonical financial metric name","constraints":{{"temporal_scope":null,"entity_scope":null,"meta":{{"currency":null,"scale":null,"unit":null}}}}}}],"precision":null}}"""
        obj = self.llm.complete_json(system, user, max_tokens=800)
        plan = QueryPlan.from_dict(obj)
        if not plan.subgoals:
            raise ValueError(f"Invalid query plan: {json.dumps(obj, ensure_ascii=False)}")
        return plan
