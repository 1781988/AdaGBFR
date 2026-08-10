from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLimits:
    max_llm_calls: int = 24
    max_total_tokens: int = 16000
    max_cost_usd: float = 0.25
    max_paths_total: int = 12
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


class BudgetTracker:
    def __init__(self, limits: Optional[BudgetLimits] = None):
        self.limits = limits or BudgetLimits()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.paths_considered = 0
        self.paths_executed = 0
        self.dynamic_requests = 0
        self.dynamic_accepts = 0
        self.verifier_calls = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 3.5))

    def check_llm_call(self, prompt_text: str = "") -> None:
        estimated = self.estimate_tokens(prompt_text)
        if self.llm_calls + 1 > self.limits.max_llm_calls:
            raise BudgetExceeded("max_llm_calls reached")
        if self.input_tokens + self.output_tokens + estimated > self.limits.max_total_tokens:
            raise BudgetExceeded("max_total_tokens would be exceeded")
        if self.estimated_cost_usd >= self.limits.max_cost_usd:
            raise BudgetExceeded("max_cost_usd reached")

    def record_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.estimated_cost_usd += (
            input_tokens * self.limits.input_price_per_million
            + output_tokens * self.limits.output_price_per_million
        ) / 1_000_000.0
        self._assert_limits()

    def record_path_considered(self, n: int = 1) -> None:
        self.paths_considered += n
        if self.paths_considered > self.limits.max_paths_total:
            raise BudgetExceeded("max_paths_total reached")

    def record_path_executed(self, n: int = 1) -> None:
        self.paths_executed += n

    def _assert_limits(self) -> None:
        if self.input_tokens + self.output_tokens > self.limits.max_total_tokens:
            raise BudgetExceeded("max_total_tokens reached")
        if self.estimated_cost_usd > self.limits.max_cost_usd:
            raise BudgetExceeded("max_cost_usd reached")

    def to_dict(self):
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "paths_considered": self.paths_considered,
            "paths_executed": self.paths_executed,
            "dynamic_requests": self.dynamic_requests,
            "dynamic_accepts": self.dynamic_accepts,
            "verifier_calls": self.verifier_calls,
            "limits": asdict(self.limits),
        }
