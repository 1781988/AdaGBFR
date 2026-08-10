from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .budget import BudgetTracker
from .schemas import Evidence, FormulaSpec


class AdaptiveVerifier:
    def __init__(self, llm: Any = None, budget: BudgetTracker | None = None, config: Dict[str, Any] | None = None):
        self.llm = llm; self.budget = budget; cfg = config or {}
        self.mode = str(cfg.get("mode", "adaptive"))
        self.deterministic_confidence = float(cfg.get("deterministic_confidence", 0.97))
        self.dynamic_requires_llm = bool(cfg.get("dynamic_requires_llm", False))

    def verify(self, metric: str, formula: FormulaSpec, value: float,
               evidences: List[Evidence], trace: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        if not evidences: return False, 0.0, "no_grounded_evidence"
        evidence_conf = min(e.confidence for e in evidences); base_conf = min(evidence_conf, formula.confidence)
        use_llm = self.mode == "llm"
        if self.mode == "adaptive":
            dynamic = formula.origin != "static"
            if dynamic and self.dynamic_requires_llm: use_llm = True
            elif base_conf < self.deterministic_confidence: use_llm = True
        if not use_llm: return True, base_conf, "deterministic_trace_check"
        if self.llm is None: return False, 0.0, "verification_requires_llm_but_llm_disabled"
        if self.budget: self.budget.verifier_calls += 1
        payload = {"metric": metric, "formula": formula.expression, "dependencies": formula.dependencies,
                   "result": value, "evidence": [e.to_dict() for e in evidences], "trace": trace}
        system = ("Verify a financial calculation trace using only supplied evidence and formula. Reject missing operands, "
                  "wrong entity/time/unit, unsupported aliases, arithmetic inconsistency, or ungrounded values. Do not repair the path.")
        user = f"Candidate trace:\n{json.dumps(payload, ensure_ascii=False)}\nReturn JSON only: {{\"valid\":true_or_false,\"confidence\":0.0,\"reason\":\"short reason\"}}"
        obj = self.llm.complete_json(system, user, max_tokens=350)
        valid = bool(obj.get("valid")); conf = max(0.0, min(1.0, float(obj.get("confidence", 0.0)))) if valid else 0.0
        return valid, min(base_conf, conf) if valid else 0.0, str(obj.get("reason", "llm_verifier"))
