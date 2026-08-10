from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .budget import BudgetExceeded, BudgetTracker
from .formula import SafeFormulaExecutor, extract_dependency_phrases, normalize_metric_name, split_formula
from .knowledge import MetricStore
from .llm import LLMError
from .schemas import FormulaSpec, MetricNode
from .source_retriever import LocalSourceIndex, RetrievedSource


class DynamicMetricExpander:
    """Evidence-grounded, per-query FMKG expansion."""

    def __init__(self, store: MetricStore, source_index: Optional[LocalSourceIndex],
                 executor: Optional[SafeFormulaExecutor] = None, llm: Any = None,
                 budget: Optional[BudgetTracker] = None, config: Optional[Dict[str, Any]] = None):
        self.store = store
        self.source_index = source_index
        self.executor = executor or SafeFormulaExecutor()
        self.llm = llm
        self.budget = budget
        self.config = config or {}
        self.top_k = int(self.config.get("top_k_sources", 5))
        self.min_source_score = float(self.config.get("min_source_score", 0.0))
        self.min_confidence = float(self.config.get("min_confidence", 0.82))
        self.min_sources = int(self.config.get("min_sources", 1))
        self.allow_rule_parser = bool(self.config.get("allow_rule_parser", True))
        self.use_llm_fallback = bool(self.config.get("use_llm_fallback", True))
        self.promotion_log = self.config.get("promotion_log")

    def expand(self, metric_name: str, query: str = "", context: str = "") -> Optional[MetricNode]:
        if self.store.get(metric_name):
            return self.store.get(metric_name)
        if self.budget:
            self.budget.dynamic_requests += 1
        sources = self._retrieve(metric_name, query)
        if len(sources) < self.min_sources:
            return None
        candidate = self._rule_candidate(metric_name, sources) if self.allow_rule_parser else None
        if candidate is None and self.use_llm_fallback and self.llm is not None:
            try:
                candidate = self._llm_candidate(metric_name, query, context, sources)
            except (LLMError, BudgetExceeded, ValueError):
                raise
            except Exception:
                candidate = None
        if candidate is None:
            return None
        valid, reason = self._validate(candidate, metric_name, sources)
        if not valid:
            self._log_candidate(candidate, accepted=False, reason=reason)
            return None
        self.store.add_overlay(candidate)
        if self.budget:
            self.budget.dynamic_accepts += 1
        self._log_candidate(candidate, accepted=True, reason="validated")
        return candidate

    def _retrieve(self, metric_name: str, query: str) -> List[RetrievedSource]:
        if self.source_index is None:
            return []
        q = f"{metric_name} financial metric formula definition {query}".strip()
        return [x for x in self.source_index.search(q, top_k=self.top_k) if x.score >= self.min_source_score]

    def _rule_candidate(self, metric_name: str, sources: List[RetrievedSource]) -> Optional[MetricNode]:
        formulas: List[FormulaSpec] = []
        source_ids: List[str] = []
        definition = ""; best_authority = 0.0
        target_norm = normalize_metric_name(metric_name)
        for rs in sources:
            doc = rs.doc; best_authority = max(best_authority, doc.authority)
            lines = [x.strip() for x in re.split(r"[\n;]", doc.text.replace("\r", "\n")) if x.strip()]
            for line in lines:
                if target_norm not in normalize_metric_name(line):
                    continue
                if not definition and "=" not in line:
                    definition = line[:500]
                if "=" not in line:
                    continue
                eq_pos = line.find("=")
                pos = line.lower().rfind(metric_name.lower(), 0, eq_pos + 1)
                formula_text = line[pos:] if pos >= 0 else line
                try:
                    lhs, _ = split_formula(formula_text)
                except Exception:
                    continue
                if SequenceMatcher(None, normalize_metric_name(lhs), target_norm).ratio() < 0.65:
                    continue
                try:
                    deps = extract_dependency_phrases(formula_text)
                except Exception:
                    deps = []
                if not deps or not self.executor.can_compile(formula_text, deps):
                    continue
                formulas.append(FormulaSpec(
                    expression=formula_text,
                    dependencies=deps,
                    confidence=min(0.96, 0.76 + 0.18 * doc.authority),
                    source_ids=[doc.source_id],
                    origin="dynamic_rule",
                ))
                source_ids.append(doc.source_id)
        if not formulas:
            return None
        confidence = min(0.96, 0.72 + 0.15 * best_authority + 0.04 * min(2, len(set(source_ids))))
        return MetricNode(name=metric_name, aliases=[], definition=definition,
                          formulas=_dedupe_formulas(formulas), origin="dynamic_rule",
                          confidence=confidence, source_ids=list(dict.fromkeys(source_ids)))

    def _llm_candidate(self, metric_name: str, query: str, context: str,
                       sources: List[RetrievedSource]) -> Optional[MetricNode]:
        source_payload = [{"source_id": x.doc.source_id, "title": x.doc.title,
                           "authority": x.doc.authority, "text": x.doc.text[:2200]}
                          for x in sources]
        system = (
            "Normalize financial metric definitions into a temporary calculation graph. "
            "Use only supplied source excerpts. Never invent a formula or dependency. "
            "If sources do not explicitly support a formula, return accepted=false."
        )
        user = f"""Target metric: {metric_name}
Question: {query}
Context excerpt (terminology only): {context[:1800]}
Sources:
{json.dumps(source_payload, ensure_ascii=False)}
Return JSON only:
{{"accepted":true_or_false,"canonical_name":"...","aliases":[],"definition":"...","confidence":0.0,"source_ids":[],"formulas":[{{"expression":"Metric = Operand A / Operand B","dependencies":["Operand A","Operand B"],"confidence":0.0,"source_ids":[]}}]}}"""
        obj = self.llm.complete_json(system, user, max_tokens=1200)
        if not obj.get("accepted"):
            return None
        formulas = [FormulaSpec(
            expression=str(item.get("expression", "")),
            dependencies=[str(x) for x in item.get("dependencies", [])],
            confidence=float(item.get("confidence", obj.get("confidence", 0.0))),
            source_ids=[str(x) for x in item.get("source_ids", obj.get("source_ids", []))],
            origin="dynamic_llm",
        ) for item in obj.get("formulas", [])]
        return MetricNode(name=str(obj.get("canonical_name") or metric_name),
                          aliases=[str(x) for x in obj.get("aliases", [])],
                          definition=str(obj.get("definition", "")), formulas=formulas,
                          origin="dynamic_llm", confidence=float(obj.get("confidence", 0.0)),
                          source_ids=[str(x) for x in obj.get("source_ids", [])])

    def _validate(self, candidate: MetricNode, requested_name: str,
                  retrieved_sources: List[RetrievedSource]) -> Tuple[bool, str]:
        if not candidate.name or not candidate.formulas:
            return False, "missing_name_or_formula"
        requested = normalize_metric_name(requested_name)
        sims = [SequenceMatcher(None, normalize_metric_name(candidate.name), requested).ratio()]
        sims += [SequenceMatcher(None, normalize_metric_name(a), requested).ratio() for a in candidate.aliases]
        if max(sims) < 0.72:
            return False, "metric_name_mismatch"
        if candidate.confidence < self.min_confidence:
            return False, "candidate_confidence_below_threshold"
        available = {x.doc.source_id for x in retrieved_sources}
        if len(set(candidate.source_ids) & available) < self.min_sources:
            return False, "insufficient_source_provenance"
        valid_formulas: List[FormulaSpec] = []
        for formula in candidate.formulas:
            if formula.confidence < self.min_confidence or not formula.dependencies:
                continue
            if any(normalize_metric_name(d) == normalize_metric_name(candidate.name) for d in formula.dependencies):
                continue
            if len(set(formula.source_ids) & available) < self.min_sources:
                continue
            try:
                lhs, _ = split_formula(formula.expression)
            except Exception:
                continue
            if SequenceMatcher(None, normalize_metric_name(lhs), normalize_metric_name(candidate.name)).ratio() < 0.7:
                continue
            if not self.executor.can_compile(formula.expression, formula.dependencies):
                continue
            if any(self.store.reaches(dep, candidate.name) for dep in formula.dependencies if self.store.contains(dep)):
                continue
            valid_formulas.append(formula)
        if not valid_formulas:
            return False, "no_safe_formula"
        candidate.formulas = _dedupe_formulas(valid_formulas)
        return True, "ok"

    def _log_candidate(self, candidate: MetricNode, accepted: bool, reason: str) -> None:
        if not self.promotion_log:
            return
        path = Path(self.promotion_log); path.parent.mkdir(parents=True, exist_ok=True)
        record = candidate.to_dict(); record["accepted"] = accepted; record["reason"] = reason
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dedupe_formulas(formulas: List[FormulaSpec]) -> List[FormulaSpec]:
    out, seen = [], set()
    for f in formulas:
        key = re.sub(r"\s+", " ", f.expression.strip().lower())
        if key not in seen:
            seen.add(key); out.append(f)
    return out
