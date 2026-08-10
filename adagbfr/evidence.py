from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .formula import normalize_metric_name
from .schemas import Constraints, Evidence, MetricNode

_NUMBER = r"(?:\(?[-+]?\s*(?:[$€£¥]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?\)?)"


def parse_number(raw: str) -> float:
    s = (raw or "").strip(); neg = s.startswith("(") and s.endswith(")"); is_percent = "%" in s
    s = s.replace(",", ""); s = re.sub(r"[$€£¥%()\s]", "", s)
    if not s: raise ValueError("empty numeric string")
    value = float(s)
    if neg and value > 0: value = -value
    if is_percent: value /= 100.0
    return value


class HybridEvidenceExtractor:
    """Strict label-adjacent extraction first, LLM fallback second. Never derives values."""
    def __init__(self, llm: Any = None, use_llm_fallback: bool = True):
        self.llm = llm; self.use_llm_fallback = use_llm_fallback
        self.cache: Dict[Tuple, Optional[Evidence]] = {}

    def find(self, metric: MetricNode, constraints: Constraints, context: str) -> Optional[Evidence]:
        key = (normalize_metric_name(metric.name), constraints.temporal_scope, constraints.entity_scope,
               constraints.currency, constraints.scale, constraints.unit, hash(context))
        if key in self.cache: return self.cache[key]
        evidence = self._rule_find(metric, constraints, context)
        if evidence is None and self.llm is not None and self.use_llm_fallback:
            evidence = self._llm_find(metric, constraints, context)
        self.cache[key] = evidence
        return evidence

    def _rule_find(self, metric: MetricNode, constraints: Constraints, context: str) -> Optional[Evidence]:
        aliases = [metric.name] + list(metric.aliases); candidates: List[Evidence] = []
        for alias in sorted(set(aliases), key=len, reverse=True):
            if not alias: continue
            escaped = re.escape(alias)
            patterns = [
                rf"(?im)(?P<label>{escaped})\s*(?:\([^\n]{{0,40}}\))?\s*[:=|]\s*(?P<value>{_NUMBER})",
                rf"(?im)(?P<label>{escaped})\s+(?:was|were|is|of|amounted\s+to|totaled)?\s*(?P<value>{_NUMBER})",
            ]
            for pattern in patterns:
                for m in re.finditer(pattern, context):
                    raw = m.group("value")
                    try: value = parse_number(raw)
                    except ValueError: continue
                    line_start = context.rfind("\n", 0, m.start()) + 1
                    line_end = context.find("\n", m.end())
                    if line_end < 0: line_end = min(len(context), m.end() + 160)
                    snippet = context[line_start:line_end].strip()
                    if not self._constraint_hint_ok(constraints, snippet, context, m.start(), m.end()):
                        continue
                    exact = normalize_metric_name(alias) == normalize_metric_name(metric.name)
                    candidates.append(Evidence(metric=metric.name, value=value, raw_value=raw.strip(),
                                               snippet=snippet[:500], confidence=0.985 if exact else 0.955,
                                               method="rule_exact" if exact else "rule_alias", constraints=constraints,
                                               notes=None if exact else f"resolved through alias: {alias}"))
        if not candidates: return None
        unique = {}
        for e in candidates: unique.setdefault(round(e.value, 12), e)
        if len(unique) == 1: return max(candidates, key=lambda x: x.confidence)
        if constraints.temporal_scope:
            matched = [e for e in candidates if str(constraints.temporal_scope).lower() in e.snippet.lower()]
            if matched and len({round(x.value, 12) for x in matched}) == 1:
                return max(matched, key=lambda x: x.confidence)
        return None

    def _constraint_hint_ok(self, c: Constraints, snippet: str, context: str, start: int, end: int) -> bool:
        window = context[max(0, start - 140):min(len(context), end + 140)]
        if c.temporal_scope:
            wanted = str(c.temporal_scope).lower()
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", window))
            wanted_years = set(re.findall(r"\b(?:19|20)\d{2}\b", wanted))
            if years and wanted_years and not (years & wanted_years): return False
        return True

    def _llm_find(self, metric: MetricNode, constraints: Constraints, context: str) -> Optional[Evidence]:
        aliases = list(dict.fromkeys([metric.name] + metric.aliases))
        system = ("Extract financial numeric evidence only. Never calculate, derive, estimate, or fill missing values. "
                  "A value is valid only if an explicit metric label or supported alias is attached to the number and all constraints match.")
        user = f"""Metric: {metric.name}
Aliases: {json.dumps(aliases, ensure_ascii=False)}
Constraints: {json.dumps(constraints.to_dict(), ensure_ascii=False)}
Context:
{context[:12000]}
Return JSON only: {{"found":true_or_false,"value_text":"number exactly as written","snippet":"minimal evidence span","confidence":0.0,"notes":"reason or alias mapping"}}"""
        obj = self.llm.complete_json(system, user, max_tokens=500)
        if not obj.get("found"): return None
        raw = str(obj.get("value_text", ""))
        try: value = parse_number(raw)
        except ValueError: return None
        snippet = str(obj.get("snippet", ""))
        digits = re.sub(r"[^0-9.-]", "", raw)
        if digits and digits not in re.sub(r"[, $€£¥%()]", "", context): return None
        return Evidence(metric=metric.name, value=value, raw_value=raw, snippet=snippet[:700],
                        confidence=max(0.0, min(1.0, float(obj.get("confidence", 0.0)))),
                        method="llm_extract", constraints=constraints,
                        notes=str(obj.get("notes")) if obj.get("notes") else None)
