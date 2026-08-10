from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .formula import extract_dependency_phrases, normalize_metric_name
from .schemas import FormulaSpec, MetricNode


class MetricStore:
    """In-memory FMKG with a per-query overlay."""

    def __init__(self, metrics: Iterable[MetricNode]):
        self.base: Dict[str, MetricNode] = {}
        self.overlay: Dict[str, MetricNode] = {}
        self._alias_index: Dict[str, str] = {}
        for metric in metrics:
            self._add_base(metric)

    @classmethod
    def from_json(cls, path: str | Path) -> "MetricStore":
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "metrics" in obj:
            records = obj["metrics"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError("Normalized KB must be a list or {'metrics': [...]} object")
        return cls(MetricNode.from_dict(x) for x in records)

    def new_session(self) -> "MetricStore":
        session = object.__new__(MetricStore)
        session.base = self.base
        session.overlay = {}
        session._alias_index = dict(self._alias_index)
        return session

    def _add_base(self, metric: MetricNode) -> None:
        if not metric.name:
            return
        key = normalize_metric_name(metric.name)
        self.base[key] = metric
        self._index_metric(metric)

    def add_overlay(self, metric: MetricNode) -> None:
        key = normalize_metric_name(metric.name)
        self.overlay[key] = metric
        self._index_metric(metric)

    def _index_metric(self, metric: MetricNode) -> None:
        canonical = normalize_metric_name(metric.name)
        self._alias_index[canonical] = canonical
        for alias in metric.aliases:
            a = normalize_metric_name(alias)
            if a:
                self._alias_index[a] = canonical

    def get(self, name: str) -> Optional[MetricNode]:
        key = normalize_metric_name(name)
        canonical = self._alias_index.get(key, key)
        return self.overlay.get(canonical) or self.base.get(canonical)

    def contains(self, name: str) -> bool:
        return self.get(name) is not None

    def aliases_for(self, name: str) -> List[str]:
        metric = self.get(name)
        return [name] if not metric else list(dict.fromkeys([metric.name] + metric.aliases))

    def all_metrics(self) -> List[MetricNode]:
        merged = dict(self.base)
        merged.update(self.overlay)
        return list(merged.values())

    def overlay_metrics(self) -> List[MetricNode]:
        return list(self.overlay.values())

    def dependencies(self, name: str) -> Set[str]:
        metric = self.get(name)
        if not metric:
            return set()
        out: Set[str] = set()
        for f in metric.formulas:
            out.update(f.dependencies)
        return out

    def reaches(self, start: str, target: str, max_depth: int = 8) -> bool:
        target_key = normalize_metric_name(target)
        seen: Set[str] = set()
        def dfs(name: str, depth: int) -> bool:
            key = normalize_metric_name(name)
            if key == target_key:
                return True
            if key in seen or depth >= max_depth:
                return False
            seen.add(key)
            metric = self.get(name)
            if not metric:
                return False
            return any(dfs(dep, depth + 1) for dep in self.dependencies(metric.name))
        return dfs(start, 0)

    def estimate_depth(self, name: str, max_depth: int = 8) -> int:
        seen: Set[str] = set()
        def depth(metric_name: str, level: int) -> int:
            key = normalize_metric_name(metric_name)
            if key in seen or level >= max_depth:
                return level
            seen.add(key)
            metric = self.get(metric_name)
            if not metric or metric.atomic:
                return level
            child_depths = [depth(dep, level + 1) for dep in self.dependencies(metric.name)]
            return max(child_depths, default=level)
        return max(0, depth(name, 0))


def convert_gbfr_record(record: dict) -> MetricNode:
    """Convert one public GBFR knowledge-base record into the AdaGBFR schema."""
    term_obj = record.get("term_en") or {}
    name = term_obj.get("name") or record.get("name") or record.get("term") or ""
    aliases_obj = term_obj.get("aliases") or record.get("aliases") or {}
    aliases: List[str] = []
    if isinstance(aliases_obj, dict):
        for key in ("english", "abbreviations", "chinese"):
            value = aliases_obj.get(key, [])
            aliases.extend([value] if isinstance(value, str) else (value or []))
    elif isinstance(aliases_obj, list):
        aliases.extend(aliases_obj)

    raw_formulas = record.get("formula_en") or record.get("formulas_en") or record.get("formulas") or []
    related = record.get("related_concepts_en") or record.get("related_concepts") or []
    related_names = []
    for item in related:
        if isinstance(item, dict):
            concept = item.get("concept") or item.get("name")
            if concept: related_names.append(str(concept))
        elif item:
            related_names.append(str(item))

    formulas: List[FormulaSpec] = []
    if isinstance(raw_formulas, str):
        raw_formulas = [raw_formulas]
    for item in raw_formulas:
        if isinstance(item, dict):
            expression = item.get("expression") or item.get("formula") or ""
            deps = item.get("dependencies") or item.get("required_vars") or []
        else:
            expression, deps = str(item), []
        if not expression:
            continue
        if not deps:
            deps = _match_dependencies(expression, related_names)
        if not deps:
            try: deps = extract_dependency_phrases(expression)
            except Exception: deps = []
        formulas.append(FormulaSpec(
            expression=expression,
            dependencies=[str(x) for x in deps],
            confidence=1.0,
            source_ids=["gbfr_fmkg"],
            origin="static",
        ))
    return MetricNode(
        name=str(name),
        aliases=list(dict.fromkeys(str(x) for x in aliases if str(x).strip())),
        definition=str(record.get("definition_en") or record.get("definition") or ""),
        formulas=formulas,
        origin="static",
        confidence=1.0,
        source_ids=["gbfr_fmkg"],
    )


def _match_dependencies(expression: str, candidates: List[str]) -> List[str]:
    if "=" not in expression:
        return []
    rhs = expression.split("=", 1)[1]
    return [term for term in sorted(candidates, key=len, reverse=True) if re.search(re.escape(term), rhs, flags=re.I)]
