from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

_FIELD_MAP = {
    "CFBenchmark": ("question", "context", "answer", None),
    "FinanceReasoning": ("question", "context", "ground_truth", None),
    "FinQA": ("question", "context", "answer", "table"),
    "TAT-QA": ("question", "context", "answer", "table"),
}


def load_dataset(dataset_name: str, data_root: str | Path) -> List[Dict[str, Any]]:
    if dataset_name not in _FIELD_MAP: raise ValueError(f"Unsupported dataset: {dataset_name}")
    qf, cf, af, tf = _FIELD_MAP[dataset_name]
    root = Path(data_root); files = [root] if root.is_file() else sorted(root.rglob("*.json")); rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as f: obj = json.load(f)
        for d in _records(obj):
            if not isinstance(d, dict) or qf not in d: continue
            context = _to_text(d.get(cf, ""))
            if tf and d.get(tf) not in (None, ""): context += "\n" + _to_text(d.get(tf))
            rows.append({"query":_to_text(d.get(qf,"")),"context":context,"answer":d.get(af),"source_file":str(path)})
    return rows


def _records(obj: Any) -> Iterable[dict]:
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for key in ("data", "examples", "items", "records"):
            if isinstance(obj.get(key), list): return obj[key]
        return [obj]
    return []


def _to_text(value: Any) -> str:
    if isinstance(value, str): return value
    if value is None: return ""
    if isinstance(value, list):
        if value and all(isinstance(x, list) for x in value):
            return "\n".join(" | ".join(str(c) for c in row) for row in value)
        return "\n".join(str(x) for x in value)
    if isinstance(value, dict): return json.dumps(value, ensure_ascii=False)
    return str(value)
